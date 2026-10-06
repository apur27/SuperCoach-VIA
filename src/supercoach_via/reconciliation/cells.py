"""Source cell semantics: what a printed or blank cell means (DESIGN section 8).

Pure functions: no I/O, no model, no production importer. For each appearance every one of the
23 statistics resolves to exactly one state from the printed cells of the player's profile row
and the same player's row on the match page, plus the match page's team totals and the source
notes. A blank is never zero merely because a teammate has a positive value.

States: ``RECORDED_VALUE`` (a printed number), ``RECORDED_ZERO`` (blank where the team total
proves the statistic was recorded), ``NOT_RECORDED`` (the source documents or structurally shows
no measurement), ``NOT_APPLICABLE`` (finals Brownlow votes), ``NOT_APPLICABLE_DNTF`` (a credited
game whose player did not take the field), ``UNRESOLVED_BLANK`` (evidence cannot distinguish
zero from unavailable), ``MALFORMED`` (unexpected text, duplicate header label) and
``SOURCE_CONFLICT`` (two source facts that index the same appearance disagree).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import NamedTuple

from supercoach_via.reconciliation.schema import STAT_FIELDS

_N = len(STAT_FIELDS)
_COUNT = re.compile(r"^(?:0|[1-9][0-9]*)$")
_PCT = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?%?$")
_BR = STAT_FIELDS.index("brownlow_votes")
_GL = STAT_FIELDS.index("goals")
_BH = STAT_FIELDS.index("behinds")
_PCT_I = STAT_FIELDS.index("time_on_ground_pct")


class St(StrEnum):
    VALUE = "RECORDED_VALUE"
    ZERO = "RECORDED_ZERO"
    NOT_RECORDED = "NOT_RECORDED"
    NA = "NOT_APPLICABLE"
    DNTF = "NOT_APPLICABLE_DNTF"
    UNRESOLVED = "UNRESOLVED_BLANK"
    MALFORMED = "MALFORMED"
    CONFLICT = "SOURCE_CONFLICT"


@dataclass(frozen=True, slots=True)
class Cell:
    state: St
    value: Decimal | None
    rule: str
    detail: str = ""


@dataclass(frozen=True)
class OneSidedRule:
    """A versioned rule (id + evidence locator in ``reconciliation_rules.toml``) that resolves a cell printed
    on one source page and blank on the other. Without one, such a cell is a ``SOURCE_CONFLICT`` (C-2)."""

    rule_id: str
    field: str
    first_season: int
    last_season: int
    printed_on: str  # "profile" | "match"


@dataclass(frozen=True)
class AppearanceCtx:
    season: int
    is_final: bool
    match_usable: bool
    player_in_lineup: bool
    column_present: frozenset[str]
    #: the team's printed Totals row (23 raw cells; ``""`` blank) or ``None`` when the match page has none
    team_totals: tuple[str | None, ...] | None
    #: the team's goals and behinds as printed in the match header score line
    team_goals: int | None
    team_behinds: int | None
    #: statistics the notes mark as recorded for the season; ``None`` before the notes table begins (1965)
    available: frozenset[str] | None
    #: statistics the notes' exception table lists as missing for this match and team
    exception_fields: frozenset[str]
    bad_fields: frozenset[str]
    #: ``%P`` is printed for at least one other player of the same team in this match
    team_pct_recorded: bool
    one_sided_rules: tuple[OneSidedRule, ...] = field(default=())
    #: the captured notes page was usable; without it, availability from 1965 cannot be established
    notes_usable: bool = True
    #: the opposition's printed Totals row (23 raw cells)
    opp_totals: tuple[str | None, ...] | None = None
    #: Brownlow, from the captured award evidence (A8, A9): no medal was awarded this season
    br_no_award: bool = False
    #: votes awarded per game this season per the evidence (``None``: the evidence states none)
    br_award_total: int | None = None
    #: this season's home-and-away match pages print per-game Brownlow votes at all (a structural fact of the
    #: season, derived from the captured pages): only then can a blank cell be a recorded zero
    br_season_recorded: bool = False


class _Bad:
    pass


BAD = _Bad()


def parse_cell(field_name: str, raw: str | None) -> Decimal | _Bad | None:
    """``None`` for a blank (or absent) cell, a ``Decimal`` for a valid printed number, ``BAD`` otherwise."""
    if raw is None or raw == "":
        return None
    if field_name == "time_on_ground_pct":
        if not _PCT.match(raw):
            return BAD
        return Decimal(raw.rstrip("%"))
    if not _COUNT.match(raw):
        return BAD
    return Decimal(raw)


def _all_blank(cells: tuple[str | None, ...]) -> bool:
    return all(c in ("", None) for c in cells)


def is_dntf(prof: tuple[str | None, ...], match: tuple[str | None, ...] | None, ctx: AppearanceCtx) -> bool:
    """A credited game whose player did not take the field (S-02): all four conditions must hold."""
    return (
        match is not None
        and ctx.match_usable
        and ctx.player_in_lineup
        and ctx.team_pct_recorded
        and _all_blank(prof)
        and _all_blank(match)
    )


def _votes(raw: str | None) -> int | None:
    """A printed Brownlow team total: blank is no votes, a count is itself, anything else is unreadable."""
    if raw in (None, ""):
        return 0
    return int(raw) if _COUNT.match(raw) else None


def _brownlow_blank(i: int, ctx: AppearanceCtx) -> Cell:
    """A blank per-game Brownlow cell in a season whose match pages record votes (A8).

    It is a recorded zero only when the two team totals printed on the match page sum to the votes the source says
    were awarded per game; otherwise the page records only part of the award and the blank proves nothing."""
    if ctx.br_award_total is None:
        why = "the season prints votes but the evidence states no award total"
        return Cell(St.UNRESOLVED, None, "R-BR-NO-AWARD-TOTAL", why)
    if ctx.opp_totals is None or ctx.team_totals is None:
        return Cell(St.UNRESOLVED, None, "R-NO-OPP-TOTALS", "the opposition's team totals are not on the page")
    mine, theirs = _votes(ctx.team_totals[i]), _votes(ctx.opp_totals[i])
    if mine is None or theirs is None:
        return Cell(St.UNRESOLVED, None, "R-BR-TOTAL-FORMAT", "a team Brownlow total is not a count")
    if mine + theirs == ctx.br_award_total:
        return Cell(St.ZERO, Decimal(0), "R-BR-AWARD-SUM")
    return Cell(
        St.UNRESOLVED,
        None,
        "R-BR-AWARD-SUM-MISMATCH",
        f"team totals {mine}+{theirs} sum to {mine + theirs}, not the {ctx.br_award_total} awarded per game",
    )


def _blank_state(i: int, f: str, ctx: AppearanceCtx) -> Cell:
    if f == "brownlow_votes":
        if ctx.is_final:
            return Cell(St.NA, None, "R-BR-FINALS")
        if ctx.br_no_award:
            return Cell(St.NA, None, "R-BR-NO-AWARD")
    if not ctx.match_usable:
        return Cell(St.UNRESOLVED, None, "R-NO-MATCH-PAGE")
    if f not in ctx.column_present:
        return Cell(St.NOT_RECORDED, None, "R-COLUMN-ABSENT")
    if ctx.team_totals is None:
        return Cell(St.UNRESOLVED, None, "R-NO-TOTALS-ROW")
    # a documented exception for this team-match beats a non-blank team total: the source's own averages exclude
    # these games, so a blank there is not a zero (A8)
    if f in ctx.exception_fields and ctx.available is not None and f in ctx.available:
        return Cell(St.NOT_RECORDED, None, "R-NOTES-EXCEPTION")
    if f == "brownlow_votes" and ctx.br_season_recorded:
        return _brownlow_blank(i, ctx)
    total = ctx.team_totals[i]
    if total not in (None, ""):
        return Cell(St.ZERO, Decimal(0), "R-TOTAL-NONBLANK")
    if not ctx.notes_usable and ctx.season >= 1965:
        return Cell(St.UNRESOLVED, None, "R-NO-NOTES")
    if f == "time_on_ground_pct":
        if ctx.available is not None and f in ctx.available and ctx.team_pct_recorded:
            return Cell(St.UNRESOLVED, None, "R-PCT-NEVER-ZERO-FILLED")
        return Cell(St.NOT_RECORDED, None, "R-NOTES-MATRIX" if ctx.available is not None else "R-PRE1965-STRUCTURE")
    if (f == "goals" and ctx.team_goals == 0) or (f == "behinds" and ctx.team_behinds == 0):
        return Cell(St.ZERO, Decimal(0), "R-HEADER-ZERO")
    if ctx.available is None:
        return Cell(St.NOT_RECORDED, None, "R-PRE1965-STRUCTURE")
    if f not in ctx.available:
        return Cell(St.NOT_RECORDED, None, "R-NOTES-MATRIX")
    return Cell(St.UNRESOLVED, None, "R-ALL-ZERO-UNPROVEN")


def classify_appearance(
    prof: tuple[str | None, ...], match: tuple[str | None, ...] | None, ctx: AppearanceCtx
) -> list[Cell]:
    """One ``Cell`` per canonical statistic for a single appearance."""
    if is_dntf(prof, match, ctx):
        dntf = Cell(St.DNTF, None, "R-DNTF")
        return [dntf if f not in ctx.bad_fields else Cell(St.MALFORMED, None, "R-LABEL-DUPLICATE") for f in STAT_FIELDS]
    out: list[Cell] = []
    for i, f in enumerate(STAT_FIELDS):
        p = prof[i]
        m = match[i] if match is not None else None
        if f in ctx.bad_fields:
            out.append(Cell(St.MALFORMED, None, "R-LABEL-DUPLICATE", f"{f}: duplicated header label"))
            continue
        pv = parse_cell(f, p)
        mv = parse_cell(f, m)
        if isinstance(pv, _Bad) or isinstance(mv, _Bad):
            out.append(Cell(St.MALFORMED, None, "R-CELL-FORMAT", f"profile={p!r} match={m!r}"))
            continue
        if match is None or m is None:
            # single-sourced: the match page has no row (or no column) for this statistic
            if pv is not None:
                out.append(Cell(St.VALUE, pv, "R-PROFILE-ONLY"))
            else:
                out.append(_blank_state(i, f, ctx))
            continue
        if pv is not None and mv is not None:
            if pv == mv:
                out.append(Cell(St.VALUE, pv, "R-BOTH-PAGES"))
            else:
                out.append(Cell(St.CONFLICT, None, "R-SOURCE-CONFLICT", f"profile={p!r} match={m!r}"))
        elif pv is None and mv is None:
            out.append(_blank_state(i, f, ctx))
        else:
            side = "profile" if pv is not None else "match"
            printed = pv if pv is not None else mv
            rule = next(
                (
                    r
                    for r in ctx.one_sided_rules
                    if r.field == f and r.first_season <= ctx.season <= r.last_season and r.printed_on == side
                ),
                None,
            )
            if rule is not None and isinstance(printed, Decimal):
                out.append(Cell(St.VALUE, printed, rule.rule_id))
            else:
                out.append(Cell(St.CONFLICT, None, "R-SOURCE-CONFLICT", f"profile={p!r} match={m!r}"))
    return out


def summary_only(
    *,
    season_value: str | None,
    game_cells: list[tuple[str | None, ...]],
    team_totals: list[tuple[str | None, ...] | None],
    field: str,
) -> bool:
    """``SOURCE_SUMMARY_ONLY`` (S-01): the season row prints a value while every per-game cell for the statistic
    is blank and the team Totals are blank on every one of the player's matches."""
    i = STAT_FIELDS.index(field)
    if not season_value or not _COUNT.match(season_value):
        return False
    if any(c[i] not in ("", None) for c in game_cells):
        return False
    return all(t is not None and t[i] in ("", None) for t in team_totals)


class LocalOutcome(NamedTuple):
    outcome: str
    detail: str = ""


def local_decimal(value: object) -> Decimal | _Bad | None:
    """A local numeric cell as an exact ``Decimal`` (``None`` for null); ``BAD`` for bool, NaN, infinity or text."""
    if value is None:
        return None
    if isinstance(value, bool):
        return BAD
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, Decimal):
        return value if value.is_finite() else BAD
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return BAD
        try:
            return Decimal(repr(value))
        except InvalidOperation:  # pragma: no cover - repr of a finite float always parses
            return BAD
    return BAD


def compare_local(cell: Cell, local: object, *, blank_is_zero: bool = False) -> LocalOutcome:
    """Judge one local value against one source cell.

    ``blank_is_zero`` is the legacy-CSV layer's declared representation (a blank counting statistic is a
    zero in a played game); the snapshot layer stores nulls and zeros distinctly.
    """
    lv = local_decimal(local)
    if isinstance(lv, _Bad):
        return LocalOutcome("malformed_local", repr(local))
    st = cell.state
    if st is St.VALUE:
        if lv is not None and lv == cell.value:
            return LocalOutcome("equal")
        return LocalOutcome("mismatch", f"expected {cell.value} actual {lv}")
    if st is St.ZERO:
        if lv == 0:
            return LocalOutcome("equal")
        if lv is None:
            return (
                LocalOutcome("equal_blank_as_zero")
                if blank_is_zero
                else LocalOutcome("mismatch_local_null", "expected 0 actual null")
            )
        return LocalOutcome("mismatch", f"expected 0 actual {lv}")
    if st is St.NOT_RECORDED:
        return (
            LocalOutcome("source_unavailable")
            if lv is None
            else LocalOutcome("unsupported_local_numeric", f"actual {lv}")
        )
    if st is St.NA:
        return (
            LocalOutcome("not_applicable")
            if lv in (None, 0)
            else LocalOutcome("mismatch", f"not applicable but actual {lv}")
        )
    if st is St.DNTF:
        return (
            LocalOutcome("not_applicable_dntf")
            if lv in (None, 0)
            else LocalOutcome("mismatch", f"did not take the field but actual {lv}")
        )
    return LocalOutcome("unresolved", st.value)
