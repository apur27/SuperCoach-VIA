"""Season, stint and career aggregates and the printed-average model (DESIGN section 8).

Pure arithmetic over cell states; no I/O. The source reference for an aggregate is the sum of
per-club-season references, each being the per-game cell sum or, for a season-summary-only
statistic, the season value (N-04). Statistics the source did not record contribute nothing; an
aggregate is source-unavailable only when every contribution is unrecorded (C-1). Printed
totals and averages are derived figures: a disagreement with the composed reference is a
source-consistency result, never a verdict on a local layer (S-07).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from supercoach_via.reconciliation.cells import Cell, St

_TWO = Decimal("0.01")
_DIGITS = re.compile(r"^(?:0|[1-9][0-9]*)$")


Num = int | Decimal


def compact(v: Decimal) -> Num:
    """An integral value as a plain ``int`` (a fraction of a Decimal's memory; small ints are shared), anything
    else unchanged. Exact either way: every reader converts back with ``Decimal(...)``."""
    return int(v) if v == v.to_integral_value() else v


class StatAcc:
    """Per-statistic tally over one club-season's games."""

    __slots__ = ("n_bad", "n_dntf", "n_na", "n_unrec", "n_value", "n_zero", "single_sourced", "total")

    def __init__(self) -> None:
        self.total: Num = 0
        self.n_value = self.n_zero = self.n_unrec = self.n_na = self.n_dntf = self.n_bad = 0
        self.single_sourced = False

    def add(self, cell: Cell, *, double_sourced: bool) -> None:
        st = cell.state
        if st is St.VALUE:
            self.n_value += 1
            assert cell.value is not None
            self.total += compact(cell.value)
            if not double_sourced:
                self.single_sourced = True
        elif st is St.ZERO:
            self.n_zero += 1
        elif st is St.NOT_RECORDED:
            self.n_unrec += 1
        elif st is St.NA:
            self.n_na += 1
        elif st is St.DNTF:
            self.n_dntf += 1
        else:
            self.n_bad += 1

    @property
    def games(self) -> int:
        return self.n_value + self.n_zero + self.n_unrec + self.n_na + self.n_dntf + self.n_bad

    @property
    def recorded(self) -> int:
        return self.n_value + self.n_zero


@dataclass(frozen=True)
class Contribution:
    kind: str  # per_game | summary_only | unavailable | not_applicable | unresolved
    value: Decimal = Decimal(0)
    single_sourced: bool = False


def contribution(acc: StatAcc, summary_value: Decimal | None, *, summary_only: bool = False) -> Contribution:
    if acc.n_bad:
        return Contribution("unresolved", Decimal(acc.total), acc.single_sourced)
    if summary_only and summary_value is not None:
        return Contribution("summary_only", summary_value, False)
    if acc.recorded:
        return Contribution("per_game", Decimal(acc.total), acc.single_sourced)
    if acc.games and acc.n_na == acc.games:
        return Contribution("not_applicable")
    return Contribution("unavailable")


@dataclass(frozen=True)
class Composed:
    kind: str  # composed | unavailable | not_applicable | unresolved
    value: Decimal
    recorded_seasons: int
    unrecorded_seasons: int
    single_sourced: bool
    summary_part: Decimal = Decimal(0)
    has_summary_only: bool = False


def compose(parts: list[Contribution]) -> Composed:
    if any(p.kind == "unresolved" for p in parts):
        return Composed("unresolved", Decimal(0), 0, 0, True)
    live = [p for p in parts if p.kind in ("per_game", "summary_only")]
    if not live:
        all_na = bool(parts) and all(p.kind == "not_applicable" for p in parts)
        return Composed("not_applicable" if all_na else "unavailable", Decimal(0), 0, len(parts), False)
    summary = sum((p.value for p in parts if p.kind == "summary_only"), Decimal(0))
    return Composed(
        kind="composed",
        value=sum((p.value for p in live), Decimal(0)),
        recorded_seasons=len(live),
        unrecorded_seasons=len(parts) - len(live),
        single_sourced=any(p.single_sourced for p in live),
        summary_part=summary,
        has_summary_only=any(p.kind == "summary_only" for p in parts),
    )


@dataclass(frozen=True)
class LocalAcc:
    total: Decimal
    n_values: int
    n_null: int
    #: the local layer stores a season-level value for this aggregate (a season award row)
    stores_summary: bool = False


@dataclass(frozen=True)
class Judgement:
    outcome: str
    detail: str = ""


def judge(composed: Composed, local: LocalAcc) -> Judgement:
    """Local aggregate versus the composed source reference (full scope; no inner join)."""
    if composed.kind == "not_applicable":
        return Judgement("not_applicable")
    if composed.kind == "unavailable":
        return Judgement("source_unavailable")
    if composed.kind == "unresolved":
        return Judgement("unresolved")
    if local.total == composed.value:
        return Judgement("equal")
    if composed.has_summary_only and composed.summary_part > 0 and not local.stores_summary:
        return Judgement(
            "local_missing_summary",
            f"source reference {composed.value} (season-summary part {composed.summary_part}) "
            f"but the local layer holds {local.total} and stores no season aggregate",
        )
    return Judgement("mismatch", f"source reference {composed.value} local {local.total}")


def check_printed(printed: str | None, composed: Composed) -> str:
    """A printed season/stint/career total against the composed reference: equal, mismatch, unresolved, malformed."""
    text = (printed or "").strip()
    if text and not _DIGITS.match(text):
        return "malformed"
    if composed.kind == "unresolved":
        return "unresolved"
    if composed.kind in ("unavailable", "not_applicable"):
        return "equal" if not text else ("unresolved" if composed.single_sourced else "mismatch")
    expected = composed.value
    got = Decimal(text) if text else Decimal(0)
    if got == expected:
        return "equal"
    return "unresolved" if composed.single_sourced else "mismatch"


# -- printed averages (S-03) -----------------------------------------------------------------


_HALF = Decimal("0.005")


def average_matches(total: Decimal, denominator: Decimal | int, printed: str) -> bool:
    """The exact rational ``total / denominator`` lies in the printed two-decimal value's rounding interval.

    The interval is closed: an exact decimal tie (for example 3.425) is consistent with both neighbours.
    Real pages print ties both ways (the capture shows 28.125 printed as 28.13 and 3.425 printed as 3.42),
    so no single tie-break convention is claimed; a value outside the interval is always a miss.
    """
    den = Decimal(denominator)
    if den <= 0:
        return False
    try:
        want = Decimal(printed)
    except ArithmeticError:
        return False
    value = total / den
    return want - _HALF <= value <= want + _HALF


def season_denominator(field_name: str, acc: StatAcc, *, home_and_away: int) -> int:
    """Games whose value the source could print: recorded games plus credited did-not-take-the-field games;
    Brownlow votes use home-and-away games. (Seasons with documented missing matches, e.g. 1975, show the
    denominator is not simply the games played.)"""
    if field_name == "brownlow_votes":
        return home_and_away
    return acc.n_value + acc.n_zero + acc.n_dntf


def career_denominator(field_name: str, acc: StatAcc, *, home_and_away_in_award_seasons: int) -> int:
    if field_name == "brownlow_votes":
        return home_and_away_in_award_seasons
    return acc.n_value + acc.n_zero + acc.n_dntf


@dataclass(frozen=True)
class AverageResult:
    outcome: str  # ok | miss | unresolved | blank
    candidates: tuple[tuple[str, bool], ...] = field(default=())


def check_career_average(
    field_name: str,
    *,
    printed: str,
    acc: StatAcc,
    games: int,
    home_and_away_in_award_seasons: int,
    composed_total: Decimal | None = None,
) -> AverageResult:
    """``composed_total``: the career value composed from per-game cells and season-summary-only values (S-01);
    when given it is the numerator, so a career whose votes are printed only per season is judged on them."""
    if acc.n_bad:
        return AverageResult("unresolved")
    if not printed.strip():
        return AverageResult("blank")
    total = Decimal(acc.total) if composed_total is None else composed_total
    model_den = career_denominator(field_name, acc, home_and_away_in_award_seasons=home_and_away_in_award_seasons)
    ok = average_matches(total, model_den, printed)
    if ok:
        return AverageResult("ok", (("model", True),))
    candidates = (
        ("model", False),
        ("games", average_matches(total, games, printed)),
        ("recorded_games", average_matches(total, acc.recorded, printed)),
        ("printed_value_games", average_matches(total, acc.n_value, printed)),
        ("home_and_away_award_seasons", average_matches(total, home_and_away_in_award_seasons, printed)),
    )
    return AverageResult("miss", candidates)


def games_average_ok(total_games: int, seasons: int, printed: str) -> bool:
    return average_matches(Decimal(total_games), seasons, printed)


def win_pct_ok(wins: int, draws: int, games: int, printed: str) -> bool:
    if games <= 0:
        return False
    value = (Decimal(wins) + Decimal(draws) / 2) * 100 / Decimal(games)
    try:
        return value.quantize(_TWO, rounding=ROUND_HALF_UP) == Decimal(printed.rstrip("%"))
    except ArithmeticError:
        return False
