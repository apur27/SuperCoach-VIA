"""One season's comparison of source appearances with one local layer (DESIGN sections 7-9).

``compare_season`` is a pure function of its input: a captured season (match facts and the
profiles' game rows for that season), the rules and notes, and one layer's local rows already
attributed to a profile. It returns additive counters, complete findings and the partial
aggregates the per-player reducer needs. Every source appearance is matched, missing or
unresolved; every cell of every requested appearance lands in exactly one bucket.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from supercoach_via.integrity.sourcepages import FINAL_TOKENS
from supercoach_via.reconciliation import avgevidence as AV
from supercoach_via.reconciliation import cells as C
from supercoach_via.reconciliation.aggregate import Num, StatAcc, compact
from supercoach_via.reconciliation.findings import make_finding
from supercoach_via.reconciliation.rules import Rules
from supercoach_via.reconciliation.schema import LABEL_TO_FIELD, STAT_FIELDS
from supercoach_via.reconciliation.source import (
    MatchFacts,
    NotesFacts,
    PlayerDetail,
    ProfileGame,
    SummaryRow,
    counter_of,
)

FINAL_NAMES = frozenset(FINAL_TOKENS.values())
_N = len(STAT_FIELDS)
_PCT_I = STAT_FIELDS.index("time_on_ground_pct")
_BR_I = STAT_FIELDS.index("brownlow_votes")


def season_records_brownlow(matches: dict[str, MatchFacts]) -> bool:
    """The season's home-and-away match pages print per-game Brownlow votes at all (a team total or a player cell).
    A structural fact derived from the captured pages: only then can a blank cell be a recorded zero (A8)."""
    for m in matches.values():
        if m.stage_text is None or not m.stage_text.isdigit():
            continue
        if any(t[_BR_I] not in ("", None) for t in m.totals.values()) or any(
            p.cells[_BR_I] not in ("", None) for p in m.players
        ):
            return True
    return False


@dataclass(frozen=True)
class QRow:
    """A quarantined local appearance row (kept visible: quarantine is neither acceptance nor absence)."""

    quarantine_id: str
    player_key: str
    season: int
    club: str
    opponent: str | None
    counter: int | None
    candidates: tuple[str, ...]


@dataclass
class ProfileView:
    url: str
    name: str | None
    games: list[ProfileGame]
    #: (club) -> canonical fields whose profile column is duplicated in that club-season table
    bad_fields: dict[str, frozenset[str]]
    sha256: str
    #: this season's profile games dropped because the source match is dated after the event boundary
    excluded_games: list[ProfileGame] = field(default_factory=list)
    #: club -> (printed season totals row, printed season averages row) for this season
    season_rows: dict[str, tuple[SummaryRow | None, SummaryRow | None]] = field(default_factory=dict)


@dataclass
class SeasonInput:
    layer: str
    season: int
    matches: dict[str, MatchFacts]
    match_sha: dict[str, str]
    #: in-scope source matches without usable facts -> reason
    absent_matches: dict[str, str]
    profiles: dict[str, ProfileView]
    notes: NotesFacts
    rules: Rules
    #: snapshot layer: (profile url, source match url) -> local rows attributed there
    local_by_pair: dict[tuple[str, str], list[Any]] = field(default_factory=dict)
    #: legacy layer: profile url -> that profile's local rows for this season
    local_by_profile: dict[str, list[Any]] = field(default_factory=dict)
    #: local rows dated after the event boundary (excluded from the audit, counted)
    local_excluded: int = 0
    #: local rows that could not be attributed to a profile/match (reason)
    unplaced: list[tuple[Any, str]] = field(default_factory=list)
    local_ids: dict[str, str] = field(default_factory=dict)
    quarantine: dict[str, list[QRow]] = field(default_factory=dict)
    #: "mapped" (a local player resolved to the profile) | "missing" (no local candidate) | "unresolved" | "conflict"
    profile_status: dict[str, str] = field(default_factory=dict)
    #: every usable captured profile URL (a lineup player absent from here is a capture gap, not a conflict)
    captured: frozenset[str] = frozenset()
    #: emit source-only facts (layer "source") from this unit; exactly one layer's units should set it
    emit_source: bool = False
    #: source match URLs whose per-appearance outcome is recorded (e.g. the latest completed final)
    track_matches: frozenset[str] = frozenset()
    #: (profile url, club) -> statistic -> season value the local layer stores for that club-season
    local_summary: dict[tuple[str, str], dict[str, Decimal]] = field(default_factory=dict)


@dataclass
class ClubSeasonAgg:
    club: str
    season: int
    games: int = 0
    home_and_away: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    src: list[StatAcc] = field(default_factory=lambda: [StatAcc() for _ in range(_N)])
    #: local sums per statistic: (total, n_values, n_null)
    loc_total: list[Num] = field(default_factory=lambda: [0] * _N)
    #: local values stored where the source records none: not in ``loc_total``, but they do reproduce a
    #: season-summary-only statistic (the local layer holds the value somewhere in that club-season)
    loc_uncounted: list[Num] = field(default_factory=lambda: [0] * _N)
    #: season-level values the local layer stores (``player_season_awards`` / the legacy awards file); they
    #: reproduce a season-summary-only statistic and nothing else
    loc_summary: list[Num] = field(default_factory=lambda: [0] * _N)
    loc_values: list[int] = field(default_factory=lambda: [0] * _N)
    loc_null: list[int] = field(default_factory=lambda: [0] * _N)
    loc_rows: int = 0
    #: where each summed local row lives (fragment#row or csv#row), so an aggregate finding names its rows (F-M4)
    loc_origins: list[str] = field(default_factory=list)
    #: per-statistic evidence for season-summary-only: all per-game profile cells blank, every team total blank
    pg_blank: list[bool] = field(default_factory=lambda: [True] * _N)
    totals_blank: list[bool] = field(default_factory=lambda: [True] * _N)
    totals_checked: int = 0
    #: source games of this club-season dropped by the event boundary: printed totals then span out-of-scope games
    excluded: int = 0


@dataclass
class SeasonResult:
    layer: str
    season: int
    counters: Counter[str] = field(default_factory=Counter)
    by_stat: Counter[tuple[str, str]] = field(default_factory=Counter)
    #: per-profile tallies of ``PROFILE_KEYS`` (a fixed-size list, not a Counter: there are hundreds of thousands)
    by_profile: dict[str, list[int]] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)
    club_seasons: dict[tuple[str, str], ClubSeasonAgg] = field(default_factory=dict)
    #: (profile url, match url) -> matched | missing | unresolved, for ``SeasonInput.track_matches``
    tracked: dict[tuple[str, str], str] = field(default_factory=dict)
    #: (profile url, declared date quality) -> (match url, source date, local date, local origin) of differing dates
    date_mismatch: dict[tuple[str, str], list[tuple[str, str, str, str]]] = field(default_factory=dict)
    #: where the unit's findings were written (set by the driver; ``findings`` is then emptied)
    chunk: Any = None


# ---------------------------------------------------------------------------
# Context for one (match, team)
# ---------------------------------------------------------------------------


def _round_of(stage_text: str | None) -> int | None:
    return int(stage_text) if stage_text and stage_text.isdigit() else None


def _alias_names(rules: Rules, season: int, names: tuple[str, ...]) -> set[str]:
    return {rules.club_alias(season, n) for n in names}


def _exception_fields(
    notes: NotesFacts, rules: Rules, season: int, rnd: int | None, team: str, opp: str
) -> frozenset[str]:
    if rnd is None:
        return frozenset()
    out: set[str] = set()
    for e in notes.exceptions:
        if e.season != season or not (e.round_lo <= rnd <= e.round_hi):
            continue
        applies = False
        if e.scope == "all":
            applies = True
        elif e.scope == "teams":
            applies = team in _alias_names(rules, season, e.teams)
        else:
            pair = {team, opp}
            applies = any({rules.club_alias(season, a), rules.club_alias(season, b)} == pair for a, b in e.matchups)
        if applies:
            if e.category == "all_but_goals":
                out |= {f for f in STAT_FIELDS if f != "goals"}
            else:
                out.add(e.category)
    return frozenset(out)


@dataclass(frozen=True)
class TeamCtx:
    usable: bool
    column_present: frozenset[str]
    totals: tuple[str | None, ...] | None
    goals: int | None
    behinds: int | None
    pct_recorded: bool
    exception_fields: frozenset[str]
    bad_fields: frozenset[str]
    opponent: str | None
    won: str | None
    opp_totals: tuple[str | None, ...] | None = None  # "W" | "L" | "D" for this team from the printed final scores


def team_ctx(m: MatchFacts, team: str, season: int, notes: NotesFacts, rules: Rules) -> TeamCtx:
    names = [t.name for t in m.teams]
    opp = next((n for n in names if n != team), None)
    cols = m.columns.get(team)
    usable = cols is not None and team not in m.bad_tables and team in m.totals
    column_present = frozenset(LABEL_TO_FIELD[lab] for lab in (cols or ()) if lab in LABEL_TO_FIELD)
    line = next((t for t in m.teams if t.name == team), None)
    goals = line.quarters[-1][0] if line else None
    behinds = line.quarters[-1][1] if line else None
    other = next((t for t in m.teams if t.name == opp), None)
    won: str | None = None
    if line and other:
        a, b = line.points[-1], other.points[-1]
        won = "W" if a > b else "L" if a < b else "D"
    pct = any(p.cells[_PCT_I] not in ("", None) for p in m.players if p.team == team)
    return TeamCtx(
        usable=usable,
        column_present=column_present,
        totals=m.totals.get(team),
        goals=goals,
        behinds=behinds,
        pct_recorded=pct,
        exception_fields=_exception_fields(notes, rules, season, _round_of(m.stage_text), team, opp or ""),
        bad_fields=frozenset(m.bad_fields.get(team, ())),
        opponent=opp,
        won=won,
        opp_totals=m.totals.get(opp) if opp else None,
    )


def appearance_ctx(
    season: int,
    is_final: bool,
    tc: TeamCtx | None,
    in_lineup: bool,
    notes: NotesFacts,
    rules: Rules,
    profile_bad: frozenset[str],
    br_season_recorded: bool = False,
) -> C.AppearanceCtx:
    if season < min(notes.availability, default=10**9):
        available: frozenset[str] | None = None
    elif season in notes.availability:
        available = frozenset(notes.availability[season])
    elif season > max(notes.availability, default=0):
        available = frozenset(STAT_FIELDS)  # the notes state every category is complete from the table's end
    else:
        available = frozenset()
    return C.AppearanceCtx(
        season=season,
        is_final=is_final,
        match_usable=tc is not None and tc.usable,
        player_in_lineup=in_lineup,
        column_present=tc.column_present if tc else frozenset(),
        team_totals=tc.totals if tc else None,
        team_goals=tc.goals if tc else None,
        team_behinds=tc.behinds if tc else None,
        available=available,
        exception_fields=tc.exception_fields if tc else frozenset(),
        bad_fields=(tc.bad_fields if tc else frozenset()) | profile_bad,
        team_pct_recorded=tc.pct_recorded if tc else False,
        one_sided_rules=rules.one_sided,
        notes_usable=bool(notes.availability) and not any(p.startswith("NOTES_MATRIX") for p in notes.problems),
        opp_totals=tc.opp_totals if tc else None,
        br_no_award=season in rules.no_award_seasons,
        br_award_total=rules.br_award_total(season),
        br_season_recorded=br_season_recorded,
    )


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

_CELL_BUCKET = {
    "equal": "cell_equal",
    "equal_blank_as_zero": "cell_equal",
    "mismatch": "cell_mismatch",
    "mismatch_local_null": "cell_mismatch",
    "malformed_local": "cell_mismatch",
    "source_unavailable": "cell_source_unavailable",
    "unsupported_local_numeric": "cell_source_unavailable",
    "not_applicable": "cell_not_applicable",
    "not_applicable_dntf": "cell_not_applicable_dntf",
    "unresolved": "cell_unresolved",
}


def _digits(token: str | None) -> int | None:
    return counter_of(token) if token else None


def _stage_name(token: str) -> str:
    """A round token as the stage it names: ``GF`` and ``Grand Final`` are the same stage."""
    return FINAL_TOKENS.get(token, token)


def _stage_matches(rd_token: str, stage_text: str | None) -> bool:
    if stage_text is None:
        return True
    if rd_token in FINAL_TOKENS:
        return FINAL_TOKENS[rd_token] == stage_text
    return rd_token == stage_text


PROFILE_KEYS = (
    "app_expected",
    "app_matched",
    "app_missing_local",
    "app_unresolved",
    "app_local_only",
    "cells_expected",
    "cell_equal",
    "cell_mismatch",
    "cell_source_unavailable",
    "cell_not_applicable",
    "cell_not_applicable_dntf",
    "cell_unresolved",
)
_PK = {k: i for i, k in enumerate(PROFILE_KEYS)}


class _Tally:
    def __init__(self, res: SeasonResult, purl: str) -> None:
        self.res = res
        self.purl = purl

    def bump(self, key: str, n: int = 1) -> None:
        self.res.counters[key] += n
        i = _PK.get(key)
        if i is not None:
            row = self.res.by_profile.get(self.purl)
            if row is None:
                row = self.res.by_profile[self.purl] = [0] * len(PROFILE_KEYS)
            row[i] += n

    def stat(self, field_name: str, bucket: str) -> None:
        self.res.by_stat[(field_name, bucket)] += 1


def _player(inp: SeasonInput, pv: ProfileView) -> dict[str, Any]:
    return {"source_url": pv.url, "local_id": inp.local_ids.get(pv.url), "name": pv.name}


@dataclass
class RowPairing:
    """Local rows paired to a profile's source rows, plus the rows that paired to none."""

    by_source: dict[int, list[Any]]
    leftover: list[Any]

    def rows_for(self, g: ProfileGame) -> list[Any]:
        return self.by_source.get(id(g), [])


def pair_snapshot(inp: SeasonInput, pv: ProfileView) -> RowPairing:
    by_source: dict[int, list[Any]] = {}
    used: set[str] = set()
    for g in pv.games:
        murl = g.match_url or ""
        if murl and murl not in used:
            used.add(murl)
            by_source[id(g)] = sorted(inp.local_by_pair.get((pv.url, murl), []), key=lambda r: r.origin)
    leftover = [
        r
        for (u, m), rows in sorted(inp.local_by_pair.items())
        if u == pv.url and m not in used
        for r in sorted(rows, key=lambda r: r.origin)
    ]
    return RowPairing(by_source, leftover)


def pair_legacy(inp: SeasonInput, pv: ProfileView) -> RowPairing:
    """Pair raw CSV rows to profile rows on (club, opponent, round token). A repeated key (a drawn final and its
    replay) is paired by career counter; anything unpaired stays visible as a leftover."""
    local = sorted(inp.local_by_profile.get(pv.url, []), key=lambda r: r.origin)
    gone = {(g.club, g.opponent, g.rd_token) for g in pv.excluded_games}
    local = [r for r in local if (r.club, r.opponent or "", r.stage) not in gone]
    src_groups: dict[tuple[str, str, str], list[ProfileGame]] = defaultdict(list)
    for g in pv.games:
        src_groups[(g.club, g.opponent, g.rd_token)].append(g)
    loc_groups: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for r in local:
        loc_groups[(r.club, r.opponent or "", r.stage)].append(r)
    by_source: dict[int, list[Any]] = {}
    leftover: list[Any] = []
    for key, rows in sorted(loc_groups.items()):
        sg = src_groups.get(key, [])
        if len(sg) == 1:
            by_source[id(sg[0])] = rows
        elif len(sg) > 1:
            by_counter = {g.counter: g for g in sg}
            for r in rows:
                g2 = by_counter.get(r.counter)
                if g2 is not None and id(g2) not in by_source:
                    by_source[id(g2)] = [r]
                else:
                    leftover.append(r)
        else:
            leftover.extend(rows)
    return RowPairing(by_source, leftover)


def _agg(res: SeasonResult, purl: str, club: str, season: int) -> ClubSeasonAgg:
    key = (purl, club)
    a = res.club_seasons.get(key)
    if a is None:
        a = res.club_seasons[key] = ClubSeasonAgg(club, season)
    return a


def _evidence(
    pv: ProfileView, g: ProfileGame, facts: MatchFacts | None, inp: SeasonInput, mp: Any, field_name: str | None
) -> dict[str, Any]:
    ev: dict[str, Any] = {
        "source_url": pv.url,
        "body_sha256": pv.sha256,
        "locator": f"table {g.table} row {g.row}" + (f" column {field_name}" if field_name else ""),
    }
    if facts is not None:
        ev["match_url"] = facts.url
        ev["match_body_sha256"] = inp.match_sha.get(facts.url)
        if mp is not None:
            ev["match_locator"] = f"{mp.team} row {mp.row}" + (f" column {field_name}" if field_name else "")
    return ev


def _local_ref(r: Any) -> dict[str, Any]:
    return {"origin": r.origin, "player_key": r.player_key, "match_key": r.match_key, "layer": r.layer}


def _source_side(
    inp: SeasonInput,
    res: SeasonResult,
    t: _Tally,
    pv: ProfileView,
    g: ProfileGame,
    facts: MatchFacts | None,
    mp: Any,
    tc: TeamCtx | None,
    cells: list[C.Cell],
    details: dict[tuple[str, str], PlayerDetail],
) -> None:
    agg = _agg(res, pv.url, g.club, inp.season)
    agg.games += 1
    is_final = g.rd_token in FINAL_TOKENS
    if not is_final:
        agg.home_and_away += 1
    if g.result == "W":
        agg.wins += 1
    elif g.result == "D":
        agg.draws += 1
    elif g.result == "L":
        agg.losses += 1
    if tc is not None and tc.totals is not None:
        agg.totals_checked += 1
    conflict_fields: list[str] = []
    malformed_fields: list[str] = []
    for i, c in enumerate(cells):
        f = STAT_FIELDS[i]
        agg.src[i].add(c, double_sourced=c.rule == "R-BOTH-PAGES")
        if g.cells[i] not in ("", None):
            agg.pg_blank[i] = False
        if tc is None or tc.totals is None or tc.totals[i] not in ("", None):
            agg.totals_blank[i] = False
        res.counters[f"src_{c.state.value}"] += 1
        res.counters[f"srcrule_{c.state.value}:{c.rule}"] += 1
        if c.state in (C.St.UNRESOLVED, C.St.CONFLICT, C.St.MALFORMED):
            res.by_stat[(f, f"srcrule_{c.state.value}:{c.rule}")] += 1
        res.by_stat[(f, f"src_{c.state.value}")] += 1
        if c.state is C.St.CONFLICT:
            conflict_fields.append(f)
        elif c.state is C.St.MALFORMED:
            malformed_fields.append(f)
    if all(c.state is C.St.DNTF for c in cells):
        t.bump("src_appearances_dntf")
    if not inp.emit_source:
        return

    def src_finding(cat: str, rule: str, field_name: str, detail: str) -> dict[str, Any]:
        return make_finding(
            cat,
            layer="source",
            rule_id=rule,
            player=_player(inp, pv),
            season=inp.season,
            match={"source_url": g.match_url},
            field=field_name,
            evidence=_evidence(pv, g, facts, inp, mp, None),
            detail=detail,
        )

    if conflict_fields:
        res.counters["source_conflict_cells"] += len(conflict_fields)
        res.findings.append(
            src_finding(
                "SOURCE_CONFLICT", "R-SOURCE-CONFLICT", ",".join(conflict_fields), "profile and match cells disagree"
            )
        )
    if malformed_fields:
        rules = sorted({c.rule for c in cells if c.state is C.St.MALFORMED})
        res.counters["source_malformed_cells"] += len(malformed_fields)
        res.findings.append(src_finding("SCHEMA_GAP", rules[0], ",".join(malformed_fields), ";".join(rules)))
    if facts is None:
        return
    problems: list[tuple[str, str]] = []
    if mp is None:
        problems.append(("lineup", f"{pv.url} is not listed for {g.club} on the match page"))
    else:
        if mp.team != g.club:
            problems.append(("club", f"profile {g.club!r} match page {mp.team!r}"))
        if tc is not None and tc.opponent is not None and g.opponent != tc.opponent:
            problems.append(("opponent", f"profile {g.opponent!r} match page {tc.opponent!r}"))
        if tc is not None and tc.won is not None and g.result != tc.won:
            problems.append(("result", f"profile {g.result!r} match page {tc.won!r}"))
        if _digits(g.jersey_token) != _digits(mp.jersey_token):
            problems.append(("jersey", f"profile {g.jersey_token!r} match page {mp.jersey_token!r}"))
        # the profile's career counter against the match page's career games to date (DESIGN S-07, T14)
        detail = details.get((mp.team, pv.url))
        if detail is None or detail.career_games is None or g.counter is None:
            res.counters["games_to_date_absent"] += 1
        else:
            res.counters["games_to_date_checked"] += 1
            if detail.career_games != g.counter:
                problems.append(("counter", f"profile {g.counter} match page {detail.career_games}"))
    if not _stage_matches(g.rd_token, facts.stage_text):
        problems.append(("round", f"profile {g.rd_token!r} match page {facts.stage_text!r}"))
    for fld, why in problems:
        res.counters["source_conflict_attrs"] += 1
        res.findings.append(src_finding("SOURCE_CONFLICT", f"R-SOURCE-{fld.upper()}", fld, why))


def _attr_pairs(g: ProfileGame, r: Any, facts: MatchFacts | None) -> list[tuple[str, Any, Any]]:
    """(attribute, source value, local value) for the attributes both layers carry."""
    out: list[tuple[str, Any, Any]] = [
        ("club", g.club, r.club),
        ("opponent", g.opponent, r.opponent),
        ("result", g.result, r.result),
        ("jersey", _digits(g.jersey_token), _digits(r.jersey)),
        ("counter", g.counter, r.counter),
        ("counter_token", g.counter_token, r.counter_token),
        ("stage", _stage_name(g.rd_token), _stage_name(r.stage)),
    ]
    if facts is not None and facts.match_date is not None and r.match_date:
        out.append(("date", facts.match_date, r.match_date))
    return out


def _date_not_compared(facts: MatchFacts | None, r: Any) -> bool:
    """The source dates the match but the local row stores no date: not compared, never a mismatch (A6)."""
    return facts is not None and facts.match_date is not None and not r.match_date


def _local_side(
    inp: SeasonInput,
    res: SeasonResult,
    t: _Tally,
    pv: ProfileView,
    g: ProfileGame,
    facts: MatchFacts | None,
    mp: Any,
    cells: list[C.Cell],
    rows: list[Any],
    status: str,
    pending_missing: list[tuple[ProfileGame, Any]],
    blank_zero: bool,
) -> None:
    t.bump("app_expected")
    t.bump("cells_expected", _N)
    layer = inp.layer
    player = _player(inp, pv)
    match = {"source_url": g.match_url}
    tracked = g.match_url in inp.track_matches
    if status in ("unresolved", "conflict"):
        t.bump("app_unresolved")
        t.bump("cell_in_unresolved_appearance", _N)
        if tracked:
            res.tracked[(pv.url, g.match_url or "")] = "unresolved"
        return
    if status == "missing" or not rows:
        if tracked:
            res.tracked[(pv.url, g.match_url or "")] = "missing"
        t.bump("app_missing_local")
        t.bump("cell_in_missing_appearance", _N)
        pending_missing.append((g, (status, mp, facts)))
        return
    primary = rows[0]
    if tracked:
        res.tracked[(pv.url, g.match_url or "")] = "matched"
    t.bump("app_matched")
    agg = _agg(res, pv.url, g.club, inp.season)
    for k, r in enumerate(rows):
        _local_agg(agg, r, cells if k == 0 else None)
    for extra in rows[1:]:
        t.bump("app_duplicate_local")
        same = extra.cells == primary.cells
        res.findings.append(
            make_finding(
                "APPEARANCE_DUPLICATE_LOCAL",
                layer=layer,
                rule_id="R-DUPLICATE-LOCAL",
                player=player,
                season=inp.season,
                match=match,
                local=_local_ref(extra),
                evidence=_evidence(pv, g, facts, inp, mp, None),
                detail="equal values" if same else "differing values",
            )
        )
    if _date_not_compared(facts, primary):
        t.bump("attr_date_not_compared")
    for attr, sv, lv in _attr_pairs(g, primary, facts):
        if sv == lv:
            t.bump("attr_equal")
            continue
        if attr == "counter_token" and layer == "legacy_csv":
            # the legacy games_played column cannot carry a sub-on/off arrow: a representation limit, counted
            t.bump("attr_legacy_token_arrow_dropped")
            continue
        t.bump("attr_mismatch")
        t.bump(f"attr_mismatch_{attr}")
        if attr == "date":
            # a stored date that differs from the source is a wrong value whatever quality flag the row declares
            # (DESIGN section 15, A6); grouped per (player, quality) to bound the stream, counted per appearance
            res.date_mismatch.setdefault((pv.url, primary.date_quality or ""), []).append(
                (g.match_url or "", str(sv), str(lv), primary.origin)
            )
            continue
        res.findings.append(
            make_finding(
                "APPEARANCE_ATTR_MISMATCH",
                layer=layer,
                rule_id=f"R-ATTR-{attr.upper()}",
                player=player,
                season=inp.season,
                match=match,
                field=attr,
                expected=sv,
                actual=lv,
                local=_local_ref(primary),
                evidence=_evidence(pv, g, facts, inp, mp, None),
            )
        )
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for i, c in enumerate(cells):
        f = STAT_FIELDS[i]
        lv = primary.cells[i]
        out = C.compare_local(c, lv, blank_is_zero=blank_zero)
        bucket = _CELL_BUCKET[out.outcome]
        t.bump(bucket)
        t.stat(f, bucket)
        if c.state is C.St.ZERO:
            t.bump("cell_recorded_zero")
        if out.outcome == "equal":
            t.bump("cell_equal_zero" if c.state is C.St.ZERO else "cell_equal_value")
        elif out.outcome == "equal_blank_as_zero":
            t.bump("cell_equal_zero")
            t.bump("cell_equal_blank_as_zero")
        elif out.outcome in ("mismatch", "mismatch_local_null", "malformed_local"):
            if out.outcome != "mismatch":  # the plain mismatch is the bucket itself
                t.bump(f"cell_{out.outcome}")
            cat = {
                "mismatch": "CELL_MISMATCH",
                "mismatch_local_null": "CELL_LOCAL_NULL",
                "malformed_local": "LOCAL_CELL_MALFORMED",
            }[out.outcome]
            res.findings.append(
                make_finding(
                    cat,
                    layer=layer,
                    rule_id=c.rule,
                    player=player,
                    season=inp.season,
                    match=match,
                    field=f,
                    expected=None if c.value is None else str(c.value),
                    actual=None if isinstance(lv, str) else lv,
                    expected_raw=g.cells[i],
                    actual_raw=(primary.raw_cells[i] if primary.raw_cells else lv),
                    evidence=_evidence(pv, g, facts, inp, mp, f),
                    local=_local_ref(primary),
                    detail=out.detail,
                )
            )
        elif out.outcome == "unsupported_local_numeric":
            t.bump("cell_unsupported_local_numeric")
            grouped[("LOCAL_UNSUPPORTED_NUMERIC", c.rule)].append(f)
        elif out.outcome == "unresolved":
            t.bump(f"cell_unresolved_{c.state.value}")
            grouped[("CELL_UNRESOLVED", f"{c.rule}")].append(f)
        elif out.outcome == "not_applicable_dntf":
            t.bump("dntf_local_null" if lv is None else "dntf_local_zero")
    for (cat, rule), fields in sorted(grouped.items()):
        res.findings.append(
            make_finding(
                cat,
                layer=layer,
                rule_id=rule,
                player=player,
                season=inp.season,
                match=match,
                field=",".join(fields),
                local=_local_ref(primary),
                evidence=_evidence(pv, g, facts, inp, mp, None),
                detail=rule,
            )
        )
    if any(c.state is C.St.DNTF for c in cells):
        nulls = sum(1 for v in primary.cells if v is None)
        res.findings.append(
            make_finding(
                "REPRESENTATION",
                layer=layer,
                rule_id="R-DNTF",
                player=player,
                season=inp.season,
                match=match,
                local=_local_ref(primary),
                evidence=_evidence(pv, g, facts, inp, mp, None),
                detail=f"credited game, player did not take the field; local {nulls} null / {_N - nulls} zero",
            )
        )


_UNCOUNTED = (C.St.NOT_RECORDED, C.St.NA, C.St.DNTF)


def _local_agg(agg: ClubSeasonAgg, r: Any, cells: list[C.Cell] | None = None) -> None:
    """Add a local row to the club-season's local totals. A value the local layer stores where the source records
    none (or a did-not-take-the-field game) is reported as a cell finding and is not added to the total, so it
    cannot masquerade as an aggregate disagreement."""
    agg.loc_rows += 1
    agg.loc_origins.append(r.origin)
    for i in range(_N):
        if cells is not None and cells[i].state in _UNCOUNTED:
            u = C.local_decimal(r.cells[i])
            if isinstance(u, Decimal):
                agg.loc_uncounted[i] += compact(u)
            continue
        v = C.local_decimal(r.cells[i])
        if v is None:
            agg.loc_null[i] += 1
        elif isinstance(v, Decimal):
            agg.loc_total[i] += compact(v)
            agg.loc_values[i] += 1


def _leftover_local(
    inp: SeasonInput,
    res: SeasonResult,
    t: _Tally,
    pv: ProfileView,
    pairing: RowPairing,
    pending_missing: list[tuple[ProfileGame, Any]],
) -> None:
    layer = inp.layer
    player = _player(inp, pv)
    free = list(pending_missing)
    qrows = inp.quarantine.get(pv.url, [])
    for r in pairing.leftover:
        # a local row that is the same game under a different match link
        partner = next(
            (
                (g, extra)
                for g, extra in free
                if g.club == r.club and (g.opponent == r.opponent) and g.rd_token == r.stage
            ),
            None,
        )
        t.bump("app_local_only")
        _local_agg(_agg(res, pv.url, partner[0].club if partner else r.club, inp.season), r)
        if partner is not None:
            free.remove(partner)
            t.bump("app_missing_wrong_link")
            g = partner[0]
            res.findings.append(
                make_finding(
                    "MATCH_LINK_MISMATCH",
                    layer=layer,
                    rule_id="R-MATCH-LINK",
                    player=player,
                    season=inp.season,
                    match={"source_url": g.match_url, "local_id": r.match_key},
                    local=_local_ref(r),
                    evidence=_evidence(pv, g, partner[1][2], inp, partner[1][1], None),
                    detail=f"local row is attached to {r.match_key}; the source appearance is {g.match_url}",
                )
            )
            pending_missing[:] = [x for x in pending_missing if x[0] is not g]
            res.counters["app_missing_local"] += 0
            continue
        facts_listing = [
            m
            for m in inp.matches.values()
            if r.match_key and any(p.link == pv.url for p in m.players) and m.url == r.match_key
        ]
        if facts_listing:
            t.bump("app_local_only_source_conflict")
            continue
        res.findings.append(
            make_finding(
                "APPEARANCE_EXTRA_LOCAL",
                layer=layer,
                rule_id="R-EXTRA-LOCAL",
                player=player,
                season=inp.season,
                match={"local_id": r.match_key},
                local=_local_ref(r),
                evidence={"source_url": pv.url, "body_sha256": pv.sha256, "locator": "no matching game row"},
                detail="no source row for this player and match",
            )
        )
    for g, (_status, mp, facts) in pending_missing:
        q = next((x for x in qrows if x.club == g.club and x.opponent == g.opponent and x.counter == g.counter), None)
        if q is not None:
            t.bump("app_missing_quarantined")
            res.findings.append(
                make_finding(
                    "APPEARANCE_QUARANTINED",
                    layer=layer,
                    rule_id="R-QUARANTINE",
                    player=player,
                    season=inp.season,
                    match={"source_url": g.match_url},
                    local={"origin": q.quarantine_id},
                    evidence=_evidence(pv, g, facts, inp, mp, None),
                    detail=f"only a quarantined row exists ({q.quarantine_id}); candidates {list(q.candidates)}",
                )
            )
        else:
            res.findings.append(
                make_finding(
                    "APPEARANCE_MISSING_LOCAL",
                    layer=layer,
                    rule_id="R-MISSING-LOCAL",
                    player=player,
                    season=inp.season,
                    match={"source_url": g.match_url},
                    evidence=_evidence(pv, g, facts, inp, mp, None),
                    detail=f"source row {g.club} v {g.opponent} round {g.rd_token}",
                )
            )


def _lineup_checks(inp: SeasonInput, res: SeasonResult, seen: set[tuple[str, str]]) -> None:
    """A usable match page lists a captured profile that has no row for the match: the two source facts disagree."""
    if not inp.emit_source:
        return
    for murl, facts in sorted(inp.matches.items()):
        for p in facts.players:
            if p.link is None or (p.link in inp.profiles and (p.link, murl) in seen):
                continue
            if p.link not in inp.captured:
                res.counters["lineup_profile_not_captured"] += 1
                continue
            res.counters["source_conflict_attrs"] += 1
            res.findings.append(
                make_finding(
                    "SOURCE_CONFLICT",
                    layer="source",
                    rule_id="R-SOURCE-MEMBERSHIP",
                    player={"source_url": p.link},
                    season=inp.season,
                    match={"source_url": murl},
                    field="membership",
                    evidence={
                        "source_url": murl,
                        "body_sha256": inp.match_sha.get(murl),
                        "locator": f"{p.team} row {p.row}",
                    },
                    detail="the match page lists the player; the profile has no game row for it",
                )
            )


def season_rows_of(core: Any, season: int) -> dict[str, tuple[SummaryRow | None, SummaryRow | None]]:
    """club -> (printed totals row, printed averages row) of ``season`` from a profile's summary tables."""
    out: dict[str, tuple[SummaryRow | None, SummaryRow | None]] = {}
    for r in core.season_totals:
        if r.year == season:
            out.setdefault(r.club, (r, None))
    for r in core.season_averages:
        if r.year == season:
            tot, _ = out.get(r.club, (None, None))
            out[r.club] = (tot, r)
    return out


@dataclass
class _Prepared:
    game: ProfileGame
    facts: MatchFacts | None
    mp: Any
    tc: TeamCtx | None
    cells: list[C.Cell]
    key: tuple[str, str]  # (match url, team as printed on the match page)
    is_final: bool


_AMBIGUOUS_RULE = "R-ALL-ZERO-UNPROVEN"
_BR_PARTIAL_RULE = "R-BR-AWARD-SUM-MISMATCH"


def _game_state(cell: C.Cell) -> str | None:
    """The cell's role in the source's average denominator; ``None`` when it makes the count uncertain."""
    if cell.state is C.St.VALUE:
        return "value"
    if cell.state is C.St.ZERO:
        return "zero"
    if cell.state is C.St.DNTF:
        return "dntf"
    if cell.state in (C.St.NOT_RECORDED, C.St.NA):
        return "excluded"
    if cell.state is C.St.UNRESOLVED and cell.rule == _AMBIGUOUS_RULE:
        return "ambiguous"
    return None


def _apply_printed_evidence(inp: SeasonInput, prepared: dict[str, list[_Prepared]]) -> None:
    """Resolve whole-team blank columns and partial Brownlow records from the source's own printed season figures
    (``avgevidence``); everything else is left exactly as classified."""
    votes: list[tuple[tuple[str, str], str, str]] = []
    br_complete: set[tuple[str, str]] = set()  # (profile url, club) whose printed season Brownlow total is complete
    for purl, preps in prepared.items():
        pv = inp.profiles[purl]
        by_club: dict[str, list[_Prepared]] = defaultdict(list)
        for p in preps:
            by_club[p.game.club].append(p)
        for club, ps in by_club.items():
            totals_row, avg_row = pv.season_rows.get(club, (None, None))
            for i, f in enumerate(STAT_FIELDS):
                if i in (_PCT_I, _BR_I):
                    continue
                states = [_game_state(p.cells[i]) for p in ps]
                if "ambiguous" not in states or None in states or avg_row is None:
                    continue
                games = [
                    AV.GameState(p.key, st or "", p.cells[i].value if st == "value" else None)
                    for p, st in zip(ps, states, strict=True)
                ]
                for key, vote in AV.player_votes(games, avg_row.cells[i]).items():
                    votes.append((key, f, vote))
            if totals_row is not None:
                printed_votes = [
                    p.cells[_BR_I].value for p in ps if p.cells[_BR_I].state is C.St.VALUE and p.cells[_BR_I].value
                ]
                if AV.brownlow_season_complete([v for v in printed_votes if v is not None], totals_row.cells[_BR_I]):
                    br_complete.add((purl, club))
    evidence = AV.team_evidence(votes)
    for purl, preps in prepared.items():
        for p in preps:
            for i, f in enumerate(STAT_FIELDS):
                cell = p.cells[i]
                if cell.state is not C.St.UNRESOLVED:
                    continue
                if cell.rule == _AMBIGUOUS_RULE:
                    ev = evidence.get((p.key, f))
                    if ev == "recorded":
                        p.cells[i] = C.Cell(C.St.ZERO, Decimal(0), "R-AVG-COUNTED")
                    elif ev == "unrecorded":
                        p.cells[i] = C.Cell(C.St.NOT_RECORDED, None, "R-AVG-EXCLUDED")
                    elif ev == "conflict":
                        why = "players' printed averages disagree"
                        p.cells[i] = C.Cell(C.St.UNRESOLVED, None, "R-AVG-CONFLICT", why)
                elif cell.rule == _BR_PARTIAL_RULE and (purl, p.game.club) in br_complete:
                    p.cells[i] = C.Cell(C.St.ZERO, Decimal(0), "R-BR-SEASON-TOTAL")


def compare_season(inp: SeasonInput) -> SeasonResult:
    res = SeasonResult(inp.layer, inp.season)
    cache: dict[tuple[str, str], TeamCtx] = {}
    detail_maps: dict[str, dict[tuple[str, str], PlayerDetail]] = {}
    br_recorded = season_records_brownlow(inp.matches)
    blank_zero = inp.layer == "legacy_csv"
    seen: set[tuple[str, str]] = set()
    prepared: dict[str, list[_Prepared]] = {}
    for purl in sorted(inp.profiles):
        pv = inp.profiles[purl]
        rows_p: list[_Prepared] = []
        for g in pv.games:
            murl = g.match_url or ""
            facts = inp.matches.get(murl)
            tc: TeamCtx | None = None
            mp = None
            if facts is not None:
                rows = [p for p in facts.players if p.link == purl]
                # the lineup row of this profile; a team-name difference is reported, never used to drop the row
                mp = next((p for p in rows if p.team == g.club), rows[0] if len(rows) == 1 else None)
                team = mp.team if mp is not None else g.club
                key = (murl, team)
                if key not in cache:
                    cache[key] = team_ctx(facts, team, inp.season, inp.notes, inp.rules)
                tc = cache[key]
            is_final = g.rd_token in FINAL_TOKENS
            actx = appearance_ctx(
                inp.season,
                is_final,
                tc,
                mp is not None,
                inp.notes,
                inp.rules,
                pv.bad_fields.get(g.club, frozenset()),
                br_recorded,
            )
            if g.malformed:
                cells = [C.Cell(C.St.MALFORMED, None, "R-ROW-MALFORMED", "row or table could not be read")] * _N
            else:
                cells = C.classify_appearance(g.cells, mp.cells if mp is not None else None, actx)
            team_name = mp.team if mp is not None else g.club
            rows_p.append(_Prepared(g, facts, mp, tc, cells, (murl, team_name), is_final))
        prepared[purl] = rows_p
    _apply_printed_evidence(inp, prepared)
    for purl in sorted(inp.profiles):
        pv = inp.profiles[purl]
        t = _Tally(res, purl)
        status = inp.profile_status.get(purl, "mapped")
        pairing = pair_snapshot(inp, pv) if inp.layer == "snapshot" else pair_legacy(inp, pv)
        pending: list[tuple[ProfileGame, Any]] = []
        for prep in prepared[purl]:
            g, facts, mp, tc, cells = prep.game, prep.facts, prep.mp, prep.tc, prep.cells
            murl = g.match_url or ""
            seen.add((purl, murl))
            if facts is not None and murl not in detail_maps:
                detail_maps[murl] = {(d.team, d.link): d for d in facts.player_details if d.link}
            _source_side(inp, res, t, pv, g, facts, mp, tc, cells, detail_maps.get(murl, {}))
            _local_side(inp, res, t, pv, g, facts, mp, cells, pairing.rows_for(g), status, pending, blank_zero)
        for g in pv.excluded_games:
            _agg(res, purl, g.club, inp.season).excluded += 1
            res.counters["source_games_excluded_after_boundary"] += 1
        _leftover_local(inp, res, t, pv, pairing, pending)
    _lineup_checks(inp, res, seen)
    for (purl, quality), drows in sorted(res.date_mismatch.items()):
        pview = inp.profiles[purl]
        eg = ", ".join(f"{m.rsplit('/', 1)[-1]}: source {a} local {b}" for m, a, b, _o in drows[:3])
        res.findings.append(
            make_finding(
                "APPEARANCE_DATE_MISMATCH",
                layer=inp.layer,
                rule_id="R-ATTR-DATE",
                player=_player(inp, pview),
                season=inp.season,
                field="date",
                expected=drows[0][1],
                actual=drows[0][2],
                local={
                    "origin": drows[0][3],
                    "rows": len(drows),
                    "date_quality": quality or None,
                    "changes": [[o, a, b] for _m, a, b, o in drows],
                },
                evidence={"source_url": purl, "body_sha256": pview.sha256, "locator": "game table Rd/date"},
                detail=f"{len(drows)} rows store a date that differs from the source match date; e.g. {eg}",
                extra_id=quality,
            )
        )
    res.date_mismatch = {}  # spent: they live on as findings and counters
    for (purl, club), values in sorted(inp.local_summary.items()):
        agg = res.club_seasons.get((purl, club))
        if agg is None:
            res.counters["local_summary_without_source_club_season"] += 1
            continue
        for fname, value in values.items():
            agg.loc_summary[STAT_FIELDS.index(fname)] += compact(value)
    res.counters["local_rows_excluded_after_boundary"] += inp.local_excluded
    for _row, reason in inp.unplaced:
        res.counters["local_unplaced"] += 1
        res.counters[f"local_unplaced_{reason}"] += 1
    return res
