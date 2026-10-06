"""Per-player aggregate reduction: stint (club-season), season and career (DESIGN section 8).

Inputs are the per-club-season accumulators the season units produced (source states and local
sums) plus the profile's printed summary rows and footers. Output: counters, per-statistic
outcome tallies and findings. Local layers are judged against the COMPOSED source reference
(per-game sums, season values for summary-only statistics); the source's own printed totals and
averages are checked separately as derived figures and never change a local layer's verdict.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from supercoach_via.reconciliation import aggregate as AG
from supercoach_via.reconciliation.findings import make_finding
from supercoach_via.reconciliation.rules import Rules
from supercoach_via.reconciliation.schema import STAT_FIELDS
from supercoach_via.reconciliation.season import ClubSeasonAgg
from supercoach_via.reconciliation.source import ProfileFacts, SummaryRow

_N = len(STAT_FIELDS)
_DIGITS = re.compile(r"^(?:0|[1-9][0-9]*)$")
_PCT_I = STAT_FIELDS.index("time_on_ground_pct")
_BR_I = STAT_FIELDS.index("brownlow_votes")


@dataclass
class ReduceResult:
    counters: Counter[str] = field(default_factory=Counter)
    #: (level, field, outcome) -> n, for the local-versus-source judgement
    by_stat: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    findings: list[dict[str, Any]] = field(default_factory=list)


def _local(a: ClubSeasonAgg, i: int, summary_only: bool) -> AG.LocalAcc:
    """The club-season's local total for statistic ``i``; a season-summary-only statistic is reproduced by a
    value the local layer holds anywhere in the club-season, so values on unrecorded cells count for it."""
    total = Decimal(a.loc_total[i]) + (Decimal(a.loc_uncounted[i] + a.loc_summary[i]) if summary_only else Decimal(0))
    return AG.LocalAcc(total, a.loc_values[i], a.loc_null[i], summary_only and a.loc_summary[i] != 0)


def _combine(parts: list[AG.LocalAcc], summary_only: list[bool]) -> AG.LocalAcc:
    """Season or career local total; it stores the season value only when every summary-only part does."""
    stored = [p.stores_summary for p, so in zip(parts, summary_only, strict=True) if so]
    return AG.LocalAcc(
        sum((x.total for x in parts), Decimal(0)),
        sum(x.n_values for x in parts),
        sum(x.n_null for x in parts),
        bool(stored) and all(stored),
    )


def merge_acc(into: AG.StatAcc, other: AG.StatAcc) -> None:
    into.total += other.total
    into.n_value += other.n_value
    into.n_zero += other.n_zero
    into.n_unrec += other.n_unrec
    into.n_na += other.n_na
    into.n_dntf += other.n_dntf
    into.n_bad += other.n_bad
    into.single_sourced = into.single_sourced or other.single_sourced


def _printed(row: SummaryRow | None, i: int) -> str | None:
    if row is None or i == _PCT_I:
        return None
    return row.cells[i]


def _as_decimal(text: str | None) -> Decimal | None:
    return Decimal(text) if text and _DIGITS.match(text) else None


def _summary_only(a: ClubSeasonAgg, i: int, printed: str | None) -> bool:
    return bool(
        printed
        and _DIGITS.match(printed)
        and a.games > 0
        and a.pg_blank[i]
        and a.totals_blank[i]
        and a.totals_checked == a.games
        and a.src[i].recorded == 0
        and a.src[i].n_bad == 0
    )


def _origins(aggs: dict[tuple[str, int], ClubSeasonAgg], level: str, club: str | None, season: int | None) -> list[str]:
    """The local rows summed into a judged aggregate, in a stable order."""
    keys = [k for k in aggs if (club is None or k[0] == club) and (season is None or k[1] == season)]
    if level == "career":
        keys = list(aggs)
    return sorted(o for k in keys for o in aggs[k].loc_origins)


def reduce_profile(
    *,
    layer: str,
    purl: str,
    local_id: str | None,
    core: ProfileFacts,
    aggs: dict[tuple[str, int], ClubSeasonAgg],
    rules: Rules,
    emit_source: bool,
    judge_local: bool = True,
    profile_sha256: str | None = None,
    career: bool = True,
) -> ReduceResult:
    """``judge_local=False`` for a profile no local player maps to: its absence is already a finding, so the
    local-versus-source totals are not judged (they would only restate it); the source's own figures still are."""
    out = ReduceResult()
    player = {"source_url": purl, "local_id": local_id, "name": core.h1}
    printed_rows: dict[tuple[int, str], SummaryRow] = {}
    for r in core.season_totals:
        if r.year is not None:
            printed_rows.setdefault((r.year, r.club), r)
    avg_rows: dict[tuple[int, str], SummaryRow] = {}
    for r in core.season_averages:
        if r.year is not None:
            avg_rows.setdefault((r.year, r.club), r)
    keys = sorted(aggs, key=lambda k: (k[1], k[0]))
    contribs: dict[tuple[str, int], list[AG.Contribution]] = {}
    per_game: dict[tuple[str, int], list[AG.Contribution]] = {}
    summary_flags: dict[tuple[str, int], list[bool]] = {}
    for key in keys:
        a = aggs[key]
        club, season = key
        prow = printed_rows.get((season, club))
        cs: list[AG.Contribution] = []
        pg: list[AG.Contribution] = []
        flags: list[bool] = []
        for i in range(_N):
            printed = _printed(prow, i)
            so = _summary_only(a, i, printed)
            flags.append(so)
            cs.append(AG.contribution(a.src[i], _as_decimal(printed), summary_only=so))
            pg.append(AG.contribution(a.src[i], None))
        contribs[key], per_game[key], summary_flags[key] = cs, pg, flags
        if any(flags):
            out.counters["summary_only_club_seasons"] += 1
            out.counters["summary_only_stats"] += sum(flags)

    def judge_level(
        level: str, i: int, comps: list[AG.Contribution], loc: AG.LocalAcc, season: int | None, club: str | None
    ) -> None:
        f = STAT_FIELDS[i]
        composed = AG.compose(comps)
        j = AG.judge(composed, loc)
        out.counters[f"agg_{level}_judged"] += 1
        out.counters[f"agg_{level}_{j.outcome}"] += 1
        out.by_stat[(level, f, j.outcome)] += 1
        if level == "season":
            return
        if j.outcome in ("mismatch", "local_missing_summary"):
            cat = "AGGREGATE_MISMATCH" if j.outcome == "mismatch" else "LOCAL_MISSING_SUMMARY_VALUE"
            out.findings.append(
                make_finding(
                    cat,
                    layer=layer,
                    rule_id=f"R-AGG-{level.upper()}",
                    player=player,
                    season=season,
                    field=f,
                    expected=str(composed.value),
                    actual=str(loc.total),
                    evidence={
                        "source_url": purl,
                        "body_sha256": profile_sha256,
                        "locator": f"{level} {club or ''} {season or ''} {f}".strip()
                        + " (summary Totals row / footer)",
                    },
                    local={"origins": _origins(aggs, level, club, season), "club": club},
                    detail=j.detail,
                    extra_id=[level, club],
                )
            )

    # -- stint (club-season) and season levels ---------------------------------------------------
    by_season: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for key in keys:
        by_season[key[1]].append(key)
    for key in keys if judge_local else []:
        a = aggs[key]
        out.counters["agg_stint_units"] += 1
        for i in range(_N):
            comps = [contribs[key][i]]
            if a.games == 0:  # the source shows no games here, but the local layer has rows
                composed_zero = AG.Contribution("per_game", Decimal(0), False)
                comps = [composed_zero]
            judge_level("stint", i, comps, _local(a, i, summary_flags[key][i]), key[1], key[0])
    for season, ks in sorted(by_season.items() if judge_local else []):
        out.counters["agg_season_units"] += 1
        for i in range(_N):
            parts = [_local(aggs[k], i, summary_flags[k][i]) for k in ks]
            loc = _combine(parts, [summary_flags[k][i] for k in ks])
            judge_level("season", i, [contribs[k][i] for k in ks], loc, season, None)
    # -- career ---------------------------------------------------------------------------------
    career_acc = [AG.StatAcc() for _ in range(_N)]
    for key in keys:
        for i in range(_N):
            merge_acc(career_acc[i], aggs[key].src[i])
    # a seasons audit holds only part of a career: career totals and printed career figures are not judged
    for i in range(_N if judge_local and career else 0):
        comps = [contribs[k][i] for k in keys]
        parts = [_local(aggs[k], i, summary_flags[k][i]) for k in keys]
        loc = _combine(parts, [summary_flags[k][i] for k in keys])
        judge_level("career", i, comps, loc, None, None)
        composed = AG.compose(comps)
        if composed.kind == "composed":
            out.counters["career_composed_recorded_seasons"] += composed.recorded_seasons
            out.counters["career_composed_unrecorded_seasons"] += composed.unrecorded_seasons
    out.counters["profiles_reduced"] += 1
    if judge_local and career:
        out.counters["agg_career_units"] += 1
    else:
        out.counters["profiles_not_judged_no_local_player"] += 1
    if emit_source:
        if core.counter_issues:
            out.counters["source_counter_sequence_profiles"] += 1
            out.findings.append(
                make_finding(
                    "SOURCE_CONFLICT",
                    layer="source",
                    rule_id="R-SOURCE-COUNTER-SEQUENCE",
                    player=player,
                    field="counter_sequence",
                    evidence={"source_url": purl, "locator": "game tables: career counter column"},
                    detail="; ".join(core.counter_issues),
                )
            )
        _derived_checks(
            out, player, purl, core, aggs, keys, per_game, summary_flags, avg_rows, career_acc, rules, contribs,
            career=career,
        )  # fmt: skip
    return out


def _derived(out: ReduceResult, kind: str, outcome: str) -> None:
    out.counters[f"derived_{kind}_{outcome}"] += 1


def _inconsistency(
    out: ReduceResult, player: dict[str, Any], purl: str, what: str, detail: str, season: int | None, extra: Any
) -> None:
    out.findings.append(
        make_finding(
            "SOURCE_DERIVED_INCONSISTENCY",
            layer="source",
            rule_id=f"R-DERIVED-{what.upper()}",
            player=player,
            season=season,
            field=what,
            evidence={"source_url": purl, "locator": what},
            detail=detail,
            extra_id=extra,
        )
    )


def _derived_checks(
    out: ReduceResult,
    player: dict[str, Any],
    purl: str,
    core: ProfileFacts,
    aggs: dict[tuple[str, int], ClubSeasonAgg],
    keys: list[tuple[str, int]],
    per_game: dict[tuple[str, int], list[AG.Contribution]],
    summary_flags: dict[tuple[str, int], list[bool]],
    avg_rows: dict[tuple[int, str], SummaryRow],
    career_acc: list[AG.StatAcc],
    rules: Rules,
    contribs: dict[tuple[str, int], list[AG.Contribution]],
    *,
    career: bool = True,
) -> None:
    """The source's printed totals/averages against its own per-game cells (never a local verdict)."""
    printed_rows = {(r.year, r.club): r for r in core.season_totals if r.year is not None}
    foots = {(cs.season, cs.club): cs for cs in core.club_seasons}
    any_excluded = any(aggs[k].excluded for k in keys)
    for key in keys:
        a = aggs[key]
        club, season = key
        if a.excluded:
            # printed season figures span games after the event boundary: out of scope, never misused
            _derived(out, "season_out_of_scope", "skipped")
            continue
        prow = printed_rows.get((season, club))
        if prow is not None:
            gm = int(prow.gm_text) if _DIGITS.match(prow.gm_text) else None
            if gm is None or gm != a.games:
                _derived(out, "gm", "mismatch")
                _inconsistency(
                    out,
                    player,
                    purl,
                    "season_games",
                    f"printed GM {prow.gm_text!r} but {a.games} game rows",
                    season,
                    club,
                )
            else:
                _derived(out, "gm", "equal")
            if prow.wdl_text != f"{a.wins}-{a.draws}-{a.losses}":
                _derived(out, "wdl", "mismatch")
                _inconsistency(
                    out,
                    player,
                    purl,
                    "season_wdl",
                    f"printed {prow.wdl_text!r} rows {a.wins}-{a.draws}-{a.losses}",
                    season,
                    club,
                )
            else:
                _derived(out, "wdl", "equal")
        cs = foots.get((season, club))
        for i in range(_N):
            if i == _PCT_I:
                continue
            comp = AG.compose([per_game[key][i]])
            if prow is not None and not summary_flags[key][i]:
                res = AG.check_printed(prow.cells[i], comp)
                _derived(out, "season_total", res)
                if res == "mismatch":
                    _inconsistency(
                        out,
                        player,
                        purl,
                        "season_total",
                        f"{STAT_FIELDS[i]}: printed {prow.cells[i]!r} game cells sum {comp.value}",
                        season,
                        [club, i],
                    )
            if cs is not None and cs.foot is not None and not summary_flags[key][i]:
                res = AG.check_printed(cs.foot.cells[i], comp)
                _derived(out, "table_footer", res)
                if res == "mismatch":
                    _inconsistency(
                        out,
                        player,
                        purl,
                        "table_footer",
                        f"{STAT_FIELDS[i]}: footer {cs.foot.cells[i]!r} game cells sum {comp.value}",
                        season,
                        [club, i],
                    )
            arow = avg_rows.get((season, club))
            printed_avg = arow.cells[i] if arow is not None else None
            if printed_avg and a.games:
                contrib = AG.contribution(
                    a.src[i], _as_decimal(prow.cells[i]) if prow else None, summary_only=summary_flags[key][i]
                )
                if contrib.kind in ("per_game", "summary_only"):
                    den = AG.season_denominator(STAT_FIELDS[i], a.src[i], home_and_away=a.home_and_away)
                    ok = AG.average_matches(contrib.value, den, printed_avg)
                    _derived(out, "season_average", "ok" if ok else "miss")
                    if not ok:
                        _inconsistency(
                            out,
                            player,
                            purl,
                            "season_average",
                            f"{STAT_FIELDS[i]}: printed {printed_avg} model {contrib.value}/{den}",
                            season,
                            [club, i],
                        )
    # career footer
    games = sum(aggs[k].games for k in keys)
    seasons = len({k[1] for k in keys})
    wins = sum(aggs[k].wins for k in keys)
    draws = sum(aggs[k].draws for k in keys)
    losses = sum(aggs[k].losses for k in keys)
    tf, af = core.totals_foot, core.averages_foot
    if any_excluded or not career:
        _derived(out, "career_out_of_scope", "skipped")
        return
    if tf is not None:
        if tf.gm_text != str(games):
            _derived(out, "career_gm", "mismatch")
            _inconsistency(
                out, player, purl, "career_games", f"printed {tf.gm_text!r} but {games} game rows", None, None
            )
        else:
            _derived(out, "career_gm", "equal")
        if tf.wdl_text != f"{wins}-{draws}-{losses}":
            _derived(out, "career_wdl", "mismatch")
            _inconsistency(
                out, player, purl, "career_wdl", f"printed {tf.wdl_text!r} rows {wins}-{draws}-{losses}", None, None
            )
        else:
            _derived(out, "career_wdl", "equal")
        for i in range(_N):
            if i == _PCT_I:
                continue
            comp = AG.compose(
                [
                    c
                    for k in keys
                    for c in [
                        AG.contribution(
                            aggs[k].src[i],
                            _as_decimal(printed_rows[(k[1], k[0])].cells[i]) if (k[1], k[0]) in printed_rows else None,
                            summary_only=summary_flags[k][i],
                        )
                    ]
                ]
            )
            res = AG.check_printed(tf.cells[i], comp)
            _derived(out, "career_total", res)
            if res == "mismatch":
                _inconsistency(
                    out,
                    player,
                    purl,
                    "career_total",
                    f"{STAT_FIELDS[i]}: printed {tf.cells[i]!r} composed {comp.value}",
                    None,
                    i,
                )
    if af is not None and games:
        if af.gm_text:
            ok = AG.games_average_ok(games, seasons, af.gm_text)
            _derived(out, "career_gm_average", "ok" if ok else "miss")
            if not ok:
                _inconsistency(
                    out,
                    player,
                    purl,
                    "career_gm_average",
                    f"printed {af.gm_text} model {games}/{seasons} seasons",
                    None,
                    None,
                )
        if af.wdl_text:
            ok = AG.win_pct_ok(wins, draws, games, af.wdl_text)
            _derived(out, "career_win_pct", "ok" if ok else "miss")
            if not ok:
                _inconsistency(
                    out,
                    player,
                    purl,
                    "career_win_pct",
                    f"printed {af.wdl_text} model {wins}W {draws}D of {games}",
                    None,
                    None,
                )
        ha_award = sum(aggs[k].home_and_away for k in keys if k[1] not in rules.no_award_seasons)
        for i in range(_N):
            if i == _PCT_I:
                continue
            printed = af.cells[i]
            if not printed:
                continue
            if i == _BR_I and not rules.no_award_seasons:
                _derived(out, "career_average", "unresolved_no_award_evidence")
                continue
            composed = AG.compose([contribs[k][i] for k in keys])
            res2 = AG.check_career_average(
                STAT_FIELDS[i],
                printed=printed,
                acc=career_acc[i],
                games=games,
                home_and_away_in_award_seasons=ha_award,
                composed_total=composed.value if composed.kind == "composed" else None,
            )
            _derived(out, "career_average", res2.outcome)
            if res2.outcome == "miss":
                cands = ", ".join(f"{n}={'fits' if ok else 'no'}" for n, ok in res2.candidates)
                _inconsistency(
                    out,
                    player,
                    purl,
                    "career_average",
                    f"{STAT_FIELDS[i]}: printed {printed}; candidate denominators: {cands}",
                    None,
                    i,
                )
