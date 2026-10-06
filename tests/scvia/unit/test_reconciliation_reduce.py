"""Stint / season / career aggregates and printed-figure consistency (T09, T32, T33; DESIGN section 8)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from supercoach_via.reconciliation import reduce as RD
from supercoach_via.reconciliation import season as SE
from supercoach_via.reconciliation import source as R
from supercoach_via.reconciliation.schema import STAT_FIELDS
from tests.scvia.unit import recon_inputs as RI
from tests.scvia.unit import recon_world as rw

W = RI.modern_world()
IDX = {f: i for i, f in enumerate(STAT_FIELDS)}


def reduce_world(
    w: rw.World,
    rows=None,
    pid: str = "a",
    season: int = 2026,
    rules=None,
    emit_source: bool = True,
    layer: str = "snapshot",
):  # type: ignore[no-untyped-def]
    inp = RI.season_input(w, season, rows=rows, layer=layer)
    if rules is not None:
        inp.rules = rules
    res = SE.compare_season(inp)
    purl = w.players[pid].url
    core = R.read_profile(rw.profile_page(w, pid), purl)
    aggs = {(club, s): a for (u, club), a in res.club_seasons.items() if u == purl for s in [a.season]}
    return RD.reduce_profile(
        layer=layer, purl=purl, local_id=pid, core=core, aggs=aggs, rules=inp.rules, emit_source=emit_source,
        profile_sha256=f"sha-{pid}",
    )  # fmt: skip


def test_a_consistent_player_has_equal_aggregates_at_every_level_and_consistent_printed_figures() -> None:
    r = reduce_world(W)
    assert [f for f in r.findings if f["severity"] != "info"] == []
    for level in ("stint", "season", "career"):
        assert r.counters[f"agg_{level}_equal"] == 23  # every statistic is recorded for this player in this world
    assert r.counters["agg_career_mismatch"] == 0 and r.counters["derived_gm_equal"] == 1
    assert r.counters["derived_wdl_equal"] == 1 and r.counters["derived_career_gm_equal"] == 1
    assert r.counters["derived_season_total_mismatch"] == 0 and r.counters["derived_career_total_mismatch"] == 0


def test_a_missing_local_game_changes_the_full_scope_totals_at_stint_and_career_level() -> None:  # T05
    rows = [r for r in RI.local_rows(W, 2026) if not r.origin.endswith(":a:041520260305")]
    r = reduce_world(W, rows)
    assert r.counters["agg_stint_mismatch"] > 0 and r.counters["agg_career_mismatch"] > 0
    f = next(
        x
        for x in r.findings
        if x["category"] == "AGGREGATE_MISMATCH" and x["rule_id"] == "R-AGG-CAREER" and x["field"] == "kicks"
    )
    assert (
        f["expected"] == "6" and f["actual"] == "5"
    )  # kicks 1+2+3 in the source; the local layer lacks the 1-kick game


def test_two_clubs_in_one_season_are_summed_once_at_season_and_career_level() -> None:  # T09
    w = RI.modern_world()
    m2 = w.matches[1]
    moved = tuple(
        replace(a, club="Beta") if a.pid == "a" else a for a in m2.apps
    )  # a changes club mid-season: Alpha, Beta
    moved_m2 = replace(m2, apps=moved)
    w2 = w.with_matches((w.matches[0], moved_m2, w.matches[2]))
    r = reduce_world(w2)
    assert r.counters["agg_stint_equal"] > 0
    # two stints in 2026 (Alpha: 2 games, Beta: 1), one season row, one career: no mismatch from double counting
    assert r.counters["agg_career_mismatch"] == 0 and r.counters["agg_season_mismatch"] == 0
    # the printed Totals foot (games) counts three games once
    assert r.counters["derived_career_gm_equal"] == 1 and r.counters["derived_gm_equal"] == 2


def test_pre_1984_season_summary_only_brownlow_is_a_reference_and_a_missing_local_value_fails() -> None:  # T33 / S-01
    ms = []
    for k, gid in enumerate(("041519350401", "041519350408", "041519350415")):
        apps = (
            rw.A("a", "Alpha", "1", {"goals": 2}),
            rw.A("b", "Alpha", "2", {"goals": 1}),
            rw.A("c", "Beta", "3", {"goals": 3}),
        )
        ms.append(rw.M(gid, 1935, str(k + 1), "Alpha", "Beta", date(1935, 4, 1 + 7 * k), apps=apps))
    w = RI.modern_world()
    w = rw.World(w.players, tuple(ms), (1935,), summary_overrides={("a", 1935, "brownlow_votes"): 13})
    rows = RI.local_rows(w, 1935)  # the local layer holds goals only: no Brownlow value anywhere
    r = reduce_world(w, rows, season=1935)
    assert r.counters["summary_only_stats"] == 1
    f = [x for x in r.findings if x["category"] == "LOCAL_MISSING_SUMMARY_VALUE"]
    assert {x["rule_id"] for x in f} == {"R-AGG-STINT", "R-AGG-CAREER"} and all(x["severity"] == "fail" for x in f)
    assert "13" in f[0]["detail"] and r.counters["agg_career_local_missing_summary"] == 1
    assert not any(x["category"] == "AGGREGATE_MISMATCH" for x in r.findings)  # counted once, not twice
    # the printed career average (13 votes / 3 home-and-away games) is reproduced from the composed season value,
    # not from the blank per-game cells
    assert not [x for x in r.findings if x["rule_id"] == "R-DERIVED-CAREER_AVERAGE" and "brownlow" in x["detail"]]
    # a local layer that does hold the season value passes
    rows2 = RI.with_cell(rows, "a", "041519350401", "brownlow_votes", 13)
    r2 = reduce_world(w, rows2, season=1935)
    assert r2.counters["agg_career_local_missing_summary"] == 0 and r2.counters["agg_career_mismatch"] == 0


def test_a_statistic_never_recorded_is_source_unavailable_not_equal_and_not_unresolved() -> None:  # T33 / N-02
    ms = (
        rw.M(
            "041519350401",
            1935,
            "1",
            "Alpha",
            "Beta",
            date(1935, 4, 1),
            apps=(rw.A("a", "Alpha", "1", {"goals": 2}), rw.A("c", "Beta", "3", {"goals": 3})),
        ),
    )
    w = rw.World({k: v for k, v in RI.modern_world().players.items() if k in ("a", "c")}, ms, (1935,))
    r = reduce_world(w, RI.local_rows(w, 1935), season=1935)
    assert r.by_stat[("career", "kicks", "source_unavailable")] == 1 and r.by_stat[("career", "goals", "equal")] == 1
    assert r.counters["agg_career_unresolved"] == 0


def test_printed_total_that_disagrees_with_double_sourced_cells_is_a_derived_inconsistency_only() -> None:  # S-07
    purl = W.players["a"].url
    core = R.read_profile(rw.profile_page(W, "a"), purl)
    bad_rows = tuple(
        r.model_copy(update={"cells": tuple("999" if i == IDX["kicks"] else c for i, c in enumerate(r.cells))})
        for r in core.season_totals
    )
    core2 = core.model_copy(update={"season_totals": bad_rows})
    inp = RI.season_input(W)
    res = SE.compare_season(inp)
    aggs = {(club, a.season): a for (u, club), a in res.club_seasons.items() if u == purl}
    r = RD.reduce_profile(
        layer="snapshot", purl=purl, local_id="a", core=core2, aggs=aggs, rules=inp.rules, emit_source=True
    )
    f = [
        x
        for x in r.findings
        if x["category"] == "SOURCE_DERIVED_INCONSISTENCY" and x["rule_id"] == "R-DERIVED-SEASON_TOTAL"
    ]
    assert len(f) == 1 and f[0]["severity"] == "info" and f[0]["layer"] == "source"
    assert [x for x in r.findings if x["severity"] == "fail"] == []  # the local layer's verdict is untouched
    assert r.counters["agg_career_equal"] >= 20


def test_a_printed_total_with_a_single_sourced_cell_is_unresolved_not_an_inconsistency() -> None:  # S-07
    purl = W.players["a"].url
    core = R.read_profile(rw.profile_page(W, "a"), purl)
    inp = RI.season_input(W)
    inp.matches = {
        u: m.model_copy(update={"players": tuple(p for p in m.players if p.name != "Able, Ann")})
        for u, m in inp.matches.items()
    }
    res = SE.compare_season(inp)
    aggs = {(club, a.season): a for (u, club), a in res.club_seasons.items() if u == purl}
    bad = tuple(
        r.model_copy(update={"cells": tuple("999" if i == IDX["kicks"] else c for i, c in enumerate(r.cells))})
        for r in core.season_totals
    )
    r = RD.reduce_profile(
        layer="snapshot",
        purl=purl,
        local_id="a",
        core=core.model_copy(update={"season_totals": bad}),
        aggs=aggs,
        rules=inp.rules,
        emit_source=True,
    )
    assert r.counters["derived_season_total_unresolved"] >= 1 and r.counters["derived_season_total_mismatch"] == 0


def test_career_average_model_miss_lists_candidates_and_is_not_a_local_verdict() -> None:  # T32 / S-03
    purl = W.players["a"].url
    core = R.read_profile(rw.profile_page(W, "a"), purl)
    af = core.averages_foot
    assert af is not None
    cells = tuple("9.99" if i == IDX["kicks"] else c for i, c in enumerate(af.cells))
    core2 = core.model_copy(update={"averages_foot": af.model_copy(update={"cells": cells})})
    inp = RI.season_input(W)
    res = SE.compare_season(inp)
    aggs = {(club, a.season): a for (u, club), a in res.club_seasons.items() if u == purl}
    r = RD.reduce_profile(
        layer="snapshot", purl=purl, local_id="a", core=core2, aggs=aggs, rules=inp.rules, emit_source=True
    )
    miss = [x for x in r.findings if x["rule_id"] == "R-DERIVED-CAREER_AVERAGE"]
    assert len(miss) == 1 and "candidate denominators" in miss[0]["detail"] and "model=no" in miss[0]["detail"]
    assert [x for x in r.findings if x["severity"] == "fail"] == []


def test_source_figures_are_emitted_by_exactly_one_layer() -> None:
    both = reduce_world(W, emit_source=True)
    only_local = reduce_world(W, emit_source=False)
    assert any(k.startswith("derived_") for k in both.counters) and not any(
        k.startswith("derived_") for k in only_local.counters
    )


def test_a_profile_no_local_player_maps_to_is_not_judged_against_empty_local_totals() -> None:
    purl = W.players["a"].url
    core = R.read_profile(rw.profile_page(W, "a"), purl)
    res = SE.compare_season(RI.season_input(W, rows=[]))
    aggs = {(club, a.season): a for (u, club), a in res.club_seasons.items() if u == purl}
    r = RD.reduce_profile(
        layer="snapshot",
        purl=purl,
        local_id=None,
        core=core,
        aggs=aggs,
        rules=RI.RULES,
        emit_source=True,
        judge_local=False,
    )
    assert r.counters["profiles_not_judged_no_local_player"] == 1 and r.counters["agg_career_judged"] == 0
    assert not any(f["category"] == "AGGREGATE_MISMATCH" for f in r.findings) and r.counters["derived_gm_equal"] == 1


def test_a_profile_whose_counters_skip_a_game_is_one_source_conflict_from_the_emitting_layer_only() -> None:  # T14
    purl = W.players["a"].url
    core = R.read_profile(
        rw.profile_page(W, "a", counters={"041520260305": "1", "041520260312": "3", "041520260926": "4"}), purl
    )
    assert core.counter_issues and core.counter_issues[0].startswith("gap:2")
    res = SE.compare_season(RI.season_input(W))
    aggs = {(club, a.season): a for (u, club), a in res.club_seasons.items() if u == purl}

    def run(emit: bool) -> RD.ReduceResult:
        return RD.reduce_profile(
            layer="snapshot", purl=purl, local_id="a", core=core, aggs=aggs, rules=RI.RULES, emit_source=emit
        )

    (f,) = [x for x in run(True).findings if x["field"] == "counter_sequence"]
    assert f["category"] == "SOURCE_CONFLICT" and f["layer"] == "source" and f["rule_id"] == "R-SOURCE-COUNTER-SEQUENCE"
    assert "gap:2" in f["detail"] and f["severity"] == "unknown"
    assert run(True).counters["source_counter_sequence_profiles"] == 1
    assert [x for x in run(False).findings if x["field"] == "counter_sequence"] == []


def test_an_aggregate_finding_names_the_profile_body_and_the_local_rows_that_were_summed() -> None:  # F-M4
    rows = [r for r in RI.local_rows(W, 2026) if not r.origin.endswith(":a:041520260305")]
    r = reduce_world(W, rows)
    f = next(x for x in r.findings if x["category"] == "AGGREGATE_MISMATCH" and x["rule_id"] == "R-AGG-CAREER")
    assert f["evidence"]["body_sha256"] == "sha-a" and f["evidence"]["locator"].startswith("career")
    origins = f["local"]["origins"]
    assert len(origins) == 2 and all(o.endswith((":a:041520260312", ":a:041520260926")) for o in origins)


def test_a_season_award_the_local_layer_stores_reproduces_a_summary_only_brownlow_value() -> None:  # T33
    from decimal import Decimal

    ms = []
    for k, gid in enumerate(("041519350401", "041519350408", "041519350415")):
        apps = (
            rw.A("a", "Alpha", "1", {"goals": 2}),
            rw.A("b", "Alpha", "2", {"goals": 1}),
            rw.A("c", "Beta", "3", {"goals": 3}),
        )
        ms.append(rw.M(gid, 1935, str(k + 1), "Alpha", "Beta", date(1935, 4, 1 + 7 * k), apps=apps))
    w = RI.modern_world()
    w = rw.World(w.players, tuple(ms), (1935,), summary_overrides={("a", 1935, "brownlow_votes"): 13})
    inp = RI.season_input(w, 1935, rows=RI.local_rows(w, 1935))
    inp.local_summary = {(w.players["a"].url, "Alpha"): {"brownlow_votes": Decimal(13)}}
    res = SE.compare_season(inp)
    purl = w.players["a"].url
    core = R.read_profile(rw.profile_page(w, "a"), purl)
    aggs = {(club, a.season): a for (u, club), a in res.club_seasons.items() if u == purl}
    r = RD.reduce_profile(layer="snapshot", purl=purl, local_id="a", core=core, aggs=aggs, rules=inp.rules,
                          emit_source=True)  # fmt: skip
    assert not [f for f in r.findings if f["category"] in ("LOCAL_MISSING_SUMMARY_VALUE", "AGGREGATE_MISMATCH")]
    assert r.counters["agg_career_equal"] >= 1
    # a stored value that disagrees with the printed season value is a mismatch, never silently accepted
    inp.local_summary = {(purl, "Alpha"): {"brownlow_votes": Decimal(12)}}
    res = SE.compare_season(inp)
    aggs = {(club, a.season): a for (u, club), a in res.club_seasons.items() if u == purl}
    r = RD.reduce_profile(layer="snapshot", purl=purl, local_id="a", core=core, aggs=aggs, rules=inp.rules,
                          emit_source=True)  # fmt: skip
    assert [f["rule_id"] for f in r.findings if f["category"] == "AGGREGATE_MISMATCH"] == [
        "R-AGG-STINT",
        "R-AGG-CAREER",
    ]
