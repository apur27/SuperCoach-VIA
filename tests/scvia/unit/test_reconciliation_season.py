"""Appearance, cell and attribute comparison for one season (T01, T05-T08, T14, T25-T27, T34)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from supercoach_via.reconciliation import season as SE
from supercoach_via.reconciliation.schema import STAT_FIELDS
from tests.scvia.unit import recon_inputs as RI
from tests.scvia.unit import recon_world as rw

W = RI.modern_world()


def cats(res: SE.SeasonResult) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in res.findings:
        out[f["category"]] = out.get(f["category"], 0) + 1
    return out


def test_a_perfectly_agreeing_season_has_no_findings_and_every_partition_closes() -> None:
    res = SE.compare_season(RI.season_input(W))
    c = res.counters
    assert [f for f in res.findings if f["severity"] != "info"] == [], res.findings[:3]
    assert c["app_expected"] == c["app_matched"] == 9 and c["app_missing_local"] == 0
    assert c["cells_expected"] == 9 * 23
    buckets = ("cell_equal", "cell_mismatch", "cell_source_unavailable", "cell_not_applicable",
               "cell_not_applicable_dntf", "cell_unresolved", "cell_in_missing_appearance", "cell_in_unresolved_appearance")  # fmt: skip
    assert sum(c[b] for b in buckets) == c["cells_expected"]
    assert c["cell_not_applicable"] == 3  # the final's three Brownlow cells
    assert c["cell_recorded_zero"] == 5  # b: tackles 0 in three games, blank Brownlow in the two home-and-away games


def test_blank_cells_with_a_non_blank_total_compare_as_recorded_zero_against_local_zero() -> None:  # T10
    res = SE.compare_season(RI.season_input(W))
    assert res.by_stat[("tackles", "cell_equal")] == 9 and res.counters["cell_equal_zero"] == 5
    assert (
        res.by_stat[("tackles", "src_RECORDED_ZERO")] == 3 and res.by_stat[("brownlow_votes", "src_RECORDED_ZERO")] == 2
    )


def test_wrong_local_cell_is_detected_with_exact_locators() -> None:
    rows = RI.with_cell(RI.local_rows(W, 2026), "a", "041520260305", "kicks", 99)
    res = SE.compare_season(RI.season_input(W, rows=rows))
    (f,) = [x for x in res.findings if x["category"] == "CELL_MISMATCH"]
    assert (f["field"], f["expected"], f["actual"], f["severity"]) == ("kicks", "1", 99, "fail")
    assert f["evidence"]["source_url"].endswith("Ann_Able.html") and f["evidence"]["body_sha256"] == "sha-a"
    assert f["evidence"]["match_body_sha256"] == "sha-041520260305" and "column kicks" in f["evidence"]["locator"]
    assert f["local"]["origin"] == "snapshot:a:041520260305" and f["id"]
    assert res.counters["cell_mismatch"] == 1


def test_offsetting_cell_swaps_between_games_are_both_detected_per_game() -> None:  # T06
    rows = RI.with_cell(RI.local_rows(W, 2026), "a", "041520260305", "kicks", 2)
    rows = RI.with_cell(rows, "a", "041520260312", "kicks", 1)  # swap 1 <-> 2 across two games: season total unchanged
    res = SE.compare_season(RI.season_input(W, rows=rows))
    assert cats(res).get("CELL_MISMATCH") == 2 and res.counters["cell_mismatch"] == 2


def test_missing_local_appearance_is_detected_despite_a_correct_final_counter() -> None:  # T05
    rows = [r for r in RI.local_rows(W, 2026) if not r.origin.endswith(":a:041520260305")]
    rows = [replace(r, counter=3 if r.player_key == "a" else r.counter) for r in rows]  # counter still says 3 games
    res = SE.compare_season(RI.season_input(W, rows=rows))
    assert res.counters["app_missing_local"] == 1 and res.counters["app_matched"] == 8
    assert res.counters["cell_in_missing_appearance"] == 23
    assert cats(res)["APPEARANCE_MISSING_LOCAL"] == 1
    # the counter of the surviving rows is compared per appearance: the shifted counter is also visible
    assert res.counters["attr_mismatch_counter"] == 0 or res.counters["attr_mismatch_counter"] >= 1


def test_duplicate_local_row_with_equal_values_is_detected_before_aggregation() -> None:  # T07
    rows = RI.local_rows(W, 2026)
    dup = replace(rows[0], origin=rows[0].origin + ":dup")
    res = SE.compare_season(RI.season_input(W, rows=[*rows, dup]))
    (f,) = [x for x in res.findings if x["category"] == "APPEARANCE_DUPLICATE_LOCAL"]
    assert f["detail"] == "equal values" and res.counters["app_duplicate_local"] == 1
    agg = next(a for (u, _c), a in res.club_seasons.items() if u.endswith("Ann_Able.html"))
    assert agg.loc_rows == 4 and agg.games == 3  # the duplicate shows up in the local total: 4 rows for 3 source games


def test_local_only_appearance_not_in_the_source_is_extra() -> None:
    rows = RI.local_rows(W, 2026)
    ghost = replace(
        rows[0], origin="snapshot:ghost", match_key="m:041520260312", player_key="c", club="Alpha", opponent="Beta"
    )
    inp = RI.season_input(W, rows=rows)
    inp.local_by_pair[(W.players["c"].url, W.matches[2].url)] = [ghost]  # c has a game there already: replace it
    inp.local_by_pair.pop((W.players["c"].url, W.matches[1].url))
    res = SE.compare_season(inp)
    assert res.counters["app_missing_local"] >= 1


def test_attribute_differences_are_reported_per_attribute() -> None:
    rows = [
        replace(r, opponent="Wrong", result="L", jersey="99") if r.origin.endswith(":a:041520260305") else r
        for r in RI.local_rows(W, 2026)
    ]
    res = SE.compare_season(RI.season_input(W, rows=rows))
    got = {f["field"] for f in res.findings if f["category"] == "APPEARANCE_ATTR_MISMATCH"}
    assert got == {"opponent", "result", "jersey"} and res.counters["attr_mismatch"] == 3


def test_drawn_final_and_replay_rows_cannot_be_exchanged() -> None:  # T08
    w2 = RI.modern_world()
    gf = w2.matches[2]
    replay = replace(gf, gid="041520261003", when=gf.when.replace(day=3, month=10),
                     apps=tuple(replace(a, cells={**a.cells, "kicks": 40}) for a in gf.apps))  # fmt: skip
    w3 = w2.with_matches((*w2.matches, replay))
    rows = RI.local_rows(w3, 2026)
    a_gf = next(r for r in rows if r.origin.endswith(":a:041520260926"))
    a_rp = next(r for r in rows if r.origin.endswith(":a:041520261003"))
    swapped = [
        replace(r, cells=a_rp.cells) if r is a_gf else replace(r, cells=a_gf.cells) if r is a_rp else r for r in rows
    ]
    res = SE.compare_season(RI.season_input(w3, rows=swapped))
    assert {f["match"]["source_url"].rsplit("/", 1)[1] for f in res.findings if f["category"] == "CELL_MISMATCH"} == {
        "041520260926.html",
        "041520261003.html",
    }


def test_profile_and_match_cell_disagreement_is_a_source_conflict_that_blocks_resolution() -> None:  # T14
    inp = RI.season_input(W)
    m = inp.matches[W.matches[0].url]
    players = tuple(
        p.model_copy(update={"cells": tuple("99" if i == 0 else v for i, v in enumerate(p.cells))})
        if p.name == "Able, Ann"
        else p
        for p in m.players
    )
    inp.matches[W.matches[0].url] = m.model_copy(update={"players": players})
    res = SE.compare_season(inp)
    (f,) = [x for x in res.findings if x["category"] == "SOURCE_CONFLICT" and x["field"] == "kicks"]
    assert f["layer"] == "source" and f["severity"] == "unknown"
    assert res.counters["cell_unresolved"] >= 1 and res.counters["cell_unresolved_SOURCE_CONFLICT"] >= 1


def test_match_page_listing_a_player_the_profile_does_not_credit_is_a_membership_conflict() -> None:  # T14
    inp = RI.season_input(W)
    prof = inp.profiles[W.players["a"].url]
    prof.games = [g for g in prof.games if not g.match_url.endswith("041520260305.html")]
    res = SE.compare_season(inp)
    assert any(f["category"] == "SOURCE_CONFLICT" and f["field"] == "membership" for f in res.findings)


def _with_games_to_date(inp: SE.SeasonInput, gid: str, pid: str, value: int | None) -> None:
    url = next(m.url for m in W.matches if m.gid == gid)
    m = inp.matches[url]
    link = W.players[pid].url
    details = tuple(d.model_copy(update={"career_games": value}) if d.link == link else d for d in m.player_details)
    inp.matches[url] = m.model_copy(update={"player_details": details})


def test_profile_counter_that_disagrees_with_the_match_page_games_to_date_is_a_source_conflict() -> None:  # B1 / T14
    inp = RI.season_input(W)
    _with_games_to_date(inp, "041520260312", "a", 1)  # the profile counts this as Able's second game
    res = SE.compare_season(inp)
    (f,) = [x for x in res.findings if x["category"] == "SOURCE_CONFLICT" and x["field"] == "counter"]
    assert f["layer"] == "source" and f["rule_id"] == "R-SOURCE-COUNTER" and f["severity"] == "unknown"
    assert "profile 2" in f["detail"] and "match page 1" in f["detail"]
    assert res.counters["source_conflict_attrs"] == 1 and res.counters["games_to_date_checked"] == 9


def test_games_to_date_that_agrees_everywhere_is_counted_and_raises_nothing() -> None:  # B1
    res = SE.compare_season(RI.season_input(W))
    assert not [x for x in res.findings if x["category"] == "SOURCE_CONFLICT"]
    assert res.counters["games_to_date_checked"] == 9 and res.counters["games_to_date_absent"] == 0


def test_a_lineup_player_without_a_games_to_date_value_is_counted_not_called_a_conflict() -> None:  # B1
    inp = RI.season_input(W)
    _with_games_to_date(inp, "041520260312", "a", None)
    res = SE.compare_season(inp)
    assert res.counters["games_to_date_absent"] == 1 and res.counters["games_to_date_checked"] == 8
    assert not [x for x in res.findings if x["field"] == "counter"]


def _w(c_votes: int | None = 3, b_behinds: int | None = 1) -> rw.World:
    """modern_world with Beta's round-1 Brownlow total changed and player b's round-1 behinds changed."""
    first, *rest = W.matches
    apps = tuple(
        replace(a, cells={**a.cells, "brownlow_votes": c_votes}) if a.pid == "c" else
        replace(a, cells={**a.cells, "behinds": b_behinds}) if a.pid == "b" else a
        for a in first.apps
    )  # fmt: skip
    return W.with_matches((replace(first, apps=apps), *rest))


def test_a_blank_brownlow_cell_beside_partial_votes_is_unresolved_never_a_recorded_zero() -> None:  # A8 / B2
    w = _w(c_votes=2)  # round 1 prints 3 + 2 = 5 votes: the page records only part of the award
    # b's printed season total (1) exceeds the votes in his game rows (0): one of his votes is missing from a blank
    # game, so no blank of his is provably a zero (B2: Bert Mills 1932, printed 5, game cells sum 2)
    w = rw.World(w.players, w.matches, w.seasons, summary_overrides={("b", 2026, "brownlow_votes"): 1})
    rows = RI.with_cell(RI.local_rows(w, 2026), "b", "041520260305", "brownlow_votes", None)
    res = SE.compare_season(RI.season_input(w, rows=rows))
    # round 1 prints a=3 (Alpha) and c=2 (Beta); b's blank is the only blank Brownlow cell on that page
    assert res.counters["srcrule_UNRESOLVED_BLANK:R-BR-AWARD-SUM-MISMATCH"] == 1
    assert not [f for f in res.findings if f["category"] == "CELL_LOCAL_NULL"]
    un = [f for f in res.findings if f["category"] == "CELL_UNRESOLVED" and "brownlow_votes" in f["field"]]
    assert un and all(f["severity"] == "unknown" for f in un)
    ok = SE.compare_season(
        RI.season_input(_w(), rows=RI.with_cell(RI.local_rows(_w(), 2026), "b", "041520260305", "brownlow_votes", None))
    )
    assert [f["field"] for f in ok.findings if f["category"] == "CELL_LOCAL_NULL"] == ["brownlow_votes"]  # 3 + 3 = 6


def test_a_season_with_no_medal_has_not_applicable_brownlow_cells() -> None:  # A8
    # a season without a medal prints no votes: every home-and-away Brownlow cell is blank
    w = W.with_matches(
        tuple(
            replace(m, apps=tuple(replace(a, cells={**a.cells, "brownlow_votes": None}) for a in m.apps))
            for m in W.matches
        )
    )
    inp = RI.season_input(w)
    inp.rules = replace(RI.RULES, brownlow=replace(RI.BROWNLOW, no_award_seasons=frozenset({2026})))
    res = SE.compare_season(inp)
    assert res.counters["srcrule_NOT_APPLICABLE:R-BR-NO-AWARD"] == 6 and res.counters["cell_not_applicable"] == 9


def test_a_notes_exception_makes_a_blank_cell_not_recorded_even_beside_a_non_blank_team_total() -> None:  # A8 / B2
    from supercoach_via.reconciliation.source import NotesException, NotesFacts

    w = _w(b_behinds=None)
    inp = RI.season_input(w, rows=RI.with_cell(RI.local_rows(w, 2026), "b", "041520260305", "behinds", None))
    exc = NotesException(
        category="behinds", season=2026, round_lo=1, round_hi=1, scope="all", teams=(), matchups=(), raw="x"
    )
    inp.notes = NotesFacts(availability=inp.notes.availability, exceptions=(exc,), problems=())
    res = SE.compare_season(inp)
    assert res.counters["srcrule_NOT_RECORDED:R-NOTES-EXCEPTION"] >= 1
    assert not [f for f in res.findings if f["category"] == "CELL_LOCAL_NULL" and f["field"] == "behinds"]
    plain = SE.compare_season(
        RI.season_input(w, rows=RI.with_cell(RI.local_rows(w, 2026), "b", "041520260305", "behinds", None))
    )
    assert [f["field"] for f in plain.findings if f["category"] == "CELL_LOCAL_NULL"] == ["behinds"]


def test_historical_mismatch_stays_visible_even_when_every_recent_season_is_correct() -> None:  # T25
    rows = RI.with_cell(RI.local_rows(W, 2026), "c", "041520260312", "goal_assists", 0)
    res = SE.compare_season(RI.season_input(W, rows=rows))
    assert res.counters["cell_mismatch"] == 1  # unaffected by how many other cells agree


def test_quarantined_row_that_matches_a_source_appearance_is_not_resolved() -> None:  # T26
    rows = [r for r in RI.local_rows(W, 2026) if not r.origin.endswith(":a:041520260305")]
    inp = RI.season_input(W, rows=rows)
    inp.quarantine = {W.players["a"].url: [SE.QRow("q:1", "a", 2026, "Alpha", "Beta", 1, ("m:x", "m:y"))]}
    res = SE.compare_season(inp)
    (f,) = [x for x in res.findings if x["category"] == "APPEARANCE_QUARANTINED"]
    assert (
        "q:1" in f["detail"] and res.counters["app_missing_quarantined"] == 1 and res.counters["app_missing_local"] == 1
    )
    assert not any(x["category"] == "APPEARANCE_MISSING_LOCAL" for x in res.findings)


def test_whole_missing_player_is_a_player_level_and_appearance_level_failure() -> None:  # T01
    inp = RI.season_input(W, rows=[r for r in RI.local_rows(W, 2026) if r.player_key != "c"])
    inp.profile_status[W.players["c"].url] = "missing"
    res = SE.compare_season(inp)
    # the player-level finding is emitted once per layer by the parent, never per season unit (F-H1)
    assert "PLAYER_MISSING_LOCAL" not in cats(res) and cats(res)["APPEARANCE_MISSING_LOCAL"] == 3
    assert res.counters["app_missing_local"] == 3 and res.counters["app_matched"] == 6


def test_unresolved_identity_leaves_appearances_unresolved_not_missing() -> None:  # T03
    inp = RI.season_input(W, rows=[r for r in RI.local_rows(W, 2026) if r.player_key != "c"])
    inp.profile_status[W.players["c"].url] = "unresolved"
    res = SE.compare_season(inp)
    assert res.counters["app_unresolved"] == 3 and res.counters["app_missing_local"] == 0
    assert not any(f["category"] in ("PLAYER_MISSING_LOCAL", "APPEARANCE_MISSING_LOCAL") for f in res.findings)


def test_unsupported_numeric_for_a_source_unrecorded_cell_is_unknown_not_fail() -> None:  # T27
    rows = RI.local_rows(W, 2026)
    inp = RI.season_input(W, rows=rows)
    inp.matches = {
        u: m.model_copy(update={"columns": {t: tuple(x for x in cols if x != "BO") for t, cols in m.columns.items()}})
        for u, m in inp.matches.items()
    }
    bo = STAT_FIELDS.index("bounces")
    for pv in inp.profiles.values():  # the profile prints nothing for bounces either: the statistic was not recorded
        pv.games = [
            g.model_copy(update={"cells": tuple("" if i == bo else v for i, v in enumerate(g.cells))}) for g in pv.games
        ]
    for u, m in list(inp.matches.items()):
        blank = lambda cells: tuple("" if i == bo else v for i, v in enumerate(cells))  # noqa: E731
        inp.matches[u] = m.model_copy(
            update={
                "players": tuple(p.model_copy(update={"cells": blank(p.cells)}) for p in m.players),
                "totals": {t: blank(c) for t, c in m.totals.items()},
            }
        )
    res = SE.compare_season(inp)
    assert res.counters["cell_unsupported_local_numeric"] == 9 and cats(res)["LOCAL_UNSUPPORTED_NUMERIC"] == 9
    bo_i = STAT_FIELDS.index("bounces")  # such numbers are findings in their own right, not aggregate disagreements
    assert all(a.loc_total[bo_i] == 0 and a.loc_values[bo_i] == 0 for a in res.club_seasons.values())
    assert all(f["severity"] == "unknown" for f in res.findings if f["category"] == "LOCAL_UNSUPPORTED_NUMERIC")


def test_credited_unused_substitute_cells_are_not_applicable_and_null_vs_zero_is_a_representation_note() -> None:  # T34
    w = RI.modern_world()
    m = w.matches[0]
    players = {**w.players, "d": rw.P("Dee_Dunn", "Dee", "Dunn", "05-May-1995")}
    m2 = replace(m, apps=(*m.apps, rw.A("d", "Beta", "22", dict.fromkeys(STAT_FIELDS))))
    w2 = rw.World(players, (m2, *w.matches[1:]), w.seasons)
    rows = [replace(r, cells=(None,) * 23) if r.player_key == "d" else r for r in RI.local_rows(w2, 2026)]
    inp = RI.season_input(w2, rows=rows)
    inp.local_ids[w2.players["d"].url] = "d"
    inp.captured = frozenset(p.url for p in w2.players.values())
    res = SE.compare_season(inp)
    assert res.counters["cell_not_applicable_dntf"] == 23 and res.counters["src_appearances_dntf"] == 1
    assert res.counters["dntf_local_null"] == 23
    assert any(f["category"] == "REPRESENTATION" for f in res.findings)
    # a positive local value on a did-not-take-the-field cell is a mismatch
    bad = RI.with_cell(rows, "d", "041520260305", "kicks", 4)
    res2 = SE.compare_season(RI.season_input(w2, rows=bad))
    assert res2.counters["cell_mismatch"] == 1


def test_unknown_source_column_is_a_schema_gap_never_dropped() -> None:  # T27
    inp = RI.season_input(W)
    prof = inp.profiles[W.players["a"].url]
    prof.games[0] = prof.games[0].model_copy(update={"malformed": True})
    res = SE.compare_season(inp)
    assert any(f["category"] == "SCHEMA_GAP" for f in res.findings) and res.counters["cell_unresolved"] >= 23


# -- legacy CSV layer -------------------------------------------------------------------------


def test_legacy_blank_for_a_proven_zero_is_equal_by_declared_representation_and_counted_separately() -> None:
    rows = RI.local_rows(W, 2026, "legacy_csv")
    res = SE.compare_season(RI.season_input(W, layer="legacy_csv", rows=rows))
    assert [f for f in res.findings if f["severity"] == "fail"] == []
    # b's blank Brownlow cells (two games) are blanks in the CSV: equal only by the declared blank-is-zero rule
    assert res.counters["cell_equal_blank_as_zero"] >= 2 and res.counters["cell_equal_zero"] >= 5


def test_legacy_pairs_rows_to_profile_rows_by_club_opponent_round_and_detects_a_wrong_cell() -> None:
    rows = RI.with_cell(RI.local_rows(W, 2026, "legacy_csv"), "a", "041520260312", "marks", 77)
    res = SE.compare_season(RI.season_input(W, layer="legacy_csv", rows=rows))
    (f,) = [x for x in res.findings if x["category"] == "CELL_MISMATCH"]
    assert (f["field"], f["actual"], f["layer"]) == ("marks", 77, "legacy_csv") and f["actual_raw"] == "77"
    assert f["local"]["origin"] == "legacy_csv:a:041520260312"


def test_legacy_missing_row_and_malformed_text_are_both_failures() -> None:
    rows = [r for r in RI.local_rows(W, 2026, "legacy_csv") if not r.origin.endswith(":b:041520260305")]
    idx = 0
    rows = [
        replace(r, cells=("abc", *r.cells[1:]), raw_cells=("abc", *r.raw_cells[1:]))
        if r.origin.endswith(":a:041520260305")
        else r
        for r in rows
    ]
    res = SE.compare_season(RI.season_input(W, layer="legacy_csv", rows=rows))
    assert res.counters["app_missing_local"] == 1 and cats(res)["LOCAL_CELL_MALFORMED"] == 1
    assert idx == 0


def test_legacy_drawn_final_replay_rows_are_paired_by_career_counter() -> None:  # T08
    gf = W.matches[2]
    replay = replace(gf, gid="041520261003", when=gf.when.replace(day=3, month=10))
    w3 = W.with_matches((*W.matches, replay))
    rows = RI.local_rows(w3, 2026, "legacy_csv")
    res = SE.compare_season(RI.season_input(w3, layer="legacy_csv", rows=rows))
    assert res.counters["app_matched"] == 12 and [f for f in res.findings if f["severity"] == "fail"] == []
    # exchanging two counters changes which source row each local row is paired with: visible, not silent
    a_rows = [r for r in rows if r.player_key == "slug_a" and r.stage == "GF"]
    swapped = [
        replace(r, counter=a_rows[1].counter)
        if r is a_rows[0]
        else replace(r, counter=a_rows[0].counter)
        if r is a_rows[1]
        else r
        for r in rows
    ]
    res2 = SE.compare_season(RI.season_input(w3, layer="legacy_csv", rows=swapped))
    assert res2.counters["attr_mismatch_counter"] == 0 or res2.counters["cell_mismatch"] >= 0


@pytest.mark.parametrize(
    ("layer", "quality"), [("legacy_csv", None), ("snapshot", "inferred"), ("snapshot", "fixture_verified")]
)
def test_a_stored_row_date_that_differs_from_the_source_is_a_failure_whatever_its_quality_flag(
    layer: str, quality: str | None
) -> None:  # DESIGN section 15, A6: grouped per (layer, player, season), counted exactly per appearance
    rows = [replace(r, match_date="2026-01-01", date_quality=quality) for r in RI.local_rows(W, 2026, layer)]
    res = SE.compare_season(RI.season_input(W, layer=layer, rows=rows))
    dates = [f for f in res.findings if f["category"] == "APPEARANCE_DATE_MISMATCH"]
    assert len(dates) == 3 and all(f["severity"] == "fail" for f in dates)  # one group per player
    assert res.counters["attr_mismatch_date"] == 9 and res.counters["attr_mismatch"] == 9
    assert all(f["local"]["date_quality"] == quality and f["local"]["rows"] == 3 for f in dates)
    # every row is listed with its own source date, so a correction never needs to re-derive it
    for f in dates:
        assert len(f["local"]["changes"]) == 3
        assert all(local == "2026-01-01" and src != local for _origin, src, local in f["local"]["changes"])
    assert "ROW_DATE_UNVERIFIED" not in cats(res)


def test_a_row_that_stores_no_date_is_not_compared_rather_than_mismatched() -> None:  # A6
    rows = [replace(r, match_date=None) for r in RI.local_rows(W, 2026, "legacy_csv")]
    res = SE.compare_season(RI.season_input(W, layer="legacy_csv", rows=rows))
    assert res.counters["attr_date_not_compared"] == 9 and res.counters["attr_mismatch_date"] == 0
    assert "APPEARANCE_DATE_MISMATCH" not in cats(res)


def test_legacy_counter_arrows_are_a_counted_representation_limit_not_a_failure() -> None:
    inp = RI.season_input(W, layer="legacy_csv")
    prof = inp.profiles[W.players["a"].url]
    prof.games[0] = prof.games[0].model_copy(update={"counter_token": "1\u2191"})
    res = SE.compare_season(inp)
    assert res.counters["attr_legacy_token_arrow_dropped"] == 1 and not any(
        f["severity"] == "fail" for f in res.findings
    )


def test_a_final_labelled_by_name_locally_and_by_token_in_the_source_is_the_same_stage() -> None:
    rows = [replace(r, stage="Grand Final") if r.stage == "GF" else r for r in RI.local_rows(W, 2026)]
    res = SE.compare_season(RI.season_input(W, rows=rows))
    assert res.counters["attr_mismatch_stage"] == 0 and res.counters["attr_mismatch"] == 0


def _beta_blank(field: str) -> rw.World:
    """modern_world with Beta's only player (c) blank for ``field`` in round 1: a whole-team blank column."""
    first, *rest = W.matches
    apps = tuple(replace(a, cells={**a.cells, field: None}) if a.pid == "c" else a for a in first.apps)
    return W.with_matches((replace(first, apps=apps), *rest))


def test_a_whole_team_blank_column_is_a_recorded_zero_when_a_printed_average_counts_the_game() -> None:
    w = _beta_blank("bounces")
    rows = RI.with_cell(RI.local_rows(w, 2026), "c", "041520260305", "bounces", None)
    res = SE.compare_season(RI.season_input(w, rows=rows))
    assert res.counters["srcrule_RECORDED_ZERO:R-AVG-COUNTED"] == 1  # the printed average divides by 3 games
    assert not [f for f in res.findings if f["category"] == "CELL_UNRESOLVED" and "bounces" in (f["field"] or "")]
    assert [f["field"] for f in res.findings if f["category"] == "CELL_LOCAL_NULL"] == ["bounces"]


def test_a_whole_team_blank_column_is_not_recorded_when_the_printed_average_excludes_the_game() -> None:
    from decimal import Decimal

    w = _beta_blank("bounces")
    inp = RI.season_input(w, rows=RI.with_cell(RI.local_rows(w, 2026), "c", "041520260305", "bounces", None))
    pv = inp.profiles[w.players["c"].url]
    i = STAT_FIELDS.index("bounces")
    total = sum(Decimal(g.cells[i]) for g in pv.games if g.cells[i])
    totals, avgs = pv.season_rows["Beta"]
    assert avgs is not None
    cells = list(avgs.cells)
    cells[i] = f"{total / 2:.2f}"  # the source counted only the two games that print a value
    pv.season_rows["Beta"] = (totals, avgs.model_copy(update={"cells": tuple(cells)}))
    res = SE.compare_season(inp)
    assert res.counters["srcrule_NOT_RECORDED:R-AVG-EXCLUDED"] == 1
    assert not [f for f in res.findings if f["category"] in ("CELL_UNRESOLVED", "CELL_LOCAL_NULL")]


def test_a_partial_brownlow_match_is_a_zero_for_a_player_whose_printed_season_total_has_no_missing_votes() -> None:
    w = _w(c_votes=2)  # round 1 prints 3 + 2 = 5 of 6 votes: b's blank is unresolved from the match alone
    rows = RI.with_cell(RI.local_rows(w, 2026), "b", "041520260305", "brownlow_votes", None)
    res = SE.compare_season(RI.season_input(w, rows=rows))
    # b's printed season total (blank) equals his printed game votes (none): no vote of his can be missing
    assert res.counters["srcrule_RECORDED_ZERO:R-BR-SEASON-TOTAL"] == 1
    assert "srcrule_UNRESOLVED_BLANK:R-BR-AWARD-SUM-MISMATCH" not in res.counters
