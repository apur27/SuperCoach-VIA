"""Tests for scripts/phantom_row_validator.py (deliverable 2).

Two layers:
  (a) deterministic games_played counter-gap check -- catches Sidebottom-type
      in-place row deletions with zero false positives on legitimately missed
      games (the counter only increments for games actually played).
  (b) drawn-final cross-check -- surfaces the counter-renumber blind spot that
      layer (a) cannot see, scoped to years whose matches file contains a drawn
      final + replay.
"""
import pandas as pd
import pytest

from scripts.phantom_row_validator import (
    check_counter_gaps,
    find_drawn_finals,
    check_drawn_final_consistency,
    validate_player_file,
    gaps_in_season,
)

PERF_COLS = ["team", "year", "games_played", "opponent", "round", "result"]


def _write_player(tmp_path, rows, name="player_test_01011990"):
    df = pd.DataFrame(rows, columns=PERF_COLS)
    p = tmp_path / f"{name}_performance_details.csv"
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------
# Layer (a): counter-gap check
# --------------------------------------------------------------------------

def test_counter_gap_flags_missing_counter(tmp_path):
    # counters 1,2,4 -> 3 is missing (a silently deleted game row)
    rows = [
        ["X", 2010, 1, "A", 1, "W"],
        ["X", 2010, 2, "B", 2, "W"],
        ["X", 2010, 4, "C", 3, "W"],
    ]
    p = _write_player(tmp_path, rows)
    res = check_counter_gaps(p)
    assert res["missing"] == [3]
    assert res["duplicated"] == []
    assert res["counter_max"] == 4
    assert res["row_count"] == 3
    assert res["ok"] is False


def test_counter_gap_clean_contiguous(tmp_path):
    rows = [["X", 2010, i, "A", i, "W"] for i in range(1, 6)]
    p = _write_player(tmp_path, rows)
    res = check_counter_gaps(p)
    assert res["missing"] == []
    assert res["duplicated"] == []
    assert res["ok"] is True


def test_counter_gap_no_false_positive_on_missed_games(tmp_path):
    # A player who sits out real rounds still has a CONTIGUOUS counter, because
    # the counter only ticks for games played. round jumps 3 -> 5 (missed r4),
    # but games_played stays 1,2,3,4 -- must NOT be flagged.
    rows = [
        # cols: team, year, games_played, opponent, round, result
        ["X", 2010, 1, "A", 1, "W"],
        ["X", 2010, 2, "B", 2, "W"],
        ["X", 2010, 3, "C", 3, "W"],
        ["X", 2010, 4, "D", 5, "W"],  # counter 4, but round jumps to 5 (missed r4)
    ]
    p = _write_player(tmp_path, rows)
    res = check_counter_gaps(p)
    assert res["ok"] is True
    assert res["missing"] == []


def test_counter_gap_flags_duplicate_counter(tmp_path):
    rows = [
        ["X", 2010, 1, "A", 1, "W"],
        ["X", 2010, 2, "B", 2, "W"],
        ["X", 2010, 2, "B", 2, "W"],  # duplicated counter
    ]
    p = _write_player(tmp_path, rows)
    res = check_counter_gaps(p)
    assert res["duplicated"] == [2]
    assert res["ok"] is False


def test_counter_gap_normalises_arrows(tmp_path):
    rows = [
        ["X", 2010, "1", "A", 1, "W"],
        ["X", 2010, "↑2", "B", 2, "W"],  # milestone arrow
        ["X", 2010, "3", "C", 3, "W"],
    ]
    p = _write_player(tmp_path, rows)
    res = check_counter_gaps(p)
    assert res["ok"] is True
    assert res["missing"] == []


# --------------------------------------------------------------------------
# Layer (b): drawn-final cross-check
# --------------------------------------------------------------------------

def _matches_frame():
    # Grand Final drawn (equal totals) then replay; plus a normal round that
    # must NOT be treated as a drawn final.
    def g(final_g, final_b):
        return final_g, final_b

    rows = []
    # Drawn GF: Collingwood 9.14 (68) v St Kilda 10.8 (68) -> DRAW
    rows.append(dict(round_num="Grand Final", date="2010-09-25", year=2010,
                     team_1_team_name="Collingwood", team_1_final_goals=9, team_1_final_behinds=14,
                     team_2_team_name="St Kilda", team_2_final_goals=10, team_2_final_behinds=8))
    # Replay GF: Collingwood 16.12 (108) v St Kilda 7.10 (52) -> WIN
    rows.append(dict(round_num="Grand Final", date="2010-10-02", year=2010,
                     team_1_team_name="Collingwood", team_1_final_goals=16, team_1_final_behinds=12,
                     team_2_team_name="St Kilda", team_2_final_goals=7, team_2_final_behinds=10))
    # A normal (non-drawn) prelim, single game -> not a drawn final
    rows.append(dict(round_num="Preliminary Final", date="2010-08-23", year=2010,
                     team_1_team_name="Collingwood", team_1_final_goals=17, team_1_final_behinds=8,
                     team_2_team_name="Geelong", team_2_final_goals=10, team_2_final_behinds=10))
    return pd.DataFrame(rows)


def test_find_drawn_finals_identifies_gf_only():
    finals = find_drawn_finals(_matches_frame())
    assert len(finals) == 1
    f = finals[0]
    assert f["round_code"] == "GF"
    assert f["teams"] == frozenset({"Collingwood", "St Kilda"})
    assert f["year"] == 2010


def test_find_drawn_finals_ignores_non_drawn():
    # Two prelims same teams but the earlier is NOT a draw -> not a drawn final.
    df = pd.DataFrame([
        dict(round_num="Semi Final", date="2010-08-10", year=2010,
             team_1_team_name="A", team_1_final_goals=10, team_1_final_behinds=0,
             team_2_team_name="B", team_2_final_goals=9, team_2_final_behinds=0),
        dict(round_num="Semi Final", date="2010-08-17", year=2010,
             team_1_team_name="A", team_1_final_goals=12, team_1_final_behinds=0,
             team_2_team_name="B", team_2_final_goals=8, team_2_final_behinds=0),
    ])
    assert find_drawn_finals(df) == []


def _finals():
    return find_drawn_finals(_matches_frame())


def test_drawn_final_both_games_present_is_clean():
    # Player played both GF games: D + W rows, distinct results -> no issue.
    pdf = pd.DataFrame([
        ["Collingwood", 2010, 35, "St Kilda", "GF", "D"],
        ["Collingwood", 2010, 36, "St Kilda", "GF", "W"],
    ], columns=PERF_COLS)
    issues = check_drawn_final_consistency(pdf, 2010, _finals())
    assert issues == []


def test_drawn_final_single_replay_row_is_review():
    # Only the replay (W) row survives, no draw row -- cannot decide from the
    # file alone whether the draw was dropped+renumbered or never played.
    pdf = pd.DataFrame([
        ["Collingwood", 2010, 56, "St Kilda", "GF", "W"],
    ], columns=PERF_COLS)
    issues = check_drawn_final_consistency(pdf, 2010, _finals())
    assert len(issues) == 1
    assert issues[0]["severity"] == "REVIEW"
    assert issues[0]["round_code"] == "GF"


def test_drawn_final_duplicate_result_is_warning():
    # Two rows with the SAME result for the drawn round -> a real duplication.
    pdf = pd.DataFrame([
        ["Collingwood", 2010, 35, "St Kilda", "GF", "W"],
        ["Collingwood", 2010, 36, "St Kilda", "GF", "W"],
    ], columns=PERF_COLS)
    issues = check_drawn_final_consistency(pdf, 2010, _finals())
    assert len(issues) == 1
    assert issues[0]["severity"] == "WARNING"


def test_drawn_final_more_than_two_rows_is_warning():
    pdf = pd.DataFrame([
        ["Collingwood", 2010, 35, "St Kilda", "GF", "D"],
        ["Collingwood", 2010, 36, "St Kilda", "GF", "W"],
        ["Collingwood", 2010, 37, "St Kilda", "GF", "W"],
    ], columns=PERF_COLS)
    issues = check_drawn_final_consistency(pdf, 2010, _finals())
    assert len(issues) == 1
    assert issues[0]["severity"] == "WARNING"


def test_drawn_final_non_finalist_ignored():
    # A player from neither finalist club must never be flagged.
    pdf = pd.DataFrame([
        ["Geelong", 2010, 20, "Fremantle", "GF", "W"],
    ], columns=PERF_COLS)
    issues = check_drawn_final_consistency(pdf, 2010, _finals())
    assert issues == []


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def test_gaps_in_season_attributes_interior_gap(tmp_path):
    # counters 1,2 in 2025; 3 missing, 4,5 in 2026 -> the missing 3 is a 2026
    # interior gap (span of 2026 surviving counters is [4,5], but 3 sits between
    # last 2025 counter and first 2026 counter, so it belongs to the 2026 season
    # boundary). We attribute a gap to the season whose surviving-counter span
    # brackets it. cols: team, year, games_played, opponent, round, result
    rows = [
        ["X", 2025, 1, "A", 1, "W"],
        ["X", 2025, 2, "B", 2, "W"],
        ["X", 2026, 4, "C", 1, "W"],  # counter 3 missing -> a 2026 game deleted
        ["X", 2026, 5, "D", 2, "W"],
    ]
    p = _write_player(tmp_path, rows)
    assert gaps_in_season(p, 2026) == [3]
    assert gaps_in_season(p, 2025) == []


def test_gaps_in_season_ignores_historical_gap(tmp_path):
    # gap firmly inside 2025, none in 2026
    rows = [
        ["X", 2025, 1, "A", 1, "W"],
        ["X", 2025, 3, "B", 3, "W"],  # counter 2 missing (2025 gap)
        ["X", 2026, 4, "C", 1, "W"],
        ["X", 2026, 5, "D", 2, "W"],
    ]
    p = _write_player(tmp_path, rows)
    assert gaps_in_season(p, 2026) == []
    assert gaps_in_season(p, 2025) == [2]


def test_gaps_in_season_none_when_clean(tmp_path):
    rows = [["X", 2026, i, "A", i, "W"] for i in range(1, 5)]
    p = _write_player(tmp_path, rows)
    assert gaps_in_season(p, 2026) == []


def test_validate_player_file_returns_both_layers(tmp_path):
    rows = [
        ["X", 2010, 1, "A", 1, "W"],
        ["X", 2010, 3, "B", 3, "W"],  # counter 2 missing
    ]
    p = _write_player(tmp_path, rows)
    res = validate_player_file(p)
    assert res["counter"]["missing"] == [2]
    assert "finals" in res
    assert res["player"].startswith("player_test")
