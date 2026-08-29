"""TDD tests for scripts/check_round_settled.py (F5).

Deterministic replacement for the day-of-week timing heuristic. The probe reads
the latest matches_<year>.csv, finds the current (highest) home-and-away round,
and confirms every game present for that round has a non-zero final score. A
round with any 0-0 (unplayed / mid-play) game is UNSETTLED -> non-zero exit, so
the weekly cycle never runs on a round whose scores are not yet confirmed.
"""
import os
import sys

import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts import check_round_settled as crs


_COLS = [
    "round_num", "team_1_team_name", "team_1_final_goals", "team_1_final_behinds",
    "team_2_team_name", "team_2_final_goals", "team_2_final_behinds", "year",
]


def _write(tmp_path, rows):
    df = pd.DataFrame(rows, columns=_COLS)
    p = tmp_path / "matches_2026.csv"
    df.to_csv(p, index=False)
    return str(p)


def _game(rnd, t1, g1, b1, t2, g2, b2, year=2026):
    return [rnd, t1, g1, b1, t2, g2, b2, year]


def test_settled_round_returns_zero(tmp_path):
    path = _write(tmp_path, [
        _game(19, "Sydney", 12, 8, "Carlton", 10, 9),
        _game(19, "Geelong", 14, 6, "Gold Coast", 9, 11),
    ])
    code, unsettled = crs.check_round_settled(path)
    assert code == 0
    assert unsettled == []


def test_unsettled_round_returns_one_with_matchups(tmp_path):
    path = _write(tmp_path, [
        _game(19, "Sydney", 12, 8, "Carlton", 10, 9),
        _game(19, "Geelong", 0, 0, "Gold Coast", 0, 0),  # not yet played
    ])
    code, unsettled = crs.check_round_settled(path)
    assert code == 1
    assert any("Geelong" in m and "Gold Coast" in m for m in unsettled)


def test_current_round_is_highest_round(tmp_path):
    """An earlier settled round must not mask an unsettled latest round."""
    path = _write(tmp_path, [
        _game(18, "A", 10, 5, "B", 9, 7),
        _game(19, "C", 0, 0, "D", 0, 0),
    ])
    code, unsettled = crs.check_round_settled(path)
    assert code == 1
    assert any("C" in m for m in unsettled)


def test_missing_file_fails_closed(tmp_path):
    code, unsettled = crs.check_round_settled(str(tmp_path / "nope.csv"))
    assert code == 1


# ---------------------------------------------------------------------------
# Finals literacy (Surveyor F2).
#
# Finals rounds are stored as STRINGS ('Grand Final' in matches_<year>.csv,
# 'GF' in player performance files). The original probe coerced round_num with
# pd.to_numeric and dropped every non-numeric row, so once the last H&A round
# was settled the probe was pinned to it FOREVER: a run mid-Preliminary-Final
# passed Phase 0 without inspecting a single finals game.
#
# NOTE: the superseded test `test_finals_labels_ignored_for_current_round`
# asserted exactly that broken behaviour (a 0-0 Grand Final row alongside a
# settled R19 returned "settled"). It encoded the defect, so it is replaced by
# `test_unsettled_grand_final_is_the_current_stage` below.
# ---------------------------------------------------------------------------

def test_unsettled_grand_final_is_the_current_stage(tmp_path):
    """A present-but-unplayed Grand Final outranks the last H&A round."""
    path = _write(tmp_path, [
        _game(25, "A", 10, 5, "B", 9, 7),
        _game("Grand Final", "A", 0, 0, "B", 0, 0),
    ])
    code, unsettled = crs.check_round_settled(path)
    assert code == 1
    assert any("A" in m and "B" in m for m in unsettled)


def test_settled_grand_final_returns_zero(tmp_path):
    path = _write(tmp_path, [
        _game(25, "A", 10, 5, "B", 9, 7),
        _game("Grand Final", "A", 13, 9, "B", 11, 8),
    ])
    code, unsettled = crs.check_round_settled(path)
    assert code == 0
    assert unsettled == []


def test_short_code_finals_labels_recognised(tmp_path):
    """Player-corpus short codes (QF/EF/SF/PF/GF) work as well as long labels."""
    path = _write(tmp_path, [
        _game(25, "A", 10, 5, "B", 9, 7),
        _game("GF", "A", 0, 0, "B", 0, 0),
    ])
    code, unsettled = crs.check_round_settled(path)
    assert code == 1


def test_finals_stage_order_prelim_beats_semi(tmp_path):
    """An unsettled Preliminary Final is not masked by a settled Semi Final."""
    path = _write(tmp_path, [
        _game("Semi Final", "A", 12, 8, "B", 10, 9),
        _game("Semi Final", "C", 11, 7, "D", 10, 10),
        _game("Preliminary Final", "A", 0, 0, "C", 0, 0),
    ])
    code, unsettled = crs.check_round_settled(path)
    assert code == 1
    assert any("A" in m and "C" in m for m in unsettled)


def test_qualifying_and_elimination_share_finals_week_one(tmp_path):
    """QF and EF are the SAME week: a settled QF must not mask an unplayed EF."""
    path = _write(tmp_path, [
        _game("Qualifying Final", "A", 12, 8, "B", 10, 9),
        _game("Qualifying Final", "C", 11, 7, "D", 10, 10),
        _game("Elimination Final", "E", 13, 6, "F", 9, 12),
        _game("Elimination Final", "G", 0, 0, "H", 0, 0),
    ])
    code, unsettled = crs.check_round_settled(path)
    assert code == 1
    assert any("G" in m and "H" in m for m in unsettled)


def test_semi_final_outranks_settled_week_one(tmp_path):
    path = _write(tmp_path, [
        _game("Qualifying Final", "A", 12, 8, "B", 10, 9),
        _game("Elimination Final", "E", 13, 6, "F", 9, 12),
        _game("Semi Final", "B", 0, 0, "E", 0, 0),
    ])
    code, unsettled = crs.check_round_settled(path)
    assert code == 1
    assert any("B" in m and "E" in m for m in unsettled)


def test_unknown_round_label_is_ignored_not_current(tmp_path):
    """A junk label must not become the current stage (and must not fail open)."""
    path = _write(tmp_path, [
        _game(25, "A", 10, 5, "B", 9, 7),
        _game("Wildcard Round", "C", 0, 0, "D", 0, 0),
    ])
    code, unsettled = crs.check_round_settled(path)
    assert code == 0


# ---------------------------------------------------------------------------
# --print-last-ha-round: the harness needs the last COMPLETED numbered round
# (for the round label and the backtest upper bound) derived from data, not
# from a prediction artifact. Separate concern from "is the latest stage
# settled?" -- during finals the answer is the last H&A round, not the final.
# ---------------------------------------------------------------------------

def test_last_ha_round_ignores_finals(tmp_path):
    path = _write(tmp_path, [
        _game(24, "A", 10, 5, "B", 9, 7),
        _game(25, "C", 11, 6, "D", 8, 8),
        _game("Grand Final", "A", 13, 9, "C", 11, 8),
    ])
    assert crs.last_settled_ha_round(path) == 25


def test_last_ha_round_skips_unsettled_round(tmp_path):
    """R25 present but mid-play -> the last SETTLED numbered round is 24."""
    path = _write(tmp_path, [
        _game(24, "A", 10, 5, "B", 9, 7),
        _game(25, "C", 0, 0, "D", 0, 0),
    ])
    assert crs.last_settled_ha_round(path) == 24


def test_last_ha_round_none_when_no_settled_ha_round(tmp_path):
    path = _write(tmp_path, [
        _game(1, "C", 0, 0, "D", 0, 0),
    ])
    assert crs.last_settled_ha_round(path) is None


def test_print_last_ha_round_cli_prints_bare_number(tmp_path, capsys):
    path = _write(tmp_path, [
        _game(24, "A", 10, 5, "B", 9, 7),
        _game(25, "C", 11, 6, "D", 8, 8),
        _game("Grand Final", "A", 13, 9, "C", 11, 8),
    ])
    rc = crs.main(["--file", path, "--print-last-ha-round"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() == "25"


def test_print_last_ha_round_cli_exits_one_when_absent(tmp_path, capsys):
    path = _write(tmp_path, [_game(1, "C", 0, 0, "D", 0, 0)])
    rc = crs.main(["--file", path, "--print-last-ha-round"])
    out = capsys.readouterr().out
    assert rc == 1
    assert out.strip() == ""


def test_print_last_ha_round_missing_file_exits_one(tmp_path, capsys):
    rc = crs.main(["--file", str(tmp_path / "nope.csv"), "--print-last-ha-round"])
    assert rc == 1
    assert capsys.readouterr().out.strip() == ""
