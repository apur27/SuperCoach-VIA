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


def test_finals_labels_ignored_for_current_round(tmp_path):
    """Non-integer round labels (finals) are not the H&A 'current round'."""
    rows = [
        _game(19, "A", 10, 5, "B", 9, 7),
        _game("Grand Final", "A", 0, 0, "B", 0, 0),
    ]
    path = _write(tmp_path, rows)
    code, unsettled = crs.check_round_settled(path)
    # Highest integer H&A round is 19 and it is settled.
    assert code == 0


def test_missing_file_fails_closed(tmp_path):
    code, unsettled = crs.check_round_settled(str(tmp_path / "nope.csv"))
    assert code == 1
