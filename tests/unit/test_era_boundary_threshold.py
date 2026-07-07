"""Tests for scripts/era_boundary_threshold.py (Decision 3 helper).

The helper encodes the INCLUDE rule: qualify players on a dropna rate (over
recorded games only) and emit coverage + N-of-M so partial-coverage rates are
visible. These tests pin the qualifying convention and the coverage bookkeeping
against synthetic fixtures — no real data files are touched.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.era_boundary_threshold import (
    Qualifier,
    canonical_games,
    load_player_games,
    threshold_scan,
)


def _player(pid, n_games, disposals, goals, games_played=None):
    """Build a player's long frame. `disposals` is a list (None => unrecorded)."""
    if games_played is None:
        games_played = [str(i + 1) for i in range(n_games)]
    return pd.DataFrame(
        {
            "pid": [pid] * n_games,
            "games_played": games_played,
            "disposals": disposals,
            "goals": goals,
        }
    )


def test_canonical_games_uses_counter_when_it_exceeds_rowcount():
    # 3 rows but the afltables counter reached 5 (e.g. collapsed drawn GF).
    g = pd.DataFrame({"games_played": ["3", "4", "5"], "disposals": [1, 2, 3]})
    assert canonical_games(g) == 5


def test_canonical_games_strips_sub_markers_and_takes_rowcount_when_larger():
    # Counter carries sub markers; leading digits extracted; rowcount wins here.
    g = pd.DataFrame({"games_played": ["1↓", "2", "3↑", None, "4"], "disposals": [1] * 5})
    assert canonical_games(g) == 5


def test_qualify_full_coverage_player():
    df = _player("full", 250, [25.0] * 250, [2] * 250)  # 250 games, 25/g, 500 goals
    res = threshold_scan(
        df, min_games=200, rate_col="disposals", min_rate=20.0,
        count_col="goals", min_count=300.0,
    )
    assert res.n_qualifying == 1
    q = res.qualifiers[0]
    assert q.games == 250 and q.rate == 25.0 and not q.partial_coverage


def test_dropna_rate_qualifies_partial_coverage_player_and_flags_it():
    # 254 games; disposals recorded for only 50 (at 22.6), rest unrecorded (None);
    # goals 330. Dropna rate = 22.6 -> qualifies. Fill-zero rate ~ 4.45 -> would NOT.
    disp = [22.6] * 50 + [None] * 204
    df = _player("barassi_like", 254, disp, [1.3] * 254)  # ~330 goals
    res = threshold_scan(
        df, min_games=200, rate_col="disposals", min_rate=20.0,
        count_col="goals", min_count=300.0,
    )
    assert res.n_qualifying == 1
    q = res.qualifiers[0]
    assert q.rate_recorded_n == 50
    assert q.coverage == "50 of 254"
    assert q.partial_coverage is True          # coverage < 90% -> flagged
    assert q.rate == pytest.approx(22.6, abs=0.01)
    assert q.rate_fillzero < 5.0               # fill-zero would fail the threshold


def test_fillzero_would_exclude_what_dropna_includes():
    # Same partial player: prove the INCLUDE decision is what admits it by
    # checking the fill-zero rate is below the min_rate the dropna rate clears.
    disp = [22.6] * 50 + [None] * 204
    df = _player("barassi_like", 254, disp, [1.3] * 254)
    res = threshold_scan(
        df, min_games=200, rate_col="disposals", min_rate=20.0,
        count_col="goals", min_count=300.0,
    )
    q = res.qualifiers[0]
    assert q.rate >= 20.0 > q.rate_fillzero


def test_below_games_or_goals_threshold_excluded():
    small_games = _player("short", 150, [30.0] * 150, [3] * 150)     # <200 games
    low_goals = _player("nogoals", 250, [30.0] * 250, [0] * 250)     # 0 goals
    df = pd.concat([small_games, low_goals], ignore_index=True)
    res = threshold_scan(
        df, min_games=200, rate_col="disposals", min_rate=20.0,
        count_col="goals", min_count=300.0,
    )
    assert res.n_qualifying == 0
    assert res.n_considered == 2


def test_n_of_m_reports_considered_population():
    df = pd.concat(
        [
            _player("q", 250, [25.0] * 250, [2] * 250),   # qualifies
            _player("a", 100, [25.0] * 100, [4] * 100),   # too few games
            _player("b", 300, [10.0] * 300, [2] * 300),   # rate too low
        ],
        ignore_index=True,
    )
    res = threshold_scan(
        df, min_games=200, rate_col="disposals", min_rate=20.0,
        count_col="goals", min_count=300.0,
    )
    assert res.n_of_m == "1 of 3"


def test_no_recorded_rate_games_cannot_qualify():
    # Player with 300 games, 400 goals, but disposals never recorded.
    df = _player("nodisp", 300, [None] * 300, [1.5] * 300)
    res = threshold_scan(
        df, min_games=200, rate_col="disposals", min_rate=20.0,
        count_col="goals", min_count=300.0,
    )
    assert res.n_qualifying == 0


def test_load_player_games_reads_tmp_csvs(tmp_path):
    p = tmp_path / "smith_john_01011990_performance_details.csv"
    pd.DataFrame(
        {"games_played": ["1", "2"], "disposals": [20, 22], "goals": [1, 0],
         "year": [2020, 2020], "round": ["1", "2"]}
    ).to_csv(p, index=False)
    df = load_player_games(str(tmp_path))
    assert set(df["pid"]) == {"smith_john_01011990"}
    assert len(df) == 2


def test_load_player_games_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_player_games(str(tmp_path))
