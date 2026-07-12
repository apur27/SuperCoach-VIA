"""Unit tests for the match dedup key + season completeness gate in
``scrapers/game_scraper.py``.

Bug 1 (dedup key too narrow): matches were deduplicated on ``date + round_num``
only. On rounds with two simultaneous kickoffs (same date + round, different
venues) one game was silently dropped -- Port Adelaide v Collingwood R2 2025 was
lost this way. Including ``venue`` in the key keeps both games.

Bug guard (completeness gate): a standalone ``check_match_completeness`` surfaces
any season where a team is short on games versus the season's modal game-count,
catching a silent drop without hard-coding bye schedules.

No network, no real data files -- these test pure key/aggregation logic.
"""

import os
import sys

import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scrapers import game_scraper  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matches_df(appearances):
    """Build a matches-style DataFrame from a flat list of team appearances.

    ``check_match_completeness`` counts appearances across both team columns,
    so pairing is irrelevant to per-team totals -- we chunk the flat list into
    (team_1, team_2) rows. ``len(appearances)`` must be even.
    """
    rows = []
    for j in range(0, len(appearances), 2):
        rows.append({
            "round_num": "1",
            "venue": "Ground",
            "team_1_team_name": appearances[j],
            "team_2_team_name": appearances[j + 1],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dedup key
# ---------------------------------------------------------------------------

def test_dedup_key_includes_venue_different_games_kept():
    """Same date + round but different venues -> distinct keys (both kept)."""
    a = {"date": "2025-04-05 13:20", "round_num": "2", "venue": "Adelaide Oval"}
    b = {"date": "2025-04-05 13:20", "round_num": "2", "venue": "MCG"}
    assert game_scraper.build_match_key(a) != game_scraper.build_match_key(b)


def test_dedup_key_same_venue_collapses():
    """Same date + round + venue -> identical key (correctly deduped)."""
    a = {"date": "2025-04-05 13:20", "round_num": "2", "venue": "MCG"}
    b = {"date": "2025-04-05 13:20", "round_num": "2", "venue": "MCG"}
    assert game_scraper.build_match_key(a) == game_scraper.build_match_key(b)


# ---------------------------------------------------------------------------
# Completeness gate
# ---------------------------------------------------------------------------

def test_completeness_gate_warns_on_missing_game():
    """17 teams at 23 games, 1 team at 21 -> warning naming the short team."""
    appearances = []
    for i in range(17):
        appearances += [f"Team{i}"] * 23
    appearances += ["WeakTeam"] * 21
    df = _matches_df(appearances)

    warns = game_scraper.check_match_completeness(df, 2025)

    assert any("WeakTeam" in w for w in warns), warns
    # Full-count teams must not be flagged.
    assert all("Team0" not in w for w in warns), warns


def test_completeness_gate_passes_balanced_season():
    """All 18 teams at 23 games -> no warnings."""
    appearances = []
    for i in range(18):
        appearances += [f"Team{i}"] * 23
    df = _matches_df(appearances)

    warns = game_scraper.check_match_completeness(df, 2025)

    assert warns == []
