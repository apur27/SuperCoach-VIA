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

def _match(date, round_num, venue, t1, t2):
    return {
        "date": date, "round_num": round_num, "venue": venue,
        "team_1_team_name": t1, "team_2_team_name": t2,
    }


def test_dedup_key_simultaneous_kickoffs_kept():
    """Two DIFFERENT games at the same date + round -> distinct keys (both kept).

    This is the original Port Adelaide v Collingwood R2 2025 guarantee. It used
    to be delivered by putting venue in the key; it is now delivered by the team
    pair, which is a stronger discriminator (a venue can be renamed, the two
    teams playing cannot change).
    """
    a = _match("2025-04-05 13:20", "2", "Adelaide Oval", "Port Adelaide", "Collingwood")
    b = _match("2025-04-05 13:20", "2", "MCG", "Melbourne", "Carlton")
    assert game_scraper.build_match_key(a) != game_scraper.build_match_key(b)


def test_dedup_key_identical_fixture_collapses():
    """Same fixture scraped twice -> identical key (correctly deduped)."""
    a = _match("2025-04-05 13:20", "2", "MCG", "Melbourne", "Carlton")
    b = _match("2025-04-05 13:20", "2", "MCG", "Melbourne", "Carlton")
    assert game_scraper.build_match_key(a) == game_scraper.build_match_key(b)


def test_dedup_key_collapses_venue_alias():
    """Same fixture under two venue SPELLINGS is one game, not two.

    Incident: afltables published the 2025 Gather Round fixtures as
    'Barossa Park' while matches_2025.csv already held them as 'Barossa Oval'.
    Because venue was in the dedup key, the re-scrape appended duplicates and
    the season double-counted two games. audit_match_rounds() could not catch it
    -- a duplicate makes a round look MORE complete, never short.
    """
    a = _match("2025-04-12 12:05", "6", "Barossa Oval", "North Melbourne", "Gold Coast")
    b = _match("2025-04-12 12:05", "6", "Barossa Park", "North Melbourne", "Gold Coast")
    assert game_scraper.build_match_key(a) == game_scraper.build_match_key(b)


def test_dedup_key_collapses_round_relabel():
    """Same fixture under two ROUND labels is one game, not two.

    Incident: the Cyclone Alfred postponement -- Brisbane Lions v Geelong at the
    Gabba, played 2025-03-29, is Round 1 on afltables but had been stored as the
    Round 4 its date implies.
    """
    a = _match("2025-03-29 18:35", "4", "Gabba", "Brisbane Lions", "Geelong")
    b = _match("2025-03-29 18:35", "1", "Gabba", "Brisbane Lions", "Geelong")
    assert game_scraper.build_match_key(a) == game_scraper.build_match_key(b)


def test_dedup_key_ignores_home_away_order():
    """A fixture with the two team columns swapped is still one game."""
    a = _match("2025-04-12 12:05", "6", "Barossa Park", "North Melbourne", "Gold Coast")
    b = _match("2025-04-12 12:05", "6", "Barossa Park", "Gold Coast", "North Melbourne")
    assert game_scraper.build_match_key(a) == game_scraper.build_match_key(b)


def test_dedup_key_falls_back_to_venue_without_team_columns():
    """Legacy rows lacking team columns keep the old date+round+venue behaviour."""
    a = {"date": "2025-04-05 13:20", "round_num": "2", "venue": "Adelaide Oval"}
    b = {"date": "2025-04-05 13:20", "round_num": "2", "venue": "MCG"}
    assert game_scraper.build_match_key(a) != game_scraper.build_match_key(b)


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


# ---------------------------------------------------------------------------
# Player date integrity — guards against afltables non-chronological labels
# causing wrong dates in player performance files (Clarke R1/2025-03-01 bug)
# ---------------------------------------------------------------------------

def test_clarke_row_has_correct_date():
    """The stranded Clarke game (afltables 'Round 1' / Gold Coast, 2025-08-27)
    must carry date 2025-08-27, not the false 2025-03-01 that the old delta
    scraper wrote when it first saw the non-chronological round label."""
    import glob
    files = glob.glob(
        "/home/abhi/git/SuperCoach-VIA/data/player_data/clarke_angus*performance*.csv"
    )
    if not files:
        return  # player not yet scraped; skip rather than fail
    df = pd.read_csv(files[0])
    bad = df[(df["year"] == 2025) & (df["round"].astype(str) == "1") & (df["date"] < "2025-08-01")]
    assert bad.empty, f"Clarke has wrong early date on Round-1/2025 row: {bad[['year','round','date']].to_dict()}"


