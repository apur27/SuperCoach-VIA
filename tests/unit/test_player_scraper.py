"""
Unit tests for scrapers/player_scraper.py — focused on the finals-round date bug.

Bug: Finals rounds (QF, EF, SF, PF, GF) strip to empty string via re.sub(r'\D', ''),
causing round_num to default to 1, which approximates game_date to March 1.
On any incremental run where since_date is mid-season, March 1 < since_date and
finals rows are permanently skipped — wiped every Wednesday by the weekly harness.

Fix: _FINALS_WEEK maps finals labels to week offsets 24-27, placing their
approximated dates in late September / October, safely after any mid-season since_date.
"""
import sys
import os
from datetime import datetime

import pytest
from bs4 import BeautifulSoup

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scrapers.player_scraper import PlayerScraper, _FINALS_WEEK


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _make_player_soup(team: str, year: int, round_str: str) -> BeautifulSoup:
    """Build a minimal afltables-style player soup with one game row."""
    # afltables tables have <th colspan="28">Team - Year</th> as the section header
    row_cells = "\n".join(
        f"<td>{v}</td>" for v in [
            1,              # games_played
            "Richmond",     # opponent
            round_str,      # round — the field under test
            "W 95-80",      # result
            10,             # jersey_num
            18, 6, 8, 26,   # kicks marks handballs disposals
            2, 1,           # goals behinds
            0, 6, 3,        # hit_outs tackles rebound_50s
            4, 2, 1, 0, 0,  # inside_50s clearances clangers free_for free_against
            0, 2, 14, 8,    # brownlow cp up contested_marks
            1, 2, 0, 1, 85, # marks_inside_50 one_pct bounces goal_assist pct_played
        ]
    )
    html = f"""
    <table>
      <tr><th colspan="28">{team} - {year}</th></tr>
      <tbody>
        <tr>{row_cells}</tr>
      </tbody>
    </table>
    """
    return BeautifulSoup(html, "html.parser")


# ---------------------------------------------------------------------------
# Test _FINALS_WEEK constant
# ---------------------------------------------------------------------------

def test_finals_week_mapping_complete():
    """All five finals labels must be present and map to week >= 23."""
    for label in ("QF", "EF", "SF", "PF", "GF"):
        assert label in _FINALS_WEEK, f"{label} missing from _FINALS_WEEK"
        assert _FINALS_WEEK[label] >= 23, f"{label} maps to week {_FINALS_WEEK[label]}, expected >= 23"


def test_finals_week_ordering():
    """GF week must be the latest; QF/EF must be earliest finals."""
    assert _FINALS_WEEK["GF"] > _FINALS_WEEK["PF"] > _FINALS_WEEK["SF"]
    assert _FINALS_WEEK["QF"] == _FINALS_WEEK["EF"]  # same week, different games


# ---------------------------------------------------------------------------
# Test that finals rows are NOT skipped during incremental scrape
# ---------------------------------------------------------------------------

def _scrape(round_str: str, since_date: datetime) -> list:
    scraper = PlayerScraper()
    soup = _make_player_soup("Collingwood", 2025, round_str)
    return scraper._scrape_player_performance_details(soup, since_date=since_date)


def test_qf_row_not_skipped_with_mid_season_since_date():
    """QF row in 2025 must be returned when since_date is mid-2025 (August)."""
    rows = _scrape("QF", since_date=datetime(2025, 8, 1))
    assert len(rows) == 1, "QF row was skipped — finals date approximated before since_date"


def test_ef_row_not_skipped():
    rows = _scrape("EF", since_date=datetime(2025, 8, 1))
    assert len(rows) == 1, "EF row was skipped"


def test_sf_row_not_skipped():
    rows = _scrape("SF", since_date=datetime(2025, 8, 1))
    assert len(rows) == 1, "SF row was skipped"


def test_pf_row_not_skipped():
    rows = _scrape("PF", since_date=datetime(2025, 8, 1))
    assert len(rows) == 1, "PF row was skipped"


def test_gf_row_not_skipped():
    rows = _scrape("GF", since_date=datetime(2025, 8, 1))
    assert len(rows) == 1, "GF row was skipped — this is the worst offender historically"


# ---------------------------------------------------------------------------
# Test that regular-round filtering still works correctly
# ---------------------------------------------------------------------------

def test_regular_round_before_since_date_is_skipped():
    """R5 in 2025 approximates to ~early May; must be skipped when since_date=Aug."""
    rows = _scrape("R5", since_date=datetime(2025, 8, 1))
    assert len(rows) == 0, "Old regular round should have been filtered out"


def test_regular_round_after_since_date_is_included():
    """R22 in 2025 approximates to ~late July; must be returned when since_date=June."""
    rows = _scrape("R22", since_date=datetime(2025, 6, 1))
    assert len(rows) == 1, "Recent regular round was incorrectly skipped"


def test_no_since_date_returns_all_rows():
    """Without since_date, all rows are returned regardless of round."""
    rows = _scrape("GF", since_date=None)
    assert len(rows) == 1

    rows = _scrape("R1", since_date=None)
    assert len(rows) == 1
