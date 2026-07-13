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


# ---------------------------------------------------------------------------
# Counter-aware delta — a genuinely-new game whose afltables round label is
# non-chronological must NOT be dropped by the approximate-date filter.
#
# Bug (Angus Clarke, 2026-07-13 weekly refresh): afltables placed his career
# game #14 (Essendon-2025 table, round label "1" vs Gold Coast, real date
# 2025-08-27) with a low round number. The date approximation
# datetime(2025,3,1)+weeks(0) = 2025-03-01 fell BEFORE since_date (2025-08-16,
# his last recorded game), so the delta filter skipped it. The genuinely-later
# 2026 R19 game (counter 15) survived, leaving a 13->15 games_played gap that
# fail-closed the phantom-row gate.
#
# Fix: the games_played career counter is the authoritative "already seen"
# signal. A row whose counter exceeds the max already on file is a new game and
# must be kept regardless of its approximated date.
# ---------------------------------------------------------------------------

def _make_player_soup_gp(team: str, year: int, round_str: str, games_played) -> BeautifulSoup:
    """Minimal afltables player soup with one row and a settable games_played."""
    row_cells = "\n".join(
        f"<td>{v}</td>" for v in [
            games_played,   # games_played — the counter under test
            "Gold Coast",   # opponent
            round_str,      # round
            "L 50-120",     # result
            36,             # jersey_num
            11, 6, 10, 21,  # kicks marks handballs disposals
            0, 0,           # goals behinds
            0, 0, 2,        # hit_outs tackles rebound_50s
            2, 0, 1, 2, 0,  # inside_50s clearances clangers free_for free_against
            0, 8, 12, 0,    # brownlow cp up contested_marks
            0, 1, 0, 0, 90, # marks_inside_50 one_pct bounces goal_assist pct_played
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


def test_new_game_with_low_round_label_not_dropped():
    """Counter 14 > max 13: genuinely-new game must survive despite old approx date."""
    scraper = PlayerScraper()
    soup = _make_player_soup_gp("Essendon", 2025, "1", 14)
    rows = scraper._scrape_player_performance_details(
        soup, since_date=datetime(2025, 8, 16), max_counter=13)
    assert len(rows) == 1, "new game (counter 14 > max 13) was dropped by the date filter"


def test_already_seen_game_with_low_round_label_still_skipped():
    """Counter 13 <= max 13: an already-recorded game must stay filtered (delta efficiency)."""
    scraper = PlayerScraper()
    soup = _make_player_soup_gp("Essendon", 2025, "1", 13)
    rows = scraper._scrape_player_performance_details(
        soup, since_date=datetime(2025, 8, 16), max_counter=13)
    assert len(rows) == 0, "already-seen game (counter 13 <= max 13) should stay filtered"


def test_max_counter_none_preserves_date_only_behaviour():
    """Without counter context the old date-only filter must be unchanged."""
    scraper = PlayerScraper()
    soup = _make_player_soup_gp("Essendon", 2025, "5", 14)
    rows = scraper._scrape_player_performance_details(
        soup, since_date=datetime(2025, 8, 16))
    assert len(rows) == 0, "date-only path (no max_counter) regressed"


# ---------------------------------------------------------------------------
# Fixture-accurate finals dates — root-cause fix for the recurring date drift.
#
# The _FINALS_WEEK approximation places finals in late August, ~1 month BEFORE
# their real September fixture dates. Every re-scrape therefore overwrites the
# fixture-accurate dates (corrected offline in commit 58f1a4f20) with wrong
# August approximations. The fix: resolve the real date from
# data/matches/matches_<year>.csv when the fixture is found.
# ---------------------------------------------------------------------------

_DATE_COL = PlayerScraper.PLAYER_COL_TITLES.index("date")
_OPP_COL = PlayerScraper.PLAYER_COL_TITLES.index("opponent")


def _write_matches_csv(tmp_path, year, rows):
    """rows: list of (round_num_fullname, date, team_1, team_2)."""
    import csv
    d = tmp_path / "matches"
    d.mkdir(exist_ok=True)
    p = d / f"matches_{year}.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["round_num", "venue", "date", "year", "attendance",
                    "team_1_team_name", "team_2_team_name"])
        for rn, date, t1, t2 in rows:
            w.writerow([rn, "M.C.G.", date, year, 90000, t1, t2])
    return str(d)


def _scrape_with_matches(team, opponent, year, round_str, matches_dir):
    """Scrape a single finals row, resolving date against matches_dir."""
    scraper = PlayerScraper()
    soup = _make_player_soup(team, year, round_str)
    # opponent in _make_player_soup is hardcoded to 'Richmond'; rebuild with the
    # opponent we want for the fixture lookup.
    html = str(soup).replace("Richmond", opponent, 1)
    soup = BeautifulSoup(html, "html.parser")
    return scraper._scrape_player_performance_details(
        soup, since_date=None, matches_dir=matches_dir)


def test_finals_date_resolved_from_matches_fixture(tmp_path):
    """A GF row must carry the REAL fixture date, not the August approximation."""
    matches_dir = _write_matches_csv(tmp_path, 2024, [
        ("Grand Final", "2024-09-28 14:30", "Sydney", "Brisbane Lions"),
    ])
    rows = _scrape_with_matches("Brisbane Lions", "Sydney", 2024, "GF", matches_dir)
    assert len(rows) == 1
    assert rows[0][_DATE_COL] == "2024-09-28", (
        f"GF date should be the fixture date 2024-09-28, got {rows[0][_DATE_COL]}")


def test_finals_date_team_order_independent(tmp_path):
    """Fixture lookup matches regardless of team/opponent ordering in the fixture."""
    matches_dir = _write_matches_csv(tmp_path, 2024, [
        ("Preliminary Final", "2024-09-21 17:15", "Geelong", "Brisbane Lions"),
    ])
    # player is Brisbane, opponent Geelong; fixture has Geelong as team_1
    rows = _scrape_with_matches("Brisbane Lions", "Geelong", 2024, "PF", matches_dir)
    assert len(rows) == 1
    assert rows[0][_DATE_COL] == "2024-09-21"


def test_finals_date_falls_back_to_approximation_when_no_fixture(tmp_path):
    """If the fixture is absent, fall back to the _FINALS_WEEK approximation
    (must still be in finals territory, never March)."""
    matches_dir = _write_matches_csv(tmp_path, 2024, [])  # empty fixtures
    rows = _scrape_with_matches("Brisbane Lions", "Sydney", 2024, "GF", matches_dir)
    assert len(rows) == 1
    # approximation: week 27 -> late Aug; must NOT be March, must be a finals month
    month = int(rows[0][_DATE_COL].split("-")[1])
    assert month >= 8, f"fallback date {rows[0][_DATE_COL]} is not in finals territory"


def test_regular_round_ignores_matches_lookup(tmp_path):
    """Home-and-away rounds keep the approximation even when a matches file exists."""
    matches_dir = _write_matches_csv(tmp_path, 2024, [
        ("Grand Final", "2024-09-28 14:30", "Sydney", "Brisbane Lions"),
    ])
    rows = _scrape_with_matches("Brisbane Lions", "Sydney", 2024, "R5", matches_dir)
    assert len(rows) == 1
    # R5 approximates to ~early May
    assert rows[0][_DATE_COL].startswith("2024-0"), rows[0][_DATE_COL]
    assert int(rows[0][_DATE_COL].split("-")[1]) <= 5


def test_no_matches_dir_uses_approximation():
    """Backwards-compat: with no matches_dir, behaviour is the pure approximation."""
    scraper = PlayerScraper()
    soup = _make_player_soup("Collingwood", 2025, "GF")
    rows = scraper._scrape_player_performance_details(soup, since_date=None)
    assert len(rows) == 1
    month = int(rows[0][_DATE_COL].split("-")[1])
    assert month >= 8
