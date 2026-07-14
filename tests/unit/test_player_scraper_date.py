"""TDD tests for the LIVE synthetic-date mint in scrapers/player_scraper.py.

Bug (F4): when the scraper stamps a numeric-round row it computes
`datetime(year, 3, 1) + timedelta(weeks=round_num - 1)`. For a game afltables
labels "Round 1" but that was actually played in August (rescheduled Opening
Round) or later, this mints a synthetic 2025-03-01-ish date. We repaired 176
existing files with scripts/fix_synthetic_dates.py, but the mint was still live
-- the next scrape re-corrupts any newly-scraped round-1 game.

Fix: when stamping a numeric-round row, cross-reference matches_<year>.csv by
team + opponent + round_num and use the real fixture date if a match is found;
otherwise fall back to the existing approximation (so rounds with no matches
data yet don't crash the scraper).

Pure fixtures: a temp matches CSV + an in-memory player soup. No network.
"""
import os
import sys

from bs4 import BeautifulSoup

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scrapers.player_scraper import PlayerScraper


def _make_player_soup(team: str, year: int, opponent: str, round_str: str) -> BeautifulSoup:
    """Minimal afltables-style player soup with one game row.

    cells[0]=games_played, cells[1]=opponent, cells[2]=round -- matching the
    scraper's positional parse.
    """
    row_cells = "\n".join(
        f"<td>{v}</td>" for v in [
            1,            # games_played
            opponent,     # opponent (cells[1])
            round_str,    # round (cells[2])
            "W 95-80",    # result
            10,           # jersey_num
            18, 6, 8, 26,
            2, 1,
            0, 6, 3,
            4, 2, 1, 0, 0,
            0, 2, 14, 8,
            1, 2, 0, 1, 85,
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


def _write_matches(tmp_path, year, rows):
    """Write a minimal matches_<year>.csv the resolver can read."""
    import pandas as pd
    cols = ["round_num", "venue", "date", "year",
            "team_1_team_name", "team_2_team_name"]
    df = pd.DataFrame(rows, columns=cols)
    path = os.path.join(str(tmp_path), f"matches_{year}.csv")
    df.to_csv(path, index=False)
    return str(tmp_path)


def _date_of(row):
    """Extract the stamped date (last column) from a scraped row."""
    return row[-1]


def test_round1_game_uses_real_fixture_date_not_synthetic(tmp_path):
    """A 'Round 1' Gold Coast vs Essendon game actually played 2025-08-27 must
    be stamped 2025-08-27, not the synthetic 2025-03-01 approximation."""
    matches_dir = _write_matches(tmp_path, 2025, [
        [1, "People First Stadium", "2025-08-27 19:20", 2025,
         "Gold Coast", "Essendon"],
    ])
    scraper = PlayerScraper()
    soup = _make_player_soup("Gold Coast", 2025, "Essendon", "1")
    rows = scraper._scrape_player_performance_details(
        soup, since_date=None, matches_dir=matches_dir
    )
    assert len(rows) == 1
    assert _date_of(rows[0]) == "2025-08-27", (
        f"expected real fixture date 2025-08-27, got {_date_of(rows[0])} "
        "-- synthetic mint still live"
    )


def test_round1_synthetic_date_would_have_been_march(tmp_path):
    """Guard: with NO matches file the fallback approximation is 2025-03-01,
    which confirms the previous test is actually exercising the fixture path."""
    scraper = PlayerScraper()
    soup = _make_player_soup("Gold Coast", 2025, "Essendon", "1")
    rows = scraper._scrape_player_performance_details(
        soup, since_date=None, matches_dir=str(tmp_path)  # empty dir, no CSV
    )
    assert len(rows) == 1
    assert _date_of(rows[0]) == "2025-03-01", (
        "fallback approximation changed -- test premise invalid"
    )


def test_no_matching_fixture_falls_back_to_approximation(tmp_path):
    """When the matches file exists but has no row for this pairing/round, the
    scraper must NOT crash and must fall back to the approximation."""
    matches_dir = _write_matches(tmp_path, 2025, [
        [1, "S.C.G.", "2025-03-07 19:40", 2025, "Sydney", "Hawthorn"],
    ])
    scraper = PlayerScraper()
    soup = _make_player_soup("Gold Coast", 2025, "Essendon", "5")
    rows = scraper._scrape_player_performance_details(
        soup, since_date=None, matches_dir=matches_dir
    )
    assert len(rows) == 1
    # R5 approx = 2025-03-01 + 4 weeks = 2025-03-29
    assert _date_of(rows[0]) == "2025-03-29"


def test_regular_round_matched_to_real_date(tmp_path):
    """A normal mid-season round should also snap to its real fixture date when
    present (the fix is not finals/round-1 specific)."""
    matches_dir = _write_matches(tmp_path, 2025, [
        [12, "M.C.G.", "2025-06-01 14:10", 2025, "Melbourne", "Carlton"],
    ])
    scraper = PlayerScraper()
    soup = _make_player_soup("Melbourne", 2025, "Carlton", "12")
    rows = scraper._scrape_player_performance_details(
        soup, since_date=None, matches_dir=matches_dir
    )
    assert len(rows) == 1
    assert _date_of(rows[0]) == "2025-06-01"
