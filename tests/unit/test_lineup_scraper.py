"""
Unit tests for the team-lineup parser in ``scrapers/game_scraper.py``.

Bug (garbage lineups since 2025 R3): ``MatchScraper._extract_player_names`` read
``cells[0]`` of each row in the afltables match-stats table. That table's first
column is the jersey ``#`` (e.g. "5", "44", "36 down-arrow"), and the *player
name* lives in the second column ("Player") formatted "Surname, Firstname".
The parser also dropped the historical "Surname, Firstname" -> "Firstname
Surname" reversal. Result: the ``players`` field filled with jersey numbers,
sub markers, and footer labels ("Rushed", "Totals", "Opposition") instead of
names, for every round from 2025 R3 onward (and all of 2026).

Fix: locate the "Player" column from the header, read that column, reverse
"Surname, Firstname" -> "Firstname Surname", and skip non-player rows. If a
populated stats table yields ZERO valid names, raise ``LineupParseError`` so a
future structure change fails loudly instead of writing garbage.

No network: every test builds a hardcoded BeautifulSoup table. There is no HTTP.
"""

import os
import sys

import pytest
from bs4 import BeautifulSoup

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scrapers.game_scraper import MatchScraper, LineupParseError  # noqa: E402


def _table(html: str) -> BeautifulSoup:
    """Parse a raw <table> HTML string into its BeautifulSoup table element."""
    return BeautifulSoup(html, "html.parser").find("table")


# A faithful slice of the CURRENT afltables match-stats table structure:
#   row 0: caption (th, spans the table)
#   row 1: header  (#, Player, KI, MK, ...)
#   rows 2+: data  (jersey#, "Surname, Firstname", stats...)
#   footer rows: Rushed / Totals / Opposition (label in col 0, numbers after)
CURRENT_STRUCTURE = """
<table class="sortable">
  <tr><th colspan="4">Brisbane Lions Match Statistics [Season][Game by Game]</th></tr>
  <tr><th>#</th><th>Player</th><th>KI</th><th>MK</th></tr>
  <tr><td>4</td><td>Ah Chee, Callum</td><td>5</td><td>5</td></tr>
  <tr><td>31</td><td>Andrews, Harris</td><td>9</td><td>8</td></tr>
  <tr><td>36 ↓</td><td>Amartey, Joel</td><td>3</td><td>1</td></tr>
  <tr><td>10 ↑</td><td>Ashcroft, Levi</td><td>9</td><td>5</td></tr>
  <tr><td>Rushed</td><td>3</td><td></td><td></td></tr>
  <tr><td>Totals</td><td>232</td><td>93</td><td>77</td></tr>
  <tr><td>Opposition</td><td>219</td><td>77</td><td>61</td></tr>
</table>
"""

# The broken shape the bug actually produced downstream: the "Player" column
# holds jersey numbers / sub-markers, not names. No comma-form name anywhere.
MALFORMED_NO_NAMES = """
<table class="sortable">
  <tr><th>#</th><th>Player</th><th>KI</th></tr>
  <tr><td>5</td><td>44</td><td>6</td></tr>
  <tr><td>12</td><td>34 ↑</td><td>20</td></tr>
  <tr><td>48</td><td>2</td><td>29</td></tr>
</table>
"""


class TestExtractPlayerNames:
    def test_current_structure_extracts_reversed_names(self):
        scraper = MatchScraper()
        names = scraper._extract_player_names(_table(CURRENT_STRUCTURE))
        # "Surname, Firstname" -> "Firstname Surname", matching historical CSVs.
        assert names == [
            "Callum Ah Chee",
            "Harris Andrews",
            "Joel Amartey",
            "Levi Ashcroft",
        ]

    def test_footer_and_header_rows_excluded(self):
        scraper = MatchScraper()
        names = scraper._extract_player_names(_table(CURRENT_STRUCTURE))
        for junk in ("Rushed", "Totals", "Opposition", "Player", "3", "232", "219"):
            assert junk not in names

    def test_no_jersey_numbers_or_markers_leak_into_names(self):
        scraper = MatchScraper()
        names = scraper._extract_player_names(_table(CURRENT_STRUCTURE))
        assert not any(ch.isdigit() for name in names for ch in name)
        assert not any(("↑" in name or "↓" in name) for name in names)

    def test_malformed_table_raises_rather_than_writing_garbage(self):
        scraper = MatchScraper()
        with pytest.raises(LineupParseError):
            scraper._extract_player_names(_table(MALFORMED_NO_NAMES))
