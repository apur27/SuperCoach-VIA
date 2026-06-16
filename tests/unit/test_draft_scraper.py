"""
Unit tests for the AFL draft-history scraper in ``scrapers/draft_scraper.py``.

Source is Wikipedia (``{year}_AFL_draft``); afltables 404s for drafts. All HTML
is mocked inline -- no test makes a real network request. The network boundary
is ``DraftScraper._fetch_soup``, which is patched to return a BeautifulSoup built
from a string fixture.

Written BEFORE the implementation (TDD).
"""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest
from bs4 import BeautifulSoup

# Make ``scrapers`` importable regardless of the directory pytest is invoked from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scrapers.draft_scraper import (  # noqa: E402
    DraftScraper,
    find_national_draft_table,
    parse_draft_table,
    clean_player_name,
    player_key,
)


def _wikitable_with_heading(heading, headers, rows, level="h2"):
    """A section heading followed by one wikitable (mimics Wikipedia layout)."""
    return f"<{level}>{heading}</{level}>" + _wikitable(headers, rows)


# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

def _wikitable(headers, rows):
    """Build a single <table class="wikitable"> HTML string."""
    thead = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    return (
        '<table class="wikitable">'
        f"<tr>{thead}</tr>"
        f"{body}"
        "</table>"
    )


# A page with three tables; only the one under a "national draft" heading is the
# national draft. The first (no heading) and last (rookie draft) must be ignored.
_THREE_TABLE_PAGE = (
    "<html><body>"
    + _wikitable(["Team", "Coach"], [["Richmond", "Hardwick"]])
    + _wikitable_with_heading(
        "2004 national draft",
        ["Round", "Pick", "Player", "Recruited from", "Club"],
        [
            ["1", "4", "Richard Tambling", "Southern Districts", "Richmond"],
            ["1", "5", "Lance Franklin", "Perth Football Club", "Hawthorn"],
        ],
    )
    + _wikitable_with_heading(
        "2005 rookie draft", ["Pick", "Player"], [["1", "Someone Else"]]
    )
    + "</body></html>"
)


# Faithful minimal copy of the 2004 national draft table structure from the brief.
_2004_PAGE = (
    "<html><body>"
    + _wikitable(["Some", "Other", "Table"], [["a", "b", "c"]])
    + _wikitable_with_heading(
        "2004 national draft",
        ["Round", "Pick", "Player", "Recruited from", "Club"],
        [
            ["Priority", "1", "Brett Deledio", "Murray Bushrangers", "Richmond"],
            ["Priority", "2", "Jarryd Roughead", "Gippsland Power", "Hawthorn"],
            ["Priority", "3", "Ryan Griffen", "South Adelaide Football Club", "Western Bulldogs"],
            ["1", "4", "Richard Tambling", "Southern Districts Football Club", "Richmond"],
            ["1", "5", "Lance Franklin", "Perth Football Club", "Hawthorn"],
        ],
    )
    + "</body></html>"
)


def _soup(html):
    return BeautifulSoup(html, "html.parser")


# ---------------------------------------------------------------------------
# Test 1: table detection by header name, not index
# ---------------------------------------------------------------------------

def test_find_national_draft_table():
    table = find_national_draft_table(_soup(_THREE_TABLE_PAGE))
    assert table is not None
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    assert "Round" in headers and "Pick" in headers and "Player" in headers
    # It is the table under the "national draft" heading (the middle one),
    # proving heading-based detection -- not first-match, not last-match.
    all_tables = _soup(_THREE_TABLE_PAGE).find_all("table")
    assert table.get_text() == all_tables[1].get_text()
    assert "Lance Franklin" in table.get_text()
    assert "Someone Else" not in table.get_text()


# ---------------------------------------------------------------------------
# Test 2: Priority round -> round=0
# ---------------------------------------------------------------------------

def test_parse_priority_round():
    table = find_national_draft_table(_soup(_2004_PAGE))
    rows = parse_draft_table(table, year=2004)
    deledio = next(r for r in rows if r["pick"] == 1)
    assert deledio["round"] == 0
    assert deledio["player_name"] == "Brett Deledio"


# ---------------------------------------------------------------------------
# Test 3: rows whose Pick is non-numeric (sub-headers) are skipped
# ---------------------------------------------------------------------------

def test_skip_non_numeric_picks():
    html = (
        "<html><body>"
        + _wikitable_with_heading(
            "2022 national draft",
            ["Round", "Pick", "Player", "Club", "Recruited from"],
            [
                ["1", "1", "Real Player", "Carlton", "Calder Cannons"],
                ["Club", "League", "", "", ""],   # sub-header row, Pick="League"
                ["", "Club", "", "", ""],          # sub-header row, Pick="Club"
                ["2", "20", "Another Player", "Sydney", "Sydney Academy"],
            ],
        )
        + "</body></html>"
    )
    table = find_national_draft_table(_soup(html))
    rows = parse_draft_table(table, year=2022)
    picks = sorted(r["pick"] for r in rows)
    assert picks == [1, 20]
    assert all(isinstance(r["pick"], int) for r in rows)


# ---------------------------------------------------------------------------
# Test 4: footnote markers stripped from player names
# ---------------------------------------------------------------------------

def test_strip_footnote_markers():
    assert clean_player_name("Lance Franklin[1]") == "Lance Franklin"
    assert clean_player_name("Lance Franklin[1][2]") == "Lance Franklin"
    assert clean_player_name("  Brett  Deledio  ") == "Brett Deledio"
    # player_key derived from the footnoted form is clean and normalised.
    assert player_key("Lance Franklin[1]") == "franklin_lance"


# ---------------------------------------------------------------------------
# Test 5: player_key generation
# ---------------------------------------------------------------------------

def test_player_key_generation():
    assert player_key("Lance Franklin") == "franklin_lance"
    assert player_key("Brett Deledio") == "deledio_brett"
    # Last token is the surname, the rest is the first-name run.
    assert player_key("Nic Naitanui") == "naitanui_nic"


# ---------------------------------------------------------------------------
# Test 6: dedup on re-run -- same (year, pick) not duplicated
# ---------------------------------------------------------------------------

def test_dedup_on_rerun(tmp_path):
    out = tmp_path / "afl_draft_history.csv"
    scraper = DraftScraper(out_path=str(out))

    with patch.object(scraper, "_fetch_soup", return_value=_soup(_2004_PAGE)):
        scraper.scrape_year(2004)
        first = pd.read_csv(out)
        scraper.scrape_year(2004)  # re-run same year
        second = pd.read_csv(out)

    assert len(first) == len(second)
    assert not second.duplicated(subset=["year", "pick"]).any()


# ---------------------------------------------------------------------------
# Test: select the NATIONAL draft table, not the mid-season / rookie one
# ---------------------------------------------------------------------------

def test_select_national_draft_among_multiple():
    """
    From 2018+, a page has several Round/Pick/Player tables (mid-season rookie,
    national, pre-season, rookie). The national draft is identified by its
    section heading containing "national draft" -- NOT by being first.
    """
    page = (
        "<html><body>"
        + _wikitable_with_heading(
            "Mid-season rookie draft",
            ["Round", "Pick", "Player", "Club", "Recruited from"],
            [["1", "1", "Midseason Guy", "Geelong", "VFL"]],
        )
        + _wikitable_with_heading(
            "2022 national draft",
            ["Round", "Pick", "Player", "Club", "Recruited from"],
            [["1", "1", "National Guy", "GWS", "GWV"]],
        )
        + _wikitable_with_heading(
            "2023 rookie draft",
            ["Round", "Pick", "Player", "Club", "Recruited from"],
            [["1", "1", "Rookie Guy", "Carlton", "VFL"]],
        )
        + "</body></html>"
    )
    table = find_national_draft_table(_soup(page))
    rows = parse_draft_table(table, year=2022)
    assert len(rows) == 1
    assert rows[0]["player_name"] == "National Guy"


def test_no_national_draft_table_returns_none():
    """A page with only rookie/mid-season drafts has no national table."""
    page = (
        "<html><body>"
        + _wikitable_with_heading(
            "1998 rookie draft",
            ["Round", "Pick", "Player", "Recruited from", "Club"],
            [["1", "1", "Rookie Only", "Perth", "Fremantle"]],
        )
        + "</body></html>"
    )
    assert find_national_draft_table(_soup(page)) is None


# ---------------------------------------------------------------------------
# Test: single-round-era table (no "Round" column) -- 1997 schema
# ---------------------------------------------------------------------------

def test_no_round_column_defaults_round_one():
    """
    1990s national drafts list Pick/Player/Recruited from/Club with NO Round
    column. The table must still be detected (Round optional) and every pick
    defaults to round=1.
    """
    page = (
        "<html><body>"
        + _wikitable_with_heading(
            "1997 national draft",
            ["Pick", "Player", "Recruited from", "Club"],
            [
                ["1", "Jeff White", "Southern U18", "Fremantle"],
                ["2", "Some Player", "Perth", "Hawthorn"],
            ],
        )
        + "</body></html>"
    )
    table = find_national_draft_table(_soup(page))
    assert table is not None
    rows = parse_draft_table(table, year=1997)
    assert {r["pick"] for r in rows} == {1, 2}
    assert all(r["round"] == 1 for r in rows)
    p1 = next(r for r in rows if r["pick"] == 1)
    assert p1["player_name"] == "Jeff White"
    assert p1["club"] == "Fremantle"
    assert p1["recruited_from"] == "Southern U18"


# ---------------------------------------------------------------------------
# Test: drafting-club column under its various era header names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("club_header", ["Club", "Recruited by", "Recruited to", "Drafted to"])
def test_drafting_club_header_aliases(club_header):
    """
    The drafting club column is headed "Club" (modern), "Recruited by" (1993),
    "Recruited to" (2000) or "Drafted to" (2012). All must populate ``club`` --
    leaving it NaN is silent data loss.
    """
    page = (
        "<html><body>"
        + _wikitable_with_heading(
            "2000 national draft",
            ["Round", "Pick", "Player", "Recruited from", club_header],
            [["1", "1", "Nick Riewoldt", "Southport Sharks", "St Kilda"]],
        )
        + "</body></html>"
    )
    table = find_national_draft_table(_soup(page))
    rows = parse_draft_table(table, year=2000)
    assert rows[0]["club"] == "St Kilda"
    assert rows[0]["recruited_from"] == "Southport Sharks"


# ---------------------------------------------------------------------------
# Test: "Rd." abbreviation header (2019 schema) is recognised as Round
# ---------------------------------------------------------------------------

def test_rd_abbreviation_header():
    page = (
        "<html><body>"
        + _wikitable_with_heading(
            "2019 national draft",
            ["Rd.", "Pick", "Player", "Club", "Recruited from"],
            [["1", "1", "Matt Rowell", "Gold Coast", "Oakleigh"]],
        )
        + "</body></html>"
    )
    table = find_national_draft_table(_soup(page))
    assert table is not None
    rows = parse_draft_table(table, year=2019)
    assert rows[0]["round"] == 1
    assert rows[0]["player_name"] == "Matt Rowell"


# ---------------------------------------------------------------------------
# Test 6b: rowspanned Round cell is carried forward to later picks in the round
# ---------------------------------------------------------------------------

def test_rowspanned_round_carried_forward():
    """
    Modern pages (2018+) write the Round number once per round via rowspan; the
    2nd..nth picks of a round omit the Round <td>. Without rowspan expansion the
    Pick column shifts left and those picks get silently dropped (the 2-6 picks
    bug). The whole round must survive.
    """
    html = (
        "<html><body>"
        "<h2>2022 national draft</h2>"
        '<table class="wikitable">'
        "<tr><th>Round</th><th>Pick</th><th>Player</th><th>Club</th>"
        "<th>Recruited from</th></tr>"
        # Round 1 spans 3 picks; only the first row carries the Round cell.
        '<tr><td rowspan="3">1</td><td>1</td><td>First Pick</td>'
        "<td>Carlton</td><td>Calder Cannons</td></tr>"
        "<tr><td>2</td><td>Second Pick</td><td>Sydney</td><td>Sydney Academy</td></tr>"
        "<tr><td>3</td><td>Third Pick</td><td>Hawthorn</td><td>Sandringham</td></tr>"
        "</table>"
        "</body></html>"
    )
    table = find_national_draft_table(_soup(html))
    rows = parse_draft_table(table, year=2022)
    by_pick = {r["pick"]: r for r in rows}
    assert sorted(by_pick) == [1, 2, 3]
    # Every pick in the round inherits round=1.
    assert all(by_pick[p]["round"] == 1 for p in (1, 2, 3))
    # And the non-Round columns stay aligned (no left-shift).
    assert by_pick[2]["player_name"] == "Second Pick"
    assert by_pick[2]["club"] == "Sydney"
    assert by_pick[3]["recruited_from"] == "Sandringham"


# ---------------------------------------------------------------------------
# Test 7: full 2004 page parse -- pick 5 = Lance Franklin / Hawthorn
# ---------------------------------------------------------------------------

def test_full_page_parse(tmp_path):
    out = tmp_path / "afl_draft_history.csv"
    scraper = DraftScraper(out_path=str(out))

    with patch.object(scraper, "_fetch_soup", return_value=_soup(_2004_PAGE)):
        scraper.scrape_year(2004)

    df = pd.read_csv(out)
    p1 = df[df["pick"] == 1].iloc[0]
    assert p1["player_name"] == "Brett Deledio"
    assert p1["club"] == "Richmond"
    assert p1["round"] == 0  # Priority

    p5 = df[df["pick"] == 5].iloc[0]
    assert p5["player_name"] == "Lance Franklin"
    assert p5["club"] == "Hawthorn"
    assert p5["recruited_from"] == "Perth Football Club"
    assert p5["round"] == 1
