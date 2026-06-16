"""
Unit tests for the DraftGuru enrichment scraper in
``scrapers/draftguru_scraper.py``.

Source is DraftGuru (``https://www.draftguru.com.au/years/{year}``). All HTML is
mocked inline -- no test makes a real network request. The network boundary is
``DraftGuruScraper._fetch_soup``, which is patched to return a BeautifulSoup
built from a string fixture.

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

from scrapers.draftguru_scraper import (  # noqa: E402
    DraftGuruScraper,
    parse_year_table,
    clean_player_name,
)


# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------
#
# DraftGuru's year table columns (per the brief + confirmed against the live
# 2004 page, in order):
#   Pick, Draft, #, Club, Signing, Player, Age, Height, Weight,
#   Original Club, Grade, Games, Goals, Coaches, Brownlow, Awards
#
# IMPORTANT (confirmed live): the ``Pick`` cell is an order-label that is often
# blank or "Priority (Club)" and is NOT a usable number. The ``#`` cell is the
# real pick-within-draft-type number (Franklin = 5). The scraper reads ``#``.

_HEADERS = [
    "Pick", "Draft", "#", "Club", "Signing", "Player", "Age", "Height",
    "Weight", "Original Club", "Grade", "Games", "Goals", "Coaches",
    "Brownlow", "Awards",
]


def _row_cells(pick, draft, num, club, signing, player, age, height, weight,
               original_club, grade, games, goals, coaches="0", brownlow="0",
               awards=""):
    """Assemble one DraftGuru table row's 16 cell strings, in column order."""
    return [
        pick, draft, num, club, signing, player, age, height, weight,
        original_club, grade, games, goals, coaches, brownlow, awards,
    ]


def _table(headers, rows):
    """Build a single <table> HTML string with thead + tbody."""
    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    return f"<table>{thead}<tbody>{body}</tbody></table>"


def _page(rows):
    """A minimal DraftGuru year page: one table inside a body."""
    return "<html><body>" + _table(_HEADERS, rows) + "</body></html>"


# Lance Franklin (#5, 2004) and Jordan Lewis (#7) -- both National. The Pick
# order-label cell is left BLANK (live pages do this for non-priority picks);
# the real pick number lives in the ``#`` column. This proves the parser keys
# off ``#``, not ``Pick``.
_FRANKLIN = _row_cells(
    "", "National", "5", "Hawthorn", "Drafted", "Lance Franklin",
    "17", "195cm", "92kg", "Dowerin / Wesley College (WA) / Perth",
    "A+", "354", "1066",
)
_LEWIS = _row_cells(
    "", "National", "7", "Hawthorn", "Drafted", "Jordan Lewis",
    "18", "188cm", "85kg",
    "Warrnambool FC / Emmanuel College (Warrnambool) / Geelong U18",
    "A", "264", "126",
)
# A Rookie-draft row that must be filtered out.
_ROOKIE = _row_cells(
    "", "Rookie", "1", "Geelong", "Rookie", "Some Rookie",
    "18", "180cm", "78kg", "Geelong Falcons", "C", "0", "0",
)
# A Preseason-draft row that must be filtered out.
_PRESEASON = _row_cells(
    "", "Pre-Season", "1", "Carlton", "Drafted", "Some Preseason",
    "22", "185cm", "82kg", "Northern Bullants", "B", "12", "3",
)


def _soup(html):
    return BeautifulSoup(html, "html.parser")


# ---------------------------------------------------------------------------
# Test 1: happy path -- parsed row dicts have the expected fields/values
# ---------------------------------------------------------------------------

def test_scrape_year_happy_path():
    table = parse_year_table(_soup(_page([_FRANKLIN, _LEWIS])), year=2004)
    assert len(table) == 2
    franklin = next(r for r in table if r["pick"] == 5)
    assert franklin["year"] == 2004
    assert franklin["player_name"] == "Lance Franklin"
    assert franklin["original_club"] == "Dowerin / Wesley College (WA) / Perth"
    assert franklin["grade"] == "A+"
    assert franklin["games"] == 354
    assert franklin["goals"] == 1066
    # Only the brief's output columns are present.
    assert set(franklin) == {
        "year", "pick", "player_name", "original_club", "grade", "games", "goals",
    }


# ---------------------------------------------------------------------------
# Test 2: only National draft rows are returned
# ---------------------------------------------------------------------------

def test_national_draft_filter():
    rows = parse_year_table(
        _soup(_page([_FRANKLIN, _ROOKIE, _PRESEASON, _LEWIS])), year=2004
    )
    names = {r["player_name"] for r in rows}
    assert names == {"Lance Franklin", "Jordan Lewis"}
    assert "Some Rookie" not in names
    assert "Some Preseason" not in names


# ---------------------------------------------------------------------------
# Test 3: Grade is extracted
# ---------------------------------------------------------------------------

def test_grade_parsed():
    rows = parse_year_table(_soup(_page([_FRANKLIN, _LEWIS])), year=2004)
    grades = {r["player_name"]: r["grade"] for r in rows}
    assert grades["Lance Franklin"] == "A+"
    assert grades["Jordan Lewis"] == "A"


# ---------------------------------------------------------------------------
# Test 4: Original Club field extracted verbatim
# ---------------------------------------------------------------------------

def test_original_club_parsed():
    rows = parse_year_table(_soup(_page([_LEWIS])), year=2004)
    assert (
        rows[0]["original_club"]
        == "Warrnambool FC / Emmanuel College (Warrnambool) / Geelong U18"
    )


# ---------------------------------------------------------------------------
# Test 5: empty / dash Grade returns "" rather than crashing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grade_cell", ["", "-", "–", "  "])
def test_missing_grade_returns_empty_string(grade_cell):
    row = _row_cells(
        "10", "National", "10", "Essendon", "Drafted", "Gradeless Guy",
        "18", "186cm", "80kg", "Calder Cannons", grade_cell, "50", "20",
    )
    rows = parse_year_table(_soup(_page([row])), year=2010)
    assert rows[0]["grade"] == ""


# ---------------------------------------------------------------------------
# Test 6: Games / Goals coerced to ints; missing -> 0
# ---------------------------------------------------------------------------

def test_numeric_fields_coerced():
    row = _row_cells(
        "", "National", "12", "Fremantle", "Drafted", "No Games Yet",
        "18", "190cm", "84kg", "Perth", "B", "", "-",
    )
    rows = parse_year_table(_soup(_page([row])), year=2024)
    assert rows[0]["games"] == 0
    assert rows[0]["goals"] == 0
    assert isinstance(rows[0]["games"], int)
    assert isinstance(rows[0]["goals"], int)


def test_games_with_parenthetical_finals():
    """
    Live Games cells read "102 (83)" (total games, finals in brackets). Only the
    leading total must be taken -- gluing the two numbers (e.g. into 10283) is a
    silent data-corruption bug.
    """
    row = _row_cells(
        "", "National", "5", "Hawthorn", "Drafted", "Lance Franklin",
        "17", "195cm", "92kg", "Perth", "A+", "354 (28)", "1066",
    )
    rows = parse_year_table(_soup(_page([row])), year=2004)
    assert rows[0]["games"] == 354
    assert rows[0]["goals"] == 1066


# ---------------------------------------------------------------------------
# Test 7: rate-limit sleep is called between years in scrape_all
# ---------------------------------------------------------------------------

def test_rate_limit_between_years(tmp_path):
    out = tmp_path / "draftguru_enrichment.csv"
    scraper = DraftGuruScraper(out_path=str(out))

    with patch.object(scraper, "_fetch_soup", return_value=_soup(_page([_FRANKLIN]))), \
            patch("scrapers.draftguru_scraper.time.sleep") as mock_sleep:
        scraper.scrape_all(start=2004, end=2006)

    # Three years scraped -> sleep called at least once per fetch boundary.
    assert mock_sleep.call_count >= 3
    assert all(c.args[0] == 0.5 for c in mock_sleep.call_args_list)


# ---------------------------------------------------------------------------
# Test 8: table missing -> empty list, no crash
# ---------------------------------------------------------------------------

def test_empty_page_returns_empty_list():
    assert parse_year_table(_soup("<html><body><p>No table</p></body></html>"), 2099) == []


# ---------------------------------------------------------------------------
# Test 9: player_name normalised (footnotes stripped, whitespace collapsed)
# ---------------------------------------------------------------------------

def test_player_key_format():
    assert clean_player_name("Lance Franklin[1]") == "Lance Franklin"
    assert clean_player_name("  Jordan   Lewis  ") == "Jordan Lewis"
    row = _row_cells(
        "", "National", "5", "Hawthorn", "Drafted", "  Lance Franklin[1] ",
        "17", "195cm", "92kg", "Perth", "A+", "354", "1066",
    )
    rows = parse_year_table(_soup(_page([row])), year=2004)
    assert rows[0]["player_name"] == "Lance Franklin"


# ---------------------------------------------------------------------------
# Test: real-page unicode (nbsp in names, zero-width in Original Club) cleaned;
# pick read from the "#" column, not the "Pick" order-label
# ---------------------------------------------------------------------------

def test_live_unicode_and_pick_from_hash_column():
    # Mirrors the live 2004 Deledio row: Pick = "Priority (Richmond)", # = "1",
    # player has a non-breaking space, Original Club has a zero-width space.
    row = _row_cells(
        "Priority (Richmond)", "National", "1", "Richmond", "Drafted",
        "Brett\xa0Deledio", "17", "188cm", "85kg",
        "Kyabram /​ Murray U18", "A", "243", "32",
    )
    rows = parse_year_table(_soup(_page([row])), year=2004)
    assert len(rows) == 1
    r = rows[0]
    assert r["pick"] == 1  # from the "#" column, not "Priority (Richmond)"
    assert r["player_name"] == "Brett Deledio"  # nbsp collapsed to a space
    assert "​" not in r["original_club"]
    assert r["original_club"] == "Kyabram / Murray U18"


# ---------------------------------------------------------------------------
# Test: end-to-end scrape_year writes the CSV with the right columns
# ---------------------------------------------------------------------------

def test_scrape_year_writes_csv(tmp_path):
    out = tmp_path / "draftguru_enrichment.csv"
    scraper = DraftGuruScraper(out_path=str(out))

    with patch.object(scraper, "_fetch_soup", return_value=_soup(_page([_FRANKLIN, _LEWIS]))):
        n = scraper.scrape_year(2004)

    assert n == 2
    df = pd.read_csv(out)
    assert list(df.columns) == [
        "year", "pick", "player_name", "original_club", "grade", "games", "goals",
    ]
    franklin = df[df["pick"] == 5].iloc[0]
    assert franklin["player_name"] == "Lance Franklin"
    assert franklin["games"] == 354


# ---------------------------------------------------------------------------
# Test: re-running a year is idempotent (dedup on (year, pick))
# ---------------------------------------------------------------------------

def test_dedup_on_rerun(tmp_path):
    out = tmp_path / "draftguru_enrichment.csv"
    scraper = DraftGuruScraper(out_path=str(out))

    with patch.object(scraper, "_fetch_soup", return_value=_soup(_page([_FRANKLIN, _LEWIS]))):
        scraper.scrape_year(2004)
        first = pd.read_csv(out)
        scraper.scrape_year(2004)
        second = pd.read_csv(out)

    assert len(first) == len(second)
    assert not second.duplicated(subset=["year", "pick"]).any()
