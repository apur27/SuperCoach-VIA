"""
Unit tests for the AFL rookie draft scraper in ``scrapers/rookie_draft_scraper.py``.

Source is Wikipedia (``{year}_AFL_draft``). Tests cover:
  - Detection of end-of-season rookie draft tables by heading
  - Detection of mid-season rookie draft tables by heading
  - "Passed" player rows are skipped
  - Footnote stripping
  - dedup on re-run (same year+type+pick not duplicated)
  - No network calls: _fetch_soup is patched in all tests

Written BEFORE the implementation (TDD).
"""

import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest
from bs4 import BeautifulSoup

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scrapers.rookie_draft_scraper import (
    RookieDraftScraper,
    find_rookie_draft_tables,
    parse_rookie_table,
    clean_player_name,
)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _wikitable(headers, rows):
    thead = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    return '<table class="wikitable">' + f"<tr>{thead}</tr>" + body + "</table>"


def _section(heading, table_html, level="h2"):
    return f"<{level}>{heading}</{level}>" + table_html


def _page(*sections):
    return "<html><body>" + "".join(sections) + "</body></html>"


def _soup(html):
    return BeautifulSoup(html, "html.parser")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_END_SEASON_TABLE = _wikitable(
    ["Pick", "Player", "Club", "Recruited from"],
    [
        ["1", "Geordie Payne", "North Melbourne", "VFL"],
        ["2", "Jacob Blight", "Richmond", "SANFL"],
        ["4", "Max Hall[1]", "St Kilda", "Dandenong"],
        ["11", "Luker Kentfield", "Melbourne", "Coburg"],
    ],
)

_MID_SEASON_TABLE = _wikitable(
    ["Pick", "Player", "Club", "Recruited from"],
    [
        ["1", "Jai Culley", "West Coast", "Dandenong Stingrays"],
        ["3", "Massimo D'Ambrosio", "Essendon", "Richmond"],
        ["5", "Wade Derksen", "Greater Western Sydney", "Peel Thunder"],
        ["11", "Hugo Hall-Kahan", "Sydney", "Sandringham Dragons"],
        ["14", "Passed", "St Kilda", "—"],
    ],
)

# A page that has three sections: national (skipped), mid-season rookie, end-of-season rookie
_MULTI_SECTION_PAGE = _page(
    _section("2022 national draft", _wikitable(
        ["Round", "Pick", "Player", "Club"],
        [["1", "1", "Jason Horne-Francis", "North Melbourne"]],
    )),
    _section("2022 mid-season rookie draft", _MID_SEASON_TABLE),
    _section("2023 rookie draft", _END_SEASON_TABLE),
)

_END_ONLY_PAGE = _page(
    _section("2024 rookie draft", _END_SEASON_TABLE),
)

_MID_ONLY_PAGE = _page(
    _section("2022 mid-season rookie draft", _MID_SEASON_TABLE),
)

_NO_ROOKIE_PAGE = _page(
    _section("2005 national draft", _wikitable(
        ["Round", "Pick", "Player", "Club"],
        [["1", "1", "Buddy Franklin", "Hawthorn"]],
    )),
)


# ---------------------------------------------------------------------------
# Test 1: find_rookie_draft_tables returns tables under "rookie" headings
# ---------------------------------------------------------------------------

def test_finds_end_season_table():
    results = find_rookie_draft_tables(_soup(_END_ONLY_PAGE))
    assert len(results) == 1
    table, draft_type = results[0]
    assert draft_type == "end_season"
    assert "Geordie Payne" in table.get_text()


def test_finds_mid_season_table():
    results = find_rookie_draft_tables(_soup(_MID_ONLY_PAGE))
    assert len(results) == 1
    table, draft_type = results[0]
    assert draft_type == "mid_season"
    assert "Wade Derksen" in table.get_text()


def test_finds_both_on_multi_section_page():
    results = find_rookie_draft_tables(_soup(_MULTI_SECTION_PAGE))
    types = {dt for _, dt in results}
    assert "mid_season" in types
    assert "end_season" in types
    # National draft table must NOT be included
    all_text = " ".join(t.get_text() for t, _ in results)
    assert "Jason Horne-Francis" not in all_text


def test_returns_empty_when_no_rookie_section():
    results = find_rookie_draft_tables(_soup(_NO_ROOKIE_PAGE))
    assert results == []


# ---------------------------------------------------------------------------
# Test 2: parse_rookie_table basic structure
# ---------------------------------------------------------------------------

def test_parse_end_season_rows():
    table = _soup(_END_SEASON_TABLE).find("table")
    rows = parse_rookie_table(table, year=2024, draft_type="end_season")
    assert len(rows) == 4
    picks = {r["pick"] for r in rows}
    assert picks == {1, 2, 4, 11}
    row1 = next(r for r in rows if r["pick"] == 1)
    assert row1["player_name"] == "Geordie Payne"
    assert row1["club"] == "North Melbourne"
    assert row1["year"] == 2024
    assert row1["draft_type"] == "end_season"


def test_parse_mid_season_rows():
    table = _soup(_MID_SEASON_TABLE).find("table")
    rows = parse_rookie_table(table, year=2022, draft_type="mid_season")
    assert len(rows) == 4  # "Passed" row excluded
    clubs = {r["club"] for r in rows}
    assert "St Kilda" not in clubs  # passed pick excluded


# ---------------------------------------------------------------------------
# Test 3: "Passed" player entries are skipped
# ---------------------------------------------------------------------------

def test_passed_rows_excluded():
    table_html = _wikitable(
        ["Pick", "Player", "Club", "Recruited from"],
        [
            ["1", "Real Player", "Carlton", "VFL"],
            ["2", "Passed", "Geelong", "—"],
            ["3", "Another Real", "Sydney", "WAFL"],
        ],
    )
    table = _soup(table_html).find("table")
    rows = parse_rookie_table(table, year=2023, draft_type="end_season")
    assert len(rows) == 2
    player_names = {r["player_name"] for r in rows}
    assert "Passed" not in player_names


# ---------------------------------------------------------------------------
# Test 4: footnote markers stripped from player names
# ---------------------------------------------------------------------------

def test_footnote_stripped_in_parse():
    rows = parse_rookie_table(
        _soup(_END_SEASON_TABLE).find("table"), year=2024, draft_type="end_season"
    )
    hall = next(r for r in rows if r["pick"] == 4)
    assert hall["player_name"] == "Max Hall"
    assert "[1]" not in hall["player_name"]


def test_clean_player_name_standalone():
    assert clean_player_name("Jack Ginnivan[1]") == "Jack Ginnivan"
    assert clean_player_name("  Oisin  Mullin  ") == "Oisin Mullin"


# ---------------------------------------------------------------------------
# Test 5: non-numeric pick rows (sub-headers) are skipped
# ---------------------------------------------------------------------------

def test_non_numeric_picks_skipped():
    table_html = _wikitable(
        ["Pick", "Player", "Club", "Recruited from"],
        [
            ["1", "Valid Player", "Adelaide", "SANFL"],
            ["Pick", "Player", "Club", ""],   # sub-header row repeated mid-table
            ["3", "Third Player", "Essendon", "VFLW"],
        ],
    )
    table = _soup(table_html).find("table")
    rows = parse_rookie_table(table, year=2021, draft_type="end_season")
    assert sorted(r["pick"] for r in rows) == [1, 3]


# ---------------------------------------------------------------------------
# Test 6: dedup on re-run — same (year, draft_type, pick) not duplicated
# ---------------------------------------------------------------------------

def test_dedup_on_rerun(tmp_path):
    out = tmp_path / "afl_rookie_draft_history.csv"
    scraper = RookieDraftScraper(out_path=str(out))

    with patch.object(scraper, "_fetch_soup", return_value=_soup(_END_ONLY_PAGE)):
        scraper.scrape_year(2024)
        first_count = len(pd.read_csv(out))
        scraper.scrape_year(2024)
        second_count = len(pd.read_csv(out))

    assert first_count == second_count
    df = pd.read_csv(out)
    assert not df.duplicated(subset=["year", "draft_type", "pick"]).any()


# ---------------------------------------------------------------------------
# Test 7: page with no rookie tables scrapes zero rows
# ---------------------------------------------------------------------------

def test_no_rookie_tables_scrapes_zero(tmp_path):
    out = tmp_path / "afl_rookie_draft_history.csv"
    scraper = RookieDraftScraper(out_path=str(out))
    with patch.object(scraper, "_fetch_soup", return_value=_soup(_NO_ROOKIE_PAGE)):
        result = scraper.scrape_year(2005)
    assert result == 0
    assert not out.exists()


# ---------------------------------------------------------------------------
# Test 8: scrape_year returns total rows scraped across both draft types
# ---------------------------------------------------------------------------

def test_scrape_year_both_types(tmp_path):
    out = tmp_path / "afl_rookie_draft_history.csv"
    scraper = RookieDraftScraper(out_path=str(out))
    with patch.object(scraper, "_fetch_soup", return_value=_soup(_MULTI_SECTION_PAGE)):
        result = scraper.scrape_year(2022)
    # end_season: 4 picks, mid_season: 4 valid picks (1 "Passed" excluded)
    assert result == 8
    df = pd.read_csv(out)
    assert set(df["draft_type"].unique()) == {"end_season", "mid_season"}
