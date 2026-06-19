"""
Unit tests for ``scrapers/free_agency_scraper.py`` (AFL 2026 free-agency / off-contract
list scraper).

Written BEFORE the implementation (TDD, per CLAUDE.md).

NO test makes a real network request. The network boundary is
``FreeAgencyScraper._fetch_html``, which is patched. Live fixtures are tiny inline
HTML strings in the SAME structure as the persisted fixtures the scraper falls back
to. Repo enrichment is tested against ``tmp_path`` CSVs -- never the real
``data/player_data`` files.
"""

import os
import sys
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

# Make ``scrapers`` importable regardless of pytest's invocation directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scrapers.free_agency_scraper import (  # noqa: E402
    FreeAgencyScraper,
    FetchBlocked,
    OUTPUT_COLUMNS,
    parse_afl_fa_list,
    parse_zerohanger,
    fa_status_note,
    merge_rows,
    compute_age,
    enrich_with_repo_stats,
)


# ---------------------------------------------------------------------------
# HTML fixtures (same structure as the persisted scrapers/fixtures/*.html)
# ---------------------------------------------------------------------------

def _afl_club(club, players):
    """players: list of (name, status)."""
    rows = "".join(
        f'<tr><td class="player">{n}</td><td class="status">{s}</td></tr>'
        for n, s in players
    )
    return (
        f'<div class="club-section" data-club="{club}">'
        f'<h3 class="club-name">{club}</h3>'
        f"<table>{rows}</table></div>"
    )


def _afl_page(clubs):
    body = "".join(_afl_club(c, p) for c, p in clubs)
    return f'<html><body><div class="free-agents-list">{body}</div></body></html>'


def _zh_club(club, names):
    lis = "".join(f"<li>{n}</li>" for n in names)
    return (
        f'<div class="club" data-club="{club}"><h3>{club}</h3>'
        f"<ul>{lis}</ul></div>"
    )


def _zh_page(clubs):
    body = "".join(_zh_club(c, n) for c, n in clubs)
    return f'<html><body><div class="off-contract">{body}</div></body></html>'


# ---------------------------------------------------------------------------
# AFL FA list parsing
# ---------------------------------------------------------------------------

def test_parse_afl_fa_list_happy():
    html = _afl_page([
        ("Adelaide", [("James Borlase", "Restricted"), ("Jordon Butts", "Unrestricted")]),
        ("Brisbane", [("Lachie Neale", "Unrestricted")]),
    ])
    rows = parse_afl_fa_list(html)
    assert len(rows) == 3
    borlase = next(r for r in rows if r["player_name"] == "James Borlase")
    assert borlase["club"] == "Adelaide"
    assert borlase["status"] == "Restricted"
    neale = next(r for r in rows if r["player_name"] == "Lachie Neale")
    assert neale["club"] == "Brisbane"
    assert neale["status"] == "Unrestricted"


def test_parse_afl_fa_list_empty():
    assert parse_afl_fa_list("") == []
    assert parse_afl_fa_list("<html><body></body></html>") == []


def test_parse_afl_fa_list_malformed_missing_status():
    # A player row with no status cell must not crash; name is still captured.
    html = (
        '<html><body><div class="free-agents-list">'
        '<div class="club-section" data-club="Carlton">'
        '<h3 class="club-name">Carlton</h3><table>'
        '<tr><td class="player">Nick Haynes</td></tr>'  # no status cell
        '</table></div></div></body></html>'
    )
    rows = parse_afl_fa_list(html)
    assert len(rows) == 1
    assert rows[0]["player_name"] == "Nick Haynes"
    assert rows[0]["club"] == "Carlton"
    assert rows[0]["status"] == ""  # gracefully empty


def test_parse_afl_fa_list_skips_blank_names():
    html = (
        '<html><body><div class="free-agents-list">'
        '<div class="club-section" data-club="Richmond">'
        '<h3 class="club-name">Richmond</h3><table>'
        '<tr><td class="player">   </td><td class="status">Unrestricted</td></tr>'
        '<tr><td class="player">Dion Prestia</td><td class="status">Unrestricted</td></tr>'
        '</table></div></div></body></html>'
    )
    rows = parse_afl_fa_list(html)
    assert [r["player_name"] for r in rows] == ["Dion Prestia"]


# ---------------------------------------------------------------------------
# ZeroHanger off-contract parsing
# ---------------------------------------------------------------------------

def test_parse_zerohanger_happy():
    html = _zh_page([
        ("Port Adelaide", ["Aliir Aliir", "Ivan Soldo"]),
        ("Carlton", ["Adam Saad"]),
    ])
    rows = parse_zerohanger(html)
    assert len(rows) == 3
    aliir = next(r for r in rows if r["player_name"] == "Aliir Aliir")
    assert aliir["club"] == "Port Adelaide"


def test_parse_zerohanger_empty():
    assert parse_zerohanger("") == []


# ---------------------------------------------------------------------------
# Status -> notes mapping
# ---------------------------------------------------------------------------

def test_fa_status_note():
    assert fa_status_note("Restricted") == "Restricted FA (AFL.com.au)"
    assert fa_status_note("Unrestricted") == "Unrestricted FA (AFL.com.au)"
    # Unknown status still produces a defensible note (no crash).
    assert "AFL.com.au" in fa_status_note("")


# ---------------------------------------------------------------------------
# Merge / dedup: AFL FA status is authoritative over ZeroHanger
# ---------------------------------------------------------------------------

def test_merge_rows_afl_status_wins_over_zerohanger():
    afl = [{"player_name": "Zak Butters", "club": "Port Adelaide",
            "notes": "Restricted FA (AFL.com.au)"}]
    zh = [
        {"player_name": "Zak Butters", "club": "Port Adelaide",
         "notes": "Off-contract (ZeroHanger)"},   # duplicate of an FA -> dropped
        {"player_name": "Aliir Aliir", "club": "Port Adelaide",
         "notes": "Off-contract (ZeroHanger)"},    # not an FA -> kept
    ]
    merged = merge_rows(afl, zh)
    names = {r["player_name"] for r in merged}
    assert names == {"Zak Butters", "Aliir Aliir"}
    butters = next(r for r in merged if r["player_name"] == "Zak Butters")
    assert butters["notes"] == "Restricted FA (AFL.com.au)"  # AFL note retained


# ---------------------------------------------------------------------------
# Age computation
# ---------------------------------------------------------------------------

def test_compute_age_before_birthday():
    # Born 08-09-2000 (DD-MM-YYYY); as of 2026-06-19 birthday not yet reached -> 25.
    assert compute_age("08-09-2000", as_of=date(2026, 6, 19)) == 25


def test_compute_age_after_birthday():
    # Born 05-04-1990; as of 2026-06-19 birthday passed -> 36.
    assert compute_age("05-04-1990", as_of=date(2026, 6, 19)) == 36


def test_compute_age_bad_input_returns_none():
    assert compute_age("", as_of=date(2026, 6, 19)) is None
    assert compute_age(None, as_of=date(2026, 6, 19)) is None


# ---------------------------------------------------------------------------
# Repo enrichment (tmp_path only -- never real data files)
# ---------------------------------------------------------------------------

def _write_player_files(player_dir, key_born_debut, perf_rows):
    """key_born_debut: (file_key, born_DDMMYYYY). perf_rows: list of games_played ints."""
    file_key, born = key_born_debut
    born_fname = born
    personal = pd.DataFrame([{
        "first_name": "Test", "last_name": "Player",
        "born_date": f"{born[:2]}-{born[2:4]}-{born[4:]}",
        "debut_date": "01-01-2018", "height": 185, "weight": 85,
    }])
    perf = pd.DataFrame({"games_played": perf_rows, "goals": [1] * len(perf_rows)})
    personal.to_csv(
        os.path.join(player_dir, f"{file_key}_{born_fname}_personal_details.csv"),
        index=False)
    perf.to_csv(
        os.path.join(player_dir, f"{file_key}_{born_fname}_performance_details.csv"),
        index=False)


def test_enrich_fills_age_and_games(tmp_path):
    pdir = tmp_path / "player_data"
    pdir.mkdir()
    _write_player_files(str(pdir), ("butters_zak", "08092000"),
                        perf_rows=[1, 2, 3, 4, 5])
    rows = [{"player_name": "Zak Butters", "club": "Port Adelaide",
             "notes": "Restricted FA (AFL.com.au)"}]
    out = enrich_with_repo_stats(rows, str(pdir), as_of_date=date(2026, 6, 19))
    r = out[0]
    assert r["games"] == 5          # max games_played
    assert r["age"] == 25
    assert r["contract_end"] == 2026
    assert r["position"] == ""      # repo has no position field


def test_enrich_matches_nickname(tmp_path):
    # Ground-truth name "Mitchell McGovern"; repo file uses the nickname "mitch".
    pdir = tmp_path / "player_data"
    pdir.mkdir()
    _write_player_files(str(pdir), ("mcgovern_mitch", "11101994"),
                        perf_rows=list(range(1, 121)))
    rows = [{"player_name": "Mitchell McGovern", "club": "Carlton",
             "notes": "Unrestricted FA (AFL.com.au)"}]
    out = enrich_with_repo_stats(rows, str(pdir), as_of_date=date(2026, 6, 19))
    assert out[0]["games"] == 120
    assert "stats not in repo" not in out[0]["notes"]


def test_enrich_ambiguous_same_name_left_blank(tmp_path):
    # Two real players share surname+first-name; the matcher must NOT guess.
    pdir = tmp_path / "player_data"
    pdir.mkdir()
    _write_player_files(str(pdir), ("lynch_tom", "15091990"), perf_rows=[1, 2, 3])
    _write_player_files(str(pdir), ("lynch_tom", "31101992"), perf_rows=[1, 2])
    rows = [{"player_name": "Thomas Lynch", "club": "Richmond",
             "notes": "Unrestricted FA (AFL.com.au)"}]
    out = enrich_with_repo_stats(rows, str(pdir), as_of_date=date(2026, 6, 19))
    assert out[0]["games"] == ""        # refused to guess
    assert "ambiguous" in out[0]["notes"].lower()


def test_enrich_no_repo_file_leaves_blank_and_notes(tmp_path):
    pdir = tmp_path / "player_data"
    pdir.mkdir()
    rows = [{"player_name": "Nobody Here", "club": "Geelong",
             "notes": "Off-contract (ZeroHanger)"}]
    out = enrich_with_repo_stats(rows, str(pdir), as_of_date=date(2026, 6, 19))
    r = out[0]
    assert r["games"] == ""
    assert r["age"] == ""
    assert "stats not in repo" in r["notes"]
    assert r["notes"].startswith("Off-contract (ZeroHanger)")  # original note kept


# ---------------------------------------------------------------------------
# Fetch-or-fixture fallback
# ---------------------------------------------------------------------------

def test_fetch_or_fixture_uses_live_when_available(tmp_path):
    fixture = tmp_path / "fx.html"
    fixture.write_text("<html>FIXTURE</html>")
    scr = FreeAgencyScraper()
    with patch.object(scr, "_fetch_html", return_value="<html>LIVE</html>"):
        html, source = scr.fetch_or_fixture("http://x", str(fixture))
    assert source == "live"
    assert "LIVE" in html


def test_fetch_or_fixture_falls_back_on_block(tmp_path):
    fixture = tmp_path / "fx.html"
    fixture.write_text("<html>FIXTURE</html>")
    scr = FreeAgencyScraper()
    with patch.object(scr, "_fetch_html", side_effect=FetchBlocked("403")):
        html, source = scr.fetch_or_fixture("http://x", str(fixture))
    assert source == "fixture"
    assert "FIXTURE" in html


# ---------------------------------------------------------------------------
# End-to-end scrape (fixtures only) + CSV shape
# ---------------------------------------------------------------------------

def test_scrape_produces_normalized_rows(tmp_path):
    afl_fx = tmp_path / "afl.html"
    zh_fx = tmp_path / "zh.html"
    afl_fx.write_text(_afl_page([
        ("Port Adelaide", [("Zak Butters", "Restricted"),
                           ("Oliver Wines", "Unrestricted")]),
    ]))
    zh_fx.write_text(_zh_page([
        ("Port Adelaide", ["Zak Butters", "Aliir Aliir"]),  # Butters dup -> dropped
    ]))
    pdir = tmp_path / "player_data"
    pdir.mkdir()

    scr = FreeAgencyScraper(
        afl_fixture=str(afl_fx), zh_fixture=str(zh_fx), player_data_dir=str(pdir))
    # Force fixture path for both sources.
    with patch.object(scr, "_fetch_html", side_effect=FetchBlocked("403")):
        rows = scr.scrape(as_of_date=date(2026, 6, 19))

    names = {r["player_name"] for r in rows}
    assert names == {"Zak Butters", "Oliver Wines", "Aliir Aliir"}
    for r in rows:
        assert r["contract_end"] == 2026
        assert set(r.keys()) >= set(OUTPUT_COLUMNS)
    butters = next(r for r in rows if r["player_name"] == "Zak Butters")
    # AFL FA status wins over the ZeroHanger duplicate; empty tmp player dir means
    # the repo-stats suffix is appended.
    assert butters["notes"].startswith("Restricted FA (AFL.com.au)")


def test_scrape_falls_back_when_live_parses_empty(tmp_path):
    # A 200 response whose body is a JS/Cloudflare shell parses to 0 rows; the
    # scraper must treat that as a failed fetch and use the fixture content.
    afl_fx = tmp_path / "afl.html"
    zh_fx = tmp_path / "zh.html"
    afl_fx.write_text(_afl_page([("Geelong", [("Patrick Dangerfield", "Unrestricted")])]))
    zh_fx.write_text(_zh_page([("Geelong", [])]))
    pdir = tmp_path / "player_data"
    pdir.mkdir()

    scr = FreeAgencyScraper(
        afl_fixture=str(afl_fx), zh_fixture=str(zh_fx), player_data_dir=str(pdir))
    shell = "<html><body><div id='app'>loading...</div></body></html>"
    with patch.object(scr, "_fetch_html", return_value=shell):
        rows = scr.scrape(as_of_date=date(2026, 6, 19))

    assert {r["player_name"] for r in rows} == {"Patrick Dangerfield"}


def test_to_csv_writes_expected_columns(tmp_path):
    out = tmp_path / "afl_2026_contracts.csv"
    rows = [{
        "player_name": "Zak Butters", "club": "Port Adelaide", "position": "",
        "age": 25, "contract_end": 2026, "games": 100,
        "notes": "Restricted FA (AFL.com.au)",
    }]
    scr = FreeAgencyScraper()
    scr.to_csv(rows, str(out))
    df = pd.read_csv(out)
    assert list(df.columns) == OUTPUT_COLUMNS
    assert df.iloc[0]["player_name"] == "Zak Butters"
    assert int(df.iloc[0]["contract_end"]) == 2026
