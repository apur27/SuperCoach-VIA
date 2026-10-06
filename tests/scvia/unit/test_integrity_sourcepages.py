"""Independent source reader: hand-annotated fixtures and cross-parser agreement on real pages."""

from __future__ import annotations

import gzip
from datetime import date
from pathlib import Path

import pytest

from supercoach_via.ingest import afltables as adapter
from supercoach_via.integrity import sourcepages as sp

REPO = Path(__file__).resolve().parents[3]
B1_RAW = REPO / "docs/rewrite/evidence/b1/raw"
R4_PAGE = B1_RAW / "08a2bf5968e7ebce40e1cc4f5d2440d41498e1aba2e965a10ac703866e1b83a8.html.gz"
R5_PAGE = B1_RAW / "e1675c9dc8b2bb6cbcf1c77a9036a5c12c53e23eee10c3c4842bff7b665dd45c.html.gz"
SEASON_PAGE = B1_RAW / "6c0d5a3bd5476ce2af7f7f558eda05ebe05679bb282fed6c434c1722f2ac7fc0.html.gz"

# Columns deliberately out of the site's usual order, a blank cell, a substitute arrow and a
# footer with rushed behinds. Expected values below were written by hand from this markup.
PERMUTED_PAGE = """<html><body>
<table><tr><td rowspan=6><a href="../2026/1.html">&larr;</a></td><td colspan=5><b>Round: </b>7 <b>Venue: </b>
<a href="../../../venues/x.html">Demo Oval</a> <b>Date: </b>Sat, 02-May-2026 7:25 PM (6:55 PM) <b>Attendance:</b> 12,345</td>
<td rowspan=6><a href="../2026/3.html">&rarr;</a></td></tr>
<tr><td><a href="../../../teams/alpha_idx.html">Alpha</a></td><td>1.2.<b>8</b></td><td>2.3.<b>15</b></td>
<td>3.4.<b>22</b></td><td>4.5.<b>29</b></td></tr>
<tr><td><a href="../../../teams/beta_idx.html">Beta</a></td><td>0.1.<b>1</b></td><td>1.1.<b>7</b></td>
<td>2.2.<b>14</b></td><td>3.3.<b>21</b></td></tr></table>
<table class="sortable"><thead><tr><th colspan=9>Alpha Match Statistics [Season]</th></tr>
<tr><th>#</th><th>Player</th><th>DI</th><th>HB</th><th>KI</th><th>%P</th><th>BH</th><th>GL</th><th>BR</th></tr></thead>
<tbody>
<tr><td>7 &uarr;</td><td><a href="../../players/A/Ann_Able.html">Able, Ann</a></td><td>12</td><td>5</td><td>7</td><td>81</td><td>1</td><td>2</td><td>&nbsp;</td></tr>
<tr><td>9</td><td><a href="../../players/B/Bob_Baker.html">Baker, Bob</a></td><td>3</td><td>1</td><td>2</td><td>100</td><td>&nbsp;</td><td>2</td><td>3</td></tr>
</tbody><tfoot><tr><td colspan=6>Rushed</td><td>4</td><td colspan=2>&nbsp;</td></tr>
<tr><td colspan=2><b>Totals</td><td>15</td><td>6</td><td>9</td><td>&nbsp;</td><td>5</td><td>4</td><td>3</td></tr></tfoot></table>
<table class="sortable"><thead><tr><th colspan=9>Beta Match Statistics [Season]</th></tr>
<tr><th>#</th><th>Player</th><th>DI</th><th>HB</th><th>KI</th><th>%P</th><th>BH</th><th>GL</th><th>BR</th></tr></thead>
<tbody>
<tr><td>1</td><td><a href="../../players/C/Cy_Cole.html">Cole, Cy</a></td><td>20</td><td>8</td><td>12</td><td>77</td><td>3</td><td>3</td><td>&nbsp;</td></tr>
</tbody></table>
</body></html>"""

ANNOTATED = {
    ("Alpha", "Able, Ann"): {
        "disposals": 12,
        "handballs": 5,
        "kicks": 7,
        "time_on_ground_pct": 81.0,
        "behinds": 1,
        "goals": 2,
        "brownlow_votes": None,
    },
    ("Alpha", "Baker, Bob"): {
        "disposals": 3,
        "handballs": 1,
        "kicks": 2,
        "time_on_ground_pct": 100.0,
        "behinds": None,
        "goals": 2,
        "brownlow_votes": 3,
    },
    ("Beta", "Cole, Cy"): {
        "disposals": 20,
        "handballs": 8,
        "kicks": 12,
        "time_on_ground_pct": 77.0,
        "behinds": 3,
        "goals": 3,
        "brownlow_votes": None,
    },
}


def _gz(path: Path) -> bytes:
    return gzip.decompress(path.read_bytes())


def test_permuted_columns_map_by_label_not_position() -> None:
    page = sp.read_match_page(PERMUTED_PAGE.encode())
    assert page.problems == []
    assert (page.stage_text, page.venue, page.attendance) == ("7", "Demo Oval", 12345)
    assert (page.match_date, page.local_start) == (date(2026, 5, 2), "2026-05-02 19:25")
    assert [(t.name, t.quarters[-1], t.points[-1]) for t in page.teams] == [("Alpha", (4, 5), 29), ("Beta", (3, 3), 21)]
    got = {(p.team, p.name): {k: sp.cell_value(k, v) for k, v in p.cells.items()} for p in page.players}
    assert got == ANNOTATED
    assert [sp.jersey(p.jersey_token) for p in page.players] == [7, 9, 1]
    assert page.rushed == {"Alpha": 4}
    assert page.totals["Alpha"]["behinds"] == "5"  # includes the rushed behinds
    assert page.totals["Alpha"]["disposals"] == "15"


def test_production_adapter_agrees_with_the_annotation() -> None:
    """The annotated fixture challenges the production adapter too: a positional mapping would fail."""
    detail = adapter.parse_match_detail(PERMUTED_PAGE.encode(), season=2026, game_id="000120260502")
    by_name = {(p.team, p.source_name): p.stats for p in detail.players}
    for key, want in ANNOTATED.items():
        for stat, value in want.items():
            assert by_name[key][stat] == value, (key, stat)


def test_unparseable_cell_is_kept_as_text_not_null() -> None:
    assert sp.cell_value("kicks", "8a") == "8a"
    assert sp.cell_value("kicks", " ") is None
    assert sp.cell_value("time_on_ground_pct", "95") == 95.0


def test_unknown_label_and_short_row_are_problems() -> None:
    bad = PERMUTED_PAGE.replace("<th>BR</th>", "<th>ZZ</th>", 1).replace("<td>12</td>", "", 1)
    page = sp.read_match_page(bad.encode())
    assert any("unknown statistic labels" in p for p in page.problems)
    assert any("row with 8 cells" in p for p in page.problems)


def test_missing_header_is_a_problem_not_an_empty_success() -> None:
    page = sp.read_match_page(b"<html><table><tr><td>nothing</td></tr></table></html>")
    assert page.problems == ["no match header table"]
    assert page.players == []


@pytest.mark.parametrize(("page_path", "game_id"), [(R4_PAGE, "131820260329"), (R5_PAGE, "091020260406")])
def test_real_match_page_agrees_with_adapter_cell_for_cell(page_path: Path, game_id: str) -> None:
    raw = _gz(page_path)
    mine = sp.read_match_page(raw)
    assert mine.problems == []
    theirs = adapter.parse_match_detail(raw, season=2026, game_id=game_id)
    assert theirs.outcome.value == "PASS"
    assert len(mine.players) == len(theirs.players) == 46
    for a, b in zip(mine.players, theirs.players, strict=True):
        assert (a.team, a.name) == (b.team, b.source_name)
        assert {k: sp.cell_value(k, v) for k, v in a.cells.items()} == b.stats
    assert [t.points[-1] for t in mine.teams] == [theirs.home_score, theirs.away_score]
    # footer evidence: team behinds = player behinds + rushed
    for team in mine.teams:
        players = [p for p in mine.players if p.team == team.name]
        bh = sum(int(p.cells["behinds"]) for p in players if p.cells["behinds"])
        # a footer without a Rushed row states no rushed behinds; the Totals row includes them
        assert bh + (mine.rushed.get(team.name) or 0) == team.quarters[-1][1] == int(mine.totals[team.name]["behinds"])


def test_real_season_page_agrees_with_adapter_on_completed_fixtures() -> None:
    raw = _gz(SEASON_PAGE)
    mine = sp.read_season_page(raw)
    assert mine.problems == []
    theirs = adapter.parse_season_page(raw, season=2026)
    done_mine = [f for f in mine.fixtures if f.home_points is not None]
    done_theirs = theirs.completed()
    assert len(done_mine) == len(done_theirs) > 200
    key_mine = sorted((f.match_date, f.home, f.away, f.home_points, f.away_points) for f in done_mine)
    key_theirs = sorted((m.match_date, m.home_name, m.away_name, m.home_score, m.away_score) for m in done_theirs)
    assert key_mine == key_theirs
    assert {sp.stage_label_for(f.stage_text) for f in done_mine} >= {"1", "Qualifying Final", "Semi Final"}


def test_person_name_and_stage_label() -> None:
    assert sp.person_name("Amiss, Jye") == "Jye Amiss"
    assert sp.person_name("Single") == "Single"
    assert sp.stage_label_for("Round 12") == "12"
    assert sp.stage_label_for("12") == "12"
    assert sp.stage_label_for("Grand Final") == "Grand Final"


def test_cell_records_rowspan_additively() -> None:
    from supercoach_via.integrity.sourcepages import Cell, read_tables

    # existing five-field construction still works; rowspan defaults to 1
    assert Cell(tag="td", text="x", links=[], colspan=1, bold=False).rowspan == 1
    html = b"<table><tr><td rowspan=3>a</td><td colspan=2>b</td><td rowspan=bad>c</td><td>d</td></tr></table>"
    cells = read_tables(html)[0].rows[0]
    assert [c.rowspan for c in cells] == [3, 1, 1, 1]
    assert [c.colspan for c in cells] == [1, 2, 1, 1]
