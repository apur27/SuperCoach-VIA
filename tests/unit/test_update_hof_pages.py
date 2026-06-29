"""TDD tests for scripts/update_hof_pages.py — written before implementation."""
import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

# Load the module under test
_MOD_PATH = Path(__file__).parent.parent.parent / "scripts" / "update_hof_pages.py"


def _load():
    spec = importlib.util.spec_from_file_location("update_hof_pages", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


uohp = None  # populated in fixture below


@pytest.fixture(autouse=True)
def load_module():
    global uohp
    uohp = _load()


TODAY = date.today().isoformat()

_LEADER_GAMES = {
    "rank": 1, "rank_label": "1", "tied": False,
    "name": "Scott Pendlebury", "teams": "Collingwood",
    "year_min": 2006, "year_max": 2026,
    "games": 434, "total": 434.0, "per_game": 1.0,
}

_LEADER_DISPOSALS = {
    "rank": 1, "rank_label": "1", "tied": False,
    "name": "Scott Pendlebury", "teams": "Collingwood",
    "year_min": 2006, "year_max": 2026,
    "games": 434, "total": 11028.0, "per_game": 25.38,
}


def _hub_content(row_suffix="435"):
    return (
        "<!-- council-pipeline:\n"
        "  DataSentinel: PASS @ 2026-01-01 (old stamp)\n"
        "-->\n"
        f"*Last refreshed: 2026-01-01. Data from 100 files.*\n"
        f"| Career games | [Games leaderboard](hall-of-fame-stat-games.md) | Scott Pendlebury {row_suffix} **[data]** |<!-- HOF-HUB:career_games -->\n"
    )


def _sub_content(games="435"):
    return (
        "<!-- council-pipeline:\n"
        "  DataSentinel: PASS @ 2026-01-01 (old stamp)\n"
        "-->\n"
        f"*Last refreshed: 2026-01-01. Data layer: Scientist.*\n"
        f"| 1 | Scott Pendlebury **[data]** | Collingwood | 2006-2026 | {games} | 11,028 | 25.35 |<!-- HOF-TOP:career_disposals -->\n"
    )


# ── test 1 ────────────────────────────────────────────────────────────────────

def test_hub_row_updated_from_json(tmp_path):
    hub = tmp_path / "hub.md"
    hub.write_text(_hub_content("435"))

    cat = uohp.CATEGORIES["career_games"]
    replacements = {
        "<!-- HOF-HUB:career_games -->": uohp.build_hub_row("career_games", cat, _LEADER_GAMES)
    }
    changed = uohp.update_file(hub, replacements, TODAY)

    assert changed
    content = hub.read_text()
    assert "Scott Pendlebury 434 **[data]**" in content
    assert "<!-- HOF-HUB:career_games -->" in content


# ── test 2 ────────────────────────────────────────────────────────────────────

def test_subpage_top_row_updated(tmp_path):
    sub = tmp_path / "stat-disposals.md"
    sub.write_text(_sub_content("435"))

    cat = uohp.CATEGORIES["career_disposals"]
    replacements = {
        "<!-- HOF-TOP:career_disposals -->": uohp.build_subpage_row("career_disposals", cat, _LEADER_DISPOSALS)
    }
    changed = uohp.update_file(sub, replacements, TODAY)

    assert changed
    content = sub.read_text()
    assert "Scott Pendlebury **[data]**" in content
    assert "11,028" in content
    assert "<!-- HOF-TOP:career_disposals -->" in content


# ── test 3 ────────────────────────────────────────────────────────────────────

def test_no_change_when_values_match(tmp_path):
    cat = uohp.CATEGORIES["career_games"]
    exact_row = uohp.build_hub_row("career_games", cat, _LEADER_GAMES)

    hub = tmp_path / "hub.md"
    hub.write_text(
        "<!-- council-pipeline:\n"
        f"  DataSentinel: PASS @ {TODAY} (auto-updated from _stat_leaders.json by update_hof_pages.py)\n"
        "-->\n"
        f"*Last refreshed: {TODAY}. Data from 100 files.*\n"
        f"{exact_row}\n"
    )
    mtime_before = hub.stat().st_mtime

    changed = uohp.update_file(hub, {"<!-- HOF-HUB:career_games -->": exact_row}, TODAY)

    assert not changed
    assert hub.stat().st_mtime == mtime_before


# ── test 4 ────────────────────────────────────────────────────────────────────

def test_stamp_date_updated(tmp_path):
    hub = tmp_path / "hub.md"
    hub.write_text(_hub_content("434"))  # row already correct, stamp old

    cat = uohp.CATEGORIES["career_games"]
    exact_row = uohp.build_hub_row("career_games", cat, _LEADER_GAMES)
    uohp.update_file(hub, {"<!-- HOF-HUB:career_games -->": exact_row}, TODAY)

    content = hub.read_text()
    assert f"DataSentinel: PASS @ {TODAY}" in content
    assert "auto-updated from _stat_leaders.json by update_hof_pages.py" in content


# ── test 5 ────────────────────────────────────────────────────────────────────

def test_missing_sentinel_warns_and_continues(tmp_path, capsys):
    hub = tmp_path / "hub.md"
    hub.write_text("No sentinel here.\n")

    # Missing sentinel → update_file finds nothing to replace → no crash, no change
    changed = uohp.update_file(hub, {"<!-- HOF-HUB:career_games -->": "new line"}, TODAY)
    assert not changed

    # run_updates() with a bad JSON path should print a warning and exit non-zero
    # (tests that the runner is fault-tolerant)
    result = uohp.run_updates(
        json_path=tmp_path / "nonexistent.json",
        repo_root=tmp_path,
        today=TODAY,
    )
    assert result != 0
    captured = capsys.readouterr()
    assert "ERROR" in captured.out or "not found" in captured.out.lower() or result != 0


# ── test 6 ────────────────────────────────────────────────────────────────────

def test_number_formatting_int_vs_thousands():
    assert uohp.fmt_value(434.0, "int") == "434"
    assert uohp.fmt_value(11028.0, "thousands") == "11,028"
    assert uohp.fmt_value(1360.0, "thousands") == "1,360"
    assert uohp.fmt_value(262.0, "int") == "262"
    assert uohp.fmt_value(10597.0, "thousands") == "10,597"


# ── Full-table tests (Task A) ─────────────────────────────────────────────────

_LEADERS_20 = [
    {
        "rank": i, "rank_label": str(i), "tied": False,
        "name": f"Player {i}", "teams": f"Club {i}",
        "year_min": 2000, "year_max": 2020,
        "games": 400 - i * 5, "total": float(400 - i * 10), "per_game": 1.23,
    }
    for i in range(1, 21)
]


def _page_with_table_sentinels(key="career_games", body=None):
    if body is None:
        body = (
            "| 1 | OldPlayer **[data]** | OldClub | 2000-2020 | 400 |\n"
            "| 2 | OldPlayer2 **[data]** | OldClub | 2000-2020 | 390 |\n"
        )
    return (
        "| # | Player | Club(s) | Span | Games |\n"
        "|--:|--------|---------|------|------:|\n"
        f"<!-- HOF-TABLE-START:{key} -->\n"
        + body
        + f"<!-- HOF-TABLE-END:{key} -->\n"
        "Some prose below.\n"
    )


# ── test 7 ────────────────────────────────────────────────────────────────────

def test_full_table_body_regenerated_from_json(tmp_path):
    sub = tmp_path / "stat-games.md"
    sub.write_text(_page_with_table_sentinels("career_games"))

    cat = uohp.CATEGORIES["career_games"]
    rows = uohp.build_full_table_body("career_games", cat, _LEADERS_20)
    text, changed = uohp.replace_table_body(sub.read_text(), "career_games", rows)

    assert changed
    assert len(rows) == 20
    assert "Player 1 **[data]**" in text
    assert "Player 20 **[data]**" in text
    assert "OldPlayer" not in text
    assert "<!-- HOF-TABLE-START:career_games -->" in text
    assert "<!-- HOF-TABLE-END:career_games -->" in text


# ── test 8 ────────────────────────────────────────────────────────────────────

def test_full_table_idempotency(tmp_path):
    sub = tmp_path / "stat-games.md"
    sub.write_text(_page_with_table_sentinels("career_games"))

    cat = uohp.CATEGORIES["career_games"]
    rows = uohp.build_full_table_body("career_games", cat, _LEADERS_20)
    text1 = sub.read_text()
    text2, _ = uohp.replace_table_body(text1, "career_games", rows)
    text3, changed_second = uohp.replace_table_body(text2, "career_games", rows)

    assert not changed_second
    assert text2 == text3


# ── test 9 ────────────────────────────────────────────────────────────────────

def test_full_table_missing_sentinel_skips_gracefully():
    text = "| # | Player | Span | Games |\n|--:|---------|------|------:|\n| 1 | A **[data]** | 2000-2020 | 400 |\n"
    cat = uohp.CATEGORIES["career_games"]
    rows = uohp.build_full_table_body("career_games", cat, _LEADERS_20)
    new_text, changed = uohp.replace_table_body(text, "career_games", rows)

    assert not changed
    assert new_text == text


# ── test 10 ───────────────────────────────────────────────────────────────────

def test_full_table_data_tags_on_all_rows():
    cat = uohp.CATEGORIES["career_games"]
    rows = uohp.build_full_table_body("career_games", cat, _LEADERS_20)

    assert len(rows) == 20
    for row in rows:
        assert "**[data]**" in row
