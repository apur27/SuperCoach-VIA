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
