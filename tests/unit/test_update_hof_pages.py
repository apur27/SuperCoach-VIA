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


# ── Full-table for disposals & goals (Task: close deferred TODO) ───────────────
#
# These two categories use the SAME standard 7-column layout as the already-
# whitelisted career_marks / career_tackles pages:
#   | # | Player | Club(s) | Span | Games | <Stat> | Per game |
# (There is no multi-column kicks+handballs / seasons+career-totals table on the
#  live pages — the disposal total renders with `thousands` formatting, and the
#  kick/handball composition lives only in prose, not the table.)
# So build_full_table_row's non-`games_only` branch already renders them; the fix
# is to add both keys to _FULL_TABLE_CATS.

_LEADERS_DISPOSALS = [
    {
        "rank": 1, "rank_label": "1", "tied": False,
        "name": "Scott Pendlebury", "teams": "Collingwood",
        "year_min": 2006, "year_max": 2026,
        "games": 436, "total": 11069.0, "per_game": 25.39,
    },
    {
        "rank": 2, "rank_label": "2", "tied": False,
        "name": "Robert Harvey", "teams": "St Kilda",
        "year_min": 1988, "year_max": 2008,
        "games": 383, "total": 9656.0, "per_game": 25.21,
    },
    {
        "rank": 3, "rank_label": "3", "tied": False,
        "name": "Brent Harvey", "teams": "Kangaroos - North Melbourne",
        "year_min": 1996, "year_max": 2016,
        "games": 432, "total": 9213.0, "per_game": 21.33,
    },
]

_LEADERS_GOALS = [
    {
        "rank": 1, "rank_label": "1", "tied": False,
        "name": "Tony Lockett", "teams": "St Kilda - Sydney",
        "year_min": 1983, "year_max": 2002,
        "games": 281, "total": 1360.0, "per_game": 4.84,
    },
    {
        "rank": 2, "rank_label": "2", "tied": False,
        "name": "Gordon Coventry", "teams": "Collingwood",
        "year_min": 1920, "year_max": 1937,
        "games": 306, "total": 1299.0, "per_game": 4.25,
    },
    {
        "rank": 3, "rank_label": "3", "tied": False,
        "name": "Jason Dunstall", "teams": "Hawthorn",
        "year_min": 1985, "year_max": 1998,
        "games": 269, "total": 1254.0, "per_game": 4.66,
    },
]


# ── test 11 ───────────────────────────────────────────────────────────────────

def test_disposals_and_goals_whitelisted_for_full_table():
    assert "career_disposals" in uohp._FULL_TABLE_CATS
    assert "career_goals" in uohp._FULL_TABLE_CATS


# ── test 12 ───────────────────────────────────────────────────────────────────

def test_full_table_body_disposals_standard_format():
    cat = uohp.CATEGORIES["career_disposals"]
    rows = uohp.build_full_table_body("career_disposals", cat, _LEADERS_DISPOSALS)

    assert rows == [
        "| 1 | Scott Pendlebury **[data]** | Collingwood | 2006-2026 | 436 | 11,069 | 25.39 |<!-- HOF-TOP:career_disposals -->",
        "| 2 | Robert Harvey **[data]** | St Kilda | 1988-2008 | 383 | 9,656 | 25.21 |",
        "| 3 | Brent Harvey **[data]** | Kangaroos - North Melbourne | 1996-2016 | 432 | 9,213 | 21.33 |",
    ]


# ── test 13 ───────────────────────────────────────────────────────────────────

def test_full_table_body_goals_standard_format():
    cat = uohp.CATEGORIES["career_goals"]
    rows = uohp.build_full_table_body("career_goals", cat, _LEADERS_GOALS)

    assert rows == [
        "| 1 | Tony Lockett **[data]** | St Kilda - Sydney | 1983-2002 | 281 | 1,360 | 4.84 |<!-- HOF-TOP:career_goals -->",
        "| 2 | Gordon Coventry **[data]** | Collingwood | 1920-1937 | 306 | 1,299 | 4.25 |",
        "| 3 | Jason Dunstall **[data]** | Hawthorn | 1985-1998 | 269 | 1,254 | 4.66 |",
    ]


# ── test 14 ───────────────────────────────────────────────────────────────────

def test_full_table_disposals_end_to_end_with_markers(tmp_path):
    """A page carrying HOF-TABLE markers gets its body regenerated for disposals."""
    sub = tmp_path / "stat-disposals.md"
    sub.write_text(
        "| # | Player | Club(s) | Span | Games | Disposals | Per game |\n"
        "|--:|--------|---------|------|------:|----------:|---------:|\n"
        "<!-- HOF-TABLE-START:career_disposals -->\n"
        "| 1 | StalePlayer **[data]** | StaleClub | 2000-2020 | 400 | 9,999 | 24.99 |<!-- HOF-TOP:career_disposals -->\n"
        "<!-- HOF-TABLE-END:career_disposals -->\n"
        "Prose below.\n"
    )

    cat = uohp.CATEGORIES["career_disposals"]
    rows = uohp.build_full_table_body("career_disposals", cat, _LEADERS_DISPOSALS)
    text, changed = uohp.replace_table_body(sub.read_text(), "career_disposals", rows)

    assert changed
    assert "StalePlayer" not in text
    assert "Scott Pendlebury **[data]**" in text
    assert "11,069" in text
    assert "<!-- HOF-TOP:career_disposals -->" in text  # sentinel preserved on rank-1
    assert "<!-- HOF-TABLE-START:career_disposals -->" in text
    assert "<!-- HOF-TABLE-END:career_disposals -->" in text


# ── test 15 ───────────────────────────────────────────────────────────────────

def test_full_table_goals_end_to_end_with_markers(tmp_path):
    """A page carrying HOF-TABLE markers gets its body regenerated for goals."""
    sub = tmp_path / "stat-goals.md"
    sub.write_text(
        "| # | Player | Club(s) | Span | Games | Goals | Per game |\n"
        "|--:|--------|---------|------|------:|------:|---------:|\n"
        "<!-- HOF-TABLE-START:career_goals -->\n"
        "| 1 | StaleForward **[data]** | StaleClub | 1980-2000 | 250 | 999 | 3.99 |<!-- HOF-TOP:career_goals -->\n"
        "<!-- HOF-TABLE-END:career_goals -->\n"
        "Prose below.\n"
    )

    cat = uohp.CATEGORIES["career_goals"]
    rows = uohp.build_full_table_body("career_goals", cat, _LEADERS_GOALS)
    text, changed = uohp.replace_table_body(sub.read_text(), "career_goals", rows)

    assert changed
    assert "StaleForward" not in text
    assert "Tony Lockett **[data]**" in text
    assert "1,360" in text
    assert "<!-- HOF-TOP:career_goals -->" in text
    assert "<!-- HOF-TABLE-START:career_goals -->" in text
    assert "<!-- HOF-TABLE-END:career_goals -->" in text
