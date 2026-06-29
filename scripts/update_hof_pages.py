#!/usr/bin/env python3
"""Deterministic HOF stat page updater.

Reads docs/hall-of-fame/_stat_leaders.json (written by compute_stat_leaders.py)
and propagates #1-leader values to sentinel-marked lines in the HOF markdown
files.  No LLM required — pure string replacement on sentinel comments.

Sentinel format:
  Hub row:      ...| Name 434 **[data]** |<!-- HOF-HUB:career_games -->
  Sub-page row: ...| 434 |<!-- HOF-TOP:career_games -->

Exit codes: 0 = success, 1 = JSON missing/invalid.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).parent.parent
JSON_PATH = REPO / "docs" / "hall-of-fame" / "_stat_leaders.json"

CATEGORIES: dict[str, dict] = {
    "career_goals": {
        "hub_label": "Career goals",
        "hub_link_text": "Goals leaderboard",
        "hub_link_href": "hall-of-fame-stat-goals.md",
        "subpage": "docs/hall-of-fame-stat-goals.md",
        "fmt": "thousands",
        "games_only": False,
    },
    "career_games": {
        "hub_label": "Career games",
        "hub_link_text": "Games leaderboard",
        "hub_link_href": "hall-of-fame-stat-games.md",
        "subpage": "docs/hall-of-fame-stat-games.md",
        "fmt": "int",
        "games_only": True,
    },
    "career_disposals": {
        "hub_label": "Career disposals",
        "hub_link_text": "Disposals leaderboard",
        "hub_link_href": "hall-of-fame-stat-disposals.md",
        "subpage": "docs/hall-of-fame-stat-disposals.md",
        "fmt": "thousands",
        "games_only": False,
    },
    "career_marks": {
        "hub_label": "Career marks",
        "hub_link_text": "Marks leaderboard",
        "hub_link_href": "hall-of-fame-stat-marks.md",
        "subpage": "docs/hall-of-fame-stat-marks.md",
        "fmt": "thousands",
        "games_only": False,
    },
    "career_tackles": {
        "hub_label": "Career tackles",
        "hub_link_text": "Tackles leaderboard",
        "hub_link_href": "hall-of-fame-stat-tackles.md",
        "subpage": "docs/hall-of-fame-stat-tackles.md",
        "fmt": "thousands",
        "games_only": False,
    },
    "career_contested_possessions": {
        "hub_label": "Career contested possessions",
        "hub_link_text": "Contested poss. leaderboard",
        "hub_link_href": "hall-of-fame-stat-contested.md",
        "subpage": "docs/hall-of-fame-stat-contested.md",
        "fmt": "thousands",
        "games_only": False,
    },
    "career_clearances": {
        "hub_label": "Career clearances",
        "hub_link_text": "Clearances leaderboard",
        "hub_link_href": "hall-of-fame-stat-clearances.md",
        "subpage": "docs/hall-of-fame-stat-clearances.md",
        "fmt": "thousands",
        "games_only": False,
    },
    "career_hit_outs": {
        "hub_label": "Career hit-outs",
        "hub_link_text": "Hit-outs leaderboard",
        "hub_link_href": "hall-of-fame-stat-hitouts.md",
        "subpage": "docs/hall-of-fame-stat-hitouts.md",
        "fmt": "thousands",
        "games_only": False,
    },
    "career_brownlow_votes": {
        "hub_label": "Career Brownlow votes",
        "hub_link_text": "Brownlow leaderboard",
        "hub_link_href": "hall-of-fame-stat-brownlow.md",
        "subpage": "docs/hall-of-fame-stat-brownlow.md",
        "fmt": "int",
        "games_only": False,
    },
    "career_goal_assists": {
        "hub_label": "Career goal assists",
        "hub_link_text": "Goal assists leaderboard",
        "hub_link_href": "hall-of-fame-stat-goalassists.md",
        "subpage": "docs/hall-of-fame-stat-goalassists.md",
        "fmt": "int",
        "games_only": False,
    },
}

_STAMP = "auto-updated from _stat_leaders.json by update_hof_pages.py"

# Categories with clean single-stat-column tables that support full body regeneration.
# TODO: add career_disposals and career_goals once per-page prose is also driven from JSON.
# TODO: add career_kicks / career_handballs (two-leader-per-row format needs custom handling).
_FULL_TABLE_CATS: frozenset[str] = frozenset({
    "career_games",
    "career_marks",
    "career_tackles",
    "career_hit_outs",
    "career_brownlow_votes",
    "career_goal_assists",
    "career_clearances",
    "career_contested_possessions",
})


def fmt_value(value: float, fmt: str) -> str:
    if fmt == "thousands":
        return f"{int(value):,}"
    return str(int(value))


def build_hub_row(key: str, cat: dict, leader: dict) -> str:
    count = fmt_value(leader["total"], cat["fmt"])
    return (
        f"| {cat['hub_label']} | [{cat['hub_link_text']}]({cat['hub_link_href']}) | "
        f"{leader['name']} {count} **[data]** |<!-- HOF-HUB:{key} -->"
    )


def build_subpage_row(key: str, cat: dict, leader: dict) -> str:
    name = leader["name"]
    teams = leader["teams"]
    span = f"{leader['year_min']}-{leader['year_max']}"
    count = fmt_value(leader["total"], cat["fmt"])
    if cat["games_only"]:
        return f"| 1 | {name} **[data]** | {teams} | {span} | {count} |<!-- HOF-TOP:{key} -->"
    games = str(int(leader["games"]))
    per_game = f"{leader['per_game']:.2f}"
    return (
        f"| 1 | {name} **[data]** | {teams} | {span} | {games} | {count} | {per_game} |"
        f"<!-- HOF-TOP:{key} -->"
    )


def build_full_table_row(key: str, cat: dict, leader: dict) -> str:
    """Build a markdown table row for any rank. Appends HOF-TOP sentinel on rank 1."""
    rank_label = leader["rank_label"]
    name = leader["name"]
    teams = leader["teams"]
    span = f"{leader['year_min']}-{leader['year_max']}"
    total_str = fmt_value(leader["total"], cat["fmt"])
    sentinel = f"<!-- HOF-TOP:{key} -->" if leader["rank"] == 1 else ""

    if cat.get("games_only"):
        return f"| {rank_label} | {name} **[data]** | {teams} | {span} | {total_str} |{sentinel}"
    else:
        games_str = str(int(leader["games"]))
        per_game_str = f"{leader['per_game']:.2f}"
        return f"| {rank_label} | {name} **[data]** | {teams} | {span} | {games_str} | {total_str} | {per_game_str} |{sentinel}"


def build_full_table_body(key: str, cat: dict, leaders: list) -> list[str]:
    """Return markdown rows for all leaders with rank <= 20."""
    return [
        build_full_table_row(key, cat, leader)
        for leader in leaders
        if leader["rank"] <= 20
    ]


def replace_table_body(text: str, key: str, rows: list[str]) -> tuple[str, bool]:
    """Replace content between HOF-TABLE-START/END markers with rows."""
    start_marker = f"<!-- HOF-TABLE-START:{key} -->"
    end_marker = f"<!-- HOF-TABLE-END:{key} -->"

    lines = text.splitlines(keepends=True)
    start_idx = end_idx = None
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n").rstrip("\r")
        if stripped == start_marker:
            start_idx = i
        elif stripped == end_marker:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        return text, False

    new_lines = lines[: start_idx + 1] + [r + "\n" for r in rows] + lines[end_idx:]
    new_text = "".join(new_lines)
    return new_text, new_text != text


def update_file(path: Path, replacements: dict[str, str], today: str) -> bool:
    """Replace sentinel lines and refresh dates in path.

    Returns True if the file was modified.
    """
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    changed = False
    new_lines: list[str] = []

    for line in lines:
        stripped = line.rstrip("\n").rstrip("\r")

        # Check sentinel replacements
        sentinel_matched = False
        for sentinel, new_content in replacements.items():
            if sentinel in stripped:
                new_line = new_content + "\n"
                if new_line != line:
                    changed = True
                new_lines.append(new_line)
                sentinel_matched = True
                break
        if sentinel_matched:
            continue

        # Update "Last refreshed:" line
        if stripped.startswith("*Last refreshed:"):
            # Preserve everything after the date token
            suffix = stripped.split(".", 1)[1] if "." in stripped else ""
            new_line = f"*Last refreshed: {today}.{suffix}\n"
            if new_line != line:
                changed = True
            new_lines.append(new_line)
            continue

        # Update DataSentinel stamp
        if "DataSentinel: PASS @" in stripped:
            new_line = f"  DataSentinel: PASS @ {today} ({_STAMP})\n"
            if new_line != line:
                changed = True
            new_lines.append(new_line)
            continue

        new_lines.append(line)

    if changed:
        path.write_text("".join(new_lines))
    return changed


def run_updates(
    json_path: Path | None = None,
    repo_root: Path | None = None,
    today: str | None = None,
) -> int:
    """Main entry point. Returns 0 on success, 1 on error."""
    if json_path is None:
        json_path = JSON_PATH
    if repo_root is None:
        repo_root = REPO
    if today is None:
        today = date.today().isoformat()

    if not json_path.exists():
        print(f"ERROR: JSON not found: {json_path}")
        return 1

    try:
        data = json.loads(json_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: JSON parse failed: {exc}")
        return 1

    categories_data = data.get("categories", {})

    # Build hub replacements (all HOF-HUB sentinels go into one file)
    hub_path = repo_root / "docs" / "hall-of-fame-stat-leaders.md"
    hub_replacements: dict[str, str] = {}

    # Build per-subpage replacements
    subpage_replacements: dict[Path, dict[str, str]] = {}

    for key, cat in CATEGORIES.items():
        if key not in categories_data:
            print(f"  WARN: category '{key}' not found in JSON, skipping")
            continue
        leaders = categories_data[key].get("leaders", [])
        if not leaders:
            print(f"  WARN: no leaders for '{key}', skipping")
            continue
        leader = leaders[0]

        hub_replacements[f"<!-- HOF-HUB:{key} -->"] = build_hub_row(key, cat, leader)

        subpage = repo_root / cat["subpage"]
        if not subpage.exists():
            print(f"  WARN: subpage not found: {subpage}")
            continue
        subpage_replacements.setdefault(subpage, {})
        subpage_replacements[subpage][f"<!-- HOF-TOP:{key} -->"] = build_subpage_row(key, cat, leader)

    # Apply hub updates
    if hub_path.exists():
        changed = update_file(hub_path, hub_replacements, today)
        if changed:
            print(f"  updated: {hub_path.relative_to(repo_root)}")
        else:
            missing = [s for s in hub_replacements if s not in hub_path.read_text()]
            if missing:
                for s in missing:
                    print(f"  WARN: sentinel not found in hub: {s}")
    else:
        print(f"  WARN: hub file not found: {hub_path}")

    # Apply subpage updates (rank-1 sentinel rows)
    for subpage, reps in subpage_replacements.items():
        changed = update_file(subpage, reps, today)
        if changed:
            print(f"  updated: {subpage.relative_to(repo_root)}")
        else:
            for s in reps:
                if s not in subpage.read_text():
                    print(f"  WARN: sentinel not found in {subpage.name}: {s}")

    # Full table body regeneration (ranks 1-20 from JSON)
    for key, cat in CATEGORIES.items():
        if key not in _FULL_TABLE_CATS:
            continue
        if key not in categories_data:
            continue
        leaders = categories_data[key].get("leaders", [])
        if not leaders:
            continue
        subpage = repo_root / cat["subpage"]
        if not subpage.exists():
            continue
        rows = build_full_table_body(key, cat, leaders)
        text = subpage.read_text()
        new_text, changed = replace_table_body(text, key, rows)
        if changed:
            subpage.write_text(new_text)
            print(f"  full-table updated: {subpage.relative_to(repo_root)}")

    print("HOF pages update complete.")
    return 0


if __name__ == "__main__":
    sys.exit(run_updates())
