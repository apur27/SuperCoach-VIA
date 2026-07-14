#!/usr/bin/env python3
"""Deterministic HOF numeric checker.

Reads _stat_leaders.json and verifies rank-1 totals in each HOF sub-page
match the JSON ground truth. Exit 0 = all OK, exit 1 = mismatch(es) found.

Replaces LLM arithmetic in the DataSentinel gate for rank-1 HOF rows.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JSON_PATH = REPO / "docs" / "hall-of-fame" / "_stat_leaders.json"

# Navigation hub — one HOF-HUB:<key> sentinel row per career category, each
# carrying that category's rank-1 total inline before the **[data]** tag. The
# hub is rewritten by update_hof_pages.py every cycle but was never gated, so its
# council-pipeline stamp had no matching DataSentinel record and the pre-commit
# hook fail-closed and reverted it (Phase 4 death, 07-13 / 07-14). This checker
# now verifies the hub's rank-1 figures against the same JSON the subpages use.
HUB_REL = "docs/hall-of-fame-stat-leaders.md"

# (subpage relative path, games_only flag)
# games_only=True  → rank-1 row: | rank | name | clubs | span | count |
# games_only=False → rank-1 row: | rank | name | clubs | span | games | count | per_game |
_SUBPAGES: dict[str, tuple[str, bool]] = {
    "career_games":                ("docs/hall-of-fame-stat-games.md",     True),
    "career_marks":                ("docs/hall-of-fame-stat-marks.md",     False),
    "career_tackles":              ("docs/hall-of-fame-stat-tackles.md",   False),
    "career_hit_outs":             ("docs/hall-of-fame-stat-hitouts.md",   False),
    "career_brownlow_votes":       ("docs/hall-of-fame-stat-brownlow.md",  False),
    "career_goal_assists":         ("docs/hall-of-fame-stat-goalassists.md", False),
    "career_clearances":           ("docs/hall-of-fame-stat-clearances.md", False),
    "career_contested_possessions":("docs/hall-of-fame-stat-contested.md", False),
    "career_disposals":            ("docs/hall-of-fame-stat-disposals.md", False),
    "career_goals":                ("docs/hall-of-fame-stat-goals.md",     False),
}


def _parse_number(s: str) -> float | None:
    try:
        return float(s.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _extract_rank1_total(line: str, games_only: bool) -> str | None:
    """Extract stat total from a HOF-TOP sentinel row."""
    sentinel_idx = line.find("<!--")
    row = line[:sentinel_idx].rstrip() if sentinel_idx != -1 else line.rstrip()
    cols = [c.strip() for c in row.split("|") if c.strip()]
    # games_only: [rank, name, clubs, span, count]  → index 4
    # multi-col:  [rank, name, clubs, span, games, count, per_game] → index 5
    idx = 4 if games_only else 5
    return cols[idx] if len(cols) > idx else None


def _extract_hub_total(line: str) -> str | None:
    """Extract the rank-1 total from a HOF-HUB row.

    Row shape: ``| Career goals | [..](..) | Tony Lockett 1,360 **[data]** |``
    The figure is the number token immediately before the ``**[data]**`` tag.
    """
    m = re.search(r"([\d][\d,]*)\s*\*\*\[data\]\*\*", line)
    return m.group(1) if m else None


def check_hof_hub(categories: dict, repo_root: Path) -> int:
    """Verify the hub's rank-1 figures against the JSON ground truth.

    Iterates the same 10 category keys the subpages use; for each, finds the
    ``<!-- HOF-HUB:<key> -->`` row and compares its inline rank-1 total to
    ``categories[key]["leaders"][0]["total"]``. Returns the mismatch count.
    A missing hub file or missing row is skipped (never a false FAIL).
    """
    hub = Path(repo_root) / HUB_REL
    if not hub.exists():
        print(f"  SKIP: hub not found: {Path(HUB_REL).name}")
        return 0

    lines = hub.read_text().splitlines()
    failures = 0
    for key in _SUBPAGES:
        if key not in categories:
            continue
        leaders = categories[key].get("leaders", [])
        if not leaders:
            continue
        expected = leaders[0]["total"]

        sentinel = f"<!-- HOF-HUB:{key} -->"
        hub_line = next((ln for ln in lines if sentinel in ln), None)
        if hub_line is None:
            print(f"  WARN: no HOF-HUB sentinel for '{key}' in {Path(HUB_REL).name}")
            continue

        actual_str = _extract_hub_total(hub_line)
        if actual_str is None:
            print(f"  WARN: could not extract hub total for '{key}' in {Path(HUB_REL).name}")
            continue
        actual = _parse_number(actual_str)
        if actual is None:
            print(f"  WARN: could not parse hub '{actual_str}' for '{key}'")
            continue

        if actual != expected:
            print(
                f"  FAIL: hub {key}: expected {expected:g}, got {actual:g} "
                f"('{actual_str}') in {Path(HUB_REL).name}"
            )
            failures += 1
        else:
            print(f"  OK: hub {key} rank-1 = {actual_str}")
    return failures


def check_hof_numbers(json_path: Path | None = None, repo_root: Path | None = None) -> int:
    if json_path is None:
        json_path = JSON_PATH
    if repo_root is None:
        repo_root = REPO

    if not json_path.exists():
        print(f"ERROR: JSON not found: {json_path}")
        return 1

    try:
        data = json.loads(json_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: JSON parse failed: {exc}")
        return 1

    categories = data.get("categories", {})
    failures = 0

    for key, (subpage_rel, games_only) in _SUBPAGES.items():
        if key not in categories:
            print(f"  SKIP: '{key}' not in JSON")
            continue

        leaders = categories[key].get("leaders", [])
        if not leaders:
            continue
        expected = leaders[0]["total"]

        subpage = Path(repo_root) / subpage_rel
        if not subpage.exists():
            print(f"  SKIP: subpage not found: {Path(subpage_rel).name}")
            continue

        sentinel = f"<!-- HOF-TOP:{key} -->"
        hof_line = next(
            (ln for ln in subpage.read_text().splitlines() if sentinel in ln),
            None,
        )
        if hof_line is None:
            print(f"  WARN: no HOF-TOP sentinel found in {Path(subpage_rel).name}")
            continue

        actual_str = _extract_rank1_total(hof_line, games_only)
        if actual_str is None:
            print(f"  WARN: could not extract rank-1 total from {Path(subpage_rel).name}")
            continue

        actual = _parse_number(actual_str)
        if actual is None:
            print(f"  WARN: could not parse '{actual_str}' in {Path(subpage_rel).name}")
            continue

        if actual != expected:
            print(
                f"  FAIL: {key}: expected {expected:g}, got {actual:g} ('{actual_str}')"
                f" in {Path(subpage_rel).name}"
            )
            failures += 1
        else:
            print(f"  OK: {key} rank-1 = {actual_str}")

    # Gate the navigation hub against the same JSON (F1).
    failures += check_hof_hub(categories, repo_root)

    if failures:
        print(f"\n{failures} mismatch(es) found.")
        return 1
    print("All HOF numbers verified OK.")
    return 0


if __name__ == "__main__":
    sys.exit(check_hof_numbers())
