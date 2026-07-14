"""
Guard that the HOF verdict loop in weekly_refresh.sh only stamps pages that
check_hof_numbers.py actually inspects.  Two pages are structurally excluded
from _SUBPAGES and must never receive a phantom PASS stamp.

The navigation hub (hall-of-fame-stat-leaders.md) is NOT in _SUBPAGES but IS now
gated by check_hof_hub() (F1) — so it earns a genuine stamp via a separate
harness step, and must NOT appear in the excluded set.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.check_hof_numbers import _SUBPAGES, HUB_REL, check_hof_hub

_CHECKED_FILES = {v[0] for v in _SUBPAGES.values()}

# These two pages are NOT in _SUBPAGES and are NOT gated by check_hof_hub either
# (kicks-handballs and single-season carry multi-figure rows the checker doesn't
# parse) — the checker never inspects them, so they must never be stamped.
_EXCLUDED = {
    "docs/hall-of-fame-stat-kicks-handballs.md",
    "docs/hall-of-fame-stat-single-season.md",
}

# The ten files the harness loop must stamp — must match _SUBPAGES exactly.
_HARNESS_LOOP_FILES = {
    "docs/hall-of-fame-stat-games.md",
    "docs/hall-of-fame-stat-marks.md",
    "docs/hall-of-fame-stat-tackles.md",
    "docs/hall-of-fame-stat-hitouts.md",
    "docs/hall-of-fame-stat-brownlow.md",
    "docs/hall-of-fame-stat-goalassists.md",
    "docs/hall-of-fame-stat-clearances.md",
    "docs/hall-of-fame-stat-contested.md",
    "docs/hall-of-fame-stat-disposals.md",
    "docs/hall-of-fame-stat-goals.md",
}


def test_subpages_matches_harness_loop():
    """_SUBPAGES and the harness loop list must be identical sets."""
    assert _CHECKED_FILES == _HARNESS_LOOP_FILES, (
        f"Drift between _SUBPAGES and harness loop.\n"
        f"  In _SUBPAGES but not harness: {_CHECKED_FILES - _HARNESS_LOOP_FILES}\n"
        f"  In harness but not _SUBPAGES: {_HARNESS_LOOP_FILES - _CHECKED_FILES}"
    )


def test_excluded_pages_not_in_subpages():
    """The three unchecked pages must not appear in _SUBPAGES."""
    overlap = _EXCLUDED & _CHECKED_FILES
    assert not overlap, f"These pages are in _SUBPAGES but checker never inspects them: {overlap}"


def test_subpages_count():
    """_SUBPAGES must cover exactly 10 stat pages (not all 13 stat-*.md files)."""
    assert len(_CHECKED_FILES) == 10, f"Expected 10 _SUBPAGES, got {len(_CHECKED_FILES)}: {_CHECKED_FILES}"


# ---------------------------------------------------------------------------
# F1 — the navigation hub is now in the gate's scope and earns a real stamp.
# ---------------------------------------------------------------------------

def test_hub_is_gated_by_checker():
    """The hub path must be the one check_hof_hub inspects, and it must not be
    in the excluded 'never stamp' set (it is now genuinely verified)."""
    assert HUB_REL == "docs/hall-of-fame-stat-leaders.md"
    assert HUB_REL not in _EXCLUDED
    assert callable(check_hof_hub)


def test_harness_stamps_the_hub():
    """weekly_refresh.sh must record a verdict for the hub (F1) — otherwise the
    pre-commit stamp gate fail-closes and reverts the regenerated hub."""
    harness = os.path.join(
        os.path.dirname(__file__), "..", "..", "scripts", "weekly_refresh.sh"
    )
    with open(harness) as fh:
        text = fh.read()
    assert "hall-of-fame-stat-leaders.md" in text
    # The hub stamp block records a verdict for the hub via record-sentinel-verdict.sh.
    assert "record-sentinel-verdict.sh --doc \"$hub\"" in text or \
           "record-sentinel-verdict.sh --doc \"$hub\" --verdict PASS" in text
