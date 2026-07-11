"""
Guard that the HOF verdict loop in weekly_refresh.sh only stamps pages that
check_hof_numbers.py actually inspects.  Three pages are structurally excluded
from _SUBPAGES and must never receive a phantom PASS stamp.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.check_hof_numbers import _SUBPAGES

_CHECKED_FILES = {v[0] for v in _SUBPAGES.values()}

# These three pages are NOT in _SUBPAGES (checker never inspects them).
_EXCLUDED = {
    "docs/hall-of-fame-stat-kicks-handballs.md",
    "docs/hall-of-fame-stat-leaders.md",
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
