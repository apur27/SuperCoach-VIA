"""TDD tests for the blocking match-completeness gate (F8 / E2).

Context: matches_<year>.csv silently lost 9 games in 2026 (R10: 6, R17: 3).
The per-round fixture audit (audit_match_rounds) logged these as WARNINGs and
the harness continued, so predictions ran on incomplete data for weeks. This
gate promotes that audit to blocking: any incomplete round -> non-zero exit.

Instrument choice: we gate on audit_match_rounds, NOT check_match_completeness.
check_match_completeness compares each team's game-count to the season mode with
a >=2 threshold; a single dropped round leaves every affected team short by only
1, so it structurally CANNOT catch the R10 bug (verified: it returns 0 warnings
on the real broken 2026 file). audit_match_rounds is exact and fixture-aware.

No network: audit_match_rounds is mocked at the gate boundary.
"""
import os
import sys

from unittest.mock import patch

import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts import match_completeness_gate as gate
from scrapers.game_scraper import check_match_completeness


# ---------------------------------------------------------------------------
# The gate: WARNING rounds -> exit 1; all-complete -> exit 0
# ---------------------------------------------------------------------------

def test_gate_exits_nonzero_on_incomplete_round():
    audit_result = [
        {"year": 2026, "round_num": 10, "n_matches": 3, "expected": 9,
         "severity": "WARNING", "missing": ["Carlton v Brisbane Lions"]},
    ]
    with patch.object(gate, "audit_match_rounds", return_value=audit_result):
        code, warns = gate.run_match_completeness_gate("dummy_path.csv")
    assert code == 1
    assert len(warns) == 1
    assert warns[0]["round_num"] == 10


def test_gate_exits_zero_when_all_rounds_complete():
    audit_result = [
        {"year": 2026, "round_num": 9, "n_matches": 9, "expected": 9,
         "severity": "INFO", "missing": []},
    ]
    with patch.object(gate, "audit_match_rounds", return_value=audit_result):
        code, warns = gate.run_match_completeness_gate("dummy_path.csv")
    assert code == 0
    assert warns == []


def test_gate_exits_zero_when_fixture_unavailable():
    """Network/fixture unavailable -> audit yields no WARNING rows -> gate
    fails OPEN (does not block on a transient afltables outage)."""
    with patch.object(gate, "audit_match_rounds", return_value=[]):
        code, warns = gate.run_match_completeness_gate("dummy_path.csv")
    assert code == 0
    assert warns == []


def test_gate_main_returns_nonzero_exit_code():
    """The CLI entrypoint must surface the non-zero code for the bash harness."""
    audit_result = [
        {"year": 2026, "round_num": 17, "n_matches": 4, "expected": 7,
         "severity": "WARNING", "missing": ["Carlton v West Coast"]},
    ]
    with patch.object(gate, "audit_match_rounds", return_value=audit_result), \
         patch.object(gate, "_resolve_matches_path", return_value="dummy.csv"), \
         patch.object(gate.os.path, "exists", return_value=True):
        rc = gate.main(["--year", "2026"])
    assert rc == 1


# ---------------------------------------------------------------------------
# Evidence guard: WHY we don't gate on check_match_completeness.
# ---------------------------------------------------------------------------

def _season(pairs):
    """Build a matches DF from (team_1, team_2) tuples."""
    rows = [{"team_1_team_name": a, "team_2_team_name": b} for a, b in pairs]
    return pd.DataFrame(rows)


def test_check_completeness_misses_single_dropped_round():
    """A single dropped round leaves each affected team short by exactly 1,
    which is below check_match_completeness's >=2 threshold -> 0 warnings.
    This is the documented reason the gate uses audit_match_rounds instead."""
    # 4 teams, 3 complete round-robin rounds = each team plays 3 games.
    full = [("A", "B"), ("C", "D"),
            ("A", "C"), ("B", "D"),
            ("A", "D"), ("B", "C")]
    # Drop ONE game (A v B) -> A and B each have 2 games (short by 1).
    dropped = [p for p in full if p != ("A", "B")]
    warns = check_match_completeness(_season(dropped), 2026)
    assert warns == [], (
        "check_match_completeness unexpectedly flagged a single dropped round; "
        "if this passes it changed contract and the gate could switch to it"
    )


def test_check_completeness_flags_team_short_by_two():
    """Positive control: check_match_completeness DOES flag when a team is short
    by >=2 (its actual contract), confirming the function itself works."""
    # A plays 3, everyone else plays enough that mode is high and A is short.
    pairs = [("B", "C"), ("B", "D"), ("B", "E"), ("B", "F"),
             ("C", "D"), ("C", "E"), ("C", "F"),
             ("D", "E"), ("D", "F"), ("E", "F"),
             ("A", "B")]  # A plays only 1
    warns = check_match_completeness(_season(pairs), 2026)
    assert any("'A'" in w for w in warns)
