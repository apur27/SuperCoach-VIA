#!/usr/bin/env python3
"""Blocking match-completeness gate for the weekly harness (F8 / E2).

Promotes the per-round fixture audit (scrapers.game_scraper.audit_match_rounds)
from warnings-only to a hard gate: if any home-and-away round in the current
season's matches file is missing scheduled matchups, exit non-zero so the
harness aborts before predictions/docs are pushed on incomplete data.

Instrument choice -- audit_match_rounds, NOT check_match_completeness:
    check_match_completeness compares each team's game-count to the season mode
    with a >=2 threshold. A single dropped round leaves every affected team
    short by only 1, so it cannot catch the "R10 2026" class of bug (verified:
    it returns 0 warnings on the real broken 2026 file). audit_match_rounds is
    exact and fixture-aware, and it names the exact missing matchups.

Fails OPEN on a fixture/network outage: audit_match_rounds skips any round whose
published fixture cannot be fetched, so a transient afltables outage yields no
WARNING rows and the gate passes (a bad scrape is caught on the next run, an
outage never blocks the pipeline).

Usage:
    python scripts/match_completeness_gate.py [--year YYYY] [--matches PATH]
Exit code 0 = all covered rounds complete; 1 = at least one incomplete round.
"""
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scrapers.game_scraper import audit_match_rounds

_MATCHES_DIR = os.path.join(_REPO_ROOT, "data", "matches")


def run_match_completeness_gate(matches_path: str) -> Tuple[int, List[Dict]]:
    """Audit one matches CSV and return (exit_code, warning_issues).

    exit_code is 1 when any round is flagged WARNING (incomplete vs the
    published fixture), else 0.
    """
    issues = audit_match_rounds(matches_path)
    warns = [i for i in issues if i.get("severity") == "WARNING"]
    return (1 if warns else 0), warns


def _resolve_matches_path(year: Optional[int], matches: Optional[str]) -> str:
    if matches:
        return matches
    yr = year if year is not None else datetime.now().year
    return os.path.join(_MATCHES_DIR, f"matches_{yr}.csv")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=None,
                        help="Season to check (default: current calendar year).")
    parser.add_argument("--matches", type=str, default=None,
                        help="Explicit matches CSV path (overrides --year).")
    args = parser.parse_args(argv)

    path = _resolve_matches_path(args.year, args.matches)
    if not os.path.exists(path):
        print(f"[match-gate] matches file not found: {path} -- nothing to check")
        return 0

    code, warns = run_match_completeness_gate(path)
    if code == 0:
        print(f"[match-gate] PASS: all covered rounds complete in {os.path.basename(path)}")
        return 0

    print(f"[match-gate] FAIL: {len(warns)} incomplete round(s) in {os.path.basename(path)}:")
    for w in warns:
        miss = "; ".join(w.get("missing", []))
        print(f"  R{w['round_num']}: {w['n_matches']}/{w['expected']} scraped -- MISSING: {miss}")
    print("[match-gate] Backfill the missing matches before shipping "
          "(re-run the match scraper for these rounds).")
    return code


if __name__ == "__main__":
    sys.exit(main())
