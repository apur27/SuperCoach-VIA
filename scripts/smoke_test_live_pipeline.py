#!/usr/bin/env python3
"""Smoke test for the post-R11 hardening of live_analysis_pipeline.py.

Loads the last R11 FT snapshot and exercises the four behaviours that broke
during the live R11 run:

  1. classify_status() correctly maps "Final Siren" -> "FT" (was the routing
     bug that wrote 8 FT polls into the Q1-live document).
  2. classify_status() correctly handles "Qtr Time", "Half Time",
     "3 Qtr Time", and short-form Q-tokens.
  3. format_analysis_block() with a prev_state whose totals match the current
     snapshot returns (None, prev_state) - i.e. the skip-if-unchanged guard
     fires for a stalled feed.
  4. format_analysis_block() with a meaningfully different prev_state DOES
     produce a block (sanity check that we didn't over-prune).

This is a unit-level smoke test. It does NOT run the polling loop, does NOT
write to any docs, and does NOT call git. Run it ad-hoc when changing
classify_status or the skip-guard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from live_analysis_pipeline import (  # noqa: E402
    TrendCache,
    build_prev_state,
    classify_status,
    format_analysis_block,
    team_totals,
    parse_score_pts,
)

FT_SNAPSHOT = REPO / "data" / "live_snapshots" / "9789_20260517_1804_full-time.json"


def assert_eq(actual, expected, label: str) -> None:
    ok = actual == expected
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {label}: got {actual!r}, expected {expected!r}")
    if not ok:
        sys.exit(1)


def main() -> int:
    print("=== Smoke test: live_analysis_pipeline post-R11 hardening ===\n")

    # ---- 1. classify_status hardening ------------------------------------
    print("Section 1: classify_status() covers all observed R11 status strings")
    cases = [
        ("Q1 6:42", "Q1"),
        ("Qtr Time", "QT"),
        ("Quarter Time", "QT"),
        ("Q2 14:23", "Q2"),
        ("Half Time", "HT"),
        ("Q3 19:11", "Q3"),
        ("3 Qtr Time", "3QT"),
        ("Three Quarter Time", "3QT"),
        ("Q4 28:36", "Q4"),
        ("Final Siren", "FT"),  # <-- the R11 routing bug
        ("Full Time", "FT"),
        ("FT", "FT"),
        ("HT", "HT"),
    ]
    for status_raw, expected in cases:
        assert_eq(classify_status(status_raw), expected, f"classify({status_raw!r})")

    # Unrecognised -> None (NOT silent fallback to Q1)
    assert_eq(classify_status("garbage string"), None, "classify('garbage') is None")
    assert_eq(classify_status(""), None, "classify('') is None")

    # ---- 2. Load FT snapshot ---------------------------------------------
    print("\nSection 2: load FT snapshot")
    if not FT_SNAPSHOT.exists():
        print(f"  [FAIL] snapshot missing: {FT_SNAPSHOT}")
        return 1
    snap = json.loads(FT_SNAPSHOT.read_text())
    print(f"  [PASS] loaded {FT_SNAPSHOT.name} ({len(snap['players'])} players)")
    print(f"        header status = {snap['header']['status']!r}")
    print(f"        score = {snap['header']['home_team_full']} {snap['header']['home_score']} "
          f"vs {snap['header']['away_team_full']} {snap['header']['away_score']}")
    status_code = classify_status(snap["header"]["status"])
    assert_eq(status_code, "FT", "FT snapshot routes to FT code")

    # ---- 3. Skip-if-unchanged: build a prev_state with identical numbers --
    print("\nSection 3: format_analysis_block returns None on stalled feed")
    players = snap["players"]
    ric_t = team_totals(players, "RI")
    stk_t = team_totals(players, "SK")
    home_pts = parse_score_pts(snap["header"]["home_score"])
    away_pts = parse_score_pts(snap["header"]["away_score"])
    if snap["header"]["home_team_full"] == "Richmond":
        ric_pts, stk_pts = home_pts, away_pts
    else:
        ric_pts, stk_pts = away_pts, home_pts

    identical_prev = build_prev_state(
        players, ric_t, stk_t, ric_pts, stk_pts, status_code,
    )
    print(f"  prev_state mirrors current: ric_pts={ric_pts}, stk_pts={stk_pts}, "
          f"ric_disp={ric_t['disposals']}, stk_disp={stk_t['disposals']}")

    # NB: the skip guard intentionally does NOT fire on quarter-break codes
    # (FT, QT, HT, 3QT) since the routing/heading transition is itself news.
    # So a stalled FT snapshot WILL still produce a block. Re-run with an
    # in-quarter status to exercise the skip path properly.
    trend = TrendCache()
    block_ft, _ = format_analysis_block(snap, "FT", trend, identical_prev)
    if block_ft is None:
        print("  [INFO] FT snapshot with identical prev_state was skipped "
              "(skip guard fires on break codes too - that's a design choice).")
    else:
        print(f"  [PASS] FT snapshot produced a block as expected "
              f"(skip guard skipped because status_code=FT). "
              f"block len = {len(block_ft)} chars.")

    # Now the real test: pretend the snapshot is still in Q4 (in-play),
    # which is when the R11 feed kept stalling and pushing duplicate blocks.
    trend2 = TrendCache()
    identical_prev_q4 = build_prev_state(
        players, ric_t, stk_t, ric_pts, stk_pts, "Q4",
    )
    block_q4_stalled, returned_prev = format_analysis_block(
        snap, "Q4", trend2, identical_prev_q4,
    )
    assert_eq(block_q4_stalled, None, "Q4 stalled snapshot -> block is None")
    assert_eq(returned_prev is identical_prev_q4, True,
              "Q4 stalled snapshot -> prev_state passed through unchanged")

    # ---- 4. Sanity: meaningfully different prev_state DOES produce block --
    print("\nSection 4: format_analysis_block produces a block when state changed")
    different_prev = dict(identical_prev_q4)
    different_prev["ric_disposals"] = ric_t["disposals"] - 30  # pretend RIC gained 30 disp
    different_prev["ric_pts"] = ric_pts - 7  # pretend they kicked a goal
    trend3 = TrendCache()
    block_real, _ = format_analysis_block(snap, "Q4", trend3, different_prev)
    if block_real is None:
        print("  [FAIL] expected a block when prev_state differs, got None")
        return 1
    print(f"  [PASS] meaningfully-different prev_state produced a block "
          f"({len(block_real)} chars, {block_real.count(chr(10))+1} lines)")

    # Print the first 800 chars of the real block so we eyeball it.
    print("\n=== Sample block (first 800 chars) ===")
    print(block_real[:800])
    print("=== End sample ===\n")

    print("All smoke-test assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
