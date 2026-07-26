---
name: measurement-vintage-crossing
description: Two Surveyor measurement errors from 2026-07-26 — how to pool backtest CSVs and how to read the README round badge
metadata:
  type: feedback
---

Two of my own README-review findings were overturned by Gaffer with correct evidence
(verified independently 2026-07-26). Rules to prevent recurrence:

1. **When pooling per-round backtest artifacts, supersede by FILE per (year, round) —
   never "latest file per round by mtime" applied independently per artifact type.**
   Independent latest-per-round selection can cross two backtest vintages (e.g. an R18
   re-score covering 14 of 18 teams leaves 4 teams from the older run alive), producing
   figures that match on n but differ on w10/bias. My 95.74/−0.113 readings were
   vintage-crossed; the correct merge (backtest_summary_*.csv concat sorted by mtime,
   drop_duplicates(year,round) keep last — the update_eval_surface.sh logic) gives
   95.78/−0.110. **How to apply:** reproduce the regenerator's documented merge, or run
   the regenerator's own selection code, before declaring a published figure wrong.

2. **The README data badge names the last SCRAPED round, not the last predicted round.**
   Weekly commits are named for the UPCOMING round (predictions), while
   matches_<year>.csv ends at the settled round. "R21 shipped" does not make a
   "round 20" data badge stale. **How to apply:** compare the badge against
   max(round_num) in matches_<year>.csv, not against commit titles or prediction CSVs.

3. **Before instructing quarantine of "orphan" artifacts, check for sibling run logs and
   same-timestamp artifact families.** backtest_by_position_* files I flagged as orphans
   had matching backtest_run_*.log + summary + by_team files — legitimate run output,
   merely outside the staging allowlist. Untracked ≠ orphaned.

**Why:** three overridden instructions in one cycle; an advisor whose measurements get
overturned loses routing authority. Gaffer's "measure before acting on Surveyor removal
instructions" memory now exists specifically because of this.
