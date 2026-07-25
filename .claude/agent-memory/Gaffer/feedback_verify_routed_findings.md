---
name: verify-routed-findings
description: Verify a routed finding's specific file claims before executing it — Surveyor/coordinator instructions have named the wrong files, and blind compliance would have deleted good output
metadata:
  type: feedback
---

Before executing a routed instruction that names specific files to delete, quarantine,
or exclude, verify the claim against the actual files. Findings arrive with confident
specifics that are sometimes wrong.

**Why:** 2026-07-25. A routed instruction said to "quarantine the 3 untracked
`backtest_by_position_*` files and 3 untracked run-1 prediction CSVs left over from
the FATALed run." Verification showed:
- The `backtest_by_position_*` files were NOT orphans from the aborted run — they came
  from four *different* legitimate runs and were untracked only because that filename
  pattern was missing from the Phase-1 allowlist. Quarantining them would have
  discarded good output; the correct fix was the opposite (add the pattern).
- Of the 3 forward CSVs, only `next_round_21_..._20260720_1557.csv` was genuinely
  tainted. The two R20 archives are load-bearing inputs to the by-archive backtest.

Separately, on the same day, a documented exclusion (`refresh_and_rank.sh`: "lineups
are intentionally excluded until their scraper corruption is fixed (S3)") turned out
to be STALE — measurement showed corruption confined to 700 legacy rows of 33,999,
with all 36 new rows clean. Checking made a "violate the documented reason" instruction
safe to carry out.

**How to apply:** when an instruction names files to remove or exclude, spend one
command measuring before acting — `git status`/`git log` on the paths, and a content
check on what actually distinguishes good output from bad. Report the discrepancy and
do the correct thing rather than the instructed thing; say plainly which parts of the
finding were wrong and why. This applies to Surveyor output specifically (already
noted as advisory-not-authoritative) and to any relayed worklist.

Related: [[consult-surveyor]] (Surveyor findings are recommendations, verify before acting).
