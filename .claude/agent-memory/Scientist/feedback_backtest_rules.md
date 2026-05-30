---
name: feedback-backtest-rules
description: Absolute rules for backtest — never re-run all rounds, always preserve existing results, only run the missing round
metadata:
  type: feedback
---

RULE 1 — INCREMENTAL ONLY: Only ever backtest the round(s) that are missing. Never pass `--start-round 1`. Detect the last completed round from `data/prediction/backtest/backtest_summary_*.csv` and start from the next one.

RULE 2 — PRESERVE ALL RESULTS: Existing backtest results are sacred. `generate_backtest_section()` must merge ALL `backtest_summary_*.csv` files (oldest-first, dedup by year+round keeping latest) so cumulative doc always shows R1 through current round — never just the latest run.

**Why:** User has been burned twice by full re-runs wiping historical data and by incremental runs replacing the cumulative summary with a single-round view. Both bugs wasted hours of compute and destroyed historical backtest records.

**How to apply:**
- `refresh_and_rank.sh` step [4/6]: detect LAST_TS from most recent summary file, set START_ROUND = last_round + 1. Already fixed in commit `855b6d225`.
- `update_team_analysis.py` `generate_backtest_section()`: merge all summary CSVs, dedup keep-last. Already fixed in commit `2edbee5f9`.
- Before ANY change to backtest code: verify both fixes are still intact.
- Never pass `--start-round 1` to `backtest.py` from any script.
- The cumulative summary table in `afl-backtest-2026.md` must always show ALL rounds from R1 to current.
