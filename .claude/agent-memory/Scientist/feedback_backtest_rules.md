---
name: feedback-backtest-rules
description: Absolute rules for backtest — never re-run all rounds, always preserve existing results, only run the missing round
metadata:
  type: feedback
---

RULE 1 — INCREMENTAL ONLY: Only ever backtest the round(s) that are missing. Never pass `--start-round 1`. Detect the last completed round from `data/prediction/backtest/backtest_summary_*.csv` and start from the next one.

RULE 2 — PRESERVE ALL RESULTS: Existing backtest results are sacred. `generate_backtest_section()` must merge ALL `backtest_summary_*.csv` files (oldest-first, dedup by year+round keeping latest) so cumulative doc always shows R1 through current round — never just the latest run.

RULE 3 — BY-ARCHIVE MODE (added 2026-07-10, commit `16384f050`): the weekly incremental backtest now scores the ALREADY-PUBLISHED forward CSV via `backtest.py --from-csv <path>` — NO retrain. This fixed a namespace-pollution bug: the old path re-ran `predictor.run()`, which wrote `next_round_*` into the LIVE prediction dir; since backtest runs after forward (step 4 > step 3) it won mtime-newest resolution and downstream shipped the backtest artifact, not the forward prediction. `--from-csv` writes ONLY to `data/prediction/backtest/`.
- **Round offset (load-bearing):** backtest scores the JUST-COMPLETED round (actuals exist); forward predicts the UPCOMING round. They differ by exactly ONE round. The archived CSV to score for round N is the `next_round_N` written in a PRIOR cycle — NOT this cycle's step-3 output (which predicts N+1). `refresh_and_rank.sh` loops `START_ROUND..(UPCOMING-1)` and resolves each round's archived CSV.
- **Semantic shift:** archive scores the published set (~320/round, n-with-actual ~284) vs retrain's broader synthetic set (~412). `update_eval_surface.sh` weights MAE by `n_players`, so headline MAE + total-N shift when archive rounds land. Expected, not a regression.

FINDING — MAE IS CLEAN (not in-sample): training filter is `df['year'] < self.target_year` (`prediction.py:493`), which drops ALL same-year rows for the 2026 within-season backtest regardless of `cutoff_round`. The `rn == cutoff_round` rows kept in `LeakProofPredictor.load_and_prepare_data` are used only for lagged prediction features, never as training targets. The `dropped 0 future rows` log is a correct no-op (round N+1 unplayed), not a leak.

**Why:** User has been burned twice by full re-runs wiping historical data and by incremental runs replacing the cumulative summary with a single-round view. Both bugs wasted hours of compute and destroyed historical backtest records.

**How to apply:**
- `refresh_and_rank.sh` step [4/6]: detect LAST_TS from most recent summary file, set START_ROUND = last_round + 1. Already fixed in commit `855b6d225`.
- `update_team_analysis.py` `generate_backtest_section()`: merge all summary CSVs, dedup keep-last. Already fixed in commit `2edbee5f9`.
- Before ANY change to backtest code: verify both fixes are still intact.
- Never pass `--start-round 1` to `backtest.py` from any script.
- The cumulative summary table in `afl-backtest-2026.md` must always show ALL rounds from R1 to current.
