---
name: backtest-partial-regen-stale-blocks
description: A weekly refresh can update SOME blocks of docs/afl-backtest-2026.md to the new round while leaving others (cumulative/team-bias/misses) pooled at the prior round — always recompute the pooled cumulative figures from the FULL current keep-last artifact set, never assume "the round-by-round table changed" means the whole doc changed.
metadata:
  type: project
---

## 2026-07-27 (R21 landing) — confirmed partial-regeneration defect

`docs/afl-backtest-2026.md` is composed of several independently-generated blocks
(`2026-BACKTEST` per-round table + top-30 table via `update_team_analysis.py`;
`CUMULATIVE` / `TEAMBIAS` / `MISSES` via `scripts/update_eval_surface.sh`). These
generators do **not** always run/land together. On the R21 refresh:

- Per-round summary table (top of doc) WAS updated: 21 rows, header says "21 rounds
  backtested", pre-registered-threshold sentence ("21 of 21 rounds ... 320 outright
  misses") computed correctly over all 21 rounds.
- Top-30 disposal-winners table WAS updated: all 30 rows' avg-actual/avg-predicted/
  avg-error/rounds-count reproduce exactly against the full 21-round keep-last pool
  (e.g. Nick Daicos 35.1/28.6/-6.5/18 rounds — confirmed via
  `sort_values` keep-last dedup across `backtest_summary_*.csv` then pooling
  `prediction_vs_actual_round_*_2026_*.csv`).
- `CUMULATIVE-START/END` block, `TEAMBIAS-START/END` block, and `MISSES-START/END`
  block were **NOT** updated — all three still pool exactly the OLD 20-round basis
  (n=7,153, matching sum of the 18 team-bias `n` values and the previous week's
  misses table, which stops at Round 20 with no Round 21 row at all). The true
  current 21-round pooled figures are n=7,524, MAE 3.959, RMSE 5.100, bias -0.134
  (vs stale -0.110), median round MAE 3.98 (vs stale 3.97).
- The "Read:" callout under the cumulative table ("under-predicted by **2.96**
  disposals on average... against a population figure of **-0.110**") also reads
  stale — recomputing mean of the top-30 table's OWN (already-updated-to-R21)
  `Avg error` column gives -2.92 (rounded-then-averaged -2.927), not -2.96. The
  callout was written against last week's top-30 table and never rechecked against
  this week's (already-correct) one sitting right above it.

**Why this matters:** the doc's own Reproducibility section declares a hard
three-way reconciliation invariant (pooled per-player rows == summed summary
`n_players` == summed per-team `n`) and claims `update_eval_surface.sh` "refuses to
write when that reconciliation fails" — but that invariant only holds **within** a
single vintage. It says nothing about cross-block staleness when one generator
(`update_team_analysis.py`) runs and lands but a sibling generator
(`update_eval_surface.sh`) doesn't (or runs against a pre-R21 snapshot). A doc can
pass its own internal reconciliation check and still be stale relative to what's on
disk right now.

**How to apply:** every time this doc is re-verified, do NOT trust "the header/
per-round table already shows round N" as proof the rest of the doc is current.
Always independently recompute the CUMULATIVE/TEAMBIAS/MISSES-block figures from
the FULL current keep-last artifact set (glob `backtest_summary_*.csv` fresh, don't
reuse a vintage map from a prior pass) and diff against what's printed. A doc that
looks "half-refreshed" (top table new, cumulative old) is the norm failure mode
here, not an edge case — check for it every pass, specifically:
1. Does `sum(team-bias n)` == `sum(dedup summary n_players)` == pooled per-player
   row count computed independently, **using every round currently in
   `data/prediction/backtest/backtest_summary_*.csv`** (not just the rounds the
   cumulative table claims)?
2. Does the misses table have one row per round shown in the per-round summary
   table above it? A missing trailing round is silent staleness, not a formatting
   choice.
3. Recompute any post-table "Read:"/interpretive derived stat directly from the
   table's own current rows — don't assume it was updated in lockstep with the
   table.

See also [[project_backtest_doc_verification_gotchas]],
[[project_backtest_tiebreak_and_readback_trap]], and
[[project_backtest_reproduction_recipes]] for the keep-last vintage-selection
mechanics reused here.
