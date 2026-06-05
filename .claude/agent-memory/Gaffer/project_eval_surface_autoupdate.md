---
name: eval-surface-autoupdate
description: scripts/update_eval_surface.sh auto-refreshes README eval section + banner.svg from backtest CSVs; the CSVs are split across runs and must be merged
metadata:
  type: project
---

`scripts/update_eval_surface.sh` (wired into `weekly_refresh.sh` as Phase 1b,
after backtest, before the Phase 4 commit) refreshes the README "Eval results —
current" table + 2 prose figures and `docs/banner.svg` (pills, Band 1 player
count, Band 2 numbers/round label) from the backtest CSVs. Idempotent, targeted
regex replacements, never touches the news block.

**Why / the load-bearing gotcha:** `data/prediction/backtest/backtest_summary_*.csv`
and `backtest_by_team_*.csv` are NOT cumulative. Early runs (through R10) wrote
all rounds 1..N per file; recent runs (R11, R12, R13) write ONLY the single
newest round. So the latest summary CSV has just 1 row. The full 13-round series
exists only by MERGING ALL summary CSVs oldest-first and deduping by (year,round)
keeping last — the exact logic `update_team_analysis.py` (~line 4158) uses to
build `docs/afl-backtest-2026.md`. Same merge+dedupe by (year,round,team) for the
season team-bias from by_team CSVs.

**How to apply:** Never compute cumulative/season backtest figures from a single
CSV — always merge across all runs. If a future cycle needs season-level eval
numbers, reuse this merge pattern (it reproduces player-rounds 4,806, weighted
MAE 4.019, Sydney -0.73 / Richmond +0.57 exactly). Headline row convention is
player-weighted throughout; named extreme rounds are recomputed each run
(argmin/argmax) so labels self-correct. This is a deterministic re-derivation of
Scientist-verified numbers, not Gaffer authoring [data] — stays inside the
truth boundary. Gate note: README is exempt from the council-stamp pre-commit
gate (see [[feedback_council_stamp_gate_scope]]).
