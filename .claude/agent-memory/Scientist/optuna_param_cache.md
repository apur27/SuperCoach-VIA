---
name: optuna-param-cache
description: prediction.py caches Optuna best-params; how it invalidates and how to force a re-tune
metadata:
  type: project
---

`supercoach/prediction.py` caches HGB + LGBM Optuna best-params (F6, committed
2026-07-10, commit 7522b7c6f).

- Cache file: `data/prediction/optuna_best_params.json`, keys `hgb` / `lgbm`,
  each with `params`, `n_training_rows`, `tuned_at`, `optuna_version`.
- Cache HIT (skip both studies, ~20 min saved) needs BOTH: row growth <5%
  (`abs(cur-cached)/cached < 0.05`) AND age <28 days. Else re-tune + re-save.
- Fail-open: missing/corrupt file → cache miss → normal tune.
- **To force a re-tune: delete the cache file.** There is deliberately no
  `--force-retune` flag.
- Only affects the weekly prediction run (`prediction.py`). `backtest.py` is
  UNTOUCHED — its ~5-6h runtime is unchanged (see [[prediction_lgbm_cpu]]).

**Why:** weekly cycle re-tuned ~100 trials on a corpus growing ~0.5%/week where
params are stable week to week; monthly re-tune cadence caps staleness risk.

**How to apply:** if a future "params didn't change" question comes up, check
this cache first. If validating a param change, delete the cache so the next
run re-tunes from scratch.
