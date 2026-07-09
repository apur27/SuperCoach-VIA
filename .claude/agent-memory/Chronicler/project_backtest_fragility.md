---
name: project-backtest-fragility
description: The eval/backtest leg is the pipeline's fragile point — non-resumable Optuna runs + a date stamp that decouples from content round
metadata:
  type: project
---

The predictive-evaluation leg is the weakest link in the weekly refresh, distinct from the (robust) documentary/HOF leg.

**Two coupled failure modes surfaced 2026-07-07 (weekly-r19):**
1. `backtest.py`'s Optuna study is **not resumable** — a mid-run kill (~trial 48) loses all output for that round, forcing manual recovery. No SQLite storage / `--resume`.
2. `afl-backtest-2026.md` stamps "Last updated · N rounds backtested" from **wall-clock date, not max round in the backtest CSV**. Result: a 2026-07-07 date shipped over R17 content — freshness the data didn't have.

**Why:** these together mean an interrupted cycle silently ships a stale eval surface while the R19 predictions look freshly validated. The one artifact that proves prediction trustworthiness is the one most likely to be missing.

**How to apply:** in Pipeline Health, always check `max(round)` in the latest `backtest_summary_*.csv` against `max(round)` in `matches_2026.csv` — if they diverge, the eval surface is frozen and the date stamp is lying. My standing #1/#2 recommendations (honest freshness stamp + Optuna persistence) address this; don't re-surface CR-1 (round-detection, CLOSED 2026-07-07 commit 12fa8202d) or treat it as open. See [[project-baselines]].
