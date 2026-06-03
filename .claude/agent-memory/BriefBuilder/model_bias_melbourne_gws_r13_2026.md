---
name: model-bias-melbourne-gws-r13-2026
description: Melbourne +0.74 and GWS -0.65 per-team biases in R13 2026 backtest; both exceed |0.5| threshold
metadata:
  type: project
---

From `backtest_by_team_20260601_225644.csv` for R13 2026:
- Melbourne bias: **+0.74** (model under-predicts Melbourne players)
- GWS bias: **-0.65** (model over-predicts GWS players)

Both exceed the |0.5| SCIENTIST REVIEW threshold. The biases partially offset each other when comparing the two teams — Melbourne predictions should be nudged slightly up, GWS slightly down.

**Why:** Systematic per-team biases captured in R13 backtest (as of 2026-06-01). These may change in future rounds as more data is added.

**How to apply:** Flag both biases for SCIENTIST REVIEW in any Melbourne vs GWS brief using R13 2026 backtest data. If a future round's backtest shows different values, update this memory.
