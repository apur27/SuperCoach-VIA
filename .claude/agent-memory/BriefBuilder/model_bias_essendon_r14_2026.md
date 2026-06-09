---
name: model-bias-essendon-r14-2026
description: Essendon per-team model bias in R13 2026 backtest — −0.86, triggers SCIENTIST REVIEW
metadata:
  type: project
---

Essendon per-team bias = −0.86 in R13 2026 backtest (`backtest_by_team_20260601_225644.csv`).

**Why:** Exceeds the |0.5| SCIENTIST REVIEW threshold. Model is systematically under-predicting Essendon players by ~0.86 disposals per player. If +0.86 correction applied: Roberts lifts to ~30.9, Merrett to ~27.9, Parish to ~26.9 at R14 prediction levels.

**How to apply:** Every Essendon brief using the R13 2026 backtest file must include a SCIENTIST REVIEW comment noting the −0.86 bias. Bias may change in later backtest files — always verify from the current `backtest_by_team_<ts>.csv` file.

Carlton bias at same timestamp = −0.52 (borderline; also triggers SCIENTIST REVIEW per |0.5| rule).

Related: [[model-bias-collingwood-2026]], [[model-bias-geelong-r13-2026]]
