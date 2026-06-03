---
name: model-bias-collingwood-2026
description: Collingwood per-team model bias is −0.91 in R13 2026 backtest — systematic over-prediction, below-average pct-within-5
metadata:
  type: project
---

As of `backtest_by_team_20260601_225644.csv`, Collingwood's per-team bias is **−0.91** disposals (model over-predicts Collingwood players) with MAE 3.87 and pct-within-5 of 69.6%.

**Why:** Discovered during WB vs Collingwood R13 2026 brief. The bias magnitude (|0.91| > 0.5) triggers the SCIENTIST REVIEW threshold from the hard rules.

**How to apply:** On every Collingwood brief, pull the current backtest_by_team file and check whether the bias has persisted or resolved. If still > |0.5|, insert a SCIENTIST REVIEW comment with both corrected and uncorrected numbers. Do not silently apply the correction — that is Scientist's call.

See also: [[model-bias-hawthorn]] for a more severe example (−2.64 at R13 2026).
