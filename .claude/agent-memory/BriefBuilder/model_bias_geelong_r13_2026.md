---
name: model-bias-geelong-r13-2026
description: Geelong per-team model bias in R13 2026 backtest — under-prediction at +1.17 disposals, triggers SCIENTIST REVIEW threshold
metadata:
  type: project
---

As of `backtest_by_team_20260601_225644.csv`, Geelong's per-team bias for R13 2026 is **+1.17 disposals** (model under-predicts Geelong players). This exceeds the |0.5| SCIENTIST REVIEW threshold.

Carlton's R13 2026 bias is **-0.52** (slight over-prediction, just at threshold).

Most impacted tracked player: Bailey Smith, predicted 28 vs season mean 32.6 (a -4.6 gap); bias-corrected estimate ~29.2.

**Why:** The Geelong positive bias (under-prediction) is consistent with their strong 9W-3L season record; the model may not fully capture their form momentum.

**How to apply:** Any future Geelong brief — check current backtest file before assuming bias magnitude. The +1.17 figure is R13-specific; bias drifts round to round.

See also: [[model-bias-hawthorn]] for the pattern of severe team-specific bias warranting SCIENTIST REVIEW.
