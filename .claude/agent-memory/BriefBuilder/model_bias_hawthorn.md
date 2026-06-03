---
name: model-bias-hawthorn
description: Hawthorn's per-team model bias is material — triggers SCIENTIST REVIEW on every brief
metadata:
  type: project
---

As of the R13 2026 backtest (`backtest_by_team_20260601_225644.csv`), Hawthorn's per-team bias = **−2.64 disposals/player** (model under-predicts Hawthorn players by 2.64 on average).

**Why:** The |>0.5| threshold in the brief spec triggers a SCIENTIST REVIEW comment. Hawthorn's bias substantially exceeds this. Corrected predictions for top-5 Hawthorn players would add ~2.64 to each.

**How to apply:** In every brief featuring Hawthorn, include a SCIENTIST REVIEW block noting the bias and showing both corrected and un-corrected numbers. Do not silently apply the correction. Let Scientist decide. This was applied in St Kilda vs Hawthorn R13 2026 without pushback.

Note: this bias figure may change as more rounds are backtested. Re-check the most recent backtest_by_team CSV at each brief run.
