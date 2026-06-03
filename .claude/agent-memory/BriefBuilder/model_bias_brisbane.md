---
name: model-bias-brisbane
description: Brisbane Lions per-team model bias — over-prediction flag for R13 2026 and future briefs
metadata:
  type: project
---

Brisbane Lions per-team bias in `backtest_by_team_20260601_225644.csv` = **+2.65 disposals** (predicted > actual). MAE = 4.22. This exceeds the |0.5| threshold and triggers a SCIENTIST REVIEW comment on every Brisbane Lions brief.

**Why:** The model systematically over-predicts Brisbane Lions player disposals, possibly due to role volatility, tagging patterns, or model feature drift from the 2024 premiership squad context.

**How to apply:** On every brief containing Brisbane Lions players, add a SCIENTIST REVIEW comment in the Model Context section noting the +2.65 bias. Present predictions as-is from CSV; do not apply correction unilaterally. Scientist decides whether §12.5 correction applies. Compare against Fremantle (−0.30, well-calibrated as of R13 2026).

Related: [[model-bias-sydney-r13-2026]], [[model-bias-hawthorn]]
