---
name: model-bias-sydney-r13-2026
description: Sydney per-team model bias in R13 2026 was −3.00 disposals (MAE 5.78, only 47.8% within 5) — major over-prediction flag for Sydney players
metadata:
  type: project
---

In Round 13 2026, the model severely over-predicted Sydney player disposals: bias −3.00, MAE 5.78, only 47.8% within 5 disposals. This is the largest team-level bias observed so far.

**Why:** Likely context-specific (opponent, venue, game-state) but could reflect a structural Sydney bias in the model. The backtest as of 20260601 captures the R13 actuals.

**How to apply:** On any future Sydney brief, check whether per-team bias in the most recent backtest for Sydney shows a persistent negative (over-prediction) pattern. If it does, raise SCIENTIST REVIEW with un-corrected and corrected numbers.

Related: [[model-bias-hawthorn]]
