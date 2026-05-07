---
name: prediction-top-end-compression
description: Disposal predictor compresses top-end output via log1p+expm1 round trip and L1 LGBM loss; both removed and post-hoc OOF linear calibration added on 2026-04-30
type: project
---

The 2026 R1-R8 backtest of `prediction.py` revealed two compounding sources of top-end compression on disposal predictions:

1. **`np.log1p(y)` target transform paired with `np.expm1` on output.** HGB's Poisson loss already carries an internal log link, so log1p was double-compressing. Combined with tree regression-to-mean, max prediction was 28 vs max actual 43. Players with 30+ disposal games were under-predicted by 10-19.
2. **LGBM `objective='regression_l1'`** predicts the median, which on right-skewed disposals is below the mean. Compounded the under-bias.

**Why:** raw model bias was -1.32 disposals across n=2879 across all 8 rounds. The bias was driven heavily by elite midfielders (Daicos, Neale, Roberts, Smith, Sheezel, Butters) who have right-tailed game profiles.

**How to apply:** When evaluating a future change to `prediction.py`,
- Train on raw disposals (do NOT re-introduce log1p).
- Keep LGBM at `objective='regression'` (L2/mean), not `regression_l1`.
- Calibration is fit per-run via OOF predictions in `_fit_calibration()`, applied at `predict_current_season_disposals()`. Calibration is bounded to slope in [0.5, 2.0] and |intercept| <= 20 - outside, falls back to identity.
- The clip is now [1, 55]. Realistic disposal range is observed up to ~50; lower bound 1 because 0 implies DNP which is upstream-modelled.
- If max prediction reverts to <30 in a future backtest, suspect either compression returning (transform/loss) or calibration failing the bounds check (look for "Calibration out of bounds" warning).
