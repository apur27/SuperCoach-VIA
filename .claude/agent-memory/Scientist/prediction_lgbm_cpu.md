---
name: LightGBM CPU vs CUDA in prediction.py
description: prediction.py uses LGBMRegressor with device parameter; this env has no NVIDIA GPU, so device must stay 'cpu'
type: project
---

`prediction.py` (method `tune_lgbm_gpu`, ~line 497-549) used to hard-code `device='cuda'` in two spots (the Optuna trial params and the final pipeline). On this host `nvidia-smi` fails (no CUDA driver / no GPU), so LightGBM aborts immediately with `LightGBMError: [CUDA] no CUDA-capable device is detected`.

**Why:** the previous Apr 30 run somehow passed despite the same code being in git. Possible explanations: a different machine (with GPU), a different lightgbm version that silently fell back, or the run simply happened on a host that did have CUDA. Either way, the current env requires CPU.

**How to apply:**
- If `prediction.py` (or any LGBM call in the repo) is reverted to `device='cuda'` and prediction/backtest aborts with `LightGBMError: [CUDA] no CUDA-capable device is detected`, change both `device='cuda'` lines to `device='cpu'` (Optuna objective params dict + final LGBMRegressor in the Pipeline).
- Methodology impact: none meaningful. LightGBM CPU and GPU produce numerically near-identical fits for the same seed; only speed differs. Backtest comparisons against prior GPU-era runs remain valid.
- Speed impact: backtest takes ~30-35 min per round on CPU (vs ~5-10 min on GPU). A full 10-round walk-forward run is ~5-6 hours wall time.
- The method is still named `tune_lgbm_gpu` for backwards compat — don't rename, just keep the docstring honest about CPU.
