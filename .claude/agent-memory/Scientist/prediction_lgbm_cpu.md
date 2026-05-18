---
name: LightGBM device selection in prediction.py
description: prediction.py auto-detects LightGBM GPU support via module-level probe; falls back to CPU on hosts without a GPU-enabled LightGBM build
type: project
---

`prediction.py` selects the LightGBM device at module load via `_detect_lgbm_device()` (defined near the top of the file). The helper attempts a tiny `LGBMRegressor(device='gpu', ...).fit(...)` on dummy data; if it raises (no GPU-enabled LightGBM build, no NVIDIA GPU, no CUDA driver, etc.) it returns `'cpu'`, otherwise `'gpu'`. The result is cached in the module-level constant `LGBM_DEVICE` and threaded through both `tune_lgbm_gpu` call sites:
- The Optuna trial params dict (`'device': LGBM_DEVICE`)
- The final `LGBMRegressor(**study.best_params, device=LGBM_DEVICE, ...)` in the Pipeline

**Why:** earlier this branch hard-coded `device='cpu'`, which disabled GPU on every CUDA-capable host even though the repo already has `prediction_cpu.py` as the explicit CPU fallback. The probe-based detection means a single entry point works on both classes of host without per-machine config. The probe redirects OS-level fd 2 to /dev/null during the test so LightGBM's C-level `[Fatal] GPU Tree Learner was not enabled in this build` line doesn't leak into the user's stderr on CPU-only hosts.

**How to apply:**
- On this dev host the venv's LightGBM is not built with `-DUSE_GPU=1`, so `LGBM_DEVICE` resolves to `'cpu'`. Backtest wall time on CPU is ~5-6 hours for the full walk-forward (vs ~30-60 min if a GPU-enabled LightGBM is available).
- If a future LightGBM upgrade or rebuild flips GPU support on, no code change is needed — the probe will pick it up automatically. The `Tuning LGBM (CPU)` vs `Tuning LGBM (GPU)` print on `tune_lgbm_gpu` invocation tells you which path is live.
- Methodology impact: none meaningful. LightGBM CPU and GPU produce numerically near-identical fits for the same seed; only speed differs. Backtest comparisons across hosts remain valid.
- The method is still named `tune_lgbm_gpu` for backwards compat — don't rename, the docstring documents the conditional behaviour.
- `prediction_cpu.py` does not set `device` at all (defaults to CPU) and is intentionally CPU-only — don't touch it when fixing GPU-related issues in `prediction.py`.
