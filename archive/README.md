# Archive — historical scripts kept for reference, not part of the active pipeline

Nothing here is imported by the pipeline or exercised by the test suite. These files
record how earlier versions worked; they are not maintained. Treat them as read-only
history — if you need this behaviour, port it into `scripts/` with tests rather than
reviving the file in place.

| Script | What it was |
|---|---|
| `analysis.py` | Early exploratory analysis over the player CSVs, superseded by the feature-engineering and walk-forward backtest paths now in `scripts/` and `backtest.py`. |
| `helper_functions.py` | Shared helpers for that exploratory work. The surviving equivalents live in `config.py` and the `supercoach/` package. |
| `testGPU.py` | One-off probe for CUDA availability, from when the ensemble's LightGBM component was first given a GPU path. `supercoach/prediction.py` now handles the CPU fallback itself. |
