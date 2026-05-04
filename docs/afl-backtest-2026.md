# 2026 backtest results

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BACKTEST-START -->
Model accuracy across 8 rounds of the 2026 season.

![Prediction accuracy by round — 2026 season](../assets/charts/backtest_accuracy_2026.png)

#### Per-round backtest summary — 2026

| Round | Players | MAE | RMSE | Within 5 disp | Within 10 disp |
|------:|--------:|----:|-----:|--------------:|---------------:|
| 1 | 230 | 4.89 | 6.17 | 58.7% | 89.6% |
| 2 | 413 | 4.14 | 5.15 | 68.0% | 94.2% |
| 3 | 320 | 4.07 | 5.28 | 69.7% | 94.7% |
| 4 | 319 | 4.15 | 5.31 | 69.9% | 94.0% |
| 5 | 365 | 3.74 | 4.74 | 70.4% | 97.3% |
| 6 | 411 | 3.98 | 5.06 | 71.3% | 95.1% |
| 7 | 410 | 4.04 | 5.14 | 68.5% | 94.6% |
| 8 | 411 | 4.13 | 5.25 | 67.9% | 94.4% |

**Overall (mean across 8 rounds):** MAE 4.14 disposals · 68.1% of predictions within 5 disposals · 94.2% within 10.

Full backtest CSVs in `data/prediction/backtest/` — run `backtest.py` to regenerate.
<!-- 2026-BACKTEST-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Brownlow predictor](afl-brownlow-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md)
