# 2026 backtest results

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BACKTEST-START -->
*Last updated: 2026-05-12 · 10 rounds backtested · auto-generated*

### What is a backtest?

Before we trust our predictions for next week, we need to check how well the model has done on rounds that are already finished — rounds where we know the real answer. A backtest does exactly that: for each completed round, the model is trained on all data **before** that round, then asked to predict it. We then compare prediction to reality.

This is the honest test. The model never gets to see the round it's predicting.

### What the numbers mean (in plain English)

| Term | What it actually means | Good or bad? |
|------|----------------------|--------------|
| **MAE** (Mean Absolute Error) | On average, our predictions were off by this many disposals. If MAE = 4.1, we were within ±4 disposals on a typical player. | Lower = better |
| **RMSE** (Root Mean Square Error) | Similar to MAE but punishes big blunders harder — if we say 30 and the player gets 10, RMSE notices that more than MAE does. | Lower = better |
| **Median error** | The middle prediction error — half of players were predicted better than this, half worse. More robust than MAE because it ignores extreme outliers. | Lower = better |
| **Bias** | Whether the model systematically over- or under-predicts. A bias of −0.7 means we tend to predict 0.7 disposals too high. A bias near 0 is ideal. | Near 0 = better |
| **Within 5 disposals** | The % of predictions that landed within 5 of the actual number (e.g. predicted 24, actual was 22 — that counts). This is the most intuitive accuracy measure for SuperCoach. | Higher = better |
| **Within 10 disposals** | Same but with a wider 10-disposal window. This is nearly always above 90%. | Higher = better |

**Rule of thumb:** an MAE around 4–5 disposals is competitive for AFL prediction — the game has too many random events (injuries, umpire decisions, tactic changes) for any model to do much better. "Within 5 disposals" above 65% is good; above 70% is strong.

![Prediction accuracy by round](../assets/charts/backtest_accuracy_2026.png)

### Round-by-round accuracy

#### Per-round backtest summary — 2026

| Round | Players | MAE | RMSE | Within 5 disp | Within 10 disp |
|------:|--------:|----:|-----:|--------------:|---------------:|
| 1 | 230 | 4.83 | 6.10 | 60.4% | 92.6% |
| 2 | 413 | 4.11 | 5.11 | 72.2% | 95.9% |
| 3 | 320 | 4.07 | 5.28 | 74.7% | 95.9% |
| 4 | 319 | 4.15 | 5.32 | 72.4% | 94.7% |
| 5 | 365 | 3.73 | 4.74 | 75.3% | 97.5% |
| 6 | 411 | 3.98 | 5.05 | 74.9% | 95.9% |
| 7 | 410 | 4.05 | 5.15 | 72.0% | 95.6% |
| 8 | 411 | 4.14 | 5.27 | 73.2% | 95.4% |
| 9 | 410 | 3.79 | 4.74 | 74.9% | 98.3% |
| 10 | 412 | 4.31 | 5.50 | 68.2% | 94.9% |

**Overall (mean across 10 rounds):** MAE 4.12 disposals · 71.8% of predictions within 5 disposals · 95.7% within 10.

> **What to look for:** MAE should stay flat or improve as the season progresses — the model gets more data per player each round. A spike in Round 1 (MAE ~4.9) is normal because many players have no 2026 history yet. If MAE rises sharply mid-season, it usually means an unusual game week (byes, interstate travel, weather).

### How accurate were predictions for the top 30 disposal players?

| # | Player | Team | Avg actual disposals | Avg predicted | Avg error | Rounds |
|--:|--------|------|---------------------:|--------------:|----------:|-------:|
| **1** | **Nick Daicos** | **Collingwood** | **35.6** | **27.8** | **−7.9 ↓** | **8** |
| **2** | **Archie Roberts** | **Essendon** | **32.7** | **23.7** | **−9.0 ↓** | **9** |
| **3** | **Bailey Smith** | **Geelong** | **32.2** | **26.4** | **−5.8 ↓** | **9** |
| **4** | **Harry Sheezel** | **North Melbourne** | **31.1** | **26.6** | **−4.6 ↓** | **9** |
| **5** | **Lachie Neale** | **Brisbane Lions** | **30.9** | **26.1** | **−4.8 ↓** | **9** |
| 6 | Lachie Whitfield | Greater Western Sydney | 30.7 | 25.9 | −4.8 ↓ | 9 |
| 7 | Zak Butters | Port Adelaide | 30.6 | 25.7 | −4.9 ↓ | 9 |
| 8 | Clayton Oliver | Greater Western Sydney | 30.4 | 25.4 | −5.0 ↓ | 9 |
| 9 | Jack Sinclair | St Kilda | 29.4 | 27.1 | −2.3 ↓ | 9 |
| 10 | Lachie Ash | Greater Western Sydney | 29.3 | 25.7 | −3.7 ↓ | 9 |
| 11 | Max Holmes | Geelong | 29.3 | 26.1 | −3.2 ↓ | 9 |
| 12 | Finn Callaghan | Greater Western Sydney | 29.1 | 26.3 | −2.8 ↓ | 9 |
| 13 | Sam Walsh | Carlton | 28.2 | 26.3 | −1.9 ↓ | 9 |
| 14 | Josh Daicos | Collingwood | 28.2 | 25.4 | −2.8 ↓ | 9 |
| **15** | **Will Ashcroft** | **Brisbane Lions** | **27.8** | **25.0** | **−2.8 ↓** | **9** |
| 16 | Marcus Bontempelli | Western Bulldogs | 27.7 | 25.0 | −2.7 ↓ | 9 |
| **17** | **Zach Merrett** | **Essendon** | **27.0** | **24.9** | **−2.1 ↓** | **9** |
| 18 | Nasiah Wanganeen-Milera | St Kilda | 26.8 | 22.8 | −4.0 ↓ | 8 |
| 19 | Luke Davies-Uniacke | North Melbourne | 26.7 | 23.3 | −3.3 ↓ | 9 |
| 20 | Matthew Kennedy | Western Bulldogs | 26.6 | 24.4 | −2.1 ↓ | 9 |
| **21** | **Wayne Milera** | **Adelaide** | **26.4** | **22.0** | **−4.4 ↓** | **9** |
| 22 | Jai Newcombe | Hawthorn | 26.4 | 23.0 | −3.4 ↓ | 9 |
| 23 | John Noble | Gold Coast | 26.3 | 23.7 | −2.7 ↓ | 9 |
| 24 | Noah Anderson | Gold Coast | 26.2 | 23.5 | −2.8 ↓ | 8 |
| 25 | Ryley Sanders | Western Bulldogs | 26.0 | 21.6 | −4.4 ↓ | 8 |
| 26 | Christian Petracca | Gold Coast | 26.0 | 24.0 | −2.0 ↓ | 7 |
| **27** | **Jake Bowey** | **Melbourne** | **26.0** | **18.0** | **−8.0 ↓** | **1** |
| 28 | Touk Miller | Gold Coast | 25.9 | 25.7 | −0.2 ↓ | 9 |
| **29** | **Dan Houston** | **Collingwood** | **25.9** | **23.2** | **−2.7 ↓** | **9** |
| 30 | Isaac Heeney | Sydney | 25.7 | 20.4 | −5.3 ↓ | 7 |

> **Reading this table:** "Avg error" tells you whether the model systematically misjudges a player. A large positive error (↑) means we over-predicted — the player gets fewer disposals than expected. A large negative error (↓) means we under-predicted — they consistently beat the model. Players with errors above ±6 (bolded) are worth investigating — they may have changed role, had an injury, or are operating in a way the model hasn't caught up with yet.

Full backtest CSVs in `data/prediction/backtest/` — run `backtest.py` to regenerate.
<!-- 2026-BACKTEST-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Brownlow predictor](afl-brownlow-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md)
