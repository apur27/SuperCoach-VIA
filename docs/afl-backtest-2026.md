# 2026 backtest results

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BACKTEST-START -->
*Last updated: 2026-05-11 · 8 rounds backtested · auto-generated*

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
| 1 | 230 | 4.89 | 6.17 | 58.7% | 89.6% |
| 2 | 413 | 4.14 | 5.15 | 68.0% | 94.2% |
| 3 | 320 | 4.07 | 5.28 | 69.7% | 94.7% |
| 4 | 319 | 4.15 | 5.31 | 69.9% | 94.0% |
| 5 | 365 | 3.74 | 4.74 | 70.4% | 97.3% |
| 6 | 411 | 3.98 | 5.06 | 71.3% | 95.1% |
| 7 | 410 | 4.04 | 5.14 | 68.5% | 94.6% |
| 8 | 411 | 4.13 | 5.25 | 67.9% | 94.4% |

**Overall (mean across 8 rounds):** MAE 4.14 disposals · 68.1% of predictions within 5 disposals · 94.2% within 10.

> **What to look for:** MAE should stay flat or improve as the season progresses — the model gets more data per player each round. A spike in Round 1 (MAE ~4.9) is normal because many players have no 2026 history yet. If MAE rises sharply mid-season, it usually means an unusual game week (byes, interstate travel, weather).

### How accurate were predictions for the top 30 disposal players?

| # | Player | Team | Avg actual disposals | Avg predicted | Avg error | Rounds |
|--:|--------|------|---------------------:|--------------:|----------:|-------:|
| **1** | **Nick Daicos** | **Collingwood** | **37.0** | **27.5** | **−9.5 ↓** | **6** |
| **2** | **Harry Sheezel** | **North Melbourne** | **32.7** | **27.5** | **−5.2 ↓** | **7** |
| **3** | **Archie Roberts** | **Essendon** | **32.1** | **23.8** | **−8.3 ↓** | **7** |
| 4 | Bailey Smith | Geelong | 32.0 | 26.2 | −5.8 ↓ | 7 |
| **5** | **Lachie Neale** | **Brisbane Lions** | **31.1** | **26.1** | **−5.1 ↓** | **7** |
| **6** | **Clayton Oliver** | **Greater Western Sydney** | **30.9** | **25.2** | **−5.6 ↓** | **7** |
| 7 | Zak Butters | Port Adelaide | 30.7 | 25.3 | −5.4 ↓ | 7 |
| 8 | Lachie Ash | Greater Western Sydney | 30.3 | 25.0 | −5.3 ↓ | 7 |
| 9 | Jack Sinclair | St Kilda | 30.1 | 27.2 | −2.9 ↓ | 7 |
| 10 | Lachie Whitfield | Greater Western Sydney | 30.1 | 25.3 | −4.8 ↓ | 7 |
| 11 | Josh Daicos | Collingwood | 29.4 | 25.2 | −4.2 ↓ | 7 |
| **12** | **Finn Callaghan** | **Greater Western Sydney** | **29.1** | **25.8** | **−3.3 ↓** | **7** |
| 13 | Max Holmes | Geelong | 28.9 | 26.0 | −2.8 ↓ | 7 |
| 14 | Sam Walsh | Carlton | 28.7 | 26.2 | −2.5 ↓ | 7 |
| **15** | **Will Ashcroft** | **Brisbane Lions** | **28.4** | **24.8** | **−3.6 ↓** | **7** |
| **16** | **Marcus Bontempelli** | **Western Bulldogs** | **26.6** | **24.6** | **−1.9 ↓** | **7** |
| 17 | Jai Newcombe | Hawthorn | 26.4 | 23.1 | −3.4 ↓ | 7 |
| 18 | Callum Wilkie | St Kilda | 26.3 | 22.2 | −4.1 ↓ | 7 |
| 19 | Matthew Kennedy | Western Bulldogs | 26.3 | 24.7 | −1.5 ↓ | 7 |
| **20** | **Zach Merrett** | **Essendon** | **26.3** | **24.6** | **−1.6 ↓** | **7** |
| 21 | Nasiah Wanganeen-Milera | St Kilda | 26.1 | 22.3 | −3.9 ↓ | 7 |
| 22 | Christian Petracca | Gold Coast | 26.0 | 24.2 | −1.8 ↓ | 5 |
| 23 | Luke Davies-Uniacke | North Melbourne | 26.0 | 24.4 | −1.6 ↓ | 7 |
| 24 | Caleb Serong | Fremantle | 25.7 | 23.8 | −2.0 ↓ | 7 |
| 25 | Luke Parker | North Melbourne | 25.6 | 23.3 | −2.3 ↓ | 7 |
| 26 | Justin Mcinerney | Sydney | 25.6 | 23.5 | −2.1 ↓ | 7 |
| 27 | Touk Miller | Gold Coast | 25.6 | 25.2 | −0.3 ↓ | 7 |
| 28 | Jayden Short | Richmond | 25.5 | 23.3 | −2.2 ↓ | 6 |
| 29 | Caleb Daniel | North Melbourne | 25.4 | 22.9 | −2.6 ↓ | 7 |
| 30 | Patrick Cripps | Carlton | 25.4 | 21.8 | −3.6 ↓ | 7 |

> **Reading this table:** "Avg error" tells you whether the model systematically misjudges a player. A large positive error (↑) means we over-predicted — the player gets fewer disposals than expected. A large negative error (↓) means we under-predicted — they consistently beat the model. Players with errors above ±6 (bolded) are worth investigating — they may have changed role, had an injury, or are operating in a way the model hasn't caught up with yet.

Full backtest CSVs in `data/prediction/backtest/` — run `backtest.py` to regenerate.
<!-- 2026-BACKTEST-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Brownlow predictor](afl-brownlow-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md)
