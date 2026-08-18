# 2026 Brownlow Medal predictor

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BROWNLOW-PREDICTOR-START -->
The **Brownlow Medal** is the AFL's individual award for the "fairest and best" player, voted on by the on-field umpires with a 3-2-1 split per game. It is impossible to predict actual votes without modelling umpire behaviour, but we *can* build a defensible **statistical proxy** — a composite score over the stats that historically correlate with vote-earning. The weights below were validated against every player-game from 2010-2025 (n=145,150) where actual `brownlow_votes` are recorded — the top 1% of proxy games captured ~70% of vote-earning performances. Players need at least 3 games played to be ranked. Suspended players are not penalised in the proxy — this is a stat-profile model, not a vote forecaster — but because any in-season suspension makes a player ineligible to win the actual Brownlow Medal, suspended players are flagged inline in the table below so the distinction stays visible.

**Composite formula** (z-scored across all eligible players, summed with weights): `0.30 × disposals + 0.22 × clearances + 0.18 × contested-poss + 0.15 × effective-disposals + 0.15 × goals`. Effective disposals are approximated as `disposals - clangers` because the raw data does not carry a true effective-disposal column. Goals are weighted higher than the conventional midfielder-only template (15% vs the ~5% common in pure-midfielder proxies) because that materially improves correlation with actual historical Brownlow votes.

![2026 Brownlow predictor](../assets/charts/brownlow_predictor_2026.png)

#### Top 15 Brownlow proxy candidates — 2026 season-to-date (after Round 24)

| Rank | Player | Team | Games | Disp/g | Clear/g | CP/g | Goals/g | Proxy | Proj. votes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Nick Daicos | Collingwood | 21 | 35.3 | 5.8 | 10.7 | 1.10 | +2.76 | +60.6 |
| 2 | Clayton Oliver | Greater Western Sydney | 22 | 30.5 | 7.4 | 15.1 | 0.18 | +2.70 | +59.3 |
| 3 | Lachie Neale | Brisbane Lions | 22 | 29.8 | 6.8 | 13.1 | 0.23 | +2.48 | +54.5 |
| 4 | Patrick Cripps | Carlton | 22 | 26.6 | 7.5 | 14.8 | 0.59 | +2.47 | +54.3 |
| 5 | Isaac Heeney | Sydney | 20 | 26.9 | 6.2 | 13.1 | 1.60 | +2.47 | +54.2 |
| 6 | Bailey Smith | Geelong | 21 | 32.3 | 5.6 | 11.4 | 0.38 | +2.38 | +52.3 |
| 7 | Jai Newcombe | Hawthorn | 22 | 26.3 | 7.7 | 13.3 | 0.32 | +2.30 | +50.6 |
| 8 | Zak Butters | Port Adelaide | 17 | 29.8 | 6.1 | 12.2 | 0.29 | +2.29 | +50.3 |
| 9 | Matt Rowell | Gold Coast | 17 | 26.1 | 7.4 | 13.8 | 0.18 | +2.23 | +49.0 |
| 10 | Marcus Bontempelli | Western Bulldogs | 21 | 26.3 | 5.8 | 11.2 | 1.33 | +2.16 | +47.6 |
| 11 | Harley Reid | West Coast | 22 | 24.8 | 6.7 | 13.4 | 0.64 | +2.06 | +45.4 |
| 12 | Will Ashcroft | Brisbane Lions | 22 | 28.9 | 5.4 | 9.6 | 0.68 | +2.04 | +45.0 |
| 13 | Harry Sheezel | North Melbourne | 22 | 30.5 | 4.5 | 9.8 | 0.55 | +2.03 | +44.7 |
| 14 | Tim Taranto | Richmond | 20 | 24.2 | 6.2 | 12.0 | 0.70 | +1.96 | +43.2 |
| 15 | Ed Richards | Western Bulldogs | 21 | 25.8 | 5.9 | 10.3 | 0.86 | +1.95 | +42.9 |

On the proxy, **Nick Daicos** (Collingwood) leads the field — built on 35.3 disposals/g, 1.1 goals/g across 21 games. The composite score (+2.76) sits 0.06 clear of second place. **Clayton Oliver** (Greater Western Sydney) is the closest challenger at +2.70, with 30.5 disposals/g and 7.4 clearances/g. The proxy is a statistical model, not actual umpire votes — it captures the stat-profile umpires *historically* reward, but it cannot model individual game narrative, suspension impact or the umpire panel's eye for a defensive midfielder.
<!-- 2026-BROWNLOW-PREDICTOR-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
