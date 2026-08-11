# 2026 Brownlow Medal predictor

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BROWNLOW-PREDICTOR-START -->
The **Brownlow Medal** is the AFL's individual award for the "fairest and best" player, voted on by the on-field umpires with a 3-2-1 split per game. It is impossible to predict actual votes without modelling umpire behaviour, but we *can* build a defensible **statistical proxy** — a composite score over the stats that historically correlate with vote-earning. The weights below were validated against every player-game from 2010-2025 (n=145,150) where actual `brownlow_votes` are recorded — the top 1% of proxy games captured ~70% of vote-earning performances. Players need at least 3 games played to be ranked. Suspended players are not penalised in the proxy — this is a stat-profile model, not a vote forecaster — but because any in-season suspension makes a player ineligible to win the actual Brownlow Medal, suspended players are flagged inline in the table below so the distinction stays visible.

**Composite formula** (z-scored across all eligible players, summed with weights): `0.30 × disposals + 0.22 × clearances + 0.18 × contested-poss + 0.15 × effective-disposals + 0.15 × goals`. Effective disposals are approximated as `disposals - clangers` because the raw data does not carry a true effective-disposal column. Goals are weighted higher than the conventional midfielder-only template (15% vs the ~5% common in pure-midfielder proxies) because that materially improves correlation with actual historical Brownlow votes.

![2026 Brownlow predictor](../assets/charts/brownlow_predictor_2026.png)

#### Top 15 Brownlow proxy candidates — 2026 season-to-date (after Round 23)

| Rank | Player | Team | Games | Disp/g | Clear/g | CP/g | Goals/g | Proxy | Proj. votes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Nick Daicos | Collingwood | 20 | 35.2 | 6.0 | 10.9 | 1.10 | +2.80 | +61.6 |
| 2 | Clayton Oliver | Greater Western Sydney | 21 | 30.5 | 7.4 | 15.0 | 0.19 | +2.71 | +59.6 |
| 3 | Lachie Neale | Brisbane Lions | 21 | 30.4 | 7.0 | 13.2 | 0.24 | +2.57 | +56.5 |
| 4 | Isaac Heeney | Sydney | 19 | 26.8 | 6.2 | 12.8 | 1.53 | +2.42 | +53.1 |
| 5 | Bailey Smith | Geelong | 20 | 32.2 | 5.8 | 11.4 | 0.40 | +2.41 | +53.0 |
| 6 | Patrick Cripps | Carlton | 21 | 26.3 | 7.3 | 14.6 | 0.62 | +2.41 | +52.9 |
| 7 | Jai Newcombe | Hawthorn | 21 | 26.4 | 7.9 | 13.3 | 0.29 | +2.33 | +51.3 |
| 8 | Zak Butters | Port Adelaide | 17 | 29.8 | 6.1 | 12.2 | 0.29 | +2.29 | +50.4 |
| 9 | Matt Rowell | Gold Coast | 16 | 25.9 | 7.3 | 13.5 | 0.19 | +2.20 | +48.4 |
| 10 | Marcus Bontempelli | Western Bulldogs | 21 | 26.3 | 5.8 | 11.2 | 1.33 | +2.16 | +47.6 |
| 11 | Harry Sheezel | North Melbourne | 21 | 31.1 | 4.4 | 9.9 | 0.52 | +2.08 | +45.7 |
| 12 | Harley Reid | West Coast | 21 | 24.8 | 6.7 | 13.3 | 0.67 | +2.06 | +45.3 |
| 13 | Will Ashcroft | Brisbane Lions | 21 | 28.8 | 5.5 | 9.6 | 0.67 | +2.05 | +45.1 |
| 14 | Tim Taranto | Richmond | 19 | 24.4 | 6.2 | 12.1 | 0.68 | +1.98 | +43.6 |
| 15 | Ed Richards | Western Bulldogs | 20 | 25.6 | 5.9 | 10.2 | 0.80 | +1.91 | +41.9 |

On the proxy, **Nick Daicos** (Collingwood) leads the field — built on 35.2 disposals/g, 1.1 goals/g across 20 games. The composite score (+2.80) sits 0.09 clear of second place. **Clayton Oliver** (Greater Western Sydney) is the closest challenger at +2.71, with 30.5 disposals/g and 7.4 clearances/g. The proxy is a statistical model, not actual umpire votes — it captures the stat-profile umpires *historically* reward, but it cannot model individual game narrative, suspension impact or the umpire panel's eye for a defensive midfielder.
<!-- 2026-BROWNLOW-PREDICTOR-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
