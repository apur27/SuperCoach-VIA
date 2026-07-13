# 2026 Brownlow Medal predictor

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BROWNLOW-PREDICTOR-START -->
The **Brownlow Medal** is the AFL's individual award for the "fairest and best" player, voted on by the on-field umpires with a 3-2-1 split per game. It is impossible to predict actual votes without modelling umpire behaviour, but we *can* build a defensible **statistical proxy** — a composite score over the stats that historically correlate with vote-earning. The weights below were validated against every player-game from 2010-2025 (n=145,150) where actual `brownlow_votes` are recorded — the top 1% of proxy games captured ~70% of vote-earning performances. Players need at least 3 games played to be ranked. Suspended players are not penalised in the proxy — this is a stat-profile model, not a vote forecaster — but because any in-season suspension makes a player ineligible to win the actual Brownlow Medal, suspended players are flagged inline in the table below so the distinction stays visible.

**Composite formula** (z-scored across all eligible players, summed with weights): `0.30 × disposals + 0.22 × clearances + 0.18 × contested-poss + 0.15 × effective-disposals + 0.15 × goals`. Effective disposals are approximated as `disposals - clangers` because the raw data does not carry a true effective-disposal column. Goals are weighted higher than the conventional midfielder-only template (15% vs the ~5% common in pure-midfielder proxies) because that materially improves correlation with actual historical Brownlow votes.

![2026 Brownlow predictor](../assets/charts/brownlow_predictor_2026.png)

#### Top 15 Brownlow proxy candidates — 2026 season-to-date (after Round 19)

| Rank | Player | Team | Games | Disp/g | Clear/g | CP/g | Goals/g | Proxy | Proj. votes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Clayton Oliver | Greater Western Sydney | 17 | 31.9 | 7.9 | 15.5 | 0.18 | +2.90 | +63.7 |
| 2 | Nick Daicos | Collingwood | 16 | 34.8 | 5.6 | 10.2 | 1.25 | +2.65 | +58.2 |
| 3 | Lachie Neale | Brisbane Lions | 17 | 30.6 | 7.2 | 13.0 | 0.24 | +2.56 | +56.4 |
| 4 | Isaac Heeney | Sydney | 15 | 27.3 | 6.2 | 12.9 | 1.67 | +2.46 | +54.0 |
| 5 | Bailey Smith | Geelong | 16 | 32.2 | 5.7 | 11.8 | 0.44 | +2.39 | +52.6 |
| 6 | Patrick Cripps | Carlton | 17 | 26.4 | 7.0 | 14.8 | 0.59 | +2.33 | +51.2 |
| 7 | Jai Newcombe | Hawthorn | 17 | 26.4 | 7.8 | 13.0 | 0.35 | +2.29 | +50.4 |
| 8 | Marcus Bontempelli | Western Bulldogs | 17 | 26.8 | 6.5 | 11.6 | 1.29 | +2.28 | +50.1 |
| 9 | Zak Butters | Port Adelaide | 17 | 29.8 | 6.1 | 12.2 | 0.29 | +2.26 | +49.7 |
| 10 | Harry Sheezel | North Melbourne | 17 | 31.6 | 4.9 | 10.4 | 0.41 | +2.14 | +47.2 |
| 11 | Christian Petracca | Gold Coast | 15 | 24.9 | 5.7 | 12.2 | 1.33 | +2.07 | +45.4 |
| 12 | Tim Taranto | Richmond | 15 | 23.9 | 5.9 | 12.3 | 0.73 | +1.89 | +41.6 |
| 13 | Caleb Serong | Fremantle | 14 | 25.0 | 5.9 | 11.6 | 0.43 | +1.85 | +40.7 |
| 14 | Harley Reid | West Coast | 17 | 23.8 | 6.1 | 12.6 | 0.76 | +1.84 | +40.6 |
| 15 | Matt Rowell | Gold Coast | 12 | 24.2 | 6.3 | 12.5 | 0.25 | +1.84 | +40.5 |

On the proxy, **Clayton Oliver** (Greater Western Sydney) leads the field — built on 7.9 clearances/g, 15.5 contested poss/g across 17 games. The composite score (+2.90) sits 0.25 clear of second place. **Nick Daicos** (Collingwood) is the closest challenger at +2.65, with 34.8 disposals/g and 5.6 clearances/g. The proxy is a statistical model, not actual umpire votes — it captures the stat-profile umpires *historically* reward, but it cannot model individual game narrative, suspension impact or the umpire panel's eye for a defensive midfielder.
<!-- 2026-BROWNLOW-PREDICTOR-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
