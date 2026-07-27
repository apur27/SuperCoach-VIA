# 2026 Brownlow Medal predictor

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BROWNLOW-PREDICTOR-START -->
The **Brownlow Medal** is the AFL's individual award for the "fairest and best" player, voted on by the on-field umpires with a 3-2-1 split per game. It is impossible to predict actual votes without modelling umpire behaviour, but we *can* build a defensible **statistical proxy** — a composite score over the stats that historically correlate with vote-earning. The weights below were validated against every player-game from 2010-2025 (n=145,150) where actual `brownlow_votes` are recorded — the top 1% of proxy games captured ~70% of vote-earning performances. Players need at least 3 games played to be ranked. Suspended players are not penalised in the proxy — this is a stat-profile model, not a vote forecaster — but because any in-season suspension makes a player ineligible to win the actual Brownlow Medal, suspended players are flagged inline in the table below so the distinction stays visible.

**Composite formula** (z-scored across all eligible players, summed with weights): `0.30 × disposals + 0.22 × clearances + 0.18 × contested-poss + 0.15 × effective-disposals + 0.15 × goals`. Effective disposals are approximated as `disposals - clangers` because the raw data does not carry a true effective-disposal column. Goals are weighted higher than the conventional midfielder-only template (15% vs the ~5% common in pure-midfielder proxies) because that materially improves correlation with actual historical Brownlow votes.

![2026 Brownlow predictor](../assets/charts/brownlow_predictor_2026.png)

#### Top 15 Brownlow proxy candidates — 2026 season-to-date (after Round 21)

| Rank | Player | Team | Games | Disp/g | Clear/g | CP/g | Goals/g | Proxy | Proj. votes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Clayton Oliver | Greater Western Sydney | 19 | 31.3 | 7.8 | 15.5 | 0.21 | +2.84 | +62.6 |
| 2 | Nick Daicos | Collingwood | 18 | 35.1 | 5.6 | 10.4 | 1.17 | +2.69 | +59.2 |
| 3 | Lachie Neale | Brisbane Lions | 19 | 30.4 | 7.0 | 13.1 | 0.26 | +2.55 | +56.1 |
| 4 | Patrick Cripps | Carlton | 19 | 26.7 | 7.4 | 14.9 | 0.68 | +2.47 | +54.3 |
| 5 | Isaac Heeney | Sydney | 17 | 27.2 | 6.1 | 12.9 | 1.53 | +2.43 | +53.5 |
| 6 | Bailey Smith | Geelong | 18 | 32.1 | 5.4 | 11.1 | 0.44 | +2.33 | +51.2 |
| 7 | Marcus Bontempelli | Western Bulldogs | 19 | 26.7 | 6.2 | 11.5 | 1.42 | +2.28 | +50.1 |
| 8 | Zak Butters | Port Adelaide | 17 | 29.8 | 6.1 | 12.2 | 0.29 | +2.28 | +50.1 |
| 9 | Jai Newcombe | Hawthorn | 19 | 26.1 | 7.5 | 12.8 | 0.32 | +2.21 | +48.6 |
| 10 | Harry Sheezel | North Melbourne | 19 | 31.6 | 4.7 | 10.3 | 0.47 | +2.15 | +47.3 |
| 11 | Christian Petracca | Gold Coast | 17 | 24.8 | 5.5 | 11.9 | 1.29 | +2.03 | +44.6 |
| 12 | Harley Reid | West Coast | 19 | 24.1 | 6.6 | 12.9 | 0.68 | +1.95 | +42.9 |
| 13 | Matt Rowell | Gold Coast | 14 | 24.6 | 6.6 | 12.6 | 0.21 | +1.92 | +42.3 |
| 14 | Tim Taranto | Richmond | 17 | 24.2 | 5.9 | 12.1 | 0.71 | +1.91 | +42.1 |
| 15 | Will Ashcroft | Brisbane Lions | 19 | 28.1 | 5.3 | 9.2 | 0.63 | +1.90 | +41.9 |

On the proxy, **Clayton Oliver** (Greater Western Sydney) leads the field — built on 7.8 clearances/g, 15.5 contested poss/g across 19 games. The composite score (+2.84) sits 0.15 clear of second place. **Nick Daicos** (Collingwood) is the closest challenger at +2.69, with 35.1 disposals/g and 5.6 clearances/g. The proxy is a statistical model, not actual umpire votes — it captures the stat-profile umpires *historically* reward, but it cannot model individual game narrative, suspension impact or the umpire panel's eye for a defensive midfielder.
<!-- 2026-BROWNLOW-PREDICTOR-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
