# 2026 Brownlow Medal predictor

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BROWNLOW-PREDICTOR-START -->
The **Brownlow Medal** is the AFL's individual award for the "fairest and best" player, voted on by the on-field umpires with a 3-2-1 split per game. It is impossible to predict actual votes without modelling umpire behaviour, but we *can* build a defensible **statistical proxy** — a composite score over the stats that historically correlate with vote-earning. The weights below were validated against every player-game from 2010-2025 (n=145,150) where actual `brownlow_votes` are recorded — the top 1% of proxy games captured ~70% of vote-earning performances. Players need at least 3 games played to be ranked. Suspended players are not penalised in the proxy — this is a stat-profile model, not a vote forecaster — but because any in-season suspension makes a player ineligible to win the actual Brownlow Medal, suspended players are flagged inline in the table below so the distinction stays visible.

**Composite formula** (z-scored across all eligible players, summed with weights): `0.30 × disposals + 0.22 × clearances + 0.18 × contested-poss + 0.15 × effective-disposals + 0.15 × goals`. Effective disposals are approximated as `disposals - clangers` because the raw data does not carry a true effective-disposal column. Goals are weighted higher than the conventional midfielder-only template (15% vs the ~5% common in pure-midfielder proxies) because that materially improves correlation with actual historical Brownlow votes.

![2026 Brownlow predictor](../assets/charts/brownlow_predictor_2026.png)

#### Top 15 Brownlow proxy candidates — 2026 season-to-date (after Round 20)

| Rank | Player | Team | Games | Disp/g | Clear/g | CP/g | Goals/g | Proxy | Proj. votes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Clayton Oliver | Greater Western Sydney | 18 | 31.4 | 7.9 | 15.6 | 0.17 | +2.87 | +63.2 |
| 2 | Nick Daicos | Collingwood | 17 | 34.8 | 5.5 | 10.0 | 1.18 | +2.62 | +57.7 |
| 3 | Lachie Neale | Brisbane Lions | 18 | 30.7 | 7.1 | 13.2 | 0.28 | +2.60 | +57.2 |
| 4 | Isaac Heeney | Sydney | 16 | 27.4 | 6.1 | 13.1 | 1.56 | +2.45 | +54.0 |
| 5 | Bailey Smith | Geelong | 17 | 32.4 | 5.5 | 11.4 | 0.47 | +2.37 | +52.1 |
| 6 | Patrick Cripps | Carlton | 18 | 26.6 | 6.9 | 14.6 | 0.67 | +2.36 | +51.9 |
| 7 | Marcus Bontempelli | Western Bulldogs | 18 | 26.6 | 6.4 | 11.8 | 1.39 | +2.30 | +50.6 |
| 8 | Zak Butters | Port Adelaide | 17 | 29.8 | 6.1 | 12.2 | 0.29 | +2.27 | +50.0 |
| 9 | Jai Newcombe | Hawthorn | 18 | 25.9 | 7.7 | 12.9 | 0.33 | +2.23 | +49.1 |
| 10 | Harry Sheezel | North Melbourne | 18 | 31.7 | 4.7 | 10.3 | 0.44 | +2.15 | +47.3 |
| 11 | Christian Petracca | Gold Coast | 16 | 24.8 | 5.5 | 11.8 | 1.38 | +2.03 | +44.6 |
| 12 | Caleb Serong | Fremantle | 15 | 25.2 | 6.1 | 11.8 | 0.40 | +1.91 | +42.0 |
| 13 | Tim Taranto | Richmond | 16 | 24.1 | 5.8 | 12.1 | 0.69 | +1.89 | +41.6 |
| 14 | Harley Reid | West Coast | 18 | 23.7 | 6.3 | 12.8 | 0.72 | +1.88 | +41.4 |
| 15 | Will Ashcroft | Brisbane Lions | 18 | 28.0 | 5.3 | 9.3 | 0.56 | +1.88 | +41.3 |

On the proxy, **Clayton Oliver** (Greater Western Sydney) leads the field — built on 7.9 clearances/g, 15.6 contested poss/g across 18 games. The composite score (+2.87) sits 0.25 clear of second place. **Nick Daicos** (Collingwood) is the closest challenger at +2.62, with 34.8 disposals/g and 5.5 clearances/g. The proxy is a statistical model, not actual umpire votes — it captures the stat-profile umpires *historically* reward, but it cannot model individual game narrative, suspension impact or the umpire panel's eye for a defensive midfielder.
<!-- 2026-BROWNLOW-PREDICTOR-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
