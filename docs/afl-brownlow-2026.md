# 2026 Brownlow Medal predictor

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BROWNLOW-PREDICTOR-START -->
The **Brownlow Medal** is the AFL's individual award for the "fairest and best" player, voted on by the on-field umpires with a 3-2-1 split per game. It is impossible to predict actual votes without modelling umpire behaviour, but we *can* build a defensible **statistical proxy** - a composite score over the stats that historically correlate with vote-earning. The weights below were validated against every player-game from 2010-2025 (n=145,150) where actual `brownlow_votes` are recorded - the top 1% of proxy games captured ~70% of vote-earning performances. Players need at least 3 games played to be ranked. Suspended players are not penalised - this proxy is a stat-profile model, not a vote forecaster.

**Composite formula** (z-scored across all eligible players, summed with weights): `0.30 × disposals + 0.22 × clearances + 0.18 × contested-poss + 0.15 × effective-disposals + 0.15 × goals`. Effective disposals are approximated as `disposals - clangers` because the raw data does not carry a true effective-disposal column. Goals are weighted higher than the conventional midfielder-only template (15% vs the ~5% common in pure-midfielder proxies) because that materially improves correlation with actual historical Brownlow votes.

![2026 Brownlow predictor](../assets/charts/brownlow_predictor_2026.png)

#### Top 15 Brownlow proxy candidates - 2026 season-to-date (after Round 9)

| Rank | Player | Team | Games | Disp/g | Clear/g | CP/g | Goals/g | Proxy | Proj. votes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Nick Daicos | Collingwood | 7 | 36.6 | 5.6 | 10.7 | 0.86 | +2.55 | +56.1 |
| 2 | Clayton Oliver | Greater Western Sydney | 8 | 31.0 | 7.4 | 14.6 | 0.25 | +2.53 | +55.7 |
| 3 | Tristan Xerri | North Melbourne | 5 | 22.8 | 8.8 | 19.2 | 0.20 | +2.35 | +51.8 |
| 4 | Lachie Neale | Brisbane Lions | 8 | 30.6 | 7.1 | 12.6 | 0.12 | +2.32 | +51.0 |
| 5 | Christian Petracca | Gold Coast | 6 | 26.5 | 5.8 | 11.8 | 2.00 | +2.20 | +48.5 |
| 6 | Zak Butters | Port Adelaide | 8 | 30.6 | 5.6 | 12.4 | 0.38 | +2.13 | +46.8 |
| 7 | Jai Newcombe | Hawthorn | 8 | 26.5 | 7.9 | 12.0 | 0.50 | +2.12 | +46.5 |
| 8 | Isaac Heeney | Sydney | 6 | 26.0 | 5.5 | 13.3 | 1.67 | +2.11 | +46.3 |
| 9 | Bailey Smith | Geelong | 8 | 31.2 | 5.5 | 11.2 | 0.38 | +2.08 | +45.7 |
| 10 | Marcus Bontempelli | Western Bulldogs | 8 | 27.4 | 5.0 | 10.6 | 1.50 | +1.92 | +42.3 |
| 11 | Matthew Kennedy | Western Bulldogs | 8 | 26.8 | 7.4 | 11.2 | 0.25 | +1.92 | +42.2 |
| 12 | Harry Sheezel | North Melbourne | 8 | 31.1 | 4.8 | 9.6 | 0.50 | +1.88 | +41.4 |
| 13 | Patrick Cripps | Carlton | 8 | 24.0 | 6.8 | 14.2 | 0.38 | +1.85 | +40.7 |
| 14 | Max Gawn | Melbourne | 8 | 21.4 | 6.5 | 14.8 | 0.75 | +1.79 | +39.4 |
| 15 | Caleb Serong | Fremantle | 8 | 24.6 | 6.1 | 12.0 | 0.62 | +1.76 | +38.8 |

On the proxy, **Nick Daicos** (Collingwood) leads the field - built on 36.6 disposals/g across 7 games. The composite score (+2.55) sits 0.02 clear of second place. **Clayton Oliver** (Greater Western Sydney) is the closest challenger at +2.53, with 31.0 disposals/g and 7.4 clearances/g. The proxy is a statistical model, not actual umpire votes - it captures the stat-profile umpires *historically* reward, but it cannot model individual game narrative, suspension impact or the umpire panel's eye for a defensive midfielder.
<!-- 2026-BROWNLOW-PREDICTOR-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
