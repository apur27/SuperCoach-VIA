# 2026 Brownlow Medal predictor

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BROWNLOW-PREDICTOR-START -->
The **Brownlow Medal** is the AFL's individual award for the "fairest and best" player, voted on by the on-field umpires with a 3-2-1 split per game. It is impossible to predict actual votes without modelling umpire behaviour, but we *can* build a defensible **statistical proxy** — a composite score over the stats that historically correlate with vote-earning. The weights below were validated against every player-game from 2010-2025 (n=145,150) where actual `brownlow_votes` are recorded — the top 1% of proxy games captured ~70% of vote-earning performances. Players need at least 3 games played to be ranked. Suspended players are not penalised — this proxy is a stat-profile model, not a vote forecaster.

**Composite formula** (z-scored across all eligible players, summed with weights): `0.30 × disposals + 0.22 × clearances + 0.18 × contested-poss + 0.15 × effective-disposals + 0.15 × goals`. Effective disposals are approximated as `disposals - clangers` because the raw data does not carry a true effective-disposal column. Goals are weighted higher than the conventional midfielder-only template (15% vs the ~5% common in pure-midfielder proxies) because that materially improves correlation with actual historical Brownlow votes.

![2026 Brownlow predictor](../assets/charts/brownlow_predictor_2026.png)

#### Top 15 Brownlow proxy candidates — 2026 season-to-date (after Round 10)

| Rank | Player | Team | Games | Disp/g | Clear/g | CP/g | Goals/g | Proxy | Proj. votes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Clayton Oliver | Greater Western Sydney | 9 | 30.4 | 7.8 | 14.6 | 0.22 | +2.56 | +56.4 |
| 2 | Lachie Neale | Brisbane Lions | 9 | 30.9 | 7.3 | 13.0 | 0.22 | +2.46 | +54.1 |
| 3 | Nick Daicos | Collingwood | 8 | 35.6 | 5.1 | 10.2 | 0.75 | +2.39 | +52.6 |
| 4 | Tristan Xerri | North Melbourne | 6 | 22.2 | 8.8 | 18.5 | 0.33 | +2.36 | +51.9 |
| 5 | Bailey Smith | Geelong | 9 | 32.2 | 5.6 | 11.6 | 0.33 | +2.22 | +48.8 |
| 6 | Isaac Heeney | Sydney | 7 | 25.7 | 5.7 | 13.6 | 1.86 | +2.22 | +48.8 |
| 7 | Christian Petracca | Gold Coast | 7 | 26.0 | 5.9 | 11.6 | 1.86 | +2.15 | +47.4 |
| 8 | Jai Newcombe | Hawthorn | 9 | 26.4 | 7.9 | 12.2 | 0.44 | +2.15 | +47.3 |
| 9 | Zak Butters | Port Adelaide | 9 | 30.6 | 5.6 | 12.0 | 0.33 | +2.12 | +46.6 |
| 10 | Patrick Cripps | Carlton | 9 | 24.9 | 7.0 | 14.7 | 0.44 | +2.05 | +45.0 |
| 11 | Marcus Bontempelli | Western Bulldogs | 9 | 27.7 | 5.2 | 10.4 | 1.44 | +2.00 | +43.9 |
| 12 | Harry Sheezel | North Melbourne | 9 | 31.1 | 4.6 | 9.6 | 0.56 | +1.91 | +42.1 |
| 13 | Matthew Kennedy | Western Bulldogs | 9 | 26.6 | 7.2 | 10.9 | 0.22 | +1.88 | +41.4 |
| 14 | Caleb Serong | Fremantle | 9 | 25.0 | 6.3 | 11.9 | 0.56 | +1.84 | +40.5 |
| 15 | Max Gawn | Melbourne | 9 | 21.3 | 6.3 | 14.7 | 0.78 | +1.81 | +39.8 |

On the proxy, **Clayton Oliver** (Greater Western Sydney) leads the field — built on 30.4 disposals/g and a balanced stat profile across 9 games. The composite score (+2.56) sits 0.10 clear of second place. **Lachie Neale** (Brisbane Lions) is the closest challenger at +2.46, with 30.9 disposals/g and 7.3 clearances/g. The proxy is a statistical model, not actual umpire votes — it captures the stat-profile umpires *historically* reward, but it cannot model individual game narrative, suspension impact or the umpire panel's eye for a defensive midfielder.
<!-- 2026-BROWNLOW-PREDICTOR-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
