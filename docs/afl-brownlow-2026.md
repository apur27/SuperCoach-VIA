# 2026 Brownlow Medal predictor

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BROWNLOW-PREDICTOR-START -->
The **Brownlow Medal** is the AFL's individual award for the "fairest and best" player, voted on by the on-field umpires with a 3-2-1 split per game. It is impossible to predict actual votes without modelling umpire behaviour, but we *can* build a defensible **statistical proxy** — a composite score over the stats that historically correlate with vote-earning. The weights below were validated against every player-game from 2010-2025 (n=145,150) where actual `brownlow_votes` are recorded — the top 1% of proxy games captured ~70% of vote-earning performances. Players need at least 3 games played to be ranked. Suspended players are not penalised in the proxy — this is a stat-profile model, not a vote forecaster — but because any in-season suspension makes a player ineligible to win the actual Brownlow Medal, suspended players are flagged inline in the table below so the distinction stays visible.

**Composite formula** (z-scored across all eligible players, summed with weights): `0.30 × disposals + 0.22 × clearances + 0.18 × contested-poss + 0.15 × effective-disposals + 0.15 × goals`. Effective disposals are approximated as `disposals - clangers` because the raw data does not carry a true effective-disposal column. Goals are weighted higher than the conventional midfielder-only template (15% vs the ~5% common in pure-midfielder proxies) because that materially improves correlation with actual historical Brownlow votes.

![2026 Brownlow predictor](../assets/charts/brownlow_predictor_2026.png)

#### Top 15 Brownlow proxy candidates — 2026 season-to-date (after Round 18)

| Rank | Player | Team | Games | Disp/g | Clear/g | CP/g | Goals/g | Proxy | Proj. votes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Clayton Oliver | Greater Western Sydney | 16 | 31.7 | 7.9 | 15.2 | 0.19 | +2.84 | +62.4 |
| 2 | Nick Daicos | Collingwood | 15 | 35.2 | 5.7 | 10.5 | 1.27 | +2.71 | +59.6 |
| 3 | Isaac Heeney | Sydney | 14 | 27.4 | 6.4 | 13.3 | 1.79 | +2.55 | +56.1 |
| 4 | Lachie Neale | Brisbane Lions | 16 | 30.1 | 7.3 | 12.9 | 0.25 | +2.52 | +55.4 |
| 5 | Jai Newcombe | Hawthorn | 16 | 26.9 | 8.2 | 13.2 | 0.38 | +2.39 | +52.6 |
| 6 | Bailey Smith | Geelong | 15 | 32.2 | 5.6 | 11.8 | 0.40 | +2.37 | +52.2 |
| 7 | Patrick Cripps | Carlton | 16 | 26.7 | 7.0 | 14.6 | 0.62 | +2.35 | +51.8 |
| 8 | Zak Butters | Port Adelaide | 16 | 30.0 | 6.2 | 12.6 | 0.31 | +2.31 | +50.9 |
| 9 | Marcus Bontempelli | Western Bulldogs | 16 | 26.9 | 6.1 | 11.4 | 1.25 | +2.20 | +48.4 |
| 10 | Harry Sheezel | North Melbourne | 16 | 31.9 | 4.9 | 10.4 | 0.44 | +2.17 | +47.8 |
| 11 | Christian Petracca | Gold Coast | 14 | 25.3 | 5.9 | 12.6 | 1.36 | +2.16 | +47.5 |
| 12 | Tim Taranto | Richmond | 14 | 23.8 | 5.9 | 12.3 | 0.71 | +1.88 | +41.5 |
| 13 | Max Gawn | Melbourne | 16 | 21.6 | 6.5 | 13.8 | 0.62 | +1.87 | +41.2 |
| 14 | Noah Anderson | Gold Coast | 15 | 28.3 | 5.3 | 9.5 | 0.33 | +1.84 | +40.5 |
| 15 | Caleb Serong | Fremantle | 13 | 24.8 | 5.8 | 11.7 | 0.46 | +1.83 | +40.3 |

On the proxy, **Clayton Oliver** (Greater Western Sydney) leads the field — built on 7.9 clearances/g, 15.2 contested poss/g across 16 games. The composite score (+2.84) sits 0.13 clear of second place. **Nick Daicos** (Collingwood) is the closest challenger at +2.71, with 35.2 disposals/g and 5.7 clearances/g. The proxy is a statistical model, not actual umpire votes — it captures the stat-profile umpires *historically* reward, but it cannot model individual game narrative, suspension impact or the umpire panel's eye for a defensive midfielder.
<!-- 2026-BROWNLOW-PREDICTOR-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
