# 2026 Brownlow Medal predictor

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BROWNLOW-PREDICTOR-START -->
The **Brownlow Medal** is the AFL's individual award for the "fairest and best" player, voted on by the on-field umpires with a 3-2-1 split per game. It is impossible to predict actual votes without modelling umpire behaviour, but we *can* build a defensible **statistical proxy** — a composite score over the stats that historically correlate with vote-earning. The weights below were validated against every player-game from 2010-2025 (n=145,150) where actual `brownlow_votes` are recorded — the top 1% of proxy games captured ~70% of vote-earning performances. Players need at least 3 games played to be ranked. Suspended players are not penalised in the proxy — this is a stat-profile model, not a vote forecaster — but because any in-season suspension makes a player ineligible to win the actual Brownlow Medal, suspended players are flagged inline in the table below so the distinction stays visible.

**Composite formula** (z-scored across all eligible players, summed with weights): `0.30 × disposals + 0.22 × clearances + 0.18 × contested-poss + 0.15 × effective-disposals + 0.15 × goals`. Effective disposals are approximated as `disposals - clangers` because the raw data does not carry a true effective-disposal column. Goals are weighted higher than the conventional midfielder-only template (15% vs the ~5% common in pure-midfielder proxies) because that materially improves correlation with actual historical Brownlow votes.

![2026 Brownlow predictor](../assets/charts/brownlow_predictor_2026.png)

#### Top 15 Brownlow proxy candidates — 2026 season-to-date (after Round 16)

| Rank | Player | Team | Games | Disp/g | Clear/g | CP/g | Goals/g | Proxy | Proj. votes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Clayton Oliver | Greater Western Sydney | 14 | 31.5 | 8.1 | 15.3 | 0.14 | +2.81 | +61.9 |
| 2 | Nick Daicos | Collingwood | 13 | 35.4 | 5.4 | 10.4 | 1.15 | +2.61 | +57.5 |
| 3 | Lachie Neale | Brisbane Lions | 14 | 30.4 | 7.4 | 12.9 | 0.29 | +2.54 | +55.8 |
| 4 | Isaac Heeney | Sydney | 12 | 27.2 | 6.2 | 13.2 | 1.92 | +2.51 | +55.2 |
| 5 | Bailey Smith | Geelong | 14 | 32.3 | 5.5 | 11.6 | 0.43 | +2.33 | +51.4 |
| 6 | Patrick Cripps | Carlton | 14 | 26.5 | 6.9 | 14.5 | 0.57 | +2.28 | +50.2 |
| 7 | Jai Newcombe | Hawthorn | 14 | 26.6 | 7.9 | 12.6 | 0.43 | +2.27 | +49.8 |
| 8 | Marcus Bontempelli | Western Bulldogs | 15 | 27.4 | 5.9 | 11.6 | 1.33 | +2.22 | +48.8 |
| 9 | Zak Butters | Port Adelaide | 14 | 29.8 | 5.8 | 12.0 | 0.36 | +2.19 | +48.3 |
| 10 | Christian Petracca | Gold Coast | 12 | 25.8 | 5.8 | 12.3 | 1.50 | +2.17 | +47.8 |
| 11 | Harry Sheezel | North Melbourne | 14 | 30.5 | 4.9 | 10.1 | 0.43 | +2.01 | +44.2 |
| 12 | Caleb Serong | Fremantle | 11 | 24.6 | 6.1 | 11.8 | 0.55 | +1.86 | +41.0 |
| 13 | Max Gawn | Melbourne | 15 | 21.6 | 6.3 | 13.7 | 0.67 | +1.82 | +40.1 |
| 14 | Tim Taranto | Richmond | 12 | 23.5 | 6.0 | 12.6 | 0.50 | +1.82 | +40.0 |
| 15 | Noah Anderson | Gold Coast | 13 | 28.5 | 5.2 | 9.7 | 0.31 | +1.81 | +39.9 |

On the proxy, **Clayton Oliver** (Greater Western Sydney) leads the field — built on 8.1 clearances/g, 15.3 contested poss/g across 14 games. The composite score (+2.81) sits 0.20 clear of second place. **Nick Daicos** (Collingwood) is the closest challenger at +2.61, with 35.4 disposals/g and 5.4 clearances/g. The proxy is a statistical model, not actual umpire votes — it captures the stat-profile umpires *historically* reward, but it cannot model individual game narrative, suspension impact or the umpire panel's eye for a defensive midfielder.
<!-- 2026-BROWNLOW-PREDICTOR-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
