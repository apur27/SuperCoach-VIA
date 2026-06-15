# 2026 Brownlow Medal predictor

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BROWNLOW-PREDICTOR-START -->
The **Brownlow Medal** is the AFL's individual award for the "fairest and best" player, voted on by the on-field umpires with a 3-2-1 split per game. It is impossible to predict actual votes without modelling umpire behaviour, but we *can* build a defensible **statistical proxy** — a composite score over the stats that historically correlate with vote-earning. The weights below were validated against every player-game from 2010-2025 (n=145,150) where actual `brownlow_votes` are recorded — the top 1% of proxy games captured ~70% of vote-earning performances. Players need at least 3 games played to be ranked. Suspended players are not penalised in the proxy — this is a stat-profile model, not a vote forecaster — but because any in-season suspension makes a player ineligible to win the actual Brownlow Medal, suspended players are flagged inline in the table below so the distinction stays visible.

**Composite formula** (z-scored across all eligible players, summed with weights): `0.30 × disposals + 0.22 × clearances + 0.18 × contested-poss + 0.15 × effective-disposals + 0.15 × goals`. Effective disposals are approximated as `disposals - clangers` because the raw data does not carry a true effective-disposal column. Goals are weighted higher than the conventional midfielder-only template (15% vs the ~5% common in pure-midfielder proxies) because that materially improves correlation with actual historical Brownlow votes.

![2026 Brownlow predictor](../assets/charts/brownlow_predictor_2026.png)

#### Top 15 Brownlow proxy candidates — 2026 season-to-date (after Round 15)

| Rank | Player | Team | Games | Disp/g | Clear/g | CP/g | Goals/g | Proxy | Proj. votes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Clayton Oliver | Greater Western Sydney | 13 | 31.7 | 8.3 | 15.5 | 0.15 | +2.87 | +63.2 |
| 2 | Lachie Neale | Brisbane Lions | 14 | 30.4 | 7.4 | 12.9 | 0.29 | +2.53 | +55.6 |
| 3 | Nick Daicos | Collingwood | 12 | 34.9 | 5.1 | 10.1 | 1.17 | +2.50 | +55.0 |
| 4 | Isaac Heeney | Sydney | 12 | 27.2 | 6.2 | 13.2 | 1.92 | +2.49 | +54.7 |
| 5 | Bailey Smith | Geelong | 14 | 32.3 | 5.5 | 11.6 | 0.43 | +2.32 | +51.1 |
| 6 | Patrick Cripps | Carlton | 13 | 26.0 | 6.9 | 14.4 | 0.62 | +2.23 | +49.0 |
| 7 | Zak Butters | Port Adelaide | 13 | 30.2 | 5.7 | 12.3 | 0.31 | +2.21 | +48.5 |
| 8 | Jai Newcombe | Hawthorn | 13 | 26.2 | 7.7 | 12.3 | 0.31 | +2.16 | +47.4 |
| 9 | Christian Petracca | Gold Coast | 11 | 25.3 | 6.0 | 11.7 | 1.55 | +2.12 | +46.6 |
| 10 | Marcus Bontempelli | Western Bulldogs | 14 | 26.9 | 5.4 | 11.4 | 1.36 | +2.09 | +45.9 |
| 11 | Harry Sheezel | North Melbourne | 13 | 30.5 | 4.9 | 9.9 | 0.46 | +1.98 | +43.6 |
| 12 | Tim Taranto | Richmond | 10 | 24.5 | 6.1 | 13.1 | 0.50 | +1.94 | +42.7 |
| 13 | Caleb Serong | Fremantle | 10 | 24.9 | 6.3 | 12.2 | 0.50 | +1.92 | +42.2 |
| 14 | Matt Rowell | Gold Coast | 8 | 24.2 | 6.4 | 12.2 | 0.38 | +1.82 | +40.1 |
| 15 | Max Gawn | Melbourne | 14 | 21.1 | 6.2 | 13.6 | 0.71 | +1.77 | +39.0 |

On the proxy, **Clayton Oliver** (Greater Western Sydney) leads the field — built on 8.3 clearances/g, 15.5 contested poss/g across 13 games. The composite score (+2.87) sits 0.35 clear of second place. **Lachie Neale** (Brisbane Lions) is the closest challenger at +2.53, with 30.4 disposals/g and 7.4 clearances/g. The proxy is a statistical model, not actual umpire votes — it captures the stat-profile umpires *historically* reward, but it cannot model individual game narrative, suspension impact or the umpire panel's eye for a defensive midfielder.
<!-- 2026-BROWNLOW-PREDICTOR-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
