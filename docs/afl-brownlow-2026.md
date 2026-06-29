# 2026 Brownlow Medal predictor

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BROWNLOW-PREDICTOR-START -->
The **Brownlow Medal** is the AFL's individual award for the "fairest and best" player, voted on by the on-field umpires with a 3-2-1 split per game. It is impossible to predict actual votes without modelling umpire behaviour, but we *can* build a defensible **statistical proxy** — a composite score over the stats that historically correlate with vote-earning. The weights below were validated against every player-game from 2010-2025 (n=145,150) where actual `brownlow_votes` are recorded — the top 1% of proxy games captured ~70% of vote-earning performances. Players need at least 3 games played to be ranked. Suspended players are not penalised in the proxy — this is a stat-profile model, not a vote forecaster — but because any in-season suspension makes a player ineligible to win the actual Brownlow Medal, suspended players are flagged inline in the table below so the distinction stays visible.

**Composite formula** (z-scored across all eligible players, summed with weights): `0.30 × disposals + 0.22 × clearances + 0.18 × contested-poss + 0.15 × effective-disposals + 0.15 × goals`. Effective disposals are approximated as `disposals - clangers` because the raw data does not carry a true effective-disposal column. Goals are weighted higher than the conventional midfielder-only template (15% vs the ~5% common in pure-midfielder proxies) because that materially improves correlation with actual historical Brownlow votes.

![2026 Brownlow predictor](../assets/charts/brownlow_predictor_2026.png)

#### Top 15 Brownlow proxy candidates — 2026 season-to-date (after Round 17)

| Rank | Player | Team | Games | Disp/g | Clear/g | CP/g | Goals/g | Proxy | Proj. votes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Clayton Oliver | Greater Western Sydney | 15 | 31.7 | 7.7 | 15.1 | 0.20 | +2.79 | +61.4 |
| 2 | Nick Daicos | Collingwood | 14 | 35.5 | 5.5 | 10.4 | 1.29 | +2.69 | +59.1 |
| 3 | Isaac Heeney | Sydney | 13 | 27.5 | 6.3 | 13.3 | 1.85 | +2.54 | +55.8 |
| 4 | Lachie Neale | Brisbane Lions | 15 | 30.2 | 7.4 | 12.9 | 0.27 | +2.52 | +55.5 |
| 5 | Jai Newcombe | Hawthorn | 15 | 26.9 | 8.3 | 13.2 | 0.40 | +2.41 | +52.9 |
| 6 | Zak Butters | Port Adelaide | 15 | 30.3 | 6.3 | 12.7 | 0.33 | +2.36 | +51.9 |
| 7 | Patrick Cripps | Carlton | 15 | 26.8 | 7.1 | 14.6 | 0.60 | +2.35 | +51.6 |
| 8 | Bailey Smith | Geelong | 14 | 32.3 | 5.5 | 11.6 | 0.43 | +2.34 | +51.6 |
| 9 | Marcus Bontempelli | Western Bulldogs | 15 | 27.4 | 5.9 | 11.6 | 1.33 | +2.23 | +49.1 |
| 10 | Harry Sheezel | North Melbourne | 15 | 31.3 | 4.9 | 10.3 | 0.40 | +2.09 | +45.9 |
| 11 | Christian Petracca | Gold Coast | 13 | 25.1 | 5.8 | 12.2 | 1.38 | +2.08 | +45.8 |
| 12 | Noah Anderson | Gold Coast | 14 | 28.9 | 5.4 | 9.6 | 0.29 | +1.86 | +41.0 |
| 13 | Tim Taranto | Richmond | 13 | 23.7 | 5.8 | 12.5 | 0.77 | +1.86 | +41.0 |
| 14 | Max Gawn | Melbourne | 15 | 21.6 | 6.3 | 13.7 | 0.67 | +1.84 | +40.4 |
| 15 | Matt Rowell | Gold Coast | 10 | 23.8 | 6.4 | 12.1 | 0.30 | +1.79 | +39.4 |

On the proxy, **Clayton Oliver** (Greater Western Sydney) leads the field — built on 7.7 clearances/g, 15.1 contested poss/g across 15 games. The composite score (+2.79) sits 0.10 clear of second place. **Nick Daicos** (Collingwood) is the closest challenger at +2.69, with 35.5 disposals/g and 5.5 clearances/g. The proxy is a statistical model, not actual umpire votes — it captures the stat-profile umpires *historically* reward, but it cannot model individual game narrative, suspension impact or the umpire panel's eye for a defensive midfielder.
<!-- 2026-BROWNLOW-PREDICTOR-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
