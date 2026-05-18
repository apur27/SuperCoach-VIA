# 2026 Brownlow Medal predictor

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BROWNLOW-PREDICTOR-START -->
The **Brownlow Medal** is the AFL's individual award for the "fairest and best" player, voted on by the on-field umpires with a 3-2-1 split per game. It is impossible to predict actual votes without modelling umpire behaviour, but we *can* build a defensible **statistical proxy** — a composite score over the stats that historically correlate with vote-earning. The weights below were validated against every player-game from 2010-2025 (n=145,150) where actual `brownlow_votes` are recorded — the top 1% of proxy games captured ~70% of vote-earning performances. Players need at least 3 games played to be ranked. Suspended players are not penalised in the proxy — this is a stat-profile model, not a vote forecaster — but because any in-season suspension makes a player ineligible to win the actual Brownlow Medal, suspended players are flagged inline in the table below so the distinction stays visible.

**Composite formula** (z-scored across all eligible players, summed with weights): `0.30 × disposals + 0.22 × clearances + 0.18 × contested-poss + 0.15 × effective-disposals + 0.15 × goals`. Effective disposals are approximated as `disposals - clangers` because the raw data does not carry a true effective-disposal column. Goals are weighted higher than the conventional midfielder-only template (15% vs the ~5% common in pure-midfielder proxies) because that materially improves correlation with actual historical Brownlow votes.

![2026 Brownlow predictor](../assets/charts/brownlow_predictor_2026.png)

#### Top 15 Brownlow proxy candidates — 2026 season-to-date (after Round 11)

| Rank | Player | Team | Games | Disp/g | Clear/g | CP/g | Goals/g | Proxy | Proj. votes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Clayton Oliver | Greater Western Sydney | 10 | 31.2 | 8.2 | 15.5 | 0.20 | +2.76 | +60.7 |
| 2 | Isaac Heeney | Sydney | 8 | 26.5 | 6.4 | 14.4 | 1.75 | +2.41 | +53.1 |
| 3 | Lachie Neale | Brisbane Lions | 10 | 30.6 | 7.1 | 12.7 | 0.20 | +2.40 | +52.9 |
| 4 | Bailey Smith | Geelong | 10 | 32.4 | 5.7 | 11.6 | 0.50 | +2.33 | +51.2 |
| 5 | Nick Daicos | Collingwood | 9 | 34.8 | 4.8 | 9.8 | 0.89 | +2.30 | +50.7 |
| 6 | Zak Butters | Port Adelaide | 10 | 31.0 | 5.9 | 12.4 | 0.30 | +2.25 | +49.5 |
| 7 | Christian Petracca | Gold Coast | 8 | 26.1 | 6.2 | 12.0 | 1.75 | +2.23 | +49.1 |
| 8 | Tristan Xerri **[SUSPENDED — BROWNLOW INELIGIBLE]** | North Melbourne | 7 | 21.1 | 8.9 | 17.6 | 0.29 | +2.22 | +48.8 |
| 9 | Patrick Cripps | Carlton | 10 | 25.3 | 7.1 | 14.9 | 0.60 | +2.17 | +47.6 |
| 10 | Jai Newcombe | Hawthorn | 10 | 26.2 | 7.4 | 12.2 | 0.40 | +2.08 | +45.8 |
| 11 | Marcus Bontempelli | Western Bulldogs | 10 | 27.3 | 4.8 | 10.5 | 1.40 | +1.92 | +42.3 |
| 12 | Caleb Serong | Fremantle | 10 | 24.9 | 6.3 | 12.2 | 0.50 | +1.87 | +41.0 |
| 13 | Harry Sheezel | North Melbourne | 10 | 30.4 | 4.5 | 9.5 | 0.50 | +1.85 | +40.6 |
| 14 | Max Gawn | Melbourne | 10 | 21.0 | 6.5 | 14.6 | 0.70 | +1.81 | +39.8 |
| 15 | Tim Taranto | Richmond | 7 | 23.1 | 6.3 | 12.9 | 0.43 | +1.76 | +38.8 |

On the proxy, **Clayton Oliver** (Greater Western Sydney) leads the field — built on 8.2 clearances/g, 15.5 contested poss/g across 10 games. The composite score (+2.76) sits 0.35 clear of second place. **Isaac Heeney** (Sydney) is the closest challenger at +2.41, with 26.5 disposals/g and 6.4 clearances/g. The proxy is a statistical model, not actual umpire votes — it captures the stat-profile umpires *historically* reward, but it cannot model individual game narrative, suspension impact or the umpire panel's eye for a defensive midfielder.

**Brownlow ineligibility note.** Under AFL rules, any player suspended by the Tribunal during the home-and-away season is automatically ineligible to win the Brownlow Medal, regardless of vote tally. Our proxy ranks on stat profile and does not remove these players — their score is still informative — but they cannot win the actual medal:

- **Tristan Xerri** (North Melbourne, proxy rank #8, +2.22) — Suspended (2026 season)
<!-- 2026-BROWNLOW-PREDICTOR-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
