# 2026 Brownlow Medal predictor

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-BROWNLOW-PREDICTOR-START -->
The **Brownlow Medal** is the AFL's individual award for the "fairest and best" player, voted on by the on-field umpires with a 3-2-1 split per game. It is impossible to predict actual votes without modelling umpire behaviour, but we *can* build a defensible **statistical proxy** — a composite score over the stats that historically correlate with vote-earning. The weights below were validated against every player-game from 2010-2025 (n=145,150) where actual `brownlow_votes` are recorded — the top 1% of proxy games captured ~70% of vote-earning performances. Players need at least 3 games played to be ranked. Suspended players are not penalised in the proxy — this is a stat-profile model, not a vote forecaster — but because any in-season suspension makes a player ineligible to win the actual Brownlow Medal, suspended players are flagged inline in the table below so the distinction stays visible.

**Composite formula** (z-scored across all eligible players, summed with weights): `0.30 × disposals + 0.22 × clearances + 0.18 × contested-poss + 0.15 × effective-disposals + 0.15 × goals`. Effective disposals are approximated as `disposals - clangers` because the raw data does not carry a true effective-disposal column. Goals are weighted higher than the conventional midfielder-only template (15% vs the ~5% common in pure-midfielder proxies) because that materially improves correlation with actual historical Brownlow votes.

![2026 Brownlow predictor](../assets/charts/brownlow_predictor_2026.png)

#### Top 15 Brownlow proxy candidates — 2026 season-to-date (after Round 12)

| Rank | Player | Team | Games | Disp/g | Clear/g | CP/g | Goals/g | Proxy | Proj. votes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Clayton Oliver | Greater Western Sydney | 11 | 31.7 | 8.5 | 16.2 | 0.18 | +2.90 | +63.8 |
| 2 | Lachie Neale | Brisbane Lions | 11 | 30.7 | 7.4 | 13.0 | 0.27 | +2.50 | +54.9 |
| 3 | Isaac Heeney | Sydney | 9 | 26.1 | 6.6 | 14.2 | 1.67 | +2.40 | +52.8 |
| 4 | Nick Daicos | Collingwood | 10 | 34.7 | 5.0 | 9.7 | 1.10 | +2.39 | +52.5 |
| 5 | Bailey Smith | Geelong | 11 | 32.4 | 5.8 | 11.6 | 0.55 | +2.38 | +52.3 |
| 6 | Christian Petracca | Gold Coast | 9 | 26.3 | 6.4 | 12.2 | 1.67 | +2.29 | +50.4 |
| 7 | Patrick Cripps | Carlton | 11 | 25.9 | 7.2 | 15.1 | 0.64 | +2.26 | +49.8 |
| 8 | Zak Butters | Port Adelaide | 11 | 30.7 | 5.7 | 12.0 | 0.27 | +2.19 | +48.1 |
| 9 | Jai Newcombe | Hawthorn | 11 | 25.5 | 7.6 | 12.2 | 0.36 | +2.05 | +45.1 |
| 10 | Marcus Bontempelli | Western Bulldogs | 11 | 27.5 | 4.9 | 11.1 | 1.45 | +2.02 | +44.4 |
| 11 | Matt Rowell | Gold Coast | 6 | 25.3 | 7.0 | 13.7 | 0.17 | +1.98 | +43.6 |
| 12 | Tristan Xerri **[SUSPENDED — BROWNLOW INELIGIBLE]** | North Melbourne | 8 | 20.0 | 8.0 | 16.1 | 0.38 | +1.95 | +43.0 |
| 13 | Caleb Serong | Fremantle | 10 | 24.9 | 6.3 | 12.2 | 0.50 | +1.88 | +41.4 |
| 14 | Harry Sheezel | North Melbourne | 11 | 30.5 | 4.6 | 9.5 | 0.45 | +1.88 | +41.3 |
| 15 | Tim Taranto | Richmond | 8 | 24.0 | 6.5 | 12.6 | 0.38 | +1.85 | +40.6 |

On the proxy, **Clayton Oliver** (Greater Western Sydney) leads the field — built on 8.5 clearances/g, 16.2 contested poss/g across 11 games. The composite score (+2.90) sits 0.41 clear of second place. **Lachie Neale** (Brisbane Lions) is the closest challenger at +2.50, with 30.7 disposals/g and 7.4 clearances/g. The proxy is a statistical model, not actual umpire votes — it captures the stat-profile umpires *historically* reward, but it cannot model individual game narrative, suspension impact or the umpire panel's eye for a defensive midfielder.

**Brownlow ineligibility note.** Under AFL rules, any player suspended by the Tribunal during the home-and-away season is automatically ineligible to win the Brownlow Medal, regardless of vote tally. Our proxy ranks on stat profile and does not remove these players — their score is still informative — but they cannot win the actual medal:

- **Tristan Xerri** (North Melbourne, proxy rank #12, +1.95) — Suspended (2026 season)
<!-- 2026-BROWNLOW-PREDICTOR-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Stat leaders](afl-stat-leaders-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
