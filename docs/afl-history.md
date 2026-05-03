# AFL history — how the game has changed across 125 years

> [← Back to AFL insights](afl-insights.md) | [← Back to main README](../README.md)

## How the game has changed — 125 years of data

This repo holds **687,920 player-game records** stretching from 1897 to 2026 and **16,900 matches**. That's a long enough run to see how AFL has actually evolved — not just how it feels different, but how it's measurably different.

The numbers below come from running `era_based_statistical_analysis.py` over every player and every match. Era buckets are deliberate: stats coverage changes with the eras. **Disposals, marks, kicks and handballs were not tracked at all before 1965. Tackles weren't reliably recorded until 1987. Clearances, contested possessions and inside-50s only start appearing in 1998.** Anywhere the cell says "n/a" below, it's because the AFL didn't track that stat yet — not because the value was zero.

### Scoring across the eras

![AFL team scoring 1965 to 2025](../assets/charts/era_scoring_trends.png)

<!-- SCORING-DECADE-CHART-START -->
![Average team score by decade, 1900s to 2020s](../assets/charts/scoring_by_decade.png)
<!-- SCORING-DECADE-CHART-END -->

| Per team, per game | pre-1965 | 1965–1990 | 1991–2010 | 2011–now |
|---|---:|---:|---:|---:|
| Goals | **10.1** | 13.9 | 13.9 | 12.3 |
| Behinds | 12.0 | 13.6 | 12.2 | **11.0** |
| Total points | 72.5 | **96.8** | 95.7 | 84.9 |
| Scoring shots | 22.1 | **27.5** | 26.1 | 23.3 |
| Goal accuracy | 45.2% | 50.1% | **53.0%** | 52.7% |
| Match total points | 145 | 193 | 191 | **170** |

A few things jump out that contradict the usual barbershop wisdom:

- **The high-scoring era was the 1980s, not now.** A team-game in 1965–1990 averaged 96.75 points — modern teams average 84.88. That's a 12% drop in per-team scoring from the Hudson/Lockett era to the Daicos era. Total match scoring is down 23 points per game from the peak.
- **Modern players are NOT more accurate in front of goal in any meaningful way.** Goal accuracy peaked in 1991–2010 at 53.0% and has actually slipped slightly to 52.7% in the modern era. The big jump was much earlier — pre-1965 footballers converted just 45.2% of their scoring shots, which lines up with rough grounds, leather balls that swelled in the rain, and longer drop kicks rather than today's set-shot routines.
- **Modern teams have fewer scoring shots, not just less accuracy.** Scoring shots per team-game: 27.5 (1965–1990) → 26.1 → 23.3. That's 4 fewer scoring shots per team per game vs the 80s.

### Player workload — the most dramatic change in footy

![Per-player stat evolution by era](../assets/charts/era_stat_evolution.png)

<!-- TACKLES-CLEARANCES-CHART-START -->
![Tackles and clearances per player per game over time](../assets/charts/era_tackles_clearances.png)
<!-- TACKLES-CLEARANCES-CHART-END -->

| Per player, per game | pre-1965 | 1965–1990 | 1991–2010 | 2011–now |
|---|---:|---:|---:|---:|
| Kicks | n/a | 11.08 | 9.15 | 9.41 |
| Handballs | n/a | 4.31 | 5.85 | **6.95** |
| Disposals | n/a | 14.89 | 14.72 | **16.21** |
| Marks | n/a | 3.76 | 4.11 | 4.22 |
| Tackles | n/a | 1.86 | 2.42 | **3.20** |
| Clearances | n/a | n/a | 2.54 | 2.87 |
| Contested possessions | n/a | n/a | 5.65 | 6.27 |

Read the kicks/handballs row carefully — this is the single biggest shift in how footy is played:

- **Handballs are up 61%** since 1965–1990 (4.31/g → 6.95/g). **Kicks are down 15%** in the same window (11.08/g → 9.41/g). The modern game is a handball game wearing a kicking game's uniform. Players today move the ball more often, but they move it shorter and faster.
- **Tackles are up 72%** from the 1965–1990 era (1.86/g → 3.20/g). Even just inside the modern stat era, tackles per player jumped 32% from 2.42 (1991–2010) to 3.20 (2011–now). Pressure is the defining tactical innovation of the post-2010 game, and the data is unambiguous about it.
- **Disposals per player are up about 9%** from the 80s (14.89 → 16.21). Combined with smaller rosters and more interchanges, that means individual ball-winners are doing more.
- **Contested possessions are up 11%** since 1991–2010 (5.65 → 6.27). The contested-ball revolution that coaches talk about is real and measurable.
- **Cohen's d for the tackles shift between 1991–2010 and the modern era is +0.41** — a moderate effect size in statistical terms. For context, that's a bigger jump than nearly every other player metric across any era boundary.

### What the numbers don't tell us

A few honest caveats — because nothing about the past 125 years is a controlled experiment.

- **Hit-outs jumped from ~10 per ruckman/game in 2016 to ~17 in 2017** and stayed there. That's almost certainly a recording-method change at AFL Stats, not a sudden ruck revolution. Treat any hit-out comparison crossing 2016/2017 with a grain of salt.
- **Tackles in 1965–1990** look low (1.86/g) partly because the AFL didn't actually start counting tackles until 1987. The mean is dragged down by 22 years of missing data inside that bucket. The real "old footy" baseline is roughly the 1987–1990 sub-window.
- **2026 is partial** — only the first 8 rounds — so it's included in the 2011–present bucket but doesn't change the pattern materially.
- **Statistical tests are reported with a "rough indicator" caveat.** The 687k player-game rows aren't independent (same players appear hundreds of times), so the p-values from the Welch tests in `data/era_significance_tests.csv` should be read alongside the effect sizes (Cohen's d), which are more honest about practical significance.

The full era-by-era breakdown lives in `data/era_stats.csv`, the matches-level scoring numbers in `data/era_team_scoring.csv`, and the year-by-year trends in `data/era_yearly_trends.csv` — all rebuilt by running `era_based_statistical_analysis.py`.

---
**Related:** [2026 live season data](afl-season-2026.md) · [5-year team profiles](afl-team-profiles.md)
