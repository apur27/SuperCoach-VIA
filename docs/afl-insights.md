# AFL insights

> [← Back to main README](../README.md)

Live season data, historical analysis, and guides for footy fans and coaches.

## What's in here

| Section | Description |
|---------|-------------|
| [2026 live season data](afl-season-2026.md) | Auto-updating team analysis, finals pathway, Brownlow predictor, and player stats |
| [5-year team profiles](afl-team-profiles.md) | How each team has played across the last 5 seasons |
| [AFL history - 125 years of data](afl-history.md) | Scoring trends, player workload evolution, era analysis |
| [For the footy expert](footy-expert-guide.md) | Challenging the all-time top-100 ranking, using Claude for deeper questions |
| [For the coaching staff](coaching-guide.md) | Data-driven game planning workflow with Claude and Scientist |
| [AFL 2026 list quality and draft pipeline](news/2026-06-17-afl-2026-list-quality-draft-pipeline.md) | All 18 clubs: squad union R1–R15, National Draft and Rookie Draft pedigree, A+–D grades, and free agency outlook — data-gated |
| [Coaches Strategy Corner](coaches-strategy-corner/README.md) | Pre-game tactical briefs grounded entirely in the dataset |
| → **[Richmond vs Adelaide R9 - executive summary](coaches-strategy-corner/richmond-vs-adelaide-round-9-2026-executive-summary.md)** | The latest brief: 1-page entry point with charts, key matchups, and win conditions |

## Round 25 — Week in Review

**Disposal leaders (rounds 1–25):** Nick Daicos (Collingwood) tops the per-game average at **34.9** **[data]**, ahead of Bailey Smith (Geelong, **32.6** **[data]**) and Errol Gulden (Sydney, **31.5** **[data]**). The samples are not equal: Gulden's average comes from just **11** **[data]** games, against **22** **[data]** each for Daicos and Smith. Harry Sheezel (North Melbourne, **30.4** **[data]** from **22** **[data]** games) and Clayton Oliver (Greater Western Sydney, **30.2** **[data]** from **23** **[data]** games) complete the top five. The league mean is **14.90** **[data]**, across **618** **[data]** eligible players with at least **3** **[data]** games.

**Team form:** Sydney lead on average score (**110.2** **[data]**) and on average margin across all games, wins and losses combined (**+28.6** **[data]**). Fremantle are second on that margin at **+22.5** **[data]**, with a lower average score of **97.0** **[data]**, so their margin leans on limiting opponents.

**Tactical note:** Clearances and contested possessions are strongly correlated this season (r = **+0.75** **[data]**), but the two overlap by definition, since a clearance is itself a contested-possession event. The relationship is partly mechanical, not an independent finding.

*Methodology: player leaderboards, the league mean, the eligibility floor, team score and margin averages, and the clearances/contested-possessions correlation come from `docs/afl-stat-leaders-2026.md`. Margin there is averaged over all games, not wins only. Games-played counts come from the same rounds window, in `data/player_data/daicos_nick_03012003_performance_details.csv`, `data/player_data/smith_bailey_07122000_performance_details.csv`, `data/player_data/gulden_errol_18072002_performance_details.csv`, `data/player_data/sheezel_harry_13102004_performance_details.csv` and `data/player_data/oliver_clayton_22071997_performance_details.csv`.*

The **FootyStrategy agent** (`@"FootyStrategy (agent)"` in Claude Code) complements Scientist with AFL tactical knowledge - use it to interpret what the data means on the ground. See [coaching-guide.md](coaching-guide.md#leveraging-the-footystrategy-agent) for the full workflow.
