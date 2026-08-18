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

**Disposal leaders (rounds 1–24):** Nick Daicos (Collingwood) leads the competition at **35.3 per game** **[data]**, ahead of Bailey Smith (Geelong) at **32.3** **[data]**; Errol Gulden (Sydney), Clayton Oliver (Greater Western Sydney) and Harry Sheezel (North Melbourne) are effectively tied for third at **30.5** **[data]** each — though Gulden's figure comes from just **10** **[data]** games this season versus **22** **[data]** each for Oliver and Sheezel — well clear of the league mean of **14.91 per game** **[data]** across **611** **[data]** eligible players with at least **3** **[data]** games.

**Team form:** Fremantle's average margin across all games this season is the highest in the competition at **+29.9** **[data]**, just ahead of Sydney's **+28.2** **[data]**. Sydney separately leads the competition in average score, at **110.0 per game** **[data]**.

**Watch in Round 25:** Harry Sheezel (North Melbourne) and Errol Gulden share the model's top projected disposal count at **29.0** each **[data]**, with Nick Daicos, Lachie Neale and Clayton Oliver next at **28.0** each **[data]**. That Daicos figure sits well below his season average of **35.3 per game** **[data]** — a **7.3**-disposal gap **[data]** wider than the cheat sheet's typical error of **±4 disposals** **[data]** — a reminder that these projections regress toward the league mean, running below season averages for the highest-volume players and above them for low-volume ones. Treat rankings as a shortlist, not a lock.

**Tactical note:** Clearances correlate strongly with contested possessions in this season's data (r = **+0.75** **[data]**) — an association between stoppage craft and ball-winning, not evidence that either stat causes the other.

*Methodology: disposal-leader figures, the league mean and eligibility threshold, and team score/margin averages (margin is averaged across ALL games, not wins only) are read from the rounds 1–24 per-game aggregates in `docs/afl-stat-leaders-2026.md` (auto-generated from `data/player_data/` and `data/matches/matches_2026.csv`), which also supplies the clearances/contested-possessions correlation. Games-played counts behind the third-place disposal tie are read directly from `data/player_data/gulden_errol_18072002_performance_details.csv`, `data/player_data/oliver_clayton_22071997_performance_details.csv`, and `data/player_data/sheezel_harry_13102004_performance_details.csv`. Round 25 disposal projections and the disposal error figure are read from `docs/afl-predictions-2026.md` and `docs/weekly/round-current-2026.md`, both sourced from `data/prediction/next_round_25_prediction_20260818_1146.csv`.*

The **FootyStrategy agent** (`@"FootyStrategy (agent)"` in Claude Code) complements Scientist with AFL tactical knowledge - use it to interpret what the data means on the ground. See [coaching-guide.md](coaching-guide.md#leveraging-the-footystrategy-agent) for the full workflow.
