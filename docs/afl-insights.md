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

## Round 24 — Week in Review

**Disposal leaders (rounds 1–23):** Nick Daicos (Collingwood) leads the competition at **35.2 per game** **[data]**, ahead of Bailey Smith (Geelong) at **32.2** **[data]** and Harry Sheezel (North Melbourne) at **31.1** **[data]** — well clear of the league mean of **14.92 per game** **[data]**.

**Team form:** Fremantle's average winning margin of **+30.1** **[data]** is the highest in the competition, just ahead of Sydney's **+27.9** **[data]**; Sydney also leads average score at **109.9 per game** **[data]** — the two clearest form lines as the season nears its end.

**Watch in Round 24:** Nick Daicos is again the model's top projected disposal getter at **30.0** **[data]**, with Harry Sheezel and Nasiah Wanganeen-Milera (St Kilda) next at **29.0** each **[data]**. The cheat sheet's typical error is **±4 disposals** **[data]**, so treat rankings as a shortlist, not a lock.

**Tactical note:** Marks inside 50 is the strongest single predictor of goal output in the correlation data (r = **+0.67** **[data]**), and Jack Gunston (Hawthorn) tops both leaderboards this season — **4.27** marks inside 50 per game **[data]** and **3.47** goals per game **[data]** — a pairing that shows association between forward positioning and scoreboard impact, not that either stat causes the other.

*Methodology: disposal-leader figures, the league mean, team margin/score averages, and the marks-inside-50/goals correlation and leaderboards are read from the rounds 1–23 per-game aggregates in `docs/afl-stat-leaders-2026.md` (auto-generated from `data/player_data/` and `data/matches/matches_2026.csv`). Round 24 disposal projections and the ±4 disposal error figure are read from `docs/afl-predictions-2026.md` and `docs/weekly/round-current-2026.md`, both sourced from `data/prediction/next_round_24_prediction_20260811_1028.csv`.*

The **FootyStrategy agent** (`@"FootyStrategy (agent)"` in Claude Code) complements Scientist with AFL tactical knowledge - use it to interpret what the data means on the ground. See [coaching-guide.md](coaching-guide.md#leveraging-the-footystrategy-agent) for the full workflow.
