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

## Round 9 — Week in Review

**Disposal leaders (rounds 1–16):** Nick Daicos (Collingwood) leads the competition at **35.4 per game** [data], with Bailey Smith (Geelong) at **32.3** [data] and Clayton Oliver (GWS) at **31.5** [data] — all more than double the league mean of **15.2 per game** [data].

**Ladder:** Sydney sit atop the margin table with a **+36.3-point average margin** [data], 4.1 points clear of Fremantle (**+32.2** [data]) — the two-team tier at the top is the clearest structural shift of the half-season.

**Watch in Round 17:** Nick Daicos (Collingwood) is the model's top projected disposal getter at **29.0** [data] — the only player reaching that threshold in next-round forecasts, consistent with his league-leading season average.

**Tactical note:** Marks inside 50 is the dataset's strongest single predictor of forward output (r = +0.67 with goals [data]); Jack Gunston leads that leaderboard at **4.56 per game** [data], with Hawthorn's **99.1-point scoring average** [data] reflecting the compounding value of manufacturing marking contests inside the arc.

The **FootyStrategy agent** (`@"FootyStrategy (agent)"` in Claude Code) complements Scientist with AFL tactical knowledge - use it to interpret what the data means on the ground. See [coaching-guide.md](coaching-guide.md#leveraging-the-footystrategy-agent) for the full workflow.
