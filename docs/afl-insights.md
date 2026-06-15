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
| [AFL 2026 team list analysis](coaches-strategy-corner/afl-2026-team-list-analysis.md) | All 18 clubs analysed by list quality, Tier 1 draft pedigree, and how list construction explains tactical identity - FootyStrategy agent output |
| [Coaches Strategy Corner](coaches-strategy-corner/README.md) | Pre-game tactical briefs grounded entirely in the dataset |
| → **[Richmond vs Adelaide R9 - executive summary](coaches-strategy-corner/richmond-vs-adelaide-round-9-2026-executive-summary.md)** | The latest brief: 1-page entry point with charts, key matchups, and win conditions |

## Round 9 — Week in Review

**Disposal leaders (season through Round 15):** Nick Daicos (Collingwood) leads the competition at **34.9 per game** [data], with Bailey Smith (Geelong) at **32.3** [data] and Clayton Oliver (GWS) at **31.7** [data] — all more than double the league average of 15.3 [data].

**Ladder:** Sydney's average winning margin of **+36.3 points** [data] places them 2.1 points clear of second-placed Fremantle (**+34.2** [data]), with a steep drop to Geelong in third (**+16.1** [data]) — the clearest sign of a two-team tier forming at the top.

**Watch in Round 16:** Bailey Smith (Geelong) tops the model's next-round projections at **29.0 predicted disposals** [data] — consistent with his 32.3 season average and second-ranked inside-50 delivery rate (**6.86 per game** [data]).

**Tactical note:** Marks inside 50 carries the strongest predictive link to forward output in this dataset (r = +0.68 with goals [data]) — teams generating marking contests inside the arc, not just entries, are where the scoring gaps between the ladder's top and bottom have opened.

The **FootyStrategy agent** (`@"FootyStrategy (agent)"` in Claude Code) complements Scientist with AFL tactical knowledge - use it to interpret what the data means on the ground. See [coaching-guide.md](coaching-guide.md#leveraging-the-footystrategy-agent) for the full workflow.
