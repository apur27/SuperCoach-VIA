# Coaches Strategy Corner

> [← Back to AFL insights](../afl-insights.md) | [← Back to main README](../../README.md)

Pre-match tactical briefs built end-to-end from the SuperCoach-VIA dataset. Every brief is forward-looking — written for an upcoming fixture, before the bounce. Every recommendation is grounded in a number from completed matches: a per-game average, a head-to-head record, a venue split, a 2026 form curve.

This is what happens when you point an AI agent and a footy-literate analyst at 130 years of structured match data and ask: **how do we beat them on Saturday?**

## What this is

A coaching brief, not a preview. The audience is the head coach walking into Tuesday's review or the senior assistant building the Friday team meeting. The questions answered are the ones a high-performance department actually asks:

- Who are they in 2026, and where can they be hurt?
- Who are we, and which of our strengths actually moves the scoreboard?
- What does the head-to-head and venue history tell us — signal, not folklore?
- Which 4–5 individual matchups decide this game?
- What 10 things must we get right by quarter time?

Tactical recommendations cite a specific stat. Verified numbers are tagged `**[data]**`; historical/contextual facts are tagged `**[historical record]**`. Nothing here is "vibes."

## What this is not

- Not a betting card. We do not predict winners; we map how the game gets won.
- Not a substitute for video. The dataset is box-score and result-level — no GPS, no spatial, no pressure tags. Where the answer requires those, we say so.
- Not a list system review. Player ratings here are 2026-form-based and matchup-relevant, not legacy or potential.

## Available briefs

| Match | Round | Venue | Date (scheduled) | Brief |
|-------|-------|-------|------|-------|
| Richmond vs Adelaide | Round 9 | M.C.G. | 10 May 2026, 3:15pm (upcoming) | [Tactical brief](richmond-vs-adelaide-round-9-2026.md) · [Player matchups](richmond-vs-adelaide-round-9-2026-player-matchups.md) · [H2H history](richmond-vs-adelaide-round-9-2026-head-to-head-history.md) |

## Methodology in one paragraph

Every brief draws from `data/matches/matches_*.csv` (130 seasons of results, quarter-by-quarter), `data/player_data/*_performance_details.csv` (per-game player box scores), and `data/lineups/team_lineups_*.csv` (round-by-round selected sides). League ranks are computed across all 18 teams from 2026 round-level aggregates. H2H records cover every meeting on the books. Venue records filter the matches table by venue string. Player consistency is the coefficient of variation (std/mean) of disposals — lower is steadier. Player trend is the linear slope of disposals across rounds. Where a number is unreliable (small sample, missing column for that era), it is flagged or excluded.

The 2026 form numbers in each brief are computed from results that have already been played as of the date the brief was written — that is, from the most recently completed round backwards. Anything about the upcoming match itself (predicted matchups, recommended structures, expected pressure points) is forward-looking by definition. Recommendations are not retrospectives.

## How a brief is structured

Every brief follows the same skeleton so that the reader knows where to look:

1. **Executive summary** — three bullets, the spine of the game plan
2. **The opposition** — identity, stat profile, best players, vulnerabilities
3. **Us** — identity, stat profile, best performers, areas to sharpen
4. **Head-to-head** — all-time, MCG-specific, last 5 meetings (full detail in the side doc)
5. **Tactical blueprint** — five concrete plans (contested ball, forward entries, defensive setup, stoppages, set pieces)
6. **Player-by-player matchups** — full detail in the side doc
7. **Pre-game checklist** — 10 items
8. **What to watch at quarter-time** — what tells you the plan is working

If you only have 90 seconds, read sections 1, 5, and 8.
