# Coaches Strategy Corner

> [← Back to AFL insights](../afl-insights.md) | [← Back to main README](../../README.md)

Pre-match tactical briefs built end-to-end from the SuperCoach-VIA dataset. Every brief is forward-looking - written for an upcoming fixture, before the bounce. Every recommendation is grounded in a number from completed matches: a per-game average, a head-to-head record, a venue split, a 2026 form curve.

This is what happens when you point an AI agent and a footy-literate analyst at 130 years of structured match data and ask: **how do we beat them on Saturday?**

## What this is

A coaching brief, not a preview. The audience is the head coach walking into Tuesday's review or the senior assistant building the Friday team meeting. The questions answered are the ones a high-performance department actually asks:

- Who are they in 2026, and where can they be hurt?
- Who are we, and which of our strengths actually moves the scoreboard?
- What does the head-to-head and venue history tell us - signal, not folklore?
- Which 4–5 individual matchups decide this game?
- What 10 things must we get right by quarter time?

Tactical recommendations cite a specific stat. Verified numbers are tagged `**[data]**`; historical/contextual facts are tagged `**[historical record]**`. Nothing here is "vibes."

## What this is not

- Not a betting card. We do not predict winners; we map how the game gets won.
- Not a substitute for video. The dataset is box-score and result-level - no GPS, no spatial, no pressure tags. Where the answer requires those, we say so.
- Not a list system review. Player ratings here are 2026-form-based and matchup-relevant, not legacy or potential.

## Available briefs

| Match | Round | Venue | Date (scheduled) | Executive summary | Full brief | Supporting docs | Live reads |
|-------|-------|-------|------------------|-------------------|------------|-----------------|------------|
| Richmond vs St Kilda | Round 11 | Marvel Stadium | 17 May 2026, 3:15pm | [Summary](richmond-vs-stkilda-round-11-2026-executive-summary.md) | [Tactical brief](richmond-vs-stkilda-round-11-2026.md) | [Player matchups](richmond-vs-stkilda-round-11-2026-player-matchups.md) · [H2H history](richmond-vs-stkilda-round-11-2026-head-to-head-history.md) | [Q1](richmond-vs-stkilda-round-11-2026-q1-live.md) · [Q2](richmond-vs-stkilda-round-11-2026-q2-live.md) · [Half-time](richmond-vs-stkilda-round-11-2026-half-time-live.md) · [Q3](richmond-vs-stkilda-round-11-2026-q3-live.md) · [Q4](richmond-vs-stkilda-round-11-2026-q4-live.md) · [Full-time verdict](richmond-vs-stkilda-round-11-2026-full-time-verdict.md) · [Post-mortem](richmond-vs-stkilda-round-11-2026-postmortem.md) |
| Richmond vs Adelaide | Round 9 | M.C.G. | 10 May 2026, 3:15pm | [Summary](richmond-vs-adelaide-round-9-2026-executive-summary.md) | [Tactical brief](richmond-vs-adelaide-round-9-2026.md) | [Player matchups](richmond-vs-adelaide-round-9-2026-player-matchups.md) · [H2H history](richmond-vs-adelaide-round-9-2026-head-to-head-history.md) | [Half-time](richmond-vs-adelaide-round-9-2026-half-time-live.md) · [Q3](richmond-vs-adelaide-round-9-2026-q3-live.md) · [Q4 + full-time](richmond-vs-adelaide-round-9-2026-q4-live.md) · [Verdict](richmond-vs-adelaide-round-9-2026-full-time-verdict.md) · [Post-mortem (data)](richmond-vs-adelaide-round-9-2026-postmortem.md) · [Post-mortem (tactical)](richmond-vs-adelaide-round-9-2026-postmortem-footystrategy.md) |

### Live reads

From Round 9, briefs are now supplemented with **in-game live reads** pulled directly from the FanFooty live feed (`fanfooty.com.au/live/<gameid>.txt`) using `scripts/fetch_live_match.py`. A live read compares pre-game predictions against in-game reality - who was right, who was wrong, and what the second half should look like. Snapshots are saved to `data/live_snapshots/` for reproducibility. Live reads can be generated at any point during a game: half-time, end of Q3, or whenever a tactical shift warrants a fresh look.

### FootyStrategy - tactical brainstorm layer

After each live read, the **FootyStrategy agent** provides a tactical interpretation of what the data shows - converting structural anomalies (Q3 AF collapses, positional resets, tackle-interaction effects) into coaching-language answers. FootyStrategy is invoked with `@"FootyStrategy (agent)"` and works alongside Scientist: Scientist reads the numbers, FootyStrategy answers "what does a coach actually do about this?" The post-match brainstorm for Richmond vs Adelaide R9 is the worked example - see the [full-time verdict](richmond-vs-adelaide-round-9-2026-full-time-verdict.md).

For list-construction analysis of each club (who their Tier 1 players are, what draft pick they were, how list depth explains their tactical identity), see the [AFL 2026 team list analysis](afl-2026-team-list-analysis.md).

## How to run a live match analysis

End-to-end recipe for reproducing the Round 9 2026 Richmond vs Adelaide live pipeline. All commands assume the repo root as the working directory and use the project venv: `/home/abhi/sourceCode/python/coding/.venv/bin/python`.

1. **Find the FanFooty game ID.** Open the live page on fanfooty.com.au; the URL is shaped `fanfooty.com.au/live/2026/<gameid>-<team-slug>.html`. The numeric `<gameid>` (e.g. `9781`) is the only argument the fetch script needs.

2. **Fetch a snapshot.** One-liner:

   ```bash
   /home/abhi/sourceCode/python/coding/.venv/bin/python scripts/fetch_live_match.py <gameid>
   ```

   The script pulls `https://www.fanfooty.com.au/live/<gameid>.txt`, parses the 65-column player rows, and writes two artifacts to `data/live_snapshots/`:
   - `<gameid>_<YYYYMMDD_HHMM>_<status>.json` - full structured snapshot (header, meta, commentary, players)
   - `<gameid>_<YYYYMMDD_HHMM>_players.csv` - the player table on its own

3. **Verify the snapshot.** On success the script prints the score, round, venue, player count, and `Schema sentry: passed`, followed by the top 5 disposal-getters per side. Two hard sentries run on every fetch:
   - **Column count**: every player row must have exactly 65 columns (catches feed schema drift).
   - **Quarter-sum**: per-player Q1+Q2+Q3+Q4 AF (cols 46/48/50/52) must equal total AF (col 5). A failure means the columns have shifted and downstream parsing is unsafe - fix before continuing.

4. **Known data quality issue: col15 ("goals") is unreliable.** The team sum of col15 does not match the actual scoreline. Do **not** attribute individual goals from this column. For score and goalkicker attribution, use the match header (`home_score` / `away_score`) and the m0nty commentary stream in the JSON snapshot. Disposals (col10 + col11), tackles (col13), marks (col12), and hitouts (col14) are reliable.

5. **Create the live read doc.** Naming convention, one file per quarter milestone:
   - `docs/coaches-strategy-corner/<match-slug>-half-time-live.md`
   - `docs/coaches-strategy-corner/<match-slug>-q3-live.md`
   - `docs/coaches-strategy-corner/<match-slug>-q4-live.md`
   - `docs/coaches-strategy-corner/<match-slug>-full-time-verdict.md` (separate post-match wrap)

   `<match-slug>` matches the brief (e.g. `richmond-vs-adelaide-round-9-2026`). Add or update the **Live reads** column in the table at the top of this README so each new doc is one click away.

6. **Run the analysis loop.** Two options:
   - **Manual**: re-run step 2 every ~90s, then update the active live read doc with what changed. Best for tight tactical commentary where each snapshot deserves a written read.
   - **Automated polling**: `scripts/live_match_monitor.py` wraps the fetch on a 90-second interval and stops automatically when status flips to `Full Time`:

     ```bash
     /home/abhi/sourceCode/python/coding/.venv/bin/python scripts/live_match_monitor.py \
       <gameid> docs/coaches-strategy-corner/<match-slug>-q4-live.md
     ```

7. **Quarter milestones.** Write a fresh doc at half-time, end of Q3, and end of Q4. Keep each doc focused on what changed in that period vs the pre-match brief - not a running log. After full-time, write the **full-time verdict** as a separate file and link it from the bottom of the Q4 doc.

8. **Commit and push.** GitHub-rendered docs are the live read; push after every meaningful update so readers see the latest version:

   ```bash
   git add docs/coaches-strategy-corner/<match-slug>-*-live.md docs/coaches-strategy-corner/README.md data/live_snapshots/
   git commit -m "Live read: <match-slug> <milestone>"
   git push origin main
   ```

### How this section works

Each brief is built from the repo's historical match, lineup, and player-performance data - see the methodology paragraph below for the underlying sources. Charts referenced inside a brief live under `assets/charts/strategy/<match-slug>/` (one folder per fixture, so briefs do not collide with each other). When a new fixture is added, a row is appended to the table above; nothing else on this page needs to change. If you only have a minute, the **Executive summary** column is the right place to start - every fuller doc is one click away from there.

## Methodology in one paragraph

Every brief draws from `data/matches/matches_*.csv` (130 seasons of results, quarter-by-quarter), `data/player_data/*_performance_details.csv` (per-game player box scores), and `data/lineups/team_lineups_*.csv` (round-by-round selected sides). League ranks are computed across all 18 teams from 2026 round-level aggregates. H2H records cover every meeting on the books. Venue records filter the matches table by venue string. Player consistency is the coefficient of variation (std/mean) of disposals - lower is steadier. Player trend is the linear slope of disposals across rounds. Where a number is unreliable (small sample, missing column for that era), it is flagged or excluded.

The 2026 form numbers in each brief are computed from results that have already been played as of the date the brief was written - that is, from the most recently completed round backwards. Anything about the upcoming match itself (predicted matchups, recommended structures, expected pressure points) is forward-looking by definition. Recommendations are not retrospectives.

## How a brief is structured

Every brief follows the same skeleton so that the reader knows where to look:

1. **Executive summary** - three bullets, the spine of the game plan
2. **The opposition** - identity, stat profile, best players, vulnerabilities
3. **Us** - identity, stat profile, best performers, areas to sharpen
4. **Head-to-head** - all-time, MCG-specific, last 5 meetings (full detail in the side doc)
5. **Tactical blueprint** - five concrete plans (contested ball, forward entries, defensive setup, stoppages, set pieces)
6. **Player-by-player matchups** - full detail in the side doc
7. **Pre-game checklist** - 10 items
8. **What to watch at quarter-time** - what tells you the plan is working

If you only have 90 seconds, read sections 1, 5, and 8.
