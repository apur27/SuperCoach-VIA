# How to use this for SuperCoach - the honest version

> [← Back to main README](../README.md) | [← Back to fan landing page](start-here-no-code.md)

This page is the plain-English bridge between what this repo does and what an actual SuperCoach player wants to do on a Sunday afternoon. No Python required to read it. No assumed background.

---

## What this repo actually does

It downloads every AFL match and player stat ever recorded (1897 onward), trains a machine-learning model on the last decade of player-game data, and produces a weekly **predicted disposal count** for the next round.

The output is a CSV file with three columns: `player`, `team`, `predicted_disposals`. Roughly one row per player expected to play.

That CSV - plus the auto-updated stat leaders, team trends, and historical Hall of Fame docs - is the deliverable.

---

## What it helps you answer

These are the questions the predictions and the team profiles are reasonably good at:

### Disposal-related questions

- "Of the inside mids in my squad, who's projected to have the highest ceiling this round?"
- "Player A is averaging 28 disposals this year. Should I expect that this week given the matchup?"
- "Is this midfielder trending up or down going into the bye?"

The model handles rolling form (3-game, 5-game, season-to-date) and opponent strength. Latest 2026 backtest performance is around **MAE 4.1 disposals** with **68% of predictions within 5 disposals** of actual.

### Form and trend questions

- "Which midfielders have improved most over the last 5 rounds?"
- "How does Team X's average disposal output compare to last year?"
- "Who are the top 30 ball-winners right now?"

The auto-updated docs ([Stat leaders](afl-stat-leaders-2026.md), [Team analysis](afl-team-analysis-2026.md), [5-year team profiles](afl-team-profiles.md)) cover this with no Python required.

### Historical-context questions

- "Has Player X's career trajectory hit its peak yet, or is there room?"
- "How does this season's leading goalkicker compare to all-time leaders?"
- "Which all-time greats had similar early-career numbers?"

The [Hall of Fame](hall-of-fame.md) and [Hall of Fame stat leaders](hall-of-fame-stat-leaders.md) docs answer these.

---

## What it does NOT include

This is the honest part. A few things this repo cannot give you, and probably never will:

### No fantasy points modelling

The model predicts **raw disposals**. SuperCoach scoring blends disposals with marks, tackles, goals, hit-outs, frees, and a bunch of multipliers. We do not predict fantasy points directly. You can use disposals as a strong proxy for midfielder output, but it is not the score itself.

### No injury or availability data

We do not know who is injured, suspended, or being managed. If a player is named but a late out, the prediction stays in the file. Always cross-check against the official team list before lockout.

### No GPS, video, or spatial data

The dataset is box-score only. We can tell you a player got 25 disposals; we cannot tell you where on the ground they got them, how many were under pressure, or how fast they ran. Anything that needs that level of detail (champion-data-style metrics) is not here.

### No price or trade-value modelling

The model has no idea what a player costs in SuperCoach this week, what their break-even is, or whether their price is about to crash. If you want price-and-value tools, this is not that. This is "what will their on-field stats look like."

### No live odds, weather, or in-game state

We do not adjust for weather, expected congestion, or live betting markets. Recent rain can shift disposal counts; the model does not see that.

### No "captaincy" recommendation

We do not pick your captain. The prediction tells you who is likely to score well; **you** decide whether the form, role, and matchup justify the captain's armband. Tools should inform decisions, not make them.

### Predictions are not certainty

Even the best AFL forecast model has an MAE of around 4 disposals - and individual rounds can be much wider than that. A prediction of 28 disposals with MAE 4 means the realistic range is something like 20-36. Use predictions as a tilt, not a guarantee.

---

## A practical Sunday workflow

This is the closest thing to "what do I actually do with this":

1. **Friday or Saturday morning** - pull the latest prediction CSV from `data/prediction/` (the most recent `next_round_*` file).
2. **Open it in Google Sheets** using the [template](../templates/google-sheets-template.md). Sort by `predicted_disposals` descending.
3. **Cross-reference with your watchlist** - filter to players you're considering trading in or captaining.
4. **Sanity check** - look at [the season stat leaders](afl-stat-leaders-2026.md) to confirm the prediction matches recent form. If a player's predicted at 28 but they've gone 18, 16, 19 the last three weeks, the model thinks something is reverting - but you should ask **why**.
5. **Check the team list** before lockout. Confirm everyone is named.
6. **Decide** based on prediction + form + role + matchup, not prediction alone.

That's the workflow. Nothing magical.

---

## When the prediction is wrong

It will be. About 32% of predictions are off by more than 5 disposals. The most common reasons:

- **Role change** - the player has moved off-ball, into defence, or been deployed differently. The model is slow to catch up.
- **Tag** - opposition has put a hard run-with on them. Disposals fall.
- **Late out / injury managed** - they came off after a quarter.
- **Weather / wet ball** - low-disposal game across the board.
- **Hot or cold streak** - regression to the mean works both ways. Sometimes a player runs hot beyond what form alone suggests.

The [backtest results page](afl-backtest-2026.md) bolds the players where the model has been most consistently off, with the average error. That's the watchlist of "model has not figured this player out yet."

---

## Bottom line

This repo is **a research and analysis tool**, not a fantasy-football oracle. It is honest about its accuracy, transparent about its methodology, and intentionally limited to what the data can defensibly tell you.

If you treat it like one input among several - alongside your eye, your team-list checks, and your read on the matchup - it will probably make you a slightly sharper SuperCoach player over time.

If you treat it like a magic 8-ball, you will be disappointed. That is true of every model, including the ones you pay for.

---

## Related

- [Start here - no code](start-here-no-code.md) - the fan landing page
- [Glossary](glossary.md) - footy and data terms in plain English
- [Latest predictions](afl-predictions-2026.md) - this round's predicted disposal leaders
- [Backtest results](afl-backtest-2026.md) - how accurate the model has been so far
- [Model report card](model-report-card.md) - hit/miss methodology
- [Google Sheets template](../templates/google-sheets-template.md) - turn the CSV into a working dashboard
