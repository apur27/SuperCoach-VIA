# Start here - no code required

> [← Back to main README](../README.md)

A landing page for footy fans. Nothing on this page requires Python, a terminal, or a GitHub account. Just a browser.

If you have ever asked "who's getting the ball this week?", "is this player worth trading in?", or "how does this season compare to history?" - this is a useful place to spend ten minutes.

---

## What you can do without writing any code

### See this round's predicted disposal leaders

The model produces a fresh prediction for the upcoming round each week, and the top 30 are formatted into a doc you can read in your browser.

→ [Round 9 cheat sheet (current round)](weekly/round-current-2026.md) - one-page top 30 plus per-club top 3.

→ [Full predictions doc (auto-updated)](afl-predictions-2026.md) - longer leaderboard, with last 5 rounds form trend per player.

> The hosted dashboard (a clickable web view of the predictions) is **coming soon** - tracked under [Phase 2 Rung 7](../README.md#whats-coming---phase-2). For now, the markdown docs above are the fan-facing surface.

---

### Download the prediction CSV (Excel / Google Sheets)

If you want to sort, filter, or build your own watchlist around the predictions, you can grab the raw CSV without installing anything.

1. Go to the latest CSV in **`data/prediction/`** in the repo (the most recent `next_round_*.csv`).
2. Click **Raw** on GitHub, then save the page as a `.csv`.
3. Open it in Google Sheets, Excel, or Numbers.

→ [Step-by-step instructions](installation.md#for-power-users---download-the-data-no-coding).

→ [Google Sheets template](../templates/google-sheets-template.md) - turn the CSV into a 5-tab dashboard (Latest predictions, Player comparison, My watchlist, Club filter, Model confidence). 30-second weekly refresh once it's set up.

---

### Read club analysis - who's in form, who's leaking

The repo auto-updates per-club analysis every weekend after the round.

→ [2026 team analysis](afl-team-analysis-2026.md) - which teams are scoring, conceding, and trending.

→ [5-year team profiles](afl-team-profiles.md) - playing-style summary for each club, generated from the data.

→ [Coaches Strategy Corner](coaches-strategy-corner/README.md) - match-by-match tactical briefs built from the data.

---

### Argue the greatest-of-all-time debate, with the receipts

The repo carries every match and player stat going back to 1897. The Hall of Fame is the result of running rank-based, era-fair statistical aggregation across the lot.

→ [All-time top 100 (Hall of Fame)](hall-of-fame.md) - the master list and the methodology.

→ [Hall of Fame - statistical leaders](hall-of-fame-stat-leaders.md) - top 20 in every major category, verified from data.

→ [Hall of Fame - the captains](hall-of-fame-captains.md) - greatest leaders in the game.

→ [Hall of Fame - courageous players](hall-of-fame-courageous.md) - on-field physical and mental courage.

→ [Hall of Fame - careers cut short](hall-of-fame-careers-cut-short.md) - the ones who could have been.

→ [Hall of Fame - dynasties](hall-of-fame-dynasties.md) - the great team eras.

→ [Hall of Fame - Indigenous Australian players](hall-of-fame-indigenous.md) - top 10.

→ [Hall of Fame - coaches](hall-of-fame-coaches.md) - the masters on the bench.

---

### Understand what the predictions are - and aren't

This is the honest version. Read it before you trade in your captain on the strength of a single CSV row.

→ [How to use this for SuperCoach](how-to-use-this-for-supercoach.md) - what it helps you answer (form, ball-winning, role), what it does NOT include (injuries, late changes, prices, breakevens, Champion Data).

→ [Glossary](glossary.md) - footy and data terms in plain English (disposals, clearances, MAE, calibration, GroupKFold).

→ [Model report card](model-report-card.md) - pre-registered hit/miss methodology and weekly accuracy log.

---

### Help us improve

The most valuable thing a fan can do is tell us when the model is wrong about a player. Your eye sees role changes and tag jobs faster than the model.

→ [Email us](mailto:careerabhi@gmail.com) — tell us when the model is wrong about a player, report a bug, or request a feature. Plain English, no account required.

---

## Quick reality check before you trust any prediction

1. **Predicted disposals are not SuperCoach points.** Disposals are a strong proxy for a midfielder's score, but the model does not include marks, tackles, goals, hit-outs, frees, or score multipliers.
2. **Typical error is ±4 disposals.** A predicted 28 means the realistic range is roughly 20-36. Use predictions as a tilt, not a guarantee.
3. **Late outs are not handled.** The model does not know who is named or rested. Always check the team list before lockout.
4. **The model is slow on role changes and tag jobs.** If you can see a player has been moved into defence, trust your eye.

---

## Coming soon

- **Hosted web dashboard** - a clickable view of the predictions you can browse in any browser, no CSV download needed. Tracked in [Phase 2 Rung 7](../README.md#whats-coming---phase-2).
- **Weekly fan pack as a downloadable bundle** - one zip per week containing the prediction CSV, the cheat sheet, the backtest, and the glossary. Scaffolded under [.github/workflows/weekly-fan-pack.yml](../.github/workflows/weekly-fan-pack.yml).
- **Player cards** - one PNG per top player with predicted disposals and trend arrow, suitable for sharing. Scaffolded under [scripts/generate_player_cards.py](../scripts/generate_player_cards.py).

---

## Related

- [Main README](../README.md)
- [How to use this for SuperCoach](how-to-use-this-for-supercoach.md)
- [Glossary](glossary.md)
- [Latest predictions](afl-predictions-2026.md)
- [Backtest results](afl-backtest-2026.md)
- [Hall of Fame](hall-of-fame.md)
