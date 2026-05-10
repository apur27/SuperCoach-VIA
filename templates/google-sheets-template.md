# Google Sheets template - turn the prediction CSV into a working dashboard

> [← Back to main README](../README.md) | [← How to use this for SuperCoach](../docs/how-to-use-this-for-supercoach.md)

This page describes how to set up a free Google Sheets workbook that turns the weekly prediction CSV into a five-tab dashboard. No coding. Once it is set up, refreshing for a new round is roughly a 30-second copy-paste.

---

## What you need

- A free Google account
- The latest prediction CSV from `data/prediction/` in this repo (the most recent file matching `next_round_<N>_prediction_<timestamp>.csv`)

The CSV has three columns:

| Column | Meaning |
|---|---|
| `player` | Player name in `Surname FirstName` format |
| `team` | AFL club name |
| `predicted_disposals` | Predicted disposal count for the next round (typically 5-30) |

---

## The five-tab structure

### Tab 1 - Latest predictions

The raw paste-target. Refresh this every week.

**How to set it up:**

1. New Google Sheet → rename the first tab to `Latest predictions`.
2. Open the prediction CSV and copy all rows including the header.
3. Paste into cell A1.
4. Highlight column C (`predicted_disposals`) → Format → Number → Number with 1 decimal.
5. Highlight A1:C1 → Format → Make first row bold and freeze it (View → Freeze → 1 row).
6. (Optional) Add a heat-map: select column C → Format → Conditional formatting → Color scale, with red→green mapping low→high.

**Weekly refresh:** open the new prediction CSV → copy all rows → paste over A1 in this tab. Other tabs auto-update because they pull from this one.

---

### Tab 2 - Player comparison

A side-by-side view of two or three specific players you're deciding between.

**How to set it up:**

1. New tab → name it `Player comparison`.
2. In A1: `Player name` (header). Below it (A2, A3, A4): the names of three players you're comparing, written exactly as they appear in the CSV (e.g., `Daicos Nick`).
3. In B1: `Predicted disposals`. In B2:

   ```
   =IFERROR(VLOOKUP(A2, 'Latest predictions'!A:C, 3, FALSE), "Not predicted")
   ```

   Drag down to B4.
4. In C1: `Team`. In C2:

   ```
   =IFERROR(VLOOKUP(A2, 'Latest predictions'!A:C, 2, FALSE), "—")
   ```

   Drag down.
5. Add notes column D for "captain pick?", "trade target?", whatever you want.

Now whenever the Latest predictions tab updates, this comparison auto-updates.

---

### Tab 3 - My watchlist

A growing list of players you're tracking - cash cows, premiums you might trade in, captain options.

**How to set it up:**

1. New tab → name it `My watchlist`.
2. Headers in row 1: `Player`, `Predicted`, `Team`, `Tag` (e.g., "captain pick", "cash cow", "POD"), `Notes`.
3. Add player names in column A.
4. Column B uses the same VLOOKUP as above against the Latest predictions tab.
5. Column C uses the team VLOOKUP.
6. Sort the tab by column B descending so the highest-projected player is always at the top.
7. (Optional) Conditional formatting on column B: highlight cells > 25 in green.

This is the tab you'll spend the most time in.

---

### Tab 4 - Club filter

A view of all predicted players for one specific club. Useful when you're worried about a single club's players (injuries, role changes, bye coming up).

**How to set it up:**

1. New tab → name it `Club filter`.
2. Cell A1: `Club name`. Cell A2: enter a club name (e.g., `Collingwood`).
3. Cell A4: paste this formula:

   ```
   =QUERY('Latest predictions'!A:C, "select A, B, C where B = '"&A2&"' order by C desc", 1)
   ```

4. Now whenever you change A2, the table below shows that club's predicted leaders.

Use this when you want to see "who from West Coast does the model think will lead disposals this week?" - useful for matchup analysis.

---

### Tab 5 - Model confidence

A reality-check tab. Reminds you of the model's accuracy and the typical error band so you don't read predictions as gospel.

**How to set it up:**

1. New tab → name it `Model confidence`.
2. Headers in row 1: `Metric`, `Value`, `Source`.
3. Manually paste in the latest backtest numbers from [afl-backtest-2026.md](../docs/afl-backtest-2026.md). Example:

   | Metric | Value | Source |
   |---|---|---|
   | MAE | 4.14 disposals | 2026 backtest, 8 rounds |
   | Within ±5 disposals | 68.1% | 2026 backtest, 8 rounds |
   | Within ±10 disposals | 94.2% | 2026 backtest, 8 rounds |
   | Bias | near 0 | 2026 backtest |
   | Last refresh | (paste date here) | data/prediction/ folder |

4. Below the table, write a note to yourself in plain English. Suggested:

   > **A predicted 28 disposals does NOT mean exactly 28.** The model's typical error is ±4 disposals. Treat predictions as a tilt, not a guarantee. Always check the team list and recent role for the player.

This tab is the conscience of the dashboard.

---

## Weekly refresh checklist

Once the workbook is set up, the weekly flow is:

1. Get the latest prediction CSV from `data/prediction/` in this repo.
2. Open `Latest predictions` tab → select all → paste new CSV contents.
3. Glance at `Player comparison`, `My watchlist`, `Club filter` - they auto-updated.
4. Update `Model confidence` if the backtest numbers in the [backtest doc](../docs/afl-backtest-2026.md) have moved.
5. Check the team list before lockout - the model does not know who is a late out.

Total time: about 30 seconds per week, plus however long you spend looking at it.

---

## Caveats

- **Player names must match exactly** for VLOOKUPs to work. If `Player comparison` returns "Not predicted" for someone you know plays, check the spelling in the CSV (it's `Surname FirstName`, not `FirstName Surname`).
- **Predictions are disposals, not fantasy points.** See [How to use this for SuperCoach](../docs/how-to-use-this-for-supercoach.md) for the difference.
- **Late outs are not handled.** The model does not know who is named or rested. Cross-check before lockout.

---

## Related

- [How to use this for SuperCoach](../docs/how-to-use-this-for-supercoach.md) - the honest version of what this is good for
- [Latest predictions](../docs/afl-predictions-2026.md) - this round's predicted disposal leaders
- [Backtest results](../docs/afl-backtest-2026.md) - how accurate the model has been
- [Glossary](../docs/glossary.md) - footy and data terms
