# Prediction Model

> [← Back to main README](../README.md)

<!-- This file is part of the SuperCoach-VIA documentation. See README.md for the live data sections. -->


**Who is this section for?** Readers who want to understand how the disposal predictions are produced, verify their accuracy, or improve the model. The math, the validation, and the algorithm behind the all-time ranking all live here.

Understand or improve the model. Plain-English notes on how predictions work, the backtest framework that grades them, and the algorithm behind the all-time top 100.

### How predictions work

#### What does the model actually do? (in plain English)

Imagine you had to guess how many disposals (kicks + handballs) Nick Daicos will rack up next Saturday. As a footy fan, you'd think about a few things - how he's been going lately, which team he's playing, where the game is, whether he came back from a soft-tissue niggle. The prediction model in this repo does exactly that, just with a lot more games and a lot more numbers in front of it.

It takes a player's history - their disposal counts in recent rounds, their season-to-date averages, the opponent's defensive style, the venue, and so on - and uses **machine learning** to guess what they'll do this week. "Machine learning" sounds fancy, but at its heart it just means: the computer looks at *thousands* of past games where we know what happened, finds patterns far too subtle for a human to spot by eye (e.g. "midfielders coming off a 6-day break, playing a slow-tempo opponent at the MCG, with this much recent form, tend to land within this disposal range"), and then applies those patterns to predict the next one.

What you see when you run the model is a number per player. If the prediction says **"Daicos: 34"**, that means the model expects him to get roughly 34 disposals - not exactly 34, but somewhere close. The model is graded by how far off it is on average. Right now, it's off by about 4 disposals per player, which is much better than guessing or a coin flip - but football is football. Players get tagged, get injured mid-game, or just have an off day. **No prediction is a sure thing.** Treat the numbers as a smart starting point, not a guarantee.

#### Under the hood

The model looks at each player's recent form - their disposals over the last 5 rounds, their average this season, how long since they last played - and uses that to estimate what they'll get next week. It tries two different machine learning approaches, picks the one that performed better in testing, then applies a final correction to remove any systematic over- or under-prediction.

**Accuracy from the 2026 season (rounds 1–8):**
- On average, predictions were off by **4.1 disposals**
- **68%** of predictions were within 5 disposals of the actual
- **94%** of predictions were within 10 disposals of the actual
- Round 1 is always the hardest because there's no form data from the current season yet

### Backtest framework

A backtest replays past rounds as if you were predicting them at the time - it only uses data that was available *before* each round, makes predictions, then checks them against what actually happened. This tells you honestly how good the model is.

> **Heads up:** each round takes about 30 minutes on a GPU, so an 8-round backtest runs for 4–5 hours. Best to kick it off and leave it running.

#### Run the backtest

> **Needs GPU.** See [GPU setup](technical-reference.md#setting-up-gpu-acceleration-optional) if you haven't configured CUDA yet - CPU-only runs are 10–30× slower.

```bash
# Check all 2026 rounds played so far
python backtest.py --start-year 2026 --start-round 1

# Check from late 2025 season onwards
python backtest.py --start-year 2025 --start-round 23

# Check just one specific round (quick sanity check, ~30 min)
python backtest.py --start-year 2026 --start-round 5 --end-year 2026 --end-round 5
```

#### What gets saved

Everything lands in `data/prediction/backtest/`. Files are timestamped so old results aren't overwritten.

| File | What's in it |
|------|-------------|
| `prediction_vs_actual_round_N_YEAR_*.csv` | Every player: what we predicted, what they actually got, how far off we were |
| `backtest_summary_*.csv` | A one-line summary per round - average error, how often we were within 5 or 10 disposals |
| `backtest_by_team_*.csv` | Same summary broken down by club |
| `backtest_by_position_*.csv` | Same summary broken down by position |
| `backtest_run_*.log` | Full details - biggest misses each round, which teams we consistently got wrong, overall accuracy |

#### Reading the per-player file

Open `prediction_vs_actual_round_N_YEAR_*.csv` in Excel. Key columns:

| Column | Meaning |
|--------|---------|
| `predicted_disposals` | What the model said |
| `actual_disposals` | What they actually got |
| `error` | predicted minus actual (negative = we under-predicted) |
| `abs_error` | How far off, ignoring direction |
| `over_under` | "over", "under", or "exact" (within 1 disposal) |

Sort by `abs_error` (largest first) to see the biggest misses. If the same players keep showing up as big misses week after week, it usually means they changed role or came back from injury and the model doesn't know yet.

#### Reading the log file

Open `backtest_run_*.log` in any text editor. Things to look for:

| What you see | What it means |
|--------------|---------------|
| Bias around 0 | Model is well-calibrated - good |
| Bias consistently below −1 | Model is under-predicting everyone - needs recalibration |
| Round 1 error much higher than other rounds | Normal - no current-season form data available yet |
| Error getting worse each round | Model is going stale as the season progresses |
| Same players always under-predicted | They've changed role and the model hasn't caught up |
| One team always off by 3+ disposals | Club-level data may be stale - refresh and re-run |

#### Options

| Option | Default | What it does |
|--------|---------|-------------|
| `--start-year` | 2025 | Which year to start from |
| `--start-round` | 22 | Which round to start from |
| `--end-year` | auto | Which year to stop at (auto = last year with data) |
| `--end-round` | auto | Which round to stop at (auto = last played round) |
| `--data-dir` | `./data/player_data/` | Where the player CSV files are |

### All-time top 100 ranking algorithm

The file `all_time_top_100.csv` ranks the 100 greatest VFL/AFL players of all time. The ranking is updated whenever you run `./refresh_and_rank.sh`.

<!-- POSITION-BREAKDOWN-CHART-START -->
![Top 100 position breakdown - donut and average score](assets/charts/top100_position_breakdown.png)
<!-- POSITION-BREAKDOWN-CHART-END -->

#### The problem it solves

Comparing players across eras is hard. A midfielder in 2024 has 20+ stats tracked. A midfielder in 1965 had 4. If you just add up stats, modern players always win - not because they were better, but because more was counted. The formula below tries to make comparisons fair.

#### How the ranking works (plain English)

**Step 1 - Score each season fairly**

Each season is scored using only the stats that existed at the time. A 1960s player isn't penalised for not having a "contested possession" count - that stat didn't exist yet.

| Era | Stats used for scoring |
|-----|----------------------|
| Before 1965 | Goals and behinds only |
| 1965–1990 | + kicks and handballs |
| 1991–2010 | + marks |
| 2011–now | + tackles, clearances, contested possessions, contested marks, one-percenters, goal assists |

No single stat can make up more than 55% of a player's score in any season. This stops one freakish goal-kicking year from drowning out everything else.

**Step 2 - Compare players against their peers, not everyone**

A key forward kicking 4 goals a game will always score higher in raw numbers than a midfielder. So players are compared within three groups based on their career goals-per-game:

| Group | Goals per game | Examples |
|-------|---------------|---------|
| Key forwards | 3.0 or more | Lockett, Dunstall, Ablett Sr, Lloyd, Franklin |
| Forward-midfielders | 0.8–2.99 | Carey, Matthews, Bartlett, Dangerfield, Ablett Jr |
| Midfielders/defenders | Under 0.8 | Pendlebury, Parker, Neale |

A midfielder ranked #1 in their group is genuinely considered the best midfielder of their era - not just "not as good as Lockett."

**Step 3 - Adjust for era completeness**

Even within a group, pre-1990 seasons have fewer stats tracked, so scores are slightly scaled down for modern players to close that gap. Post-2010 seasons are still scaled down a little too - GPS distance, defensive pressure acts, and other modern measures still aren't in the data.

**Step 4 - Calculate the final score**

```
Final score = average of best 8 seasons × (1 + career bonus) + peak bonus
```

- **Best 8 seasons** - rewards sustained excellence. Using only top 5 was tried but let a few players with 2–3 exceptional seasons rank too high.
- **Career bonus** - up to +30% for playing 300+ games. Capped so a long-but-average career can't beat a shorter-but-brilliant one.
- **Peak bonus** - extra credit for having a season where you were clearly the best player in the competition.
- Minimum 150 games required to be ranked.

**Step 5 - Guarantee historical coverage**

The best player from each decade (1900s, 1910s … 2020s) is guaranteed a spot. This ensures the list isn't dominated by recent players just because the data is richer.

#### Re-run the rankings

> **Needs GPU.** See [GPU setup](technical-reference.md#setting-up-gpu-acceleration-optional) if you haven't configured CUDA yet - CPU-only runs are 10–30× slower.

```bash
# Quick re-run (uses cached data, ~5–10 min)
python top_players_comprehensive.py

# Full re-run from scratch (clears cache first)
rm -f data/top100/all_time_top_100.csv
python top_players_comprehensive.py

# Full pipeline (refresh all data + re-rank)
./refresh_and_rank.sh
```
