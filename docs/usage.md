# Running predictions & backtests

> [← Back to main README](../README.md)

<!-- This file is part of the SuperCoach-VIA documentation. See README.md for the project overview. -->

**Who is this page for?** Anyone who already has the project installed and wants to predict next week's disposals, refresh the data, or run a backtest.

> **First time here?** Start with [Installation & first-time setup](installation.md).

## Predict next week's disposals

Run this command and it will automatically figure out the current year and the next round that hasn't been played yet, then generate predictions for every player:

> **Needs GPU.** See [GPU setup](technical-reference.md#setting-up-gpu-acceleration-optional) if you haven't configured CUDA yet — CPU-only runs are 10–30× slower.

```bash
python prediction.py
```

The result is saved to `data/prediction/next_round_N_prediction_<timestamp>.csv`. Open it in Excel or any spreadsheet app — it has three columns:

| Column | What it means |
|--------|---------------|
| `player` | Player name |
| `team` | Their club |
| `predicted_disposals` | How many disposals the model thinks they'll get |

**Want more detail while it runs?**
```bash
python prediction.py --debug
```

**Want to predict for a specific year?**
```bash
python prediction.py --year 2026
```

**No GPU? Use the CPU version (slower but works anywhere):**
```bash
python prediction_cpu.py
```

## What you'll actually get

Run the prediction script and you'll get a file like this — open it in Excel or Google Sheets:

| player | team | predicted_disposals |
|---|---|---|
| Nick Daicos | Collingwood | 34 |
| Clayton Oliver | Greater Western Sydney | 31 |
| Lachie Neale | Brisbane Lions | 30 |
| Zak Butters | Port Adelaide | 28 |
| Patrick Cripps | Carlton | 27 |
| … | … | … |

**How accurate is it?** Based on the 2026 season backtest (rounds 1–8):

- Average error: roughly **4–5 disposals per player per game**
- About **65–70% of predictions land within ±5 disposals** of the actual result
- About **90%+ land within ±10 disposals**

That's not perfect — football is unpredictable — but it's meaningfully better than guessing, and it gives you a consistent, data-driven starting point for your SuperCoach trades.

The `data/prediction/backtest/` folder shows the model's performance on every past round in full detail — nothing is hidden.

## Run a backtest

A backtest replays past rounds as if you were predicting them at the time, so you can see how accurate the model has been on data it's never seen.

> **Needs GPU.** See [GPU setup](technical-reference.md#setting-up-gpu-acceleration-optional) if you haven't configured CUDA yet — CPU-only runs are 10–30× slower.

```bash
python backtest.py --start-year 2026 --start-round 1
```

> **Heads up:** each round takes about 30 minutes on a GPU, so an 8-round backtest runs for 4–5 hours. Best to kick it off and leave it running.

Output lands in `data/prediction/backtest/`. For the full command reference, file layout, and how to read the log, see [Backtest framework](prediction-model.md#backtest-framework) under the prediction-model section.

## Refresh data and rankings

To pull the latest match and player data and recalculate the all-time top 100 ranking + 2026 team analysis in one command:

> **Needs GPU.** See [GPU setup](technical-reference.md#setting-up-gpu-acceleration-optional) if you haven't configured CUDA yet — CPU-only runs are 10–30× slower.

```bash
./refresh_and_rank.sh
```

## What does this script actually do?

Think of `refresh_and_rank.sh` as the **"update everything" button** for the whole project. One command, four big jobs, and when it finishes the README and every chart you see in this file have been re-built from scratch with the latest numbers.

1. **It downloads the latest match and player data from the internet** — match results, every player's stats line, fresh from AFL Tables.
2. **It recalculates the all-time top 100 player rankings** — re-scores every player from 1897 onwards using the current ranking formula and writes the updated `all_time_top_100.csv`.
3. **It rebuilds all the 2026 team analysis, finals pathway, Brownlow predictor and stat leaders** — every paragraph in the AFL Insights section is regenerated from the freshest data.
4. **It regenerates all the charts and updates this README** — the auto-marker sections in the file (team analysis, finals pathway, Brownlow predictor, stat leaders, 5-year profiles) are overwritten with the new content. You don't edit those by hand; they belong to this script.

The full run takes roughly **10 to 15 minutes on a GPU**, and longer without one. It's the right thing to run after each round of footy is played. You can also have it run automatically every week — see [Claude Code setup on Ubuntu](claude-code-setup.md) for one way to do that on a schedule.

What it actually executes, end-to-end:
1. `refresh_data.py` — scrape the latest match and player results from AFL Tables
2. `top_players_comprehensive.py` — recompute and write `all_time_top_100.csv`
3. `update_team_analysis.py` — regenerate the 2026 team analysis section + 5-year team-style profiles + the embedded charts in this README


---

## Related

- [Installation & first-time setup](installation.md)
- [Troubleshooting](troubleshooting.md)
- [How predictions work](prediction-model.md)
- [Technical reference (GPU setup, etc.)](technical-reference.md)
