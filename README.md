# AFL SuperCoach VIA
![AFL Data Analysis Banner Image](/assets/readme_banner.png)
<div align="center">
  <img src="https://img.shields.io/github/last-commit/akareen/AFL-Data-Analysis">
  <img src="https://img.shields.io/github/contributors/akareen/AFL-Data-Analysis">
  <img src="https://img.shields.io/github/stars/akareen/AFL-Data-Analysis?style=social">
  <img src="https://img.shields.io/github/forks/akareen/AFL-Data-Analysis?style=social">
</div>
<br>

A personal AFL data project that does three things:
1. **Stores every AFL match and player stat** going back to 1897
2. **Ranks the greatest players of all time** using a fair, era-adjusted formula
3. **Predicts how many disposals each player will get** in the next round

No AFL coding knowledge required — if you can open a terminal and run a command, you can use this.

## Table of Contents
- [What's in this repo](#whats-in-this-repo)
- [Getting started](#getting-started)
- [Predict next week's disposals](#predict-next-weeks-disposals)
- [Backtest — check how accurate the predictions were](#backtest--check-how-accurate-the-predictions-were)
- [All-time top 100 ranking](#all-time-top-100-ranking)
- [Setting up GPU acceleration (optional)](#setting-up-gpu-acceleration-optional)
- [How the data is organised](#how-the-data-is-organised)
- [Using Claude and the Scientist agent](#using-claude-and-the-scientist-agent)
- [Data sources](#data-sources)
- [Contributing](#contributing)
- [License](#license)


## What's in this repo

| What | Where | Details |
|------|-------|---------|
| Every AFL match result (1897–now) | `data/matches/` | Scores, margins, venues, quarter-by-quarter breakdown |
| Stats for every player ever | `data/player_data/` | Kicks, marks, goals, disposals, tackles and more — 5,700+ players, 682,000 game records |
| All-time top 100 ranking | `all_time_top_100.csv` | Era-adjusted ranking updated whenever you run the pipeline |
| Next round disposal predictions | `data/prediction/` | A CSV telling you how many disposals each player is predicted to get |
| Backtest results | `data/prediction/backtest/` | How accurate past predictions were, round by round |


## Getting started

### 1. Download the repo

```bash
git clone https://github.com/apur27/SuperCoach-VIA.git
cd SuperCoach-VIA
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Refresh to latest data (optional)

This pulls the latest match and player data from the web, then recalculates the top 100:

```bash
./refresh_and_rank.sh
```

That's it. You're ready to run predictions.


## Predict next week's disposals

Run this command and it will automatically figure out the current year and the next round that hasn't been played yet, then generate predictions for every player:

```bash
/home/abhi/sourceCode/python/coding/.venv/bin/python prediction.py
```

The result is saved to `data/prediction/next_round_N_prediction_<timestamp>.csv`. Open it in Excel or any spreadsheet app — it has three columns:

| Column | What it means |
|--------|---------------|
| `player` | Player name |
| `team` | Their club |
| `predicted_disposals` | How many disposals the model thinks they'll get |

**Want more detail while it runs?**
```bash
/home/abhi/sourceCode/python/coding/.venv/bin/python prediction.py --debug
```

**Want to predict for a specific year?**
```bash
/home/abhi/sourceCode/python/coding/.venv/bin/python prediction.py --year 2026
```

**No GPU? Use the CPU version (slower but works anywhere):**
```bash
/home/abhi/sourceCode/python/coding/.venv/bin/python prediction_cpu.py
```

### How the predictions work (plain English)

The model looks at each player's recent form — their disposals over the last 5 rounds, their average this season, how long since they last played — and uses that to estimate what they'll get next week. It tries two different machine learning approaches, picks the one that performed better in testing, then applies a final correction to remove any systematic over- or under-prediction.

**Accuracy from the 2026 season (rounds 1–8):**
- On average, predictions were off by **4.1 disposals**
- **68%** of predictions were within 5 disposals of the actual
- **94%** of predictions were within 10 disposals of the actual
- Round 1 is always the hardest because there's no form data from the current season yet


## Backtest — check how accurate the predictions were

A backtest replays past rounds as if you were predicting them at the time — it only uses data that was available *before* each round, makes predictions, then checks them against what actually happened. This tells you honestly how good the model is.

> **Heads up:** each round takes about 30 minutes on a GPU, so an 8-round backtest runs for 4–5 hours. Best to kick it off and leave it running.

### Run the backtest

```bash
# Check all 2026 rounds played so far
/home/abhi/sourceCode/python/coding/.venv/bin/python backtest.py --start-year 2026 --start-round 1

# Check from late 2025 season onwards
/home/abhi/sourceCode/python/coding/.venv/bin/python backtest.py --start-year 2025 --start-round 23

# Check just one specific round (quick sanity check, ~30 min)
/home/abhi/sourceCode/python/coding/.venv/bin/python backtest.py --start-year 2026 --start-round 5 --end-year 2026 --end-round 5
```

### What gets saved

Everything lands in `data/prediction/backtest/`. Files are timestamped so old results aren't overwritten.

| File | What's in it |
|------|-------------|
| `prediction_vs_actual_round_N_YEAR_*.csv` | Every player: what we predicted, what they actually got, how far off we were |
| `backtest_summary_*.csv` | A one-line summary per round — average error, how often we were within 5 or 10 disposals |
| `backtest_by_team_*.csv` | Same summary broken down by club |
| `backtest_by_position_*.csv` | Same summary broken down by position |
| `backtest_run_*.log` | Full details — biggest misses each round, which teams we consistently got wrong, overall accuracy |

### Reading the per-player file

Open `prediction_vs_actual_round_N_YEAR_*.csv` in Excel. Key columns:

| Column | Meaning |
|--------|---------|
| `predicted_disposals` | What the model said |
| `actual_disposals` | What they actually got |
| `error` | predicted minus actual (negative = we under-predicted) |
| `abs_error` | How far off, ignoring direction |
| `over_under` | "over", "under", or "exact" (within 1 disposal) |

Sort by `abs_error` (largest first) to see the biggest misses. If the same players keep showing up as big misses week after week, it usually means they changed role or came back from injury and the model doesn't know yet.

### Reading the log file

Open `backtest_run_*.log` in any text editor. Things to look for:

| What you see | What it means |
|--------------|---------------|
| Bias around 0 | Model is well-calibrated — good |
| Bias consistently below −1 | Model is under-predicting everyone — needs recalibration |
| Round 1 error much higher than other rounds | Normal — no current-season form data available yet |
| Error getting worse each round | Model is going stale as the season progresses |
| Same players always under-predicted | They've changed role and the model hasn't caught up |
| One team always off by 3+ disposals | Club-level data may be stale — refresh and re-run |

### Options

| Option | Default | What it does |
|--------|---------|-------------|
| `--start-year` | 2025 | Which year to start from |
| `--start-round` | 22 | Which round to start from |
| `--end-year` | auto | Which year to stop at (auto = last year with data) |
| `--end-round` | auto | Which round to stop at (auto = last played round) |
| `--data-dir` | `./data/player_data/` | Where the player CSV files are |


## All-time top 100 ranking

The file `all_time_top_100.csv` ranks the 100 greatest VFL/AFL players of all time. The ranking is updated whenever you run `./refresh_and_rank.sh`.

### The problem it solves

Comparing players across eras is hard. A midfielder in 2024 has 20+ stats tracked. A midfielder in 1965 had 4. If you just add up stats, modern players always win — not because they were better, but because more was counted. The formula below tries to make comparisons fair.

### How the ranking works (plain English)

**Step 1 — Score each season fairly**

Each season is scored using only the stats that existed at the time. A 1960s player isn't penalised for not having a "contested possession" count — that stat didn't exist yet.

| Era | Stats used for scoring |
|-----|----------------------|
| Before 1965 | Goals and behinds only |
| 1965–1990 | + kicks and handballs |
| 1991–2010 | + marks |
| 2011–now | + tackles, clearances, contested possessions, contested marks, one-percenters, goal assists |

No single stat can make up more than 55% of a player's score in any season. This stops one freakish goal-kicking year from drowning out everything else.

**Step 2 — Compare players against their peers, not everyone**

A key forward kicking 4 goals a game will always score higher in raw numbers than a midfielder. So players are compared within three groups based on their career goals-per-game:

| Group | Goals per game | Examples |
|-------|---------------|---------|
| Key forwards | 3.0 or more | Lockett, Dunstall, Ablett Sr, Lloyd, Franklin |
| Forward-midfielders | 0.8–2.99 | Carey, Matthews, Bartlett, Dangerfield, Ablett Jr |
| Midfielders/defenders | Under 0.8 | Pendlebury, Parker, Neale |

A midfielder ranked #1 in their group is genuinely considered the best midfielder of their era — not just "not as good as Lockett."

**Step 3 — Adjust for era completeness**

Even within a group, pre-1990 seasons have fewer stats tracked, so scores are slightly scaled down for modern players to close that gap. Post-2010 seasons are still scaled down a little too — GPS distance, defensive pressure acts, and other modern measures still aren't in the data.

**Step 4 — Calculate the final score**

```
Final score = average of best 8 seasons × (1 + career bonus) + peak bonus
```

- **Best 8 seasons** — rewards sustained excellence. Using only top 5 was tried but let a few players with 2–3 exceptional seasons rank too high.
- **Career bonus** — up to +30% for playing 300+ games. Capped so a long-but-average career can't beat a shorter-but-brilliant one.
- **Peak bonus** — extra credit for having a season where you were clearly the best player in the competition.
- Minimum 150 games required to be ranked.

**Step 5 — Guarantee historical coverage**

The best player from each decade (1900s, 1910s … 2020s) is guaranteed a spot. This ensures the list isn't dominated by recent players just because the data is richer.

### Re-run the rankings

```bash
# Quick re-run (uses cached data, ~5–10 min)
/home/abhi/sourceCode/python/coding/.venv/bin/python top_players_comprehensive.py

# Full re-run from scratch (clears cache first)
rm -f data/top100/all_time_top_100.csv
/home/abhi/sourceCode/python/coding/.venv/bin/python top_players_comprehensive.py

# Full pipeline (refresh all data + re-rank)
./refresh_and_rank.sh
```


## Setting up GPU acceleration (optional)

The prediction scripts run faster with an NVIDIA GPU. If you don't have one, everything still works — just use `prediction_cpu.py` instead of `prediction.py`, and the ranking script falls back to CPU automatically.

### Do you need GPU setup?

- **No GPU / not sure** — use `prediction_cpu.py`, skip this section
- **NVIDIA GPU on Linux** — follow the steps below
- **NVIDIA GPU on Windows** — GPU DataFrame support doesn't work on native Windows; either use WSL2 (Windows Subsystem for Linux) or use `prediction_cpu.py`

### Step 1 — Check your GPU works

```bash
nvidia-smi
```

You should see your GPU name and a CUDA version number. If this command fails, install the latest NVIDIA driver from [nvidia.com/drivers](https://www.nvidia.com/drivers) first.

### Step 2 — Install the CUDA toolkit

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

Then add CUDA to your terminal path (paste into `~/.bashrc`):

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Reload: `source ~/.bashrc`, then verify: `nvcc --version`

### Step 3 — Install the GPU libraries

```bash
# GPU DataFrame library (speeds up ranking)
pip install cudf-cu12 cupy-cuda12x --extra-index-url=https://pypi.nvidia.com

# GPU-enabled LightGBM (speeds up predictions)
pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON
```

### Step 4 — Verify everything works

```bash
python cuDF_test.py   # should print "cuDF working"
python testGPU.py     # should print "LightGBM GPU working"
python prediction.py  # should say "Running on GPU" at the top
```

### Laptop tips

If you're on a gaming laptop and predictions feel slow:
- Set your laptop to Performance power mode
- Check that it's using the NVIDIA GPU, not the integrated Intel/AMD one (look in BIOS or Armoury Crate)
- If you see `CUDA out of memory`, switch to `prediction_cpu.py`


## How the data is organised

```
data/
  matches/          — one CSV per year: every match result 1897–now
  lineups/          — team lineups by season
  player_data/      — one CSV per player: their full career stats
  top100/           — all-time and yearly rankings
  prediction/       — disposal predictions (next round)
    backtest/       — historical accuracy results
```

### Match data columns

Each row in `data/matches/matches_YEAR.csv` is one game:

`year, round, venue, date, home_team, away_team, home_q1_g, home_q1_b, ...`  
(goals and behinds for each quarter, final totals, winning team, margin)

### Player data columns

Each player has two files in `data/player_data/`:

**Performance file** (`*_performance_details.csv`) — one row per game played:  
`team, year, round, opponent, kicks, marks, handballs, disposals, goals, behinds, tackles, clearances, brownlow_votes, ...`

**Personal file** (`*_personal_details.csv`):  
`first_name, last_name, born_date, debut_date, height, weight`


## Using Claude and the Scientist agent

This project is designed to be improved over time using [Claude Code](https://claude.ai/code) — an AI coding assistant you run in your terminal. You don't need to understand the code to use it. You just describe what you want in plain English.

### Opening the project in Claude

From the project folder, start Claude Code:

```bash
cd /home/abhi/git/SuperCoach-VIA
claude
```

Then just type what you want. Some examples:

```
run the backtest for 2026
what was the accuracy last round?
why is the model under-predicting midfielders?
push my changes to main
```

---

### The Scientist agent

The Scientist is a specialised Claude agent that reads code and data, finds problems, and fixes them. It's the main tool for improving prediction accuracy.

Summon it by typing `@"Scientist (agent)"` at the start of your message, then describe what you want it to do.

#### After a backtest run — improve the model

After running `backtest.py`, hand the results to the Scientist and it will read the logs, find the systematic errors, and fix `prediction.py`:

```
@"Scientist (agent)" analyse the backtest and improve the prediction model
```

It will:
- Read all the backtest CSVs and the log file
- Find patterns — which teams are consistently wrong, which players are always missed, whether the model over- or under-predicts
- Make targeted changes to `prediction.py` to fix what it found
- Tell you what it changed and why

#### Check on a run

If you started a backtest or prediction and aren't sure if it finished:

```
@"Scientist (agent)" check the status of the last backtest run
```

#### Investigate a specific problem

```
@"Scientist (agent)" the model keeps under-predicting high-disposal midfielders — find out why and fix it
@"Scientist (agent)" look at prediction.py and find anything that could cause wrong results
@"Scientist (agent)" the backtest for round 1 always has bad accuracy — can you explain why and improve it?
```

#### Optimise slow code

```
@"Scientist (agent)" prediction.py is running slowly — find optimisations
```

---

### Typical improvement loop

This is the recommended workflow for improving prediction accuracy over a season:

```
1. Refresh data       →  ./refresh_and_rank.sh
2. Run backtest       →  python backtest.py --start-year 2026 --start-round 1
3. Ask Scientist      →  @"Scientist (agent)" analyse the backtest and improve the model
4. Re-run backtest    →  verify MAE dropped
5. Push to main       →  push the changes to main
6. Predict next round →  python prediction.py
```

You can repeat steps 3–5 as many times as you like. Each iteration typically improves MAE by 0.2–0.5 disposals.

---

### Other useful Claude commands

These work without the Scientist agent — just type them directly:

| What you type | What happens |
|---------------|-------------|
| `push the changes to main` | Commits everything and pushes to GitHub |
| `update the readme` | Updates this file based on recent changes |
| `what does prediction.py do?` | Plain-English explanation of the code |
| `run the prediction for next round` | Gives you the exact command to run |
| `the backtest crashed with [error] — fix it` | Diagnoses and fixes the error |
| `show me the top 10 most over-predicted players` | Reads the backtest CSV and answers |

---

### Tips for getting good results from Claude

- **Be specific about what went wrong.** Instead of "fix the prediction", say "the model predicted Daicos at 18 disposals but he got 34 — why?"
- **Paste error messages directly.** If a script crashes, copy the full error and paste it into Claude. It will fix it.
- **Tell it what you care about.** "I care more about getting the top 10 players right than overall accuracy" helps it make better trade-offs.
- **Ask it to explain first.** Before making big changes, ask "what would you change and why?" — you can then say yes or redirect it.
- **Push after each session.** Type "push the changes to main" at the end of any session where improvements were made.

## Data sources

- Match results and player stats: [AFL Tables](https://afltables.com/afl/afl_index.html)
- Historical betting odds: [AusSportsBetting](https://www.aussportsbetting.com/data/historical-afl-results-and-odds-data/)


## Contributing

Got an idea, found a bug, or want to add a new feature? Open an issue or send a pull request — all contributions welcome.


## License

MIT License — see the [LICENSE](LICENSE) file.
