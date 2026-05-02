# AFL SuperCoach VIA
![AFL Data Analysis Banner Image](/assets/readme_banner.png)
<div align="center">
  <img src="https://img.shields.io/github/last-commit/akareen/AFL-Data-Analysis">
  <img src="https://img.shields.io/github/contributors/akareen/AFL-Data-Analysis">
  <img src="https://img.shields.io/github/stars/akareen/AFL-Data-Analysis?style=social">
  <img src="https://img.shields.io/github/forks/akareen/AFL-Data-Analysis?style=social">
</div>
<br>

If you're curious about AI and large language models but don't know where to start, this repo is a practical, hands-on way to learn — and you don't need a computer science degree to follow along. All you need is a gaming laptop running Ubuntu, an interest in AFL, a copy of [Claude Code](https://claude.ai/code), and a Claude subscription — the entry level plan is plenty for everything in this repo. Everything in this project — the prediction model, the backtest framework, the all-time player rankings — was built and improved by having plain-English conversations with AI agents, including a specialised "Scientist" agent that reads data, finds problems, and fixes code on its own. You'll see how to use agents to analyse real AFL data, predict player disposals each week, and continuously improve accuracy by feeding results back into the model. Whether you want to understand how LLMs can write and improve code, how machine learning predictions actually work in practice, or just who the greatest AFL player of all time is — this repo shows you all of it, one conversation at a time.

---

A personal AFL data project that does three things:
1. **Stores every AFL match and player stat** going back to 1897
2. **Ranks the greatest players of all time** using a fair, era-adjusted formula
3. **Predicts how many disposals each player will get** in the next round

## Table of Contents
- [What's in this repo](#whats-in-this-repo)
- [Getting started](#getting-started)
- [Predict next week's disposals](#predict-next-weeks-disposals)
- [Backtest — check how accurate the predictions were](#backtest--check-how-accurate-the-predictions-were)
- [All-time top 100 ranking](#all-time-top-100-ranking)
- [For the footy expert — finding the greatest 100 players of all time](#for-the-footy-expert--finding-the-greatest-100-players-of-all-time)
- [Setting up GPU acceleration (optional)](#setting-up-gpu-acceleration-optional)
- [How the data is organised](#how-the-data-is-organised)
- [Setting up Claude Code on Ubuntu](#setting-up-claude-code-on-ubuntu)
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


## For the footy expert — finding the greatest 100 players of all time

If you live and breathe footy — you can rattle off Lockett's goals tally, you've argued about Carey vs. Matthews more times than you can count, you have strong opinions about whether Bontempelli is already a top-20 player of all time — but you've never opened a terminal in your life, this section is for you.

This repo gives you something you can't get anywhere else: a ranking formula you can actually **change, challenge, and re-run yourself** — without writing a single line of code. You just talk to an AI agent (Claude) in plain English, the same way you'd argue at the pub. You ask "why is Carey ranked above Matthews?" and it tells you. You say "I reckon goals should count for more — show me what happens" and it changes the formula and re-runs it.

You don't need to know what Python is. You don't need to understand machine learning. You just need to be opinionated about footy.

---

### What the ranking is actually doing (in footy terms)

The hardest problem in ranking all-time greats is that **you can't just add up stats**. Tony Lockett kicked 1,360 goals — but in his era, contested possessions weren't even tracked. Haydn Bunton Sr. won three Brownlows in the 1930s when the only stats kept were goals and behinds. Compare that to Patrick Cripps, who has 20+ different stats logged every single game.

If you just totted up everything, modern players would crush every list — not because they were better, but because more was being counted.

So the formula does three things to make it fair:

**1. It only uses the stats that existed at the time.** Bunton's score uses goals and behinds. Bartlett's uses kicks, handballs, marks, and goals. Bontempelli's uses everything modern, including contested possessions and clearances. Nobody is penalised for not having a stat that didn't exist yet.

**2. It groups players by position type so you're comparing apples to apples.** A key forward like Lockett or Dunstall will always pile up more raw "score" than a midfielder like Pendlebury — because goals are weighted heavily and forwards kick more of them. So the ranking splits players into three buckets:

- **Key forwards** (3+ goals per game) — Lockett, Dunstall, Ablett Sr, Lloyd, Franklin, Brown
- **Forward-midfielders** (0.8 to 2.99) — Carey, Matthews, Bartlett, Dangerfield, Ablett Jr, Dustin Martin
- **Midfielders / defenders** (under 0.8) — Pendlebury, Neale, Cripps, Parker

A midfielder who finishes #1 in their group is genuinely the best midfielder of the era — not just "good but not as good as Lockett."

**3. It rewards both sustained excellence and peak greatness.** The final score is the **average of a player's best 8 seasons**, plus a bonus for very long careers (capped, so a 15-year average career can't beat a 10-year brilliant one), plus a bonus for having a season where you were clearly the best player in the league. The minimum to be ranked at all is 150 games — that filters out brief careers, no matter how brilliant.

Unsure why the best 8? Because using only the best 5 let players with 2 freak years rank too high. Using the whole career punished anyone who hung on too long. Best 8 is the sweet spot — long enough to reward consistency, short enough to ignore decline years.

---

### How to challenge the ranking by talking to Claude

Once you've followed the [Setting up Claude Code on Ubuntu](#setting-up-claude-code-on-ubuntu) steps, you'll have a thing called Claude running in your terminal. **Forget what "terminal" means** — just think of it as a window where you type questions to a very smart footy analyst who has read every line of code in this project and every player's stats.

To start it up:

```bash
cd /path/to/SuperCoach-VIA
claude
```

You'll see a `>` prompt. That's where you type. Hit Enter to send. Claude reads everything in this project and answers in plain English. If it needs to change something or run something, it'll tell you what it's about to do first.

Here's what conversations actually look like.

#### Asking why a player ranked where they did

```
Why is Wayne Carey ranked above Leigh Matthews on the all-time list?
```

Claude will open `all_time_top_100.csv`, look up both players' scores, see which group (forward-midfielders) they're both in, and explain the breakdown — Carey's per-season scores, Matthews' per-season scores, the era adjustment applied to each, the peak bonus, the longevity bonus. You'll get a paragraph back like "Carey's average top-8 season scored higher because the formula weights kicks and marks heavily and Matthews played in an era where marks weren't tracked until 1991 — he gets credit for goals and disposals but the formula can't see his contested marks. Try this: tell me to lower the weight on marks and we'll see if Matthews moves up."

#### Asking where a specific player ranks

```
Where does Scott Pendlebury rank and why?
Where does Gary Ablett Jr finish and what's his peak season according to the formula?
Why didn't Lance Franklin make the top 10?
```

For any of these, Claude will pull the actual numbers and explain. If a player isn't on the list, Claude will tell you exactly where they fell short — not enough games, no peak Brownlow-level season, or their group ranking just wasn't high enough.

#### Challenging the formula and re-running it

This is where it gets fun. You can change anything about the formula and see what happens.

```
I think goals should count for more in the modern era — increase the goal weight by 30% and re-run the ranking. Show me how the top 20 changes.
```

```
The formula uses the best 8 seasons. I think it should be best 10 — too many short-career players are sneaking in. Change it and re-run.
```

```
Drop the minimum games requirement from 150 to 100 and tell me which players newly make the list.
```

```
The career longevity bonus is capped at 30%. I think Pendlebury and Bartlett are getting underrated for their 350+ games — raise the cap to 40% and show me what shifts.
```

Claude will make the change in the code, re-run the ranking, save the new top 100, and tell you which players moved up, which moved down, and which dropped off entirely. If you don't like it, just say "revert that change" and you're back to the original.

#### Asking Claude to explain the era adjustments

```
Explain the era adjustment in plain English — why does a 1985 season score differently to a 2015 season for the same raw stats?
```

```
Walk me through how Haydn Bunton's 1933 season is scored. He won the Brownlow but I don't see a "Brownlow" stat in the formula — how does the model recognise his greatness?
```

```
The formula caps any single stat at 55% of a season's score. What does that actually do? Show me a player whose ranking changed because of that cap.
```

---

### Using the Scientist agent for deeper questions

Claude (the regular one) is great for opinion arguments and quick formula tweaks. But for **proper number-crunching** — running statistical sensitivity analyses, checking whether changes are real or just noise, comparing rankings across different formula versions — there's a heavier-duty version called the **Scientist agent**.

> **READ THE DISCLAIMER** in the [STOP. READ THIS FIRST.](#stop-read-this-first-do-not-waste-the-scientist) section above before invoking the Scientist. It runs on Claude's most expensive model and burns through tokens fast. **Use it for the questions that actually require it. For everything else, just use plain Claude.**

You invoke the Scientist by typing `@"Scientist (agent)"` at the start of your message:

```
@"Scientist (agent)" double the weight on contested possessions in the ranking formula and tell me how the top 10 changes. I want a proper sensitivity analysis — not just the new list, but how confident we should be that the changes are meaningful versus formula noise.
```

```
@"Scientist (agent)" the ranking puts Wayne Carey at #1. Run a sensitivity analysis — try 5 different reasonable variations of the formula (different season-count, different group thresholds, different era adjustments) and tell me whether Carey stays #1 in all of them or whether his top spot is fragile.
```

```
@"Scientist (agent)" how much does the top 20 actually change if I remove the era adjustment entirely? Is the era adjustment doing real work or is it cosmetic?
```

```
@"Scientist (agent)" the formula gives a peak-season bonus. Strip it out and re-rank — which players were riding on one freakish season versus genuine sustained excellence?
```

```
@"Scientist (agent)" simulate what the top 100 would look like if pre-1965 players had access to the modern stats — i.e. fill in the missing stats with reasonable estimates based on similar modern players, and show me how Bunton, Coleman and Whitten move.
```

The Scientist will read the code, do the proper analysis, and report back with not just the answer but **how confident you should be in it** — which is what separates a real analysis from just another opinion.

---

### Questions worth asking — for stress-testing the formula

If you're a footy expert who wants to genuinely challenge whether this ranking holds up, here are the questions that will tell you the most. You can paste any of these straight into Claude.

**Era fairness**

- "Show me the top 5 players from each decade in this list. Are pre-1970 players underrepresented? If so, is that a real reflection of quality or a flaw in the formula?"
- "The best 8 seasons average rewards longevity. Most pre-WWII players had shorter careers due to the war and shorter seasons. Has the formula adjusted for that, or are players like Dyer and Bunton being penalised?"
- "Modern players have GPS distance, defensive pressure acts, and other stats that aren't in this dataset. The formula scales them down to compensate — but is the scale-down enough? Try doubling it and show me what happens."

**Position group fairness**

- "There are 3 position groups. What if I split key forwards into 'true power forwards' (Lockett, Dunstall, Coleman) and 'lead-up forwards' (Franklin, Ablett Sr)? Does the ranking change in a way that feels right?"
- "Pendlebury, Neale and Cripps are in the same group as 1960s-era pure defenders. Does that make sense? Test what happens if we split midfielders out from defenders."
- "Carey is classified as a forward-midfielder. Move him into the key forwards group and re-rank. Does that change his position?"

**Stat weighting**

- "Goals dominate forwards' scores. What's the actual weight on goals in the formula, and what happens if I halve it? Do midfielders take over the top of the list?"
- "Show me what happens to the top 20 if I weight contested possessions twice as heavily as kicks. The argument is that contested ball is harder, so should count more."
- "Brownlow votes aren't used as a stat. Should they be? Add them as a 10% weighting and re-rank."

**Peak vs longevity**

- "Tony Lockett's career peak was higher than Buddy Franklin's, but Franklin played longer. The formula uses best 8 seasons + a longevity bonus. Show me what each player gets from each component."
- "Cut the longevity bonus to zero. Who falls off the top 50? Are those players great because of who they were, or because they hung on?"
- "Use only best 3 seasons (peak only — no longevity at all). Who rises to the top?"

**Specific player debates**

- "Dustin Martin won 3 Norm Smiths and a Brownlow. Where does the formula put him, and is he being properly credited for finals heroics? (The dataset is regular-season only — flag that as a limitation.)"
- "Gary Ablett Sr vs Gary Ablett Jr — which one ranks higher and what's the gap? Walk me through the breakdown."
- "Lachie Neale has two Brownlows but doesn't crack the top 30. Is that a fair reflection or is the formula missing something about his game?"
- "Bontempelli is still active. How does the formula treat current players whose best seasons might still be ahead of them?"

**Structural questions**

- "The minimum is 150 games. How many genuinely great careers does that exclude — players who were brilliant but injured early, or war-shortened careers like Dick Reynolds-era players?"
- "The list guarantees one player per decade. Is the 1900s rep just there because they had to put someone — or are they genuinely top-100 quality on the merits?"
- "What's the standard deviation of scores in the top 100? If the gap between #1 and #50 is small, the rankings within those 50 are basically noise. If it's big, the order is meaningful."

---

### One last thing

The whole point of this is that **you can argue with the formula and win**. If you genuinely believe Leigh Matthews is the greatest of all time and the formula has him at #4 — say so to Claude, ask why, and challenge the assumptions. If your argument holds up, change the formula. If your changes produce a list you can defend at the pub, push them to the repo (just type `push the changes to main`).

This isn't a list handed down from on high. It's a starting point for the argument — and now you've got the tools to make your case with actual numbers behind it.


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


## Setting up Claude Code on Ubuntu

This section walks you through getting Claude Code running on a fresh Ubuntu laptop — from zero to having an AI agent that can read your AFL data, write code, and improve the prediction model.

### What you need before you start

- An Ubuntu laptop (20.04 or newer) — a gaming laptop works great, specs don't need to be high for Claude itself
- A Claude subscription — go to [claude.ai](https://claude.ai) and sign up; the **entry level (Pro) plan is enough** for everything in this repo
- Internet connection

---

### Step 1 — Install Node.js

Claude Code runs on Node.js. Install it via the official NodeSource repository (the version in Ubuntu's default package manager is often too old):

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

Verify:
```bash
node --version   # should show v20 or higher
npm --version
```

---

### Step 2 — Install Claude Code

Claude Code is installed as a global npm package:

```bash
npm install -g @anthropic-ai/claude-code
```

Verify:
```bash
claude --version
```

---

### Step 3 — Log in to your Claude account

```bash
claude login
```

This opens a browser window. Log in with the same account you used to sign up at claude.ai. Once authenticated, your terminal will confirm you're logged in.

---

### Step 4 — Install Python and project dependencies

This repo uses Python 3.12. Set up a virtual environment:

```bash
sudo apt install -y python3.12 python3.12-venv python3-pip

# Create a virtual environment
python3.12 -m venv ~/.venv/afl
source ~/.venv/afl/bin/activate

# Install project dependencies
cd /path/to/SuperCoach-VIA
pip install -r requirements.txt
```

To activate the environment every time you open a terminal, add this to your `~/.bashrc`:
```bash
echo 'source ~/.venv/afl/bin/activate' >> ~/.bashrc
source ~/.bashrc
```

---

### Step 5 — Open the project in Claude Code

```bash
cd /path/to/SuperCoach-VIA
claude
```

You'll see the Claude Code prompt. You're now inside an AI-powered terminal that understands your entire codebase. Try:

```
what does this project do?
```

Claude will read the code and explain it to you in plain English.

---

### Step 6 — Verify everything works end to end

```bash
# In the Claude Code prompt, type:
run the prediction for next round
```

Claude will give you the exact command. Run it. If it works, you're set up correctly.

---

### Troubleshooting common issues

| Problem | Fix |
|---------|-----|
| `claude: command not found` | Run `npm install -g @anthropic-ai/claude-code` again; make sure `/usr/bin` is in your PATH |
| `claude login` doesn't open a browser | Copy the URL it prints and paste it into your browser manually |
| Python packages fail to install | Make sure your venv is activated: `source ~/.venv/afl/bin/activate` |
| `ModuleNotFoundError` when running scripts | You're using the wrong Python — use the full venv path: `/path/to/.venv/afl/bin/python prediction.py` |
| Claude says it can't find files | Make sure you're in the repo directory when you start Claude: `cd SuperCoach-VIA && claude` |

---

## Using Claude and the Scientist agent

Once Claude Code is set up (see above), you interact with it entirely in plain English — no commands to memorise, no code to write. This section covers how to get the most out of it, and specifically how to use the Scientist agent to improve the prediction model.

---

> ## **STOP. READ THIS FIRST. DO NOT WASTE THE SCIENTIST.**
>
> ### **The Scientist is expensive. Use it wisely.**
>
> The Scientist agent runs on **Claude Opus** — the most powerful and **most expensive** Claude model that Anthropic ships. Every time you invoke the Scientist, you are spending **a lot more tokens** than a normal Claude conversation would use.
>
> If you are on an **entry-level Claude subscription** (Pro, or any plan with a monthly token budget), you have a **finite number of tokens per month**. Burning them on trivial questions to the Scientist is the fastest way to **run out before the end of the month** and find yourself unable to use Claude at all.
>
> **The rule is simple:**
>
> > **Use the Scientist ONLY when you need it to read your data and your code, and do something analytical with it.**
> >
> > **For literally everything else, just talk to plain Claude — or use Google.**
>
> Plain Claude (no `@"Scientist (agent)"` prefix) is already extremely capable. It can answer questions, explain code, write small scripts, and chat with you — all on a much cheaper model. Save the Scientist for the heavy lifting it was built for.

### **Good questions for Scientist vs. Don't waste Scientist on this**

| Good questions for Scientist (worth the cost) | Don't waste Scientist on this (use plain Claude or Google) |
|---|---|
| Analyse the backtest and improve the model | What is the weather today? |
| Why is the model consistently under-predicting midfielders? | Who won the 2024 AFL grand final? |
| Find any data leakage in the prediction pipeline | What does `print()` do in Python? |
| The backtest shows a bias of -1.3 — find the root cause and fix it | How do I open a CSV file in Excel? |
| Optimise `prediction.py` without changing the results | What is machine learning? |
| Which teams is the model most wrong about, and why? | Can you write me a haiku about footy? |
| Round 1 accuracy is always worse than other rounds — investigate and fix it | What time is it? |
| Check `prediction.py` for bugs that could cause wrong predictions | How many players are in an AFL team? |
| Run a statistical analysis on the backtest errors and identify the biggest contributors | Explain what a `for` loop is |
| Look for feature engineering improvements based on the residuals | What's the capital of Australia? |
| Compare the last three backtests and tell me whether MAE is genuinely improving or just noise | Can you summarise the README for me? |
| Investigate why predictions for a specific player are systematically off | Say hello |

### **A simple rule of thumb**

Ask yourself one question before typing `@"Scientist (agent)"`:

> **"Does answering this require Claude to actually open my CSV files, read my Python code, and do real analytical work on it?"**
>
> - **Yes** → Use the Scientist. This is what it's for.
> - **No** → Use plain Claude. Or Google. Or a calculator. Anything cheaper.

If you just want a chat, a definition, a quick code explanation, or general help — **don't @ the Scientist**. Just type your question normally. Plain Claude will handle it for a fraction of the cost.

Treat the Scientist like calling in a senior consultant. You wouldn't pay a consultant $500/hr to tell you what time it is. Same idea here.

---

### How Claude Code works

Start it from the project folder:

```bash
cd /path/to/SuperCoach-VIA
claude
```

You'll see a `>` prompt. Type what you want in plain English. Claude reads every file in the project and can run commands, edit code, and explain what it's doing — all in response to plain-English instructions.

```
what does this project do?
run the prediction for next round
the backtest crashed — fix it
push my changes to main
```

Press `Enter` to send. Claude will respond, and if it needs to run a command or edit a file, it will ask for your permission first (or just do it, depending on your settings).

---

### The Scientist agent — your data science expert

The Scientist is a specialised sub-agent built into this project. It has deep expertise in data science methodology — feature engineering, model evaluation, bias detection, statistical testing — and is the main tool for improving prediction accuracy.

**How to invoke it:** type `@"Scientist (agent)"` at the start of your message, then describe the task.

```
@"Scientist (agent)" [your task here]
```

That's it. The Scientist will spin up, read the relevant files, do the analysis, and report back. You don't need to tell it where the files are or how the code works — it figures that out itself.

---

### What to ask the Scientist

#### After every backtest — improve the model

This is the most important use. After running `backtest.py`, hand the results to the Scientist:

```
@"Scientist (agent)" analyse the backtest and improve the prediction model
```

The Scientist will:
1. Read all the backtest CSVs and the log file
2. Calculate where the model is systematically wrong — which teams, which players, which disposal ranges
3. Identify the root cause (e.g. "the model is compressing high-disposal predictions due to the log transform")
4. Make targeted changes to `prediction.py` to fix what it found
5. Tell you exactly what it changed and what improvement to expect

You then re-run the backtest to verify the improvement.

#### Check if a run finished

If you started a backtest or prediction run and your laptop restarted or you closed the terminal:

```
@"Scientist (agent)" check the status of the last backtest run
```

It will look at the output files and logs and tell you exactly what completed, what's missing, and whether you need to re-run anything.

#### Investigate a specific problem

```
@"Scientist (agent)" the model keeps under-predicting Daicos — find out why and fix it
@"Scientist (agent)" round 1 accuracy is always much worse than other rounds — why?
@"Scientist (agent)" look at prediction.py and find any bugs that could cause wrong results
@"Scientist (agent)" which teams is the model most consistently wrong about?
```

#### Optimise slow code

```
@"Scientist (agent)" prediction.py is running slowly — find optimisations without changing the results
```

#### Understand the data

```
@"Scientist (agent)" what does the distribution of disposals look like across positions and teams?
@"Scientist (agent)" are there any data quality issues in the player CSVs I should know about?
```

---

### The improvement loop — how to get better predictions week by week

```
1. Refresh data        →  type: refresh the data
2. Run backtest        →  python backtest.py --start-year 2026 --start-round 1
3. Invoke Scientist    →  @"Scientist (agent)" analyse the backtest and improve the model
4. Re-run backtest     →  verify MAE dropped
5. Push changes        →  type: push the changes to main
6. Predict next round  →  python prediction.py
```

Repeat steps 3–5 as many times as you like. Each iteration typically improves MAE by 0.2–0.5 disposals. The improvement compounds — the model that predicted 2026 R8 at MAE 4.1 started at MAE 4.9 after several rounds of Scientist-driven improvements.

---

### Other useful Claude commands (no Scientist needed)

These work by just typing them at the Claude prompt:

| What you type | What Claude does |
|---------------|-----------------|
| `push the changes to main` | Commits all changes and pushes to GitHub |
| `update the readme` | Updates this file based on what changed |
| `what does prediction.py do?` | Explains the code in plain English |
| `the backtest crashed with [paste error] — fix it` | Diagnoses and fixes the error |
| `show me the top 10 most over-predicted players from the last backtest` | Reads the CSV and answers |
| `what is the current prediction accuracy?` | Summarises the latest backtest results |
| `explain why the model under-predicted [player name]` | Looks at their stats and explains |

---

### Tips for better results

- **Paste the full error message.** If something crashes, copy everything from the terminal and paste it in. Claude fixes it faster with the full stack trace.
- **Be specific about what you care about.** "I care more about getting the top 20 players right than overall accuracy" helps the Scientist make better trade-offs.
- **Ask it to explain before changing.** Type "what would you change and why?" first — you can redirect it before it touches any code.
- **One session, one push.** At the end of any session where the model improved, type "push the changes to main" so you don't lose the work.
- **Tell the Scientist what surprised you.** "The model got Pendlebury badly wrong last round — why?" is better than "improve accuracy." Specific questions get specific answers.

## Data sources

- Match results and player stats: [AFL Tables](https://afltables.com/afl/afl_index.html)
- Historical betting odds: [AusSportsBetting](https://www.aussportsbetting.com/data/historical-afl-results-and-odds-data/)


## Contributing

Got an idea, found a bug, or want to add a new feature? Open an issue or send a pull request — all contributions welcome.


## License

MIT License — see the [LICENSE](LICENSE) file.
