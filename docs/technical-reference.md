# Technical Reference

> [← Back to main README](../README.md)

<!-- This file is part of the SuperCoach-VIA documentation. See README.md for the live data sections. -->


**Who is this section for?** Anyone setting up the environment or contributing back. GPU configuration, the on-disk data layout, where the source data comes from, and how to contribute or use the project.

Setup guides and reference docs. GPU acceleration is optional, the data layout is documented, and the source links and contributing notes round out the section.

### Setting up GPU acceleration (optional)

The prediction scripts run faster with an NVIDIA GPU. If you don't have one, everything still works — just use `prediction_cpu.py` instead of `prediction.py`, and the ranking script falls back to CPU automatically.

#### Do you need GPU setup?

- **No GPU / not sure** — use `prediction_cpu.py`, skip this section
- **NVIDIA GPU on Linux** — follow the steps below
- **NVIDIA GPU on Windows** — GPU DataFrame support doesn't work on native Windows; either use WSL2 (Windows Subsystem for Linux) or use `prediction_cpu.py`

#### Step 1 — Check your GPU works

```bash
nvidia-smi
```

You should see your GPU name and a CUDA version number. If this command fails, install the latest NVIDIA driver from [nvidia.com/drivers](https://www.nvidia.com/drivers) first.

#### Step 2 — Install the CUDA toolkit

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

#### Step 3 — Install the GPU libraries

```bash
# GPU DataFrame library (speeds up ranking)
pip install cudf-cu12 cupy-cuda12x --extra-index-url=https://pypi.nvidia.com

# GPU-enabled LightGBM (speeds up predictions)
pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON
```

#### Step 4 — Verify everything works

```bash
python cuDF_test.py   # should print "cuDF working"
python testGPU.py     # should print "LightGBM GPU working"
python prediction.py  # should say "Running on GPU" at the top
```

#### Laptop tips

If you're on a gaming laptop and predictions feel slow:
- Set your laptop to Performance power mode
- Check that it's using the NVIDIA GPU, not the integrated Intel/AMD one (look in BIOS or Armoury Crate)
- If you see `CUDA out of memory`, switch to `prediction_cpu.py`

### How the data is organised

```
data/
  matches/          — one CSV per year: every match result 1897–now
  lineups/          — team lineups by season
  player_data/      — one CSV per player: their full career stats
  top100/           — all-time and yearly rankings
  prediction/       — disposal predictions (next round)
    backtest/       — historical accuracy results
```

#### Match data columns

Each row in `data/matches/matches_YEAR.csv` is one game:

`year, round, venue, date, home_team, away_team, home_q1_g, home_q1_b, ...`  
(goals and behinds for each quarter, final totals, winning team, margin)

#### Player data columns

Each player has two files in `data/player_data/`:

**Performance file** (`*_performance_details.csv`) — one row per game played:  
`team, year, round, opponent, kicks, marks, handballs, disposals, goals, behinds, tackles, clearances, brownlow_votes, ...`

**Personal file** (`*_personal_details.csv`):  
`first_name, last_name, born_date, debut_date, height, weight`

### Data sources

- Match results and player stats: [AFL Tables](https://afltables.com/afl/afl_index.html)
- Historical betting odds: [AusSportsBetting](https://www.aussportsbetting.com/data/historical-afl-results-and-odds-data/)

### Quick command reference

Bookmark this — it's everything you'll ever need to type:

| What you want to do | Command |
|---|---|
| Get this week's predictions (GPU) | `python prediction.py` |
| Get this week's predictions (CPU, slower) | `python prediction_cpu.py` |
| Update all data + rankings + README | `./refresh_and_rank.sh` |
| Check how accurate past predictions were | `python backtest.py --start-year 2026 --start-round 1` |
| Pull the latest code changes | `git pull` |
| Ask Claude a question about the data | `claude` (then type your question) |
| Activate your Python environment | `source .venv/bin/activate` |

### Glossary

A quick plain-English reference for the footy and tech terms used in this README. If something here didn't make sense above, this is the place to look it up.

**Footy stats**

- **Disposal** — a kick or handball; the main currency of AFL statistics, and the thing this project's prediction model is built around.
- **Contested possession** — winning the ball when an opponent is trying to take it from you; a sign of physicality and willingness to compete at ground level.
- **Clearance** — getting the ball away from a stoppage (a ball-up or boundary throw-in); a key indicator of midfield dominance.
- **Inside 50** — moving the ball into your forward 50-metre arc; more inside 50s = more scoring opportunities.
- **Hitout** — a ruckman tapping the ball at a ruck contest. Not all hitouts are useful, which is why "hitouts to advantage" matters in modern analysis.
- **Clanger** — a mistake: a turnover, a dropped mark, an errant kick. Lower is better.
- **Brownlow Medal** — the AFL's individual award for the "fairest and best" player, voted on by on-field umpires using a 3-2-1 system per game.
- **SuperCoach** — Australia's most popular AFL fantasy competition. You pick a squad of real AFL players and score points based on their actual in-game stats each week. Disposals are the biggest scoring category.

**Tech and AI**

- **LightGBM** — a fast machine-learning library (the kind of maths the computer uses to find patterns); this project uses it to predict disposals.
- **Machine learning** — letting a computer find patterns in lots of past data and then use those patterns to make a guess about a new situation. No, it isn't sentient.
- **z-score** — a number that says how many "standard steps" above or below average something is; a z-score of +2 means "much better than average".
- **Walk-forward backtest** — testing a prediction model by simulating how it would have performed in the past, round by round, without using future data to cheat. The honest way to grade a model.
- **GPU** — the graphics card in a computer; originally built for video games, but also very fast at the kind of maths machine learning needs. Running without a GPU is like doing your tax return by hand instead of with a calculator — possible, just much slower.
- **CLI / terminal** — the text window where you type commands. Looks old-fashioned, but it's the fastest way to run data pipelines.
- **Python** — the programming language all the scripts in this project are written in. You don't need to know Python to use this project.
- **venv** — short for "virtual environment". It's a Python self-contained box of software libraries so this project doesn't mess with anything else on your laptop.
- **Claude / Claude Code** — Claude is Anthropic's AI assistant; Claude Code is the version that runs in your terminal so it can read your files, edit code, and run commands on your behalf.
- **Scientist agent** — a specialised version of Claude built into this project that runs on the most powerful (and most expensive) Claude model. Use it sparingly — see the warning above the Scientist examples.
