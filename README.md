# 🏈 AFL Data Analysis
![AFL Data Analysis Banner Image](/assets/readme_banner.png)
<div align="center">
  <img src="https://img.shields.io/github/last-commit/akareen/AFL-Data-Analysis">
  <img src="https://img.shields.io/github/contributors/akareen/AFL-Data-Analysis">
  <img src="https://img.shields.io/github/stars/akareen/AFL-Data-Analysis?style=social">
  <img src="https://img.shields.io/github/forks/akareen/AFL-Data-Analysis?style=social">
</div>
<br>
An in-depth analysis of Australian Football League (AFL) data. This repository contains comprehensive data, tools and code for exploring and analysing AFL match and player statistics, as well as historical odds data.

## Table of Contents
- [🔦 Overview](#overview)
  - [🛠 Features](#features)
- [💾 Installation](#installation)
  - [📖 Usage](#usage)
  - [📁 Repository Structure](#repository-structure)
  - [🔍 Scraping Examples](#scraping-examples)
- [📚 Data Guide](#data-guide)
- [🔗 Data Sources](#data-sources)
- [🤝 Contributing](#contributing)
- [🎓 Learning Pointers](#learning-pointers)
- [⚖️ License](#license)


## Overview

The AFL Data Analysis project provides a comprehensive platform for examining and deriving insights from AFL match and player data. Whether you're a sports enthusiast, a tipper, a data scientist, or a student, this repository offers valuable resources for diving into the world of Australian Rules Football. 

The repository currently stores match scores data from 1897 to 2024, in depth personal and game statistics for every player who have ever played in the VFL/AFL and historical odds data from 2009 to 2024. All the data is  conveniently stored in CSV format for seamless access and analysis.

Download the repository and explore the **/data/** directory for the complete dataset. 

Contributions are encouraged; don't hesitate to submit a pull request or contact me with the details on my GitHub profile.

## Features

**Current Offerings:**
- Profiles for 5,700+ players, 682,000 rows of player performance data with 19 million data points
- Statistics for 15,000+ matches, inclusive of individual player performance
- Historical odds data for strategic tipping and betting
- Cleansed data, primed for analysis
- Analytical Jupyter notebooks showcasing potential insights
- Python classes for players, teams, and matches

**In the Pipeline:**
- Expanding the classes to allow for complex analysis
- Dedicated database system
- Advanced scoring algorithms
- Visualization tools for performance metrics

**Suggestions?**
- Pitch in your wishlist. One current suggestion: Player GPS Data

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/apur27/SuperCoach-VIA.git
   cd SuperCoach-VIA
   ```

2. (Skip if you just want the data) Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running GPU-Accelerated Code on a Gaming PC or Laptop

Several scripts in this repo — `prediction.py`, `top_players_comprehensive.py`, `testGPU.py`, `cuDF_test.py` — can use an NVIDIA GPU for faster processing. GPU support is always optional; every script falls back to CPU automatically when a GPU isn't available or isn't set up.

This section covers the full setup for a gaming PC or laptop with an NVIDIA RTX GPU (3000, 4000, or 5000 series).

### What GPU acceleration is used for

| Script | GPU library | What it speeds up |
|--------|-------------|-------------------|
| `prediction.py` | LightGBM CUDA | Hyperparameter tuning + model training |
| `top_players_comprehensive.py` | cuDF (RAPIDS) | DataFrame operations on player CSVs |
| `testGPU.py` | LightGBM CUDA | GPU smoke test |
| `cuDF_test.py` | cuDF + CuPy | RAPIDS smoke test |

### Step 1 — Check your GPU

Open a terminal and run:

```bash
nvidia-smi
```

You should see your GPU name, driver version, and CUDA version. If this command is not found, install the latest NVIDIA driver from [nvidia.com/drivers](https://www.nvidia.com/drivers) first.

Key requirements:
- NVIDIA GPU with CUDA Compute Capability 6.0+ (all GTX 10-series and newer, all RTX series)
- Driver version 520+ recommended
- CUDA 11.8 or 12.x (check the `CUDA Version` field in `nvidia-smi` output)

### Step 2 — Windows vs Linux

**Linux (native):** All GPU libraries work out of the box. Skip to Step 3.

**Windows:** RAPIDS (cuDF/CuPy) does **not** run on native Windows. You have two options:

- **Option A — WSL2 (recommended):** Run everything inside Windows Subsystem for Linux. Install WSL2, then follow the Linux steps below inside your WSL2 terminal. NVIDIA's WSL2 driver passes your GPU through automatically — no separate Linux GPU driver needed inside WSL2.

  ```powershell
  # In PowerShell (admin)
  wsl --install
  # Restart, then open Ubuntu from the Start menu
  ```

- **Option B — CPU only:** Use `prediction_cpu.py` instead of `prediction.py`. The ranking scripts always fall back to CPU automatically.

### Step 3 — Install the CUDA Toolkit

Even if your driver already shows a CUDA version, you need the toolkit for libraries to compile against.

```bash
# Ubuntu / Debian (WSL2 or native Linux)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

After installation, add CUDA to your PATH (add to `~/.bashrc`):

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

Verify with:

```bash
nvcc --version
```

### Step 4 — Install RAPIDS (cuDF + CuPy)

RAPIDS is the GPU DataFrame library used by `top_players_comprehensive.py`. It requires Linux (or WSL2).

Use the RAPIDS selector at [rapids.ai/start](https://rapids.ai/start) for the exact command matching your CUDA version. For CUDA 12.x:

```bash
pip install cudf-cu12 cupy-cuda12x --extra-index-url=https://pypi.nvidia.com
```

For CUDA 11.8:

```bash
pip install cudf-cu11 cupy-cuda11x --extra-index-url=https://pypi.nvidia.com
```

Verify:

```bash
python cuDF_test.py
```

> **Note:** RAPIDS requires significant VRAM. Gaming laptops with 6 GB VRAM or less may run into out-of-memory errors on large datasets. The scripts automatically fall back to pandas/CPU in that case.

### Step 5 — Install LightGBM with CUDA support

The default `pip install lightgbm` does NOT include GPU support. You need to build or install the GPU variant:

```bash
# Option A: pre-built wheel (easiest)
pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON

# Option B: if that fails, install from source
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build && cd build
cmake -DUSE_CUDA=1 ..
make -j4
cd ../python-package
pip install .
```

Verify:

```bash
python testGPU.py
```

### Step 6 — Laptop-specific tips

Gaming laptops often throttle GPU performance by default. Before running long training jobs:

1. **Set power mode to Performance** — Windows: Battery icon → Best Performance. Linux: `sudo nvidia-smi -pm 1`
2. **Disable GPU switching** — Some laptops (MUX switch) run games through the NVIDIA GPU but route other apps through the integrated Intel/AMD GPU. Check your laptop's BIOS or Armoury Crate / Control Center and set discrete GPU mode.
3. **Monitor GPU temperature** — `watch -n 1 nvidia-smi` shows real-time temperature. Above 85°C sustained means your cooling needs attention (clean fans, repaste).
4. **VRAM limits** — `prediction.py` batches data to avoid VRAM exhaustion, but if you hit `CUDA out of memory`, reduce `max_bin` in the LightGBM parameters or switch to `prediction_cpu.py`.

### Quick verification checklist

```bash
# 1. Driver
nvidia-smi

# 2. CUDA toolkit
nvcc --version

# 3. cuDF + CuPy (RAPIDS)
python cuDF_test.py

# 4. LightGBM CUDA
python testGPU.py

# 5. Full prediction pipeline (GPU)
python prediction.py

# 5b. Full prediction pipeline (CPU fallback)
python prediction_cpu.py
```

All scripts print whether they are running on GPU or CPU at startup, so you can confirm the right path is taken.

## Usage

I regularly update the CSV data files in the **/data** directory with the latest AFL match and player data. But you can also do your own data scraping using the provided scripts in the "scripts" directory. Scripts, using the Beautiful Soup library, are available for web scraping.

For up-to-date statistics in your own copy of the repository, the primary pipeline entry point is:

```bash
./refresh_and_rank.sh
```

This runs the full end-to-end workflow: refreshes the underlying CSVs (player + match scrapers), regenerates the all-time top 100 ranking, and produces the formatted `all_time_top_100.csv` enriched with player bios. If you only need a partial refresh, the individual scripts (`player_scraper.py`, `game_scraper.py`, `refresh_data.py`, `top_players_fast.py`, `formatTop100.py`) can also be invoked directly.


## Repository Structure

The project organizes data and scripts into several key locations:

- `data/matches/` – yearly match results (`matches_<year>.csv`).
- `data/lineups/` – team lineups by season.
- `data/player_data/` – individual player profiles and performance details.
- `data/top100/` – yearly and all-time top player rankings.
- `data/era_stats.csv` – aggregate statistics for each AFL era.

### Scripts

**Data acquisition & refresh**

- `main.py` – entry point for the full data scraping pipeline (kicks off the scrapers from scratch).
- `player_scraper.py` – scrapes player personal and performance data from AFL Tables using concurrent requests.
- `game_scraper.py` – scrapes match results and team lineups.
- `refresh_data.py` – orchestrates an incremental data refresh (calls the scrapers and updates CSVs in place).
- `refresh_and_rank.sh` – full pipeline shell wrapper: refresh data → rank → format the top 100.

**Ranking**

- `top_players_comprehensive.py` – **sole ranking script**: computes the era-normalised all-time top 100 using a single-pass file ingestion (reads each of ~13k player files exactly once for ~100× speed improvement over year-by-year approaches). See [All-Time Top 100 Ranking Algorithm](#all-time-top-100-ranking-algorithm).
- `formatTop100.py` – reads `data/top100/all_time_top_100.csv`, enriches with player bios, and writes the root-level `all_time_top_100.csv`.

**Analysis & visualisation**

- `analysis.py` – team performance analysis and heatmap generation.
- `era_based_statistical_analysis.py` – era-by-era statistical comparison with per-100-minute normalisation.
- `charts.py` / `bar_chart.py` – visualisation scripts for team and player metrics.

**Prediction**

- `prediction.py` / `prediction_cpu.py` – match outcome prediction models (GPU and CPU variants).
- `prediction_accuracy.py` – evaluates the prediction model's accuracy against held-out matches.

**Utilities**

- `helper_functions.py` – shared utilities used across scripts.
- `cuDF_test.py` / `testGPU.py` – GPU/cuDF availability checks for the prediction stack.


## All-Time Top 100 Ranking Algorithm

The ranking in `top_players_comprehensive.py` uses an **era-normalised z-score dominance** approach. The goal is to produce a defensible all-time list that does not unfairly favour modern players (more tracked stats, no shrinkage) over historical legends. The methodology has five steps.

### Step 1 — Era-aware raw scoring (per season, per player)

Each season's raw score is computed using only the statistics tracked in that era:

| Era | Years | Stats available |
|-----|-------|----------------|
| Pre-1965 | 1897–1964 | Goals, behinds |
| 1965–1990 | 1965–1990 | + kicks, handballs |
| 1990–2010 | 1991–2010 | + marks |
| Post-2010 | 2011–present | + tackles, clearances, contested possessions, contested marks, one-percenters, goal assists |

**Scoring weights:** `goals=55, kicks=4.5, handballs=3.0, clearances=5.5, contested_possessions=5.5, contested_marks=7, goal_assist=4, tackles=3.5, one_percenters=3, marks=2.5, behinds=1.5`.

> **Note on disposals:** `disposals` is intentionally absent — since `disposals = kicks + handballs`, weighting all three double-counts every kick. Kicks and handballs are scored separately instead.

**55% single-stat cap:** No single statistic can contribute more than 55% of a player's raw score for a season. This prevents extreme goal-kicking seasons from totally dominating while still allowing goal kickers to be properly rewarded (the old 40% cap was confirmed to over-penalise elite forwards like Lockett and Dunstall in sparse-stat eras).

### Step 2 — Three-group career position stratification

Players are z-scored within their career position group, not against the full cohort. The goal weight (55 per goal) means a full-forward kicking 4+ goals/game scores vastly more in raw terms than a midfielder — without stratification, every year's top z-scores are monopolised by goal kickers, unfairly depressing complete players.

Groups are determined by **career** goals/game (not single-year, to prevent year-to-year classification flipping):

| Group | Career goals/game | Examples |
|-------|-------------------|---------|
| `key_forward` | ≥ 3.0 | Lockett (4.84), Dunstall (4.66), Ablett Sr (~4.1), Lloyd (4.17), Franklin (3.01) |
| `forward_mid` | 0.80 – 2.99 | Carey (2.67), Matthews (2.75), Bartlett (1.93), Dangerfield (~1.0), Ablett Jr (~1.25) |
| `other` | < 0.80 | Pendlebury (0.48), Parker (0.70), Neale (0.45) |

The earlier binary split (≥ 1.0 g/game → forward) failed because it lumped Wayne Carey (2.67) with Tony Lockett (4.84) in the same pool. Carey consistently appeared below-average within elite full-forwards despite being a more complete player. Three groups give each type fair peer comparison. If a group has fewer than 5 players in a given year, it falls back to the full-cohort z-score.

### Step 3 — Era completeness shrinkage

Each season's z-score is multiplied by `sqrt(era_completeness)`. Values are calibrated to close the structural gap between eras — the original 1.0 (no shrinkage) for the post-2010 era was creating a mathematical ceiling that pre-1990 players could never reach regardless of their true dominance:

| Era | Completeness | Shrinkage factor |
|-----|--------------|------------------|
| Pre-1965 | 0.65 | × 0.806 |
| 1965–1990 | 0.78 | × 0.883 |
| 1990–2010 | 0.90 | × 0.949 |
| Post-2010 | 0.92 | × 0.959 |

Note: post-2010 is 0.92 rather than 1.0 because modern stats still omit GPS distance, pressure acts, and defensive ratings — even full modern tracking is not a complete picture.

### Step 4 — All-time score formula

A player's final score is:

```
all_time_score = mean_z_top8 × (1 + career_bonus) + peak_bonus
```

- **`mean_z_top8`** — average era-adjusted z-score across the player's best **8** seasons. Top-5 was trialled but proved too vulnerable to outlier seasons — Stewart Loewe and Barry Hall reached top-10 all time on 2–3 exceptional goal-kicking years. Top-8 rewards sustained excellence without penalising long careers for weak tail seasons.
- **`career_bonus`** = `0.30 × min(career_games / 300, 1.0)` — **additive** bonus (max +30%) for longevity. Critically, this is *additive* not *multiplicative*: a 364-game player and a 300-game player both receive the same 30% bonus, so a longer career cannot overcome lower per-season z-scores. The old multiplicative formula (×1.5 max) was producing Brad Johnson #1, Wayne Carey #16 — directly contradicting expert consensus. True career game count used (all games played, not just top-100 seasons; a longstanding bug was silently dropping injury seasons, under-counting Carey by 60 games, Voss 83, Hird 89). Minimum 150 career games required.
- **`peak_bonus`** = `0.25 × best_season_z_adj` — raised from 0.15 to reward "seasons where you were clearly the #1 player in the competition" (Carey 1996, Matthews 1975, Ablett Sr's 1989 Grand Final).

> **Brownlow bonus removed:** it created recency bias — modern midfielders accumulate votes across far more fully-tracked seasons.

### Step 5 — Decade representation guarantee

To ensure historical breadth, the top scorer from each decade (1897–1909, 1910s … 2020s) is guaranteed inclusion. Remaining spots are filled by overall score. Reduced from top-3 to top-1 per decade: the old 3-per-decade rule reserved 39 of 100 slots for decade anchors, displacing genuine greats (e.g. Tony Lockett at #90 on merit) in favour of 1890s–1900s players whose sparse stats cannot be fairly compared.

The output is written to `data/top100/all_time_top_100.csv` and then enriched with player bios by `formatTop100.py` into the root-level `all_time_top_100.csv`.

### Re-running the rankings from scratch

To force a full recalculation (e.g. after algorithm changes), delete the cached yearly files before running:

```bash
rm -f data/top100/all_time_top_100.csv
rm -f data/top100/yearly/*.csv
python top_players_comprehensive.py   # single-pass, processes 1897–present (~5–10 min)
python3 formatTop100.py               # enriches with player bios → all_time_top_100.csv
```

Or use the full pipeline (also refreshes match/player data first):

```bash
./refresh_and_rank.sh
```


## Data Guide

### Match Data -  Explanation

The repository contains the information for all matches from 1897-2023.

![Match Data Example](/assets/matchdata_example.png)

The above includes part of the data the columns are too numerous to show completely. An example of a selection of the match data can be seen here: [matches_2023.csv](data/matches/matches_2023.csv)

**The columns for each match are as follows:**   
Year, Round, Venue, Date,  
Home Team, Away Team,   
Home Teams Goals by Quarter, Home Teams Behinds by Quarter,  
Away Teams Goals by Quarter, Away Teams Behinds by Quarter,  
Home Total Goals, Home Total Behinds,  
Away Total Goals, Away Total Behinds,  
Winning Team, Margin  

All the columns are in **snake case** and there is a column for each quarter such as **home_q1_g** for Home Team Quarter 1 goals and **away_q1_b** for Away team Quarter 1 Behinds.

----
### Player Data - Explanation

#### Player Performance Data

![Player Performance Data Example](/assets/playerstats_example.png)

An example of the player performance data can be seen here: [BONTEMPELLI_MARCUS_24-11-1995_STATS.csv](data/players/bontempelli_marcus_24111995_performance_details.csv)

**The columns for each player are as follows:**  

Team, Year, Games Played, Opponent, Round, Result,   
Jersey Num, Kicks, Marks, Handballs, Disposals, Goals, Behinds, Hit Outs,   
Tackles, Rebound 50s, Inside 50s, Clearances, Clangers, Free Kicks For, Free Kicks Against,   
Brownlow Votes, Contested Possessions, Uncontested Possessions, Contested Marks, Marks Inside 50,   
One Percenters %, Bounces, Goal Assist, % Percentage of Game Played

Inside the data all the columns are in **snake case**. The file format is *{last_name}_{first_name}_{born_date}_performance.csv*.


#### Player Personal Data

**The columns for each player are as follows:**
First Name, Last Name, Born Date, Debut Date, Height, Weight

Inside the data all the columns are in **snake case** and the players born date along with first and last name are used to create a unique identifier for each player. The file format is The file format is *{last_name}_{first_name}_{born_date}_personal.csv*.


## Data Sources

This project uses publicly available AFL data sources, including match scores, player statistics, and historical odds data. The data sources are as follows:

- Match and Player Data: [AFL Tables](https://afltables.com/afl/afl_index.html)
- Historical Odds Data: [AusSportsBetting](https://www.aussportsbetting.com/data/historical-afl-results-and-odds-data/)

## Scraping Examples
While data is readily available in the repository, here's how you can use scraping if needed.

- To scrape match scores data from 1897 to 2023 and store it in the "data/match_scores" directory:

  ```bash
  python main.py
  ```

This used to be more granular, but the data is now fully available in the repository. I'll be making a small update soon to bring back the ability to only scrape most recent data without rescraping existing data.


## Contributing

AFL Data Analysis thrives on collaboration! Got a novel analysis idea or data source? Open an issue or send a pull request. Your expertise is invaluable in elevating this project.

## Learning Pointers

- Explore the CSV layouts in the `data` directory to understand the available metrics.
- Experiment with `top_players_comprehensive.py` by adjusting weighting schemes or era definitions.
- Review the generated heatmaps and charts in the `charts` folder for visualization examples.
- Try extending the scrapers or analysis scripts to incorporate new metrics or plots.

## License

AFL Data Analysis is under the MIT License. Refer to the [LICENSE](LICENSE) file for a complete understanding.
