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

The ranking in `top_players_comprehensive.py` uses an **era-normalised z-score dominance** approach. The goal is to produce a defensible all-time list that does not unfairly favour either modern players (who have more tracked stats) or pre-1965 goal kickers (who played in a 2-stat era where goals dominate everything). The methodology has four steps.

### Step 1 — Era-aware raw scoring (per season, per player)

Each season's raw score is computed using only the statistics that were tracked in that era:

| Era | Years | Stats available |
|-----|-------|----------------|
| Pre-1965 | 1897–1964 | Goals, behinds |
| 1965–1990 | 1965–1990 | + kicks, handballs |
| 1990–2010 | 1991–2010 | + marks, disposals |
| Post-2010 | 2011–present | + tackles, clearances, contested possessions, contested marks, one-percenters, goal assists |

**Scoring weights:** `goals=55, kicks=4.5, handballs=3.0, clearances=5.5, contested_possessions=5.5, contested_marks=7, goal_assist=4, tackles=3.5, one_percenters=3, marks=2.5, behinds=1.5`.

> **Note on disposals:** `disposals` is intentionally absent from the weights and era stat lists. Since `disposals = kicks + handballs`, weighting all three would double-count every kick. Instead, kicks and handballs are scored separately — handballs at ~65% of kick value (3.0 vs 4.5), reflecting their lower distance and accuracy.

**40% single-stat cap:** No single statistic can contribute more than 40% of a player's raw score for a season. This prevents one-dimensional specialists — most notably pre-1965 goal kickers in a goals-only era — from running away with the ranking on the back of a single metric.

### Step 2 — Position-stratified z-scores

Each player is classified by role (forwards: ≥1 goal/game; everyone else lumped together as midfielders / defenders / rucks) and z-scored **within their position group for that season**. This avoids the trap where a pre-1965 forward looks +5σ simply because midfielders in that era had zero recorded goals — the comparison is now within forwards only.

### Step 3 — Era completeness shrinkage

Each season's z-score is multiplied by `sqrt(era_completeness)`, where completeness reflects how many of the modern stat categories were tracked in that era:

| Era | Completeness | Shrinkage factor |
|-----|--------------|------------------|
| Pre-1965 | 0.40 | × 0.632 |
| 1965–1990 | 0.65 | × 0.806 |
| 1990–2010 | 0.82 | × 0.906 |
| Post-2010 | 1.00 | × 1.000 |

This is epistemic humility encoded into the score: a +3σ season evidenced by 2 stats is genuinely less convincing than a +3σ season evidenced by 12 stats, and the shrinkage formalises that.

### Step 4 — All-time score formula

A player's final score is:

```
all_time_score = mean_z_top8 × longevity + peak_bonus
```

- **`mean_z_top8`** — the average era-adjusted z-score across the player's best 8 seasons. Using the top 8 rather than career average rewards sustained excellence without penalising long careers for weak tail seasons.
- **`longevity`** = `min(career_games / 250, 1.5)` — games-based durability multiplier, capped at 1.5×. Using actual games played (not seasons) meaningfully separates a 9-season/185-game current player from a 19-season/400-game all-time great. Minimum 150 career games required to qualify.
- **`peak_bonus`** = `0.15 × best_season_z_adj` — small additive nudge rewarding peak dominance (e.g. a Norm Smith Medal year) without letting it dominate the career signal.

> **Brownlow bonus removed:** an earlier version included a Brownlow vote prior, but it created recency bias — modern midfielders accumulate votes across many more fully-tracked seasons, unfairly inflating their scores relative to pre-1990 players.

### Step 5 — Decade representation guarantee

To ensure the list spans the full history of the game, the top 3 scorers from each decade (1897–1909, 1910s, 1920s … 2020s) are guaranteed inclusion. Remaining spots are filled by overall score. The final list is re-sorted by score so rank order always reflects statistical merit.

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
