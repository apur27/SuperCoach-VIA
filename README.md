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

For up-to-date statistics in your own copy of the repository, run `player_scraper.py` and `game_scraper.py` every week. This fetches the latest player info and match results so your data stays current.


## Repository Structure

The project organizes data and scripts into several key locations:

- `data/matches/` – yearly match results (`matches_<year>.csv`).
- `data/lineups/` – team lineups by season.
- `data/player_data/` – individual player profiles and performance details.
- `data/top100/` – yearly and all-time top player rankings.
- `data/era_stats.csv` – aggregate statistics for each AFL era.

Important scripts include:

- `player_scraper.py` – collects player data from AFL Tables using concurrent requests.
- `game_scraper.py` – downloads match results and team lineups.
- `analysis.py` and `era_based_statistical_analysis.py` – generate team metrics and era comparisons.
- `top_players_comprehensive.py` – ranks players for each season and all time.


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
