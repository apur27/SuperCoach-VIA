# AFL SuperCoach VIA
![AFL SuperCoach VIA Banner](/assets/banner.svg)
<div align="center">
  <img src="https://img.shields.io/github/last-commit/apur27/SuperCoach-VIA">
  <img src="https://img.shields.io/github/contributors/apur27/SuperCoach-VIA">
  <img src="https://img.shields.io/github/stars/apur27/SuperCoach-VIA?style=social">
  <img src="https://img.shields.io/github/forks/apur27/SuperCoach-VIA?style=social">
</div>

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Data](https://img.shields.io/badge/data-2026%20season%20round%208-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

⭐ **If this helped your SuperCoach team this week, please star the repo** — it helps other footy fans find it.

> **⚡ TL;DR — I just want this week's predictions (no reading required)**
>
> 1. Open Terminal → paste these 4 lines one by one:
>    ```bash
>    git clone https://github.com/apur27/SuperCoach-VIA.git
>    cd SuperCoach-VIA
>    pip install -r requirements.txt
>    python prediction_cpu.py   # works on any laptop — or prediction.py if you have a GPU
>    ```
> 2. Open the file `data/prediction/next_round_*_prediction_*.csv` in Excel or Google Sheets.
> 3. Done — you now have predicted disposals for every player this week.
>
> Takes roughly 5–10 minutes on a normal laptop. Full details in [the docs](docs/).

<br>

A personal AFL data project that does three things:
1. **Stores every AFL match and player stat** going back to 1897
2. **Ranks the greatest players of all time** using a fair, era-adjusted formula
3. **Predicts how many disposals each player will get** in the next round

## Who is this for?

| I am… | Start here |
|---|---|
| **A footy fan** who wants to understand their team better | [Live AFL insights](docs/afl-insights.md) — current-season team profiles, finals pathway, who's playing well |
| **A SuperCoach player** wanting a data edge | [Running predictions & backtests](docs/usage.md) — produces this week's predicted disposals CSV |
| **Curious about AI applied to sport** | [Using Claude and the Scientist agent](docs/scientist-agent.md) — plain-English questions over your data |
| **A developer or data scientist** | [How predictions work](docs/prediction-model.md) and [Technical reference](docs/technical-reference.md) |
| **New to all of this** on Ubuntu | [Installation & first-time setup](docs/installation.md) — Git, GitHub, Python, end to end |

You don't need to write any code to use most of this. The [AFL insights page](docs/afl-insights.md) updates automatically every week — just read it.

> **What is SuperCoach?** SuperCoach is Australia's most popular AFL fantasy competition — you pick a squad of real AFL players and score points based on their actual in-game statistics each week. Disposals (kicks + handballs) are the biggest scoring category, which is why this project focuses on predicting them.

## Table of Contents

### Getting started
- [Quick start](docs/quick-start.md) — TL;DR, three commands
- [Installation & first-time setup](docs/installation.md) — Git, GitHub and Python from scratch
- [Running predictions & backtests](docs/usage.md) — predict next round, backtest, refresh data
- [Troubleshooting](docs/troubleshooting.md) — common errors and fixes

### AFL insights & live data
- [AFL insights hub](docs/afl-insights.md)
  - [2026 season hub](docs/afl-season-2026.md)
    - [Team analysis](docs/afl-team-analysis-2026.md) *(auto-updates)*
    - [Finals pathway](docs/afl-finals-2026.md) *(auto-updates)*
    - [Brownlow predictor](docs/afl-brownlow-2026.md) *(auto-updates)*
    - [Player stat leaders](docs/afl-stat-leaders-2026.md) *(auto-updates)*
    - [Next round predictions](docs/afl-predictions-2026.md) *(auto-updates)*
    - [Backtest results](docs/afl-backtest-2026.md) *(auto-updates)*
  - [5-year team profiles](docs/afl-team-profiles.md) *(auto-updates)*
  - [AFL history — 125 years](docs/afl-history.md)
  - [For the footy expert](docs/footy-expert-guide.md)
  - [For the coaching staff](docs/coaching-guide.md)
- [AFL Hall of Fame](docs/hall-of-fame.md) — all-time top 100, captains, coaches, courageous players

### Technical guides
- [Claude Code setup on Ubuntu](docs/claude-code-setup.md) — install Node.js, Claude Code, Python venv, default model
- [Using the Scientist agent](docs/scientist-agent.md) — when plain Claude vs the Scientist, the improvement loop
- [How predictions work](docs/prediction-model.md) — the model, the backtest framework, the all-time-100 algorithm
- [Technical reference](docs/technical-reference.md) — GPU setup, data layout, scripts

### About
- [Roadmap & contributing](docs/roadmap.md)
