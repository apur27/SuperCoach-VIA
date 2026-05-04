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

⭐ **If this project is useful to you, please star the repo.**

## What this is

An applied data science and AI project using Australian rules football as the domain. The game has been played since 1897, which means 125+ years of structured match data — enough to do something genuinely interesting with machine learning, statistical modelling, and large language models.

Three problems drive the work:

1. **Prediction** — Can a model learn a player's disposal patterns well enough to forecast next-round output better than intuition? The answer, with a gradient-boosted ensemble trained on rolling form, opponent, venue and context features, is: often yes.
2. **Historical ranking** — How do you compare players across eras when the game itself has changed radically? Era-normalised z-scoring gives a principled answer, not just a pub argument.
3. **Natural language over structured data** — Claude (via the Scientist agent) can answer questions about the dataset in plain English, write and run its own analysis code, and update the docs automatically. This is what "AI applied to sport" looks like in practice.

The entire pipeline — data scrape → feature engineering → model training → prediction → backtest → doc generation — runs from a single shell script and updates this repo every week.

## Who is this for?

| I am… | Start here |
|---|---|
| **A data scientist or ML engineer** | [How predictions work](docs/prediction-model.md) — the model, feature engineering, walk-forward backtest |
| **Curious about AI applied to sport** | [Using Claude and the Scientist agent](docs/scientist-agent.md) — LLMs writing and running analysis code over live data |
| **A footy fan** who wants to understand their team | [Live AFL insights](docs/afl-insights.md) — current-season team profiles, finals pathway, stat leaders |
| **A SuperCoach player** wanting a data edge | [Running predictions & backtests](docs/usage.md) — produces this week's predicted disposals CSV |
| **New to all of this** on Ubuntu | [Installation & first-time setup](docs/installation.md) — Git, GitHub, Python, end to end |

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
