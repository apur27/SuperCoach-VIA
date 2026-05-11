# AFL SuperCoach VIA
![AFL SuperCoach VIA Banner](/assets/banner.svg)
<div align="center">
  <img src="https://img.shields.io/github/last-commit/apur27/SuperCoach-VIA">
  <img src="https://img.shields.io/github/contributors/apur27/SuperCoach-VIA">
  <img src="https://img.shields.io/github/stars/apur27/SuperCoach-VIA?style=flat-square">
  <img src="https://img.shields.io/github/forks/apur27/SuperCoach-VIA?style=flat-square">
</div>

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Data](https://img.shields.io/badge/data-2026%20season%20round%209-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

SuperCoach VIA turns 125+ years of AFL data into weekly player predictions, team trends, and debate-ready footy insights - no coding required.

⭐ **If this project is useful to you, please star the repo.**

## Start here - I want to...

| I want to... | Go to | Setup needed |
|---|---|---|
| **See this week's predicted disposal leaders** | [docs/afl-predictions-2026.md](docs/afl-predictions-2026.md) | None - browser only |
| **Browse the no-code fan landing page** | [docs/start-here-no-code.md](docs/start-here-no-code.md) | None - browser only |
| **Understand what this is good for in SuperCoach** | [docs/how-to-use-this-for-supercoach.md](docs/how-to-use-this-for-supercoach.md) | None - browser only |
| **Get the prediction CSV into Google Sheets** | [templates/google-sheets-template.md](templates/google-sheets-template.md) | A free Google account |
| **Read the auto-updated 2026 season hub** | [docs/afl-season-2026.md](docs/afl-season-2026.md) | None - browser only |
| **See the all-time top 100 and Hall of Fame** | [docs/hall-of-fame.md](docs/hall-of-fame.md) | None - browser only |
| **Look up a footy or data term** | [docs/glossary.md](docs/glossary.md) | None - browser only |
| **See how accurate the model has been** | [docs/afl-backtest-2026.md](docs/afl-backtest-2026.md) | None - browser only |
| **Read the model's pre-registered report card** | [docs/model-report-card.md](docs/model-report-card.md) | None - browser only |
| **Run predictions or retrain the model myself** | [docs/installation.md](docs/installation.md) (For Contributors section) | Python, Git, terminal |
| **Get tactical analysis on an AFL team's list and draft picks** | [docs/coaches-strategy-corner/afl-2026-team-list-analysis.md](docs/coaches-strategy-corner/afl-2026-team-list-analysis.md) | None - browser only |

## What this is

Australian rules football, used as the domain for an applied data science and AI project. The game has been played since 1897, which gives 125+ years of structured match data - enough surface area to do something genuinely interesting with machine learning, statistical modelling, and large language models.

The repo runs a full, weekly-refreshed pipeline: scrape new match and player data, retrain the disposal model, run a leak-proof walk-forward backtest, regenerate the all-time top-100, and update the documentation - all from a single shell script. Three problems drive the work.

### The prediction problem
Can a model learn a player's disposal patterns well enough to forecast next-round output better than intuition? Latest 2026 backtest (8 rounds, ~2,900 player-round predictions): **MAE ≈ 4.1 disposals, 68% within 5, 94% within 10**. See ["For data scientists"](#for-data-scientists) below for the model details.

### The historical ranking problem
How do you compare players across eras when the game itself has changed radically - different stats recorded, different rules, different season lengths? The all-time top-100 is built to be era-fair by construction rather than by quota.

### AI applied to sport
Claude - via the Scientist agent in this repo - does not just answer questions about the dataset. It reads the actual code, writes and runs Python analysis, regenerates charts, updates the auto-generated documentation sections, and commits the result. The pattern is "natural language as a thin wrapper over structured data" - and the agent is held to inspect-before-transform, baseline-first, leakage-aware practice via its system prompt.

A second specialist agent - **FootyStrategy** - brings AFL tactical knowledge to the same dataset: half-time structural resets, ruck-rotation patterns, list-construction analysis by draft pick, and the football-coach answer to "what does this number actually mean on the ground?" The two agents are designed to work together - Scientist reads the data, FootyStrategy interprets it tactically.

## Table of Contents

### For fans (no code)
- [Start here - no code](docs/start-here-no-code.md)
- [How to use this for SuperCoach](docs/how-to-use-this-for-supercoach.md)
- [Glossary](docs/glossary.md)
- [Google Sheets template](templates/google-sheets-template.md)
- [Weekly cheat sheet (current round)](docs/weekly/round-current-2026.md)

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
  - [Coaches strategy corner](docs/coaches-strategy-corner/README.md) - match-by-match tactical briefs built from the data
  - [AFL history - 125 years](docs/afl-history.md)
  - [For the footy expert](docs/footy-expert-guide.md)
  - [For the coaching staff](docs/coaching-guide.md)
  - [AFL 2026 team list analysis](docs/coaches-strategy-corner/afl-2026-team-list-analysis.md) - all 18 clubs: list identity, Tier 1 draft pedigree, tactical identity explained by list construction
- [AFL Hall of Fame](docs/hall-of-fame.md) - all-time top 100, statistical leaders, captains, coaches, courageous players, careers cut short, and the great dynasties

### About
- [Roadmap & contributing](docs/roadmap.md)
- [Changelog](CHANGELOG.md)

---

## For data scientists

If you're here for the methodology, the technical pages below have the full picture.

### The prediction model in detail
The current pipeline trains a tuned `LightGBM` / `HistGradientBoosting` / `RandomForest` ensemble on rolling-form, opponent, venue, and context features, with `GroupKFold` (player-grouped) cross-validation to prevent leakage and a post-hoc out-of-fold calibration to correct top-end compression.

> Full technical detail → **[How it works: data science deep-dive](docs/data-science.md)** - the dataset, model, backtest, ranking algorithm, current accuracy, and roadmap, written in three layers from layperson to ML practitioner.

### The all-time-100 algorithm
Built from per-year, position-stratified within-cohort z-scores, capped and shrunk by era completeness, then aggregated using a rank-based formula with a season-count career bonus. Era-fair by construction.

### Creating a Release (weekly fan pack)

The repo ships a small shell script that bundles the weekly fan pack into a single zip ready to attach to a GitHub Release. From the repo root:

```bash
./scripts/package_fan_pack.sh                    # default tag: weekly-YYYY-MM-DD
./scripts/package_fan_pack.sh weekly-2026-05-10  # custom tag
```

The script picks up the most recent prediction CSV, copies the fan-friendly docs (cheat sheet, predictions, backtest, glossary, how-to, Google Sheets template), writes a small README into the bundle, and zips it as `fan-pack-<tag>.zip` in the repo root.

To upload as a release with the [`gh` CLI](https://cli.github.com/):

```bash
TAG="weekly-$(date -u +%Y-%m-%d)"
gh release create "$TAG" \
  --title "Weekly fan pack - $TAG" \
  --notes "Latest prediction CSV plus fan-friendly docs." \
  "fan-pack-${TAG}.zip"
```

The same packaging happens automatically on a Sunday-night schedule via [.github/workflows/weekly-fan-pack.yml](.github/workflows/weekly-fan-pack.yml) - the local script is for ad-hoc releases between scheduled runs.

### Technical guides
- [Quick start](docs/quick-start.md) - TL;DR, three commands
- [Installation & first-time setup](docs/installation.md) - For Fans / For Power Users / For Contributors
- [Running predictions & backtests](docs/usage.md) - predict next round, backtest, refresh data
- [Troubleshooting](docs/troubleshooting.md) - common errors and fixes
- [Claude Code setup on Ubuntu](docs/claude-code-setup.md) - install Node.js, Claude Code, Python venv, default model
- [Using the Scientist agent](docs/scientist-agent.md) - when plain Claude vs the Scientist, the improvement loop
- [Using the FootyStrategy agent](docs/coaching-guide.md#leveraging-the-footystrategy-agent) - tactical brainstorming, list analysis by draft pick, Scientist x FootyStrategy workflow
- [How predictions work](docs/prediction-model.md) - the model, the backtest framework, the all-time-100 algorithm
- [AI system architecture](docs/ai-architecture.md) - RAG, tool router, eval harness, MCP gateway, sovereign deployment
- [How this repo uses Claude](docs/how-this-repo-uses-claude.md) - custom agent design, policy-as-code, feedback governance framework, multi-agent orchestration
- [Building The Crumb - a footy AI that runs the club better than the coach](docs/footy-ai-chatbot-setup.md) - 13-agent Claude staff (senior coach, line coaches, specialists, analysts, methodology, performance, list, data steward) on the SuperCoach-VIA dataset, end-to-end build guide
- [Technical reference](docs/technical-reference.md) - GPU setup, data layout, scripts
- [Model report card](docs/model-report-card.md) - hit/miss methodology and weekly accuracy log

## What's coming - Phase 2

The project works well for one operator on one machine. Phase 2 makes it work for anyone, on any machine, with visible failures when something breaks.

Work is underway on the `feature/phase2` branch. The plan in order of priority:

| Rung | What | Why it matters |
|------|------|----------------|
| 1 ✅ | `pyproject.toml`, pinned deps, `src/supercoach/` package layout, GitHub Actions CI, first tests | Anyone can clone and reproduce the environment exactly |
| 2 | Ruff linting, mypy type checking, pre-commit hooks, expanded test suite | Catch bugs before they reach the data |
| 3 | Replace `refresh_and_rank.sh` local cron with a GitHub Actions scheduled pipeline | No more hardcoded local paths - failures are visible, not silent |
| 4 | Pandera schema contracts at every data boundary | When AFL Tables renames a column, the pipeline fails loudly on row one rather than silently corrupting a feature |
| 5 | DVC for large data files, MLflow for model experiment tracking | Clean git history; traceable model versions |
| 6 | Scientist agent hardening - tool allowlist, PR-only mode, golden-task eval harness | Safe to give the agent more autonomy |
| 7 | Optional Streamlit/FastAPI frontend | Predictions browsable in a web UI without touching a CSV |

The modernisation is sequenced so each rung unblocks the next. Rung 1 is complete; Rungs 2 and 3 are the next weekend of work.

## Why this repo exists

This is not a commercial project. It is not affiliated with any gambling service, and nothing here is intended to encourage betting of any kind. The motivation is the game itself - the patterns inside it, the history it carries, and the people it brings together.

It started, honestly, as competitive edge. I have been playing SuperCoach with the same group for over a decade, and this repo exists in no small part because of the arguments about who the better player really was. Somewhere along the line a Sunday-night lineup tweak turned into feature engineering, then into a backtest framework, then into this.

But this repo is also a return gift. To the friends and colleagues who got me up to speed on this game - who explained what a clearance was, why ruck craft matters, how to read a scoreline - and who introduced me to SuperCoach in the first place. You did not have to, and you did. This is, in part, a thank you back.

A specific and heartfelt thank you goes to the families, coaches and community of Cranbourne Junior Football Club, who welcomed my son and trained him in the right spirit of the game. The coaches who give their time freely on cold mornings, the families who stand on the boundary in the rain - these are the people who actually make the game what it is. AFL doesn't exist without them, and a polished dataset of senior careers means very little without remembering where every one of those players came from.

It is also why I think AFL is one of the things that can make Australia genuinely multicultural. Sport breaks boundaries in a way that policy never quite manages to. A new Australian turning up at a junior football club and being welcomed onto a team is not a small thing - it is one of the more honest forms of belonging this country has to offer.

And finally, this work is an homage to the giants of the game - past, present and future. To the players whose careers are quietly recorded in the rows of this dataset, who gave everything on the field, and who made generations of fans care deeply about something together. The numbers in here are theirs. The rest of us are just keeping the ledger.
