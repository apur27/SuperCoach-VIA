# AFL SuperCoach VIA
![AFL SuperCoach VIA Banner](/assets/banner.svg)
<div align="center">
  <img src="https://img.shields.io/github/last-commit/apur27/SuperCoach-VIA">
  <img src="https://img.shields.io/github/contributors/apur27/SuperCoach-VIA">
  <img src="https://img.shields.io/github/stars/apur27/SuperCoach-VIA?style=flat-square">
  <img src="https://img.shields.io/github/forks/apur27/SuperCoach-VIA?style=flat-square">
</div>

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Data](https://img.shields.io/badge/data-2026%20season%20round%2010-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## What this is

**For the footy fan:** 130 years of AFL data plus a set of AI agents that reason like a coaching staff - weekly player predictions, team trends, all-time rankings, and debate-ready insight, no coding required.

**For the ML engineer:** a production-architecture reference in one legible repo - a feature pipeline with strict temporal cutoffs, a three-model ensemble, multi-agent LLM reasoning, deterministic RAG over structured data, a walk-forward eval harness, and an MCP tool gateway. Small enough to read end to end, complete enough to map onto a real deployment.

The repo runs a full weekly-refreshed pipeline from a single shell script: scrape new match and player data, retrain the disposal model, run a leak-proof walk-forward backtest, regenerate the all-time top-100, and update the documentation. The football is the domain; the architecture is the point.

The current model misses a player's next-round disposal count by about **[data]** 4.09 disposals on average, measured honestly across **[data]** 3,701 player-round predictions (Rounds 1-10, 2026).

⭐ **If this project is useful to you, please star the repo.**

---

## AFL News & Analysis

Long-form footy journalism where the numbers are not decoration - they are the argument. Every piece is co-authored by two agents in this repo: **Scientist** pulls verified stats from 130 years of match data (every claim reproducible from the CSVs), and **FootyStrategy** turns them into coach-grade tactical reads. No hot takes, no recycled commentary.

**Latest:** [Michael Voss steps down from Carlton](docs/news/2026-05-13-voss-carlton.md) - what went wrong structurally at the Blues, why it doesn't diminish the legend of a Brisbane three-peat captain, the Melbourne 2021 rebuild parallel, and the path forward. *(13 May 2026)*

→ [All news entries](docs/news/README.md)

---

## What's in this repo - for the fan

Plain English, with a football analogy for each piece.

### The data
130 years of AFL history, structured. Every match since 1897, and **[data]** 13,321-plus individual player files - one CSV per player, a row for every game they ever played. A scraper refreshes it weekly so the numbers stay current. Think of the club's archivist who has kept a card for every player in every game since 1897 - every kick, mark, and goal, filed and cross-referenced. Each week after the round finishes, a runner goes out, collects the latest match sheets, and adds them to the cabinet before the analysts come in Monday morning. The whole system is useless if the cabinet is out of date or has gaps, so keeping it complete and current is the unglamorous job everything else depends on.

### The prediction model
Three different prediction models look at a player's recent form, who they're playing, where, and under what conditions - then they vote on how many disposals that player will get next round. Averaging three models is steadier than trusting any one. Across the 2026 season so far it has been within 5 disposals **[data]** 72% of the time and within 10 **[data]** 96% of the time. Instead of trusting one analyst's gut, you put three experienced analysts in the room - each with a different way of reading the game - and ask them all to call the result independently. When all three lean the same way, you can be confident. When they split, that disagreement is itself a useful signal that the match is genuinely hard to call. You take their combined verdict, not any single one's.

### The Scientist agent
An AI that does the honest statistical work. It reads the actual data, writes and runs its own analysis code, checks for the traps that make sports stats lie (a player appearing in both the training and the test set, using future games to "predict" the past), and refuses to oversell a result. If the answer is "we can't tell from this data," it says so. This is the stats analyst in the coaches' box who doesn't just answer a question - they go and do the work. Ask "does this player drop off in wet weather?" and they pull the data, run the numbers, draw the chart, then write up what they found with the honest caveats attached. They think, act, look at the result, and think again - and they file the finished report so the coaches can read it.

### The FootyStrategy agent
An AI that thinks like a coaching panel. Where the Scientist tells you what the numbers say, FootyStrategy tells you what to *do* about it in football terms - and tags every recommendation with how confident it actually is. Picture a panel of eight assistant coaches, each obsessed with one thing - fitness, structure, match-ups, list management, and so on - all reviewing the same question at once. They hand back a single recommendation, but they're upfront about how sure they are and where they disagreed. And every call comes with a "tripwire": the specific thing you'd see on the ground that means the plan is wrong and you change it.

### The Crumb
A 13-agent AI coaching staff - a senior coach, line coaches, specialists, analysts, a data steward - that you ask one question and it dispatches the right specialists and merges their answers. Named after the crumber: the small forward who reads where the ball will spill before the pack resolves. This is the entire football department drawn as an org chart. At the top sits the Senior Coach, who doesn't crunch numbers personally - they take the big question, hand the right pieces to the right specialists below them, and pull the answers back together into one plan. Thirteen roles in all, from the boss down to the data steward who keeps the filing cabinet clean. Everyone has a lane.

### The weekly fan pack
A Round 11 cheat sheet and prediction bundle, packaged every Sunday night - the kind of thing you'd actually open before locking in a SuperCoach lineup. [ANALOGY: the weekly fan pack]

### The news section
Data-grounded footy journalism. Every number in every article is reproducible from the CSVs in this repo - no remembered stats, no quoted-without-source figures. This is the Monday press conference, done properly. The stats analyst (the Scientist) brings what the numbers honestly say about the weekend's footy; the strategy panel (FootyStrategy) translates that into what it actually means for how teams should play. Together they produce the current-affairs piece - grounded in real data, framed in real football language.

---

## What's in this repo - for the engineer

Each layer below is small on purpose. The interest is that all of them are present at once.

### Data layer
130 years of AFL match and player CSVs. **[data]** 13,321-plus player performance files, one row per player per game, 1897-present, plus per-season match files. Weekly scrape via `refresh_data.py`. Feature engineering builds rolling-window features per player (3-game, 5-game, season-to-date form), opponent strength, venue effects, and contextual flags (home/away, day/night, season stage). The `LeakProofPredictor` enforces a strict temporal cutoff: when predicting round N, only data strictly before round N is visible during feature construction and fitting.

### ML inference
A `VotingRegressor` ensemble of three diverse base learners: `HistGradientBoostingRegressor`, `LightGBM` (GPU-capable, CPU fallback), and `RandomForestRegressor`. Hyperparameters tuned via Optuna's TPE sampler over a 50-trial budget. A post-hoc out-of-fold linear calibration step corrects top-end compression. Walk-forward backtest: **[data]** MAE 4.086 across 3,701 player-rounds (R1-R10, 2026). Cross-validation is `GroupKFold` keyed on player ID, so no player appears in both train and validation folds.

### LLM reasoning - Scientist
Claude (`claude-sonnet-4-6` / Opus) running a ReAct loop - Reason, Act, Observe, repeat - for 50-plus turns on complex tasks. Tool surface: Bash, Read/Write/Edit, WebFetch, Agent subagents. `CLAUDE.md` is the versioned system prompt and policy doc: data-coverage caveats, ranking constants, behavioural constraints, all in source control and diffable.

### LLM reasoning - FootyStrategy
An 8-lens tactical council: Conditioner, Tempo Architect, Structuralist, Match-up Tactician, Talent Developer, Innovator, Culture Custodian, List Strategist. Each lens is produced separately, then reconciled. Output is tiered - Settled, Probationary, Contested, Insufficient Evidence - and every Settled or Probationary recommendation must carry a **tripwire**: an explicit observable that would overturn it. Caveats from the Scientist's upstream findings propagate through unchanged; the data tier caps the recommendation tier.

### The Crumb (Phase 1)
A 13-agent, 6-tier hierarchy. `claude-opus-4-7` orchestrator (Senior Coach), `claude-sonnet-4-6` specialists (line coaches, analysts), `claude-haiku-4-5` data steward. Invoked through the Claude Code `@"Agent (agent)"` pattern. Phase 1 uses prompt-based scoping and model-driven tool calls.

### RAG layer
Deterministic retrieval - pandas filters over CSVs. No embedding model, no vector store for structured numeric data, because semantic similarity adds noise where the query maps directly to a structured filter. Hybrid upgrade path is documented: pgvector or Qdrant for unstructured commentary, the existing pandas layer staying authoritative for any numeric claim.

### Eval harness
Walk-forward backtest with strict temporal cutoff. Per-round MAE, RMSE, within-5, within-10, signed bias, and a top-10-player MAE slice that surfaces the worst failure mode. Team-level bias across all 18 teams. Top-error analysis per round. `backtest.py --start-round N --end-round N` for incremental runs; output persisted as CSV under `data/prediction/backtest/`.

### MCP gateway
Claude Code's built-in MCP implementation. Tool surface: Bash, Read/Write/Edit, WebFetch/WebSearch, Agent subagents. JSON-schema tool definitions; tool selection is model-driven, no hand-coded routing logic.

### Observability
`git log` as the audit trail - every doc change is an attributable commit with author, timestamp, diff, message. Backtest CSVs are the ML performance history; a regression is visible by diffing two runs. Chart filenames are timestamped. `CLAUDE.md` is version-controlled, so the agent's policy state at any past commit is reconstructable.

---

## The Crumb - Phase 2: Production-Grade Multi-Agent Architecture

This is the major piece of new design work. Full spec: **[docs/footy-ai-chatbot-phase2.md](docs/footy-ai-chatbot-phase2.md)**.

### What's changing and why

**Plain English:** Phase 1 of The Crumb works - you ask the Senior Coach a question, it calls the right specialists, it merges their answers. But it works the way a well-run training session works: everyone knows their job because they were *told* their job, and they cooperate because they choose to. Phase 2 makes the structure load-bearing instead of cooperative - the rules are enforced by the system, not by good behaviour.

**Technical:** Phase 1 scopes agents through their system prompts and lets each agent emit arbitrary tool calls (including running arbitrary Python). That is flexible and cheap to evolve, but it gives no hard guarantee that an agent stays in its lane, no schema contract on what agents hand each other, and no automated check that the system as a whole still works after a change. Phase 2 applies three patterns drawn from real multi-agent deployments to close those gaps - without losing the legible, markdown-in-repo character of the project.

### The three patterns driving Phase 2

These are patterns João Moura (CrewAI) has described from production multi-agent work. Each maps cleanly onto a football idea.

1. **Planner-Executor split.** One component decides *what* to do and writes a plan; separate components *execute* the plan steps. The planner never touches data; the executors never decide strategy. Footy analogy: the match committee picks the plan on Thursday; the players run it on Saturday. You don't want the bloke running the plan also rewriting it mid-quarter.

2. **Role-based crew with IAM isolation.** Each agent gets only the access it needs for its role - the credentials, the tools, the data paths - and nothing else, enforced at the infrastructure level rather than by instruction. Footy analogy: the goal umpire can signal a goal; they cannot also bounce the ball. The role *is* the permission set.

3. **Supervisor-worker graph.** The orchestration is an explicit graph - declared nodes, declared edges, durable state - so a crashed step can be replayed from its last good state instead of re-running the whole workflow. Footy analogy: a structured game plan with named phases and triggers, not a verbal "see how we go" - when something breaks down you reset to the last set position, you don't restart the game.

### The five Phase 2 design changes

1. **Planner-Executor split of the Senior Coach Agent.** The current Senior Coach both decides which specialists to call *and* synthesises their answers. Phase 2 splits this: a **Planner** node reads the user question and emits a structured plan (which agents, what each is asked, what the dependency order is); **Executor** nodes run the plan; a separate **Synthesis** node merges results. The Synthesis node has no data access at all - it can only combine what the executors verified.

2. **Parameterised query templates.** Today the Data Steward runs arbitrary pandas/Python. Phase 2 replaces arbitrary code execution with a fixed set of parameterised query templates - `player_form(player_id, n_games)`, `team_round_results(team, year, round)`, and so on. The agent fills parameters; it cannot author new code paths. This kills an entire class of failure (and attack surface) at once.

3. **`FootyFinding` Pydantic envelope.** Every agent-to-agent handoff is wrapped in a single validated schema - the finding itself, its data citations, its confidence tier, its caveats, its era-coverage flags. A handoff that does not validate is rejected at the boundary, not discovered three steps later in a wrong answer.

4. **HITL routing on confidence threshold.** When the synthesised answer's confidence falls below a configured floor - genuine lens disagreement, thin data, an era-coverage refusal - the answer is routed to a human review queue instead of being returned directly. High-confidence answers flow through; the uncertain ones get a human in the loop.

5. **Evaluation harness for the agent layer.** A nightly regression suite across five test categories: **citation precision** (does every number trace to a real file?), **era-coverage refusal** (does the system correctly refuse a pre-1987 tackle query?), **role isolation** (does the Planner stay out of the data?), **calibration** (do "Settled" answers actually hold up more often than "Contested" ones?), and **falsifiability** (does every recommendation carry a real tripwire?).

### Data isolation without GCP IAM

João's pattern assumes cloud IAM - role-scoped credentials issued by the platform. This repo runs locally, so Phase 2 adapts the *principle* without the cloud machinery:

- The data layer is mounted **read-only** to the agent process - the OS enforces it, not a prompt.
- Data access goes only through the **parameterised query templates** - there is no general code-execution tool for an agent to misuse.
- The **Synthesis node has no data access at all** - it physically cannot fabricate a number, only combine verified findings.

Three cheap, local mechanisms that together give the same guarantee as cloud IAM: an agent can only touch what its role allows.

### Build order

| Priority | Change | Effort |
|----------|--------|--------|
| 1 | Parameterised query templates (kill arbitrary code execution) | Low |
| 2 | `FootyFinding` Pydantic envelope for all handoffs | Low-Medium |
| 3 | Planner-Executor split of the Senior Coach | Medium |
| 4 | HITL routing on confidence threshold | Medium |
| 5 | Evaluation harness (5 test categories, nightly) | Medium-High |

Sequenced so each rung unblocks the next: templates and the envelope are the contracts the split depends on; the split is what makes HITL routing and the eval harness meaningful.

### What this proves beyond football

The football is incidental. The same five patterns - a Planner-Executor split, role-scoped data isolation, schema-validated handoffs, HITL routing on a confidence threshold, and a nightly eval regression - apply to any multi-agent deployment: a customer-support crew, a research assistant, a code-review pipeline. SuperCoach-VIA is small enough to read in an afternoon and complete enough to be the reference implementation for all five. That is the actual deliverable; the disposal predictions are the demo.

---

## How to run it - quick start

```bash
git clone https://github.com/apur27/SuperCoach-VIA
cd SuperCoach-VIA
# Install venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Refresh data + predictions
bash refresh_and_rank.sh
# Run backtest (latest round only - incremental)
python backtest.py --start-year 2026 --start-round 10 --end-year 2026 --end-round 10
# Generate weekly fan pack
python scripts/generate_weekly_cheat_sheet.py --year 2026
```

Full setup (GPU notes, data layout, first-time troubleshooting) is in [docs/installation.md](docs/installation.md).

---

## Eval results - current

Walk-forward backtest, 2026 season, Rounds 1-10. For each round the model is retrained using only data from before that round, predicts every player who played, and is scored against actuals. All figures reproducible from `data/prediction/backtest/backtest_summary_20260511_191837.csv`.

| Round | Player-rounds | MAE | RMSE | Within 5 | Within 10 |
|-------|--------------:|----:|-----:|---------:|----------:|
| 1  | **[data]** 230 | **[data]** 4.83 | **[data]** 6.10 | **[data]** 60.4% | **[data]** 92.6% |
| 2  | **[data]** 413 | **[data]** 4.11 | **[data]** 5.11 | **[data]** 72.2% | **[data]** 95.9% |
| 3  | **[data]** 320 | **[data]** 4.07 | **[data]** 5.28 | **[data]** 74.7% | **[data]** 95.9% |
| 4  | **[data]** 319 | **[data]** 4.15 | **[data]** 5.32 | **[data]** 72.4% | **[data]** 94.7% |
| 5  | **[data]** 365 | **[data]** 3.73 | **[data]** 4.74 | **[data]** 75.3% | **[data]** 97.5% |
| 6  | **[data]** 411 | **[data]** 3.98 | **[data]** 5.05 | **[data]** 74.9% | **[data]** 95.9% |
| 7  | **[data]** 410 | **[data]** 4.05 | **[data]** 5.15 | **[data]** 72.0% | **[data]** 95.6% |
| 8  | **[data]** 411 | **[data]** 4.14 | **[data]** 5.27 | **[data]** 73.2% | **[data]** 95.4% |
| 9  | **[data]** 410 | **[data]** 3.79 | **[data]** 4.74 | **[data]** 74.9% | **[data]** 98.3% |
| 10 | **[data]** 412 | **[data]** 4.31 | **[data]** 5.50 | **[data]** 68.2% | **[data]** 94.9% |
| **All** | **[data]** 3,701 | **[data]** 4.09 | **[data]** 5.18 | **[data]** 72.3% | **[data]** 95.8% |

**Plain English:** the typical prediction misses by about four disposals. On a per-player range of roughly 0-45, that is usable signal, not a solved problem. Round 1 is the hardest (**[data]** 4.83 MAE) because there are no within-season form features before any 2026 game has been played; Round 5 is the best so far (**[data]** 3.73 MAE, **[data]** 97.5% within 10).

**Technical:** weighted across 3,701 player-rounds the overall signed bias is **[data]** -0.09 disposals - the model is very close to unbiased in aggregate. The known failure mode is the elite tier: top-10-player MAE runs roughly 2.5x the global figure, driven by a residual ceiling effect and context (tag absorption, role rotations) the feature set captures only partially.

### Team-level bias

Signed bias by team, weighted across all 10 rounds (negative = model over-predicts, positive = model under-predicts). Reproducible from `data/prediction/backtest/backtest_by_team_20260511_191837.csv`.

| Direction | Team | Mean signed bias |
|-----------|------|-----------------:|
| Most over-predicted | Sydney | **[data]** -0.59 |
| | Collingwood | **[data]** -0.45 |
| | Hawthorn | **[data]** -0.45 |
| Most under-predicted | Richmond | **[data]** +0.53 |
| | West Coast | **[data]** +0.36 |
| | Brisbane Lions | **[data]** +0.36 |

Mean absolute team bias is **[data]** 0.25 disposals - no team is badly mis-calibrated, but the spread is real and is a calibration target for the next model iteration.

Full pre-registered methodology, per-round notable misses, and the weekly accuracy log: [docs/afl-backtest-2026.md](docs/afl-backtest-2026.md).

---

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
| **See how accurate the model has been (backtest + pre-registered report card)** | [docs/afl-backtest-2026.md](docs/afl-backtest-2026.md) | None - browser only |
| **Run predictions or retrain the model myself** | [docs/installation.md](docs/installation.md) (For Contributors section) | Python, Git, terminal |
| **Get tactical analysis on an AFL team's list and draft picks** | [docs/coaches-strategy-corner/afl-2026-team-list-analysis.md](docs/coaches-strategy-corner/afl-2026-team-list-analysis.md) | None - browser only |
| **Read the AFL news desk - data-grounded long-form on current stories** | [docs/news/README.md](docs/news/README.md) | None - browser only |
| **Understand the AI system architecture** | [docs/ai-architecture.md](docs/ai-architecture.md) | None - browser only |
| **Read the Crumb Phase 2 design doc** | [docs/footy-ai-chatbot-phase2.md](docs/footy-ai-chatbot-phase2.md) | None - browser only |

---

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
  - [AFL history - 130 years](docs/afl-history.md)
  - [For the footy expert](docs/footy-expert-guide.md)
  - [For the coaching staff](docs/coaching-guide.md)
  - [AFL 2026 team list analysis](docs/coaches-strategy-corner/afl-2026-team-list-analysis.md) - all 18 clubs
- [AFL Hall of Fame](docs/hall-of-fame.md) - all-time top 100, statistical leaders, captains, coaches, dynasties

### AI architecture & agents
- [AI system architecture](docs/ai-architecture.md) - RAG, tool router, eval harness, MCP gateway, sovereign deployment
- [Building The Crumb (Phase 1)](docs/footy-ai-chatbot-setup.md) - 13-agent Claude staff, end-to-end build guide
- [The Crumb - Phase 2 design doc](docs/footy-ai-chatbot-phase2.md) - Planner-Executor, parameterised tools, FootyFinding envelope, HITL routing, eval harness
- [How this repo uses Claude](docs/how-this-repo-uses-claude.md) - custom agent design, policy-as-code, multi-agent orchestration

### About
- [Roadmap & contributing](docs/roadmap.md)
- [Changelog](CHANGELOG.md)

---

## Further reading

- [How it works: data science deep-dive](docs/data-science.md) - dataset, model, backtest, ranking algorithm, written in three layers from layperson to ML practitioner
- [AI system architecture](docs/ai-architecture.md) - the full architecture write-up
- [Building The Crumb (Phase 1)](docs/footy-ai-chatbot-setup.md) - the 13-agent staff build guide
- [The Crumb - Phase 2 design doc](docs/footy-ai-chatbot-phase2.md) - the production-grade multi-agent spec
- [How predictions work](docs/prediction-model.md) - the model, the backtest framework, the all-time-100 algorithm
- [Using the Scientist agent](docs/scientist-agent.md) - when plain Claude vs the Scientist, the improvement loop
- [Using the FootyStrategy agent](docs/coaching-guide.md#leveraging-the-footystrategy-agent) - tactical brainstorming, list analysis, Scientist x FootyStrategy workflow
- [Backtest results - 2026](docs/afl-backtest-2026.md) - pre-registered methodology, per-round MAE/RMSE, team-level bias, notable misses
- [Quick start](docs/quick-start.md) / [Installation](docs/installation.md) / [Usage](docs/usage.md) / [Troubleshooting](docs/troubleshooting.md)
- [Claude Code setup on Ubuntu](docs/claude-code-setup.md) - install Node.js, Claude Code, Python venv
- [Technical reference](docs/technical-reference.md) - GPU setup, data layout, scripts

---

## Why this repo exists

This is not a commercial project. It is not affiliated with any gambling service, and nothing here is intended to encourage betting of any kind. The motivation is the game itself - the patterns inside it, the history it carries, and the people it brings together.

It started, honestly, as competitive edge. I have been playing SuperCoach with the same group for over a decade, and this repo exists in no small part because of the arguments about who the better player really was. Somewhere along the line a Sunday-night lineup tweak turned into feature engineering, then into a backtest framework, then into this.

But this repo is also a return gift. To the friends and colleagues who got me up to speed on this game - who explained what a clearance was, why ruck craft matters, how to read a scoreline - and who introduced me to SuperCoach in the first place. You did not have to, and you did. This is, in part, a thank you back.

A specific and heartfelt thank you goes to the families, coaches and community of Cranbourne Junior Football Club, who welcomed my son and trained him in the right spirit of the game. The coaches who give their time freely on cold mornings, the families who stand on the boundary in the rain - these are the people who actually make the game what it is. AFL doesn't exist without them, and a polished dataset of senior careers means very little without remembering where every one of those players came from.

It is also why I think AFL is one of the things that can make Australia genuinely multicultural. Sport breaks boundaries in a way that policy never quite manages to. A new Australian turning up at a junior football club and being welcomed onto a team is not a small thing - it is one of the more honest forms of belonging this country has to offer.

And finally, this work is an homage to the giants of the game - past, present and future. To the players whose careers are quietly recorded in the rows of this dataset, who gave everything on the field, and who made generations of fans care deeply about something together. The numbers in here are theirs. The rest of us are just keeping the ledger.
