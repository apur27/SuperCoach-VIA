# Building The Crumb - a multi-agent footy AI that runs the club better than the coach

> [← Back to main README](../README.md) · [← AI Agents hub](ai-agents.md)

<!-- This file is part of the SuperCoach-VIA documentation. See README.md for the project overview. -->

**Who is this page for?** Engineers who want to build a conversational AFL coaching assistant on top of the SuperCoach-VIA dataset. This is a "how to build the thing" guide - it assumes Python, an Anthropic API key, and familiarity with the [Claude Code agent SDK](https://docs.anthropic.com/) (or the underlying tool-use loop).

If you only want to *use* the existing Scientist or FootyStrategy agents, read [scientist-agent.md](scientist-agent.md) instead.

---

## 1. Overview

**The Crumb** is the AI that challenges the coach's intuition. Named after the AFL crumber - the small forward who reads where the ball will spill from a pack before the contest resolves. The Crumb sees the pattern before the coach has called it: pure pattern recognition, one beat ahead of everyone else. Where the coach trusts his gut on a selection or a matchup, The Crumb puts the data on the table and asks the question the box was about to lose to vibes.

Under the hood, The Crumb is an **AI coaching staff simulation** built on top of the SuperCoach-VIA dataset. Instead of a single monolithic chatbot, the system is structured the way a real AFL coaching staff is structured: a senior coach who runs the meeting, line coaches who own a part of the ground, specialists who own a phase of play, analysts who own the evidence, and a data steward who owns the source of truth.

A user asks a question in plain English - "*Who should I pick in my midfield for Round 11?*" - and the **Senior Coach Agent** decides which agents in the staff to call, dispatches the work in parallel, then merges their answers into a single response. When the answer disagrees with conventional wisdom, The Crumb says so plainly and shows the data behind the disagreement. Honest answers over impressive answers, even when "honest" means "the coach is wrong on this one".

Every agent is grounded in the same source of truth: the CSV files inside this repo. Agents read from the data layer, never write to it. There is no separate vector store, no fine-tune - the dataset is the context. **[data]** The repo ships:

- ~14,500 per-player performance CSVs in `data/player_data/`
- Match results 1897–present in `data/matches/`
- Current-season team lineups in `data/lineups/`
- Walk-forward backtest output and weekly predictions in `data/prediction/`

The pattern is the same one the existing [Scientist](scientist-agent.md) and [FootyStrategy](ai-agents.md#footystrategy-agent) agents use - natural language as a thin wrapper over structured data - extended into a multi-agent hierarchy.

---

## 2. Architecture

### Agent hierarchy

The Crumb is a 13-agent staff arranged in six tiers. The Senior Coach at the top is the only agent the user talks to; everything below is delegated work. The tiers mirror how a real AFL football department is built - line coaches own zones, specialists own phases, analysts own evidence, performance and list-management own the off-field decisions, and a data steward owns the source of truth.

```
                                  ┌──────────────────────────┐
                       user ───▶  │  Senior Coach Agent      │  Tier 1  (claude-opus-4-7)
                                  │  - routes queries        │
                                  │  - challenges intuition  │
                                  │  - synthesises answer    │
                                  └────────────┬─────────────┘
                                               │
                ┌──────────────────────────────┼──────────────────────────────┐
                ▼                              ▼                              ▼
        ┌───────────────┐              ┌───────────────┐              ┌───────────────┐
        │ Midfield      │              │ Forward Line  │              │ Back Line     │   Tier 2
        │ Coach         │              │ Coach         │              │ Coach         │
        └───────┬───────┘              └───────────────┘              └───────────────┘
                │
        ┌───────┴───────────────────────────────────┐
        ▼                                           ▼
┌────────────────────┐                  ┌────────────────────┐                          Tier 3
│ Stoppage           │                  │ Defensive Press    │
│ Specialist         │                  │ Specialist         │
└────────────────────┘                  └────────────────────┘

┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  ┌──────────────────────┐  Tier 4
│ Match        │  │ Opposition   │  │ Stats/Methodology    │  │ Strategy Council     │
│ Analyst      │  │ Analyst      │  │ ("The Scientist")    │  │ ("FootyStrategy")    │
└──────────────┘  └──────────────┘  └──────────────────────┘  └──────────────────────┘

       ┌─────────────────────────────┐       ┌─────────────────────────────┐           Tier 5
       │ High Performance Agent      │       │ List Manager Agent          │
       └─────────────────────────────┘       └─────────────────────────────┘

                            ┌────────────────────────────────────┐                     Tier 6
                            │ Data Steward Agent                 │
                            │ - owns the read path to the CSVs   │
                            └─────────────────┬──────────────────┘
                                              ▼
                                  ┌──────────────────────────┐
                                  │   Data layer (read-only) │
                                  ├──────────────────────────┤
                                  │  data/player_data/*.csv  │
                                  │  data/matches/*.csv      │
                                  │  data/lineups/*.csv      │
                                  │  data/prediction/*.csv   │
                                  └──────────────────────────┘
```

### Roles at a glance

| Tier | Agent | Model | Owns | Reads from |
|---|---|---|---|---|
| 1 | **Senior Coach** | claude-opus-4-7 | routing, synthesis, the "challenge the coach" framing, final answer | nothing directly - delegates |
| 2 | **Midfield Coach** | claude-sonnet-4-6 | disposal chains, clearances, contested possessions, centre bounces, inside-50 generation | via Data Steward |
| 2 | **Forward Line Coach** | claude-sonnet-4-6 | scoring shots, goal-kicking accuracy, forward-50 entries, marks inside 50, forward pressure | via Data Steward |
| 2 | **Back Line Coach** | claude-sonnet-4-6 | intercept marks, rebound 50s, one-percenters, spoils, opposition forward output | via Data Steward |
| 3 | **Stoppage Specialist** | claude-sonnet-4-6 | centre bounce setups, around-the-ground stoppage wins, ruck/midfield combos | via Data Steward |
| 3 | **Defensive Press Specialist** | claude-sonnet-4-6 | forward-half turnover pressure, kick-in setups, opposition exit-66% rate | via Data Steward |
| 4 | **Match Analyst** | claude-sonnet-4-6 | head-to-head history, venue records, recent results, fixture context | via Data Steward |
| 4 | **Opposition Analyst** | claude-sonnet-4-6 | next opponent's recent form, tactical tendencies, key personnel, weak slots | via Data Steward |
| 4 | **Stats/Methodology Agent ("The Scientist")** | claude-sonnet-4-6 | model assumptions, leakage checks, holdout integrity, uncertainty quantification, p-hacking refusal | via Data Steward |
| 4 | **Strategy Council Agent ("FootyStrategy")** | claude-sonnet-4-6 | game-plan framing, list strategy, tactical narratives, "is this a real trend or a vibe" calls | via Data Steward |
| 5 | **High Performance Agent** | claude-sonnet-4-6 | load, soreness/injury proxies, games-played cadence, return-from-injury form curves | via Data Steward |
| 5 | **List Manager Agent** | claude-sonnet-4-6 | per-player form, tier rankings, availability, lineups, contract/age structure context | via Data Steward |
| 6 | **Data Steward** | claude-haiku-4-x | the *only* agent that reads files; enforces read-only, schema, era-coverage rules, citation | `data/player_data/`, `data/matches/`, `data/lineups/`, `data/prediction/` |

Two things to notice. First: every agent above tier 6 delegates data reads to the **Data Steward**. That is deliberate. It centralises the "what year does this column start?" rule (tackles from 1987, clearances from 1998, hit-out recording change in 2017), the citation requirement, and the read-only constraint. None of the analyst agents can lie about a number because none of them touch a file.

Second: the **Stats/Methodology Agent** and **Strategy Council Agent** sit alongside the analysts, not above them. When the Senior Coach is about to give an answer that disagrees with conventional coaching wisdom, it consults the Scientist on the methodology and FootyStrategy on the framing before speaking. That is what makes The Crumb willing to say "the coach is wrong on this one" without it being a vibe.

### Data layer (read-only)

| Path | Contents | Approx. rows |
|---|---|---|
| `data/player_data/<surname>_<firstname>_<DDMMYYYY>_performance_details.csv` | One row per player per game, 1897–present | varies per player |
| `data/matches/matches_<year>.csv` | All matches for a season, scores and venues | ~200 per recent season |
| `data/lineups/team_lineups_<club>.csv` | Current-season selections per club | 1 file × 18 clubs |
| `data/prediction/next_round_<N>_prediction_*.csv` | Weekly disposal predictions | ~360 per round |
| `data/prediction/backtest/` | Walk-forward backtest output | per round |

Agents never write to these files. Every write path goes through the existing pipeline scripts (`refresh_data.py`, `prediction.py`, `update_team_analysis.py`).

---

## 3. Technology stack

| Layer | Choice | Reason |
|---|---|---|
| Senior Coach model | `claude-opus-4-7` | Routing, synthesis, and the "is this answer ready to challenge the coach?" judgement benefit from the larger model; one Opus call per user query is affordable |
| Line coach / specialist / analyst models | `claude-sonnet-4-6` | Each agent is doing one bounded thing - Sonnet is the right cost/quality point |
| Data Steward model | `claude-haiku-4-x` (or Sonnet if Haiku is unavailable) | Mechanical work - find file, read columns, return rows. Cheapest tier that still handles tool use cleanly |
| Orchestration | Claude Code agent SDK (`tool_use` + subagent spawning) | Same pattern the existing Scientist and FootyStrategy agents use |
| Data access | Python venv at `/home/abhi/sourceCode/python/coding/.venv` | Pinned pandas/numpy versions; matches the rest of the repo |
| CSV reads | `pandas` | Already used everywhere in the pipeline |
| Interface | CLI (Claude Code) day one; web UI later | Lowest friction - the CLI is already the primary interface for Scientist |

The same Anthropic API key powers every agent in the hierarchy. The Claude Code SDK handles subagent spawning, so the orchestrator does not need to manage HTTP connections by hand - it issues `tool_use` calls and the framework dispatches them.

---

## 4. Implementation guide

### Step 1 - Anthropic API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

For Claude Code specifically, the CLI handles login via the browser the first time you run `claude` - see [claude-code-setup.md](claude-code-setup.md) for the full setup. If you are building a standalone process (not running inside Claude Code), an env var is enough.

### Step 2 - Dependencies

The repo's venv already has everything you need:

```bash
source /home/abhi/sourceCode/python/coding/.venv/bin/activate
pip list | grep -E "anthropic|pandas|numpy"
```

If `anthropic` is missing:

```bash
pip install anthropic
```

### Step 3 - Define each staff agent

Every non-orchestrator agent is a Claude Sonnet subagent with a tight system prompt. The system prompt scopes the agent to one part of the ground (or one phase, or one analytical responsibility). Only the **Data Steward** owns the `query_data` tool; every other agent calls the Data Steward when it needs numbers.

Sketch (simplified):

```python
from anthropic import Anthropic

client = Anthropic()

MIDFIELD_SYSTEM_PROMPT = """You are the Midfield Coach in The Crumb staff.
You answer only midfield questions: disposal chains, clearances, contested ball,
centre bounce attendance, inside-50 generation.

You do not read files yourself. When you need numbers, call the data_steward
tool with a specific request ("disposals for player X over last 4 games"),
and the steward returns rows already cited and era-checked.

Never invent numbers. If the steward says a column does not exist for that
era, say so plainly. Honest answers over impressive answers."""

def call_midfield_coach(question: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        system=MIDFIELD_SYSTEM_PROMPT,
        max_tokens=2048,
        tools=[DATA_STEWARD_TOOL_SCHEMA],
        messages=[{"role": "user", "content": question}],
    )
    return run_tool_loop(response)
```

Repeat the same pattern for the other line coaches, specialists, and analysts - each with its own scoped system prompt. The Stats/Methodology Agent ("The Scientist") and Strategy Council Agent ("FootyStrategy") reuse the system prompts of the existing custom agents already in the repo (see `.claude/agents/`).

### Step 4 - Build the Senior Coach orchestrator

The Senior Coach is a Claude Opus model whose tools are *the staff agents themselves*. From the orchestrator's perspective, calling `call_midfield_coach(...)` is just another `tool_use`.

```python
SENIOR_COACH_SYSTEM_PROMPT = """You are the Senior Coach of The Crumb AI
staff. You receive a user question, decide which staff agents to call,
dispatch their work, and synthesise their answers into a single reply.

Your remit is to challenge the coach's intuition with evidence. When the
data disagrees with conventional coaching wisdom, you say so plainly and
show the working. You do not soften an honest answer to sound polite.

Your staff:
  Line coaches: midfield_coach, forward_line_coach, back_line_coach
  Specialists:  stoppage_specialist, defensive_press_specialist
  Analysts:     match_analyst, opposition_analyst
  Methodology:  stats_methodology ("The Scientist"),
                strategy_council ("FootyStrategy")
  Off-field:    high_performance, list_manager
  Data:         data_steward (only via the agents above)

Rules:
1. Call agents in parallel when their work is independent.
2. Never answer a data question yourself - always delegate.
3. Before giving an answer that contradicts conventional wisdom, consult
   stats_methodology to validate the evidence and strategy_council to
   frame the disagreement.
4. If an agent returns "no data", surface that honestly; do not invent.
5. End every answer with: "Consulted: <list of agents>"."""

SENIOR_COACH_TOOLS = [
    {"name": "midfield_coach",            "description": "Midfield/clearance questions",          "input_schema": {...}},
    {"name": "forward_line_coach",        "description": "Forward/scoring questions",             "input_schema": {...}},
    {"name": "back_line_coach",           "description": "Defensive questions",                   "input_schema": {...}},
    {"name": "stoppage_specialist",       "description": "Centre bounce/stoppage setups",         "input_schema": {...}},
    {"name": "defensive_press_specialist","description": "Forward-half pressure, kick-in setups", "input_schema": {...}},
    {"name": "match_analyst",             "description": "H2H, venue, recent form",               "input_schema": {...}},
    {"name": "opposition_analyst",        "description": "Next opponent profile and weak slots",  "input_schema": {...}},
    {"name": "stats_methodology",         "description": "Methodology, leakage, uncertainty",     "input_schema": {...}},
    {"name": "strategy_council",          "description": "Game-plan framing, trend-vs-vibe",      "input_schema": {...}},
    {"name": "high_performance",          "description": "Load, soreness, return-from-injury",    "input_schema": {...}},
    {"name": "list_manager",              "description": "Player form / availability / tiers",    "input_schema": {...}},
]
```

The Opus model is well-suited to "which agents do I need, what do I ask each, and how do I combine the answers into something that holds up under challenge" - which is where most of the answer quality comes from.

### Step 5 - The Data Steward and its `query_data` tool

The Data Steward is the only agent that actually touches disk. Every analyst above it calls the Data Steward; the Data Steward calls `query_data`. The benefit: schema rules, era-coverage rules, and citation discipline live in *one* system prompt instead of being copy-pasted into eleven.

```python
QUERY_DATA_TOOL_SCHEMA = {
    "name": "query_data",
    "description": "Execute a pandas snippet against the SuperCoach-VIA CSVs. "
                   "Read-only. Returns stdout (truncated to 4000 chars).",
    "input_schema": {
        "type": "object",
        "properties": {
            "python_code": {
                "type": "string",
                "description": "Python code to run. Must use pandas. Read-only."
            }
        },
        "required": ["python_code"]
    }
}

def query_data(python_code: str) -> str:
    import subprocess
    result = subprocess.run(
        ["/home/abhi/sourceCode/python/coding/.venv/bin/python", "-c", python_code],
        capture_output=True, text=True, timeout=30,
        cwd="/home/abhi/git/SuperCoach-VIA",
    )
    return (result.stdout + result.stderr)[:4000]
```

This is the only path from agent to disk. Lock down `subprocess` with a timeout and a working directory; do not pass user input directly into shell commands.

### Step 6 - CLI entry point

```python
# bin/the-crumb
if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) or input("crumb> ")
    answer = run_senior_coach(question)
    print(answer)
```

Inside Claude Code, the same setup can be exposed as a custom agent with `@"The Crumb"` - the existing Scientist and FootyStrategy agents are defined the same way and end up as tier-4 members of the same staff.

---

## 5. Example system prompts

### Senior Coach (orchestrator)

```
You are the Senior Coach of The Crumb AI staff, working on the
SuperCoach-VIA dataset (125+ years of match and player data, weekly-refreshed
predictions). The Crumb's job is to challenge the coach's intuition with
evidence - not to flatter it.

You have eleven staff agents you can call as tools:
  Line coaches: midfield_coach, forward_line_coach, back_line_coach
  Specialists:  stoppage_specialist, defensive_press_specialist
  Analysts:     match_analyst, opposition_analyst
  Methodology:  stats_methodology, strategy_council
  Off-field:    high_performance, list_manager

Your job:
1. Read the user's question. Decide which agents are relevant. Call only
   what you need - don't fan out to all eleven for a midfield-only question.
2. When agents can work independently, call them in parallel.
3. Each agent returns a short data-grounded answer. Your job is to
   synthesise them into a single coherent reply.
4. Before giving an answer that contradicts conventional coaching wisdom,
   consult stats_methodology to validate the evidence is real (not a small-
   sample artefact) and strategy_council to frame the disagreement well.
5. Never invent numbers. If an agent says "no data", say so. Honest answers
   over impressive answers - especially when "honest" means the coach is
   wrong on this one.
6. End every reply with: "Consulted: <list of agents>". This is auditable.

You do not read data files. Always delegate.
```

### Midfield Coach (line coach)

```
You are the Midfield Coach in The Crumb staff.

Your domain:
- Disposal counts and chains
- Clearances (centre, stoppage, total)
- Contested possessions
- Centre bounce attendance
- Inside-50 generation from midfield

You do not read files. When you need numbers, call the data_steward tool
with a specific request - for example:
  "disposals + clearances + contested_possessions for player X over
   last 6 games, returned with file citation"

The data_steward enforces era coverage (tackles from 1987, clearances from
1998, hit-out recording change in 2017) and returns rows already cited.

Rules:
1. Never invent a stat. If the steward says the column does not exist for
   that era, surface that.
2. Quote the steward's file citation in your answer.
3. Keep answers under 200 words unless the user explicitly asks for more.
4. Stay in your lane. If the user asks about forwards, defenders, or list
   management, say "out of scope, ask the senior coach to route this".
```

### Data Steward (tier 6)

```
You are the Data Steward in The Crumb staff. You are the only agent that
reads files.

You receive specific data requests from line coaches, specialists, and
analysts. You answer with rows + the exact file path you read.

You read from (via the query_data tool):
- data/player_data/<surname>_<firstname>_<DDMMYYYY>_performance_details.csv
- data/matches/matches_<year>.csv
- data/lineups/team_lineups_<club>.csv
- data/prediction/next_round_<N>_prediction_*.csv
- data/prediction/backtest/

Hard rules:
1. Read-only. Never write, never modify.
2. Inspect before returning - print shape, dtypes, and head() of what you
   loaded. Catches encoding / dtype surprises early.
3. Era coverage:
     - tackles: from 1987 only
     - clearances, contested_possessions: from 1998 only
     - hit_outs: recording change in 2017, do not compare across that break
   If a request asks for these stats outside the supported era, refuse and
   say why.
4. Every response includes the exact file path(s) read. No exceptions.
5. There is no `position` column. If asked, return the disposal-volume +
   centre-bounce-attendance proxy and flag it as a proxy.
6. If you can't find what was asked, say so plainly. Do not guess.
```

### List Manager (tier 5)

```
You are the List Manager in The Crumb staff.

Your domain:
- Current player form (last 4-6 weeks of disposals, goals, SC score)
- Tier rankings - elite / first-choice / fringe / depth
- Availability - who is in the named lineup vs absent
- Weekly model predictions for the next round

You do not read files. Ask the data_steward for what you need.

Rules:
1. When the user asks "who should I pick", interpret as: who is named in the
   round's lineup AND has favourable recent form AND a strong prediction.
2. Predictions have meaningful error (MAE ~4 disposals - see the model report
   card). Quote the prediction with its uncertainty, not as a point estimate.
3. There is no `position` column. If asked for "midfielders", note you are
   using disposal volume + centre bounce attendance as a proxy.
4. Never invent a name. If a player is not in the lineup files, say so.
5. Quote the steward's file citation in your answer.
```

The other agents (Forward Line Coach, Back Line Coach, Stoppage Specialist, Defensive Press Specialist, Match Analyst, Opposition Analyst, High Performance) follow the same pattern - tight domain, no direct disk access, scope discipline. The Stats/Methodology Agent ("The Scientist") and Strategy Council Agent ("FootyStrategy") reuse the existing `.claude/agents/` system prompts already in this repo - they slot in unchanged.

---

## 6. Example query flow

**User:** *"The coach wants to drop player B from the midfield rotation - should he?"*

This is exactly the kind of question The Crumb is built for: a decision where the coach has a view, and the question is whether the data supports it.

1. **Senior Coach** parses the question. Recognises it needs current form (List Manager + Midfield Coach), opposition matchup (Opposition Analyst), and a methodology check (Stats/Methodology) before challenging the coach's preference. Dispatches all four in parallel.

2. **List Manager** asks the Data Steward for B's last 6 games and lineup status. Returns: "B is named in this week's lineup; last 6 games SC scores trending flat at 95 +/- 12. **[data]** Cited: `data/lineups/team_lineups_<club>.csv`, `data/player_data/B_*.csv`."

3. **Midfield Coach** asks the Data Steward for B's clearances and contested ball over the same window. Returns: "Contested-possession share 47%, stable. Centre bounce attendance up 8% over last 3. Clearances per game 5.2 (career median 5.0). No drop signal in midfield-specific markers."

4. **Opposition Analyst** asks for the upcoming opponent's midfield and stoppage record. Returns: "Opponent ranks 14th for opposition midfielder disposals allowed over last 4 - favourable matchup for an inside mid."

5. **Stats/Methodology** is consulted *because the answer is going to disagree with the coach*. It checks: is a 6-game window enough to call the trend "stable"? Yes - SE on disposal mean ~1.4 over n=6, well inside the noise band. No leakage. Effect size of any "drop" claim would be small. Returns: "Evidence for dropping B is absent at conventional significance. The case to keep him is the stronger one."

6. **Senior Coach** synthesises: "On the data, dropping B is the wrong call this week. His midfield markers are stable (contested share 47%, centre bounces up), the SC volatility you're seeing is within noise (+/-12 on a 95 base, n=6), and the matchup favours him (opponent 14th vs inside mids). The Scientist confirms a 6-game window is enough to rule out a real drop in performance. **Recommendation: hold.** If the coach's concern is a specific behaviour rather than output, name it and I will pull the corresponding markers."

7. **Final answer to user** ends with: `Consulted: list_manager, midfield_coach, opposition_analyst, stats_methodology, data_steward`.

The whole loop is one Opus turn and four parallel Sonnet turns (plus the Data Steward calls inside each). Total latency ~8-12 seconds on a healthy API; total cost roughly 1 Opus + 4 Sonnet + a handful of Haiku completions per user query. The structural cost of the methodology check is one extra Sonnet call - cheap insurance against confidently telling the coach he is wrong when he isn't.

---

## 7. Extending the system

The hierarchy is intentionally easy to extend. New staff agents slot in alongside the existing tier 4 or 5 layer; the Data Steward absorbs the new read paths in one place.

### Adding a Brownlow Predictor Agent

1. Write a system prompt scoped to umpire votes, the existing `data/afl-brownlow-2026.md` outputs, and the proxy weights used in the model. (The proxy uses disposals - clangers, goals weighted at 15% - see `brownlow_proxy_weights.md` in agent memory.)
2. Wire it to the Data Steward, like every other analyst.
3. Add it as a new tool on the Senior Coach: `brownlow_predictor`.
4. Update the Senior Coach system prompt: route Brownlow-related questions to this agent.

That is the whole change. The data layer does not need to move.

### Adding a Set Shot Specialist

1. System prompt scoped to goal-kicking accuracy: distance, angle, fatigue, set-shot conversion vs general play.
2. Goes in at tier 3 (specialist), under the Forward Line Coach.
3. Register as `set_shot_specialist` on the Senior Coach for direct routing on goal-kicking questions.

The pattern is: one new system prompt + one new tool registration on the orchestrator (and optionally on the relevant line coach). The Data Steward picks up any new file paths in one place.

---

## 8. Limitations and honest caveats

What this system does **not** have:

- **No GPS or tracking data.** Player movement, heat maps, ground-coverage metrics are out of scope. The dataset is event-level (a disposal, a goal, a clearance) - not spatial.
- **No video.** The agents cannot watch a game. They can describe stat patterns; they cannot describe vision.
- **No real-time injury feed.** Availability is inferred from the named lineup files in `data/lineups/`, which are refreshed weekly. A late withdrawal is not in the data until the next refresh.
- **No position labels.** There is no `position` column in the player CSVs. "Midfielder" / "forward" / "defender" are proxies (disposal volume, goal frequency, intercept counts) - not source-of-truth position assignments.
- **Prediction uncertainty is real.** The disposal model has a meaningful error bar. **[data]** The 2026 walk-forward backtest reports Round 1 MAE of ~4.83 disposals improving to Round 5 MAE of ~3.73 - meaning a "predicted 28 disposals" answer is realistically a 24-32 range, not a point estimate. The Senior Coach is prompted to surface this, not hide it.
- **Pre-1987 / pre-1998 stats are incomplete.** Tackles are recorded from 1987. Clearances and contested possessions from 1998. Hit-outs have a recording change in 2017 that is not a real shift. The Data Steward refuses historical computations on columns that did not exist.
- **No causal claims.** Agents report patterns and predictions. They do not say "trade in player X *because* he will get more game time" unless the lineup file actually shows that - and even then the language is "the lineup indicates", not "this will happen".
- **"Challenge the coach" is not "override the coach".** The Crumb presents evidence and disagreement; it does not make the selection. The coach is still the one who walks the team out on Saturday. The system is designed to lose arguments cleanly when the data does not support its position.

If any of these matter for your use case - add an explicit data source, do not paper over the gap with a confident-sounding answer.

---

## Related

- [Using Claude and the Scientist agent](scientist-agent.md) - the existing single-agent pattern this system extends
- [AI agents hub](ai-agents.md) - Scientist and FootyStrategy, the two custom agents already in the repo
- [How this repo uses Claude](how-this-repo-uses-claude.md) - policy-as-code, multi-agent orchestration, feedback governance
- [AI system architecture](ai-architecture.md) - longer-form architecture write-up (RAG, tool router, eval harness)
- [Model report card](model-report-card.md) - prediction accuracy by round, what the MAE actually means in practice
