# SuperCoach-VIA - AI System Architecture

> [← Back to main README](../README.md)
>
> **Scope:** This is the **AI-system reference architecture** write-up. It walks each layer of the AI stack (feature pipeline, ML inference, LLM reasoning, RAG, tool router / MCP, eval harness, observability, multi-agent hierarchy), maps the project to Australia's AI Ethics Principles, and describes the production gaps and the sovereign-deployment design. Read this when you want to understand the architecture as a reference design or audit it from an ML-engineering / AI-ethics perspective.
>
> **Companion doc:** For the **repository map and operations manual** — full data inventory, code inventory, docs inventory, match lifecycle, live-pipeline internals, prediction-model implementation, ops runbooks, the incident log, and the agent guardrails — see [`ARCHITECTURE.md`](ARCHITECTURE.md). That doc tells you how the repo *runs*; this one tells you how the system is *designed*.

> How a weekend football project maps to production-grade AI system design

SuperCoach-VIA is a working AI system that ingests 130 years of Australian Football League data, trains an ensemble disposal-prediction model, and lets a Claude-powered "Scientist" agent reason over the dataset, write its own analysis code, generate charts, and publish updated documentation back into the repo. The interesting part is not the football - it is that the system contains, in miniature, every layer you would expect in a production AI deployment: a feature pipeline, an ML inference layer with temporal validation, an LLM reasoning loop, a tool gateway, a deterministic RAG layer over structured data, an offline eval harness, and a lightweight audit trail. This document walks each layer, explains it in plain English, then says what an ML engineer would want to add for production. The aim is to use a small, legible system as a reference point for the bigger ones - and to be honest about the gaps a weekend project leaves open.

---

## Architecture overview

```mermaid
flowchart TB
    subgraph Sources["Data sources"]
        A1[AFL match results<br/>web scrape]
        A2[Player performance<br/>13,321 CSV files]
    end

    subgraph Pipeline["Feature engineering pipeline"]
        B1[Rolling-window features<br/>form / opponent / venue / context]
        B2[Target: disposals<br/>log1p transform until 2026]
    end

    subgraph Store["Feature store"]
        C1[(CSV files +<br/>pandas DataFrames)]
    end

    subgraph ML["ML inference layer"]
        D1[HGB]
        D2[LightGBM-GPU]
        D3[RandomForest]
        D4[VotingRegressor<br/>Optuna 50-trial TPE<br/>OOF linear calibration]
        D1 --> D4
        D2 --> D4
        D3 --> D4
    end

    subgraph LLM["LLM reasoning layer — six-agent council"]
        E1[Scientist<br/>Opus, ReAct loop]
        E2[CLAUDE.md<br/>system prompt /<br/>policy guardrails]
        E3[FootyStrategy<br/>Opus, 8-lens council]
        E4[BriefBuilder<br/>Sonnet, brief skeleton]
        E5[DataSentinel<br/>Haiku, data gate]
        E6[Skeptic<br/>Opus, adversarial review]
        E7[Codex<br/>External, blind read]
        E2 -.governs.-> E1
        E4 --> E1
        E1 --> E3
        E3 --> E5
        E3 --> E6
        E1 --> E7
    end

    subgraph Tools["Tool router - MCP gateway"]
        F1[Bash]
        F2[Read / Write / Edit]
        F3[WebFetch / WebSearch]
        F4[Agent subagents]
    end

    subgraph RAG["RAG layer"]
        G1[Deterministic retrieval<br/>pandas queries over CSV]
    end

    subgraph Eval["Eval harness"]
        H1[Walk-forward backtest<br/>strict temporal cutoff<br/>MAE / RMSE / within-N]
    end

    subgraph Output["Output layer"]
        I1[Markdown docs]
        I2[Charts PNG]
        I3[Git commits + push]
    end

    subgraph Obs["Observability"]
        J1[git log<br/>backtest CSVs<br/>chart timestamps]
    end

    A1 --> B1
    A2 --> B1
    B1 --> B2 --> C1
    C1 --> D4
    D4 --> H1
    D4 --> E1
    C1 --> G1
    G1 --> E1
    E1 --> F1
    E1 --> F2
    E1 --> F3
    E1 --> F4
    E4 --> F4
    F2 --> I1
    F2 --> I2
    F1 --> I3
    I3 --> J1
    H1 --> J1
```

The diagram traces a single complete loop: AFL data is scraped, transformed into rolling-window features, persisted as CSVs, and consumed both by the ML ensemble (which produces calibrated disposal predictions) and by the Scientist agent (which queries the same store deterministically). The agent reasons under CLAUDE.md guardrails, calls tools through Claude Code's MCP gateway, and writes its outputs back to disk and git - closing the loop with the backtest harness as the system's sole automated quality gate.

---

## Component deep-dives

### 1. Data ingestion and feature engineering

**In plain English:** Every player who has ever played AFL has a CSV of their game-by-game stats. We scrape new games each week, then turn those raw stats into rolling averages and trends the model can learn from.

**The methodology:** The pipeline is a batch ETL. AFL Tables and AFL.com.au are scraped into a flat structure of 13,321 player performance CSVs covering 1897–present, plus per-season match files. Feature engineering (in `prediction.py`) constructs rolling-window features on a per-player basis: recent form (3-game, 5-game, season-to-date), opponent strength, venue effects, and contextual flags (home/away, day/night, season stage). The target is disposals per game. Coverage caveats are explicit in the data layer: tackles only from 1987, clearances and contested possessions from 1998, and the 2017 hit-out jump is a recording change rather than a real shift in play (see `agent-memory/Scientist/data_stat_coverage_eras.md`).

**For the ML practitioner:** Leakage prevention is the load-bearing piece. The `LeakProofPredictor` enforces a strict temporal cutoff - when predicting round N, only data strictly before round N is visible during feature construction and model fitting. Cross-validation uses GroupKFold keyed on player ID, so a single player never appears in both train and validation folds (this prevents the model memorising player-specific levels rather than learning generalisable signal). What would change at production scale: a streaming feature pipeline (Kafka/Flink) instead of weekly batch, a real feature store (Feast or Tecton) with point-in-time correctness guarantees rather than ad-hoc CSV reads, and feature drift monitoring to detect when scraping format changes silently corrupt a column.

---

### 2. ML inference layer

**In plain English:** Three different machine-learning models each predict next-round disposals, and we average their answers. Averaging different model types is more robust than any one of them alone.

**The methodology:** A `VotingRegressor` ensembles three diverse base learners - `HistGradientBoostingRegressor`, `LightGBMRegressor` (GPU-accelerated), and `RandomForestRegressor`. Hyperparameters are tuned via Optuna's TPE sampler over a 50-trial budget against MAE on a held-out temporal slice. A post-hoc out-of-fold linear calibration step is applied to fix top-end compression observed in the raw outputs (see `agent-memory/Scientist/prediction_top_end_compression.md` - log1p targets and L1 LightGBM loss were both contributing to a compressed prediction range and were removed).

**For the ML practitioner:** The validation design is walk-forward, not random k-fold. For each completed 2026 round, the model is retrained from scratch using only pre-round data, predictions are generated for that round, and metrics are computed against actuals. Random splits would inflate every metric by leaking future games into the training set - a common failure mode in sports prediction codebases. Known failure modes: top-10 player MAE sits around 10.8 disposals (≈ 2.6× the global MAE of 4.11), driven by a residual ceiling effect and the long right tail of elite-player game counts. Production gaps: no model registry (currently the model is rebuilt on demand and not versioned), no shadow deployment for new model variants, no automated distribution shift monitoring on either inputs or predicted outputs, and no rollback path if a regression slips into production.

---

### 3. LLM reasoning layer - the Scientist agent

**In plain English:** A Claude agent sits on top of the data and the model. You can ask it questions in plain English and it will write Python, run it, look at the results, write more Python, generate a chart, edit a markdown file, and commit the change - all without you touching a keyboard.

**The methodology:** The Scientist agent runs the ReAct pattern (Reason → Act → Observe → repeat). Claude Opus reads the user's task, decides which tool to invoke, executes it via the Claude Code harness, observes the result (stdout, file diff, error trace), and iterates until the task is closed out. Multi-turn reasoning is the default; the agent routinely writes throwaway Python scripts to inspect data before committing to an analysis approach. CLAUDE.md acts as the persistent system prompt - it encodes domain rules (data coverage caveats, ranking constants, stat era boundaries), behavioural constraints (no emojis, absolute paths only, no summary `.md` files), and project workflow (push to main after major changes, commit with co-author tag). Subagent spawning lets long tasks fork specialised workers (e.g. one subagent rebuilds Hall of Fame docs while another runs the next-round prediction).

**For the ML practitioner:** The substrate is Anthropic's tool-use API - each tool is registered with a JSON schema describing its inputs and outputs, and the model is trained to emit structured tool calls when appropriate. The harness routes those calls to local executors. Production gaps are real: Opus latency is 3–10 seconds per turn and a non-trivial task chains 50+ turns, costs scale linearly with token volume and there is no per-task budget enforcement, there is no fallback to a smaller cheaper model when the task is simple, and there is no formal token accounting per session. A production deployment would tier the agent (Haiku for trivial retrieval, Sonnet for normal tasks, Opus only for the hardest reasoning) and enforce hard token ceilings before invocation rather than after.

---

### 3b. FootyStrategy - tactical interpretation agent

**In plain English:** The Scientist tells you what the numbers honestly say. FootyStrategy tells you what to do about it in football terms. The two roles are deliberately kept apart - one is a statistician, the other is a coaching council - so that a tactical recommendation never silently inherits more confidence than the underlying data supports.

**The methodology:** FootyStrategy deliberates through an 8-lens council, with each lens a distinct coaching perspective: the *Conditioner* (preparation, work-rate, repeatable effort), *Tempo Architect* (game-speed control), *Structuralist* (zones, defensive shape), *Match-up Tactician* (named opposition threats and weekly profiling), *Talent Developer* (role fit over résumé), *Innovator* (exploits in the prevailing meta), *Culture Custodian* (standards, identity), and *List Strategist* (multi-year horizon, draft discipline). Every output is tagged with a confidence tier - *Settled* (multiple lenses converge and the underlying data is robust), *Probationary* (lenses converge but data is exploratory), *Contested* (genuine lens disagreement), or *Insufficient Evidence* (neither data nor lens consensus). Every Settled or Probationary recommendation must include a **tripwire**: an explicit observable that, if seen, would overturn the recommendation. No tripwire means the recommendation is automatically downgraded. Caveats from the Scientist's findings propagate through unchanged - associational evidence cannot become causal in the tactical layer, and the data tier caps the recommendation tier. The agent emits a structured envelope (Tier, Horizon, active lenses, per-lens reads, Convergence/Tensions, recommendation, tripwire, caveats, Out of Scope statement). The handoff to the Scientist is one-directional: the Scientist produces numbers, FootyStrategy interprets tactically, and FootyStrategy never fabricates numbers or overrides upstream confidence labels. Invocation is via `@"FootyStrategy (agent)"` in Claude Code; the system prompt lives in the agent definitions directory and persistent memory at `.claude/agent-memory/FootyStrategy/MEMORY.md`.

**For the ML practitioner:** The interesting design choice is the *structured multi-perspective council* pattern rather than a single composite prompt. A single prompt asking "give me a tactical read with confidence" tends to anchor on whichever framing arrived first in the prompt and to bury tensions inside hedged paragraphs. The 8-lens structure does two things differently: it reduces anchoring bias by forcing each perspective to be produced separately before they are reconciled, and it surfaces disagreement explicitly through the *Contested* tier rather than averaging tensions away. The production-grade equivalent is a "society of mind" pattern - in LangGraph, each lens would be a separately-prompted subagent node with its own working memory, edges that aggregate into a deliberation node, and an explicit reconciliation step that emits the final tier. Known limitations are honest: no GPS or spatial data, no video, no real-time injury feed - the layer is pattern-recognition only and its output quality is bounded by the Scientist's data quality.

---

### 4. RAG layer - deterministic retrieval over structured data

**In plain English:** When the agent needs a fact like "Sam Berry's average tackles in 2026," it runs a pandas filter over a CSV. There is no fancy semantic search, and there shouldn't be - for clean structured numbers, plain queries are exactly right.

**The methodology:** Retrieval is fully deterministic: pandas reads the relevant CSV, filters by player ID and season, computes the requested aggregate, and returns the number. No embedding model, no vector store, no nearest-neighbour search. This is the correct architecture for the data shape - semantic similarity adds noise, not signal, when the user query maps directly to a structured filter. Vector search is great for "find me documents about clearance work in wet-weather games"; it is strictly worse than `df.query(...)` for "give me Marcus Bontempelli's disposal average in round 3 home games."

**For the ML practitioner:** Vector search becomes appropriate when unstructured text enters the corpus - match commentary, scouting reports, post-game interviews, injury notes. The production upgrade path is hybrid retrieval: pgvector or Qdrant for the unstructured side, the existing pandas/CSV layer for structured queries, and a routing layer that decides which to hit based on query parse. Chunking strategy for added commentary would be paragraph-level with a 1–2 sentence overlap, with metadata filters on round, team, and player ID so the retriever can scope to relevant context before semantic ranking. The structured layer should remain authoritative for any numeric claim.

---

### 5. Tool router and MCP gateway

**In plain English:** MCP is a standard way for AI models to talk to tools - a USB-C port for AI. Instead of every tool needing its own bespoke integration, MCP defines one protocol the model uses to discover and call any registered tool.

**The methodology:** SuperCoach-VIA uses Claude Code's built-in MCP implementation. The registered tool surface includes Bash (shell command execution), Read / Write / Edit (filesystem), WebFetch and WebSearch (HTTP and search), and Agent (subagent spawning). Tool selection is model-driven: Claude inspects the task, the available tool schemas, and the conversation state, then emits a structured tool call which the harness dispatches. There is no hand-coded routing logic - the model is the router. Tool definitions are JSON-schema documents declaring inputs, outputs, and behavioural notes that condition selection.

**For the ML practitioner:** The MCP specification (Anthropic, 2024) defines a JSON-RPC transport with a server/client model - tools live in MCP servers, and any MCP-aware client can discover and invoke them. Production gaps in this deployment: no sandboxing on Bash (commands execute directly against the host filesystem with the user's privileges), no rate limiting on tool invocation, no input sanitisation on shell arguments, and the spec itself is still pre-1.0 with breaking changes possible. A production hardening path: route every Bash invocation through a gVisor or Firecracker microVM sandbox with a read-only mount of the data layer and a tightly-scoped write area; validate every tool call's inputs and outputs against the registered schemas before dispatch; emit a structured audit record (tool name, arguments hash, caller agent ID, timestamp, latency, exit status) on every call; pin the MCP spec version and gate version upgrades behind regression tests.

---

### 6. Eval harness

**In plain English:** Before publishing a number, the system pretends it's the past. It re-trains the model using only data that would have been available at the time, predicts the round it doesn't yet know the result for, then compares against what actually happened. That gives an honest read on how good the predictions are.

**The methodology:** A walk-forward backtest covers all completed 2026 rounds. For round N, the pipeline trains exclusively on data from before round N (strict temporal cutoff enforced inside `LeakProofPredictor`), generates predictions for every player who played in round N, and joins to actuals. Reported metrics: MAE, RMSE, within-5, within-10, signed bias, and a top-10 MAE slice that surfaces the most-elevated-profile failure mode. Backtest output is persisted as CSV under `data/prediction/backtest/` so each run leaves a permanent record.

**For the ML practitioner:** The eval covers ML model outputs only - LLM output quality is currently measured by post-hoc human review (~70–75% factual accuracy on the first pass of Hall of Fame documents, raised to ~99% after a systematic correction process). Production gap: automated LLM evaluation. RAGAS would give faithfulness and context-relevance scores on every retrieval-augmented response; DeepEval would flag hallucinations against a reference corpus. Online eval is also missing - once a real round is played, the system does not automatically score its earlier predictions against actuals and alert on MAE regression. Adding an online eval loop (cron job that ingests results, joins to predictions, computes metrics, emits to Langfuse/Phoenix) would close the most important production blind spot.

---

### 7. Observability

**In plain English:** When something looks wrong, you want to know who or what produced it, when, and based on which version of the system. Right now we lean on git for that.

**The methodology:** Git history serves as a lightweight audit trail - every doc change is a commit with author, timestamp, diff, and message. Backtest CSVs persist model performance over time, so a regression is visible by diffing two runs. Chart filenames are timestamped, so generation history is recoverable. CLAUDE.md is version-controlled, so the agent's policy state at any past commit is reconstructable.

**For the ML practitioner:** What's missing is structured LLM trace logging - for every agent turn, you want the full prompt (including system prompt and tool definitions), the model's response, the tool calls emitted, latency per turn, token counts, and dollar cost. Without that, debugging "why did the agent do X" is a forensic exercise on git diffs. You also want model performance dashboards (MAE by round, by player tier, by team, drift charts) and alerting on MAE regression beyond a threshold. A solid open-source stack: OpenTelemetry as the trace collection plane (it has wide language support and a stable spec), Langfuse as the LLM-specific trace store and dashboard layer (self-hostable, MIT licensed). For on-prem with stronger compliance posture, Arize Phoenix is the leading open-weight equivalent.

---

### 8. Multi-agent hierarchy - The Crumb

**In plain English:** The Crumb is the project's coaching brain - thirteen specialist agents organised into six tiers, with a Senior Coach orchestrator on top. You ask one question at the front door ("what should we do about our forward line going into round 11?") and the Senior Coach decomposes it, dispatches the relevant specialists, and integrates their answers back into a single coherent recommendation. It is named after the AFL crumber - the small forward who reads where the ball will spill from a pack before the contest resolves. The Crumb sees the pattern before the coach has called it.

**The methodology:** Architecture and per-agent role definitions live in `docs/footy-ai-chatbot-setup.md`. The 13-agent, 6-tier structure:

| Tier | Agent | Role |
|------|-------|------|
| 1 | Senior Coach Agent | Orchestrator - decomposes queries, integrates responses, owns final framing |
| 2 | Midfield Coach Agent | Clearance, contested ball, stoppage, rotations |
| 2 | Forward Line Coach Agent | Inside-50 entries, forward structure, conversion, set pieces |
| 2 | Back Line Coach Agent | Defensive setup, intercept structure, rebound chain |
| 3 | Stoppage Specialist | Centre-bounce and around-ground stoppage designs |
| 3 | Defensive Press Specialist | Forward-half pressure, kick-in coverage, transition defence |
| 4 | Match Analyst Agent | Pattern recognition across recent fixtures |
| 4 | Opposition Analyst Agent | Weekly opposition profile, personnel, historical tendencies |
| 4 | Stats/Methodology Agent | Quantitative work - the existing Scientist agent |
| 4 | Strategy Council Agent | Tactical interpretation - the existing FootyStrategy agent |
| 5 | High Performance Agent | Load, fitness, injury risk, return-to-play |
| 5 | List Manager Agent | Multi-year list strategy, contract, draft |
| 6 | Data Steward Agent | Owns data files, enforces CLAUDE.md verification rule |

Technology mapping: the Senior Coach runs on `claude-opus-4-7` because orchestration and final framing are the highest-reasoning workload; line coaches, specialists, and analysts run on `claude-sonnet-4-6` (fast, capable enough for scoped specialist work); the Data Steward runs on `claude-haiku-4-5` because its job is structured retrieval and verification, not open-ended reasoning. Orchestration uses the Claude Code agent SDK's tool_use plus subagent-spawning model - the Senior Coach calls each tier-2/3/4 agent through the Agent tool, receives a structured envelope back, and integrates. Data access is shared across the tiers via the existing repo layout: `data/player_data/`, `data/matches/`, `data/lineups/`, and `data/prediction/`. The Scientist and FootyStrategy agents already documented above are slotted into Tier 4 - The Crumb is the structural envelope that gives them a coordinated calling pattern rather than ad-hoc invocation.

**For the ML practitioner:** The Crumb is a specialisation of the **hierarchical agent** pattern - one orchestrator, many workers, with role boundaries enforced through system prompt scoping rather than code-level access control. The trade-off is honest: prompt-scoped boundaries are cheap to define and flexible to evolve but provide no hard guarantee that, say, the List Manager will not attempt to write a tactical recommendation outside its scope. The production upgrade path is two-part. First, LangGraph to make state transitions explicit and handoffs durable - each agent becomes a node with declared inputs and outputs, the graph is the orchestration contract, and a crashed agent turn can be replayed from the last persisted state rather than re-running the whole workflow. Second, Temporal.io for retry semantics when an agent turn fails mid-workflow - exactly-once execution, exponential backoff on transient model errors, and workflow versioning so an in-flight session does not silently switch behaviour when the agent definitions are updated. Hard role enforcement (least-privilege credentials per agent, capability-scoped MCP tool sets) becomes meaningful once those two are in place.

---

### 9. Claude Code setup - building and extending the agent stack

**In plain English:** Claude Code is the substrate the whole agent stack runs on. It is not a chatbot in a browser tab - it is a CLI/IDE harness that gives Claude direct access to your filesystem, shell, web, and other agents, governed by a policy document checked into the repo. This section is how someone with the repo and an API key gets the full Scientist + FootyStrategy + Crumb stack running, and how they would extend it with a new agent or data source.

**The methodology:**

1. **What Claude Code is.** Anthropic's official CLI and IDE-integrated AI tool. It is the agent harness for this project - the thing that turns a Claude API call into a usable agent. It implements the MCP gateway (tool discovery and dispatch), the tool-use API loop (ReAct cycles with structured tool calls), multi-agent orchestration (parent agents can spawn subagents via the Agent tool), and the CLAUDE.md policy system (a versioned, repo-resident system prompt that governs every session). Distinguishing it from a chatbot wrapper: local filesystem read/write, shell execution, git operations, web fetch, and subagent spawning are all native primitives, not workarounds.

2. **Installation and prerequisites.** Install the CLI with `npm install -g @anthropic-ai/claude-code`, or use the desktop app. Authenticate by setting `ANTHROPIC_API_KEY` as an environment variable. For the SuperCoach-VIA project specifically, the Python environment used by every agent is the project venv at `/home/abhi/sourceCode/python/coding/.venv` - it already contains pandas, scikit-learn, lightgbm, matplotlib, optuna, and the rest of the analysis stack. Clone the repo with `git clone https://github.com/apur27/SuperCoach-VIA` and run Claude Code from inside the repo root so CLAUDE.md is auto-loaded.

3. **Agent definitions.** Custom agents are registered as markdown files under `.claude/agents/`. Each file is a frontmatter block plus a system prompt: `name`, `model` (e.g. `sonnet`, `opus`), `description`, and `tools` (whitelist of tools the agent is allowed to invoke) in the frontmatter, then the system prompt as the body. The Scientist and FootyStrategy definitions live there now. Any new agent - say an Opposition Analyst for The Crumb - is added by writing a new `.md` file in the same directory following the same shape.

4. **CLAUDE.md - the policy layer.** The project-level `CLAUDE.md` at the repo root governs every agent in every session. It encodes the data verification rule (no player stat is written without first reading it from the data files), absolute-path requirements, push policy (push to main, no PRs), and the broader behavioural constraints (no emojis, no summary `.md` files, no ad-hoc filesystem changes outside the agreed scope). This is the agent's constitution - it is read first on every turn and is enforced for the entire session. Versioning it in the repo means policy state at any past commit is reconstructable, which is a real audit primitive.

5. **Extending the stack - adding a new agent.** Four steps. *Step 1:* create `.claude/agents/<agent-name>.md` with frontmatter declaring name, model, description, and the tool whitelist, and the system prompt as body. *Step 2:* define the agent's data access scope in the system prompt - which directories it is allowed to read, which directories it must not write to, which data files are authoritative for which claims. *Step 3:* invoke it from a chat session with `@"<Agent Name> (agent)"`, or programmatically from a parent agent via the Agent tool. *Step 4:* register it in the tier table in `docs/footy-ai-chatbot-setup.md` and mirror the entry in this architecture doc, so the system has a single source of truth for what agents exist and what they do.

6. **MCP server extensions.** Claude Code supports custom MCP servers beyond its built-in tools. Example: a live AFL data MCP server exposing `get_live_scores`, `get_player_stats`, and `get_team_form`. Define it in `.claude/settings.json` under the `mcpServers` key, declaring the transport (stdio or HTTP), the entry point, and any required environment variables. Each MCP server declares its own tool schemas, and Claude Code discovers them at session start - no client-side wiring needed beyond the registration entry. This is the path to integrating real-time data sources without baking them into the agent code.

7. **Memory system.** Persistent agent memory lives under `.claude/agent-memory/<agent-name>/`. The convention is a top-level `MEMORY.md` acting as an index of one-line pointers, with individual topic files (`data_quirks.md`, `user_preferences.md`, etc.) as the entries. Memory persists across sessions and is loaded into the agent's context at conversation start. The Scientist's memory currently carries data-quirk notes, baseline numbers, and reproducibility recipes - all written from prior sessions so future sessions do not re-derive them. FootyStrategy's memory at `.claude/agent-memory/FootyStrategy/MEMORY.md` does the same for tactical patterns. The memory system is what lets the agent tailor responses without re-explaining context on every invocation.

**For the ML practitioner:** The interesting choice is that the entire agent definition surface - system prompts, tool whitelists, memory, policy - lives in the repo as plain markdown and JSON. There is no separate "agent registry" service. That makes the stack diff-able, reviewable, and rollback-able via the same git workflow as the code, which is the right default for a small-team or single-operator deployment. The production upgrade path is to lift the agent definitions into a managed registry once you have multiple teams editing concurrently - but the markdown-in-repo pattern scales further than people expect, and the audit benefits are real.

---

### 10. The six-agent council — architecture and interaction model

**In plain English:** The repo has six specialised agents. Each one does a narrowly-defined job, deliberately separated so that no single agent has to be good at everything - and so that errors made by one are caught before they propagate into the final output. This section documents who they are, what they hand off to each other, and how to invoke them correctly.

**Agent registry:**

| Agent | Job | Model | Tool surface | Invocation |
|-------|-----|-------|-------------|-----------|
| **Scientist** | Data analysis, statistical tests, code execution, prediction runs | Claude Opus | All tools (Bash, Read, Write, Edit, WebFetch, Agent) | `@"Scientist (agent)"` |
| **FootyStrategy** | Tactical interpretation, 8-lens council, confidence tiering, tripwire writing | Claude Opus | All tools | `@"FootyStrategy (agent)"` |
| **BriefBuilder** | Pre-match brief skeleton - auto-populates [data] tags, leaves FOOTYSTRATEGY INSERT placeholders | Claude Sonnet | All tools | `@"BriefBuilder (agent)"` |
| **DataSentinel** | Pre-commit verification gate - walks every [data] tag, confirms against source CSV, emits PASS/FAIL | Claude Haiku | All tools | `@"DataSentinel (agent)"` |
| **Skeptic** | Adversarial review - checks tripwire observability, caveat-hierarchy fidelity, lens-tension smoothing | Claude Opus | All tools | `@"Skeptic (agent)"` |
| **Codex** | Outside-the-frame second opinion, model-internals inspection, blind adversarial read | External (GPT series) | Shared runtime | `@"Codex (agent)"` |

**Standard pre-match pipeline — sequence and handoffs:**

```
 Monday
    │
    ▼
┌─────────────┐   team names + round
│ BriefBuilder│──────────────────────► writes data skeleton
│  (Sonnet)   │                        [data] tags on every number
└─────────────┘                        <!-- FOOTYSTRATEGY INSERT --> placeholders
    │ brief skeleton
    ▼
┌─────────────┐   skeleton + target question
│  Scientist  │──────────────────────► statistical tests, effect sizes, CIs
│   (Opus)    │                        caveats propagated explicitly
└─────────────┘
    │ findings with uncertainty
    ▼
┌─────────────┐   skeleton + Scientist findings
│ FootyStrategy──────────────────────► 8-lens council deliberation
│   (Opus)    │                        tiered recommendations + tripwires
└─────────────┘
    │ complete draft
    ▼
┌─────────────┐   complete assembled brief
│ DataSentinel│──────────────────────► verifies every [data] tag vs source CSV
│   (Haiku)   │                        PASS or annotated FAIL list
└─────────────┘
    │ verified brief
    ▼
┌─────────────┐   verified brief + upstream chain
│   Skeptic   │──────────────────────► tripwire observability check
│   (Opus)    │                        caveat-hierarchy audit
└─────────────┘                        PASS / PASS_WITH_CONCERNS / BLOCK
    │
    ▼
 Brief ready to print
    │
    └── optional: Codex blind read for outside-the-frame gaps
```

**Handoff contracts — what each agent receives and what it must not invent:**

| Handoff | What must be passed explicitly | What the receiving agent must not do |
|---------|-------------------------------|--------------------------------------|
| BriefBuilder → Scientist | The brief skeleton with [data] tags already placed | Scientist must not re-derive stats already in the skeleton; augment, don't redo |
| Scientist → FootyStrategy | Effect size, CI, sample size, and the verbatim caveat on each finding | FootyStrategy must not upgrade "associational" to "causal"; must not fabricate numbers not in Scientist's output |
| FootyStrategy → DataSentinel | The complete assembled draft with all [data] tags and sources named in the methodology para | DataSentinel must not interpret meaning; it checks numbers only |
| FootyStrategy → Skeptic | The full draft + Scientist findings + BriefBuilder skeleton as context | Skeptic must not rewrite the draft; must report line-number + quote + proposed fix |
| Any agent → Codex | The complete brief and the specific question | Codex should not act as a primary fact source for AFL stats - it is a reasoning check, not a data source |

**Standalone invocation rules:**

- **Scientist**: can be invoked standalone for any data analysis task. No prior context needed.
- **FootyStrategy**: works best with some data context (from Scientist or BriefBuilder). Can be invoked standalone for post-match tactical reads where the question is conceptual rather than statistical.
- **BriefBuilder**: always invoked first in any match-preparation cycle. No prior context needed.
- **DataSentinel**: invoked on any draft document with [data] tags. Can be run standalone mid-draft for a partial check.
- **Skeptic**: invoke only after FootyStrategy has produced a *complete* draft. Premature invocation before the recommendation is settled generates noise.
- **Codex**: invoke when you want a second opinion uncorrupted by the other agents' reasoning chains, or when the question touches model internals (feature contributions, why did the predictor give this player that score).

**Model selection rationale:**

The council tiers its model selection by task complexity, not by prestige. Using Opus everywhere is slower and more expensive without improving quality on scoped tasks.

| Agent | Why this model |
|-------|---------------|
| DataSentinel → Haiku | Pure structured retrieval and pattern matching against a known schema. No complex reasoning; speed and cost matter more than chain-of-thought depth. |
| BriefBuilder → Sonnet | Data-assembly templating: open files, read rows, write tagged output to a fixed structure. Moderate complexity — Sonnet handles it cleanly without Opus latency. |
| Scientist → Opus | Long reasoning chains: statistical test selection, feature engineering decisions, caveat propagation, uncertainty quantification. The hardest reasoning workload in the council. |
| FootyStrategy → Opus | Multi-perspective deliberation: activating 3–5 distinct coaching lenses, surfacing tensions, tiering a recommendation with a falsifiable tripwire. Requires broad knowledge and nuanced judgement. |
| Skeptic → Opus | Adversarial reasoning: detecting caveat drift, assessing tripwire observability from first principles, identifying smoothed tensions. Must be as capable as the agent it reviews. |
| Codex → External | Deliberate provider separation: an independent model family avoids anchoring on the same training-data priors as the rest of the council. The value of Codex is the difference in perspective, not raw capability. |

**For the ML practitioner:** The interesting design choice is that error containment is structural rather than prompt-based. Rather than adding "check your numbers" to every agent's system prompt, DataSentinel is a dedicated gate — a separate agent whose only job is verification, running on a cheap model because the task is cheap. Similarly, rather than adding "play devil's advocate" to FootyStrategy's prompt (which causes self-critique to be soft and polite), Skeptic is an adversarial agent with no social stake in the draft it reviews. The pattern is: *separate agents for separate failure modes*. The production equivalent is a pipeline with explicit checkpoints — a lint step for data tags, a confidence-threshold gate before publication, a separate evaluator that does not share weights with the generator. These are well-established patterns in ML pipelines (train/eval split, holdout sets, A/B test gating); the council applies the same logic to the LLM layer.

---

## Eval results

| Metric | Value | Notes |
|--------|-------|-------|
| Prediction MAE | **[data]** 4.063 disposals | Player-weighted across 11 rounds, 4,074 player-rounds (Rounds 1–11, 2026) |
| RMSE | **[data]** 5.195 | |
| Within 5 disposals | **[data]** 72.3% | |
| Within 10 disposals | **[data]** 95.6% | |
| Signed bias | **[data]** -0.069 | Essentially unbiased in aggregate |
| Round 1 MAE | **[data]** 4.83 | Elevated — no within-season rolling features |
| Round 5 MAE | **[data]** 3.73 | Best MAE so far |
| Round 11 MAE | **[data]** 3.83 | Best within-5 so far (**[data]** 77.2%) |
| Top-10 player MAE | ~10.8 | Top-end compression — known failure mode |
| LLM factual accuracy | ~70–75% pre-correction → ~99% post | Measured by external review of Hall of Fame docs; systematic correction process in place |

A 4.063-disposal player-weighted MAE across 4,074 player-rounds means the typical prediction misses by about four disposals, which on a per-player range of 0–45 is roughly 9% of the active range — usable signal, not a solved problem. The 95.6% within-10 figure says coarse predictions are reliable; the 72.3% within-5 figure says fine predictions are improving but still leave headroom. Round 1 sits ~1.0 MAE above the rolling-rounds floor because there are no within-season rolling features available before any 2026 game has been played. The top-10 player MAE remains the headline failure: elite players are systematically harder to predict because their week-to-week ceilings move on context (tag absorption, opponent matchups, role rotations) that the current feature set captures only partially. Roadmap targets: cut top-10 MAE below 8.0 via an opponent-tag feature and a within-season rolling feature for round 1, raise within-5 above 78% via better calibration on the upper tail, and add an automated online eval loop so post-round actuals score the predictions without manual triggering.

**Full per-round table (all 11 rounds), team-level bias for every club, and round-by-round biggest misses:** [docs/afl-backtest-2026.md](afl-backtest-2026.md).

---

## Australia's AI Ethics Principles — how this project maps

Australia's [AI Ethics Principles](https://www.industry.gov.au/publications/australias-ai-ethics-principles) are a set of eight voluntary principles published by the Department of Industry, Science and Resources to guide responsible AI design, development and deployment. They have since been complemented by the *Guidance for AI Adoption* (October 2025) and the *Voluntary AI Safety Standard*, but the eight principles remain the canonical reference point for whether an AI system has been built with the right things in mind. This section maps each principle to the concrete artefacts in this repo — what's actually in place, and where the project is honest about not meeting the principle in full.

SuperCoach-VIA is a personal weekend project, not a production deployment, and the mapping reflects that: some principles are well-served by the architecture; others are partially met; one (contestability) has no meaningful surface here because there are no external users to contest anything. The aim is to be useful to an auditor or regulator reading the repo, not to claim more than the artefacts support.

### 1. Human, societal and environmental wellbeing

> *"Throughout their lifecycle, AI systems should benefit individuals, society and the environment."*

The system's purpose is football analysis and pedagogy — the README's "Why this repo exists" framing positions it as a reference architecture for AI engineers and a learning resource for fans, not a decision-support tool with material stakes. The harm surface is correspondingly narrow: no individual is denied a service, opportunity, or resource based on a model output. **Partial gap:** environmental footprint is not measured. The system runs ML training on a local GPU and makes paid Anthropic API calls; neither the per-run kWh nor the per-session token-driven inference compute is tracked. A production deployment would account for this; this project does not.

### 2. Human-centred values

> *"Throughout their lifecycle, AI systems should respect human rights, diversity, and the autonomy of individuals."*

The system processes only publicly-published AFL match and player statistics. Player names appear in outputs because the data is inherently per-player, but no demographic inferences are made, no automated decisions affect any player or coach, and the agent council's tactical recommendations are explicitly framed as advisory — the FootyStrategy envelope ends with an "Out of Scope" statement and the system has no mechanism to act on its own recommendations. **Honest qualifier:** human review is the *default* publication path for documents authored by the agent council, but the live in-game analysis pipeline (`live_analysis_pipeline.py`) is rule-based code that auto-commits and `git push`es every ~90 seconds during a match without per-block human approval. Those blocks are deterministic outputs of Scientist-authored code (no LLM inference at runtime), but they reach `main` without a human in the loop on each push.

### 3. Fairness

> *"Throughout their lifecycle, AI systems should be inclusive and accessible, and should not involve or result in unfair discrimination against individuals, communities or groups."*

The prediction model uses only on-field statistical features (rolling form, opponent, venue, context) and does not ingest demographic attributes. **Honest framing:** absence of demographic features rules out one specific class of risk (direct discrimination on a protected attribute) but is not by itself a fairness audit — proxy variables can still encode group membership, and equal-accuracy across groups has not been measured. The closest available slice is performance tier: top-10 player MAE sits at ~10.8 disposals vs the global 4.06, meaning the model is meaningfully less accurate for elite players. This is disclosed in the eval-results table and in `agent-memory/Scientist/prediction_top_end_compression.md`. **Partial gap:** no formal slice analysis has been run across player position, age, team, or era. Per-position analysis is blocked by a known data limitation (no position column in player CSVs, noted in `agent-memory/Scientist/data_no_position.md`); other slices have not been prioritised. A regulator-grade fairness review is not on the artefact list.

### 4. Privacy protection and security

> *"Throughout their lifecycle, AI systems should respect and uphold privacy rights and data protection, and ensure the security of data."*

All data at rest is publicly-published AFL match and player statistics — no private personal information, no protected health information, no minors' data beyond what AFL publishes officially for senior-team players. The data layer is read-only flat CSVs in a public GitHub repository (github.com/apur27/SuperCoach-VIA). **The data-at-rest privacy story is essentially trivial; the runtime security story is not, and several gaps are open.** Bash invocations through the MCP gateway execute against the host filesystem without sandboxing (documented in section 5), which is a software-supply-chain risk if the agent stack is ever run against an untrusted prompt source. Agent sessions also exchange the local repo contents with the Anthropic API (and Codex with OpenAI), which is fine for public AFL data but would not be acceptable for any deployment that touched private information without first solving the model-hosting question. Section 4 of the sovereign-deployment notes describes the gVisor / Firecracker sandboxing and self-hosted-model path that would close both gaps; neither is in place now.

### 5. Reliability and safety

> *"Throughout their lifecycle, AI systems should reliably operate in accordance with their intended purpose."*

This is the principle the architecture most fully addresses, and the strongest evidence in the section. The `LeakProofPredictor` enforces strict temporal cutoffs and GroupKFold by player ID to prevent leakage. The walk-forward backtest harness re-runs every completed 2026 round under the same constraints it would face in real prediction, and reports MAE / RMSE / within-N / signed bias / top-10 MAE — all published in the eval-results table above and in `docs/afl-backtest-2026.md`. The six-agent council *describes* two additional review steps for documents: DataSentinel verifies every `[data]` tag against the source CSV, and Skeptic performs adversarial review on tripwire observability and caveat fidelity. **Honest qualifier on those two:** as called out in `docs/ARCHITECTURE.md` §13.4 (Gap 8), the council sequence is "descriptive, not enforced" — each agent is invoked manually and there is no orchestrator or pre-commit hook that blocks publication if a step is skipped. Known reliability limits are reported alongside the numbers, not buried: top-10 MAE elevation, round 1 cold-start, lack of an automated online eval loop.

### 6. Transparency and explainability

> *"There should be transparency and responsible disclosure so people can understand when they are being significantly impacted by AI, and can find out when an AI system is engaging with them."*

The agent stack is fully documented in this file. Every model used is named, every tool surface is enumerated, the council's handoff contracts are documented, and the policy layer (CLAUDE.md) is version-controlled and diff-able. Published documents carry `[data]` tags that point to the source statistic, and FootyStrategy outputs carry a confidence tier (Settled / Probationary / Contested / Insufficient Evidence) plus a tripwire that names what observation would overturn the recommendation. **Honest limit on explainability:** the ensemble exposes global `feature_importances_` from its tree-based base learners — a user can see which features the model relies on overall — but no per-prediction (local) attribution like SHAP or LIME is generated, and no per-prediction confidence interval is exposed to the reader (the backtest gives an aggregate MAE / within-N envelope, not a player-specific one). A reader cannot answer "this specific prediction is high because feature X contributed +N for this player this week" or "what is the 80% interval on this single prediction." Closing both is straightforward (SHAP over the ensemble; quantile regression or conformal prediction for intervals) but is not yet on the artefact list.

### 7. Contestability

> *"When an AI system significantly impacts a person, community, group or environment, there should be a timely process to allow people to challenge the use or outcomes of the AI system."*

**Largely not applicable in the current deployment, and stated as such.** No AI output from this repo significantly impacts an external person, community, or group — there is no user-facing service, no decision is being made about anyone based on a model output, and the tactical recommendations are advisory text published to a documentation site. The GitHub issue tracker exists as an informal feedback channel for any reader who disagrees with a published claim, but that is a feedback surface, not a formal challenge process — there is no SLA, no documented review procedure, and no escalation path. If the system were ever to be used to inform decisions with real stakes (a wagering product, a fantasy-sports tool with paid users, a coaching-staff recommendation system), a formal challenge process would be required and is not in place.

### 8. Accountability

> *"Those responsible for the different phases of the AI system lifecycle should be identifiable and accountable for the outcomes of the AI systems, and human oversight of AI systems should be enabled."*

A single named operator (the repo owner) is accountable for the data layer, the model, the agent stack, the policy document, and every published artefact. The git log is an immutable record of who or what (human author or co-authored agent) made every change, with timestamp, diff, and commit message. CLAUDE.md is version-controlled, so the agent's policy state at any past commit is reconstructable for audit. Human oversight is the *intended* workflow for agent-authored documents — DataSentinel + Skeptic are the documented pre-publication review steps for documents containing `[data]` tags or tactical recommendations. **Honest limit on enforcement (Codex-flagged):** the review steps are convention, not a machine-enforced gate — there is no pre-commit hook or CI check that blocks a push if DataSentinel was skipped (`docs/ARCHITECTURE.md` §13.4 Gap 8 names this explicitly). The live in-game analysis pipeline also bypasses human-in-the-loop entirely, auto-committing rule-based blocks to `main` every ~90s during a match. **Honest limit at scale:** with a single accountable operator, role separation is not enforced — the same person owns data ingest, model training, policy, and publication. The sovereign-deployment notes describe the role-separation path (least-privilege credentials per agent, separate retrain / publish / review identities) that would be required if this were ever staffed by a team. The accountability story is therefore better described as *attributable* (git log makes every change traceable to a named author or co-authored agent) than as *gated* (no automated control prevents an unreviewed change from reaching `main`).

### Summary

| Principle | Status | Most credible evidence | Most honest gap |
|-----------|--------|------------------------|-----------------|
| 1. Wellbeing | Met for the project's narrow scope | Pedagogical / reference-architecture framing; no decision-stakes | No environmental footprint tracking; "low stakes" is not the same as positive wellbeing contribution |
| 2. Human-centred values | Partial | Advisory-only outputs; no demographic inferences | Live in-game pipeline auto-commits to `main` without per-block human review |
| 3. Fairness | Partial | No demographic features; top-10 MAE elevation disclosed | No formal slice analysis; absence of demographic features ≠ a fairness audit |
| 4. Privacy & security | Met for data; partial for runtime | Public data only; no PII at rest | MCP Bash gateway unsandboxed; agent sessions exchange repo contents with external model APIs |
| 5. Reliability & safety | Strongest in the section | Leak-proof predictor, walk-forward backtest, published metrics | DataSentinel/Skeptic gates are descriptive not enforced; no automated online eval loop |
| 6. Transparency & explainability | Met for system; partial for predictions | Versioned policy doc, `[data]` tags, confidence tiers + tripwires | Global feature importance only; no per-prediction SHAP/LIME or confidence interval |
| 7. Contestability | Not applicable at current scope | GitHub issues as informal feedback channel | No formal challenge mechanism (none needed at current stakes) |
| 8. Accountability | Attributable, not gated | Git as audit trail; versioned CLAUDE.md; named operator | Review gates are convention not machine-enforced; live pipeline auto-pushes; no role separation |

---

## What I'd do differently in a sovereign deployment

"Sovereign AI" means a deployment under the operator's full control - on-premises hardware or a VPC-isolated cloud tenancy, with data residency guarantees, no information leaving the relevant jurisdiction, and (for sensitive use cases) air-gapped or semi-air-gapped network postures. Australia's AI Ethics Framework and the broader push toward on-shore AI capability make this a live concern: government, defence, regulated industries, and any system handling protected data increasingly need an AI architecture that does not depend on hyperscaler API endpoints. The seven changes below are what I would actually build if SuperCoach-VIA had to run inside that envelope.

### 1. Replace API calls with self-hosted models

Move the agent from the Anthropic API to vLLM serving an open-weight model - Llama 3.1 70B Instruct or Mistral Large would be the realistic candidates, deployed on 2× H100 or 4× A100 hardware. Trade-offs are honest: latency per turn drops (no network hop, no shared queue) but per-token throughput on equivalent hardware is lower than Anthropic's optimised serving stack; cost flips from per-token to per-GPU-hour, which is cheaper at high utilisation and more expensive at low; capability gap on hard reasoning tasks is real - a 70B open-weight model is roughly Sonnet-equivalent on most benchmarks and meaningfully behind Opus on the long, complex reasoning chains the Scientist agent runs at its hardest. The right call: tier the deployment, with self-hosted Llama 70B handling 90% of routine work and an isolated, audited API path for tasks that genuinely need frontier capability.

### 2. Proper feature store and data lineage

Replace the ad-hoc CSV layer with Feast (open-source, Kubernetes-native) for online and offline feature serving with point-in-time correctness, and dbt for declarative transformation lineage so every feature has a graph from raw source through every transformation. Append-only audit logs on the feature store so any historical query about "what was this player's rolling-3 disposal average as of round 5" returns a deterministic answer regardless of when it is run. In regulated environments this matters because model decisions have to be reproducible months later for audit, and ad-hoc CSVs fail that test the moment someone updates a file in place.

### 3. Extend the eval harness to LLM outputs

Add RAGAS for faithfulness (does the answer reflect retrieved evidence) and context relevance (are the retrieved chunks actually about the question) on every agent response that touches structured retrieval. Add DeepEval for hallucination detection against a curated reference corpus of known-true claims about AFL history and rules. Wire in continuous online eval: a scheduled job that, after each completed round, joins predictions to actuals, computes MAE / RMSE / bias / top-10 MAE, persists the result, and triggers an alert if any metric regresses beyond a configured threshold. The same job should re-run a sample of recent agent responses through RAGAS and emit hallucination-rate trend.

### 4. Harden the MCP gateway

Every Bash invocation routes through a gVisor or Firecracker microVM sandbox - read-only mount of the data layer, write access only to a scoped scratch area, no network egress unless the tool definition explicitly grants it. Input/output schema validation on every tool call, with the schema versioned and pinned. A structured audit log (JSON, one record per call) capturing tool name, agent session ID, timestamp, input hash, output hash, exit status, and latency, written to an append-only store. Per-session rate limits to bound the worst-case behaviour of a runaway agent. Pin the MCP spec version and gate upgrades behind a regression suite.

### 5. Full observability stack

Instrument every LLM call with OpenTelemetry - trace ID, span ID, prompt template version, retrieved context, tool calls emitted, latency, token counts, model version, and (in dev) the full prompt and response. Self-host Langfuse as the trace store and dashboarding layer, with retention policies aligned to the data classification (e.g. 90 days for unclassified, 7 years for audit-relevant). Alerting on three conditions: prediction MAE regression more than 10% week-on-week, LLM hallucination rate above 2% on the rolling RAGAS sample, and tool call failure rate above 5% in any 1-hour window.

### 6. Human-in-the-loop gates

Confidence thresholds on agent outputs - when the model's self-reported confidence (or an external classifier's hallucination score) falls below a configured floor, the output is held in a review queue rather than published. A formal approval workflow before any document publishes to a user-facing surface, with the reviewer's identity and approval timestamp captured in the audit log. Chain-of-custody: every published document carries metadata linking it to the data snapshot, model version, agent session ID, and reviewer who signed off, so any future investigation can reconstruct the full provenance from a single document.

### 7. Multi-agent orchestration

Replace the single-agent, single-process model with LangGraph for stateful multi-agent workflows - defined roles (data-scrape agent, training agent, prediction agent, writer agent, reviewer agent), explicit handoffs, and persistent state across the workflow. Run the orchestration on Temporal.io for durable execution: retry semantics, exactly-once guarantees, workflow versioning, and the ability to replay a failed workflow from any prior step. Role separation matters for least-privilege - the writer agent should not be able to retrain models, the training agent should not be able to publish documents, and no single agent should hold credentials for all of data, model, and output stages.

---

## What this project already gets right

- Temporal leakage prevention in the ML pipeline - the `LeakProofPredictor` enforces a strict round-by-round cutoff and uses GroupKFold by player; this is genuinely rigorous and would survive a production review.
- CLAUDE.md as a versioned system prompt and policy document - agent behaviour is in source control, diffable, reviewable, and auditable rather than living in a cloud console.
- Git as an immutable audit trail - every doc change is attributable to an author (human or agent co-author), with timestamp, diff, and message.
- Backtest-first evaluation before any output is published - predictions ship with their MAE, RMSE, and within-N coverage already measured on out-of-sample data, not on the same data the model was trained on.
- Explicit data coverage caveats - pre-1965 incomplete records, tackles only from 1987, the 2017 hit-out recording change; these are documented in agent memory and surfaced in any analysis that touches the affected windows, so the agent does not fabricate confidence in numbers it cannot defend.

---

## Further reading

- [Anthropic MCP specification](https://modelcontextprotocol.io)
- [RAGAS - RAG evaluation framework](https://ragas.io)
- [Langfuse - open-source LLM observability](https://langfuse.com)
- [LangGraph - stateful multi-agent orchestration](https://langchain-ai.github.io/langgraph/)
- [vLLM - high-throughput LLM serving](https://vllm.ai)
- [Temporal - durable execution for agent workflows](https://temporal.io)
- [Australia's AI Ethics Framework](https://www.industry.gov.au/publications/australias-artificial-intelligence-ethics-framework)
- [Australia's AI Ethics Principles (the 8 principles)](https://www.industry.gov.au/publications/australias-ai-ethics-principles) — mapped to this project in the section above
