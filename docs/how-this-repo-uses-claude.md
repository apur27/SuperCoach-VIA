# How This Repo Uses Claude — A Practitioner's Showcase

> This document explains, in technical depth, how Claude AI and Claude Code are used to build, maintain, and govern the SuperCoach-VIA project. It is written for technical readers — engineers, hiring managers, and collaborators — who want to understand not just *that* Claude is used, but *how* and *at what level of sophistication*.

---

## Overview

SuperCoach-VIA is not a project that uses Claude as a chatbot. It uses Claude as an **active engineering collaborator** operating across four distinct layers:

1. **Custom agent design** — a purpose-built Scientist agent with its own model, memory, methodology rules, and escalation protocol
2. **Policy-as-code** — `CLAUDE.md` as a versioned system prompt that governs every agent in the repo
3. **Mathematical feedback governance** — a formal framework controlling how agent-driven changes propagate through the system
4. **Multi-agent orchestration** — parallel and serial subagent spawning for long-horizon tasks

Each layer is described below with the actual implementation details.

---

## 1. Custom Agent Design: The Scientist

### What it is

The Scientist is a Claude sub-agent defined in `.claude/agents/Scientist.md`. It runs on **Claude Opus** (the highest-capability model) and is invoked with `@"Scientist (agent)"` from within any Claude Code session.

### Why a custom agent rather than plain Claude

Plain Claude Code runs on Sonnet by default — fast, cheap, appropriate for 80% of tasks (file edits, git operations, doc updates, question answering). The Scientist exists because a narrow class of tasks — statistical analysis, model debugging, feature engineering, backtest interpretation — require both a stronger model and a different behavioural contract. Mixing those requirements into plain Claude would either waste tokens on routine tasks or let analytical work slip through without the rigour it requires.

The separation also makes cost legible: every `@"Scientist"` invocation is a deliberate, high-value decision. The docs explicitly tell users to treat it like calling in a senior consultant.

### The agent definition

The Scientist's system prompt encodes:

**A prime directive with a clear priority ordering:**
```
Honest answers over impressive answers.
A null result, reported clearly, beats a positive result built on a leaky pipeline.
Methodology integrity is non-negotiable. Speed and elegance are negotiable.
```

**Three interaction modes with different definitions of "done":**
- *Exploratory* — honest characterisation of what's there, fast
- *Decision-support* — defensible answer with uncertainty quantified
- *Production* — reproducible code, tested on held-out data, data contract written down

**A blast-radius classification system** (LOW / MEDIUM / HIGH) that governs how much rigour and reporting is required before any output is published.

**12 hard rules that are never relaxed**, including:
- Inspect before transforming (print shape, dtypes, missingness)
- No data leakage (train/test, temporal, group)
- Holdout sets are sacred — if you peek, it is burned
- No silent data loss — every filter reports rows-in vs rows-out
- No p-hacking — report all comparisons or correct for multiple testing
- Baselines before complex models

**A structured response contract** — every Scientist output begins with a one-line classification (`[Mode: X] [Type: Y] [Blast: Z]`) and returns exactly five sections: Did / Found / Caveats / Didn't / Assumed. This makes outputs machine-parseable and forces completeness — the "Didn't" section is where a model would normally hide failures.

**An escalation protocol** — the Scientist is explicitly instructed to *stop and wait* when a decision exceeds methodology and enters business or product territory. It does not pick the "obvious" answer to keep moving.

**Persistent cross-session memory** — the Scientist has its own agent-memory directory at `.claude/agent-memory/Scientist/` with a structured `MEMORY.md` index. It saves data quirks, established baselines, approved methodology choices, and reproducibility recipes — so each session builds on prior ones rather than starting cold.

### What this demonstrates

- **Custom agent engineering**: structuring a system prompt as a rigorous methodology contract rather than a generic instruction set
- **Model tiering**: deliberately using Opus only where the capability delta justifies the cost
- **Behavioural specification**: encoding hard rules, soft defaults, and escalation conditions as first-class constructs rather than implicit expectations
- **Memory architecture**: persistent, file-based cross-session memory with an index layer and typed memory categories

---

## 2. Policy-as-Code: CLAUDE.md as Versioned System Prompt

### What it is

`CLAUDE.md` is a Markdown file at the project root that Claude Code automatically injects as a system prompt into every session in this repository. It is version-controlled, diffable, and reviewed like any other source file.

### What it encodes

The current `CLAUDE.md` encodes one critical domain rule: **the data verification requirement**.

```
Before writing any player stat into a document — games played, goals, Brownlow votes,
premierships, career averages — you MUST verify it against the actual data files in this repo.
Do NOT rely on training-data memory or general knowledge for specific numbers.
```

This rule exists because Claude's training data contains AFL statistics — but those statistics can be wrong, stale, or hallucinated at the margins. The repo contains the authoritative source (13,321 player CSVs from 1897–present), so any stat written into a document must come from there.

The rule is non-negotiable and enforced by the system prompt rather than by hoping the model remembers. It includes the exact code pattern to use for verification, the Python venv path, and the explicit fallback (`**[historical record - unverified in data]**`) for stats that genuinely cannot be found.

### Why this matters

`CLAUDE.md` solves the **alignment drift problem** in long-running AI-assisted projects. Without a policy document, agent behaviour is only as consistent as the current session's context — you re-explain the same rules, the agent drifts on edge cases, and you have no audit trail. With `CLAUDE.md`:

- Every session starts with the same policy, regardless of conversation history
- Policy changes are tracked in git with diffs, commit messages, and dates
- The state of the policy at any past commit is reconstructable
- Multiple agents (plain Claude, the Scientist, future agents) all operate under the same governance

This is **prompt engineering as software engineering** — the policy document is a source artifact, not a chat history.

---

## 3. Mathematical Feedback Governance

### The problem it solves

In an AI-assisted repo, feedback arrives continuously: from users, from external reviewers, from automated analysis. Each feedback item has a different scope, criticality, and reversibility profile. Without a governance framework, you either apply everything (risky) or nothing (useless). The right answer is a principled way to classify each feedback by its impact magnitude and decide autonomously vs. escalate.

### The framework

This repo formalises feedback governance as a mathematical model. The state of the system is:

```
S ∈ [0, 1],  S₀ = 1.0  (100% complete)
```

When feedback arrives, its **impact magnitude** is computed as:

```
I(f) = κ · φ(σ) · ψ(r)
```

Where:
- **κ (criticality)** — the tier of feedback: 0.1 (cosmetic), 0.5 (localised correction), 1.0 (cross-cutting), 2.0+ (structural/canonical)
- **φ(σ) = min(1 + log₁₀(1 + σ), 4)** — scope multiplier (log-scaled count of affected artifacts, capped at 4)
- **ψ(r) = 2 − r** — reversibility penalty (irreversible changes double the impact; r=1 means trivially undoable)

The result is **quantized** onto a discrete action grid:

| Raw impact I | Applied delta | Action |
|---|---|---|
| (0, 0.3] | 0.1% | Autonomous |
| (0.3, 0.75] | 0.5% | Autonomous |
| (0.75, 2.0] | 1.0% | Autonomous |
| > 2.0 | **HALT** | Manual approval + blast radius report |

The **hard ceiling** is per-item: no single autonomous change may exceed 2% of the system state. Changes above this threshold trigger a mandatory blast radius report:

```
BLAST RADIUS REPORT
  raw_impact:         I = <value>  (threshold = 2.0)
  drivers:            κ=<criticality>, σ=<scope>, r=<reversibility>
  affected artifacts: [list]
  irreversible ops:   [list]
  rollback plan:      [git ref / backup path / "none - destructive"]
  recommended split:  [decomposition into sub-feedbacks each with I ≤ 2.0]
  approval requested: yes
```

### Applied to a real feedback event

When a detailed 3,000-word repositioning brief arrived (see "The Forge" feedback in the session that built this document), the Scientist decomposed it into 12 atomic sub-feedbacks, computed I for each, and produced:

- **7 autonomous items** (glossary, how-to pages, issue templates, CHANGELOG, templates) — each I < 2.0, applied serially
- **3 Tier B items** requiring explicit approval (README restructure at I=1.95, social preview, scheduled workflow at I=1.82)
- **6 HALT items** (hosted dashboard at I=6.47, player card generator at I=2.85, weekly cheat sheet at I=2.56, model report card at I=2.22, fan pack releases at I=2.67, strategic repositioning thesis at I=8.8) — each with a full blast radius report and a "smallest reversible probe" decomposition

The user approved all HALTs. The Scientist then executed all 15 items, created 22 files, and pushed to `origin/main` in a single commit — with every decision traceable back to its I-value.

### What this demonstrates

- **Control-theory thinking applied to AI governance** — treating agent autonomy as a bounded control variable rather than an on/off switch
- **Formal decomposition of compound feedback** — breaking a strategic brief into atomic items with independent impact profiles
- **Blast radius reporting as a first-class output** — not just "HALT" but "here is exactly what would be affected, why it's above threshold, and here is the smallest safe step that keeps progress moving"
- **Per-item vs per-session ceiling design** — explicit reasoning about which is appropriate and why

---

## 4. Multi-Agent Orchestration

### The pattern

Claude Code supports spawning subagents — separate Claude instances with their own context, tools, and (optionally) models. This repo uses subagent spawning for two purposes:

**Parallelisation**: independent tasks (rebuild Hall of Fame docs, run next-round predictions) run in separate agents simultaneously, reducing wall-clock time.

**Context protection**: a long data-analysis task that produces hundreds of lines of Python output would pollute the main conversation context and increase latency and cost on every subsequent turn. Delegating it to a subagent keeps the main context clean.

### The Scientist as a subagent

The Scientist is implemented as a named subagent (`.claude/agents/Scientist.md`). When invoked via `@"Scientist (agent)"`, Claude Code:

1. Instantiates a new agent context with the Scientist's system prompt prepended
2. Routes tool calls through the same MCP gateway as the parent session
3. Returns a structured result to the parent conversation

The parent agent (plain Claude/Sonnet) acts as **orchestrator**: it receives the task, decides whether the Scientist is warranted, delegates with a fully-specified prompt, and integrates the result. The Scientist acts as **specialist**: it runs the heavy analytical work, reports in its structured contract format, and does not make business decisions.

This is the same pattern used in production multi-agent systems (LangGraph, AutoGen, CrewAI) — role separation, explicit handoffs, structured interfaces between agents.

### Worktree isolation

For large execution tasks (like the Forge feedback execution), the agent runs in an isolated git worktree (`isolation: "worktree"`). This means:
- The agent writes to a temporary branch, not directly to main
- If the agent fails partway, main is unaffected
- Changes are reviewed before being merged
- The worktree is automatically cleaned up if no changes are made

### What this demonstrates

- **Orchestrator/specialist pattern** — the same mental model as production multi-agent frameworks, applied at the Claude Code layer
- **Context budget management** — deliberate delegation to protect main context window from expensive intermediate outputs
- **Structured agent interfaces** — the Scientist's response contract is its API; the orchestrator integrates it without needing to parse free text
- **Worktree isolation for safe execution** — agent changes are staged, not live, until verified

---

## 5. The Full Workflow Loop

End-to-end, the repo operates on this weekly loop:

```
1. Data refresh
   └── AFL results scraped, player CSVs updated

2. Prediction run
   └── python prediction.py
   └── Ensemble VotingRegressor (HGB + LightGBM-GPU + RandomForest)
   └── Optuna 50-trial TPE hyperparameter search
   └── OOF linear calibration on upper tail

3. Backtest
   └── Walk-forward, strict temporal cutoff (LeakProofPredictor)
   └── GroupKFold by player ID (no player in both train and val)
   └── MAE / RMSE / within-5 / within-10 / signed bias / top-10 MAE slice

4. Scientist analysis
   └── @"Scientist (agent)" analyse the backtest and improve the model
   └── Scientist reads CSVs, runs Python, identifies systematic errors
   └── Proposes targeted changes to prediction.py
   └── User reviews, approves, re-runs backtest to verify

5. Doc updates
   └── Plain Claude updates weekly predictions doc, cheat sheet, model report card
   └── Data verification rule enforced: every stat read from CSV before writing

6. Governance check
   └── Any structural change > 2% I-value triggers HALT + blast radius report
   └── User approves or decomposes into safe sub-steps

7. Push to main
   └── Commit with co-author tag (Claude Sonnet 4.6)
   └── CHANGELOG updated
   └── GitHub Actions weekly fan pack runs on schedule
```

Every step has a human in the loop at the right point — not blocking routine work, but required for decisions above the governance threshold.

---

## 6. Key Design Decisions and Trade-offs

### Why Markdown files as agent memory, not a vector store

Agent memory (both the user-level memory in `~/.claude/projects/` and the Scientist's project memory in `.claude/agent-memory/Scientist/`) is implemented as plain Markdown files with a `MEMORY.md` index. This is a deliberate choice:

- **Auditability**: git tracks every memory change with timestamp, diff, and author
- **Simplicity**: no embedding model, no vector database, no retrieval latency
- **Editability**: a human can read, correct, or delete any memory entry directly
- **Appropriateness**: the memory corpus is small and structured; semantic similarity adds noise, not signal, compared to a simple index lookup

The trade-off is that memory is not semantically searchable — but for a single-project agent with a bounded memory corpus, that is the right call. Vector search becomes appropriate only when the memory corpus grows large enough that the index becomes unwieldy or when unstructured text (commentary, notes) enters the corpus.

### Why CLAUDE.md rather than a per-session system prompt

Encoding policy in `CLAUDE.md` rather than in each session's opening message:
- Survives conversation compaction (long sessions lose early context)
- Is version-controlled and diffable
- Applies automatically to all agents in the repo, including the Scientist
- Puts policy authorship in the hands of the engineer, not the operator

The downside is that `CLAUDE.md` is always loaded regardless of task type — a pure git operation does not need the AFL data verification rule. For this repo, the policy is small enough that the overhead is negligible. At scale, a tiered policy approach (global `CLAUDE.md` + task-specific overrides) would be more appropriate.

### Why the feedback governance framework uses a hard ceiling rather than a soft guideline

A soft guideline ("prefer small changes") relies on the agent's judgement about what "small" means — which varies with context, framing, and session state. A hard ceiling with a mathematical I-value creates:

- **Consistency**: the same feedback always lands in the same tier regardless of how it is phrased
- **Legibility**: "I = 2.56, HALT" is a fact; "this feels big" is an opinion
- **Escalation without ambiguity**: the blast radius report is triggered by a threshold crossing, not by the agent's comfort level

The ceiling is calibrated (κ, φ, ψ are design choices, not derivations), and the calibration is documented. A future improvement would fit the calibration to a labelled set of historical feedback events.

---

## 7. Observability and Audit Trail

| Layer | Mechanism |
|---|---|
| Code changes | Git commit history with co-author tags (human + Claude Sonnet 4.6) |
| Agent policy | `CLAUDE.md` version-controlled in git |
| Scientist memory | `.claude/agent-memory/Scientist/*.md` in git |
| Model performance | Backtest CSVs in `data/prediction/backtest/` (one per run) |
| Doc accuracy | `docs/model-report-card.md` — pre-registered methodology, per-round actuals |
| Feedback governance | HALT events and blast radius reports in session logs |
| Weekly automation | GitHub Actions run log (`weekly-fan-pack.yml`) |

What is currently missing (and known):
- Structured LLM trace logging (prompt, tool calls, latency, token cost per turn)
- Per-session token budget enforcement
- Automated LLM eval (RAGAS for faithfulness, DeepEval for hallucination rate)
- Online eval loop (automatic scoring of predictions against post-round actuals)

These gaps are documented in `docs/ai-architecture.md` under "What I'd do differently in a sovereign deployment."

---

## Summary: What This Demonstrates

| Capability | Implementation |
|---|---|
| Custom agent engineering | `.claude/agents/Scientist.md` — methodology contract, hard rules, response contract |
| Model tiering (Sonnet / Opus) | Default Sonnet for routine work; Opus only for the Scientist |
| Policy-as-code | `CLAUDE.md` as versioned, version-controlled system prompt |
| Mathematical feedback governance | I = κ·φ(σ)·ψ(r) with hard 2% ceiling and blast radius reports |
| Multi-agent orchestration | Orchestrator (plain Claude) + specialist (Scientist) with structured interfaces |
| Persistent cross-session memory | File-based memory with typed categories and MEMORY.md index |
| Worktree isolation | Agent execution in isolated git branch, merged on verification |
| Walk-forward ML validation | `LeakProofPredictor` + GroupKFold — production-grade temporal holdout |
| Human-in-the-loop gates | Autonomous below threshold, mandatory approval above |
| Full audit trail | Git history + backtest CSVs + model report card + CHANGELOG |

---

## Further Reading

- [`docs/ai-architecture.md`](ai-architecture.md) — the full seven-layer architecture with ML practitioner notes and sovereign deployment design
- [`docs/scientist-agent.md`](scientist-agent.md) — day-to-day playbook for working with the Scientist
- [`CLAUDE.md`](../CLAUDE.md) — the live policy document governing all agents in this repo
- [`.claude/agents/Scientist.md`](../.claude/agents/Scientist.md) — the full Scientist agent definition
- [`docs/model-report-card.md`](model-report-card.md) — pre-registered methodology and per-round accuracy results
