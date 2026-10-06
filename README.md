# AFL SuperCoach VIA

<div align="center">
  <img src="https://img.shields.io/github/last-commit/apur27/SuperCoach-VIA">
  <img src="https://img.shields.io/github/contributors/apur27/SuperCoach-VIA">
  <img src="https://img.shields.io/github/stars/apur27/SuperCoach-VIA?style=flat-square">
  <img src="https://img.shields.io/github/forks/apur27/SuperCoach-VIA?style=flat-square">
  <img src="https://img.shields.io/badge/python-3.12-blue">
  <img src="https://img.shields.io/badge/data-audit%20findings%20open-orange">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey">
</div>

---

SuperCoach VIA is a static AFL statistics website with a local Python data pipeline. It provides player and match history, comparisons, rankings, downloads and forecast evaluation. The repository also retains its earlier CSV pipeline and dated football analysis.

**For readers:** browse the site or the linked reports. Check the snapshot date and data status before using a total, ranking or prediction.

**For contributors:** the `scvia` package imports and validates snapshots, builds static releases, and runs deterministic integrity checks. The Astro frontend reads release files. Claude agent definitions and project memories are in [`.claude/agents/`](.claude/agents/).

**[Open the provisional website](https://apur27.github.io/supercoach-via/)** · [Data status](https://apur27.github.io/SuperCoach-VIA/data-status/) · [Publication and rollback](docs/pages-preview.md).

The legacy weekly harness and the new local pipeline coexist. The static preview is published manually. Numeric pipeline switch-over and independent acceptance of the all-player reconciler remain pending.

⭐ **If this project is useful to you, please star the repo.**

---

## Data status

**Checked 3 October 2026: the data has known integrity failures.** Completing an audit does not mean its inputs passed.

| Input or check | Recorded result | Meaning |
|---|---|---|
| September 29 candidate snapshot `3de6597513b5…` | **FAIL** against captured AFL Tables pages, with unresolved evidence | Includes the Grand Final, but has missing appearances, zero/null differences and other discrepancies |
| Legacy CSVs in `data/` | **FAIL** against the same source capture | An older data layer; the audited CSVs lack the Grand Final |
| Hosted preview release `20261003T124334Z-08e8e0eb65d4` | **PASS** from `scvia check-integrity` | Rebuilt from the same candidate; release files agree with their declared inputs. This does not establish full AFL Tables agreement |
| Independent acceptance of the new reconciler | **Pending** | Implementation is still in `work/afltables-reconciliation`; it has not been merged into `main` |

The source audit used pages captured on 1–2 October, with matches scoped through 30 September. Its canonical report SHA-256 is `58eff517a29d30b932087f919a31d36567e8be330301ecff53feb06494d8db28`. The local evidence lives under `var/reconciliations/afltables/2026-10-01-full/`; it is not included in a fresh clone.

**Hall of Fame:** [provisional candidate tables](docs/hall-of-fame/provisional/README.md) are regenerated from the September 29 snapshot at the owner's request. They report snapshot values, not a clean source-audit verdict or official AFL Hall of Fame selections. Missing games, historical statistic coverage and unresolved discrepancies can affect totals and ranks. The older narrative pages retain their original data vintage. Regenerate the provisional tables after the data is corrected.

**Forecasts:** the September 29 candidate reports `unavailable / no_valid_future_fixture`. Archived prediction pages are not current forecasts.

The lowercase `/supercoach-via/` address now opens the canonical `/SuperCoach-VIA/` site through a small redirect. The repository name is unchanged. With JavaScript disabled, the entry page provides a link. Every hosted page displays the provisional notice and requests `noindex` while this snapshot's audit failures remain open.

Read the [integrity checker guide](docs/data-integrity.md) and [reconciliation status and handoff](docs/rewrite/afltables-reconciliation/README.md). The latter records the pending review and the preserved Claude worktree.

## Recorded legacy evaluation

The figures below belong to the legacy R1–R25 evaluation and retained CSV inventory. They were not recomputed for the September 29 candidate or the October source audit.

| Metric | Value | Source |
|---|--:|---|
| AFL history covered | **[data]** 1897–present | `data/matches/` |
| Player performance files | **[data]** 13,364 | `data/player_data/` (one CSV per player, one row per game) |
| Backtest window | **[data]** R1–R25, 2026 | `data/prediction/backtest/` |
| Player-round predictions scored | **[data]** 9,007 | walk-forward backtest |
| Mean absolute error (disposals) | **[data]** 3.961 | player-weighted across all rounds |
| Within 5 disposals | **[data]** 74.3% | player-weighted |
| Within 10 disposals | **[data]** 95.7% | player-weighted |
| Aggregate bias | **[data]** -0.120 | essentially unbiased at population level |

**Plain English:** the model misses a player's next-round disposal count by about four disposals on average — usable signal on a 0–45 range, measured honestly across 9,007 predictions. The known weak spot is the elite tier, where error runs roughly 2.5× the global figure.

---

## How to run it — quick start

Use Python from [`.python-version`](.python-version) and Node from [`.node-version`](.node-version). Install `uv`, then run:

```bash
git clone https://github.com/apur27/SuperCoach-VIA
cd SuperCoach-VIA
uv sync --locked --group dev --group legacy --extra ml
(cd web && npm ci)
uv run scvia doctor
# Start the browser with the bundled DEMO fixture.
(cd web && npm run dev)
```

Astro prints the local URL. The bundled demo contains synthetic data. For a real browser build, set `SCVIA_RELEASE_DIR` to the **absolute path to a release's `public/` directory**, and use the same `SCVIA_PUBLIC_BASE` as that release. An unset release directory selects the demo fixture.

To view an already built site without regenerating its sealed files:

```bash
node web/scripts/serve.mjs --dir /absolute/path/to/release/site --base / --port 4321
```

Use the release's actual base path if it is not `/`. The [operations guide](docs/operations.md) covers import, forecast, build, validation, sealing and preview. The [legacy installation guide](docs/installation.md) describes the older pipeline.

The legacy `scripts/weekly_refresh.sh` performs network refreshes and can commit/push generated outputs. It is an operator workflow, not the browser quick start; its existing gates and [harness rules](CLAUDE.md#6-harness-change-discipline--non-negotiable) still apply.

### Start here — I want to...

| I want to... | Go to | Setup needed |
|---|---|---|
| **Read the archived prediction report** | [docs/afl-predictions-2026.md](docs/afl-predictions-2026.md) | None - browser only; check its date |
| **Browse the no-code fan landing page** | [docs/start-here-no-code.md](docs/start-here-no-code.md) | None - browser only |
| **Understand what this is good for in SuperCoach** | [docs/how-to-use-this-for-supercoach.md](docs/how-to-use-this-for-supercoach.md) | None - browser only |
| **Get the prediction CSV into Google Sheets** | [templates/google-sheets-template.md](templates/google-sheets-template.md) | A free Google account |
| **Read the retained 2026 season hub** | [docs/afl-season-2026.md](docs/afl-season-2026.md) | None - browser only; check its date |
| **See the provisional candidate top 100 and statistical leaders** | [Candidate Hall of Fame tables](docs/hall-of-fame/provisional/README.md) | None - browser only; audit failures remain open |
| **Browse historical Hall of Fame narratives** | [Hall of Fame archive](docs/hall-of-fame.md) | None - browser only; original source dates apply |
| **Look up a footy or data term** | [docs/glossary.md](docs/glossary.md) | None - browser only |
| **See how accurate the model has been (backtest + pre-registered report card)** | [docs/afl-backtest-2026.md](docs/afl-backtest-2026.md) | None - browser only |
| **Run predictions or retrain the model myself** | [docs/installation.md](docs/installation.md) (For Contributors section) | Python, Git, terminal |
| **Get tactical analysis on an AFL team's list and draft picks** | [docs/news/2026-06-17-afl-2026-list-quality-draft-pipeline.md](docs/news/2026-06-17-afl-2026-list-quality-draft-pipeline.md) | None - browser only |
| **Browse archived AFL news and analysis** | [docs/news/README.md](docs/news/README.md) | None - browser only; publication dates apply |
| **AI design patterns — how this maps to a production deployment (RAG, MCP, eval harness, AI Ethics, sovereign deployment)** | [docs/ai-architecture.md](docs/ai-architecture.md) | None - browser only |
| **AI security — risks, controls, prompt injection, data poisoning, governance and secure design** | [docs/ai-architecture.md#ai-security--risks-controls-and-secure-design-in-this-repo](docs/ai-architecture.md#ai-security--risks-controls-and-secure-design-in-this-repo) | None - browser only |
| **Operator's manual — how this specific repo works end-to-end (data inventory, scripts, match lifecycle, live pipeline, runbooks)** | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | None - browser only |
| **Read the Crumb Phase 2 design doc** | [docs/footy-ai-chatbot-phase2.md](docs/footy-ai-chatbot-phase2.md) | None - browser only |

---

## The existing agent workflow

The roles below describe the legacy analysis workflow. Agent definitions in `.claude/agents/` are authoritative for current defaults. Per-task model choices and actual review records determine which model ran; browsing the static app does not launch these agents.

This is the differentiator. Nine agents live in `.claude/agents/`; the tenth (Codex) is an external model queried for outside-the-frame commentary. Each has a bounded role, and together they form a methodology layer that makes every published claim falsifiable against a CSV in this repo.

| # | Agent | Model | Primary role | One-line description |
|---|-------|-------|--------------|----------------------|
| 1 | **Scientist** | Opus | Data, code, model, pipeline | Owns the data layer — EDA, stat verification, prediction code, live pipeline, doc structure. Enforces the CLAUDE.md verification rule. |
| 2 | **FootyStrategy** | Sonnet | Tactical interpretation | Eight-lens coaching council (Conditioner, Tempo Architect, Structuralist, Match-up Tactician, Talent Developer, Innovator, Culture Custodian, List Strategist). Translates Scientist's numbers into coach-grade reads. Never names specific coaches without attribution. |
| 3 | **DataSentinel** | Sonnet | Pre-commit verification gate | Walks every `**[data]**` tag in a draft, confirms it against the source CSV. Flags untagged numbers, coach-name violations, schema violations. Emits machine-readable JSON for a pre-commit hook to consume. |
| 4 | **BriefBuilder** | Sonnet | Brief data-skeleton drafter | Given two teams and a round, auto-populates the data skeleton of a pre-match brief — H2H ledger, season form, model predictions, top-5-per-side tracking list. Leaves `<!-- FOOTYSTRATEGY INSERT -->` placeholders for the interpretation layer. |
| 5 | **Skeptic** | Opus | Adversarial reviewer | Probes tripwire observability, caveat-hierarchy fidelity, and lens-tension smoothing on FootyStrategy drafts. Outputs `PASS / PASS_WITH_CONCERNS / BLOCK`. Never modifies the doc — the author decides what to incorporate. |
| 6 | **Gaffer** | Opus | Delivery lead / editor-in-chief | Delivery Lead / Editor-in-Chief — orchestrates the chain, decides 'ready to ship' on PASS; boss of process, not of truth. Never authors or edits a `**[data]**` number, never overrides a DataSentinel FAIL or Skeptic BLOCK. |
| 7 | **QA** | Sonnet | Quality-assurance gate | Runs the full test suite, validates pipeline output schemas, checks for data regressions, and verifies every mandatory artifact exists and is well-formed. Emits a structured report; a QA FAIL blocks the ship with the same authority as a DataSentinel FAIL. |
| 8 | **Chronicler** | Opus | End-of-run documentation | Invoked after Gaffer ships. Produces the run report for each cycle — what shipped, what the data is saying, pipeline health — and ranks concrete forward-looking expansion recommendations grounded in what already exists. |
| 9 | **Surveyor** | Fable | Strategic advisor & repo diagnostician | Read-only consultant external to the commit chain. Inspects pipeline health, ranks bottlenecks by impact-per-engineering-day, routes every fix to its owning agent. Never ships code or authors a `**[data]**` number. Invoked judiciously — before structural changes, after each sprint, or when the refresh feels slow or fragile. |
| 10 | **Codex** | Selected per task | Engineering and independent review | Supports coding and review tasks. Record the actual model and executed checks for each task; an agent's opinion is not a data-integrity verdict. |

**The publication chain:** BriefBuilder → DataSentinel (pass 1) → FootyStrategy → DataSentinel (pass 2) → Skeptic → QA → Gaffer → Chronicler. Scientist owns the data/code work feeding the chain. See [Gaffer's current definition](.claude/agents/Gaffer.md) for the applicable workflow and gates.

**The data verification contract:** every specific number in any published doc must be tagged `**[data]**` and verified against the actual CSV before commit. CLAUDE.md is the policy; DataSentinel is the gate. Verification reports land in [`docs/sentinel-reports/`](docs/sentinel-reports/).

### The Surveyor — strategic oversight on Fable

The Surveyor is the one agent that never touches the commit path. It runs on **Claude Fable** — a model chosen specifically for long-horizon strategic reasoning and cross-system pattern recognition — and operates as a read-only diagnostician that sits outside the council's weekly execution chain.

**What it does.** The Surveyor inspects the entire repo in one pass: pipeline health, data coverage, test coverage, agent behaviour, documentation currency, process gaps, and prediction-accuracy risk. It ranks every finding by impact-per-engineering-day and routes each fix to its owning agent — Scientist for code and model work, Gaffer for process and harness, FootyStrategy for interpretation, BriefBuilder for data skeletons. It maintains an open-findings ledger at `.claude/agent-memory/Surveyor/survey_open_findings.md` so no issue falls through across sprint boundaries.

**Why Fable, and why judiciously.** Fable is not used routinely in the weekly pipeline — that would waste a model best suited for open-ended strategic synthesis on mechanical CSV verification or brief drafting. It is invoked deliberately: after each sprint ships, before any structural change to the pipeline or data model, or when the weekly refresh feels fragile. Each survey takes one deep pass (hundreds of tool calls, multiple files read end to end) and produces a concrete, prioritised advisory report. The cost is front-loaded; the benefit is that it catches architectural problems — leakage in the feature pipeline, gate logic defects, stale surface docs — before they compound.

**How it has performed.** As of 2026-07-10:

- **Sprint 3 post-mortem (07-09):** Identified a target-leakage defect in S1b — `percentage_time_played` had been added raw to `feature_columns`, producing train/serve skew and a backtest that would have rubber-stamped the error. Fixed by the Scientist before R20 (2026-07-14), the first live exposure.
- **Gate logic defect (07-10):** Found that the stamp gate accepted any PASS at a content hash while ignoring a co-existing FAIL — a hole that had allowed a FAIL-stamped document to ship on main. Escalated to human decision; dustin-martin remediated, DataSentinel upgraded from Haiku to Sonnet (see below) for consistency.
- **S6 structural mis-scope:** Confirmed that the by-position backtest plan (Sprint 4 candidate) is structurally impossible — the lineup CSV has no position field. Re-scoped before a sprint slot was wasted.
- **S3 backfill tracking:** Verified 700 corrupt lineup rows (100% of 2026 data) and kept them on the open-findings ledger across sprints until a network window is available.
- **Anti-pattern ledger:** Added standing rules from each survey — "no unlagged feature may enter `feature_columns`", "a same-hash FAIL is not cleared by a same-hash PASS", "never park a gated doc across days" — so agents don't re-create the same class of error.

**Why DataSentinel moved from Haiku to Sonnet.** DataSentinel's two tasks have very different difficulty profiles. Verifying a `**[data]**` tag against a CSV is mechanical — run the pandas query, compare to the claimed value, pass or fail. Haiku handles this reliably. But scanning prose for *untagged* stat-shaped numbers — detecting that "22 disposals" in a sentence body has no tag, while "Round 22" or "1965" should not be flagged — is open-ended pattern recognition where the model must reason about context. The Surveyor's 07-09 audit caught the evidence: the same document at the same content hash produced PASS at 05:07Z, FAIL at 05:11Z, then PASS at 06:00Z. That non-determinism is Haiku reaching its reliability ceiling on the harder of the two tasks. Sonnet runs this gate with materially better consistency. DataSentinel runs on every commit and on every doc remediation cycle — accuracy here is load-bearing, not cosmetic, and the cost difference is justified by what the gate protects.

**What the Surveyor never does.** It does not commit, does not fix code, does not author or verify `**[data]**` numbers, and does not re-litigate a DataSentinel or Skeptic verdict. Its job is to tell the right agent what to look at — not to look at it for them. Every finding in its report has a named owner and a concrete next step; none of them are open-ended observations.

### The council in plain football terms

- **Scientist** — the stats analyst in the coaches' box who doesn't just answer a question, they go and do the work. Ask "does this player drop off in wet weather?" and they pull the data, run the numbers, draw the chart, then write it up with honest caveats. If the answer is "we can't tell from this data," they say so.
- **FootyStrategy** — a panel of eight assistant coaches, each obsessed with one thing (fitness, structure, match-ups, list management). They hand back a single recommendation, upfront about how sure they are and where they disagreed — and every call comes with a **tripwire**: the specific thing you'd see on the ground that means the plan is wrong.
- **DataSentinel** — the fact-checker at the door of every commit. It refuses to let a stat past if the number does not actually match the CSV it came from.
- **BriefBuilder** — the analyst's apprentice who lays out the bones of next week's match brief (season records, head-to-head ledger, model predictions, players worth tracking) so the senior coaches start from a populated draft, not a blank page.
- **Skeptic** — the devil's advocate the panel keeps in the room. It reads finished briefs and asks the awkward questions ("is that tripwire really observable on the day?", "did you upgrade the call beyond what the data supports?") and refuses to silently rewrite anything — the call stays with the author.
- **Gaffer** — the delivery lead and editor-in-chief who runs the week. They sequence the agents, hold the line on every gate, and only call "ready to ship" once DataSentinel and the Skeptic have signed off. They are boss of process, not of truth: they never touch a `**[data]**` number and never publish around a FAIL or a BLOCK.

### The Crumb — a 13-agent coaching staff

A 13-agent, 6-tier AI coaching staff — a senior coach, line coaches, specialists, analysts, a data steward — that you ask one question and it dispatches the right specialists and merges their answers. Named after the crumber: the small forward who reads where the ball will spill before the pack resolves. At the top sits the Senior Coach, who doesn't crunch numbers personally — they hand the right pieces to the right specialists and pull the answers back into one plan. Everyone has a lane.

**Phase 1** uses Claude Opus (Senior Coach), Claude Sonnet specialists, and a Claude Haiku data steward, invoked through the Claude Code Agent pattern with prompt-based scoping and model-driven tool calls.

**Phase 2** makes the structure load-bearing: agents talk through a validated schema, the Senior Coach is split into a Planner (picks the agents) and a Synthesiser (merges findings, with no data access at all), data reads go through parameterised query templates instead of arbitrary code, low-confidence answers are routed to a human queue, and a nightly eval harness measures citation precision, era-coverage refusal, role isolation, calibration, and falsifiability. The patterns come from João Moura (CrewAI): Planner-Executor split, role-based crew with IAM isolation, supervisor-worker graph with durable state. The football is incidental — the same five changes apply to any multi-agent deployment.

Full spec — build order, sample Planner output, the `FootyFinding` Pydantic envelope, and the local IAM adaptation: [docs/footy-ai-chatbot-phase2.md](docs/footy-ai-chatbot-phase2.md).

---

## The data

The legacy corpus contains **[data]** 13,364 individual player files and season match CSVs beginning in 1897. The newer pipeline stores immutable, content-addressed snapshot fragments and release metadata. These layers have different vintages; source reconciliation found gaps in both.

Historical statistics have different recording periods. A missing value is not automatically a zero, and a career counter can disagree with the appearances actually stored. The provisional reports expose recorded-game denominators and preserve these limitations instead of asserting complete careers.

### The prediction model

The legacy ensemble's recorded evaluation was within 5 disposals **[data]** 74.3% of the time and within 10 **[data]** 95.7% of the time. Those figures belong to the legacy evaluation window below. The candidate's model and fixture eligibility are separate: inspect its release metadata and [model card](docs/model-card.md). The currently reviewed candidate has no valid future fixture and publishes no forecast.

### The weekly fan pack

The repo retains weekly cheat sheets and prediction bundles from earlier runs. Check each bundle's generation time and fixture coverage. This documentation refresh does not run, enable or verify a publication schedule.

### The news section

The [news archive](docs/news/README.md) keeps each article's publication date, methodology and original review status. Its statistical and tactical claims should be read in that historical context.

---

## Legacy architecture reference

This section describes the earlier CSV and council implementation. For the static app and `scvia` package, use the [current operations guide](docs/operations.md) and [rewrite architecture](docs/architecture.md).

Each layer below is small on purpose. The interest is that all of them are present at once.

| Layer | What it is |
|---|---|
| **Data** | 130 years of AFL match and player CSVs — **[data]** 13,364 player performance files (one row per player per game, 1897–present) plus per-season match files. Weekly scrape via `refresh_data.py`. Feature engineering builds rolling-window features per player (3-game, 5-game, season-to-date form) and a one-hot flag for which club the player is facing. The `LeakProofPredictor` enforces a strict temporal cutoff: predicting round N sees only data strictly before round N. |
| **ML inference** | A `VotingRegressor` ensemble of three diverse base learners: `HistGradientBoostingRegressor`, `LightGBM` (GPU-capable, CPU fallback), and `RandomForestRegressor`. Hyperparameters tuned via Optuna's TPE sampler over a 50-trial budget. Post-hoc out-of-fold linear calibration corrects top-end compression. Walk-forward backtest: **[data]** MAE 3.961 across 9,007 player-rounds (R1–R25, 2026). Cross-validation is `GroupKFold` keyed on player ID, so no player appears in both train and validation folds. |
| **LLM reasoning — Scientist** | Claude Opus running a ReAct loop (Reason, Act, Observe, repeat) for 50+ turns on complex tasks. Tool surface: Bash, Read/Write/Edit, WebFetch, Agent subagents. `CLAUDE.md` is the versioned system prompt and policy doc — data-coverage caveats, ranking constants, behavioural constraints, all in source control and diffable. |
| **LLM reasoning — FootyStrategy** | An 8-lens tactical council, each lens produced separately then reconciled. Output is tiered — Settled, Probationary, Contested, Insufficient Evidence — and every Settled or Probationary recommendation must carry a **tripwire**: an explicit observable that would overturn it. Caveats from the Scientist's upstream findings propagate through unchanged; the data tier caps the recommendation tier. |
| **LLM reasoning — extended council** | **DataSentinel** (Sonnet) is a pre-commit verification gate that walks every `**[data]**` tag and emits machine-readable JSON (`PASS \| FAIL` with per-violation detail) for a pre-commit hook. **BriefBuilder** (Sonnet) is a structured-assembly drafter that pulls H2H, season form, model predictions, and a top-5-per-side tracking list. **Skeptic** (Opus) is an adversarial critic that probes tripwire observability, caveat-hierarchy honour, and lens-tension smoothing, then emits `PASS / PASS_WITH_CONCERNS / BLOCK` — never silently modifying the doc. Ship order: DataSentinel first (closes the runtime-enforcement gap on CLAUDE.md), then BriefBuilder, then Skeptic. Full design in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §2.4 and §13. |
| **RAG** | Deterministic retrieval — pandas filters over CSVs. No embedding model, no vector store for structured numeric data, because semantic similarity adds noise where the query maps directly to a structured filter. Hybrid upgrade path documented: pgvector or Qdrant for unstructured commentary, the pandas layer staying authoritative for any numeric claim. |
| **Eval harness** | Walk-forward backtest with strict temporal cutoff. Per-round MAE, RMSE, within-5, within-10, signed bias, and a top-10-player MAE slice that surfaces the worst failure mode. Team-level bias across all 18 teams. `backtest.py --start-round N --end-round N` for incremental runs; output persisted as CSV under `data/prediction/backtest/`. |
| **MCP gateway** | Claude Code's built-in MCP implementation. Tool surface: Bash, Read/Write/Edit, WebFetch/WebSearch, Agent subagents. JSON-schema tool definitions; tool selection is model-driven, no hand-coded routing logic. |
| **Observability** | `git log` as the audit trail — every doc change is an attributable commit with author, timestamp, diff, message. Backtest CSVs are the ML performance history; a regression is visible by diffing two runs. `CLAUDE.md` is version-controlled, so the agent's policy state at any past commit is reconstructable. |

---

## Eval results — recorded legacy run

These retained results describe the legacy evaluation window below. They are not a fresh evaluation of the candidate snapshot, and their date has not been advanced by the documentation update.

Walk-forward backtest, 2026 season, Rounds 1–25. For each round the model is retrained using only data from before that round, predicts every player who played, and is scored against actuals.

| Window | Player-rounds | MAE | Within 5 | Within 10 | Bias |
|---|--:|--:|--:|--:|--:|
| **R1-R25 player-weighted** | **[data]** 9,007 | **[data]** 3.961 | **[data]** 74.3% | **[data]** 95.7% | **[data]** -0.120 |
| Round 1 (hardest) | **[data]** 230 | **[data]** 4.83 | **[data]** 60.4% | **[data]** 92.6% | — |
| Round 13 (best MAE) | **[data]** 320 | **[data]** 3.51 | **[data]** 79.4% | **[data]** 96.9% | — |

**Plain English:** the typical prediction misses by about four disposals. On a per-player range of roughly 0–45 that is usable signal, not a solved problem. Round 1 is hardest because there are no within-season form features before any 2026 game has been played.

**Technical:** the model is essentially unbiased in aggregate. The known failure mode is the elite tier — top-10-player MAE runs ~2.5x the global figure, driven by a residual ceiling effect and context (tag absorption, role rotations) the feature set captures only partially. Team-level signed bias spans **[data]** -0.57 (St Kilda, most under-predicted) to **[data]** +0.34 (Richmond, most over-predicted), with mean absolute team bias **[data]** 0.22 disposals.

Full per-round table (all 25 rounds), team-level breakdown for every club, biggest misses per round, and pre-registered methodology: **[docs/afl-backtest-2026.md](docs/afl-backtest-2026.md)**.

---

## AFL News & Analysis

The articles below are archived analysis. Their publication dates, source windows and original review records apply; the current source audit does not retrospectively certify every claim.

<!-- NEWS-LATEST-START -->
**Latest:** [Dustin Martin — The Storm](docs/news/2026-06-21-dustin-martin-the-storm.md) - Career retrospective on Dustin Martin, his role in Richmond’s premiership era and the absence he leaves behind. *(2026-06-21)*

[AFL 2026–2030: Five-Year Grand Final Strategy — All 18 Clubs](docs/news/2026-06-19-afl-2026-5yr-grand-final-strategy.md) - Club paths to a Grand Final, competitive tiers, structural gaps and recruitment needs, based on a partial-season snapshot. *(2026-06-19)*
<!-- NEWS-LATEST-END -->

→ [All news entries](docs/news/README.md)

---

## AI architecture & security

- [Repository architecture](docs/ARCHITECTURE.md) — how this repo works end-to-end: ten-agent council, data inventory, scripts inventory, match lifecycle, live pipeline, prediction model
- [AI system architecture](docs/ai-architecture.md) — RAG, tool router, eval harness, MCP gateway, sovereign deployment
  - [Australia's AI Ethics Principles — how this project maps to the 8 principles](docs/ai-architecture.md#australias-ai-ethics-principles--how-this-project-maps)
  - [AI security — risks, controls, and secure design in this repo](docs/ai-architecture.md#ai-security--risks-controls-and-secure-design-in-this-repo)
- [Building The Crumb (Phase 1)](docs/footy-ai-chatbot-setup.md) — 13-agent Claude staff, end-to-end build guide
- [The Crumb — Phase 2 design doc](docs/footy-ai-chatbot-phase2.md) — Planner-Executor, parameterised tools, FootyFinding envelope, HITL routing, eval harness
- [How this repo uses Claude](docs/how-this-repo-uses-claude.md) — custom agent design, policy-as-code, multi-agent orchestration

---

## Hall of Fame & all docs

### Hall of Fame

- [Provisional candidate tables](docs/hall-of-fame/provisional/README.md) — refreshed top 100, career leaders and single-season leaders from the named snapshot, with known audit failures
- [AFL Hall of Fame archive](docs/hall-of-fame.md) — retained narratives, captains, coaches and dynasties
- [100 Forgotten Heroes](docs/hall-of-fame-forgotten-heroes.md) — retained player profiles; see [current data status](#data-status) before using their statistics

### For fans (no code)
- [Start here - no code](docs/start-here-no-code.md)
- [How to use this for SuperCoach](docs/how-to-use-this-for-supercoach.md)
- [Glossary](docs/glossary.md)
- [Google Sheets template](templates/google-sheets-template.md)
- [Retained weekly cheat sheet](docs/weekly/round-current-2026.md) — check its recorded round and date

### Retained season reports and captured live data

These pages retain their own generation dates. A documentation refresh does not run the data pipeline or make a captured match snapshot live.
- [AFL insights hub](docs/afl-insights.md)
  - [2026 season hub](docs/afl-season-2026.md)
    - [Team analysis](docs/afl-team-analysis-2026.md)
    - [Finals pathway](docs/afl-finals-2026.md)
    - [Brownlow predictor](docs/afl-brownlow-2026.md)
    - [Player stat leaders](docs/afl-stat-leaders-2026.md)
    - [Archived predictions](docs/afl-predictions-2026.md)
    - [Recorded backtest results](docs/afl-backtest-2026.md)
  - [Retained 5-year team profiles](docs/afl-team-profiles.md)
  - [Coaches strategy corner](docs/coaches-strategy-corner/README.md) - match-by-match tactical briefs built from the data
  - [AFL history - 130 years](docs/afl-history.md)
  - [For the footy expert](docs/footy-expert-guide.md)
  - [For the coaching staff](docs/coaching-guide.md)
  - [AFL 2026 list quality and draft pipeline](docs/news/2026-06-17-afl-2026-list-quality-draft-pipeline.md) - all 18 clubs

### Further reading
- [AI Harness 101: How to Turn a Language Model Into a System That Actually Ships](https://medium.com/@abh1shek/ai-harness-101-how-to-turn-a-language-model-into-a-system-that-actually-ships-b4d0ab5bdf21) *(Medium)* — uses this repo as the worked example: deterministic Python + bounded agent tasks + Git as audit trail
- [How it works: data science deep-dive](docs/data-science.md) - dataset, model, backtest, ranking algorithm, written in three layers from layperson to ML practitioner
- [How predictions work](docs/prediction-model.md) - the model, the backtest framework, the all-time-100 algorithm
- [Using the Scientist agent](docs/scientist-agent.md) - when plain Claude vs the Scientist, the improvement loop
- [Using the FootyStrategy agent](docs/coaching-guide.md#leveraging-the-footystrategy-agent) - tactical brainstorming, list analysis, Scientist x FootyStrategy workflow
- [Quick start](docs/quick-start.md) / [Installation](docs/installation.md) / [Usage](docs/usage.md) / [Troubleshooting](docs/troubleshooting.md)
- [Claude Code setup on Ubuntu](docs/claude-code-setup.md) - install Node.js, Claude Code, Python venv
- [Technical reference](docs/technical-reference.md) - GPU setup, data layout, scripts

### About
- [Roadmap & contributing](docs/roadmap.md)
- [Changelog](CHANGELOG.md)

---

## Why this repo exists

> What this repo is for, in the end, is none of the engineering above — it is this.

This is not a commercial project. It is not affiliated with any gambling service, and nothing here is intended to encourage betting of any kind. The motivation is the game itself - the patterns inside it, the history it carries, and the people it brings together.

It started, honestly, as competitive edge. I have been playing SuperCoach with the same group for over a decade, and this repo exists in no small part because of the arguments about who the better player really was. Somewhere along the line a Sunday-night lineup tweak turned into feature engineering, then into a backtest framework, then into this.

But this repo is also a return gift. To the friends and colleagues who got me up to speed on this game - who explained what a clearance was, why ruck craft matters, how to read a scoreline - and who introduced me to SuperCoach in the first place. You did not have to, and you did. This is, in part, a thank you back.

A specific and heartfelt thank you goes to the families, coaches and community of Cranbourne Junior Football Club, who welcomed my son and trained him in the right spirit of the game. The coaches who give their time freely on cold mornings, the families who stand on the boundary in the rain - these are the people who actually make the game what it is. AFL doesn't exist without them, and a polished dataset of senior careers means very little without remembering where every one of those players came from.

It is also why I think AFL is one of the things that can make Australia genuinely multicultural. Sport breaks boundaries in a way that policy never quite manages to. A new Australian turning up at a junior football club and being welcomed onto a team is not a small thing - it is one of the more honest forms of belonging this country has to offer.

And finally, this work is an homage to the giants of the game - past, present and future. To the players whose careers are quietly recorded in the rows of this dataset, who gave everything on the field, and who made generations of fans care deeply about something together. The numbers in here are theirs. The rest of us are just keeping the ledger.
