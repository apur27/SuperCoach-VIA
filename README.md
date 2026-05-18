# SuperCoach-VIA

![AFL SuperCoach VIA Banner](/assets/banner.svg)

SuperCoach-VIA is an AFL data, modelling, and strategy analysis repo. It contains 130 years of match results (1897-2026), 26,000+ per-player per-game CSVs, a three-model LightGBM/HGB/RF disposal-prediction ensemble with walk-forward backtesting, a 90-second FanFooty live-match pipeline that auto-commits per-poll analysis, and a structured publishing workflow for pre-match briefs, live reads, and post-match analysis - all gated by a six-agent council that enforces data verification at every commit.

---

## The six-agent council

Five agents live in `.claude/agents/`; the sixth (Codex) is an external model queried for outside-the-frame commentary. Each agent has a clearly bounded role; together they form a methodology layer that makes every published claim falsifiable against a CSV in this repo.

| # | Agent | Model | Primary role | One-line description |
|---|-------|-------|--------------|----------------------|
| 1 | **Scientist** | Opus | Data, code, model, pipeline | Owns the data layer - EDA, stat verification, prediction code, live pipeline, doc structure. Enforces the CLAUDE.md verification rule. |
| 2 | **FootyStrategy** | Opus | Tactical interpretation | Eight-lens coaching council (Conditioner, Tempo Architect, Structuralist, Match-up Tactician, Talent Developer, Innovator, Culture Custodian, List Strategist). Translates Scientist's numbers into coach-grade reads. Never names specific coaches without attribution. |
| 3 | **DataSentinel** | Haiku | Pre-commit verification gate | Walks every `**[data]**` tag in a draft, confirms it against the source CSV. Flags untagged numbers, coach-name violations, schema violations. Emits machine-readable JSON for a pre-commit hook to consume. |
| 4 | **BriefBuilder** | Sonnet | Brief data-skeleton drafter | Given two teams and a round, auto-populates the data skeleton of a pre-match brief - H2H ledger, season form, model predictions, top-5-per-side tracking list. Leaves `<!-- FOOTYSTRATEGY INSERT -->` placeholders for the interpretation layer. |
| 5 | **Skeptic** | Opus | Adversarial reviewer | Probes tripwire observability, caveat-hierarchy fidelity, and lens-tension smoothing on FootyStrategy drafts. Outputs `PASS / PASS_WITH_CONCERNS / BLOCK`. Never modifies the doc - the author decides what to incorporate. |
| 6 | **Codex (GPT-5.4)** | External | Outside-the-frame commentary | Queried for views from outside this repo's data frame. All Codex outputs are attributed explicitly as external commentary and cross-checked against repo data where possible. |

The pre-match flow is BriefBuilder -> Scientist -> FootyStrategy -> DataSentinel -> (optionally Skeptic). Full architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §2 and §6.

---

## What this repo produces

Three concrete artefacts come out of the agent loop, all version-controlled and grounded in the data layer.

- **Pre-match briefs** combining a data spine with a tactical interpretation layer. Example: [Richmond vs St Kilda, Round 11, 2026](docs/coaches-strategy-corner/richmond-vs-stkilda-round-11-2026.md).
- **Live match analysis** updated every 90 seconds during games, auto-routed to per-quarter docs and auto-committed. Example: [Richmond vs St Kilda R11 half-time live](docs/coaches-strategy-corner/richmond-vs-stkilda-round-11-2026-half-time-live.md).
- **Published news and opinion** grounded in data - every specific number reproducible from a CSV. Example: [Who should be the next Carlton coach?](docs/news/2026-05-15-carlton-next-coach.md).

---

## The data verification contract

Every specific number written into any doc - games played, goals, Brownlow votes, premierships, career averages, model outputs - must be tagged `**[data]**` and verified against the actual data files in this repo before commit. Memory and general knowledge are not sources. The rule is recorded in [`CLAUDE.md`](CLAUDE.md), loaded into every Claude session, and applies to every agent equally. **DataSentinel** is the runtime enforcement: it walks every `**[data]**` tag in a draft, confirms the cited number against the source CSV, and emits JSON that a pre-commit hook consumes. CLAUDE.md is the policy; DataSentinel is the gate. Verification reports land in [`docs/sentinel-reports/`](docs/sentinel-reports/).

---

## What's in the data

All paths relative to `data/`. Full inventory in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §3.

| Layer | Path | Coverage | Notes |
|-------|------|----------|-------|
| Match results | `matches/matches_<year>.csv` | 1897-2026, 130 files, continuous | Quarter-by-quarter scoring, attendance, venue |
| Per-player per-game | `player_data/<surname>_<first>_<DOB>_performance_details.csv` | ~13,300 players, 26,644 CSVs | 30 columns; no `position` field. Stat coverage: goals from 1897, disposals from 1965, tackles from 1987, clearances/I50s from 1998 |
| Prediction model output | `prediction/next_round_<N>_prediction_<ts>.csv` | R15-R25 2025, R2-R11 2026 | `player, team, predicted_disposals` |
| Backtest results | `prediction/backtest/backtest_summary_<ts>.csv` | Per (year, round) | MAE, RMSE, bias, within-5, within-10 |
| Live snapshots | `live_snapshots/<gameid>_<ts>_<status>.json` | Per-90-second polls during games | 65-column player schema (3 fields unreliable - see §9.1 of ARCHITECTURE) |
| All-time rankings | `top100/all_time_top_100.csv` and `top100/yearly/year_<YYYY>.csv` | 130 per-season top-100s + canonical all-time | Score methodology in `.claude/agent-memory/Scientist/all_time_formula.md` |

---

## Key docs

| Doc | What it is |
|-----|------------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | End-to-end repo map: agent council, data inventory, scripts inventory, match lifecycle, live pipeline, prediction model, known limitations |
| [`docs/coaches-strategy-corner/README.md`](docs/coaches-strategy-corner/README.md) | House rules for per-game tactical briefs; live-pipeline operator recipe |
| [`docs/news/README.md`](docs/news/README.md) | House rules for data-grounded long-form journalism; `**[data]**` / `**[historical record]**` tag glossary |
| [`docs/sentinel-reports/`](docs/sentinel-reports/) | DataSentinel verification reports per published doc |
| [`CLAUDE.md`](CLAUDE.md) | Operational policy - the data-verification rule loaded into every session |

---

## How to run things

All commands assume the project venv at `/home/abhi/sourceCode/python/coding/.venv/bin/python`. Full setup in [`docs/installation.md`](docs/installation.md).

```bash
# Start the live pipeline for a fixture (exits on Full Time)
/home/abhi/sourceCode/python/coding/.venv/bin/python scripts/live_analysis_pipeline.py <gameid>

# Run the prediction model (auto-detects next round; CPU ~30-60 min)
/home/abhi/sourceCode/python/coding/.venv/bin/python prediction.py

# Run the live-pipeline smoke test (post-R11 hardening regressions)
/home/abhi/sourceCode/python/coding/.venv/bin/python scripts/smoke_test_live_pipeline.py

# Full weekly refresh: scrape -> rank -> predict -> backtest -> docs
bash refresh_and_rank.sh
```

Per-match operator detail (gameid lookup, `DOC_BASE` / `KEY_PLAYERS` config, log-line meanings) is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §10.

---

## House rules

- **Verify before tagging.** No specific player stat, no model output, no historical figure goes into a doc without being read from a CSV first and tagged `**[data]**`. DataSentinel will block the commit if the tag is missing or wrong. Pre-1965 records are incomplete - use `**[historical record - unverified in data]**` rather than inventing a number.
- **No causal claims from associational data.** A team's H2H record, a player's good run against an opponent, a stat correlation - these are flags worth investigating, not causes. FootyStrategy's `Caveat propagation` line and Scientist's response contract carry this through to every brief.
- **No betting tips.** This repo is not affiliated with any gambling service and nothing here is intended to encourage betting. Model predictions are decision support, not picks.
- **No specific coach appointments from analysis alone.** FootyStrategy reasons in archetypes and tactical principles. Specific names - "X should coach Y" - require explicit attribution to a source (Codex external read, a public report, etc.), never a synthesised "the data says hire X" claim.
