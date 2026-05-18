# SuperCoach-VIA — Architecture

> [← Back to main README](../README.md)
>
> **Status:** Reference document. Describes how the repository is organised, who (which agent) is responsible for which layer, what data and code lives where, and how a match flows from pre-game preview to full-time verdict.
>
> Where this document and CLAUDE.md disagree, CLAUDE.md wins — it is the operational policy and is loaded into every Claude session. This document is the map; CLAUDE.md is the rule book.

---

## Table of contents

1. [Repository purpose and overview](#1-repository-purpose-and-overview)
2. [The two-agent model](#2-the-two-agent-model-scientist-and-footystrategy)
3. [Data inventory](#3-data-inventory)
4. [Code and scripts inventory](#4-code-and-scripts-inventory)
5. [Docs inventory](#5-docs-inventory)
6. [Workflow — the match lifecycle](#6-workflow--the-match-lifecycle)
7. [Live pipeline architecture](#7-live-pipeline-architecture)
8. [Prediction model architecture](#8-prediction-model-architecture)
9. [Known limitations and caveats](#9-known-limitations-and-caveats)
10. [How to run things](#10-how-to-run-things)
11. [Recent problems encountered (and fixes shipped)](#11-recent-problems-encountered-and-fixes-shipped)
12. [Planned improvements](#12-planned-improvements)
13. [Agent guardrails and sequencing](#13-agent-guardrails-and-sequencing)

---

## 1. Repository purpose and overview

SuperCoach-VIA is a working AFL (Australian Football League) analytics platform that does four things from one repository:

1. **Stores the history.** Every AFL match since 1897 and a per-player per-game performance row for every player in the league, kept current weekly. 130 seasons of match files, ~13,300+ player files, plus team lineups, fixtures, and an all-time top-100 ranking.
2. **Predicts the next round.** A walk-forward, leak-proof disposals model (`prediction.py`) that emits a per-player prediction CSV for the next round of the current season. A backtest harness (`backtest.py`) re-runs the model round-by-round under strict temporal cutoffs to measure honest MAE.
3. **Writes coach-grade tactical briefs.** Pre-match briefs in `docs/coaches-strategy-corner/` and grounded AFL journalism in `docs/news/` — every specific number tagged `**[data]**` and reproducible from a CSV in the repo.
4. **Reads matches live.** A 90-second polling pipeline (`scripts/live_analysis_pipeline.py`) that pulls the FanFooty live feed, writes per-quarter analysis blocks, and auto-commits to git so every poll is a reviewable snapshot.

The repo is the substrate; two LLM agents (Scientist and FootyStrategy) operate on it. The data layer makes their work falsifiable — every claim in a published doc points at a row in a CSV.

### Top-level layout

```
SuperCoach-VIA/
├── CLAUDE.md                   Operational policy: data-verification rule
├── README.md                   Fan + engineer landing page
├── CHANGELOG.md                Versioned project changes
├── CONTRIBUTING.md             How to extend the repo
├── prediction.py               Main disposal-prediction model (3-model ensemble)
├── backtest.py                 Walk-forward backtest harness
├── refresh_data.py             Targeted weekly scrape (matches + active players)
├── refresh_readme.py           Re-renders auto-generated doc sections
├── refresh_and_rank.sh         Master cron driver (scrape → rank → predict → docs)
├── game_scraper.py             afltables match scraper
├── player_scraper.py           afltables per-player scraper
├── top_players_comprehensive.py  Recomputes all-time top-100 ranking
├── update_team_analysis.py     Writes team-style sections into season + profile docs
├── generate_readme_charts.py   Era + team chart generators
├── era_based_statistical_analysis.py  Historical era comparisons
├── analysis.py / charts.py / bar_chart.py   Ad-hoc EDA + chart scripts
├── prediction_accuracy.py      Standalone one-round prediction-vs-actual scorer
├── prediction_cpu.py           Explicit CPU-only fallback predictor
├── gpu_disposal_prediction_old.py  Deprecated cudf/cupy predecessor
├── cuDF_test.py / testGPU.py   GPU smoke tests
├── helper_functions.py / main.py   Thin glue
├── scripts/                    Operational scripts (see §4)
├── data/                       All data assets (see §3)
├── docs/                       All documentation (see §5)
├── assets/                     Banner, social card, generated PNG charts
├── charts/                     Older static chart PNGs from era analysis
├── templates/                  google-sheets-template.md
├── models/                     (currently empty placeholder)
├── tests/unit/                 Unit-test scaffold
├── scratch/                    Three diagnostic scripts (not for prod)
├── .claude/                    Agent definitions + persistent memory
└── .github/                    CI workflows + issue templates
```

---

## 2. The two-agent model — Scientist and FootyStrategy

> **Note:** the original architecture rested on two agents — Scientist (methodology) and FootyStrategy (interpretation) — described in §2.1 to §2.3. §2.4 introduces the extended council (DataSentinel, BriefBuilder, Skeptic) that wraps a production loop around them. Together the five agents are the "agent layer" referenced elsewhere in this document.

Two LLM agents are defined under `.claude/agents/`, each with its own persistent memory under `.claude/agent-memory/`. They are invoked from inside Claude Code (`@"Scientist (agent)"` / `@"FootyStrategy (agent)"`) and operate as a methodology layer (Scientist) and an interpretation layer (FootyStrategy) on top of the same data.

### 2.1 Scientist

**File:** `.claude/agents/Scientist.md` (~840 lines)
**Memory:** `.claude/agent-memory/Scientist/` (MEMORY.md index + 10 topic files)
**Model:** Opus.

**Operational responsibilities (this repo):**

- All data work: EDA, stat verification, feature engineering, statistical testing.
- Writing and maintaining the prediction code (`prediction.py`, `backtest.py`, the scrapers).
- Operating the live polling pipeline (`scripts/live_analysis_pipeline.py`) and writing per-poll structured stats blocks.
- Producing the **data layer** of every news article and strategy brief: tables, season records, career stats, H2H ledgers — each number tagged `**[data]**` and citing the source file.
- Doc structure itself: the marker-driven auto-generated sections (`<!-- YEAR-TEAM-ANALYSIS-START -->` etc.) are written by Scientist-owned code (`update_team_analysis.py`, `refresh_readme.py`).
- Enforcing reproducibility: explicit seeds, GroupKFold by player, pinned versions, the row-count-at-every-filter discipline from its prompt.

Methodology contract (hard rules, response contract, escalation protocol) is described in [`ai-architecture.md` §3](ai-architecture.md#3-llm-reasoning-layer---the-scientist-agent) and [`how-this-repo-uses-claude.md` §1](how-this-repo-uses-claude.md#1-custom-agent-design-the-scientist). The full agent prompt is the source of truth in `.claude/agents/Scientist.md`.

### 2.2 FootyStrategy

**File:** `.claude/agents/FootyStrategy.md` (~810 lines)
**Memory:** `.claude/agent-memory/FootyStrategy/` (MEMORY.md index + 3 topic files)
**Model:** Opus.

**Operational responsibilities (this repo):**

- Tactical interpretation: converting Scientist's findings into football-language recommendations a coaches' panel could defend.
- The **interpretation layer** of news articles and strategy briefs: what the numbers mean structurally, what a coach typically does about it, what the cultural read is. FootyStrategy's prose sits between Scientist's tables.
- Quarter-break and post-match analyst commentary on live docs (after Scientist's automated blocks fire).
- Multi-game pattern recognition — it owns the `recurring_tensions.md` memory that records when council lenses systematically disagree on this list's questions.
- Coach-anonymity enforcement (see `coach_anonymity_lint.md`).

The 8-lens council methodology (Conditioner, Tempo Architect, Structuralist, Match-up Tactician, Talent Developer, Innovator, Culture Custodian, List Strategist), the confidence tiers, the tripwire rule, and the output envelope are described in [`ai-architecture.md` §3b](ai-architecture.md#3b-footystrategy---tactical-interpretation-agent) and the operator playbook is in [`coaching-guide.md` §"The eight-lens council"](coaching-guide.md#the-eight-lens-council---how-footystrategy-thinks). The full agent prompt is the source of truth in `.claude/agents/FootyStrategy.md`.

### 2.3 How they collaborate

The split is visible in the published docs.

| Concern | Scientist | FootyStrategy |
|---|---|---|
| Pull a season record from `data/matches/matches_2026.csv` | Yes | No |
| Tag a specific stat `**[data]**` | Yes | No (it inherits the tag from Scientist's row) |
| Decide what the stat *means* tactically | No | Yes |
| Run the prediction model | Yes | No |
| Tier a recommendation Settled / Probationary / Contested | No | Yes |
| Write the polling-loop block at half-time | Yes (automated) | Yes (analyst commentary block) |
| Add a name to a tactical brief | Player names: yes. Coach names: no. | Player names: yes. Coach names: never. |
| Author a backtest run | Yes | No |

The convention used across every news entry and strategy brief:

1. **Scientist writes the data layer first.** Tables, ranks, H2H records, career averages. Each verified figure carries the inline tag `**[data]**`. Numbers from public record but not in the repo carry `**[historical record]**`. The draft contains explicit placeholders like `<!-- FOOTYSTRATEGY INSERT: ... -->`.
2. **FootyStrategy fills the placeholders.** Tactical reads, structural reasoning, what a coach should do about it. The same `**[data]**` tags propagate through — FootyStrategy does not strip or add them.
3. **The two-layer file is reviewed once** for tone consistency and committed. Both layers are visible in the final doc; the data and the interpretation are not blended.

This is the pattern used in `docs/coaches-strategy-corner/` (pre-match briefs and post-mortems) and `docs/news/` (long-form journalism). The README for each directory documents the rule.

### 2.4 The extended agent council

Scientist and FootyStrategy carry the analytical and tactical layers. Three additional agents wrap the production loop around them — a verifier in front of every commit, a structured-assembly drafter for the pre-match brief, and an adversarial critic that probes finished work before it ships. Each closes a specific gap the Council identified in the original two-agent design.

#### DataSentinel (Haiku)

**File:** `.claude/agents/DataSentinel.md`.
**Model:** Haiku.
**Role:** pre-commit verification gate.

**What it does:** walks every `**[data]**` tag in a doc and confirms the cited number against the source CSV named in the methodology paragraph. Flags any number that is untagged, any tag whose source file is missing or mis-cited, any coach-name violation (FootyStrategy rule), and any FanFooty schema violation (e.g. a `goals` figure pulled from per-player snapshot rather than afltables, given §9.1).

**Output:** machine-readable JSON — `{status: PASS|FAIL, violations: [{kind, line, expected, actual}]}` — so a pre-commit hook can consume it without LLM-parsing prose.

**When it runs:** at every brief, news article, and post-mortem before commit. High-frequency, latency-sensitive, low-judgement — exactly the shape Haiku is for.

**Why Haiku:** mechanical comparison task. No interpretation, no tactical reasoning, no methodology trade-offs. The job is "is this number in this CSV." Sonnet and Opus would be over-spec and over-priced for it.

#### BriefBuilder (Sonnet)

**File:** `.claude/agents/BriefBuilder.md`.
**Model:** Sonnet.
**Role:** auto-populates the data skeleton of a pre-match brief.

**What it does:** given two team names and a round, pulls the H2H ledger from `data/matches/`, the season form from per-player CSVs, the model predictions from `data/prediction/`, and assembles a top-5-per-side tracking list. Writes `**[data]**` tags with source-file annotations into the draft. Leaves `<!-- FOOTYSTRATEGY INSERT -->` placeholders for the interpretation layer to fill.

**What it does not do:** decide the tier, choose the tripwires, name structural reads, write any prose interpretation. Those remain FootyStrategy's responsibility. BriefBuilder produces a tabular spine — not a finished brief.

**Gating:** BriefBuilder output goes through DataSentinel like every other doc. It cannot self-certify the numbers it tagged.

**Why Sonnet:** structured assembly with surfacing judgement. Choosing which 5 of 25 past H2Hs to surface is judgement (recency, structural similarity, marquee context); choosing which players belong in the top-5 tracking list is judgement (form vs role vs matchup vs availability). Not deep analytical novelty — that's Scientist's territory — but more than mechanical string substitution. Sonnet is the right rung.

#### Skeptic (Opus)

**File:** `.claude/agents/Skeptic.md`.
**Model:** Opus.
**Role:** adversarial review of FootyStrategy-authored drafts before commit.

**What it probes:**

- **Tripwire observability.** Every Settled or Probationary recommendation must include a tripwire. Skeptic asks: is the named observable actually computable from this repo's data schema? An inside-50-differential tripwire is currently unobservable (§9.2) — Skeptic flags it.
- **Caveat hierarchy honour.** Did FootyStrategy upgrade the tier above what Scientist's upstream tag supports? An associational `[Blast: LOW]` Scientist finding cannot become a Settled tactical recommendation. Skeptic checks the caveat propagation line against the upstream data layer.
- **Lens-tension smoothing.** Did the brief paper over genuine disagreement between coaching lenses? If the Conditioner and the Tempo Architect would reach opposite conclusions on the same evidence, the brief must surface the tension, not blend it.

**Output:** structured `PASS / PASS_WITH_CONCERNS / BLOCK` verdict with a per-concern critique. Skeptic never silently modifies the document. The author decides what to incorporate.

**Why Opus:** must reason adversarially across both the methodology layer (does this tripwire honour upstream caveat?) and the tactical layer (would these lenses really agree?). Subtle violations — a tier upgrade hidden inside a footnote, a tripwire that is observable in theory but not in this repo's snapshot schema — require the strongest model. Haiku and Sonnet will miss them.

### 2.5 The non-negotiable CLAUDE.md rule

`CLAUDE.md` is the operational policy loaded into every session. Its single load-bearing rule:

> Before writing any player stat into a document — games played, goals, Brownlow votes, premierships, career averages — you MUST verify it against the actual data files in this repo. Do NOT rely on training-data memory or general knowledge for specific numbers.

This binds all five agents equally. Scientist enforces it through its inspect-before-transforming discipline; FootyStrategy enforces it by refusing to overwrite Scientist-tagged numbers; BriefBuilder must verify before it tags; Skeptic flags any unverified number it spots; DataSentinel is the runtime mechanism that actually closes the loop — CLAUDE.md is policy, DataSentinel is the enforcement. If the player's data file is missing or the stat is genuinely unavailable (e.g. pre-1965), the claim is tagged `**[historical record — unverified in data]**` rather than fabricated.

---

## 3. Data inventory

All paths relative to `data/` unless stated.

### 3.1 `data/matches/` — match results by year

**Files:** 130 CSVs, one per AFL season, `matches_1897.csv` through `matches_2026.csv`.
**Total size:** ~1.9 MB.
**Source:** scraped from afltables.com by `game_scraper.py` and refreshed weekly through `refresh_data.py`.

**Schema** (one row per match):

| Column | Type | Notes |
|---|---|---|
| `round_num` | int | Round number; finals codes vary by era |
| `venue` | string | Free text, e.g. "M.C.G.", "Docklands", "Brunswick St" |
| `date` | string | "YYYY-MM-DD HH:MM" |
| `year` | int | |
| `attendance` | int | May be missing for older games |
| `team_1_team_name` / `team_2_team_name` | string | Home / away by source convention |
| `team_X_q1_goals` / `team_X_q1_behinds` | int | Quarter-by-quarter scoring through Q3 |
| `team_X_final_goals` / `team_X_final_behinds` | int | Full-time score |

**Coverage:** 1897–present, continuous. Used by every team-style aggregation, H2H ledger, era analysis, and chart that needs scoreboard data.

### 3.2 `data/player_data/` — per-player per-game CSVs

**Files:** 26,644 CSVs (~13,322 players × 2 files each).
**Total size:** ~145 MB.
**Naming convention:** `<surname>_<firstname>_<DDMMYYYY>_performance_details.csv` and `<surname>_<firstname>_<DDMMYYYY>_personal_details.csv`. The 8-digit date is the player's date of birth.

**`*_performance_details.csv` — one row per game played, 30 columns:**

```
team, year, games_played, opponent, round, result, jersey_num,
kicks, marks, handballs, disposals,
goals, behinds, hit_outs, tackles,
rebound_50s, inside_50s, clearances, clangers,
free_kicks_for, free_kicks_against, brownlow_votes,
contested_possessions, uncontested_possessions, contested_marks,
marks_inside_50, one_percenters, bounces, goal_assist,
percentage_of_game_played, date
```

**`*_personal_details.csv` — single row per player, 6 columns:**

```
first_name, last_name, born_date, debut_date, height, weight
```

**There is no `position` column** in either file. Any per-position analysis requires an external position source. See `.claude/agent-memory/Scientist/data_no_position.md`.

**Coverage by stat (see `.claude/agent-memory/Scientist/data_stat_coverage_eras.md`):**

| Stat | First reliable year | Note |
|---|---|---|
| `goals`, `behinds` | 1897 | Full history |
| `kicks`, `marks`, `handballs`, `disposals` | 1965 | No AFL stats kept before |
| `hit_outs` | 1966 | A 2016 → 2017 jump is a recording change, not on-field |
| `tackles` | 1987 | |
| `clearances`, `inside_50s` | 1998 | |
| `contested_possessions`, `uncontested_possessions` | 1999 | |

**Used by:** `prediction.py` (training and prediction), `backtest.py` (walk-forward), `top_players_comprehensive.py` (all-time ranking), `update_team_analysis.py` (team-style aggregation), `era_based_statistical_analysis.py`, every news article that cites a career stat, every strategy brief that cites a per-game average.

### 3.3 `data/live_snapshots/` — FanFooty live feed captures

**Files:** 254 files for two games observed to date (`9781` = Richmond vs Adelaide R9 2026; `9789` = St Kilda vs Richmond R11 2026). Paired JSON + CSV per poll.
**Total size:** ~12 MB.
**Source:** `scripts/fetch_live_match.py` pulls `https://www.fanfooty.com.au/live/<gameid>.txt` every 90 seconds during a live game.

**Naming:** `<gameid>_<YYYYMMDD>_<HHMM>_<status>.json` and `<gameid>_<YYYYMMDD>_<HHMM>_players.csv`. The `<status>` segment is one of `q1-1-54`, `q2-...`, `half-time`, `q3-...`, `q4-...`, `final-siren`, `full-time` — derived from the FanFooty status string.

**JSON snapshot structure:**

```
{
  "gameid": "9789",
  "fetched_at_utc": "...",
  "header":     { home_team_full, away_team_full, round, home_score, away_score, status, raw, ... },
  "meta":       { date, year, time, venue, weather, temp_c, raw },
  "commentary": [ { quarter, time, text }, ... ],   // m0nty stream
  "chat":       [ ... ],                            // coach chat stream
  "players":    [ { 65 columns per row }, ... ]
}
```

**Player-row schema (65 columns, defined in `scripts/fetch_live_match.py`):**

Reliable: `player_id, first_name, surname, team, af, sc, proj_af, proj_low, proj_sc, kicks, handballs, marks, tackles, hitouts, frees_for, frees_against, status, position, jumper, de_pct, tog_pct, af_q1/q2/q3/q4, sc_q1/q2/q3/q4`.

**Unreliable** (cross-checked against afltables and found to misindex): `goals` (col15), `behinds` (col16), `clangers` (col39). See `.claude/agent-memory/Scientist/snapshot_data_quality.md`. For goals/behinds/clangers, use afltables or the header scoreline.

**Not in the schema** (cannot be computed from FanFooty alone): `inside_50s`, `clearances`, `contested_possessions`. The live pipeline uses kick-share as a territory proxy and labels it as such.

**Two sentries run on every fetch:** every player row must have exactly 65 columns; per-player Q1+Q2+Q3+Q4 AF must equal total AF. Failure indicates feed schema drift.

### 3.4 `data/prediction/` — model output

**Files (root):** `next_round_<N>_prediction_<YYYYMMDD_HHMM>.csv`, one per prediction run.
**Schema:** `player, team, predicted_disposals` (three columns).
**Coverage:** R15–R25 2025 (production season) and R2–R11 2026 (current season). Multiple timestamps per round when the model was re-run.

**Files (`backtest/`):**

- `backtest_run_<ts>.log` — full diagnostic log of a walk-forward run.
- `backtest_summary_<ts>.csv` — one row per (year, round) with `n, mae, rmse, bias, median_abs_error, pct_within_5, pct_within_10`.
- `backtest_by_team_<ts>.csv` — same metrics broken down by team-round.
- `backtest_by_position_<ts>.csv` — currently a single `Unknown` bucket because position is missing from player data.
- `prediction_vs_actual_round_<N>_<year>_<ts>.csv` — per-player actual vs predicted for one round.

**Files (`oldModelData/`):** Predictions from the deprecated `gpu_disposal_prediction_old.py` (cudf/cupy era). Retained for reproducibility comparisons.

### 3.5 `data/top100/` — all-time aggregated rankings

- `all_time_top_100.csv` — the canonical all-time top-100 player ranking. Mirrored to repo root as `all_time_top_100.csv`. Columns: `Serial Number, Player Name, Footy Teams, Comment`. The `Comment` column contains a one-paragraph career synopsis built from the player's CSV.
- `yearly/year_<YYYY>.csv` — per-season top-100 for every season from 1897 onward (130 files). Schema: `player, score, percentile_rank, games_played`. Score is the impact metric computed by `top_players_comprehensive.py`.

The ranking methodology (cap z-scores, take top-11 seasons, gamma curvature on rank-to-year-score mapping) is recorded in `.claude/agent-memory/Scientist/all_time_formula.md`.

### 3.6 Other `data/` subdirectories

| Path | Contents | Purpose |
|---|---|---|
| `data/lineups/` | 24 `team_lineups_<teamname>.csv` files (one per AFL club, current + historical) | Round-by-round selected sides; used by strategy briefs for opposition study |
| `data/conceded_stats/` | `team_stats_conceded_2025.csv` | Per-team stats conceded — derived from the 2025 dataset only |
| `data/fixtures/` | Empty (placeholder for forward fixture data) | |
| `data/era_*.csv` and `data/era_summary.json` | Outputs of `era_based_statistical_analysis.py` | Mean and significance-test results per era for every metric; powers the era charts in `docs/afl-history.md` |

---

## 4. Code and scripts inventory

### 4.1 Top-level: data pipeline and model

| File | Purpose |
|---|---|
| `prediction.py` | Main disposal-prediction model. Three-model ensemble (HGB + LightGBM + RandomForest, voted), Optuna tuning, GroupKFold-by-player CV, post-hoc OOF linear calibration. See §8. |
| `backtest.py` | Walk-forward backtest. For every (year, round) in the requested window, strips all rows at or after that round, runs `AFLDisposalPredictor` with `target_year=year`, joins predictions to actuals, writes the four `data/prediction/backtest/` artifacts. |
| `prediction_cpu.py` | Explicit CPU-only fork of `prediction.py`. Used on hosts where the GPU probe is undesirable. Don't touch when fixing GPU issues in `prediction.py`. |
| `gpu_disposal_prediction_old.py` | Deprecated `cudf`/`cupy` predecessor. Kept for archaeological reference; not in any pipeline. |
| `prediction_accuracy.py` | Standalone one-round prediction-vs-actual scorer. Used ad-hoc to inspect a single round's predictions without re-running the full backtest. |
| `refresh_data.py` | Targeted weekly scrape: matches in delta mode, then re-scrapes only active players (last game ≥ ACTIVE_SINCE) or new debutants discovered on team pages. Skips ~12,500 retired players. |
| `refresh_and_rank.sh` | Master cron driver. Runs (1) `refresh_data.py`, (2) `top_players_comprehensive.py`, (3) `prediction.py`, (4) `backtest.py --start-round 1 --end-round auto`, (5) `refresh_readme.py`. |
| `refresh_readme.py` | Re-renders every marker-driven section across `docs/afl-season-2026.md`, `docs/afl-team-profiles.md`, `docs/hall-of-fame-top100.md` after upstream data refreshes. Idempotent. |
| `game_scraper.py` | afltables match scraper. `MatchScraper.scrape_all_matches()` writes both `data/matches/` and `data/lineups/`. |
| `player_scraper.py` | afltables per-player scraper. Handles the `performance_details` + `personal_details` pair. |
| `top_players_comprehensive.py` | Rewrites `all_time_top_100.csv` (root + `data/top100/`) and `data/top100/yearly/year_<YYYY>.csv` for every season. Methodology: z-cap → top-11 seasons → rank-to-year gamma mapping. |
| `update_team_analysis.py` | Generates the current-season team-style section and the 5-year team-profile section. Writes between marker pairs in `docs/afl-season-2026.md` and `docs/afl-team-profiles.md`. Re-run is idempotent. |
| `generate_readme_charts.py` | Five PNG charts under `assets/charts/` (era scoring, era stat evolution, team radar, team heatmap, team-style scatter). |
| `era_based_statistical_analysis.py` | Buckets player-game rows by match year into four canonical eras (pre-1965, 1965–1990, 1991–2010, 2011+). Welch t-tests between adjacent eras. Writes `data/era_*.csv` and `data/era_summary.json`. |
| `analysis.py` / `bar_chart.py` / `charts.py` | Ad-hoc EDA + chart scripts predating the structured pipeline. Still useful for one-off plots. |
| `helper_functions.py` | Tiny soup-fetch helper used by older scripts. |
| `main.py` | Thin glue that calls `MatchScraper().scrape_all_matches()` and `PlayerScraper().scrape_all_players()`. The "scrape everything from scratch" entry point. |
| `cuDF_test.py` / `testGPU.py` | GPU smoke tests (cudf/cupy and LightGBM-GPU). Diagnostic, not production. |

### 4.2 `scripts/` — operational scripts

| File | Purpose |
|---|---|
| `fetch_live_match.py` | Downloads `fanfooty.com.au/live/<gameid>.txt`, parses the 65-column player schema, writes the JSON + CSV snapshot pair under `data/live_snapshots/`. Runs the column-count and quarter-sum sentries on every fetch. |
| `live_analysis_pipeline.py` | The 90-second polling pipeline. Wraps `fetch_live_match.py`, classifies status into a quarter code, routes to the right per-quarter doc, writes a structured stats block + quarter-break summary + ANALYST BLOCK on transitions, commits and pushes every cycle. See §7. |
| `live_match_monitor.py` | Older, simpler polling monitor. Pre-dates the routing/transition logic in `live_analysis_pipeline.py`; kept for the pre-R11 lineage. Writes a single doc instead of routing across quarter docs. |
| `smoke_test_live_pipeline.py` | Unit-level smoke test for the post-R11 hardening of the pipeline: `classify_status` covers every observed FanFooty status string; the skip-if-unchanged guard fires on a stalled feed; the guard does NOT over-prune when state actually changes. Loads the last R11 FT snapshot and exercises four assertions; no git, no doc writes. |
| `compute_r10_player_table.py` | Generates the full Richmond + Adelaide player stat table for R10 2026 by combining afltables canonical box-score numbers (kicks/marks/handballs/disposals/goals/behinds/hit-outs/tackles/clangers) with snapshot AF/SC/quarter-AF (the snapshot fields that are reliable). Hard-codes the AFL truth table as a dict. |
| `compute_r10_team_aggregates.py` | Aggregates Richmond and Adelaide team-level totals from the R10 game (now in player CSVs) for the post-mortem: tackles, inside-50s, clearances, contested possessions, clangers, hit-outs. Also pulls 2026-season aggregates for the "was this representative" framing. |
| `update_r10_player_data.py` | Appends R10 (Richmond vs Adelaide, 2026-05-10) rows to every named player's performance CSV. Uses afltables truth for goals/behinds/clangers because the snapshot is unreliable on those three. |
| `generate_player_cards.py` | Renders per-player PNG prediction cards from the latest prediction CSV. Trend indicator vs last-5-round actual mean. |
| `generate_weekly_cheat_sheet.py` | Reads the most recent `next_round_<N>_prediction_<ts>.csv`, sorts by predicted disposals, writes `docs/weekly/round-<N>-<year>.md` and overwrites `docs/weekly/round-current-<year>.md`. Pure-pandas, deterministic, no model retraining. |
| `package_fan_pack.sh` | Bundles the weekly cheat-sheet + prediction CSV + charts into a downloadable artefact. |

### 4.3 In-doc scripts

A few generator scripts live next to the docs they feed, because they exist to produce one specific section.

| File | Purpose |
|---|---|
| `docs/hall-of-fame/compute_stat_leaders.py` | Sums career totals across every player file, ranks the top 20 in each stat category. Writes `_stat_leaders.json` + `_stat_leaders.md` consumed by `hall-of-fame-stat-leaders.md`. |
| `docs/hall-of-fame/generate_records_charts.py` | Records-related chart generator for the hall-of-fame section. |
| `docs/coaches-strategy-corner/generate_strategy_charts.py` | Six PNGs per fixture (MCG form, H2H wins by era, team radar, quarterly differential, key-player disposal comparison, last-10 H2H lollipop). Saved under `assets/charts/strategy/<match-slug>/`. |

### 4.4 Other

- `scratch/diag_gamma.py` / `diag_ranks.py` / `validate_position_classification.py` — diagnostic notebooks-as-scripts for the ranking formula and a position-classification experiment. Not in any pipeline.
- `tests/unit/` — scaffold directory. The current functional smoke test is `scripts/smoke_test_live_pipeline.py`.
- `requirements.txt` — runtime deps.

---

## 5. Docs inventory

All paths relative to `docs/`.

### 5.1 Top-level docs

| File | Purpose | Status |
|---|---|---|
| `afl-insights.md` | Landing page for live season data, historical analysis, and guides | Index, evergreen |
| `afl-season-2026.md` | Current-season team analysis, finals pathway, Brownlow predictor, stat leaders | Auto-generated by `update_team_analysis.py` and `refresh_readme.py` between marker pairs |
| `afl-team-profiles.md` | 5-year team playing-style profiles for all 18 clubs | Auto-generated (`5YEAR-TEAM-PROFILES-START/END` markers) |
| `afl-team-analysis-2026.md` | Hand-authored team-by-team analysis for 2026 | Manual |
| `afl-stat-leaders-2026.md` | 2026 stat leader tables | Manual / partial auto |
| `afl-backtest-2026.md` | Backtest result writeup for the current season | Updated from `data/prediction/backtest/` after every run |
| `afl-brownlow-2026.md` | Brownlow predictor outputs | Auto-generated section |
| `afl-finals-2026.md` | Finals pathway analysis | Auto-generated section |
| `afl-predictions-2026.md` | Latest prediction summary | Refreshed after every `prediction.py` run |
| `afl-history.md` | Era scoring, player workload, historical trend writeup | Manual |
| `model-report-card.md` | Honest summary of model performance + caveats | Updated after each backtest |
| `prediction-model.md` | Model description for fan + ML reader | Manual |
| `data-science.md` | Data-science process documentation | Manual |
| `coaching-guide.md` | Workflow for using Claude + agents as a coaching aid | Manual |
| `footy-expert-guide.md` | For the footy expert: challenging the all-time ranking, using Claude for deeper questions | Manual |
| `start-here-no-code.md` / `how-to-use-this-for-supercoach.md` | Fan-facing onboarding | Manual |
| `quick-start.md` / `installation.md` / `usage.md` / `claude-code-setup.md` / `technical-reference.md` / `troubleshooting.md` | Engineer-facing setup + reference | Manual |
| `roadmap.md` | Open work and direction | Manual |
| `glossary.md` | AFL + ML term definitions | Manual |
| `ai-agents.md` / `ai-architecture.md` / `scientist-agent.md` | Documentation of the agent system | Manual |
| `footy-ai-chatbot-setup.md` / `footy-ai-chatbot-phase2.md` | Phase 1 and Phase 2 design for "The Crumb" multi-agent chatbot | Manual / design |
| `how-this-repo-uses-claude.md` | How Claude Code is wired into this repo's workflow | Manual |
| `hall-of-fame*.md` (10 files) | Top-100, captains, careers cut short, coaches, courageous, dynasties, indigenous, stat leaders | Hall-of-fame section; `hall-of-fame-top100.md` is auto-generated |

### 5.2 `docs/coaches-strategy-corner/` — per-game tactical briefs

**Index:** `README.md` documents the layout, methodology, and live-pipeline recipe. Also includes `afl-2026-team-list-analysis.md` (all 18 clubs by list quality / Tier 1 draft pedigree) and `generate_strategy_charts.py`.

**Game: Richmond vs Adelaide, Round 9 2026, M.C.G., 10 May 2026** — 10 docs:

| File | Layer |
|---|---|
| `richmond-vs-adelaide-round-9-2026.md` | Full tactical brief (pre-match) |
| `richmond-vs-adelaide-round-9-2026-executive-summary.md` | 1-page entry point with charts |
| `richmond-vs-adelaide-round-9-2026-player-matchups.md` | Player-by-player matchup guide |
| `richmond-vs-adelaide-round-9-2026-head-to-head-history.md` | H2H history brief |
| `richmond-vs-adelaide-round-9-2026-half-time-live.md` | Live read, half-time |
| `richmond-vs-adelaide-round-9-2026-q3-live.md` | Live read, end of Q3 |
| `richmond-vs-adelaide-round-9-2026-q4-live.md` | Live read, Q4 / full-time |
| `richmond-vs-adelaide-round-9-2026-full-time-verdict.md` | Full-time verdict |
| `richmond-vs-adelaide-round-9-2026-postmortem.md` | Data-layer post-mortem (Scientist) |
| `richmond-vs-adelaide-round-9-2026-postmortem-footystrategy.md` | Tactical post-mortem (FootyStrategy) |

**Game: Richmond vs St Kilda, Round 11 2026, Marvel Stadium, 17 May 2026** — 11 docs:

| File | Layer |
|---|---|
| `richmond-vs-stkilda-round-11-2026.md` | Full tactical brief (pre-match) |
| `richmond-vs-stkilda-round-11-2026-executive-summary.md` | 1-page summary |
| `richmond-vs-stkilda-round-11-2026-player-matchups.md` | Player-by-player matchup guide |
| `richmond-vs-stkilda-round-11-2026-head-to-head-history.md` | H2H history brief |
| `richmond-vs-stkilda-round-11-2026-q1-live.md` | Live read, Q1 |
| `richmond-vs-stkilda-round-11-2026-q2-live.md` | Live read, Q2 |
| `richmond-vs-stkilda-round-11-2026-half-time-live.md` | Live read, half-time |
| `richmond-vs-stkilda-round-11-2026-q3-live.md` | Live read, Q3 |
| `richmond-vs-stkilda-round-11-2026-q4-live.md` | Live read, Q4 |
| `richmond-vs-stkilda-round-11-2026-full-time-verdict.md` | Full-time verdict |
| `richmond-vs-stkilda-round-11-2026-postmortem.md` | Joint post-mortem (data + tactical) |

### 5.3 `docs/news/` — long-form journalism

**Index:** `README.md` documents the two-layer (Scientist + FootyStrategy) house rules and a glossary of `**[data]**` / `**[historical record]**` tags.

| File | Topic | Status |
|---|---|---|
| `2026-05-13-voss-carlton.md` | Carlton coaching change | Complete (data + FootyStrategy layers) |
| `2026-05-15-richmond-vs-stkilda-r11.md` | Match preview / Milera injury impact | Complete |
| `2026-05-15-richmond-vs-stkilda-r11-data.md` | Working data brief (Scientist) — input for the published piece | Draft |
| `2026-05-15-carlton-next-coach.md` | Carlton coaching succession analysis | Complete |
| `2026-05-15-carlton-next-coach-data.md` | Working data brief (Scientist) for the Carlton piece | Draft |
| `2026-05-15-carlton-next-coach-footystrategy.md` | Standalone FootyStrategy layer for the Carlton piece | Draft |

### 5.4 `docs/weekly/` — fan-pack cheat sheets

| File | Purpose |
|---|---|
| `round-09-2026.md`, `round-11-2026.md` | Per-round predicted-disposal leaderboards (Top 30 + appendix) |
| `round-current-2026.md` | Overwritten on every cheat-sheet run so the README link is stable |

### 5.5 `docs/hall-of-fame/`

Auxiliary scripts and data for the hall-of-fame docs.

| File | Purpose |
|---|---|
| `compute_stat_leaders.py` | All-time stat leader computation |
| `generate_records_charts.py` | Records-related charts |
| `_stat_leaders.json` / `_stat_leaders.md` | Auto-generated outputs consumed by `hall-of-fame-stat-leaders.md` |

---

## 6. Workflow — the match lifecycle

A match in this repo passes through five phases. Both agents work at every phase; the artefacts they produce sit alongside each other and are wired into git so the timeline is a reviewable commit log.

### 6.1 Pre-match (typically 3–7 days before the bounce)

The pre-match flow is a five-agent pipeline: **BriefBuilder → Scientist → FootyStrategy → DataSentinel → Skeptic (optional)**. BriefBuilder assembles the tabular spine (H2H, season form, model predictions, top-5-per-side tracking list) with `**[data]**` tags and `<!-- FOOTYSTRATEGY INSERT -->` placeholders; Scientist adds non-routine analysis (per-team form vs league, era-coverage caveats, model bias for these teams, the six strategy charts under `assets/charts/strategy/<match-slug>/`); FootyStrategy fills the interpretation layer including a headline call with explicit tripwire and the load-bearing structural reads; DataSentinel walks every tag against its CSV and emits PASS/FAIL JSON that a pre-commit hook consumes; Skeptic runs for high-stakes briefs (finals, news-cited) and emits PASS / PASS_WITH_CONCERNS / BLOCK without ever modifying the doc.

The handoff contracts (what each agent receives and what it must not invent) are in [`ai-architecture.md` §10](ai-architecture.md#10-the-six-agent-council--architecture-and-interaction-model). The operator playbook with copy-paste prompts is [`coaching-guide.md` §"Getting all six agents to brainstorm together"](coaching-guide.md#getting-all-six-agents-to-brainstorm-together).

**Commit cadence:** one or two commits per session, message scoped to the brief slug. DataSentinel must pass before commit; Skeptic concerns are addressed at the author's discretion.

### 6.2 Live — during the match

**The pipeline runs:** `scripts/live_analysis_pipeline.py <gameid>` polls FanFooty every 90 seconds. Behaviour detail in §7.

Per poll the pipeline:

1. Fetches a snapshot via `scripts/fetch_live_match.py`, writes the JSON + CSV pair under `data/live_snapshots/`.
2. Classifies the FanFooty status into one of `Q1, QT, Q2, HT, Q3, 3QT, Q4, FT`.
3. Routes to the correct per-quarter doc under `docs/coaches-strategy-corner/<match-slug>-<phase>-live.md`.
4. If the status is the same as last poll and **all** numbers (score, disposals, tackles, hit-outs) are identical, **skips** writing (no duplicate blocks).
5. Otherwise builds a structured analysis block: score, disposal leaders, current-quarter AF leaders, team-totals table, tripwire status, key-player tracking, and a dynamic "Read" paragraph that describes what *changed* since the last poll.
6. Inserts the block at the top of the doc (newest first) under the `<!-- LIVE_ANALYSIS_AUTO_BLOCKS_BELOW -->` marker.
7. On a quarter-transition: writes a `QUARTER BREAK` summary to the OUTGOING doc (closing read) and an `ANALYST BLOCK` to the INCOMING doc (forward-looking opening read with top-5 movers and updated key-player tracking).
8. `git add` the touched docs and the last two snapshot JSON+CSV pairs; commit with `Live analysis <gameid>: <status>`; `git push origin main`.

**Scientist:** owns the pipeline. The automated blocks are Scientist output in the sense that Scientist wrote and maintains the code; they are not LLM-generated prose.

> **Note:** the per-poll analysis blocks are generated by rule-based Python code (`live_analysis_pipeline.py`), not by LLM inference at runtime. They are Scientist-authored in the sense that Scientist wrote and maintains the code, but the 90-second blocks are deterministic outputs of that code, not agent-generated prose. The "Read" paragraph composes its sentences from a `prev_state` delta table; the quarter-break analyst block composes from the `af_qN` columns. No LLM is in the live loop.

**FootyStrategy:** can be invoked at any quarter break for a multi-lens tactical read of the ANALYST BLOCK. These are appended manually after the automated block and tier-labelled as Probationary at most (the live-data sample is n=1 quarter).

### 6.3 Full-time — immediate

**Scientist:**

1. Pipeline detects `Full Time` / `Final Siren`, writes the final block to `<match-slug>-full-time-verdict.md`, and exits.
2. Appends the final-game player rows to each named player's `data/player_data/<surname>_<first>_<dob>_performance_details.csv` — for goals/behinds/clangers, uses afltables (not the snapshot, which is unreliable on those three).
3. Computes team-level aggregates (`scripts/compute_r10_team_aggregates.py` pattern): tackles, inside-50s, clearances, contested possessions, clangers, hit-outs — for the post-mortem.

### 6.4 Post-match — same evening or next day

**Scientist** writes the **data-layer post-mortem** under `<match-slug>-postmortem.md` or `<match-slug>-postmortem-data.md`:

- Final result vs each pre-match call, with the actual number tagged `**[data]**`.
- Section-by-section breakdown of which pre-match reads held and which broke.
- Honest "what we missed" section — usually one or two breakout players or structural reads the brief did not name.
- Methodology paragraph naming every source file used.

**FootyStrategy** writes a **tactical post-mortem** under `<match-slug>-postmortem-footystrategy.md` (or merges into the joint file):

- Council lens deliberation across 3–5 archetypes.
- Tier the result evidence at (Settled / Probationary / Contested / Insufficient).
- Update the `recurring_tensions.md` memory if a new tension between lenses surfaced.
- Explicit tripwires from the brief: which fired, which held, which were unfalsifiable in hindsight.

### 6.5 Weekly refresh — overnight Monday

`refresh_and_rank.sh` runs end-to-end:

1. `refresh_data.py` — scrape new match results and active-player updates.
2. `top_players_comprehensive.py` — recompute the all-time top-100.
3. `prediction.py` — retrain and predict for the next round.
4. `backtest.py --start-round 1 --end-round auto` — incremental walk-forward over the season so far.
5. `refresh_readme.py` — regenerate every marker-driven section in the season and team-profile docs.

The CI workflow at `.github/workflows/weekly-fan-pack.yml` then packages the cheat sheet for distribution.

### 6.6 Git as the audit trail

Every step above produces a commit. The diff of any doc is the change history; the message is the context. Both agents push directly to `main` (see `feedback_push_to_main.md` in user memory) — PRs are not used for routine in-repo work. This is deliberate: the doc *is* the publication, and the commit *is* the audit.

---

## 7. Live pipeline architecture

`scripts/live_analysis_pipeline.py` (~1211 lines) is the load-bearing live system. It runs as a long-lived process started by the operator at first bounce and exits on Full Time.

> **Note (clarification, not change):** every block this pipeline writes is generated by rule-based Python code, not by LLM inference at runtime. The pipeline is "Scientist output" only in the sense that Scientist authored and maintains the code. The 90-second poll cycle, the "Read" paragraph, the QUARTER BREAK summary, and the ANALYST BLOCK are all deterministic functions of the snapshot and the `prev_state` delta — no agent is called during the live loop. LLM-authored quarter-break commentary is listed as planned work in §12.1.

### 7.1 Status classification

`classify_status(status_raw) -> str | None` maps the FanFooty status string to one of 8 quarter codes.

```python
QUARTER_ORDER = ["Q1", "QT", "Q2", "HT", "Q3", "3QT", "Q4", "FT"]
```

The function's check order is load-bearing — getting it wrong was the root cause of the R11 routing bug where 8 polls wrote end-of-game data into the Q1 live document. The order is:

1. **Game-over states first.** `"full time"`, `"ft"`, `"final siren"`, `"fs"` → `"FT"`. Must match before any in-quarter regex so a scoreboard frozen at the siren is not re-classified as Q4.
2. **Three-quarter-time variants.** `"three quarter time"`, `"3qt"`, `"3 qtr"`, `"3qtr"` → `"3QT"`. Must match before generic `"qtr time"`.
3. **In-quarter strings.** `"q4"`, `"q3"`, `"q2"`, `"q1"` → matching code. Must match before break tokens because in-play strings like `"Q3 19:11"` still contain the quarter token.
4. **Break states.** `"half time"`, `"ht"` → `"HT"`. `"quarter time"`, `"qtr time"`, `"qt"` → `"QT"`.
5. **Unrecognised → `None`.** The caller treats `None` as "skip this cycle and log a warning" — never as a silent fallback to Q1. That silent fallback was the bug.

### 7.2 Doc routing

`DOC_FOR_QUARTER` is a status-code → filename map:

| Status code | Document |
|---|---|
| `Q1` | `<match-slug>-q1-live.md` |
| `QT` | `<match-slug>-q1-live.md` (end-of-Q1 summary on Q1 doc) |
| `Q2` | `<match-slug>-q2-live.md` |
| `HT` | `<match-slug>-half-time-live.md` |
| `Q3` | `<match-slug>-q3-live.md` |
| `3QT` | `<match-slug>-q3-live.md` |
| `Q4` | `<match-slug>-q4-live.md` |
| `FT` | `<match-slug>-full-time-verdict.md` |

Routing is computed every poll. The match-slug is currently hard-coded per pipeline invocation (`DOC_BASE = "richmond-vs-stkilda-round-11-2026"`).

### 7.3 `prev_state` delta tracking

The pipeline keeps a `prev_state` dict in memory across polls so the "Read" paragraph can describe what *changed*, not just the current snapshot. The dict is built by `build_prev_state()` and contains:

```
ric_disposals / stk_disposals     # cumulative disposals per side
ric_tackles   / stk_tackles
ric_hitouts   / stk_hitouts
ric_pts       / stk_pts            # cumulative points per side
status_code                        # the prior status
player_disp:  { player_key -> disposals }   # per-player on the prior poll
player_tk:    { player_key -> tackles }
```

Per poll, the "Read" paragraph composes 5 sentences from delta evidence:

1. **Score delta** — unanswered goals, scoreboard stalemates, trades.
2. **Possession / pressure delta** — net disposal swing or tackle swing this block.
3. **Current-quarter AF leader** — who is hot *right now*, not cumulatively.
4. **Rising / falling players** — per-player +4 disp or +3 tackle jumps since last poll, capped at the two biggest movers.
5. **Game-state closer** — frame the read by score band (close / chasing / leading / blowout) and status code.

`prev_state` is reset to `None` at quarter transitions so deltas don't straddle break boundaries — a 14-minute break would otherwise look like a "block" with massive deltas.

### 7.4 Quarter-break analyst blocks

At every transition out of a live quarter (e.g. `Q3 19:11` → `Q3T`), the pipeline writes **two** blocks:

- **`format_quarter_break(prev_code, snap)`** — written to the OUTGOING doc. Closes out the quarter just finished: top-3 quarter-AF leaders per side, the kick-share tripwire state, and a one-line score verdict that compares the quarter's margin to the pre-match target (Q1 ≤15 down, Q2 ≤25 down at half time, Q3 ≤30 down at 3QT).
- **`format_quarter_break_analyst(prev_code, snap)`** — written to the INCOMING doc. Opens the new quarter's doc with a forward-looking structured read: score verdict vs target, top-5 movers in both teams by quarter AF, key-player tracking vs pre-match predictions (Short / Sinclair / Hill), one forward-looking sentence for the next quarter. Equivalent to having a Scientist commentary block fire automatically — Q3 and Q4 docs never start on a blank page.

The transition handler is wrapped so an analyst-block exception cannot crash the main poll loop.

### 7.5 Skip-if-unchanged guard

`format_analysis_block()` returns `(None, prev_state)` when the FanFooty feed has stalled — score, disposals, tackles, AND hit-outs all identical to the prior poll. The caller writes nothing and commits nothing (a no-op commit is worse than no commit — it inflates the log).

The guard is **disabled at quarter-break codes** (`QT`, `HT`, `3QT`, `FT`) because at a break the routing change *is* the news; the numbers can be frozen and the doc still needs the heading.

### 7.6 The insertion marker

Every per-quarter doc has the structure:

```
# Title
> back-links / intro paragraph(s)

## Auto-updated live analysis (newest first)

<!-- LIVE_ANALYSIS_AUTO_BLOCKS_BELOW -->

<newest block>
<older block>
...

---

<pre-existing hand-authored sections>
```

`insert_block(path, block)` finds the `<!-- LIVE_ANALYSIS_AUTO_BLOCKS_BELOW -->` marker and inserts the new block immediately after it, pushing older blocks down. If the marker is absent (because the doc was hand-authored before the pipeline started), `_find_header_end()` splices the auto-section in right after the H1 title and intro, before any existing `##` / `###` / `---` section.

### 7.7 Git commit and push

`git_commit_push(paths, status)` runs after every block write that produced output:

1. `git add <touched docs> <last 2 snapshot JSON+CSV pairs>`. Limiting the snapshots prevents a long-running game from accumulating hundreds of binary diffs in one commit.
2. `git commit -m "Live analysis <gameid>: <status>\n\nAuto-pushed by live_analysis_pipeline.py\n\nCo-Authored-By: ..."`.
3. `git push origin main`.

Commit failures other than "nothing to commit" are logged but do not stop the loop. A commit happens roughly every 90s during a live game; over a typical 2.5-hour match that is ~100 commits per fixture.

### 7.8 Hardening lineage

The pipeline went through a known R11 incident where unrecognised status strings silently fell back to `Q1`, dumping end-of-game blocks into the Q1 doc. The fixes are tracked in `.claude/agent-memory/Scientist/live_pipeline_glitch.md` and exercised by `scripts/smoke_test_live_pipeline.py`. Three asserts cover the regression:

- `classify_status("Final Siren")` → `"FT"` (not `"Q1"`).
- A stalled in-quarter feed with identical `prev_state` returns `(None, prev_state)` — no block written.
- A meaningfully-different `prev_state` still produces a block (the guard does not over-prune).

---

## 8. Prediction model architecture

`prediction.py` (~1100 lines, class `AFLDisposalPredictor`) is the disposal-prediction system.

### 8.1 Inputs

- `data/player_data/*_performance_details.csv` — every per-player per-game row across history.
- A `target_year` (auto-detected as the max `year` observed across files; fallback to current calendar year).
- A `birth_year_threshold = target_year - 40` filter — pre-filters retired or otherwise-implausible players.

### 8.2 Feature engineering

Base columns rolled up: `disposals, kicks, handballs, tackles, clearances, inside_50s` (the columns present in modern data; era coverage means pre-1998 rows have fewer features). Extra columns: `cba_percent`, `percentage_time_played` (where available).

Per base column, four features are produced:

| Feature | How it is computed |
|---|---|
| `across_season_rolling_avg_<col>_5` | 5-game rolling mean grouped by `player`, shifted by 1 (so the row's target value is excluded) |
| `within_season_rolling_avg_<col>_3` | 3-game rolling mean grouped by `(player, year)`, shifted by 1 |
| `season_to_date_mean_<col>` | Expanding mean grouped by `(player, year)`, shifted by 1 |
| `recent_form_<col>` | EWM (span=3) grouped by `player`, shifted by 1 |

Plus: `round_number`, `days_since_last_game`, the extra features (`cba_percent`, `percentage_time_played`), and one-hot dummies for `venue` and `opponent`.

The shift-by-1 is the leak-prevention discipline — when predicting row N, the rolling feature on row N must not include row N's own target value.

### 8.3 LightGBM device detection

`_detect_lgbm_device() -> 'gpu' | 'cpu'` runs a tiny `LGBMRegressor(device='gpu', n_estimators=1).fit(...)` on dummy data at module load. If LightGBM raises (no GPU build, no CUDA, etc.) the helper catches the exception and returns `'cpu'`. The result is cached in the module-level constant `LGBM_DEVICE` and threaded through both Optuna trial params and the final `LGBMRegressor` in the Pipeline.

The probe redirects OS-level fd 2 to `/dev/null` for the duration of the probe so LightGBM's C-level `[Fatal] GPU Tree Learner was not enabled in this build` log line does not leak into the user's stderr on CPU-only hosts.

On the current dev host, `LGBM_DEVICE` resolves to `'cpu'`. Full walk-forward backtest takes ~5–6 hours on CPU vs ~30–60 min on GPU. See `.claude/agent-memory/Scientist/prediction_lgbm_cpu.md`.

### 8.4 Models

Three base learners, all Optuna-tuned (50 trials, 10-minute timeout, GroupKFold by player):

| Model | Notes |
|---|---|
| `HistGradientBoostingRegressor` (Poisson loss) | Internal log link; gradient-boosted on raw disposals |
| `LGBMRegressor` (`objective='regression'`, L2/mean) | Switched from `regression_l1` after the 2026-04-30 top-end compression incident — L1 predicted the median, which on right-skewed disposals is below the mean and under-predicted 30+ disposal games by 7–15. L2 is more sensitive to outliers but better calibrated at the top end. |
| `RandomForestRegressor(n_estimators=200, max_depth=6)` | Untuned, diversifier |

These are combined in a `VotingRegressor` ensemble. The best individual model by GroupKFold MSE is selected as `self.best_name` and used for prediction.

Cross-validation throughout is `GroupKFold(n_splits=5)` keyed on player — no player ever appears in both train and validation folds. The groups array is set in `prepare_features_and_target()` and consumed positionally by `cross_val_score`.

### 8.5 The calibration step

Target is raw `disposals` (no `log1p`). The earlier log1p target paired with expm1 on output, combined with HGB's internal log link, double-compressed the high tail — max prediction was 28 vs max actual 43. See `prediction_top_end_compression.md`.

After model fit, `_fit_calibration()` runs a `GroupKFold` pass with the chosen best model, collects OOF predictions, and fits a single `(slope, intercept)` linear regression of `actual_disposals` on `oof_prediction`. Applied at predict time as `predictions = slope * raw_pred + intercept`.

Calibration is bounded: `slope ∈ [0.5, 2.0]`, `|intercept| ≤ 20`. Outside those bounds, falls back to identity (`1.0, 0.0`) and logs a warning.

Predictions are clipped to `[1, 55]`. Lower bound 1 because 0 implies DNP (which the upstream pipeline doesn't model). Upper bound 55 because historical max single-game disposals in the modern era is ~50.

### 8.6 Outputs

| File | Schema | Purpose |
|---|---|---|
| `data/prediction/next_round_<N>_prediction_<YYYYMMDD_HHMM>.csv` | `player, team, predicted_disposals` | Headline next-round predictions |
| `data/prediction/backtest/backtest_summary_<ts>.csv` | per (year, round): `n, mae, rmse, bias, median_abs_error, pct_within_5, pct_within_10` | Backtest summary |
| `data/prediction/backtest/backtest_by_team_<ts>.csv` | same metrics broken down by `team` | Team-bias inspection |
| `data/prediction/backtest/backtest_by_position_<ts>.csv` | same metrics by `position` (always `"Unknown"` until a position source is wired) | Schema placeholder |
| `data/prediction/backtest/prediction_vs_actual_round_<N>_<year>_<ts>.csv` | per-player row with `predicted_disposals`, `actual_disposals` | For per-round inspection |
| `data/prediction/backtest/backtest_run_<ts>.log` | full diagnostic log | |

### 8.7 The backtest

`backtest.py` implements the walk-forward. For every (year, round) in the requested window:

1. Strip every row from the player-data DataFrame at or after (year, round). All year > target_year rows are dropped, plus same-year rows with round >= target_round.
2. Instantiate `AFLDisposalPredictor` with `target_year=year`.
3. Run the full pipeline. Predictions come out for the target round.
4. Join predictions to actuals, write a per-round `prediction_vs_actual_*` CSV.
5. Aggregate up to `backtest_summary_*`, `backtest_by_team_*`, `backtest_by_position_*`.

Strict temporal cutoff. The same model that runs in production runs in backtest; nothing is held back, nothing is added. This is what makes the headline MAE number defensible.

---

## 9. Known limitations and caveats

These are the load-bearing caveats. Both agents are expected to honour them; CLAUDE.md operationalises them.

### 9.1 FanFooty snapshot — three unreliable per-player columns

`goals` (col15), `behinds` (col16), `clangers` (col39) in the per-player CSV are misindexed in the upstream FanFooty feed. Cross-checked against afltables on the R10 2026 Richmond vs Adelaide snapshot: 102 field mismatches concentrated in those three fields; zero mismatches in kicks / handballs / marks / tackles / hit-outs / AF / SC / quarter splits / TOG% / DE%.

**Use:** afltables for goals/behinds/clangers; header `home_score` / `away_score` for the scoreboard; the m0nty commentary stream for individual goal attribution. See `snapshot_data_quality.md`.

### 9.2 FanFooty snapshot — three stats not in the schema at all

`inside_50s`, `clearances`, `contested_possessions` are not in the FanFooty per-player schema. The live pipeline cannot compute true I50 differential. It uses **kick-share** as a territory proxy and labels it explicitly (`*Inside 50s / contested poss / clearances are not in the FanFooty per-player snapshot schema. Kick-share used as a proxy below.*`).

### 9.3 Goals data — header is authoritative, per-player is not

For an individual game's team goal count, the team sum of the per-player `goals` column does not match the scoreboard. The match-header `home_score` / `away_score` is the authoritative scoreline. Do not attribute individual goals from the per-player column; use the commentary stream.

### 9.4 Pre-1965 records are incomplete

AFL stats kept before 1965 are limited to goals and behinds. No disposals, kicks, marks, handballs, tackles, clearances, or possessions exist before 1965. Any claim about a pre-1965 player's disposal count, tackle count, or per-game average is unverifiable in this data.

**Use the tag `**[historical record — unverified in data]**`** rather than inventing a number, per CLAUDE.md.

Other coverage boundaries: hit-outs from 1966; tackles from 1987; clearances and inside-50s from 1998; contested and uncontested possessions from 1999. A 2016/2017 hit-out jump is a recording-method change, not a tactical shift — see `data_stat_coverage_eras.md`.

### 9.5 Player files have no position column

Neither `*_performance_details.csv` nor `*_personal_details.csv` carries a `position` field. Per-position aggregates currently bucket every row as `"Unknown"`. Any per-position analysis requires a new data source (likely an extension to `player_scraper.py`).

**Do not** fabricate positions from height/weight or guess from kicks/handball ratios — flag the missing source and ask.

### 9.6 Model team-bias

The walk-forward backtest surfaces team-level bias that is not uniform across the league. Richmond, in particular, has been over-predicted by +0.53 average on its 2026 season — the model has not fully absorbed the structural collapse of Richmond's 2026 disposal volume (worst in league). The team-by-team breakdown is in `backtest_by_team_<ts>.csv`; consult it before treating a Richmond-specific prediction as decision-grade.

### 9.7 Top-end compression — fixed but watch for regression

The 2026-04-30 backtest showed max prediction 28 vs max actual 43 because of compounding `log1p`+`regression_l1`+tree-mean-regression. The fix: train on raw disposals, switch LGBM to `objective='regression'`, add OOF linear calibration. If max prediction reverts to <30 in a future backtest, the first suspects are compression returning (transform/loss reintroduced) or calibration failing its bounds check (look for "Calibration out of bounds" in the run log).

### 9.8 Live pipeline polling-end-of-game glitch

FanFooty sometimes keeps stamping end-of-game scores onto polls after the siren. Earlier versions of the pipeline routed these into earlier-quarter docs. The fixes (1) treat `Final Siren` as `FT` not `Q1`; (2) skip writes when score+disposals+tackles+hit-outs are all unchanged; (3) refuse to silently fall back when `classify_status` returns `None`. Smoke-tested by `scripts/smoke_test_live_pipeline.py`.

### 9.9 H2H is associational, not causal

A team's H2H ledger against another team is associational evidence — sample sizes are small, lists change year to year, structural matchups vary. Treat a 5-0 H2H streak as a flag worth investigating, not a deterministic prediction. FootyStrategy's `Caveat propagation` line carries this through to every tactical brief.

---

## 10. How to run things

### 10.1 The Python environment

All commands assume the project venv:

```
/home/abhi/sourceCode/python/coding/.venv/bin/python
```

Activate via `source /home/abhi/sourceCode/python/coding/.venv/bin/activate` if running interactively. The venv ships with pandas, sklearn, lightgbm, optuna, requests, beautifulsoup4, matplotlib, seaborn — see `requirements.txt`.

### 10.2 Start a live pipeline for a new game

1. Find the FanFooty game ID. The URL is shaped `fanfooty.com.au/live/2026/<gameid>-<team-slug>.html`. Only the numeric `<gameid>` is needed.
2. (One-time) Update the per-match constants in `scripts/live_analysis_pipeline.py`:
   - `DOC_BASE = "richmond-vs-stkilda-round-11-2026"` — the match slug for routing.
   - `KEY_PLAYERS` — the three pre-match named players to track.
3. Start the loop. It exits automatically on Full Time.

```bash
/home/abhi/sourceCode/python/coding/.venv/bin/python \
  scripts/live_analysis_pipeline.py <gameid>
```

The pipeline prints one log line per poll: timestamp, classified status code, scoreboard, and which doc the block was routed to. A `[skip] feed stalled` line means the skip-if-unchanged guard fired — no block was written and nothing was committed. A `[warn] unrecognised status string` line means `classify_status` returned `None` and the cycle was skipped intentionally.

### 10.3 One-off snapshot (no polling)

For a single ad-hoc snapshot of a live game without the loop:

```bash
/home/abhi/sourceCode/python/coding/.venv/bin/python \
  scripts/fetch_live_match.py <gameid>
```

Writes a single JSON + CSV pair under `data/live_snapshots/` and prints the score, round, venue, player count, and the top 5 disposal-getters per side. The two sentries (65 columns per row; per-player quarter-AF sums to total AF) run automatically.

### 10.4 Run the prediction model

```bash
/home/abhi/sourceCode/python/coding/.venv/bin/python prediction.py
```

Auto-detects target year and next round from the latest player data. Writes `data/prediction/next_round_<N>_prediction_<YYYYMMDD_HHMM>.csv`. CPU run on this host takes ~30–60 minutes for full re-train + predict; GPU run on a CUDA-enabled host takes ~5–10 minutes.

### 10.5 Run the backtest

Full walk-forward over the current season:

```bash
/home/abhi/sourceCode/python/coding/.venv/bin/python backtest.py \
  --start-year 2026 --start-round 1 --end-year 2026 --end-round auto
```

Incremental (last round only):

```bash
/home/abhi/sourceCode/python/coding/.venv/bin/python backtest.py \
  --start-year 2026 --start-round 10 --end-year 2026 --end-round 10
```

Full season on CPU takes ~5–6 hours; single round takes ~5–10 minutes. Outputs land in `data/prediction/backtest/`.

### 10.6 Run the smoke test

```bash
/home/abhi/sourceCode/python/coding/.venv/bin/python \
  scripts/smoke_test_live_pipeline.py
```

Exercises the four hardenings of the live pipeline post-R11. No git, no doc writes — pure unit. Exits with `All smoke-test assertions passed.` on success. Run this after any change to `classify_status` or the skip-if-unchanged guard.

### 10.7 Refresh everything (weekly cron)

```bash
bash refresh_and_rank.sh
```

Runs scrape → top-100 → predict → backtest → docs refresh end-to-end. Designed to be safe to re-run; each step is independently idempotent given fresh upstream inputs.

### 10.8 Generate the weekly cheat sheet

```bash
/home/abhi/sourceCode/python/coding/.venv/bin/python \
  scripts/generate_weekly_cheat_sheet.py --year 2026
```

Reads the latest `next_round_<N>_prediction_*.csv` and writes `docs/weekly/round-<N>-2026.md` (plus overwrites `docs/weekly/round-current-2026.md`). Pure-pandas; no model retraining; safe to run any time after `prediction.py`.

### 10.9 Verify a player stat (CLAUDE.md flow)

The non-negotiable rule before writing any specific player stat. Example for Vic Aanensen:

```python
import pandas as pd, glob

PLAYER_DIR = "data/player_data"
files = glob.glob(f"{PLAYER_DIR}/aanensen_vic_*_performance_details.csv")
df = pd.read_csv(files[0])
print(f"Games: {len(df)}")
print(f"Goals: {df['goals'].sum()}")
print(f"Disposals: {df['disposals'].sum()}")
print(f"Years: {df['year'].min()}–{df['year'].max()}")
```

Same pattern for any career stat. If the player's file is missing or the stat is genuinely unavailable for that era, tag the claim `**[historical record — unverified in data]**` rather than inventing a number.

---

## 11. Recent problems encountered (and fixes shipped)

A log of significant defects observed during live operation and what was done about each. Commit hashes link the fix to its diff. Where a single commit covers multiple subsections (`deaa918b0` bundles five R11 hardenings), the same hash is cited under each affected item.

### 11.1 Hardcoded `device='cpu'` in `prediction.py` (Codex review finding)

- **Problem:** `prediction.py` hardcoded `device='cpu'` for every LightGBM fit, ignoring available GPU hardware. On a CUDA-enabled host the cost was a 5–10x training slowdown that nobody noticed because the runs still completed.
- **Fix:** Added `_detect_lgbm_device()` — probes the local LightGBM build by attempting a 1-tree fit on dummy data with `device='gpu'`; falls back to `'cpu'` if the GPU build is unavailable. The result is cached in the module-level constant `LGBM_DEVICE` and threaded through both the Optuna trial params and the final `LGBMRegressor` in the Pipeline. Probe stderr is redirected to `/dev/null` so the C-level "GPU Tree Learner was not enabled" log line does not leak to the user.
- **Commit:** `e153046db` (Fix hardcoded device='cpu' in prediction.py - use CUDA when available, fall back to CPU).

### 11.2 Pipeline pushing only every 3rd poll

- **Problem:** The initial `live_analysis_pipeline.py` had an `if iteration % 3 == 0` guard on the git push step. Docs went 4.5 minutes between pushes — a long time during a fast-moving quarter, and an unforgiving cadence for any external reader pulling from `origin/main`.
- **Fix:** Removed the iteration guard. The pipeline now pushes after every poll that actually wrote a block (every 90s during a live game; skipped polls do not push because there is nothing to commit).
- **Commit:** `f20d45fad` (Push to main on every 90s poll, not every 3rd).

### 11.3 Pipeline inserting new blocks at bottom not top

- **Problem:** New analysis blocks were appended to the bottom of the per-quarter doc. The newest analysis was the hardest to find — a reader had to scroll past every older block to see what just happened.
- **Fix:** Added `_find_header_end()` helper plus a `<!-- LIVE_ANALYSIS_AUTO_BLOCKS_BELOW -->` insertion marker. All new blocks now prepend immediately after the marker, pushing older blocks down. If the marker is absent (older hand-authored doc), `_find_header_end()` splices the auto-section in right after the H1 title and intro, before any existing `##` / `###` / `---` boundary.
- **Commit:** `9a045bc3e` (Fix live_analysis_pipeline: insert new blocks at top of doc, not bottom).

### 11.4 Repetitive "Read" paragraph — identical template every 90 seconds

- **Problem:** The `Read:` paragraph in every auto-block was a hardcoded f-string template ("Saints dominating possession X-Y… Short running hot… Sinclair on track… tripwire territory"). Across 50+ blocks in Q3 and Q4 only the numbers changed. Same sentence structure every poll. No evolving narrative, no score-movement delta, no quarter-specific story — pure Mad Libs.
- **Fix:** Added `_build_dynamic_read()`. The paragraph is now composed from deltas against a `prev_state` dict carried across polls. Narrative content changes based on score movement since the last poll, who is hot in the *current* quarter (via the `af_qN` columns), rising / falling player flags (4+ disposals or 3+ tackles in the last 90s), and the game-state band (close / chasing / blowout / final / break). New `top_quarter_af()` helper surfaces the in-quarter leader rather than the cumulative leader.
- **Commit:** `1e137356a` (Live pipeline: replace Mad-Libs Read with delta-aware dynamic narrative).

### 11.5 "Final Siren" routing bug — 8 polls misfiled to Q1 doc

- **Problem:** FanFooty emits `"Final Siren"` as a status after the final siren sounds but before the official `"Full Time"` string is published — a window that lasted 8 polls (~12 minutes) during R11. The pipeline only checked for `"Full Time"` / `"ft"` as game-over signals, so the `"Final Siren"` polls fell through to the Q1 default and wrote end-of-game scores into the Q1 live document. The Q1 doc ballooned to 75KB of mostly post-game data.
- **Fix:** `classify_status()` now matches `"final siren"` (and `"FS"`) and returns `"FT"`. Check ordering was reorganised so all game-over signals fire before any in-quarter regex — a scoreboard frozen at the siren is no longer re-classified as Q4.
- **Commit:** `deaa918b0` (Harden live_analysis_pipeline.py after R11 Richmond vs St Kilda postmortem).

### 11.6 Thin quarter-break blocks

- **Problem:** The auto-generated `QUARTER BREAK` block had only 4 cumulative stat lines. No quarter-specific leaders, no score verdict against the pre-match target ladder, no tripwire state. A reader scanning the half-time doc could not tell at a glance whether the quarter just played met expectations.
- **Fix:** `format_quarter_break()` expanded to 18 lines: score verdict against the target ladder (≤15 down at Q1, ≤25 down at HT, ≤30 down at 3QT), top-3 quarter-AF leaders per side using `af_qN`, kick-share tripwire state, and the existing cumulative stat lines.
- **Commit:** `deaa918b0` (Harden live_analysis_pipeline.py after R11 Richmond vs St Kilda postmortem).

### 11.7 No auto analyst block at quarter transitions

- **Problem:** After Q2 there was no automatic Scientist or FootyStrategy commentary written to the Q3 or Q4 doc. Q3 and Q4 docs started on a blank page until the next per-poll block landed. Key players were missed in live analysis as a result — Jack Macrae (STK) finished R11 with 31 disposals and 7 tackles and was never named in any auto-block because nothing was forcing the pipeline to look at quarter-specific movers.
- **Fix:** Added `format_quarter_break_analyst()` plus `_write_quarter_break_analysis()`. Fires at every Q→Q transition and writes the top-5 quarter movers per side (by `af_qN`), key-player matchup tracking vs the pre-match disposal predictions (Short / Sinclair / Hill), and a forward-looking line for the next quarter. The transition handler is wrapped so an analyst-block exception cannot crash the main poll loop.
- **Commit:** `deaa918b0` (Harden live_analysis_pipeline.py after R11 Richmond vs St Kilda postmortem).

### 11.8 Skip-if-unchanged missing

- **Problem:** When FanFooty's feed stalled (no new numbers between two consecutive polls), identical blocks were written multiple times. The Q3 doc had four byte-identical "3 Qtr Time" blocks visible in succession — pure noise from the operator's perspective.
- **Fix:** `format_analysis_block()` now returns `(None, prev_state)` when score, disposals, tackles, AND hit-outs are all unchanged vs the previous poll. The main loop skips the write and the git commit in that case. The guard is intentionally disabled at quarter-break codes (`QT`, `HT`, `3QT`, `FT`) because at a break the routing change itself is the news.
- **Commit:** `deaa918b0` (Harden live_analysis_pipeline.py after R11 Richmond vs St Kilda postmortem).

### 11.9 Status string detection fragile

- **Problem:** `'Qtr Time'` (note: capitalised, not `"quarter time"`) was falling through to the Q1 default because the matcher only checked the lowercase long form. `'3 Qtr Time'` was at risk of matching a generic `"qtr time"` rule before the specific `"3qt"` rule. Unknown strings silently routed to Q1 instead of being skipped.
- **Fix:** All FanFooty status strings catalogued from actual R11 snapshots: `'Qtr Time'`, `'Half Time'`, `'3 Qtr Time'`, `'Final Siren'`, `'Full Time'`, plus the in-quarter `Qn HH:MM` forms. Check order made load-bearing — game-over first, three-quarter-time variants before generic quarter-time, in-quarter strings before break tokens. Unknown strings now return `None`; the main loop logs a warning and skips the cycle rather than misfiling.
- **Commit:** `deaa918b0` (Harden live_analysis_pipeline.py after R11 Richmond vs St Kilda postmortem).

### 11.10 Live docs bloating with repetitive blocks

- **Problem:** After the R11 game the live docs were 42KB–75KB of near-identical auto-pipeline blocks (Q1: 75KB, Q2: 42KB, HT: 19KB, Q3: 33KB, Q4: 27KB). Combination of the Final Siren misrouting (11.5) and the missing skip-if-unchanged guard (11.8). Effectively unreadable.
- **Fix:** Post-game pruning pass. For each doc: kept the first block of the quarter, the last block before the break, one instance per distinct transition header (`QUARTER BREAK` / `Qtr Time` / `Half Time` / `3 Qtr Time` / `Full Time` / `Final Siren`), and any block where the margin jumped ≥6 points since the last kept block. Out-of-scope end-of-game blocks were dropped entirely from earlier-quarter docs. Result: Q1 75KB→50KB, Q2 42KB→21KB, HT 19KB→4KB, Q3 33KB→9KB, Q4 27KB→11KB. The fixes in 11.4 and 11.8 prevent this from recurring on future games.
- **Commit:** `5ae20feee` (Prune repetitive auto-pipeline blocks from R11 live docs).

### 11.11 Pre-match brief missed key player (Macrae, R11)

- **Problem:** Jack Macrae (STK) finished R11 with 31 disposals and 7 tackles and was not tracked in the pre-match brief. Only three players were tracked explicitly in `KEY_PLAYERS` (Short / Sinclair / Hill). A tackling midfielder who was also a high-handball-receive distributor turned out to be the game's dominant presence and went undetected for the full 2.5 hours of live coverage.
- **Fix (process, not code):** Pre-match tracking list extended to top-5 players per side. Auto analyst blocks now surface the top-5 quarter-AF movers instead of the top-3, on the principle that the brief's named list will always be incomplete and the auto-layer needs to catch the rest.
- **Commit:** n/a (process change; the structural surfacing of top-5 movers is included in `deaa918b0`, but the brief-authoring policy is documented in the R11 postmortem rather than enforced in code).

---

## 12. Planned improvements

A frank list of what is planned but not yet shipped. Items are marked aspirational where they require new wiring (subagent invocation, secondary data sources, scheduled jobs) rather than just a code change.

### 12.1 LLM-generated quarter-break commentary

The current auto analyst block is rule-based — top-N by `af_qN`, matchup delta vs pre-match prediction, score margin against the target ladder. Planned: invoke the Scientist / FootyStrategy subagents at each quarter break to write actual prose commentary in place of the structured tables.

- **Blocked by:** subagent invocation from inside the polling loop is not currently wired. Would need an in-process call path (or a deferred queue that a human triggers at break) so the loop is not held up by a multi-second LLM round-trip.
- **Status:** aspirational.

### 12.2 Inside-50 and clearance tracking

Inside-50s, clearances, and contested possessions are the most important possession-chain stats in modern AFL analysis. None of them are in the FanFooty per-player snapshot schema (see §9.2). The live pipeline currently uses kick-share as a labelled territory proxy.

- **Planned:** investigate whether the FanFooty commentary event stream can be parsed to approximate inside-50 counts in-game; alternatively, pull authoritative numbers from a secondary source post-game and back-fill the auto-blocks.
- **Why this is now load-bearing:** Skeptic (§2.4) will block any FootyStrategy tripwire that references inside-50 differential because it is unobservable in this repo's live schema. Until §12.2 ships, every tripwire that wants to use inside-50s must be reframed in observable terms (kick-share proxy, post-game backfill from afltables) or it will fail Skeptic's tripwire-observability probe. This makes §12.2 a prerequisite for a class of tactical recommendations that are otherwise routine in modern AFL analysis.
- **Status:** unstarted.

### 12.3 Pre-match brief auto-population

Originally planned as a one-off script. Now implemented as the **BriefBuilder agent** (Sonnet) — see §2.4 and §6.1. BriefBuilder takes two team names and a round, pulls the H2H history from `data/matches/`, the season form from per-player CSVs, and the model predictions from `data/prediction/`, then writes the data skeleton with `**[data]**` tags and source-file annotations. FootyStrategy fills the interpretation layer between BriefBuilder's tables; DataSentinel gates the commit.

- **Status:** agent file at `.claude/agents/BriefBuilder.md` is defined; integration into the pre-match flow per §6.1 is the next deliverable, gated on DataSentinel landing first. Ship order per §13: DataSentinel first, then BriefBuilder, then Skeptic.

### 12.4 Player position tagging

Player CSVs lack a `position` column (§9.5). All position-based analysis — forward vs midfielder vs defender splits, `backtest_by_position` — currently buckets every row as `"Unknown"`.

- **Planned:** enrich player data with a position lookup table sourced from afltables or a similar reliable source. Likely an extension to `player_scraper.py` that parses the position field already present on player profile pages, plus a back-fill pass for the existing 13,000-player corpus.
- **Status:** unstarted.

### 12.5 Model bias correction per team

The walk-forward backtest surfaces a +0.53 disposals-per-player over-prediction for Richmond on the 2026 season. Other teams have smaller but non-zero biases. Consumers currently have to mentally shade the headline number for Richmond.

- **Planned:** apply a per-team bias correction factor at predict time, sourced from the most recent `backtest_by_team_<ts>.csv`. Apply only to teams where the bias is statistically distinguishable from zero on the season-to-date sample.
- **Status:** identified, not yet implemented.

### 12.6 Smoke test expansion

`scripts/smoke_test_live_pipeline.py` currently exercises four assertions against one R11 Full Time snapshot.

- **Planned:** build a snapshot library of edge cases — stalled feed, unknown status string, first poll of a game, quarter-transition race condition, mid-game schema drift — and exercise all of them as a regression suite before any change to `classify_status` or the skip-if-unchanged guard.
- **Status:** partially started (the smoke test exists; more snapshot cases needed).

### 12.7 Automated weekly data refresh

Player data, match results, and prediction files need updating after every round. Currently this is a manual `bash refresh_and_rank.sh` invocation by the operator.

- **Planned:** scheduled weekly refresh script with validation gates — row counts, date-range checks, schema diffs — before any new CSV is allowed to overwrite the existing file. A failed validation should leave the existing data in place and surface a clear "refresh blocked because X" message.
- **Status:** unstarted.

---

## 13. Agent guardrails and sequencing

The extended council (DataSentinel + BriefBuilder + Skeptic, §2.4) introduces three new agents with distinct trust boundaries. The guardrails below are non-negotiable: they exist to make the architecture honest rather than ceremonial.

### 13.1 Output contracts

- **DataSentinel must emit machine-readable JSON, never prose.** The pre-commit hook is the consumer; a hook cannot parse paragraphs. Schema: `{status: "PASS" | "FAIL", violations: [{kind, line_number, expected, actual, source_file}]}`. Any prose explanation is for the human reviewing the hook output, not the hook itself. If DataSentinel ever returns prose-only, the hook silently passes and the gate has failed.
- **BriefBuilder must never write `**[data]**` tags it hasn't verified.** Tagging a number is a claim that the number was read from the cited source CSV. BriefBuilder's output goes through DataSentinel like every other doc — it cannot self-certify. If BriefBuilder is unsure of a number, it leaves the slot empty with a `<!-- TODO Scientist: pull X -->` placeholder rather than tag a guess.
- **Skeptic never silently modifies the doc.** Its output is critique only — `PASS / PASS_WITH_CONCERNS / BLOCK` with a per-concern explanation. The author (FootyStrategy or Scientist) decides what to incorporate. This preserves the audit trail: every change to a published doc is attributable to a named author, not to an adversarial reviewer who quietly rewrote a paragraph.

### 13.2 Inherited rules

- **All three new agents inherit the CLAUDE.md verification rule with no exceptions.** No agent — regardless of model, regardless of role — is exempt from "verify against the data file before writing a number." DataSentinel is the runtime enforcement; BriefBuilder is the most-likely violator (it is doing the tagging); Skeptic is the second-line auditor that catches what DataSentinel's mechanical comparison misses (e.g. a number that matches the CSV but cites the wrong file).
- **No coach names.** FootyStrategy's coach-anonymity rule extends to all agents that touch published docs. DataSentinel flags coach-name violations as a `kind: "coach_name_violation"` in its JSON output.
- **No business decisions.** Threshold setting, model-vs-source preference, tier overrides — these escalate to the human author. No agent makes a stakeholder-implicated call autonomously.

### 13.3 Recommended ship order

Ship the council in this order. Each rung depends on the one below it.

1. **DataSentinel first.** It is the closure on Gap 1 (CLAUDE.md is aspirational policy, not runtime-enforced). Until DataSentinel exists, every commit is a trust exercise; once it exists, every commit is gated by a deterministic check. This is the highest-leverage agent in the council.
2. **BriefBuilder second.** Only after DataSentinel is in place. Reason: BriefBuilder is a fast tagger of `**[data]**` numbers — if its output is not gated, it will accelerate the rate of unverified claims entering the repo. Shipping BriefBuilder before DataSentinel is a regression on the existing manual-Scientist baseline.
3. **Skeptic third — after ~2 months of manual skepticism.** Before automating adversarial review, the team should manually adversarially-review FootyStrategy briefs for a sustained window and record what was caught. That catalogue becomes Skeptic's calibration set. Shipping Skeptic earlier risks an agent that either (a) catches the same things FootyStrategy already self-catches, or (b) BLOCKs on style differences rather than methodology violations. The 2-month manual window proves the defects are real and recurrent before any agent is wired to flag them.

### 13.4 Known limitations carried forward

These are the Council's gaps that were *not* closed by the new agents and remain on the limitations ledger.

- **Gap 6: `DOC_BASE` hardcoded per pipeline invocation.** `live_analysis_pipeline.py` requires the operator to edit `DOC_BASE = "richmond-vs-stkilda-round-11-2026"` for each new fixture. There is no auto-detection of which match-slug to route to. A wrong slug routes blocks into the wrong fixture's docs and is only caught by the operator's eye.
- **Gap 8: orchestration is implicit.** The five-agent pre-match flow described in §6.1 (BriefBuilder → Scientist → FootyStrategy → DataSentinel → optionally Skeptic) is documented but not enforced by any orchestrator. Each step is launched manually via `@"Agent (agent)"` in Claude Code. Skipping a step is possible; reordering is possible; running steps out of dependency order is possible. The architecture is descriptive, not enforced. A proper orchestrator (or even a shell script that walks the sequence) is open work.

---

## Appendix — load-bearing references

- `CLAUDE.md` — operational policy. The data-verification rule wins over everything in this document.
- `.claude/agents/Scientist.md` — full Scientist prompt and contract.
- `.claude/agents/FootyStrategy.md` — full FootyStrategy prompt and contract.
- `.claude/agent-memory/Scientist/MEMORY.md` — index of Scientist's persistent memory files.
- `.claude/agent-memory/FootyStrategy/MEMORY.md` — index of FootyStrategy's persistent memory files.
- `docs/coaches-strategy-corner/README.md` — house rules for strategy briefs; live-pipeline operator recipe.
- `docs/news/README.md` — house rules for news articles; tag glossary.
- `docs/footy-ai-chatbot-phase2.md` — design spec for "The Crumb" multi-agent system.
