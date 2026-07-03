# SuperCoach-VIA — Architecture

*A meta/architecture document. This is not a council-authored brief or news article, so it does not carry a `council-pipeline` provenance stamp.*

*Last updated: 2026-07-02.*

---

## Table of contents

1. [Repo overview](#1-repo-overview)
2. [Weekly refresh pipeline](#2-weekly-refresh-pipeline)
3. [Agent council](#3-agent-council)
4. [Known recurring problems](#4-known-recurring-problems)
5. [Agent prompts](#5-agent-prompts)

---

## 1. Repo overview

### What this repo does

SuperCoach-VIA is an AFL statistics, prediction, and editorial pipeline. It maintains a **complete AFL match and player statistics history (1897–present)**, derives all-time and season leaderboards from that corpus, produces a **leak-proof disposal prediction model** for the upcoming round, **backtests** that model week-over-week, and publishes a set of human-readable analysis documents. A **six-agent LLM council** sits on top of the numeric pipeline to author and gate any interpretive, reader-facing writing (news articles, Hall of Fame stat pages, pre-match briefs, weekly recaps).

The core design principle runs through the whole repo: **numbers are derived deterministically from `data/` and verified before they are presented.** Prose is downstream of verification, never the other way around.

### Data sources

| Source | What it provides | How it enters the repo |
|--------|------------------|------------------------|
| AFLTables (match pages) | Match results, team scores, quarter scores, rounds | `scrapers/game_scraper.py` (`MatchScraper`), delta mode |
| AFLTables (player profile + team all-time pages) | Per-player per-game performance rows; roster discovery | `scrapers/player_scraper.py` (`PlayerScraper`), targeted mode |
| DraftGuru / external draft grades | Draft pick grades (for list-quality articles) | WebFetch, verified into `[data]` tags manually |
| afl.com.au / zerohanger | Contract / news context for briefs | WebFetch (other domains are blocked; live Python scrapes often hit SPA shells and fall back to verified fixtures) |

**Canonical on-disk data (never hand-edited — only the Scientist / scrapers write here):**

- `data/matches/matches_<year>.csv` — match results and team scores.
- `data/player_data/<surname>_<firstname>_<DDMMYYYY>_performance_details.csv` — one file per player, one row per game. The `DDMMYYYY` is date-of-birth, used to disambiguate same-name players.
- `data/prediction/next_round_<N>_prediction_<timestamp>.csv` — model output for the upcoming round.
- `data/prediction/backtest/backtest_summary_<timestamp>.csv` + `backtest_by_team_*` / `backtest_by_position_*` / `prediction_vs_actual_round_*` — walk-forward backtest outputs.
- `data/top100/all_time_top_100.csv` and root `all_time_top_100.csv` — all-time aggregated rankings.
- `docs/hall-of-fame/_stat_leaders.json` — machine-computed career + single-season leaderboards (the ground truth that drives the HOF pages).

### Output surfaces

These are the reader-facing artifacts the pipeline keeps fresh:

- **`README.md`** — the front door. Carries an auto-updated **Eval results** section and a **news block hard-limited to 2 entries** (`<!-- NEWS-LATEST-START -->` / `<!-- NEWS-LATEST-END -->`). The full news archive lives in `docs/news/README.md`.
- **`docs/banner.svg`** — headline eval numbers rendered as a banner; regenerated alongside the README eval surface.
- **Season docs** — `docs/afl-season-2026.md`, `afl-team-analysis-2026.md`, `afl-finals-2026.md`, `afl-brownlow-2026.md`, `afl-stat-leaders-2026.md`, `afl-team-profiles.md`, `afl-insights.md`.
- **Predictions & backtest** — `docs/afl-predictions-2026.md`, `docs/afl-backtest-2026.md`, and the weekly cheat sheets under `docs/weekly/round-<N>-2026.md` plus the stable-link `docs/weekly/round-current-2026.md`.
- **Hall of Fame pages** — the hub `docs/hall-of-fame-stat-leaders.md` plus the per-stat leaderboards `docs/hall-of-fame-stat-{games,goals,disposals,marks,tackles,brownlow,clearances,contested,hitouts,goalassists,kicks-handballs,single-season}.md`, and `docs/hall-of-fame-top100.md`. Editorial HOF pages (captains, courageous, forgotten-heroes, dynasties, indigenous, careers-cut-short, coaches) are **manually curated and never auto-updated**.
- **News desk** — `docs/news/` (articles, archive index, sentinel logs). Each published article carries a `council-pipeline` provenance stamp enforced by the pre-commit hook.
- **Charts** — `assets/charts/` and the HOF records charts under `docs/hall-of-fame/`.

### Enforcement substrate

- `.githooks/pre-commit` → `scripts/check-council-stamp.sh`: a **deterministic** pre-commit gate. Git is wired to it via `git config core.hooksPath .githooks` so every fresh clone inherits the gate (the older `.git/hooks/pre-commit` was local-only and lost on clone — see Known Problems). It greps staged Markdown: any `docs/news/*.md` or `docs/hall-of-fame-stat-*.md` (and any HOF page that already declares itself a council doc) must carry a `<!-- council-pipeline: ... -->` stamp whose DataSentinel and Skeptic lines both read PASS. It never invokes an LLM — it only enforces the recorded verdict.
- `scripts/log-agent-turn.sh`: appends structured JSONL audit lines to `.claude/audit/YYYY-MM-DD.jsonl` (gitignored; only `.gitkeep` is committed).
- `scripts/check_hof_numbers.py`: a **deterministic HOF numeric gate** that re-reads `_stat_leaders.json` and confirms every rank-1 total in the HOF sub-pages matches the JSON ground truth. This replaced LLM arithmetic for rank-1 HOF rows.

---

## 2. Weekly refresh pipeline

**Cadence.** The pipeline runs weekly, after the round has fully settled. A round completes Sunday; data is settled by Tuesday. `scripts/weekly_refresh.sh` is the top-level entry point and is intended to run **Tuesday evening UTC (Wednesday morning AEST)**:

```
# Wednesday 6 AM AEST = Tuesday 8 PM UTC
0 20 * * 2 cd /home/abhi/git/SuperCoach-VIA && bash scripts/weekly_refresh.sh
```

`weekly_refresh.sh` runs `set -euo pipefail`, logs every step to `.claude/audit/weekly_refresh_<date>.log`, and orchestrates two big phases: **Phase 1 is the entire data+model+season-docs pipeline (`refresh_and_rank.sh`)**, and **Phase 2 onward is the Hall of Fame + cheat-sheet + insights layer**.

### Phase 1 — `refresh_and_rank.sh` (data → model → season docs → push)

`weekly_refresh.sh` calls `bash refresh_and_rank.sh` first. `refresh_and_rank.sh` is a 6-step sprint that owns its own git add/commit/push:

| Step | Script | What it does | Key inputs → outputs |
|------|--------|--------------|----------------------|
| 1 | `refresh_data.py` | Delta match scrape + **targeted** player scrape (only active players `last_game >= ACTIVE_SINCE`, plus newly-discovered debuts — avoids re-fetching ~12.5k retired players). Runs post-write self-audits: `audit_match_rounds` (catches silently truncated rounds like the "R10 2026" bug) and `audit_player_career_totals` (reconciles updated players against their AFLTables profile). Both are **WARNING-only**, never abort. | AFLTables → `data/matches/*.csv`, `data/player_data/*.csv` |
| 2 | `top_players_comprehensive.py` | Recomputes and formats the all-time top 100. | player_data → `all_time_top_100.csv`, `data/top100/all_time_top_100.csv` |
| 3 | `python -m supercoach.prediction` | Predicts next-round disposals with the leak-proof model. Auto-detects current year + next round from latest player data. Invoked as a module so its package-relative imports resolve. | player_data → `data/prediction/next_round_<N>_prediction_<ts>.csv` |
| 4 | `backtest.py` | **Incremental** walk-forward backtest. Detects the last complete run (the last `backtest_summary_*.csv`), starts from the next round, runs to `--end-round auto`. | player_data + predictions → `data/prediction/backtest/backtest_summary_*.csv`, `backtest_by_team_*`, `backtest_by_position_*`, `prediction_vs_actual_round_*` |
| 5 | `refresh_readme.py` + `docs/hall-of-fame/compute_stat_leaders.py` + `docs/hall-of-fame/generate_records_charts.py` | Embeds fresh prediction + backtest CSVs into the prediction/backtest docs; recomputes HOF stat leaders JSON; regenerates HOF charts so no data-derived image goes stale. | CSVs → `docs/afl-predictions-2026.md`, `docs/afl-backtest-2026.md`, `_stat_leaders.json`, charts |
| 6 | `git add` (explicit allowlist) + commit + push | Stages only the deliberate list of season docs / charts / CSVs (never `git add .`, to avoid sweeping in scratch CSVs), commits, pushes to `origin/main`. | → `origin/main` |

Season docs committed by Phase 1: `afl-season-2026.md`, `afl-team-analysis-2026.md`, `afl-finals-2026.md`, `afl-brownlow-2026.md`, `afl-stat-leaders-2026.md`, `afl-predictions-2026.md`, `afl-backtest-2026.md`, `afl-team-profiles.md`, `afl-insights.md`, `hall-of-fame-top100.md`, plus `assets/charts/` and the top-100 CSVs.

### Phase 1b — eval surface refresh

After Phase 1, `weekly_refresh.sh` runs `scripts/update_eval_surface.sh`, which re-derives the already-verified backtest figures (using the same merge logic as the backtest doc — merge all per-run CSVs, dedupe by `(year, round)`) and updates only the README **Eval results** table and `docs/banner.svg`. It never touches the news block. Idempotent.

### Round detection

Between Phase 1 and Phase 2, the script detects the next round **from the CSV `prediction.py` just wrote** (`ls data/prediction/next_round_*_prediction_*.csv | sort | tail -1`), not from stale on-disk state. This `ROUND` value feeds the cheat sheet and the FootyStrategy recap.

### Phase 2a — weekly cheat sheet

`scripts/generate_weekly_cheat_sheet.py` reads the latest prediction CSV and writes `docs/weekly/round-<N>-<year>.md` plus the stable link `docs/weekly/round-current-<year>.md`.

### Phase 2b — Hall of Fame stat-leaders refresh (deterministic)

This is the HOF pipeline, run as four deterministic steps with a hard gate at the end:

1. `docs/hall-of-fame/compute_stat_leaders.py` — recompute career + single-season stat totals from the freshly-updated player corpus → `_stat_leaders.json` (13 career categories, 20 leaders each, plus single-season; `meta.player_count`).
2. `docs/hall-of-fame/generate_records_charts.py` — regenerate stat records charts.
3. `scripts/update_hof_pages.py` — **deterministic string replacement** (no LLM). Reads `_stat_leaders.json` and propagates values into sentinel-marked lines: hub rows (`<!-- HOF-HUB:<key> -->`), rank-1 sub-page rows (`<!-- HOF-TOP:<key> -->`), and full ranks-1-20 table bodies between `<!-- HOF-TABLE-START:<key> -->` / `<!-- HOF-TABLE-END:<key> -->` markers (for the eight clean single-stat categories in `_FULL_TABLE_CATS`). Also refreshes each page's `*Last refreshed:` date and its `DataSentinel: PASS @` stamp.
4. `scripts/check_hof_numbers.py` — **deterministic numeric gate**. Re-reads `_stat_leaders.json`, extracts each rank-1 total from the HOF-TOP sentinel rows, compares. **Exit 1 aborts Phase 2b** (`weekly_refresh.sh` halts and refuses to ship) if any number drifts.

### News-block enforcement

Before Phase 3, `enforce_news_limit()` trims the README news block to its **2-most-recent-entries hard limit** (matching the rule in `CLAUDE.md`).

### Phase 3 — FootyStrategy weekly recap

`weekly_refresh.sh` invokes the **FootyStrategy agent** via the `claude` CLI (`--agent FootyStrategy --model sonnet --permission-mode bypassPermissions`, tools limited to `Read,Write,Edit,Glob,Grep`). It writes a tight 150–200 word `## Round <N> — Week in Review` section into `docs/afl-insights.md`, grounded only in the freshly-updated stat-leaders / season / predictions / cheat-sheet docs, tagging stats `[data]`. Hard rules forbid it from touching navigation, links, news, or HOF files.

### Phase 4 — commit + push Phase 2/3 outputs

`git add` (explicit allowlist: README, banner, insights, weekly, all HOF stat pages, `_stat_leaders.json`, charts) → commit → `git push origin main`.

### Excluded from auto-update (manually curated only)

`docs/hall-of-fame-captains.md`, `hall-of-fame-courageous.md`, `hall-of-fame-forgotten-heroes.md`, `hall-of-fame-dynasties.md`, `hall-of-fame-indigenous.md`, `hall-of-fame-careers-cut-short.md`, `hall-of-fame-coaches.md`, and `docs/news/README.md`.

### One-line data flow

```
AFLTables ─▶ refresh_data.py ─▶ data/matches, data/player_data
                                     │
                 ┌───────────────────┼─────────────────────────┐
                 ▼                    ▼                          ▼
      top_players_comprehensive   supercoach.prediction     compute_stat_leaders.py
                 │                    │                          │
                 ▼                    ▼                          ▼
        all_time_top_100.csv   next_round_*.csv  ──▶ backtest.py   _stat_leaders.json
                                     │                 │              │
                                     ▼                 ▼              ▼
                          refresh_readme.py   backtest_summary_*  update_hof_pages.py
                                     │                 │              │ (deterministic)
                                     ▼                 ▼              ▼
                    afl-predictions / afl-backtest docs      HOF stat pages
                                     │                                │
                                     ▼                                ▼
                       update_eval_surface.sh              check_hof_numbers.py (GATE)
                                     │                                │
                                     ▼                                ▼
                        README eval + banner.svg          generate_weekly_cheat_sheet.py
                                                                      │
                                                                      ▼
                                                     FootyStrategy → afl-insights.md → push
```

---

## 3. Agent council

Six LLM agents plus the human operator. The operator is the named, accountable owner (Australia's AI Ethics Principle 8); every agent is a delegate, never a replacement for that accountability.

### The canonical chain

For every brief or news article:

```
BriefBuilder
   → DataSentinel (Pass 1: data skeleton)
      → FootyStrategy
         → DataSentinel (Pass 2: full doc, incl. all interpretation prose)
            → Skeptic
               → Gaffer (SHIP)
                  → (optional) Codex blind read
```

**DataSentinel runs twice.** Pass 1 gates the numeric skeleton before FootyStrategy is allowed to interpret it; Pass 2 gates the whole document — every interpretive sentence included — before the Skeptic sees it. A Pass-1 PASS is **not** final clearance; the ship is gated on Pass 2.

On a clean PASS through the chain, Gaffer stamps the doc:

```
<!-- council-pipeline: BriefBuilder@<sha>, Scientist@<sha>, FootyStrategy@<sha>,
     DataSentinel:PASS(pass1)@<ts>, DataSentinel:PASS(pass2)@<ts>,
     Skeptic:PASS@<ts>, Gaffer:SHIP@<ts> -->
```

The pre-commit hook then refuses the commit unless that stamp exists and both gating lines read PASS.

### Roles

**BriefBuilder** (`model: sonnet`; tools: Read, Grep, Glob, Write, Edit)
- **Owns:** the *data skeleton* of a pre-match brief. Given two teams and a round, it pulls season form, the head-to-head ledger, model predictions, and a top-5-per-side tracking list, and writes `[data]` tags annotated with their source file.
- **Cannot:** write the interpretation layer — it leaves `<!-- FOOTYSTRATEGY INSERT -->` placeholders. Its output is gated by DataSentinel (Pass 1) before anyone builds on it.

**DataSentinel** (`model: haiku`; tools: Read, Grep, Glob, Bash, Write, Edit)
- **Owns:** the verification gate. It walks every `[data]` tag, confirms it against the source CSV named in the doc's methodology paragraph, flags untagged numbers, coach-name violations, and FanFooty schema violations. Emits machine-readable JSON for the hook.
- **Cannot:** author interpretation, or be trusted for arithmetic it does by "reading" a CSV in prose. **Its arithmetic is not authoritative** — deterministic Python re-reads (`check_hof_numbers.py`) exist precisely because the LLM mis-sums CSVs. Use it as a tag-walker and line-locator; verify disputed sums in pandas.

**FootyStrategy** (`model: opus`; tools: all)
- **Owns:** the interpretation layer — pre-game strategy, live-match inputs, post-match analysis, and the weekly `afl-insights.md` recap. Fills the BriefBuilder placeholders with data-grounded prose.
- **Cannot:** invent numbers, or touch navigation / links / news / HOF files in the weekly job. Its full draft goes back through DataSentinel (Pass 2) and then Skeptic.

**Skeptic** (`model: opus`; tools: all)
- **Owns:** adversarial review of the FootyStrategy draft. Probes three things: tripwire observability in this repo's data, caveat-hierarchy fidelity versus the upstream Scientist findings, and lens-tension smoothing (claims quietly rounded up). Emits `PASS` / `PASS_WITH_CONCERNS` / `BLOCK`.
- **Cannot:** silently modify the doc. A `BLOCK` halts the ship — Gaffer may not override it.

**Scientist** (`model: opus`; tools: all)
- **Owns:** all code, the model, and — critically — **all `[data]`-tagged numbers**. Only the Scientist derives numbers from `data/`. Owns the load-bearing ML invariants: strict temporal cutoff in `LeakProofPredictor`, GroupKFold-by-player, seeded determinism. Owns data-defect fixes (dedup bugs, truncation, backfill).
- **Cannot:** be bypassed on numbers or model logic. Gaffer flags ML-invariant violations but never rewrites ML logic without Scientist sign-off.

**Gaffer** (`model: opus`; tools: all)
- **Owns:** *process*, not *truth*. Sequences the chain, enforces handoffs, maintains the harness backlog, writes the provenance stamp on PASS, applies presentation polish downstream of verification, commits, pushes.
- **Cannot:** override / soften / re-run-until-green a `DataSentinel: FAIL` or `Skeptic: BLOCK`; author or edit a `[data]` number; mark/simulate/infer a verdict; edit anything under `data/`; `git push --force`. When a request is user-initiated, the USER REQUEST WAIVER tells Gaffer to execute it and report Done / Not-done rather than block on preflight.

### Two lanes

The council governs **editorial** output (briefs, news, HOF pages, recaps). The **numeric weekly refresh** (Section 2) is mostly deterministic scripts with two automated council touchpoints: the FootyStrategy recap (Phase 3) and the DataSentinel stamp that `update_hof_pages.py` refreshes on the HOF pages (backed by the deterministic `check_hof_numbers.py` gate, not by LLM arithmetic).

---

## 4. Known recurring problems

A frank list of what has actually broken over the last ~8 weeks, with root cause and the fix applied. Several of these are why the deterministic gates exist.

### 4.1 Pendlebury / HOF staleness (repeated)

- **Symptom:** HOF stat pages (notably a Pendlebury games/disposals leaderboard row) kept showing stale totals after a refresh — the JSON updated but the published Markdown did not, or only the rank-1 row moved while the hand-written ranks 2–20 drifted.
- **Root cause:** HOF numbers were once propagated by LLM/manual editing, and only the rank-1 sentinel was automated; the rest of the table and the kicks/handballs page were hand-written and drifted out of sync with `_stat_leaders.json`.
- **Fix:** `scripts/update_hof_pages.py` now regenerates full ranks-1-20 table bodies deterministically from JSON for the clean categories (`_FULL_TABLE_CATS`), plus hub + rank-1 sentinels for all. `scripts/check_hof_numbers.py` is a hard gate in Phase 2b that aborts the refresh if any rank-1 total drifts from JSON. **Update (arch-review pass):** the deferred TODO for `career_disposals` / `career_goals` is closed — the TODO's "multi-column format" premise was stale (both pages use the standard 7-column layout the existing builder already renders), so both are now in `_FULL_TABLE_CATS` with unit-test coverage in `tests/unit/test_update_hof_pages.py`. Regeneration for them activates once the two doc pages gain `<!-- HOF-TABLE-START/END -->` markers (a doc edit, deliberately not bundled with the code change). Remaining gap: `career_kicks`/`career_handballs` — a genuine two-leader-per-row format — still awaits custom handling.

### 4.2 Scraper timing — ran before the round was played

- **Symptom:** A refresh produced predictions/backtests against an incomplete round because it ran before all matches had settled.
- **Root cause:** Cadence ambiguity — the round completes Sunday but stats are not fully settled until Tuesday.
- **Fix:** `weekly_refresh.sh` is scheduled for **Tuesday 8 PM UTC / Wednesday 6 AM AEST**, after settlement. Round is detected *after* Phase 1 from the CSV `prediction.py` just wrote, not from stale disk state. The backtest is incremental (`--start-round = last_complete + 1`), so a premature/partial round is not silently baked in as "complete."

### 4.3 Gaffer deadlock on push-to-main

- **Symptom:** Gaffer refused to push to `origin/main` on explicit user requests, blocking on preflight/permission checks and demanding extra confirmation for work the user had already asked for.
- **Root cause:** The original Gaffer prompt treated the pre-commit gate / audit-log substrate as a *deploy blocker* ("do not deploy until Gap 1 and Gap 2 are live"), which turned into a hard refusal to operate.
- **Fix:** The **USER REQUEST WAIVER** at the top of `Gaffer.md` now overrides everything below it: on a user request, Gaffer executes, pushes, and reports **Done / Not done** with specific blockers — it does not refuse or demand re-confirmation. Preflight is now warn-and-continue, not refuse. (Corroborated by recent commits `Restore Gaffer with user-request waiver` and `Simplify Gaffer agent: remove preflight deadlocks`.)

### 4.4 LLM DataSentinel arithmetic failures

- **Symptom:** DataSentinel (haiku) reported false mismatches — in one run, four numbers it "re-summed" from a CSV were wrong, producing false FAILs on correct data.
- **Root cause:** An LLM asked to sum a CSV column in prose is unreliable; it is a suggestion, not a gate.
- **Fix:** Deterministic checks replaced LLM arithmetic wherever the data allows. `check_hof_numbers.py` re-reads JSON and compares in Python for rank-1 HOF rows. General rule (recorded in Gaffer memory): treat DataSentinel as a tag-walker / line-locator; re-measure any disputed number in pandas before acting on a FAIL. Canonical games metric is `max(rowcount, games_played.max())`, not naive `len(df)`, to avoid false FAILs.

### 4.5 Sidebottom / drawn-GF dedup bug

- **Symptom:** A player's career totals were under-counted after a "dedup" pass; drawn-and-replayed finals (e.g. a drawn Grand Final plus its replay) lost a legitimate game row.
- **Root cause:** Deduplication keyed on `(year, round, opponent)` treated the drawn match and its replay as duplicates and deleted one — but they are two distinct, real games. A related class: a 2024-finals backfill appended real-dated rows while leaving `YYYY-03-01` placeholder rows, double-counting finals stats (241 dup rows).
- **Fix (Scientist-owned):** the dedup key must not collapse drawn-and-replayed finals; placeholder rows must be removed when real-dated rows are appended. These are data-defect fixes that require Scientist sign-off — Gaffer flags, does not rewrite.

### 4.6 Leaderboard drift (stale hand-written tables)

- **Symptom:** The stat-leaders hub and per-page tables (kicks/handballs especially, and ranks 2–20 generally) showed numbers that no longer matched the computed JSON.
- **Root cause:** Only rank-1 was automated; the rest was hand-maintained Markdown that nobody re-derived each week.
- **Fix:** Full-table deterministic regeneration from `_stat_leaders.json` via `update_hof_pages.py` for the clean categories, gated by `check_hof_numbers.py`. Same root fix as 4.1; listed separately because it also affects the hub page and the two-leader-per-row kicks/handballs format still awaiting custom handling.

### 4.7 `weekly_refresh.sh` not calling `refresh_and_rank.sh` as Phase 1

- **Symptom:** The weekly job ran the HOF / cheat-sheet / insights layer against **stale** match and player data, because the data+model refresh had not run first.
- **Root cause:** The two scripts had drifted apart — `weekly_refresh.sh` did not invoke `refresh_and_rank.sh`, so Phase 2 read yesterday's CSVs.
- **Fix:** `weekly_refresh.sh` Phase 1 now explicitly runs `bash refresh_and_rank.sh` before anything else, and round detection reads the CSV that run just wrote. The two-phase contract is documented in the script header.

### 4.8 False DataSentinel FAILs from the parallelisation race

- **Symptom:** When multiple council agents committed to `main` concurrently (up to 7 at once), git index races and `index.lock` contention produced spurious failures; verification sometimes ran against a half-written or superseded file.
- **Root cause:** Parallel agents pushing to the same branch collide on the git index; "command succeeded" is not the same as "the right content landed."
- **Fix / operating rule (Gaffer memory):** wait for `index.lock` to clear, serialize commits to `main`, `pull --rebase` before push, and **verify by content, not by command exit code**. Concurrent rebuild agents can leave malformed rows, so re-read the file after a parallel run.

### 4.9 Architecture-review hardening (2026-07-03)

An external expert review (Ng / Willison / Rajpal, plus a quality/sales/innovation pass) surfaced that the numbers were verified but the *process* was not. Addressed in the architecture-review commit (`Arch review: prompt lint, gate config promotion, stamp verifiability, commit serialisation`):

- **Q1 — provenance stamp forgeable in practice → FIXED (hook side).** The pre-commit gate greps for `DataSentinel: PASS` text, which any agent can type. It now cross-checks that text against a content-hash-keyed audit record. `scripts/council-content-hash.sh` is the single source of truth for the canonical hash (strips the volatile stamp + trust-badge lines and trailing blanks, so the pre-stamp hash DataSentinel records equals the post-stamp hash the hook recomputes). `scripts/record-sentinel-verdict.sh` is the producer helper DataSentinel calls to write `.claude/audit/sentinel-<hash>-<ts>.json`. `scripts/check-council-stamp.sh` now fails a PASS stamp whose content hash has no matching PASS record (tamper/stale-stamp), and — under `AUDIT_ENFORCE=1` — a stamp with no record at all. Default is warn-not-block so it lands safely before the DataSentinel prompt-side wiring; flip `AUDIT_ENFORCE` when the producer is live. Tests: `tests/unit/test_council_stamp_audit.py`. *Follow-up (backlog): wire `record-sentinel-verdict.sh` into the DataSentinel agent turn.* Note: audit records are gitignored (local-only), which is acceptable for the single-machine pipeline.
- **Q5 — prompt hygiene → FIXED.** `Scientist.md` had two conflicting frontmatter blocks (opus vs sonnet) collapsed by `\r` line endings; now a single opus block with the correct data-scientist description. All six agent files had `\r` endings (stripped) and ~130 lines of duplicated memory boilerplate (trimmed to a per-agent delta; the general memory spec is inherited from the session). `Gaffer.md`'s `description:` no longer carries the "do not deploy until Gap 1/2" deadlock text, and the waiver is now an explicit precedence table (overrides preflight friction, never overrides FAIL/BLOCK or the `[data]` prohibition). `Skeptic.md`'s aspirational "runs twice → identical critique" replaced with verdict-*category* stability + machine-readable JSON. Audience blocks + a trust-badge rule added to `Gaffer.md`/`FootyStrategy.md`.
- **Q7 — gate references promoted out of mutable agent memory → FIXED.** Coach-anonymity list → `config/coach_names.txt`; FanFooty schema facts → `config/fanfooty_schema.yaml` (read-only, reviewed config). `DataSentinel.md` and `Skeptic.md` now read these files; the agent-memory notes are demoted to advisory rationale. Tests: `tests/unit/test_gate_config.py`.
- **Q6 — concurrency fixed by mechanism, not memory → FIXED.** `scripts/git_commit_safe.sh` (flock wrapper) added and wired into `refresh_and_rank.sh` + `scripts/weekly_refresh.sh` for automated commits; the "only Gaffer commits, serialise" rule is now in the Gaffer prompt body under WHAT YOU MUST NEVER DO, not just memory.
- **Q3 — untagged player-stat numbers → FIXED (policy).** `DataSentinel.md` now treats a player-stat-shaped untagged number as a hard FAIL (was advisory); the verdict rule counts `untagged_numbers_flagged` in the PASS condition. Verification is no longer opt-in via tagging.
- **Q2 (partial) — HOF full-table TODO → FIXED; hard-abort audits + pandera still backlog.** `career_disposals`/`career_goals` closed (see 4.1). Still backlog: promoting Phase-1 audits (`audit_match_rounds`, `audit_player_career_totals`) to hard aborts, and a `pandera`/JSON-Schema gate on CSV write.
- **Also added (this pass):** a DataSentinel hard rule that every numeric check must be an executed pandas one-liner quoted verbatim (never in-token arithmetic; unverifiable ⇒ UNVERIFIED, not PASS) — the same root cause as 4.4.

**Deferred (backlog, not in this pass):** I1 executable data-tags / render-time numbers (design-first), I3 council eval golden-set, Q4 haiku→deterministic rearchitect, Q8 deterministic Skeptic sampling, S1–S4 sales surfaces. Broader `scripts/` unit-coverage backfill beyond the gate scripts also remains.

### Related lower-severity items on the backlog

- `docs/afl-finals-2026.md` prose can carry stale round labels ("Round 8" / "7 games") even when the ladder data is correct — a script-template bug, not a hand-fix.
- Player CSVs can lag current-season finals rounds even when the matches CSV has the Grand Final; scope player-level premiership metrics to completed seasons.
- Player-audit career-total WARNINGs are often false positives from a URL builder that discards DOB and collides same-name players — triage by DOB-stamped filename before re-scraping.
- `charts.py` Era `KeyError` (needs Scientist); `live_analysis_pipeline.py --help` auto-starts the poll loop.

---

## 5. Agent prompts

> **Note:** Verbatim agent prompts are maintained in `.claude/agents/` — the single source of truth.
> Embedding copies here creates a drift source; see the Known Problems log (§4, anti-pattern: hand-maintained derived state).
> To read the current prompts: `cat .claude/agents/<AgentName>.md`
