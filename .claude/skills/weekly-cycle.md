---
name: weekly-cycle
description: Run the full weekly refresh pipeline with the Tuesday-settlement guard — scrape actuals, rank, predict, backtest, HOF recompute, weekly recap, QA gate, ship, and Chronicler. Invoke as "/weekly-cycle". Use for the routine weekly data refresh after a round settles, not for a brief.
---

# /weekly-cycle — weekly refresh pipeline

Invoked by the coordinator as `/weekly-cycle`. This is the no-brief cadence cycle:

```
refresh_and_rank.sh → HOF pipeline → weekly recap → QA → Gaffer(SHIP) → Chronicler
```

Gaffer owns process, not truth: never override a QA FAIL or a `check_hof_numbers.py` non-zero exit, never author a `[data]` number, never `git push --force`.

## Cadence rule

Run **on Tuesday after the round settles.** afltables does not have the weekend's actuals until then — the round that just played is available from the Tuesday after. Do not hardcode a specific round or date here; the evergreen Tuesday-settlement rule is the whole cadence.

## Phase 0 — Timing guard

Before doing anything, check the calendar:

- If today is **Tuesday or later** and the weekend round has completed, proceed.
- If run **before Sunday's round completes**, afltables won't have the data yet. **Warn the user** ("round not settled — afltables won't have this round's actuals; scraping now will pull stale/partial data") and **ask for explicit confirmation** before proceeding. Do not auto-proceed on an unsettled round.

## Phase 1 — refresh_and_rank.sh

- **Run:** `bash refresh_and_rank.sh` — scrape matches + players → rank → predict → backtest → refresh docs → commit its own auto-update. This script handles its own git add/commit/push for the data-refresh outputs.
- **Long-running (~30–60 min).** Run in the background and wait for completion before Phase 2.
- **Watch for:** a player game-count drop or other clearly-wrong data (the "wrong data" escalation trigger). If the scrape produces obviously broken numbers, halt and route to Scientist — do not build HOF pages on corrupt actuals.

## Phase 2a — Cheat sheet

- After Phase 1 completes, run cheat-sheet generation **if applicable** for this round. Skip if not part of this cycle.

## Phase 2b — HOF pipeline (hard-gated)

Run in order; each depends on the previous:

1. `python docs/hall-of-fame/compute_stat_leaders.py`
2. `python docs/hall-of-fame/generate_records_charts.py`
3. `python scripts/update_hof_pages.py`
4. `python scripts/check_hof_numbers.py` — **HARD GATE.**

- **On `check_hof_numbers.py` non-zero exit:** abort Phase 2b. Route the discrepancy to **Scientist** (HOF numbers not matching source data is a data/compute issue). Do not ship HOF pages that fail their own number check.

## Phase 3 — Weekly recap

- **Invoke:** FootyStrategy for the weekly recap in `docs/afl-insights.md`.
- **Note:** `afl-insights.md` gating is **deferred to Sprint 2** — it is currently ungated. Do **not** add it to the stamp gate yet.

## Phase 4 — QA gate

- **Invoke:** QA agent.
- **On PASS or PASS_WITH_WARNINGS:** proceed. Log warnings in the retro.
- **On FAIL:** abort. Route each specific failure to its owning agent, fix, re-run QA. A QA FAIL blocks ship with the same authority as a DataSentinel FAIL.

## Phase 5 — Gaffer SHIP

Stage the regenerated files by the **weekly refresh allowlist** (the exact list `scripts/weekly_refresh.sh` stages — keep in sync with that script):

```
README.md
docs/banner.svg
docs/afl-insights.md
docs/weekly/
docs/hall-of-fame-stat-leaders.md
docs/hall-of-fame-stat-disposals.md
docs/hall-of-fame-stat-games.md
docs/hall-of-fame-stat-goals.md
docs/hall-of-fame-stat-brownlow.md
docs/hall-of-fame-stat-tackles.md
docs/hall-of-fame-stat-marks.md
docs/hall-of-fame-stat-clearances.md
docs/hall-of-fame-stat-contested.md
docs/hall-of-fame-stat-hitouts.md
docs/hall-of-fame-stat-kicks-handballs.md
docs/hall-of-fame-stat-goalassists.md
docs/hall-of-fame-stat-single-season.md
docs/hall-of-fame/_stat_leaders.json
assets/charts/
```

- Commit via `scripts/git_commit_safe.sh commit -m "Weekly refresh round <N> — stat leaders + cheat sheet + insights (<date>)"`.
- Push to `origin/main`. Never `--force`.
- README news block: at most 2 entries (`enforce_news_limit` in `scripts/weekly_refresh.sh` handles this; a manual publish must follow the same rule).

## Phase 6 — Chronicler

- **Invoke:** Chronicler agent after the push.
- **Pass:** the commit hash and cycle type `"weekly-refresh"`.
- **It produces:** the run report and top-3 expansion recommendations.

## Retro

Log one line to Gaffer memory: what broke, the top Chronicler recommendation, what to add to the backlog.
