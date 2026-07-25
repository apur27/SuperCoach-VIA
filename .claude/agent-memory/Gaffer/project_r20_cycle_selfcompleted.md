---
name: r20-cycle-selfcompleted
description: 2026-07-13 R20 weekly refresh completed end-to-end on its own; open gap is untracked Phase-1 prediction/backtest CSVs
metadata:
  type: project
---

The 2026-07-13 Round 20 weekly refresh completed the FULL council chain autonomously across two commits, both pushed to origin/main:
- `9e617a167` — Phase 1 auto-update (insights/predictions/backtest docs + charts)
- `2c50e6736` — Phase 2/3/4: README eval surface (R18→R19), afl-insights.md (DataSentinel PASS @ hash `9357e497`, weekly-recap Skeptic-exempt), cheat sheet round-20 + round-current, HOF stat sub-pages, banner.svg
- `6149b1ed4` — Chronicler run report (`docs/run-reports/2026-07-13-weekly-r20.md`)

**Why:** A dispatched Gaffer task assumed Phase 2–5 hadn't landed, but a concurrent/prior weekly_refresh.sh run finished them. Lesson: verify HEAD/origin and content hashes before re-shipping — a duplicate commit would have been wrong.

**How to apply — the README news block is NOT a weekly-refresh output.** Phase 1b updates the README *eval surface* (stats banner + backtest table); the news block (`NEWS-LATEST-START/END`) only carries user-published news articles and `enforce_news_limit` merely trims it to 2. A weekly recap does not add a news-block entry. So "news block unchanged" is correct, not a miss.

**Open gap (route to Scientist/harness):** Phase 1 did NOT stage its ground-truth CSVs — `data/prediction/next_round_20_prediction_20260713_2050.csv` and `data/prediction/backtest/*_20260713_205008.csv` are untracked (not gitignored; historically these ARE committed). afl-insights.md's methodology paragraph cites the R20 prediction CSV as a source, so the published doc references an uncommitted file — a provenance loose end. See [[project_sprint2_execution]] ("weekly ship must commit scraped ground truth").
