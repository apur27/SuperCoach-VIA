---
name: weekly-r20-r21-retro
description: 2026-07-20 R20→R21 refresh — phantom-row gate caught genuine 833-file doubling, fail-closed recovery via dedup fix + reset --mixed, clean re-ship
metadata:
  type: project
---

2026-07-20 weekly refresh (R20 detected → predicting R21). First run after F1–F8 harness hardening. Fail-closed incident, then clean re-ship.

**What broke.** First run aborted at the phase-1c phantom-row gate (F8/F12): 833 player CSVs had a clean +1 row doubling (`missing=[] duplicated=[N]`, 0 missing). The bad data reached a LOCAL Phase-1 commit (`135b9194b`) but the gate blocked the push — nothing reached origin. Root cause (Scientist): `dedup_player_performance` in `scrapers/player_scraper.py` ran `drop_duplicates` on raw key columns, so an existing `int64` year/round (from `read_csv`) never matched a freshly-scraped string year/round after `concat`. Trigger: a game's fixture date drifted forward past the delta scraper's `since_date`, so an already-recorded row re-emitted and survived the dtype-blind dedup. Fixed by normalising key cols to stripped strings before dedup — commit `8badf8dc5`, TDD (+2 tests in `test_player_scraper_dedup.py`).

**Load-bearing learning — a doubled file does NOT self-heal on re-run.** Once doubled, the file's max date == the drifted date, so next run `game_date <= since_date` is true → no re-emit → `_write_player_details` never runs → the double persists and the gate fails again. Remediation MUST explicitly repair the already-doubled files (read → dedup → write) before re-running; a plain re-run is not enough. See [[project_phantom_row_dedup_gate.md]].

**Recovery pattern validated (reusable).** (1) `git reset --mixed HEAD^` to drop the bad Phase-1 commit while keeping the working tree — NEVER `--hard` (would strand `backtest_summary_*.csv` and risk a full R1 recompute). (2) Commit the code fix standalone via `scripts/git_commit_safe.sh`. (3) Re-run the full harness. The standalone code-fix commit rides along automatically in the Phase-1c push (`git rev-list origin/main..HEAD` pushes all pending), giving clean linear history: `[code fix] → [Phase-1 auto-update] → [Phase-4 weekly]`.

**Surveyor confirmed no sweep hazard** (pre-ship read): both `refresh_and_rank.sh` (Phase 1) and `weekly_refresh.sh` (Phase 4) stage by EXPLICIT allowlist. Pre-existing dirty state (18 lineups, `top100/yearly`, `.claude/agent-memory|surveys|audit`, `backtest_by_position_*`) is all OUTSIDE the allowlists — do NOT stash before a recovery re-run, the allowlist protects the commit.

**Commit-name correction to prior mental model.** "Auto-update: refresh AFL insights, predictions and backtest" is the **Phase-1 deferred-push** commit written by `refresh_and_rank.sh`, NOT a Phase-4 commit. Phase-4 is "Weekly refresh round N — stat leaders + cheat sheet + insights". So an amend+push of the auto-update commit would ship a Phase-1-ONLY tree and skip the whole rest of the cycle — always re-run, never amend-and-ship.

**Clean re-ship gates:** phantom-row counter-gap 0, match-completeness PASS, HOF numeric gate PASS + HOF verdict records (F1), insights DataSentinel PASS pass-1 (F2, no retry). Model: overall backtest MAE 3.958 (player-weighted, R1–R20, 7,153 scored); R20 round MAE 3.920 (n=361). Shipped `8badf8dc5` → `268cd00b8` → `64b366702`.

**QA verdict: PASS WITH WARNINGS** (proceeds — warnings logged here). 384 tests pass, phantom-row counter-gap 0, MAE 3.958 / n=7,153 independently re-derived to exact match. Zero failures.

**Backlog items created:**
- **F3 (Scientist).** `data/top100/yearly/year_2026.csv` is regenerated every cycle but sits outside both harness `git add` allowlists → drifts uncommitted indefinitely; a fresh clone gets a stale yearly table. Same class as the D3 "R18 actuals stranded" fix. Confirm whether `data/top100/yearly/` belongs in the Phase-1 allowlist.
- **Banner aria-label (Gaffer, harness gap).** `docs/banner.svg` visible text updates every cycle, but the root `<svg aria-label>` accessibility attribute has been stale for weeks (still "R1–R13: MAE 4.020, 73.0%"). The banner-update step (eval-surface refresh) patches visible text nodes only, not the aria-label. Needs a TDD fix so screen-reader users get current figures. Non-blocking, chronic.
- HOF chart count (4 vs checklist "≥6"): by-design/chronic, no action unless it drops below 4.
