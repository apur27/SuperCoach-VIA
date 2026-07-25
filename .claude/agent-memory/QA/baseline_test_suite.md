---
name: baseline_test_suite
description: Baseline pytest count for tests/ so future QA runs can detect regressions vs growth
metadata:
  type: project
---

As of 2026-07-13 (Round 20 prediction / Round 19 backtest weekly-refresh QA
gate), `pytest tests/ -v` reports **352 passed, 0 failed, 0 skipped** in ~2s.
Prior baseline (2026-07-07) was 244; the +108 growth reflects several
untracked/new test modules landing between cycles (council-stamp audit,
precommit hook fail-closed, requires-stamp routing, staged-blob check,
tag/verdict vocabulary, prediction selection/features, top100 consistency,
update_hof_pages, etc. — see `git log --oneline` for the Surveyor-audit and
M11 commits between 2026-07-07 and 2026-07-13). Confirmed non-regression:
same green baseline, much more coverage. In particular
`test_qa_rank1_cross_check_runs` (tests/unit/test_qa_stat_leaders_schema.py)
PASSED — the Pendlebury CSV=437 vs HOF-JSON=436 mismatch reported mid-cycle
was confirmed to be a mid-abort staleness artifact that self-healed once
Phase 2b (`update_hof_pages.py`) completed; independently re-verified via
`scripts/check_hof_numbers.py` (exit 0, career_games rank-1 = 437) and a
direct CSV cross-check (data/player_data/pendlebury_scott_07011988_performance_details.csv
== 437 rows == JSON total).

As of 2026-07-07 (Round 19 weekly-refresh QA gate), `pytest tests/ -v` reported
**244 passed, 0 failed, 0 skipped** in ~1s. Prior baseline (2026-07-03) was 239;
the +5 growth is `tests/unit/test_prediction_selection.py` (3 tests) added for
the CR-1 mtime-vs-lexicographic-sort fix in `scripts/weekly_refresh.sh`, plus
2 more from other untracked test files landing between cycles. Confirmed
non-regression: same green baseline, more coverage.

Note: a prior request assumed "~250+ passing" as the expected baseline — that
number was not grounded in an actual prior QA run recorded in memory (this was
the first QA memory entry). 239 passed with zero failures is a clean, complete
green run; treat 239 (now 244) as the current baseline going forward, not a shortfall.

**Why:** without a recorded baseline, QA can't distinguish "test count dropped
because something broke" from "test count is just lower than someone's
unverified estimate." Recording the actual number here closes that gap.

As of 2026-07-20 (Round 21 prediction / Round 20 backtest weekly-refresh QA
gate, post-incident-recovery verification), `pytest tests/ -v` reports
**384 passed, 0 failed, 0 skipped** in ~2s. Prior baseline (2026-07-13) was
352; the +32 growth includes `tests/unit/test_player_scraper_dedup.py` (6
tests, added in commit `8badf8dc5` to fix a dtype-blind `drop_duplicates` in
`dedup_player_performance` that let fixture-date-drift re-emits double-count
833 player CSVs — the first weekly run that cycle aborted fail-closed at the
phantom-row gate before any push). Re-verified post-fix: counter-gap hits = 0
across all player files (`scripts/phantom_row_validator.py`), `check_hof_numbers.py`
exit 0, HOF rank-1 career_games = 438 (Pendlebury, cross-checked CSV==JSON).

As of 2026-07-25 (harness + correctness ship — no weekly cycle run; new
`scripts/backtest_completeness.py` + `scripts/skeptic_verdict.py` modules,
banner aria-label fix, top100-consistency-gate single-sourcing), `pytest
tests/ -v` reports **442 passed, 0 failed, 0 skipped** in ~4.6s. Prior baseline
(2026-07-20) was 384; the +58 growth is explained by the new test files in
this commit: `tests/unit/test_backtest_completeness.py` (20 tests, confirmed
by direct collection), `tests/unit/test_skeptic_gate.py` (**12 tests**, not
11 — the PR description undercounted by one; verified by direct collection,
not a red flag), `tests/unit/test_eval_surface_banner.py` (5 tests), plus
`test_update_team_analysis_top100_gate.py` and
`test_refresh_readme_top100_gate.py` (untracked, landed same cycle) and
`test_harness_wiring.py`. All new modules' tests pass; `check_hof_numbers.py`
exit 0 with rank-1 career_games = 438 (unchanged from 2026-07-20, no
player-game-count regression — `data/matches/` and `data/player_data/` are
untouched this cycle per `git status`).

**How to apply:** on future QA runs, compare the new pass count against 442
(current baseline as of 2026-07-25; was 239 → 244 → 352 → 384 → 442).
- Count drops with same file set → investigate (deleted/skipped tests).
- Count rises → expected as new modules ship (e.g. this cycle added
  test_commit_authorization.py, test_inject_trust_badge.py,
  test_skeptic_sample_tags.py, test_requires_stamp_routing.py,
  test_staged_blob_check.py, test_tag_vocabulary.py — several of these were
  untracked/unstaged at QA time, see [[project_council_stamp_gate]]).
- Update this memory's number after each QA run so the baseline stays current.

Also noted 2026-07-13: `assets/charts/hall/` only ever contains 4 files
matching the `alltime_top20_*.png` glob (goals, games, disposals, tackles)
plus a 5th non-matching chart (`alltime_stat_categories_leaders.png`) used
only on the stat-leaders hub page. `docs/hall-of-fame/generate_records_charts.py`
is hard-coded to emit exactly these 4 — it has never produced 6. The QA
checklist's "at least 6 chart files" mandatory-artifact check is therefore
chronically unmet by design, not by regression; git history shows this
directory byte-stable across the last 5+ auto-update commits. Treat this as
a known WARN (spec vs. pipeline mismatch), not a FAIL, unless the file count
drops below 4 or a referenced chart goes missing from a stat page.
