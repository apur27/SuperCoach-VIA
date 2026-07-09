---
name: survey-open-findings
description: Findings routed by past surveys not yet confirmed fixed — the first thing the next survey re-checks
metadata:
  type: project
---

Re-check these first on the next survey; retire each line only after verifying the
fix by content, not by claim.

## OPEN — from 2026-07-09 STANDARD survey (see .claude/surveys/2026-07-09-survey.md)
- **F1 CRITICAL — entire Sprint 3 uncommitted** (prediction.py, game_scraper.py,
  conceded CSV repair, BriefBuilder F06, 5 test files, 2 scripts, experiment-log,
  five badge-passed docs) while experiment-log:100 and pending-decisions:71 claim
  "committed/shipped". Verify: `git ls-files tests/unit/test_lineup_scraper.py` etc.
  non-empty AND the claiming docs now true. Owner Gaffer.
- **F2 HIGH — S3 lineup backfill unrun**: 700 garbage rows (2025:412, 2026:288 =
  100% of 2026). Verify by the semicolon fingerprint scan (experiment-log:126) → 0,
  AND the lineups exclusion lifted from refresh_and_rank.sh allowlist. Owner Scientist.
- **F3 HIGH — S6 position-source assumption false**: lineup schema is
  [year,date,round_num,team_name,players] — NO position field. Verify pending-tasks
  S6 card re-scoped to a verified source. Owner Scientist+Gaffer.
- **F4 MED — backtest queue serialization**: S1b-effect run → S7 ON-vs-OFF → S4 →
  S5, one change per comparison. Verify experiment-log entries per run. Owner Scientist.
- **F5 MED — afl-insights.md:18 "latest brief" pointer stale at R9** (R13 briefs
  exist). Fold into F17. Owner Gaffer.
- **F6 MED — conceded stats orphaned**: 2025-only, no consumer computes from it,
  4 columns unverifiable, yet listed as BriefBuilder/DataSentinel source. Verify a
  recorded load-bearing-or-demote decision. Owner Scientist+Gaffer.
- **F7 LOW — 3 stale worktrees ~560MB** (June 16; branches 524fa0ffe/04216cd2c/
  9293d831a unmerged but likely superseded on main). Verify .claude/worktrees/ empty.
  Owner Gaffer.
- **WATCH — F04 first live test**: after Tuesday 2026-07-14 cycle, BOTH
  .claude/audit/weekly_refresh_2026-07-14.log and
  .claude/audit/last_refresh_complete.json must exist. If sentinel absent → escalate
  (single-entry discipline failed first exercise).
- **WATCH — TOG% era coverage** (unaudited, per experiment-log S1b "NOT verified");
  **prediction double-runs** each round (2 CSVs per cycle — flaky phase?);
  **sentinel-*.json accumulation** (~70 files, no retention rule).

## Still open, carried from earlier surveys
- F8 (07-07) FootyStrategy tripwire learning loop never fired — re-verified dormant
  2026-07-09 (no fired-tripwire memory exists). MEDIUM, Gaffer+FootyStrategy.
- F19 (07-07) Chronicler commit ownership rule still unwritten (its outputs landed
  only via catch-all commit 0b07b3048). MEDIUM, Gaffer.
- 0703-F2 HOF stamp attribution — still unverified (no stamp lines grep-matched in
  hall-of-fame-stat-leaders.md 2026-07-09). LOW.

## Verified fixed 2026-07-09
Retired: 0703-F1 (skeptic_sample_tags spec-form — fixed, script line 28),
0703-F3 (AUDIT_ENFORCE defaults 1 — check-council-stamp.sh:35),
0703-F4/F04 (single-entry guard refresh_and_rank.sh:9-22 + sentinel code
weekly_refresh.sh:310-317 — code verified; live exercise pending, see WATCH),
D1 (QA mtime selector, commit 7d6a4e40f), D2 (ai-architecture venue residuals),
D3 (R18 ground truth committed + auto-stage allowlist e525b82c7; lineups exclusion
deliberate pending F2). Fill-zero-vs-Decision-3 checked: NO conflict (per-game
averages use fill-zero; threshold-scan inclusion uses dropna+coverage — context-
separated correctly). Suite 315 passed; HOF gate PASS; prompt drift fully explained.

## Prior retirements (2026-07-07, HEAD 798ff2557)
F1-F21 of the 07-07 DEEP survey verified/retired as recorded in that survey file;
F02 five-doc correction committed 5a5789bd1 (the 07-09 badge pass on those docs is
part of the new F1 above).
