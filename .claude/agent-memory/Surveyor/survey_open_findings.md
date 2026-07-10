---
name: survey-open-findings
description: Findings routed by past surveys not yet confirmed fixed — the first thing the next survey re-checks
metadata:
  type: project
---

Re-check these first on the next survey; retire each line only after verifying the
fix by content, not by claim.

## OPEN — from 2026-07-10 DEEP survey (see .claude/surveys/2026-07-10-survey.md)
- **F1 CRITICAL (ESCALATED TO HUMAN) — stamp gate accepts same-hash FAIL; dustin-martin
  shipped despite HELD + FAIL record.** dustin-martin-the-storm.md council hash 16ddae68
  has PASS (050709Z) AND later FAIL (051141Z); committed 13ad59e69 anyway. Verify:
  check-council-stamp.sh fails on any same-hash non-PASS record; human disposition of
  dustin-martin recorded. Owner Gaffer + human.
- **F2 HIGH — DataSentinel untagged-number sweep non-deterministic** (Gaffer memory
  feedback_datasentinel_nondeterminism.md). Verify: deterministic script sweep exists in
  the gate path. Owner Gaffer.
- **F3 HIGH — S3 lineup backfill STILL unrun** (carried from 07-09 F2): re-measured
  2026-07-10 → 700 garbage rows (2025:412, 2026:288 = 100% of 2026). Verify: fingerprint
  scan → 0 AND refresh_and_rank.sh:101 lineups exclusion lifted. Owner Scientist.
- **F4 HIGH (SHARPENED 07-10 audit) — S1b wired TOG% as a RAW SAME-GAME feature: train/serve
  skew in production + leak in backtest harness.** Coverage concern RETIRED: training population
  (DOB>1986 filter, prediction.py:333) is 2005+ and TOG% is 100.0% non-null there (129,007 rows
  measured 07-10). Real defect: extra_features enters feature_columns unlagged
  (prediction.py:439-442) while every other feature is shift(1); training row's TOG% is the
  target game's own TOG%. Production predicts on the round-(N-1) row (run(): filter >= next_round-1,
  head(1)) so serves last-game TOG% — semantics mismatch. Backtest keeps cutoff-round row with raw
  stats intact (backtest.py:180-185, no masking) so an S1b backtest would consume the scored game's
  actual TOG% — optimistically biased; harness cannot fairly evaluate S1b as-is. No post-S1b
  prediction has shipped yet (latest CSV 07-07 16:06 < S1b commit 07-09 16:37); R20 cycle 07-14 is
  first exposure. Verify: raw TOG% out of feature_columns (lagged variant or opt-in flag) BEFORE
  R20 run; backtest invariant "no same-row raw feature OR cutoff-row stats masked" enforced;
  then OFF-vs-lagged-ON backtest logged in experiment-log. Owner Scientist.
- **F5 MED — un-gated remediation parked in working tree**: forgotten-heroes modified
  16:37:53 (+13 bare [data] tags), council hash 2bb69904 has NO sentinel record. Verify:
  tree clean, current content hash has PASS. Owner Scientist→Gaffer.
- **F6 MED — pending-decisions footer false**: says both docs HELD; forgotten-heroes
  shipped gated, dustin-martin shipped around FAIL. Verify footer updated. Owner Gaffer.
- **F7 MED — README.md:55 instructs `bash refresh_and_rank.sh` which now refuses**
  (single-entry guard). Also last weekly_refresh log 06-22 while R18/R19 predictions
  shipped — sanctioned entry point unused 18 days. Verify: README fixed; 07-14 cycle run
  via weekly_refresh.sh. Owner Gaffer.
- **F8 MED — S6 position-source assumption false** (carried 07-09 F3, unchanged;
  pending-tasks:216). Owner Scientist+Gaffer.
- **F9 MED — conceded stats: doc-layer consumer only** (revised 07-10 audit): consumed by
  BriefBuilder skeletons (.claude/agents/BriefBuilder.md:84) + DataSentinel verification list;
  NOT consumed by prediction model or update_team_analysis.py (grep-verified). S2 rotation fix
  = doc correctness value, zero prediction-accuracy impact. Still 2025-only. Verify recorded
  load-bearing-or-demote decision. Owner Scientist+Gaffer.
- **F10 MED — afl-insights.md:18 latest-brief pointer stale at R9; actual latest R14**
  (carried 07-09 F5; fold into F17). Owner Gaffer.
- **F11 MED — model training core untested**: no behavioural test of _engineer_features
  rolling/expanding values or train/predict round-trip. Verify synthetic-fixture test
  module exists. Owner Scientist.
- **F12 LOW — 3 stale worktrees ~560MB** (carried 07-09 F7, unchanged, June 16). Owner Gaffer.
- **WATCH — F04 first live test due 2026-07-14**: BOTH weekly_refresh_2026-07-14.log AND
  .claude/audit/last_refresh_complete.json must exist after the cycle; absent → escalate.
- **WATCH — prediction double-runs** (2 CSVs per round R17–R19); **sentinel-*.json
  retention** (71 files); **weekly-fan-pack.yml Wed-23:00-UTC ordering** vs Tuesday refresh.

## Still open, carried from earlier surveys
- F8 (07-07) FootyStrategy tripwire learning loop never fired. MEDIUM, Gaffer+FootyStrategy.
- F19 (07-07) Chronicler commit ownership rule unwritten. MEDIUM, Gaffer.
- 0703-F2 HOF stamp attribution — still unverified. LOW.

## Verified fixed 2026-07-10
Retired: **07-09 F1 CRITICAL (Sprint 3 uncommitted)** — all 8 artifacts tracked
(git ls-files verified), committed 13ad59e69, main == origin/main, experiment-log
"committed" claims now true. Suite 315 passed in 2.63s; HOF gate exit 0; prompt
drift zero (all 11 hashes == 07-09 baseline; BriefBuilder F06 hash committed as
the baseline note predicted). Sprint 3 code audited line-by-line: S1b/S2/S3/S7
all sound; no correctness defects found (coverage gap = F11 above).

## Prior retirements (2026-07-09 and earlier)
See .claude/surveys/2026-07-09-survey.md — 0703-F1/F3/F4, D1/D2/D3 retired there;
F1–F21 of 07-07 retired in that survey file.
