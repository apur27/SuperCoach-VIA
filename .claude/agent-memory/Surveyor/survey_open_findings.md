---
name: survey-open-findings
description: Findings routed by past surveys not yet confirmed fixed — the first thing the next survey re-checks
metadata:
  type: project
---

Re-check these first on the next survey; retire each line only after verifying the
fix by content, not by claim.

## NEW OPEN — 2026-07-25 testing-layer META consult (advisory, reported in chat)
- **T25-R1 HIGH (Gaffer) — TDD rule has no deterministic enforcement point**: .githooks/pre-commit
  exits 0 when no .md staged (:33-37) → pure-code commits get ZERO automated checks; QA coverage
  audit is WARN-only + diffs HEAD~1..HEAD only (QA.md:146-149); harness has no pytest phase
  (grep=0 in both .sh, re-verified 07-25). Suite = 384 green in ~2s, so hook cost trivial.
  Verify: hook runs unit suite fail-closed when .py staged + CLAUDE.md §5 codifies it.
- **T25-R2 MED (Gaffer routes; Scientist writes tests) — QA validators live only as prompt
  snippets** (QA.md:59-137 prediction schema / HOF cross-check / match completeness) — class 6.
  Verify: tests/integration/ tier exists, QA.md invokes `pytest -m integration` instead.
- **T25-R3 MED (Scientist) — no golden-file regression tests for incident-bearing transforms**
  (dedup=8badf8dc5 class, top100 ranking, feature engineering — subsumes open F11 07-10).
  Verify: tests/regression|fixtures exist and run in QA tier.

## NEW OPEN — 2026-07-25 pending-state check (verified by measurement, reported in chat)
- **P25-F1 CRITICAL (ESCALATED) — HOF profile stat-lines/prose stale ON MAIN; the gate CLAUDE.md
  cites never runs in the production path.** hall-of-fame-top100.md table :36 = Pendlebury
  438/209/11,108 (matches all_time_top_100.csv + measured CSV 438 games) BUT regen-owned profile
  stat-line :216 = "436 games · 208 goals · 11,069 disposals" and [data]-tagged prose :163/:218
  say 436/11,069. Cause: 07-20 doc write was refresh_readme.py Step 2b (:412, marker table only);
  update_team_analysis.py [13/14] (regenerate_top100_profiles + check_top100_consistency hard
  gate, :5004-5015) invoked NOWHERE in weekly_refresh.sh/refresh_and_rank.sh (grep 0; log has
  zero "[13/14]"/"[profile]" lines). CLAUDE.md tag-exemption claims the gate verifies this doc —
  false in practice. = F14 fix regressed to dead code; classes 7+1. Owner Scientist (single
  gated writer). Verify: [13/14] output in next harness log + stat-line==table.
- Untracked run-1 artifacts to quarantine (ties to P20-F1): backtest_by_position_20260710/13/20,
  next_round_20_prediction_20260713_2050 + 20260714_0730, next_round_21_prediction_20260720_1557.
  Note WF-F7 stays retired: afl-insights.md:30 cites next_round_21_..._20260720_2007.csv which IS
  tracked (git ls-files verified 07-25).
- MED (Gaffer) — allowlist orphans, class 6: data/top100/yearly/year_2026.csv tracked-but-dirty,
  outside both harness allowlists (Gaffer retro F3, confirmed dirty 07-25); 18 data/lineups CSVs
  same state. Verify: files in Phase-1 allowlist or documented as intentionally unshipped.
- MED-LOW (Gaffer) — docs/banner.svg root aria-label still "R1–13: MAE 4.020, 73.0%" (grep 07-25);
  visible text current at R1-20. 8 rounds stale for screen readers; chronic (QA memory
  project_banner_aria_label_stale.md).
- MED-LOW (Chronicler) — no run report for the 07-20 R21 cycle; docs/run-reports latest is
  2026-07-13-weekly-r20.md.
- Verified fixed 07-25: requirements.txt repaired (sklearn/lightgbm/optuna/scipy/pyyaml present,
  bogus `datetime` gone) — S11-F5 retired pending one CI-green confirmation; Harvey/Pendlebury
  record-claim STRUCTURE corrected (:163 now past-tense, correct) though numbers stale per P25-F1;
  harness FootyStrategy invocation now `--agent FootyStrategy` with no --model override (M2 core
  retired); last_refresh_complete.json exists (M5 retired).

## NEW OPEN — from 2026-07-20 PULSE post-run review (.claude/surveys/2026-07-20-pulse-survey.md)
- **P20-F1 HIGH — published R20 backtest tainted-provenance**: backtest_*_20260720_155725 computed
  during run 1 on the corpus that FAILED the phantom gate at 16:04 (mass doubled rows, pre-8badf8dc5);
  run 2 skipped recompute ("Last complete backtest: round 20") and committed it (268cd00b8);
  published afl-backtest-2026.md:56 MAE 3.92 n=361. Verify: R20 re-scored from archived forward CSV
  on deduped corpus + FATALed-run artifacts quarantined from incremental check. Owner Scientist + Gaffer.
- S1b MAE needle still unmeasured: R20 reading exists but tainted (above); series R16 3.981 / R17 3.825 /
  R18 3.767 / R19 3.973 / R20 3.920 — no attributable signal. Next clean reading = R21 backtest.

## Retired/verified 2026-07-20 PULSE (all by measurement, see survey file)
- WF-F1 RETIRED: hub verdict sentinel written (stat-leaders 4e461fa0, check_hof_hub at
  check_hof_numbers.py:182); 13/13 stamp checks vs audit records; HOF gate "all verified OK".
- WF-F2 RETIRED: DataSentinel PASS pass-1 (12/12 tags, hash matches current file); FootyStrategy
  followed datasentinel_gate_traps playbook; retry wired (1d634db75).
- WF-F3 RETIRED: HARNESS_PHASE on every log line; last_refresh_complete.json round 21; run-1 FATAL
  fully captured in log this time.
- WF-F4 spot-clean: 5 files 0 synthetic-date suspects; clarke_angus 2025 row now 2025-08-27
  (round label still "1" — cosmetic residue only).
- WF-F5 RETIRED: settlement probe fired phase 0 both runs. WF-F6 RETIRED: match-completeness gate
  blocks before push (20:14:08 PASS). WF-F7 RETIRED: 10 CSVs + optuna params in 268cd00b8.
- WF-F8 RETIRED (data side): matches_2026 = 162 rows, R10=9 R17=7, blocking match-gate PASS.
- C2-class vector CLOSED live: dtype-blind dedup (8badf8dc5) caused run-1 mass duplication; phantom
  gate blocked push (S11-F7 fix re-proven a second time); run-2 clean, 0 dup rounds in spot checks.

## OPEN — from 2026-07-14 STANDARD weekly-flow survey (.claude/surveys/2026-07-14-weekly-flow-survey.md)
- **WF-F1 CRITICAL (ESCALATED) — HOF hub staged-but-unverdictable; killed 07-14 Phase 4; 8 of 13
  rank-1 [data] figures stale ON MAIN** (Pendlebury 436 vs 437 etc., measured vs _stat_leaders.json;
  hub has HOF-HUB sentinels check_hof_numbers.py ignores). Verify: checker reads HOF-HUB rows +
  verdict recorded in Phase 2b + main corrected. Owner Scientist+Gaffer.
- **WF-F2 HIGH — DataSentinel gate single-shot; methodology-paragraph requirement absent from
  FootyStrategy.md and harness -p prompt** (guaranteed first-pass FAIL). Verify: authoring checklist
  in FootyStrategy.md + bounded retry in weekly_refresh.sh Phase 3b. Owner Gaffer.
- **WF-F3 HIGH — Phase 4 commit output not tee'd (weekly_refresh.sh:289), no resume-from-phase**;
  07-14 death left zero log evidence. Owner Gaffer.
- **WF-F4 HIGH — synthetic-date mint still open** (= old C1 scraper half): player_scraper.py:262
  approx date for numeric-round rows, fixture resolution finals-only (:432-444); fix_synthetic_dates.py
  not wired into harness; no pytest/QA phase in harness (grep=0). Owner Scientist.
- **WF-F8 CRITICAL (ESCALATED, recurrence of S11-F1 class) — matches_2026.csv missing 9 games
  (R10: 6, R17: 3)**; audit_match_rounds WARNs every run, nothing acts; published ladder claims
  computed+DataSentinel-verified against the incomplete CSV. Owner Scientist + human disposition.
- WF-F5 MED — timing guard = LLM weekday heuristic in weekly-cycle.md; 3 surfaces conflict on
  cadence. Verify: deterministic settlement probe in harness Phase 0. Owner Gaffer.
- WF-F6 MED — phantom gate placement after full Phase-1 compute (~2h loss per scraper defect).
  Verify: validator runs post-scrape inside refresh_and_rank. Owner Gaffer.
- WF-F7 MED — 10 prediction/backtest CSVs untracked incl. the CSV afl-insights.md cites
  (next_round_20_prediction_20260713_2050.csv); by-archive backtest depends on this archive.
  Verify: staged by explicit pattern + files committed. Owner Gaffer.
- WF-F9 LOW — false QA-phase claims (refresh_and_rank.sh:12, weekly-cycle.md description).

## Retired/updated 2026-07-14
- C1 on-disk half RETIRED: Clarke row date corrected (a742b37bf); 176-file corpus sweep committed
  (4c798e665). C1 scraper half continues as WF-F4.
- 07-13 AMBER "cycle incomplete" RESOLVED: Phase 2-5 shipped 2c50e6736 (07-13); insights PASS
  genuine on 07-14 re-run (12/12 tags, fresh timestamp). BUT 07-14 re-run itself died at Phase 4
  (WF-F1) and its sentinel was never written.
- kicks-handballs/single-season stamp passes verified LEGITIMATE (badge-only changes, hash-stripped;
  07-07 records still match recomputed hashes) — no gate bypass.
- Two R20 forward CSVs byte-equivalent on predictions (maxdiff 0) — no cross-doc inconsistency.

## OPEN — from 2026-07-13 scraper-fix consult (counter-aware delta, findings in-chat)
- **C1 ELEVATED TO CRITICAL/ESCALATED (PULSE 07-13): the wrong-date Clarke row is now COMMITTED
  to main in 9e617a167** — clarke_angus game 14 = year=2025 round=1 date=2025-03-01 after round 25
  (measured post-commit). Original finding below stands; disposition now includes published-wrong.
- **C1 HIGH — restored Clarke game #14 on disk carries wrong date 2025-03-01** (real date
  2025-08-27 per Scientist's own test comment, test_player_scraper.py). Generalizes: every
  future counter-rescued row gets the known-bad approximate date; numeric-round rows never
  get fixture-date resolution (only finals do, player_scraper.py:434-440). Model temporal
  ordering poisoned for such rows. Verify: fixture-date lookup extended to numeric-round
  anomalies + Clarke row date corrected. Owner Scientist.
- **C2 MED — counter-aware delta opens a renumbering-duplicate vector**: afltables retro-insert
  BEFORE on-file rows shifts later site counters +1 → last on-file game re-scrapes as
  counter=max+1 → is_new_counter → re-appended; dedup key includes gp → both copies kept;
  phantom gate sees contiguous counters → silent. Old date filter blocked this. Verify:
  same-game-different-counter check in gate or scraper. Owner Scientist.
- **C3 LOW/backlog — Gate 1 year-skip residual accepted**: earlier-season backfill stranded
  until next new game fires the gate (active players); retired players (RETIREMENT_THRESHOLD
  skip) silent forever. Backlog: periodic corpus counter sweep (~30s for 13,353 files; ran
  clean 07-13: 0 gaps, 0 dups). Owner Scientist.
## Verified live 2026-07-13 PULSE (R19/R20 cycle) — see .claude/surveys/2026-07-13-pulse-survey.md
- RETIRED S11-F7: phantom gate aborted 18:41 BEFORE any push; push deferred to parent; live-verified.
- RETIRED S11-F4: RUN_START freshness guard (weekly_refresh.sh:97-101) + round-20 detection correct.
- --from-csv backtest verified live (R19 scored from archived 07-07 CSV, no retrain, 0 log errors).
- NEW AMBER: cycle INCOMPLETE — DataSentinel FAILed afl-insights.md 21:00, harness FATAL at Phase 3b;
  README/insights/cheat-sheet gated-but-uncommitted (insights has PASS at current hash 9357e497,
  23:20). Verify next survey: Phase 2-5 shipped via sanctioned path, last_refresh_complete.json exists.
- NEW AMBER (Scientist): matches_2026.csv still missing 9 games (R10:6, R17:3, S11-F1 class);
  brown_callum_15082000 csv=76 games vs afltables=49 / goals 80 vs 19 — suspect wrong-player merge;
  unwin_rhys csv=1 vs afltables=3.
- S1b MAE needle unmeasurable at R19 (prediction predates fix); first reading = R20 backtest next cycle.
  R19 MAE=3.973 vs R18 3.767 / R17 3.825 / R16 3.981.
- Missing-game class (Clarke 13→15) fix verified in working tree 07-13, pending commit;
  suite re-run by measurement: 351 passed, 1 failed (Pendlebury 436 JSON vs 437 CSV —
  partial-scrape staleness, expect self-heal on harness re-run; escalate if still red after).

## OPEN — from 2026-07-11 status-aware-refresh consultation (findings returned in-chat)
- **F14 CRITICAL (ESCALATED) — hall-of-fame-top100.md publishes a false games-record claim.**
  Lines 163 + 216-218 say Pendlebury is "joint record holder at 432 / will pass the record
  next time he runs out"; same doc's auto-table line 36 + CSV show 436 games (measured
  07-11). Record broken ~4 games ago; [data]-tagged claim now false. Frozen profile prose
  sits OUTSIDE the ALL-TIME-TOP100 markers (7-121) that update_team_analysis.py regenerates
  weekly; 5 of 6 active players' profile ranks wrong (#34/#41/#45/#53/#94 vs table
  29/40/61/52/92). No guard: check_hof_numbers.py covers only the 10 stat sub-pages; doc
  carries no council stamp. Verify: profile stat-lines/ranks generated in the marker pass +
  deterministic table-vs-profile consistency check + Pendlebury/Harvey prose corrected.
  Owner Scientist (generation) + FootyStrategy (prose) + human (published-wrong disposition).
- **F15 MED — active-player career totals in frozen narrative docs have no as-of protection.**
  dustin-martin peer table (news 2026-06-21, lines 173-199) matches CSVs exactly today
  (measured) but 4 peers are 2026-active; drift begins next refresh they play (07-14).
  Verify: as-of-date editorial rule in FootyStrategy/BriefBuilder prompts + DataSentinel
  check keyed off a derived player-status artifact. Owner Gaffer (prompts) + Scientist
  (status artifact: active := max(year)==current season; no status field exists today —
  personal_details has only name/dob/debut/height/weight; player_scraper.py:114
  RETIREMENT_THRESHOLD=5 is scrape-skip only, unfit for publication status).

## OPEN — from 2026-07-11 DEEP survey (see .claude/surveys/2026-07-11-survey.md)
- **S11-F1 CRITICAL (ESCALATED) — matches_2025.csv missing PA v Collingwood R2 2025** (PA/Coll
  22 H&A games vs 23; Daicos file has the game; 2025 has 0 simultaneous-kickoff pairs vs 6-11
  in 2022-24; dedup key game_scraper.py:686/699 lacks venue+teams; forward-only delta never
  backfills). Verify: row restored + venue/teams in dedup key + per-team completeness gate.
  Owner Scientist.
- **S11-F2 CRITICAL (ESCALATED) — fan-pack releases ship stale round-9 CSV as latest**
  (proven: weekly-2026-07-08 latest-prediction.csv md5 == next_round_9_prediction_20260430_1822;
  weekly-fan-pack.yml:82 ls -1t on fresh checkout). Verify: filename-parsed selection + release
  replaced. Owner Gaffer.
- **S11-F3 CRITICAL (ESCALATED) — weekly_refresh.sh:159-164 records check_hof_numbers PASS
  for 3 docs _SUBPAGES never inspects** (kicks-handballs, stat-leaders hub, single-season).
  Verify: verdict loop keyed to _SUBPAGES + coverage extended. Owner Gaffer.
- **S11-F4 HIGH — R20 stale-round silent rerun**: dead "unknown" guard (weekly_refresh.sh:83-95,
  61 CSVs always on disk), no freshness assertion; harness:113 omits --csv to cheat sheet.
  Verify before 07-14 run. Owner Gaffer.
- **S11-F5 HIGH — tests.yml red 40/40** (ModuleNotFoundError sklearn/yaml); requirements.txt
  unusable (bogus `datetime`, missing lightgbm/optuna/sklearn/scipy/pyyaml). Verify: CI green
  from repaired requirements.txt. Owner Gaffer.
- **S11-F6 HIGH — Optuna cache feature-set-blind + shared prod/backtest path**
  (prediction.py:86-116, keys 'hgb'/'lgbm' :677/:752, optuna_version unchecked, :278 inherited
  by LeakProofPredictor). Verify: feature fingerprint in key + isolated backtest cache BEFORE
  serialized S1b/S7 backtests. Owner Scientist.
- **S11-F7 HIGH — phantom-row gate at weekly_refresh.sh:66-71 runs AFTER refresh_and_rank
  ([1/5]) has computed and pushed**; abort cannot un-push. Verify: gate before Phase 1 read or
  single end-of-cycle commit. Owner Gaffer.
- Below-cut confirmed queue (see survey file): afl-backtest-2026.md:107-113 R1-R13 headline
  block vs 18-round header; player_status unclamped max(last_year) poisoning; scraper
  no-timeout/no-retry/no-atomic-write; backtest.py:386-391 residual mtime glob +
  refresh_and_rank.sh:101 fallback without --from-csv; prediction.py name-only output dedup /
  DOB-filter drops / date-less rolling sort; no run-lock; pylint 3.8/3.9; audit O(N) scan;
  Chronicler/BriefBuilder prompt drift; README bare [data] tags.

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
- **F13 CRITICAL (DIAGNOSED 07-10 harness survey) — backtest.py writes its re-run
  predictor CSV into the live data/prediction/next_round_* namespace** (backtest.py:364-371
  globs newest-by-mtime after predictor.run()). Backtest runs AFTER the forward prediction
  in refresh_and_rank step 4, so every mtime-based consumer (weekly_refresh.sh:83 round
  detect, generate_weekly_cheat_sheet.py:44-48, refresh_readme step 2d) reads the BACKTEST
  artifact, not the forward prediction. Proven: R17 06-22 pair differs (Keane Mark 14 vs 15);
  cheat sheet shipped the 21:25 backtest file. This IS the double-run. Also proven:
  `[cutoff y=2026 r=18] dropped 0 future rows` in backtest_run_20260707_124402.log — weekly
  single-round backtest cutoff is a no-op; R18/R19 pairs byte-identical to forward run →
  scored round's data present at train time (ties to F4). Verify: backtest predictor writes
  to backtest-scoped path; cheat sheet consumes explicit --csv handoff; Scientist confirms
  whether scored rows were in training set (published MAE may be optimistic). Owner Scientist
  (namespace + leak) + Gaffer (explicit CSV handoff in harness).
- **WATCH — sentinel-*.json retention** (71 files); **weekly-fan-pack.yml Wed-23:00-UTC
  ordering** vs Tuesday refresh.

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
