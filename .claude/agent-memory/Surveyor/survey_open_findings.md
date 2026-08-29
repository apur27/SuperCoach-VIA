---
name: survey-open-findings
description: Findings routed by past surveys not yet confirmed fixed — the first thing the next survey re-checks
metadata:
  type: project
---

Re-check these first on the next survey; retire each line only after verifying the
fix by content, not by claim.

## VERIFIED 2026-07-26 — post-ship audit of Gaffer's four commits (a46ca60e8, fa76f13a3, 24c047286, ce6ff6163)
All four on main, HEAD==origin/main==ce6ff6163. Everything below verified by measurement.

RETIRED this pass:
- R26-F1: README numbers now reproduce exactly (N=7153, MAE 3.958, w5 74.4, w10 95.78→95.8,
  bias −0.110, StK −0.584, WC +0.571, mean-abs 0.269; team-n==summary-n 7153 asserted in
  script + integration tier). Regenerator = scripts/update_eval_surface.sh, WIRED at
  weekly_refresh.sh:168. NOTE: my badge/w10/bias sub-claims were WRONG (see
  [[measurement-vintage-crossing]]); Gaffer's overrides upheld 3-of-3.
- R26-F2: council registry table derived+gated (tests/integration/test_published_artifacts.py
  :198-251); "ten-agent" = 9 registered + Codex external, stated consistently; banner
  aria-label fully current (ten-agent, R1–R20, 3.958, 74.4%, 13,357).
- R26-F3 (my count was 7+7; actual 7 R13 + 8 R14), R26-F4, F7-0710 quick-start
  (--allow-direct shown): all fixed in 24c047286.
- T25-R1/R2/R3: pre-commit Python gate live (.githooks/pre-commit:41-118, fail-closed,
  timeout-bounded, git-env-stripped, nested flock); failure paths test-exercised
  (test_precommit_python_gate.py: blocked on failing suite/untested module/unrunnable
  runner/hang/direct commit). Fast tier RUN: 491 passed 16.0s. Integration tier RUN:
  13 passed 0.7s, wired weekly Phase 0b (:85, FATAL on fail). Golden tier: 5 "matched
  nothing" ValueErrors added to update_team_analysis.py replacers.
- P25-F1: HOF profile prose corrected (Pendlebury 438/11,108 at :163/:216/:218 == table
  :36); refresh_readme.py:130 delegates to uta.update_top100_hof_doc() (hard gate inside),
  invoked from refresh_and_rank.sh:120 — gate now reachable in production path. F14 (ancestor
  finding) retired with it.
- P20-F1: R20 re-scored 20260725_173602, prediction_vs_actual frame-equal to 20260720_155725
  (413 rows) — provenance established by re-derivation. Completion manifest
  data/prediction/backtest/completed_runs.json live.
- by_position quarantine instruction (P25): correctly OVERRIDDEN — files are outputs of real
  runs (matching run logs + summary/by_team timestamps), now tracked; allowlist gap fixed.
- lineups allowlist (P25 MED): committed in a46ca60e8; re-allowlist safe — fingerprint scan
  re-run 07-26: 700 garbage/33,999 all legacy (2025:412, 2026:288 early-season), 36 recent
  rows clean. Only corrupted forward CSV next_round_21_..._20260720_1557.csv stays untracked
  on disk (accurate per Gaffer's minor note).
- WF-F2 residue: insights Skeptic exemption withdrawn AND codified (Gaffer.md:60,
  Skeptic.md:238); Phase 3c + scripts/skeptic_verdict.py deterministic three-way verdict
  (weekly_refresh.sh:359-380).
- banner aria-label (MED-LOW 07-25) fully retired.

## VERIFIED 2026-07-26 (second pass) — final review of 71fa430b3 + 4cf8a7da2; six-commit arc closed
HEAD==origin/main==4cf8a7da2. All prior pending-human items resolved by recorded decision:
- SV26-N1 RETIRED: CLAUDE.md two-tier budget (~20s) committed in 71fa430b3.
- 1360 RETIRED: table now renders f"{goals:,}" (uta:4626); golden fixtures updated in same
  commit; invariant test is generic (>=1000 must match stat-line), 39/39 pass.
- R18 gap RETIRED as accepted limitation: documented OUTSIDE 2026-BACKTEST markers (:100 end,
  note at :216+); I re-measured every figure: 284 non-null actuals/320 rows 14 teams
  (20260710_214217), 412/18 teams (20260707_154033), 9 R18 matches, delta 128. All exact.
- afl-insights double-BLOCK RETIRED: shipped hash 5dba9fa5... verified sha256-exact at HEAD;
  gate trail hash-keyed and clean (DataSentinel PASS 06:26Z + Skeptic PASS_WITH_CONCERNS
  06:32Z on 5dba9fa5; the BLOCK sits on prior hash 392e8d46 — no PASS/FAIL coexistence).
- Vintage bugs (71fa430b3): both regression tests proven NON-VACUOUS by running scenarios
  against pre-fix code in scratchpad (old top30 picks stale 10.0; old bootstrap adopts
  orphan). No-live-contamination claim CONFIRMED by measurement: on the PRE-fix tracked set,
  old HHMMSS key picked the authoritative vintage in all 10 multi-vintage 2026 rounds
  (incl. R1 where the NEW key alone would have picked orphan 20260525_182141 — the orphan
  deletion in the same commit is what makes "0 of 10 move" true). Session-wide diff of
  docs/afl-backtest-2026.md: only the R18 note added, zero figure changes.

NEW LOW/watch (route via Gaffer when convenient, none blocking):
- SV26-N2 LOW (Gaffer): Skeptic pass-4 concerns SK-R21-04/05 exist only in the 4cf8a7da2
  commit message; the verdict JSON has no reason field (record-sentinel-verdict.sh has no
  --reason flag despite the invocation prompt requesting one), and the IDs COLLIDE with
  pass-2/3 numbering (SK-R21-04 = ladder finding in Gaffer/FootyStrategy memories). QA's
  "5 warnings" list likewise unpersisted (counts 498+13=511/0-fail ARE backed by QA
  baseline memory). Non-PASS verdicts should persist their payload.
- SV26-N3 RETIRED 2026-07-28 (BL-02, c43229a96, verified independently): loader now filters
  to manifest-complete runs; guard test proven red-first against parent in worktree
  (99.0!=20.0); pool reproduced 21rds/7,524 with mark vs 20rds/7,153 without (delta 371 ==
  R21 n_scored per run log); retroactive R21 mark 20260728_004513 LEGITIMATE (log ends DONE,
  artifacts on origin in 28f5847a0 after DS PASS 18/18; unmarked would have quarantined
  git-tracked published files); manifest bytes == mark_complete() serialization; doc 0-delta;
  status: 15 complete / 0 orphans. Unit tier 532 exact.
- SV28-N4 MED (Gaffer): §6.2 scope boundary uncodified — policy text says "any script the
  harness invokes" must smoke-run; update_team_analysis.py IS transitively harness-invoked
  (refresh_and_rank.sh:120 → refresh_readme.py → import uta), yet BL-02 shipped it as
  "app-code-only, §6 doesn't apply" with no smoke run. One-day-old policy already being
  interpreted case-by-case by its own author. Class 10. Outcome: one canonical sentence
  defining scope (direct .sh invocations only vs transitive python imports). §6.1 genuinely
  didn't apply (exit_code set in last_refresh_status.json = no active cycle).
- Chart PNG non-reproducibility + generate_backtest_section() side-effect write: logged in
  Gaffer project_open_backlog.md item 3, owner Scientist. Tracked there, not duplicated.

## NEW 2026-07-27 — weekly refresh aborted at final commit (incident diagnosis)
- **I27-F1 HIGH (Gaffer): gated-doc regeneration has no re-verify hop.** 54a71c9ce brought
  docs/afl-backtest-2026.md under the hash-keyed council-stamp gate; the very next refresh
  regenerated it (Step 2d, new R21 numbers + new timestamped CSV names) → new content hash
  8da80746 has no sentinel record (last PASS was da1fc808 @09:54Z) → check-council-stamp
  blocked refresh_and_rank [6/6] commit at 20:55 → set -euo pipefail killed weekly_refresh
  at :99 → phases 2–5 never ran. Gate behaved correctly. Deterministic recurrence: any
  re-run fails the same way unless DataSentinel re-verifies the doc AFTER regeneration,
  before [6/6]. Pattern exists already for HOF hub (weekly_refresh :251-255) and insights
  (:321-354) — omitted for the newly gated doc. Class 7/NEW.
- **I27-F2 HIGH (Gaffer): no failure marker + orchestrator went idle on a background run.**
  Harness writes last_refresh_complete.json only on success (still shows R20-cycle
  2026-07-20); an abort leaves ONLY the log tail. Orchestrating Gaffer launched the run in
  background, reported progress at ~300/923, then ended its turn; the ERROR sat unread at
  the log tail for 2+ hours. Fix outcome: terminal status marker written on BOTH success
  and failure (phase + exit code), and Gaffer protocol = poll until terminal marker before
  ending turn. Related: pending-tasks.md:111 standing rule (background + artifact gate).
- State left on disk 20:55 2026-07-27: CLEAN PAUSE at commit boundary. 461 staged paths
  (R21 scrape complete 923 players, next_round_22 CSV, R21 backtest quartet 20260727_204827,
  all regenerated docs/charts), zero commits, no index.lock, no orphaned processes,
  7 orphan backtest logs correctly quarantined. Nothing half-written.

## VERIFIED 2026-07-28 — BL-03 (4bd5db06b) reviewed clean, second backlog clearance
- F3-lineups RETIRED and S11-F1 dedup-key half RETIRED: independently re-measured — before
  34,017 rows/700 garbage (2025:412, 2026:288), after 34,015/0; 698 of 700 keys repaired
  in place with real names, the 2 absent = Gabba R4 phantom (Cyclone Alfred opener), its
  R1 copies present. Dedup key now date+sorted-team-pair (build_match_key + write-site);
  my own sweep of all 130 matches_*.csv: 0 dupes under new key, 0 false collisions, 2,742
  same-timestamp slots preserved. Barossa Park canonical confirmed (2026 file 2-0-park;
  "Barossa Oval" survives only in stale worktrees/tests/memories; draft-file hit is the
  junior club "Barossa / Central District", not a venue). Gabba R4→R1 corroborated by
  player_data CSVs (untouched by commit, already round=1) and by all-18-teams-at-23-H&A
  invariant (before: Coll/PA at 22 — the added R2 M.C.G. row was genuinely missing).
  Tests re-run: 536 unit pass; integration 19/20, sole fail = BL-13 stat-leaders staleness
  (11,108 vs 11,137 exact). BL-13 self-heal wiring VERIFIED real: Phase 2b
  update_hof_pages.py rewrites the hub deterministically + check_hof_numbers.py aborts on
  mismatch — heals or fails loudly. S11-F1 missing-game half also closed (PA v Coll R2
  2025 now present). BL-04 green-lit.

## VERIFIED 2026-07-28 — BL-04 (5a4394ee3) reviewed, holds with two corrections
- Chart-PNG backlog item (Gaffer project_open_backlog item 3 / SV28 line above) RETIRED:
  all 7 PNGs refreshed; regeneration proven deterministic (2 renders byte-identical) and
  committed==regenerated re-verified independently for top10_alltime_hall + team_2026_heatmap
  + era_scoring_trends. Red-first proven by hash: pre-fix blob 02e97b9d, current-env render
  691acd8f (matches commit message exactly). Guard test renders via monkeypatched CHARTS_DIR
  into tmp_path with an escape assertion — does not touch assets/. Suites: 536 unit /
  20+1 integration, sole fail = BL-13 (unchanged).
- **Evidence correction (Gaffer's report, not the fix):** "underlying data byte-identical
  to 2026-06-22" is FALSE — data/top100/all_time_top_100.csv changes EVERY weekly refresh
  (17/100 score rows differ since 06-22, incl. top-10 ranks 8+10 at ~5e-06). Conclusion
  survives anyway: I rendered with the 06-22 CSVs in the current env → byte-identical to
  current render, so the old-blob drift was 100% environmental. Right verdict, wrong
  stated evidence. Also: era_scoring_trends refresh embeds BL-03's matches_2025 repair
  (a content delta), and the diff touched docs/pending-tasks.md (9 files, not 8).
- **SV28-N5 MED (Gaffer): new integration test lands inside weekly_refresh Phase 3d
  fail-closed gate (:400, exit 1 pre-commit) — §6.2 behaviour test 2 ("alters what a gate
  verifies") arguably triggered, yet BL-04's scope reasoning said "no gate touched."
  Consequence unstated in commit: next matplotlib/font upgrade aborts the WEEKLY RUN at 3d
  (no phase regenerates charts) instead of failing in a dev loop. Measured risk low (weekly
  score wobble does NOT flip bytes; only real env/data shifts trip it — by design), but the
  recovery path ("regenerate charts deliberately, commit, re-run") exists only in a test
  docstring. Outcome: codify whether tests/integration additions are §6.2-in-scope, and
  document the 3d chart-failure recovery. Extends SV28-N4 (same class-10 boundary).**
- §6.2 process delta vs BL-02: genuine improvement — reasoning stated explicitly in the
  commit, engages both tests. The call itself still under-scoped (above). SV28-N4 remains
  open; N5 is its concrete instance.

## REVIEWED 2026-07-28 — BL-05: work verified sound, but THE COMMIT DOES NOT EXIST
- **SV28-N6 CRITICAL (Gaffer, escalated to human): reported commit 9c33a5fe1 is not in any
  ref, reflog, or dangling object. The entire BL-05 change set is UNCOMMITTED in the live
  working tree**: record-sentinel-verdict.sh (+60), refresh_and_rank.sh REPO_ROOT fix (+8),
  test_harness_wiring.py (+19), plus UNTRACKED scripts/smoke_harness.sh,
  tests/unit/test_verdict_findings.py, test_smoke_harness_contract.py. HEAD==origin/main==
  3988a8205. pending-tasks.md BL-05 NOT marked DONE (consistent); claimed BL-14 entry
  EXISTS NOWHERE (grep docs/ + Gaffer memory = 0 hits). A `git checkout -- .` would silently
  resurrect the worktree-isolation hole. Until committed: BL-05 not shipped.
- Substance all verified GOOD: 549 unit pass (22.7s); red-first proven (6/7 new tests fail
  against HEAD's script in scratchpad); escaping proven end-to-end (quotes/newline/CJK
  roundtrip valid JSON; printf counterfactual demonstrably invalid); stamp-grep claim real
  (check-council-stamp.sh:66 `"verdict":"[^"]*"`, compact separators used + commented);
  fail-closed on malformed findings-file; COUNCIL_REQUIRE_FINDINGS opt-in correct; findings
  persistence observed working in run-3 smoke log (PASS records with counts, BLOCK with
  full findings array).
- **Fake-sandbox claim CONFIRMED REAL (class 8/NEW): refresh_and_rank.sh REPO_ROOT hardcode
  dates to 6f4cecda5 (2026-04-27).** weekly_refresh.sh:32 was already BASH_SOURCE-safe, so
  exposure = refresh_and_rank only. This session: smoke runs 1 (16:09–16:12) + 2 (16:13–
  16:36) ran PRE-fix (fix mtime 16:39:20, run 3 start 16:39:57) and run 2's log names its
  LIVE-repo writes: next_round_22_prediction_20260728_1630.csv (STILL PRESENT untracked =
  uncleaned smoke dropping), assets/charts incl. hall/*, _stat_leaders.{md,json},
  year_2026.csv @16:17. Doc/chart writes byte-identical to HEAD (BL-04 determinism absorbed
  the hit — verified `git diff docs/hall-of-fame/ assets/` empty). Run 3 post-fix: exactly
  1 live-path ref in log; worktree's own dirty status proves sandboxing now real. No
  evidence of earlier-session contamination (BL-01..04 committed directly on main; only
  stale June worktrees predate; /tmp/supercoach-smoke has only 2026-07-28 runs).
- Phase claims match run-3 log: 1/1c/3b PASS, Phase 3c Skeptic BLOCK (2 blocking findings
  on afl-insights.md L24 prose) → FATAL abort before Phase 4. "Green enough" question real
  but BL-14 never filed.
- **SV28-N7 MED (Scientist via Gaffer): data/top100/yearly/year_2026.csv uncommitted drift —
  last committed 2026-07-07 (4d289665b); R21+R22 refreshes regenerate it (200-line delta,
  live diff == smoke-worktree diff) but no allowlist ever commits it.** Predates smoke runs.
  Decide: allowlist it with round data, or stop regenerating in the live tree.

## VERIFIED 2026-08-29 — SV28-N6 RETIRED; finals-cycle DEEP survey (round 25)
- SV28-N6 RETIRED: BL-05 change set committed 2026-07-28 18:15 as 635ec3d73 (all 6 files +
  smoke_harness.sh + pending-tasks update; phantom hash 9c33a5fe1 in 0 refs — was a
  misreported hash, not lost work). smoke_harness.sh applies working-tree diff by design.
- Context for next survey: 08-18 R25-label cycle FAILED at Phase 3c (last_refresh_status.json
  rc=1 phase 3c 12:08:43) and was MANUALLY completed (commit 6c9c2ff3d 12:30:17, sentinel
  hand-written 12:31:04). The two markers now disagree permanently for that cycle.
- **SV29-F1 CRITICAL (Gaffer+Scientist): finals blindness — unmodified harness run during
  finals fabricates "Round 26"** (get_next_round = max int round+1; prediction.py:999-1027;
  cheat sheet find_latest_prediction picks HIGHEST round from year-less filename, so a
  phantom next_round_26 CSV would poison selection into 2027). Full analysis in the
  2026-08-29 engagement report (returned in-chat, not filed under surveys/).
- **SV29-F2 HIGH (Scientist): scrape-side gates fail OPEN on finals rows** —
  check_round_settled.py:52-59 numeric-only; audit_match_rounds (game_scraper.py:248-252)
  excludes finals. First finals scrape ships gated only by phantom_row_validator (which IS
  finals-aware). Human decision needed on whether September gating is wanted.
- **SV29-F3 HIGH (Gaffer): completed_runs.json entry 20260818_114620 (R24 scoring run)
  uncommitted since 08-18** — fresh clone would quarantine + re-score R24. Manual-completion
  protocol never codified: mark ran, commit didn't; status marker not maintained.
- **SV29-F4 MED (Gaffer): generate_predictions_section stamps gen_date=today on any refresh
  (update_team_analysis.py ~3861)** — post-season reruns publish false freshness on a played
  round. Related: prediction filenames carry no year (next_round_25_* exists for BOTH 2025
  and 2026 seasons; by-archive scoring resolves by mtime only).
- SV28-N7 STILL OPEN (re-confirmed: year_2026.csv dirty in worktree 08-29).

## STILL OPEN (carried, unaffected by this ship)
- **F13 (07-10) backtest writes into live next_round_* namespace** — quarantine now hides
  ORPHANS from mtime consumers but the namespace collision itself unverified as fixed.
  Owner Scientist+Gaffer. Check next cycle log.
- **S11-F6** Optuna cache feature-set-blind (Scientist). **S11-F2** fan-pack stale-CSV
  selection (Gaffer) — recheck next release. **S11-F3** SUBPAGES verdict loop — recheck.
- **WF-F4** synthetic-date mint in player_scraper.py numeric-round rows (Scientist);
  fix_synthetic_dates.py still unwired last checked.
- **F4 (07-10)** TOG% raw same-game feature / backtest masking (Scientist) — status unknown,
  re-audit at next model-touching change.
- S1b MAE needle: first clean reading = R21 backtest (next cycle); series R16 3.981 / R17
  3.825 / R18 3.767 / R19 3.973 / R20 3.920 (R20 now clean via re-derivation).
- F15 (07-11) active-player as-of protection (Gaffer+Scientist). F8/F19 (07-07) +
  0703-F2: unchanged. F12 stale worktrees (~560MB, still present — 3 under
  .claude/worktrees). C2 renumbering-duplicate vector, C3 year-skip residual (Scientist
  backlog).
- MED-LOW (Chronicler): no run report for 07-20 R21 cycle; and this four-commit hardening
  ship (07-25/26) also has no run report yet.

## Prior retirements
See .claude/surveys/ files and git history of this memory for the full retirement trail
(2026-07-09 … 2026-07-26).
