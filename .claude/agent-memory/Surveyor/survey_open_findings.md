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
- SV26-N3 MED-LOW (Scientist): _load_top30_player_deviation still globs the backtest dir
  and trusts any timestamped file — it does not consult completed_runs.json. A future
  orphan landing between quarantine sweeps with a newer timestamp would win under the
  (now-correct) selector. Same family as Gaffer's own F13 backlog item; fix is
  manifest-filtering or namespace isolation, already on Gaffer's open-backlog memory.
- Chart PNG non-reproducibility + generate_backtest_section() side-effect write: logged in
  Gaffer project_open_backlog.md item 3, owner Scientist. Tracked there, not duplicated.

## STILL OPEN (carried, unaffected by this ship)
- **F3 (07-10) lineup garbage — Scientist half only**: 700 garbage rows still in committed
  CSVs (re-measured 07-26: 700/33,999); delete+re-scrape recipe at experiment-log.md:330.
  Allowlist half DONE. Owner Scientist.
- **F13 (07-10) backtest writes into live next_round_* namespace** — quarantine now hides
  ORPHANS from mtime consumers but the namespace collision itself unverified as fixed.
  Owner Scientist+Gaffer. Check next cycle log.
- **S11-F6** Optuna cache feature-set-blind (Scientist). **S11-F1** matches_2025 missing
  PA v Coll R2 2025 + dedup key (Scientist). **S11-F2** fan-pack stale-CSV selection
  (Gaffer) — recheck next release. **S11-F3** SUBPAGES verdict loop — recheck.
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
