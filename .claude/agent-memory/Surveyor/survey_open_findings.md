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

NEW — SV26-N1 MED (Gaffer): CLAUDE.md carries an UNCOMMITTED working-tree edit raising the
fast-tier budget 10s→~20s and documenting the two tiers, while Gaffer's pending-human list
frames the budget breach (15.1s; I measure 16.0s) as awaiting the human's call. A drafted
resolution sitting ahead of the decision — either commit as human-approved or revert
pending it. Verify: CLAUDE.md clean OR budget decision recorded.

PENDING HUMAN (confirmed accurately described, do not route):
- afl-insights.md double Skeptic-BLOCK: main still carries "extended their lead"/"same
  skill expressed twice" prose; softened fix HELD in working tree (diff verified).
- 1360 vs 1,360: REAL — hall-of-fame-top100.md:26 goals "1360" beside disposals "2,867";
  golden fixture expected_top100_section.md:12 now PINS the bare form (fix needs renderer
  + fixture together).
- R18 2026 coverage gap: per-round table R18 n=284 (14-team archived forward CSV) vs ~371
  played — both README and doc now use the same vintage consistently; disclosure wording
  is the human call.
- Fast-tier 10s budget (see SV26-N1). Monday cycle carries new blocking Skeptic gate.

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
