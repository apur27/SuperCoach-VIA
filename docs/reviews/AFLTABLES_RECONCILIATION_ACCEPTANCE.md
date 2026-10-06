# AFL Tables reconciliation: final acceptance

**Date:** 2026-10-06. **Decision (Gaffer): ACCEPTED WITH CONDITIONS.** This covers the corrections, the weekly changed-season gate and the harness repair described in `docs/reviews/AFLTABLES_RECONCILIATION_RUN.md` Part A. It is an acceptance of the work as delivered. It does not upgrade the data verdict: that stays **UNKNOWN, not PASS**.

Reviewers inspected the final code and run artifacts under `var/reconciliations/afltables/`, not the run summary. Opus Surveyor did this read-only. QA ran twice, and the second run was on the rebased tree.

## What was confirmed, and how

| Claim | Confirmed by | Result |
|---|---|---|
| Data verdict is UNKNOWN for both layers, not PASS | QA and Surveyor parsed `2026-10-05-corrected/reports/final-cold-4/report.json` (sha256 `81ac70ba…`) | UNKNOWN for the snapshot and for the legacy CSVs. Each layer has 63 `CELL_UNRESOLVED` cells (41 `R-BR-AWARD-SUM-MISMATCH`, 20 `R-ALL-ZERO-UNPROVEN`, 2 `R-PCT-NEVER-ZERO-FILLED`), 35 unresolved aggregates and 0 confirmed discrepancies |
| Acceptance was not pre-marked | Surveyor, `completion.json` (`5cdad784…`) | `.acceptance.status` = PENDING with `report_sha256` null; `.overall_data_verdict` = UNKNOWN |
| Warm-compare target miss is stated | Surveyor | 127.2 s against a 120 s target, stated in `completion.json` `.open_items` and in run record A3 and A9 |
| §6.2 smoke run passed | Surveyor read `2026-10-06-gate-live/20261005T215424Z.log` | Phases 0 to 4, exit 0. Commit and push were suppressed by the shim (log lines 18313, 18580, 18581). The three earlier failed logs fail for the reasons A6 gives. No harness file changed after the run started (mtimes; the wrapper recorded no diff hash). |
| No cycle is active (§6.1) | QA and Surveyor | `.claude/audit/last_refresh_status.json` in the main checkout reads phase 4, exit 0, 2026-08-29, and no harness process is running |
| Corrections applied as recorded | Surveyor | 80 of 80 sampled cell changes from rounds 2 and 3 are present in `data/`; all 107 of the 2026 live-gate changes are present |
| The 4 deleted players are duplicates | Surveyor and Scientist | Every row of each deleted file matches a kept canonical record (paternoster, ross, green, steele). Steele's 2 stub rows differ only in date. |
| Data commit is structurally safe | QA | All 13,193 modified player files were checked: 0 header changes and 0 row-count decreases. The 4 match files keep their columns; 2026 gains its Grand Final. The awards file has 6,989 rows with no duplicate keys. |
| Allowlist and ignore rules | Surveyor and QA | Every changed path is on the A8 allowlist. No new file is git-ignored, and 0 files under `var/` are tracked. `web/src/lib/provisional.ts` was not touched, and the snapshot candidate was not promoted. |
| Fast tiers on the rebased tree | QA re-gate | Legacy: 595 passed, 18 skipped, 0 failed. scvia: 1,501 passed, 0 failed (this includes the 8 new `sink.py` tests). |
| Integration tier on the rebased tree | QA re-gate | 20 passed, 1 skipped, 1 failed. The failure, `test_top100_chart_reproduces_byte_identically`, is pre-existing: it fails with identical hashes (`8c0fef10…` vs `691acd8f…`) on `origin/main` `39985abb1`. |

## Findings, and what was done with each

**QA FAIL (first run), now cleared.** The banner and README still said 13,367 player files while the tree has 13,364 (4 duplicates deleted, 1 added). Fix: the canonical generator `scripts/update_eval_surface.sh` was re-run on the rebased tree, and the resulting count-only diff in `README.md` and `docs/banner.svg` was committed. The generator also rewrites `docs/afl-backtest-2026.md`. That output was **discarded, not shipped**: in this worktree the backtest run logs are absent (they are untracked and exist only in the main checkout), so every round's provenance fell from "attested" to "not attested". Its stale "1,818 of 13,367" line is left for the next weekly cycle, which regenerates it under DataSentinel with the logs present.

**Fixed before commit (Scientist):**
* A2's Brownlow row count is 6,989 (6,972 from round 2 plus 17 from round 3), not 6,972.
* A2's provenance claim now covers cell-level lines only. The 12,628 identity-level lines and the 2 deletions carry no finding id or body hash, and the record now says so.
* A2's per-rule counts are labelled by layer.
* A1 and A7 now separate the integrity report's embedded digest (`6b3b776e…`) from its byte sha256 (`6f6eb3e2…`).
* A7's `ruff format` PASS now names its true scope.
* A6 now discloses that the smoke run needed `RECON_DATA_ROOT` and `RECON_GATE_BASE` overrides. Their values were not recorded. As a result, the gate's audit/fix/commit path has not yet run inside a harness cycle.
* A9 now records the model-input vintage break (M3) and the re-scrape revert risk (M4).
* `test_reconciliation_real.py` was formatted.
* The pre-commit hook rejected `sink.py`, a new module with no test reference. `tests/scvia/unit/test_reconciliation_sink.py` was added (8 tests).

**Open, queued as next-cycle harness work under §6.1 (not patched now):**
* **H1 (HIGH, Gaffer):** a skipped or warned reconciliation gate logs "Reconciliation gate passed". On a clone with no `var/finalized/data` the gate skips every cycle while the log reads as a pass. Skip, warn and pass need distinct log lines and a persisted record.
* **H2 (HIGH, Scientist):** the gate's audit/fix/commit path has never run inside a harness cycle. The first real changed-season cycle is its first end-to-end test.
* **M2 (MEDIUM, Gaffer):** `--fix` can make row or identity edits after the phantom-row and match-completeness gates without re-running them.
* **M4 (MEDIUM, Scientist):** a re-scrape can overwrite a corrected row (`dedup_player_performance`, `keep='last'`), and a fail-open week leaves it reverted.
* **M3 (MEDIUM, Scientist):** about 175,631 rows dated 2005 or later have corrected dates, which feed `days_since_last_game`. Record this as a vintage break before any 2027 backtest is pooled.
* **M5 (MEDIUM, Gaffer):** the fast tier takes about 150 s (scvia 127 s, legacy 19 s) against CLAUDE.md's 20 s budget, inside the hook's 300 s timeout.
* **L2 (LOW, Gaffer):** the smoke wrapper should record the sha of the diff it tested.
* **Pre-existing:** the top-100 chart no longer reproduces byte-for-byte in the current rendering environment. This will abort Phase 3d of the next weekly cycle unless the charts are regenerated and confirmed rendering-only first (CLAUDE.md §6.2 recovery).
* **Test fragility (new finding):** `scripts/scvia_weekly.sh` refuses to run when `pgrep -af` matches the harness script names on *any* command line. A commit message naming those scripts, passed through a shell heredoc, made 19 `test_weekly_candidate` tests fail inside the pre-commit hook; it also explains the 19 failures A7 recorded. Workaround: pass commit messages with `-F <file>`.
* **Condition:** run a full weekly cycle with `FINALS_MODE=1` soon after this merge, so the backtest doc and charts are regenerated under the gate chain (Surveyor M1).

## Out of scope, not done
* The snapshot candidate `var/candidates/20261005-afltables-corrected` was not promoted.
* `web/src/lib/provisional.ts` was not changed.
* No historical truth is claimed beyond agreement with AFL Tables. Its figures are unofficial, and 63 cells per layer remain unresolved because the source itself is incomplete.
