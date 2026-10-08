# Survey — 2026-10-07 — scope: DEEP + META (Surveyor role, read-only)

## Executive read

The corrected code and data are merged into main, while the overall source-data verdict remains **UNKNOWN: zero confirmed discrepancies and 63 unresolved source cells per layer**. The corrected candidate is not yet published, and promotion should wait for release evidence bound to that candidate and an explicit UNKNOWN notice on the website. The current live preview correctly retains its older snapshot's FAIL notice. All five latest push workflows failed overall, although the web job passed; Fable also found that GitHub does not currently enforce required checks on main. The local fast tier passed all 2,127 tests in 65.7 seconds, above the documented budget.

This is the consolidated review from Cursor Fable's initial DEEP + META survey and follow-up. Codex incorporated the corrections, checked key claims against source and evidence, and rehashed the candidate fragments. Original responses and logs are retained beside this report. This review made no application, data, harness, policy, publication or Git changes.

## Identity, state, method, limits

- Reviewed HEAD: `8324dc0be7a99641dc356dac9352a113270ddece` = `origin/main` (both re-read at start and end). Date: 2026-10-07, Melbourne.
- Working tree at start and end: ` M .claude/audit/last_refresh_complete.json`, ` M .claude/audit/last_refresh_status.json`, `?? insights_datasentinel_2026-10-06.json`, `?? insights_skeptic_2026-10-06.json` — unchanged by this review (`git status` diffed before/after every probe that could write).
- Routing observed by Codex: Cursor CLI `2026.10.01-e373342`; session `e66fdf23-9f86-4fa6-a750-ca62a1063f9e`; requested model `claude-fable-5-1-thinking-high`; Cursor init reported `Claude Fable 5.1 300K High`. These record the requested route and the service-reported identity.
- Method: read-only file reads, `git` queries, `gh` read-only API calls, two live HTTPS GETs against the public site, Python probes with the repo `.venv` (pyarrow/pandas) over `var/` artifacts, and three pytest timing runs with `-p no:cacheprovider` and a throwaway `COUNCIL_GIT_LOCK`. No agent was launched, no production script executed, no network capture made, no tracked application or data changes. Probe outputs are in `/tmp/fable_*`; Codex retained the reports and logs in this ignored review directory. Tests may create ordinary ignored runtime artifacts. HEAD, Git status and the four pre-existing audit-file hashes were checked for preservation.
- Limits: I did not re-run `compare` to reproduce `81ac70ba…`; I did not recompute the 13,193-file diff claim; I did not read capture objects; the `.claude/audit/*.jsonl` logs were only sampled; `docs/reviews/CLAUDE_OPUS55_REVIEW.md` was read by section headers and §0 only.

## Ranked findings

### F1 — New snapshots lose the explicit source-audit notice and noindex  [HIGH]

- Evidence: `web/src/lib/provisional.ts:2-12` returns audit metadata only for `3de65975…`. `ProvisionalAudit` and `ProvisionalRobots.astro:6` consequently emit no audit notice or `noindex` for other snapshots. `web/tests/unit/provisional-status.test.ts` explicitly expects empty output for another snapshot.
- Failure path: building a release from `b9830cbf…` drops the explicit UNKNOWN verdict, report links and robots restriction unless audit handling changes.
- Qualification: `web/src/pages/data-status/index.astro:22` reports freshness separately: it shows a stale banner when appropriate and a current banner otherwise. Coverage and dataset-status rows remain visible. The absence of the source-audit notice does **not** itself label every other snapshot clean. The currently published `3de65975…` preview is correctly labelled FAIL.
- Owner: Gaffer.
- Outcome: bind audit verdict and report identity to the selected snapshot. An unaudited real snapshot gets an explicit unaudited/provisional state; an UNKNOWN snapshot keeps its UNKNOWN notice. Do not apply the old snapshot's FAIL report to a new snapshot.
- Acceptance: negative tests cover an unpinned real snapshot; it cannot silently lose the audit notice or preview `noindex`. A candidate build shows UNKNOWN, the correct evidence references and 63 unresolved cells. Freshness and source-audit confidence remain distinct. Demo behavior is checked separately.
- Effort: S.

### F2 — The corrected candidate lacks its own completed publication chain  [HIGH]

- Evidence: `var/candidates/20261005-afltables-corrected-releases/build-release.json` names parent snapshot `f1abd8c2…`. Its release directory has public resources and validation, but no sealed site. The full source report `81ac70ba…` names that parent. The newer report `6fecac8e…` names `b9830cbf…` but covers only season 2026. Parent release consistency does not certify a release built from the child.
- Impact: the corrected snapshot exists, but its release, input-consistency report, seal and hosted site still need to be produced and bound together.
- Owner: Scientist.
- Outcome: build and validate a release from the exact candidate; produce a complete input-consistency report; build the website with correct audit metadata; seal it and verify the deployment bundle. Preserve the overall UNKNOWN source verdict.
- Source-evidence acceptance: either produce a correctly bound new full audit, or document a composite argument using both existing reports with their original snapshot IDs and hashes. The latter must prove historical partitions unchanged, use the newer report for 2026, inspect the changed quality metadata, and locate all residual unresolved cells in the covered partitions. Neither existing report may be relabelled as a full audit of the child.
- Independent Codex check: all 411 unique fragment files referenced by the parent and candidate were rehashed with **zero hash mismatches**. Only `player_games/2026` and `quality_issues` differ. Evidence: `fragment-verification.json`. This verifies fragment identity; it does not replace the required source-coverage record or candidate release audit.
- Release acceptance: `public/release.json.snapshot_id` equals the full candidate ID. Its actual bytes match `checksums.json.files["release.json"].sha256` and the seal entry `files["data/<release_id>/release.json"].sha256`. Validation is PASS, names the same release, and binds the checksum file digest and `seal_sha256`. Re-run the canonical validator against the candidate output root to check the whole inventory; these field comparisons alone are insufficient.
- Integrity acceptance: the report has `inputs.snapshot.snapshot_id` equal to the candidate, `inputs.release.release_id` equal to the release, `outcome == "PASS"`, and both `scope.complete` and `scope.semantic_complete` true. Supply the source/content/model inputs needed for full semantic coverage; missing inputs cannot be accepted as complete.
- Contract correction: `seal.json` has **no top-level snapshot ID**. It binds the snapshot through the hashed release manifest. The source audit remains UNKNOWN even when release consistency and seal validation PASS.
- Effort: M. Building/validating writes new output artifacts; no such candidate build was performed during this review.

### F3 — Current CI is red and required checks are not enforced on main  [HIGH] [class 7: documented sequence vs actual]
- Evidence: `gh run list` 2026-10-06: `scvia-ci`, `Pylint`, `Python Package using Conda`, `Unit Tests`, `.github/workflows/weekly-fan-pack.yml` all `failure` on every push. (a) `weekly-fan-pack.yml` is invalid YAML — `yaml.safe_load` fails at line 84 (heredoc inside `run: |` not indented); last change `967cc9359` (2026-07-11); last successful release `weekly-2026-07-08`. It is `schedule`d, has `permissions: contents: write`, and uses `softprops/action-gh-release@v2` by tag, unlike the SHA-pinned `scvia-pages.yml`. (b) `scvia-ci` python job: `1 failed, 1504 passed` — `tests/scvia/unit/test_builder.py:629` hard-codes `/usr/bin/node` (known since `docs/pages-preview.md:39`, still open). (c) `tests.yml` (Python 3.10), `pylint.yml` (3.8–3.10), `python-package-conda.yml` (3.10) contradict `requires-python >=3.12`. GitHub also still lists `.github/workflows/ci.yml`, which does not exist in the tree.
- Impact: standing failures obscure new regressions. The scheduled fan-pack workflow requires an explicit release-policy decision before re-enabling it. Fable separately queried GitHub: main reports `protected: false`, branch protection returns “Branch not protected”, and the one listed ruleset has `enforcement: disabled`. CI is currently informational as a remote merge/push gate; the local commit hook still enforces its own checks.
- Owner: Gaffer.
- Outcome: repair active CI and explicitly decide the fate of legacy/fan-pack workflows. The Python CI job must install the pinned Node runtime and `web` dependencies, then run the cross-language contract test as a required check. Resolve Node through PATH rather than `/usr/bin/node`; missing required runtime or Vitest must fail CI. The ordinary web job cannot substitute: its Python-release test is skipped when `SCVIA_PY_PUBLIC` is unset. Do not hide the contract test behind a CI skip.
- Acceptance: the required jobs for the exact new HEAD are green, with evidence that the Python-to-TypeScript contract test actually ran. A negative check with missing required Node/Vitest fails. Audit active workflows explicitly rather than assuming the five most recent runs cover all of them. GitHub enforcement changes remain a separate owner decision. Effort: S.

### F4 — The weekly gate audits a stale snapshot layer it then ignores  [MEDIUM] [class 2/7]
- Evidence: `scripts/weekly_refresh.sh:168` → `--data-root ${RECON_DATA_ROOT:-var/finalized/data}`; `var/finalized/data/current.json` → `aa836549…` (promoted 2026-09-28; the control snapshot with 3,010,288 findings in RUN.md Part B §6). `reconciliation_gate.py:72-89` decides on `legacy_csv` only. The live gate run's proposal (`2026-10-06-gate-live/fix-verify/proposal/summary.json`) confirms: `snapshot_id aa836549…`, 86,550 proposed changes of which 86,443 are snapshot-layer (`R-TOTAL-NONBLANK 77,140`, `R-BR-AWARD-SUM 8,901`) against 107 legacy changes.
- Impact: every changed-season gate run computes and discards a snapshot comparison of a vintage nobody ships; the "accepted snapshot data root" label in the log and docs is misleading; after any promotion the two roots will diverge silently.
- Owner: Scientist.
- Outcome: the gate's plan is legacy-only, or `RECON_DATA_ROOT` is bound to the promoted corrected root and the binding is asserted in `gate.json`.
- Acceptance: `gate.json.layers` has only `legacy_csv`, or `plan.json` snapshot equals the promoted id. Effort: S.

### F5 — Published status text is stale on three surfaces  [MEDIUM] [class 1: staleness drift]
- Evidence: `README.md:38` "Independent acceptance of the new reconciler | Pending | Implementation is still in `work/afltables-reconciliation`; it has not been merged" — but `git merge-base --is-ancestor work/afltables-reconciliation main` is true and `docs/reviews/AFLTABLES_RECONCILIATION_ACCEPTANCE.md` records ACCEPTED WITH CONDITIONS on 2026-10-06. `README.md:36` "Legacy CSVs in `data/` | FAIL" — the current `data/` content hash `460dba95…` (recomputed with `inventory.pin_legacy`) is exactly what `6fecac8e…` audited PASS for 2026, and the full-run legacy verdict is UNKNOWN with 0 fail findings. `docs/rewrite/afltables-reconciliation/README.md:9-20` ("Status — 3 October", "not yet in main", "paused"). `docs/data-integrity.md:3-7` ("reconciliation reports FAIL for its snapshot and for the legacy CSVs"). `docs/reviews/AFLTABLES_RECONCILIATION_RUN.md` A9 still lists H1/M2 as "queued, not applied" although `238cacf46` landed them.
- Impact: the README is the owner-designated public truth surface and is linked from every hosted page's notice; it currently understates what was done and overstates the legacy verdict.
- Owner: Gaffer. The 3de6… FAIL rows remain correct for the hosted site and must stay until F1/F2 are resolved; "Why this repo exists" is untouched by this.
- Acceptance: each row cites a report sha and a commit; DataSentinel PASS on the README block. Effort: S.

### F6 — CLAUDE.md's test contract names a dead interpreter and a budget for a tier that no longer exists in that shape  [MEDIUM] [class 10: unwritten/stale convention]
- Evidence: CLAUDE.md §5 "Run with: `/home/abhi/sourceCode/python/coding/.venv/bin/python -m pytest tests/ -m "not integration"`" — path absent (RUN.md A6; `.githooks/pre-commit:29` now defaults to repo `.venv`). Budget "under ~20 seconds… raised at ~490 tests". `pyproject.toml:67` says "~10s". Measured today (see Test budget).
- Owner: Gaffer (CLAUDE.md is the council's contract; budget change is the owner's).
- Acceptance: CLAUDE.md names the interpreter resolution rule (`harness_env.sh`) and a budget per tier that matches `tests/unit/test_precommit_*` expectations. Effort: S.

### F7 — The hook's test gate fires only on staged `.py`; harness `.sh`/hook edits commit untested  [MEDIUM] [class 7]
- Evidence: `.githooks/pre-commit:40` `staged_py=… grep -E '\.py$'`; everything from line 42 to 122 is inside `if [ -n "$staged_py" ]`. `tests/unit/test_harness_wiring.py` and `test_smoke_harness_contract.py` exist but run only when a `.py` happens to be staged. `238cacf46` and `f3878f3d0` were covered only because they also touched tests.
- Owner: Gaffer. Outcome: the tier also runs when `scripts/*.sh`, `refresh_and_rank.sh` or `.githooks/*` are staged (the §6.2 smoke remains the merge condition; this closes the "lint-free but untested" gap). Effort: S.

### F8 — Harness prompt instructs a definitional claim the Skeptic flags every run  [LOW] [class 3: instruction conflict]
- Evidence: `scripts/weekly_refresh.sh:412` tells FootyStrategy "say that they overlap by definition"; `.claude/audit/insights_skeptic_2026-10-06.json` and all three smoke runs (`smoke.out`, `smoke-h2.out`, `smoke-m5.out`) return PASS_WITH_CONCERNS on exactly that sentence as unsourced. Owner: Gaffer. Outcome: either the source table (`docs/afl-stat-leaders-2026.md`) carries the "(mechanically related)" label for that pair, or the prompt stops asserting it. Effort: S.

### F9 — Audit markers and verdict records drift out of git every cycle  [LOW] [hygiene]
- Evidence: `last_refresh_status.json`/`last_refresh_complete.json` are tracked (last committed 2026-08-29) and modified by every cycle; `insights_*_2026-07-13…08-29.json` are tracked, `…10-06.json` are not; the Phase 4 allowlist (`weekly_refresh.sh:524-544`) stages none of them. A fresh clone reads an 08-29 §6.1 marker. Owner: Gaffer. Outcome: either untrack the machine-local markers or stage them deliberately. Effort: S.

### F10 — Agent permission file is over-broad and references the dead interpreter  [LOW] [hygiene]
- Evidence: `.claude/settings.local.json` allows `Bash(git *)`, `Bash(python *)`, `Bash(kill *)`, and `Bash(/home/abhi/sourceCode/python/coding/.venv/bin/python *)`; the CLI printed a wildcard-position warning on a `cp … *.csv` rule. Owner: Gaffer. Effort: S.

### F11 — QA has conflicting test and pre-existing-failure rules  [MEDIUM]

- Evidence: `QA.md:35` invokes the retired interpreter and serial `pytest tests/ -v`, mixing the fast and integration tiers. Lines 37, 40 and 203 respectively say any failure is FAIL, pre-existing failures do not block, and PASS permits pre-existing warnings. `Gaffer.md:20` treats QA FAIL as absolute.
- Owner: Gaffer. Outcome: one explicit command per tier and one consistent policy for pre-existing failures, matching Gaffer's handling.
- Acceptance: the same documented failing scenario produces one unambiguous decision in QA and Gaffer. Effort: S.

### F12 — Agent instructions describe different weekly approval chains  [MEDIUM]

- Evidence: `Gaffer.md:107` inserts a QA agent before shipment. `.claude/skills/weekly-cycle.md` names the shell harness as the single orchestrator; that harness runs deterministic gates and Phase 3d tests, but no QA agent. `QA.md:23` also describes a standalone role after refresh.
- Owner: Gaffer. Outcome: document whether the weekly cycle requires an agent verdict or the harness's deterministic checks, and align all three instructions with the intended behavior.
- Acceptance: trace the documented approval chain against a recorded cycle; every required verdict/check has an actual producing step. Any behavior change follows CLAUDE.md §6.2. Effort: S.

### F13 — Skeptic's declared scope omits the weekly recap it checks  [LOW]

- Evidence: `Skeptic.md:17-20` lists briefs, news and post-mortems, while `Gaffer.md:60` and the harness also use it for `docs/afl-insights.md`.
- Owner: Gaffer. Outcome: make the recap remit and applicable checks explicit.
- Acceptance: the definition and actual recap invocation agree; sample a recap against the stated checks. Effort: S.

Labelled inference: F1 describes the source-audit notice behavior inferred from code; the corrected candidate has not been published. F4's production behaviour is inferred from the gate-live artifacts plus the default in the harness; the smoke runs used overrides whose values were not recorded (RUN.md A6).

## Reconciliation of Claude's latest claims

| Claim | Status | Evidence |
|---|---|---|
| Work finished and merged | Supported | `work/afltables-reconciliation` is an ancestor of `main`; `main..work/…` is empty; commits `0f9a3fe90`…`1add8cdc5`, `238cacf46`, `f3878f3d0` on `main` |
| No cycle running; last cycle 2026-10-06 13:15, phase 4, exit 0 | Supported | `last_refresh_status.json` `{"phase":"4","exit_code":0,"round":"25","ts":"2026-10-06T13:15:11+1100"}`; `pgrep` finds no harness process; no crontab, no user timer |
| Data verdict UNKNOWN, not PASS; 0 confirmed discrepancies; 63 unresolved cells per layer | Supported | `final-cold-4/report.json` sha `81ac70ba…`; findings stream sha `3166d1b6…`, 1,478 lines: legacy 63 unknown + 431 info, snapshot 63 unknown + 426 info, 0 fail; 63 = 41 `R-BR-AWARD-SUM-MISMATCH` + 20 `R-ALL-ZERO-UNPROVEN` + 2 `R-PCT-NEVER-ZERO-FILLED` |
| Full non-2026 captures date to 1–2 Oct; the later 2026 capture found revisions | Supported | `completion.json.open_items[2]`; `gate-run/gate.json` block with 106 `CELL_MISMATCH`; `candidate/r/report.json` 2026 PASS/PASS |
| Audit agreement, release consistency, capture freshness, activation are distinct | Supported | README and data-status keep them in separate rows; `SWITCH_PLAN.md` status "not activated" |
| "H1/M2 queued for the next cycle, not applied" (RUN.md A9, ACCEPTANCE) | Stale | landed in `238cacf46` at 16:06 on 2026-10-06, after the cycle ended 13:15 (§6.1 respected); the documents were not updated |
| "The gate's audit/fix/commit path has never run inside a harness cycle" (H2) | Qualified | it has now run inside a §6.2 smoke cycle (`2026-10-06-h2-smoke`: planted Perkins marks 7→9, live capture, block, fix, re-audit PASS, re-gate, stubbed commit) but not inside a production cycle |
| "Shipped data differs from the four full runs only in the 107 2026 cells" | Partially verified | main `data/` content sha `460dba95…` equals the 2026 PASS report's `legacy_inputs.content_sha256`; the 107-cell-only delta against `1309b4ec…` was not recomputed here |
| Fast tier "~66 s on 4 workers" (commit `f3878f3d0`) | Supported | measured 65.7 s / 2,127 passed today under load average ≈1.7 |

## End-to-end trace of one corrected value

Karl Amon, 2026 round 12, Hawthorn v Adelaide, 2026-05-21, `one_percenters`:
1. Source evidence **[historical record]**: `fix-verify/proposal/changes.jsonl` line 1 — `R-BOTH-PAGES`, `https://afltables.com/afl/stats/players/K/Karl_Amon.html`, body sha `8bb49287…`, old 3 → new 2 (Oct 6 capture; the Oct 1–2 page printed 3).
2. Legacy CSV **[data]**: `data/player_data/amon_karl_19081995_performance_details.csv` row 205 reads `2.0` at HEAD (`67217df40`); `git show 39985abb1:…` reads `3.0`.
3. Snapshot **[data]**: `b9830cbf…` `player_games/2026` fragment `62aab810…` → `m:2026:r12:adelaide:hawthorn:0` `one_percenters=2`; parent `f1abd8c2…` → 3.
4. Aggregation/public resource **[data]**: the only candidate release (`20261005T105910Z-94be51774067`, built from `f1abd8c2…`) `player-games/k.bGVnYWN5OmFtb25fa2FybF8xOTA4MTk5NQ/2026.json` → `3` (stale hop — no release from `b9830cbf…` exists).
5. Web presentation **[data]**: live `https://apur27.github.io/SuperCoach-VIA/data/20261003T124334Z-08e8e0eb65d4/player-games/…/2026.json` → `3`, under the `3de65975…` FAIL notice (correctly labelled provisional).
6. Gate **[data]**: `candidate/r/report.json` (`6fecac8e…`) 2026 seasons audit PASS/PASS for both layers, GF 2026-09-26 present 46/46; the weekly gate will not revisit 2026 because the season no longer changes (RUN.md A6).

Hops 4–5 are where the corrected value is not yet visible to anyone.

## Snapshot promotion recommendation: HOLD (conditional)

Consumers and pins, current values:
- `var/finalized/data/current.json` → `aa836549…` (production data root; also the reconciliation gate's `RECON_DATA_ROOT` default and the correction-evidence root).
- Correction-evidence pin: `SCVIA_CORRECTION_EVIDENCE_SNAPSHOT=sha256:aa836549…` with root `var/finalized/data` (`docs/rewrite/SWITCH_PLAN.md:116-117`; used by `scvia_weekly.sh:205`; the candidate-smoke `corrections.json` shows evidence snapshot `aa836549…`, season capture `87c99c1a…`). The candidate root contains `aa836549….json` and the Sep-28 `raw/`, so this pin remains resolvable inside the candidate root; it is not an obstacle but must be stated.
- Web UI audit pin: `web/src/lib/provisional.ts` `3de65975…` + report `58eff517…`; live site matches.
- Hall of Fame provisional tables and `docs/hall-of-fame/provisional/provenance.json` → `3de65975…`.
- Candidate: `var/candidates/20261005-afltables-corrected/current.json` → `b9830cbf…`, parent `f1abd8c2…`, status `legacy_unverified`.

Evidence supporting the candidate: full compare of its parent UNKNOWN/UNKNOWN with 0 fails (`81ac70ba…`); 2026 seasons audit of the candidate itself PASS/PASS (`6fecac8e…`); legacy spot-check 1,501/1,501; integrity PASS on the parent's release; GF present; the candidate-smoke rehearsal (`compare.json`: 130/130 yearly lists, all-time delta 0, `verdict.ok true`) ran on an import of corrected CSVs (`eae223d7…`), not on `b9830cbf…`.

Prerequisites before any promotion:
1. F1 fixed: an unpinned real snapshot cannot silently omit its source-audit state.
2. F2: a release built from `b9830cbf…`, a complete input-consistency PASS for that release, and a sealed site. The source-coverage record must use either a new full report or the explicitly documented parent/2026 combination described in F2; preserve each report’s actual snapshot ID. The notice must say UNKNOWN with the 63-cell explanation.
3. Decide and record how the gate's `RECON_DATA_ROOT` follows the promotion (F4).
4. Regenerate the provisional HOF tables from the new snapshot (their README says "Redo them after the audit corrections are verified").
5. README/data-status rows updated through DataSentinel; "Why this repo exists" verbatim.

Distinguish: preview publication (steps 1–2, 4–5, `scvia-pages.yml` dispatch) is independent of pipeline activation (`SWITCH_PLAN.md` §3 steps B/C, two real shadow cycles, owner decisions §6 — none of which are satisfied yet). Promotion settles none of the 63 cells; UNKNOWN stays UNKNOWN.

## Test-budget recommendation (owner decision; coverage preserved)

Measurements today (12 CPUs, load ≈1.7, repo `.venv`, `-p no:cacheprovider`):
- Hook tier as the hook runs it (`tests/ -m "not integration" -n 4`): 2,127 passed, 65.7 s (wall 66.0 s). Slowest: `test_precommit_python_gate` (3.6/3.2 s, spawn real pytest), `test_builder` parallel-build (3.6 s), integrity determinism/cache suites (1.5–3.4 s each).
- `tests/unit` alone, serial: 622 passed, 31.2 s — this is the lineage of the CLAUDE.md "20 s at ~490 tests" figure.
- `tests/scvia` alone, 4 workers: 1,505 passed, 57.0 s — this is the tier the `scvia-ci` "≤30 s" budget (`IMPLEMENTATION_STATUS.md` item 4, O55-07) refers to.
- Comparability: the "59 s" and "66 s" figures (commit `f3878f3d0`) and today's 65.7 s are the same scope and same machine; the "150 s"/"173 s" figures were serial; the 2026-09-25 "66 s" in IMPLEMENTATION_STATUS was `tests/scvia` only on a 4-vCPU box. The time tool reported maximum RSS of about 596 MiB. This is not an aggregate process-tree measurement; it does not establish per-worker memory or a controller-only allocation.

Options, none of which skip coverage:
- A. Keep one hook tier, raise its documented budget to the measured value with the test count (as the 2026-07-26 raise did). Cheapest; keeps the hook at ~1 min.
- B. Re-scope: the hook runs the tier the staged files belong to (`tests/unit` for legacy paths, `tests/scvia` for `src/`/`web/`, both when either is ambiguous), each with its own budget; the full tier runs in `scvia-ci`. Requires `scvia-ci` to be green first (F3).
- C. Reduce wall time without removing tests: the ~15 determinism/cache tests that build a demo corpus per test share a session fixture; the two precommit tests that spawn a real pytest keep running but are marked and still counted. Potential gains are unmeasured. Preserve independent test state and cache-invalidation probes; demonstrate a measured improvement before claiming any saving.
Recommendation: A now (truthful number), C as routine Scientist/QA work, B only after F3. Approval and the number remain the owner's.

## Prioritized implementation sequence (review only)

1. F1 fail-closed notice (Gaffer) — DoD: unpinned real snapshots display an explicit unaudited notice and preview `noindex`; the candidate displays its actual UNKNOWN evidence. Depends on nothing.
2. F3 CI green (Gaffer) — DoD: all active workflows succeed on `main`; fan-pack decision recorded. Depends on nothing.
3. F5/F6 documentation currency (Gaffer, DataSentinel-gated) — DoD: README/data-status/`afltables-reconciliation/README.md`/CLAUDE.md cite current shas and commits. Depends on 1 for the final wording of the web rows.
4. F2 candidate artifacts (Scientist) — DoD: the candidate release and integrity report name `b9830cbf…`; the seal hashes that manifest; the source-coverage record meets F2 without rewriting old report identities. Depends on the capture-binding decision.
5. F4 gate data-root binding (Scientist) — DoD: `gate.json` layers legacy-only or snapshot = promoted id. Depends on 4 if the "follow promotion" option is chosen. In §6.2 scope: smoke run required.
6. Promotion decision (owner) — depends on 1–5.
7. F7 hook trigger scope, F8 prompt/source alignment, F9/F10 hygiene and F11–F13 agent-contract consistency (Gaffer). F7 and any changed harness/gate behavior require §6.2 smoke evidence. Resolve F11/F12 before relying on those instructions for another weekly approval.

## Checked and found clean

- Harness ordering matches its documentation: Phase 0 → 1 → 1c (phantom, completeness, reconciliation gate, re-gate on fix, push) → 1b → 2a → 2b (deterministic HOF gate + recorded verdicts) → 3/3b/3c/3d → 4; commits only via `git_commit_safe.sh`; `core.hooksPath=.githooks`.
- §6.1 respected: `238cacf46`, `f3878f3d0` committed after the 13:15 cycle end; each cites a §6.2 smoke with recorded diff sha (L2 in place).
- `reconciliation_gate.py` column assumptions hold (`year` is column 1 in player CSVs, column 3 in match CSVs; no quoted fields in either corpus).
- `scvia_weekly.sh` process-detection regex fixed (anchored `bash … scripts/weekly_refresh.sh`), locks per data/out/var root, refuses without a cycle marker `exit_code`.
- `scvia-pages.yml`: manual-only, `contents: read` on verify, SHA-pinned actions, HTTPS-only bundle, both digests supplied out-of-band, no Astro rebuild.
- Release path safety: `isSafeRelPath` rejects absolute, `..`, `%`, whitespace; `publish/release.py` private-content scan for `/home/<user>/`; nh3 allowlist for article HTML.
- Live site: canonical base `/SuperCoach-VIA/`, 200; "Why this repo exists" present; provisional banner present; lowercase alias returns 200 and redirects client-side as documented.
- Integration tier (Phase 3d) passes 22/22 in all three 2026-10-06 smoke runs; the chart-reproducibility failure was fixed by `04784d509`.
- 2026 grand final present in legacy `matches_2026.csv` (218 matches) and in the candidate snapshot; `latest_completed_final` matched 46/46 in both layers.
- Agent definitions: every harness `claude -p` call passes task parameters only, no `--model`/`--allowedTools` (class 9 clean); model tiers in frontmatter consistent with roles.

## Focused publication-security review

Fable's follow-up read `publish/deploy.py`, the publication/sealing/rollback sections of `publish/release.py`, `web/integrations/release-tree.mjs`, the resource-reader boundaries and the Pages workflow.

- Archive extraction rejects links, device entries, absolute paths, parent traversal and files outside the permitted layout. Member-count and extracted-byte limits apply; the supplied archive digest is checked before extraction (`deploy.py:74,134-161`).
- Bundle verification checks the expected seal, recomputed inventory identity, validation binding and matching release ID (`deploy.py:88-95`). This relies on trusted expected digests and the validated packing process; a self-consistent attacker-supplied digest is not authenticity evidence.
- Packing checks validation, the seal and embedded data, then uses a fresh tree walk that rejects symlinks (`deploy.py:38-63`).
- Publication captures the validated inventory once; local upload rechecks its copied files before replacement. Rollback deliberately verifies integrity without requiring the newest schema (`release.py:712-733,766-839`).
- Release paths and strict JSON parsing reject unsafe paths, duplicate keys and non-finite values. Web copying checks path safety, symlinks and exact checksum inventory closure before copying (`release-tree.mjs:91-133`).
- Pages verification has read-only repository permission; publication permissions are scoped to the deploy job. Deployment is manually dispatched and uploads the verified tree.

No additional blocking security defect was identified in this focused source review. It was not a penetration test, dependency audit or concurrent-filesystem stress test. Minor documentary constants and download/extraction cost observations remain advisory, not demonstrated vulnerabilities.

## Coverage table

| Area | Checked | Not checked |
|---|---|---|
| 1 Code quality | package layout, config duplication (`config/` vs packaged copy identical), stale paths (`requirements.txt` unpinned legacy, `tests.yml` py3.10, dead interpreter path in CLAUDE.md/settings, duplicate `docs/ARCHITECTURE.md` vs `docs/architecture.md`) | module-level review of the 42k-line `src/`; mypy/ruff re-run |
| 2 Data integrity | one value traced across 6 hops; 63-cell recount; legacy content hash recomputed; GF presence; blanks rule read; dedup `keep='last'` confirmed; snapshot ancestry | reproducing `81ac70ba…`; the 13,193-file delta; Brownlow award table contents; capture immutability re-hash |
| 3 Pipeline/process | ordering, freezes, locks, staged-vs-worktree hop, changed-season detection, warn/block policy, pending persistence, M3 vintage break doc | a live cycle; `check_round_settled`/FINALS_MODE for the 2027 rollover; `backtest_completeness` logic |
| 4 Publication/security | Pages workflow, seal/digest flow, path safety, private-content scan, CI permissions, action pinning, public/private boundary (`var/` untracked, 0 tracked) | secret stores (by rule); dependency audit (`pip-audit` not run) |
| 5 Website | stale notice logic, data-status page, base paths, live probes, a11y table wrappers | mobile rendering, Lighthouse/axe, asset budgets beyond recorded 264 MB |
| 6 Tests/CI | three timing runs, durations, collection counts, CI run status and failing test | serial full-tier timing today; `web` test suites; flakiness over repeats |
| 7 Agents | frontmatter/model/tools, substantive Gaffer/QA/Scientist/DataSentinel/Skeptic contracts, weekly-cycle and council-brief skills, harness prompts and verdict handling | agent memory directories in depth; `.claude/audit/*.jsonl` beyond sampling; full bodies of BriefBuilder, FootyStrategy and Chronicler |
| 8 Stewardship | worktree ancestry and dirty state, stashes, releases, tracked markers | contents of `var/finalized/site.tar` (479 MB), `scratch/`, `archive/` |

## Unrun commands and limitations

Not run: `scvia reconcile-afltables compare` (reproduction of `81ac70ba…`), `scvia check-integrity`, `npm run build/test`, `pip-audit`, any harness or smoke script, serial full-tier pytest. Timing was taken under background load (≈1.7), not idle. The `/usr/bin/time %M` figure is maximum RSS; it is not aggregate process-tree memory and does not establish a controller-only allocation.

## Watch list (speculation, unranked)

- Hard-coded `--train-cutoff 2025-06-01 --calibration-end 2026-05-01` in `scvia_weekly.sh:213` vs `2025-01-01/2026-01-01` in `docs/operations.md:41`: two sources, both will be wrong for 2027.
- `dedup_player_performance keep='last'` still lets a re-scrape revert a correction between gate runs (M4 pending-list mitigates, fail-open weeks do not).
- `weekly_refresh.sh` header still advertises a cron line; no crontab exists — fine if intentional, worth one sentence in `docs/operations.md`.
- `git add -A -- data/player_data data/matches data/awards` (`weekly_refresh.sh:183`) would sweep any stray untracked file in those directories into the corrections commit.
- The 1931–34 Brownlow handling and the 1974 hit-out column will stay UNKNOWN until a second source is pinned; no rule should be invented to close them.
- `docs/afl-backtest-2026.md:215` reads "1,817 of 13,364" — regenerated in the 10-06 cycle; verify provenance "attested" after the next cycle with logs present.

## Safe cleanup recommendations (recommend only)

- `var/worktrees/afltables-reconciliation` + branch `work/afltables-reconciliation`: fully merged (ancestor of `main`). Before removal: copy `.claude/agent-memory/Gaffer/project_afltables_acceptance_retro.md` and the one-line `MEMORY.md` index edit into the main checkout via Gaffer (they exist nowhere else); the 33 ignored entries are caches and a 1.4 GB `.venv`. Run artifacts under `var/reconciliations/` are evidence, not worktree contents — retain.
- `.claude/worktrees/agent-a0656b…`, `agent-a2c502…`, `agent-a8be40…` (2026-06-16): their HEAD commits are on no branch; removing the worktrees orphans them. Tag or branch them first if the June career-total reconciler / draft scraper work is wanted; otherwise record the decision.
- `.claude/worktrees/agent-a5cae8…`, `var/worktrees/cursor-rewrite-grok47`, `var/worktrees/opus55-integrity`: HEADs are ancestors of `main`; safe to remove once their untracked state is inspected.
- `stash@{1}` (2026-04-30) is five months old; `stash@{0}` is the 2026-09-28 pre-merge preservation — keep until the owner confirms.
- Legacy workflows (`tests.yml`, `pylint.yml`, `python-package-conda.yml`) and the broken `weekly-fan-pack.yml` are deletion candidates pending the owner's decision in F3.
- Retain: all four `preview-*` release assets and digests (rollback), `var/finalized/`, `var/candidates/`, all `var/reconciliations/` runs.

## Standing anti-pattern list

Carried forward unchanged: never trust an LLM sum; never verify by exit code; never `git add .`; never hand-edit `data/`; never let Pass-1 stand for Pass-2; never soften an upstream caveat; never run before settlement; never push from parallel agents; never define role/model/tools in two places; never leave a gate convention unwritten.

Additions (evidence this survey):
- Never let a new real snapshot silently lose its source-audit notice. Freshness text cannot replace an audit verdict (F1).
- Never promote a snapshot pointer without an artifact chain (release, integrity report, seal, audit report) that names the same id (F2).
- Never treat a red CI as background noise; one broken workflow file hides every other failure (F3).
- Never leave a status table row without a report sha and a commit beside it (F5).

## Next handoff

Gaffer: F1, F3, F5, F6, F7–F13, worktree memory custody. Scientist: F2 (capture binding decision first), F4. Owner: test-budget option, fan-pack fate, promotion go/no-go after prerequisites. Data verdict remains UNKNOWN with 63 unresolved cells per layer; nothing in this review settles any of them.

## Delivery evidence

- Full review and follow-up: `result.json`, `followup-result.json`; original wording: `REVIEW_ORIGINAL.md`, `ADDENDUM.md`. This consolidated report supersedes their overstatements and draft command examples.
- Cursor session/tool trace: `events.jsonl`; launch brief: `PROMPT.md`; follow-up brief: `FOLLOWUP.md`.
- Tests: `fast-tier.log` (2,127 passed, exit 0). Additional tier measurements are in the Cursor trace.
- Independent fragment hashes: `fragment-verification.json` (411 unique files, zero hash mismatches).
- Preservation: `before.json`, `after.json`, `preservation-final.json`; existing audit records are unchanged.
- No commit, push, deployment, activation, branch deletion or budget change was made for this review.
