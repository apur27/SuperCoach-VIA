# AFL Tables reconciliation: audit, corrections and weekly gate

**Mode:** decision-support → production, blast radius HIGH. **Branch:** `work/afltables-reconciliation` (worktree `var/worktrees/afltables-reconciliation`), base commit `21ca717978d4f573abcd2e048c801a5997eaace7`; `origin/main` has moved 8 commits (the Codex web UI) and no file overlaps. **Nothing is committed**: the tree is handed to Gaffer (A8).

Part A is the corrections run (2026-10-03 to 2026-10-06, Scientist on `claude-opus-5-5`), authorised by the owner to fix the data "once and for all": both layers edited directly from the frozen capture, a season-awards table for pre-1984 Brownlow totals, and a weekly gate on changed seasons with the harness repaired first. Part B is the first, audit-only run (2026-10-01/02); its verdicts are superseded by Part A and kept as the baseline.

## A1 Verdicts after correction

| Claim | Status | Evidence |
|---|---|---|
| Snapshot `sha256:f1abd8c2b7f8d6b3812e73f91a4403add1844f3ad9037139e7dc373a0ed91703` (candidate `var/candidates/20261005-afltables-corrected`; its child `b9830cbf…` adds the 107 cells of A6) | **UNKNOWN**: 0 confirmed discrepancies; 63 cells AFL Tables itself leaves unresolved | `2026-10-05-corrected/reports/final-cold-4/report.json` |
| Legacy CSV layer (this worktree's `data/`, content sha256 `1309b4ec…e96fe1`) | **UNKNOWN**: 0 confirmed discrepancies; the same 63 cells | same report |
| Release `20261005T105910Z-94be51774067` (`scvia check-integrity --scope full --as-of 2026-10-05T12:00:00Z`) | PASS, 26 of 26, complete | `integrity-asof1005/integrity-full.json`: embedded `report_sha256` field `6b3b776e…a6ea607` (the checker's own content hash, also the value `completion.json` records for the release); file byte sha256 `6f6eb3e2…21c28c16` |
| Independent spot-check of applied corrections (production ingest parser, no shared code) | 1,501 of 1,501 sampled values agree; 1 date unverifiable, 0 disagree | `spotcheck-round2.json`, seed 20261005, 200 per rule |
| Weekly gate on changed seasons | Wired; live run caught a real AFL Tables revision (fixed); §6.2 smoke run PASS, with two environment overrides, and the gate's audit/fix/commit path not yet run inside a cycle (A6) | `scripts/reconciliation_gate.py`, `scripts/weekly_refresh.sh` |
| Final acceptance (Opus Surveyor + QA + Gaffer) | **PENDING, not run** | command in A8 |

Overall data verdict: **UNKNOWN, not PASS**. Every remaining finding is a cell the source cannot settle:

| Residual (per layer) | Cells | Why it cannot be settled from AFL Tables |
|---|---:|---|
| 1932 and 1934 Brownlow votes (`R-BR-AWARD-SUM-MISMATCH`) | 40 + 1 | the per-game votes printed for those seasons do not sum to the printed award totals, so the source's game record is partial |
| 1974 hit-outs, Essendon v St Kilda **[data]** (`R-ALL-ZERO-UNPROVEN`) | 20 | both teams' hit-out columns are blank and no printed season average uniquely proves a zero |
| 2004 and 2005 time on ground (`R-PCT-NEVER-ZERO-FILLED`) | 1 + 1 | a blank percentage the source never zero-fills |

They also leave 35 dependent aggregates unresolved. No rule was invented to close them; turning them into PASS would need evidence AFL Tables does not hold.

## A2 What was corrected, and on what evidence

Baseline: the re-audit of the original inputs with the full rule set (`2026-10-03-rerun/reports/round2-audit`, snapshot `3de65975…`): legacy 68,257 fail and 82 unknown findings; snapshot 104,647 fail and 836 unknown.

| Round | Changes | After it (fail findings: legacy / snapshot) | Snapshot after |
|---|---:|---|---|
| 1 | 36,468 proposed on the unchanged baseline; superseded by round 2's complete proposal, not applied separately | | `3de65975…` |
| 2 | 1,413,507 (both layers) | 96 / 245 | `225ad40b…` |
| 3 | 2,132 | 0 / 2 | `1f0294e8…` |
| 4 | 4 | 0 / 0 | `f1abd8c2…` |

Round 2 by rule and layer (change lines, from `2026-10-03-rerun/corrections/round2/summary.json` `by_layer_rule`; 681,219 legacy + 732,288 snapshot = 1,413,507):
* **Both layers** (legacy / snapshot): row dates `R-ATTR-DATE` 674,150 / 674,232 (source fixture date replaces inferred/synthetic dates); stint totals `R-AGG-STINT` 6,972 / 6,971; jerseys `R-ATTR-JERSEY` 16 / 16; the two 1976 cells `R-BOTH-PAGES` 4 / 4; identity repairs `R-ID-REPAIR` 15 / 35; duplicates `R-ID-DUPLICATE` 2 / 6.
* **Snapshot only** (no legacy change lines under these rules in round 2): proven zeros `R-TOTAL-NONBLANK` 21,598 and `R-HEADER-ZERO` 98; Brownlow `R-BR-AWARD-SUM` 14,686 and `R-BR-SEASON-TOTAL` 848; printed-average evidence `R-AVG-COUNTED` 310 and `R-AVG-EXCLUDED` 259; notes exceptions `R-NOTES-EXCEPTION` 551; quarantined replays `R-QUARANTINE` 12; counters `R-ATTR-COUNTER` 46 and `R-ATTR-COUNTER_TOKEN` 47; profile bindings `R-ID-BIND` 12,569.
* **Legacy only**: missing appearances `R-MISSING-LOCAL` 55; missing matches `R-MATCH-MISSING` 4 (including the extra-time 1994 qualifying final, 2007 semi final and 2017 elimination final, written with the final score line); missing player `R-PLAYER-MISSING` 1.

Every cell-level change line carries its finding id, rule id, source URL, page body SHA-256 and the audited old value; `apply` aborts before writing anything if one old value no longer matches. **Exception: identity-level lines carry the rule id, source URL and old value but no finding id (`""`) and no page body SHA-256 (`null`).** Measured: 12,628 such lines in round 2 (all `R-ID-BIND` 12,569, all `R-ID-REPAIR` 15 legacy + 35 snapshot, all `R-ID-DUPLICATE` 2 + 6, and the 1 `R-PLAYER-MISSING` create) and the 2 round-4 `delete_files` lines (`R-ID-DUPLICATE`, `green_william_08092005`, `steele_roan_19092002`); rounds 3 and 4 otherwise have none missing. The deletions' evidence is a row comparison, re-measured against the pre-correction files at `HEAD`: every performance row of each deleted file equals a row of the kept canonical record bound to the same AFL Tables profile on all 30 non-date columns (numeric values compared after normalising `5` vs `5.0`): `paternoster_henry` 2 of 2 rows in `paternoster_norman_08071882`, `ross_jonathan` 20 of 20 in `ross_jonathon_03111973`, `green_william_08092005` 1 of 1 in `green_will_08092005`, `steele_roan_19092002` 2 of 2 in `steele_roan_22102001`. The date column also matches for the first three; the two Steele rows carry dates (2025-06-28, 2025-07-05) that differ from the kept rows (2025-07-04, 2025-07-11), and the kept record is the one audited against the source.

Legacy files changed: 13,184 player files (13,193 with the 2026 revision in A6) rewritten byte-preservingly (only named cells change), 4 match files, 2 files created (`dalton_jack_05042007`, a player the local layer lacked), 8 deleted (the 4 players above), and `data/awards/brownlow_season_votes.csv` added: 6,989 rows, of which 6,972 were written in round 2 and 17 in round 3 (`R-AGG-STINT`, `2026-10-05-corrected/corrections/round3/summary.json`).

**Season awards.** Pre-1984 Brownlow votes exist at source only as season totals. They now live in the snapshot table `player_season_awards` and the legacy file above (key player, season, club, award), and `analytics/players.py` career Brownlow totals add them, so a pre-1984 career total is complete. The table is a versioned schema addition (`domain/schemas.py`, `docs/data-contracts.md`) with tests. Rankings (`legacy_v1`) are unchanged: that frozen parity method does not read the new table.

## A3 Four compare runs of the corrected inputs (offline, `unshare -rn`)

| Run | Workers | Wall | Peak tree RSS |
|---|---:|---:|---:|
| cold-1 | 1 | 1,091.7 s | 1,139.0 MiB |
| cold-4 | 4 | 350.3 s | 1,777.0 MiB |
| warm-4 | 4 | 127.2 s | 1,116.3 MiB |
| changed-since | 2 | 127.5 s | 1,114.9 MiB |

All four: exit 8 (UNKNOWN), `report.json` sha256 `81ac70ba8aa25110119a91779d2c2df91088b2f0f87408db3546de8109a4509c`, `findings.jsonl` `3166d1b6ca57b1ef…` (1,478 findings). Targets: cold-4 under 10 minutes **met**; peak RSS under 2 GiB **now met** (1,777 MiB, was 2,150.6: season units then players reduced one layer at a time, integral totals stored as ints); warm under 2 minutes **missed by 7.2 s** (127.2 s; open).

## A4 Probes on the corrected inputs

| Probe (on a copy) | Recomputed / total units | Result |
|---|---|---|
| source cell: a 2006 kicks cell 5 → 9 on one profile | 42 / 260 (that player's seasons, both layers) | both layers UNKNOWN (source conflict) |
| availability: 1975 hit-outs notes grid X → blank | 2 / 260 | both UNKNOWN |
| rule: a notes-club alias added for 1990 | 2 / 260 | both UNKNOWN |
| local row: one legacy cell 10 → 17 | 1 / 260 | legacy **FAIL**, snapshot UNKNOWN |

Capture objects, manifest, plan, previous report and cache re-hashed identical after the probes (`probes/probes-summary.json`, `originals_untouched: true`).

## A5 Code changes after Part B

* Surveyor acceptance findings (`docs/reviews/afltables-reconciliation/surveyor-acceptance-review-2026-10-03-run1.md`): B1 match-page games-to-date and profile counter-sequence checks (`R-SOURCE-COUNTER`, `R-SOURCE-COUNTER-SEQUENCE`); B2 notes precedence and the 1931–34 Brownlow handling; B3 an unmapped notes club is a `SCHEMA_GAP`, not info; B4 `--out`/`--cache` containment against every input root, with cleanup; B5 `ValidationStep` counts and validator plus the committed writer `scripts/reconciliation_completion.py`; B6 an unfinished revalidation keeps a capture incomplete; F-H1 duplicate finding ids; F-M2 unit-cache salt covers the code that shapes cached results. F-H2 (row dates) is resolved by correcting the dates rather than reclassifying them: date mismatches are now `APPEARANCE_DATE_MISMATCH` failures.
* New: `avgevidence.py`, `corrections.py`, `idfix.py`, `propose.py`, `legacy_rows.py`, CLI `propose-corrections` / `apply-corrections`, plan `--scope seasons --season N` and `--capture-plan`, `scripts/reconciliation_gate.py`, `scripts/reconciliation_spotcheck.py`, `scripts/reconciliation_completion.py`. One-line fix in `storage/snapshots.py` (`apply_upserts` KeyError on a delete-only table).

## A6 Weekly gate and harness repair (CLAUDE.md section 6)

**Scope determination: in scope by both tests.** The diff touches harness entry points (`scripts/weekly_refresh.sh`, `refresh_and_rank.sh`, `scripts/update_eval_surface.sh`, `.githooks/pre-commit`, `scripts/record-sentinel-verdict.sh`) and adds a gate that changes what a cycle refuses to ship. A §6.2 smoke run is the merge condition. No cycle was active (`.claude/audit/last_refresh_status.json`: phase 4, exit 0, 2026-08-29).

The harness could not start on this machine: the hard-coded interpreter `/home/abhi/sourceCode/python/coding/.venv/bin/python` and the agent CLI `/home/abhi/.claude/local/claude` no longer exist. Repair:
* `scripts/harness_env.sh` (sourced by every harness script): `harness_python` resolves `SUPERCOACH_PYTHON`, else the repository's locked `.venv` (`uv sync --locked --group dev --group legacy --extra ml`), and fails with that command when missing; `harness_claude` resolves `CLAUDE` or `claude` on PATH. The pre-commit hook uses `COUNCIL_PYTHON`, then `SUPERCOACH_PYTHON`, then the repo `.venv`. It first lived in `scripts/lib/`, which a generic `lib/` rule in `.gitignore` ignores: a commit would have left every harness script sourcing a file that is not in the repository. A test now asserts it is not ignored.
* The eval-surface tests had been silently skipped (`skipif` on the vanished interpreter path); they run again (28 pass), and they exposed a selection defect: `update_eval_surface.sh` chose backtest vintages by file mtime, which a fresh checkout or copy scrambles. It now orders by the run timestamp in the file name, with a test where mtimes contradict the names.
* Gate phase `[1c/5]` after the match-completeness gate and before the Phase 1 push: `reconciliation_gate.py --legacy-root REPO --data-root ${RECON_DATA_ROOT:-var/finalized/data} --base ${RECON_GATE_BASE:-origin/main} --fix`. Seasons come from `git diff <base>` against the working tree. Corrections are committed through `scripts/git_commit_safe.sh`. Block (exit 1) on a confirmed discrepancy or a duplicate/unresolved player that survives correction; warn and continue on a capture outage, an incomplete source record or a missing data root.
* **Operator, when it fires mid-cycle:** the Phase 1 commit is local and unpushed. Open `gate.json` in the run directory the log names (`reports/after-fix/findings.jsonl` lists what remains). A scraper defect: fix it and re-run the cycle. A change on AFL Tables: route to Scientist. Never push the Phase 1 commit past the gate by hand.

**Live run of the gate (season 2026, 2026-10-05/06; evidence `var/reconciliations/afltables/2026-10-06-gate-live`).** A fresh, polite capture of the 2026 season (1 season page, 218 match pages, 669 profiles, one request at a time) BLOCKED: 106 legacy cell mismatches **[data]**. Every one sits on a page AFL Tables changed after the October 1–2 capture (36 of 36 match pages, 61 of 61 profiles; 98 of the 891 pages fetched both times differ). Example, 2026-05-22 **[data]**: one player's marks 8→7, frees for blank→1, contested possessions 3→4, uncontested 16→15, one-percenters 3→2; our row matched the October 1 page. The gate did what it exists for: it caught a post-season source revision.

The live run also exposed two gate defects, both fixed test-first:
1. **Season-scope identity.** Within one season, teammates share identical appearance sets, so four players whose local surname lacks a prefix the source prints (Ah Chee, De Koning ×2, van Rooyen) could not be matched by appearances and came out `IDENTITY_UNRESOLVED`, which blocks. The full audit resolves them by career appearances. A seasons audit now identifies legacy players by career too (the legacy key needs no match URL); full-audit behaviour is unchanged. Test: `test_a_seasons_audit_resolves_a_renamed_player_by_career_not_by_teammates_identical_season_games`.
2. **Revision to a proven zero.** In 4 cells the source turned a 1 into a blank its team totals prove zero; `propose` corrected only nulls to zero. A local number against the same proof (`R-TOTAL-NONBLANK`, `R-HEADER-ZERO`, `R-BR-AWARD-SUM`, `R-AVG-COUNTED`, `R-BR-SEASON-TOTAL`, expected exactly 0) is now corrected; other single-sourced mismatches stay unsupported.

With the fixes, the gate's fix path on a copy of the data: 107 legacy changes (103 `R-BOTH-PAGES`, 4 `R-TOTAL-NONBLANK`) in 62 files, no identity edits, legacy PASS on re-audit. Because the season is over, no future scrape would touch 2026 and the weekly gate would never revisit it, so the same 107 corrections were applied to this worktree's `data/player_data` (old values re-checked; 0 conflicts) and to the snapshot candidate (promoted `sha256:b9830cbf1093544e234f26159440a2cc84eb834a223da13d05e931ae8f5815c4`). A 2026 audit of both layers against the October 6 pages: **PASS / PASS, 0 findings** (report sha256 `6fecac8e…`).

Consequence for A1/A3: the four full runs describe the inputs before these 107 cells, against the October 1–2 capture. The shipped data now differs from them only in those 2026 cells. Other seasons were not re-captured, so a source revision outside 2026 after October 2 would not be seen until a full audit is re-run.

**§6.2 smoke run: PASS on the fourth run** (`var/reconciliations/afltables/2026-10-06-gate-live/20261005T215424Z.log`, wrapper `smoke-wrapper.sh`). A scratch worktree at `origin/main` (including the Codex UI commits) plus this branch's diff, this branch's corrected data, the main checkout's `.claude/audit` state, `FINALS_MODE=1`, and git commit/push shimmed out. All phases ran 0 → 4: round-settlement probe, scrape and model with DataSentinel re-verification of the regenerated backtest doc (PASS), phantom-row, match-completeness and reconciliation gates, eval surface, Hall of Fame leaders/pages/numeric gate, the round-25 recap (DataSentinel PASS on pass 1, Skeptic cleared), and the Phase 3d integration tier (passed). `origin/main` on the remote is unchanged (`39985abb1`). Deviation from `scripts/smoke_harness.sh`: audit state from the main checkout, because the worktree holds only the 11 committed audit records.

The first three runs failed. Each failure was a real defect, fixed test-first before the next run:
1. **`update_eval_surface.sh` crashed on finals rows.** It ran `int()` on round labels such as "Elimination Final". Those rows are already on `origin/main`, so every post-finals cycle would have died in Phase 1.
2. **The backtest page was built from stale vintages.** `update_team_analysis.generate_backtest_section` picked backtest summaries by file mtime, so the page's per-round table showed superseded results for R1–R8 and R18. DataSentinel rejected it. The Phase 3d checks in `tests/integration/test_published_artifacts.py` used the same mtime selection, so they now use filename timestamps too, matching the generators.
3. **The Phase 3d agent-table check rejected Codex.** It treated Codex as an unregistered agent because a README edit on `origin/main` (`1510eb967`) changed Codex's model column from "External" to "Selected per task". By the owner's decision the README text is untouched. The check now names external agents (`EXTERNAL_AGENTS = {"Codex"}`), and any other unregistered name still fails.

Run 3 also showed my corrections commit firing on any dirty data file. It now runs only when the gate itself changed the data (`RECON_BEFORE`).

Within the cycle the gate printed "no legacy season changed": the season is over and the scrape added nothing. **That outcome needed two environment overrides, which are deviations from a default cycle.** `smoke-wrapper.sh` (byte-identical to the launcher used) sets neither `RECON_DATA_ROOT` nor `RECON_GATE_BASE`, so both came from the launching shell and their values are not recorded in the run artifacts. They must have been set, because with the defaults the gate cannot print that line: (1) the scratch worktree has no `var/` (untracked, not copied by the wrapper), so the default `--data-root var/finalized/data` has no `current.json` and the gate would print `WARN no accepted snapshot ... gate skipped (fails open)` instead; (2) with the default `--base origin/main`, `seasons_since` over this branch's corrected data returns 130 seasons (1897 to 2026; re-measured in this worktree against both `origin/main` and `HEAD`), which would have started a 130-season capture. So the smoke run validated the harness ordering around phase `[1c/5]`, not the gate's audit. **The gate's audit, fix and corrections-commit path has not yet run inside a harness cycle.** It was exercised only standalone, in the live 2026 run above.

## A7 Verification of this part

Written by the committed writer `scripts/reconciliation_completion.py`. Every step is run and its status is computed, not typed in. Records: `2026-10-05-corrected/implementation-validation.json` (sha256 `dc3360bf…`) and `completion.json` (`5cdad784…`). In `completion.json`, `data.release.report_sha256` (`6b3b776e…`) is the integrity report's embedded `report_sha256` field, not the byte sha256 of `integrity-full.json` (`6f6eb3e2…`); the file is left as written. Implementation **COMPLETE**, execution COMPLETE (four runs byte-identical), data **UNKNOWN** for both layers, acceptance PENDING.

| Step | Result |
|---|---|
| `ruff check src tests scripts` | PASS |
| `ruff format --check src/supercoach_via/reconciliation scripts` (the recorded scope, `implementation-validation.json` `steps[1]`, 28 files) | PASS. Format was not checked on `tests/` or the rest of `src/`: `ruff format --check src tests scripts` re-run on 2026-10-06 reports 69 files that would be reformatted. `tests/scvia/integration/test_reconciliation_real.py` was reformatted after the record (format only); it now passes `ruff format --check` and `ruff check` |
| `mypy` | PASS, 104 source files |
| scvia fast tier | 1,487 passed, 0 failed, 0 skipped |
| legacy fast tier | 595 passed, 0 failed, 18 skipped (4 `_stat_leaders.json` not generated; 14 still keyed to the dead interpreter path, pre-existing, not changed here) |
| integration tier (Phase 3d, in the smoke run) | passed |

A first writer run during the smoke run recorded 19 failures in `test_weekly_candidate`. Those tests start `scvia_weekly.sh`, which deliberately refuses to run while a weekly cycle is running. The records above come from a re-run with nothing else running. Libraries: Python 3.12.3 and the versions in `implementation-validation.json`. No stochastic step, so no seed.

## A8 Handoff to Gaffer

Commit only through `scripts/git_commit_safe.sh`, never `--no-verify`, with `COUNCIL_PYTHON=/home/abhi/git/SuperCoach-VIA/var/worktrees/afltables-reconciliation/.venv/bin/python` (the old interpreter path no longer exists). Rebase or merge onto `origin/main` first (8 Codex UI commits, no overlapping file). Run artifacts stay under `var/reconciliations/afltables/` and are not committed.

**Allowlist** (worktree `var/worktrees/afltables-reconciliation`):
* Code: `src/supercoach_via/reconciliation/` (all), `src/supercoach_via/{cli.py,pipeline.py,ingest/http.py,integrity/sourcepages.py,integrity/derived_expect.py,storage/snapshots.py,domain/schemas.py,analytics/players.py}`
* Config and schemas: `config/reconciliation_*`, the identical copies under `src/supercoach_via/config/`, `schemas/reconciliation/*.schema.json`
* Harness: `scripts/harness_env.sh`, `scripts/weekly_refresh.sh`, `refresh_and_rank.sh`, `scripts/update_eval_surface.sh`, `scripts/record-sentinel-verdict.sh`, `.githooks/pre-commit`, `update_team_analysis.py` (backtest vintage selection), `tests/integration/test_published_artifacts.py`
* Scripts: `scripts/reconciliation_{gate,spotcheck,completion,mutation_probes}.py`, `scripts/measure_process_tree.py`
* Tests: `tests/scvia/unit/test_reconciliation_*.py`, `recon_*.py`, `test_season_awards.py`, `test_measure_process_tree.py`, edits to `test_http.py`, `test_integrity_sourcepages.py`, `test_storage_upserts.py`, `test_analytics_players.py`; `tests/scvia/integration/test_reconciliation_real.py`; `tests/scvia/fixtures/reconciliation/*`; `tests/unit/test_harness_{wiring,interpreter_paths}.py`, `tests/unit/test_eval_surface_{readme,backtest_doc,banner}.py`, `tests/unit/test_top30_deviation_vintage.py`, `tests/scvia/unit/test_reconciliation_completion.py`
* Docs: `docs/afltables-reconciliation.md`, `docs/reviews/AFLTABLES_RECONCILIATION_RUN.md`, `docs/data-contracts.md`, `docs/rewrite/afltables-reconciliation/DESIGN.md`, `docs/reviews/afltables-reconciliation/surveyor-acceptance-review-2026-10-03-run1.md`, `.claude/surveys/2026-10-03-afltables-reconciliation-acceptance-survey-run1.md`, the Surveyor memory edits
* Data (owner-authorised direct correction, a separate commit): `data/player_data/` (13,193 modified, 2 added, 8 deleted; includes the 107 cells of 2026 from A6), `data/matches/` (4), `data/awards/brownlow_season_votes.csv`
* Main checkout (separate commit): `.claude/agent-memory/Scientist/` (MEMORY.md and the reconciliation notes)

Not in the allowlist: the snapshot candidate `var/candidates/20261005-afltables-corrected` (promote it with the release flow; the web UI pins `3de65975…` and report `58eff517…` in `web/src/lib/provisional.ts`, so update that pin only together with a new release).

**Exact final-acceptance command** (not yet run):

```bash
cd /home/abhi/git/SuperCoach-VIA/var/worktrees/afltables-reconciliation && claude --agent Gaffer --model claude-opus-5-5 --effort high \
  "Final acceptance for the AFL Tables reconciliation corrections, gate and harness repair. Commission Surveyor (Opus 5.5) and QA to inspect the FINAL code and artifacts, not this summary: docs/reviews/AFLTABLES_RECONCILIATION_RUN.md Part A, var/reconciliations/afltables/2026-10-05-corrected/{completion.json,implementation-validation.json,reports/final-cold-4/report.json,probes/probes-summary.json,spotcheck-round2.json,integrity-asof1005/integrity-full.json}, the corrections under 2026-10-03-rerun/corrections/round2 and 2026-10-05-corrected/corrections/round{3,4}, the live-gate evidence in 2026-10-06-gate-live, and the smoke-run log named in section A6. Confirm the data verdict is UNKNOWN (not PASS) with 63 source-unresolvable cells per layer, that the warm-time miss is stated, that the harness change passed its smoke run, and that no cycle is active. Record accept or reject with evidence. If accepted, rebase onto origin/main, commit the allowlist (code, data, memory as separate commits) through scripts/git_commit_safe.sh with COUNCIL_PYTHON set, then deliver to main per the owner's standing request."
```

## A9 What was not done, and what could change the conclusion

* **Not PASS.** 63 cells per layer stay UNKNOWN because AFL Tables' own record is incomplete there. A source other than AFL Tables would be needed.
* **Historical truth is not verified**: this is agreement with AFL Tables, which calls its figures unofficial.
* **Warm compare** misses its 2-minute target by 7.2 s.
* **Gate cost:** a changed season costs about 1,000 polite requests (roughly 35 to 40 minutes) per weekly cycle. A season-final cycle that rescrapes many seasons would take hours; the gate fails open on a capture outage, so a slow capture delays rather than breaks a cycle.
* **Root fix not made:** the legacy import still guesses zeros from team totals; the gate catches the result for every changed season, but `domain.blanks` itself was not changed.
* **Model feature-input vintage break.** The legacy `R-ATTR-DATE` corrections change the `date` column of 175,631 legacy rows dated 2005 or later in round 2 (plus 376 in round 3; re-measured from the `changes.jsonl` of each round, by year of the corrected date). `supercoach/prediction.py` derives `days_since_last_game` from `date` (lines 538, 583; a fitted feature) and `player_age_at_match` from `date` (line 517; used only when the opt-in `include_age_experience` is on, which no production caller sets). The next retrain or backtest therefore runs on different feature inputs and is not comparable with earlier backtest vintages.
* **A re-scrape can undo a correction.** `scrapers/player_scraper.py` `dedup_player_performance` keeps the last copy (`keep='last'`), so a re-scraped row replaces a corrected one with whatever AFL Tables then prints. The gate re-corrects only when it runs on that season, and it fails open (exit 0) on a capture outage or a missing data root.
* **Queued for the next cycle, not applied (CLAUDE.md §6.1):** (H1) when the gate skips or warns it exits 0, and `scripts/weekly_refresh.sh` (line 177) then logs "Reconciliation gate passed", which hides a skipped gate; (M2) the gate's `--fix` can make row and identity edits after the phantom-row and match-completeness gates have already passed (lines 127 to 148), and neither is re-run on the corrected data.
* **Residual risk:** corrections rest on one 17-hour capture of AFL Tables (2026-10-01/02); a later source edit to an older season is caught only when that season next changes locally or a full audit is re-run.

---

# Part B: first run (audit only, 2026-10-01/02; verdicts superseded by Part A)

## 1 Verdicts

| Claim | Status | Evidence |
|---|---|---|
| Design accepted | COMPLETE | `docs/reviews/AFLTABLES_RECONCILIATION_DESIGN_REVIEW.md`; Opus approval after three Surveyor reviews; commit `09ac3a750` |
| Implementation | COMPLETE | `implementation-validation.json`; section 8 |
| Execution (full capture, four compare runs, control, probes) | COMPLETE, two performance targets missed | `completion.json`; sections 4 to 7 |
| **Primary snapshot** `sha256:3de6597513b5c6eaca1b102d304a2cf128d8c08f229ee1de384c1d40fe746db0` | **FAIL** (confirmed mismatches; also UNKNOWN on unresolved evidence) | `report.json` layers.snapshot |
| **Raw legacy CSV layer** (`data/player_data`, `data/matches`, content sha256 `ac92ea35…b471`) | **FAIL** (same) | `report.json` layers.legacy_csv |
| **Release** `20260929T213900Z-c8938f4ddb83` (`scvia check-integrity --scope full`) | PASS, 26 of 26 checks | separate report, sha256 `46d77e23…ada089`. It makes no claim about AFL Tables agreement, and this audit makes no claim about the release |
| Overall data verdict | **FAIL** | `completion.json` overall_data_verdict |
| Final acceptance (Opus Surveyor + QA + Gaffer) | **PENDING, not run** | command in section 11 |

PASS in this audit would mean agreement with captured AFL Tables evidence inside the declared availability scope. AFL Tables states its figures are unofficial; the capture is `reference_mode=observed_current` over a 17-hour window, not a snapshot of the 2026-09-30 boundary. No correction was applied to any input. This is an audit; the finding list is a worklist, not a patch.

## 2 Scope and denominators

Men's senior VFL/AFL premiership matches from 1897 to 2026-09-30 inclusive (finals, drawn finals and replays included). Excluded and stated in every report: preseason, reserves, representative football, AFLW, coaches, umpires, mutable height/weight, fantasy scores, predictions and locally invented proxy measures.

| | Source | Snapshot layer | Legacy CSV layer |
|---|---|---|---|
| In-scope matches | 17,056 (all pages usable) | 17,056 paired | 17,052 paired; 3 local-only, 4 missing locally |
| Season pages | 130 of 130 usable | | |
| Profiles in directory, required, usable | 13,364 of 13,364 | 13,348 mapped | 13,345 mapped |
| Local players | | 13,368 (4 local-only) | 13,367 (4 local-only) |
| Appearances expected (source) | 695,499 | 694,561 matched; 244 missing locally (12 only as quarantined rows); 694 unresolved | 694,380 matched; 215 missing; 904 unresolved |
| Cells expected (appearances × 23 statistics) | 15,996,477 | see below | see below |

Cell accounting (snapshot): 7,777,444 equal (4,882,597 equal values, 2,894,847 equal zeros), 36,577 mismatched (36,573 are a null where the source proves a recorded zero), 29,740 not applicable (finals Brownlow votes etc.), 8,110,781 source-unavailable (not recorded in that era or documented in the notes page), 11,092 unresolved, 435 local numbers where the source records none, 5,612 inside a missing appearance, 15,962 inside an unresolved appearance. Legacy: 7,808,020 equal, 4 mismatched, 29,699 not applicable, 8,112,663 source-unavailable, 11,085 unresolved, 4,945 and 20,792 inside missing and unresolved appearances. All accounting identities hold (`execution.json` `accounting_identities_hold: true`, and the report states each identity).

Source cell states, identical for both layers: 4,888,370 recorded value, 2,934,300 recorded zero, 8,123,619 not recorded, 29,811 not applicable, 9,269 credited-but-did-not-take-the-field, 11,108 unresolved blank. Aggregates judged (snapshot): career 307,004 (102,868 equal, 38 mismatched, 2,287 local-missing-summary, 2,313 unresolved), season 1,351,825 (580,766 equal, 39 mismatched), stint 1,357,529 (581,718 equal, 39 mismatched). Source-unavailable aggregates are counted separately (career 199,492, season 759,281). Verified numeric fraction 0.486; available-statistic fraction 0.991 (snapshot).

**Latest completed final before the boundary:** the 2026 Grand Final, Fremantle v Brisbane Lions on 2026-09-26 **[data]**. All 46 participating profiles were compared in the snapshot layer (46 matched); the legacy CSV layer **does not contain that final** (46 missing, 215 missing appearance findings). That is a genuine legacy-layer gap, found without any local seed.

## 3 Findings (primary run; full stream `reports/cold-4/findings.jsonl`, 195,668 findings, sha256 `c538a53d01d182e2c063bf1372b67e0ee2c37fe79973b363f85554dc67aefaa4`)

Every finding carries a stable id, rule id, source URL, body hash and table/row/cell locator, and the local fragment or file/row.

| Category | Snapshot | Legacy CSV | What it is |
|---|---:|---:|---|
| CELL_LOCAL_NULL (fail) | 36,573 | 0 | Source proves a recorded zero; the snapshot stores null. 14,756 are Brownlow votes. By decade: 14,400 in the 1930s (Brownlow cells), 5,774 in the 1960s, 9,585 in the 1970s, 2,173 in the 1980s, 3,414 in the 1990s, 1,129 in the 2000s |
| LOCAL_MISSING_SUMMARY_VALUE (fail) | 9,258 | 9,260 | Pre-1984 Brownlow season totals the source publishes and the local layer does not hold (per-game cells are blank at source, so the season total is the only reference) |
| APPEARANCE_MISSING_LOCAL (fail) | 232 | 215 | Snapshot: e.g. a 1971 Richmond v Carlton row; legacy: the 2026 Grand Final |
| APPEARANCE_QUARANTINED (fail) | 12 | 0 | Source appearance exists only as a quarantined local row (drawn-final replays, e.g. 1928 and 2010) |
| APPEARANCE_ATTR_MISMATCH (fail) | 109 | 16 | Counter mismatches (snapshot, 2026 Grand Final) and jersey numbers (legacy, e.g. 1977) |
| AGGREGATE_MISMATCH (fail) | 77 | 1,336 | Legacy stint/career totals off by small amounts, e.g. 1976 kicks and disposals, 2026 time-on-ground |
| CELL_MISMATCH (fail) | 4 | 4 | Two players in one 1976 match (kicks and disposals each), one higher locally than at source; identical in both layers **[data]** |
| PLAYER_MISSING_LOCAL / MATCH_MISSING_LOCAL / MATCH_EXTRA_LOCAL (fail) | 16 / 0 / 0 | 12 / 4 / 3 | A profile with no local player; legacy lacks one 1994 and one 2007 final and holds two others the source does not show |
| CELL_UNRESOLVED (unknown) | 11,047 | 11,040 | Blank cells whose zero-ness cannot be proven: hit-outs 1966 to 1979 (5,746), bounces 1990s to 2020s (5,054, most in 2010 to 2025), goal assists 2020 (133) |
| LOCAL_UNSUPPORTED_NUMERIC (unknown) | 395 | 0 | The local layer holds a number where the source records none |
| IDENTITY_UNRESOLVED / IDENTITY_CONFLICT (unknown) | 14 / 2 | 16 / 3 | No unique mapping without guessing |
| Reported only (info) | 57,847 | 57,848 | Row dates (inferred or synthetic in the legacy layer, not verified), representation, identity variance, plus 324 source-internal inconsistencies |

Interpretation, kept separate from the evidence: the snapshot FAIL is dominated by one representation question (null versus recorded zero) and one coverage question (pre-1984 Brownlow). The confirmed value-level disagreements are few (4 cells, 109 attribute and 77 aggregate findings in the snapshot). Whether a null that stands for a proven zero should be a FAIL or a representation note is a stakeholder decision; the design classifies it as a FAIL, and I did not reclassify it to obtain a different verdict. The control (section 7) shows how much this one choice moves the counts.

## 4 Capture (network)

* Plan: `plan_id 0b931df12131e849f48d70c7b9e93557dcfbd52a0c7af7880cd8ff45b89a2466`, `capture_identity 9f2cbee4c9515a4bca4444adb930d924e0b66ac3d2fcc21d99c51fb68b0fbc28`. The first plan (`fd6bc85d…`, archived in `plans/`) was re-issued once, after the Brownlow sum rule was added; `capture_identity` is unchanged, so the capture stayed valid (deviation 1).
* One request at a time, at least 2 s apart, no conditional requests, three attempts. 30,736 observations: 30,734 HTTP 200, one genuine 404 (robots.txt, recorded as absent), one non-HTTP observation. No 403, 429 or challenge page; the supervisor needed one pass (`logs/supervisor.log`).
* Terminal records: 30,734 done, 0 failed, 0 blocked, 0 missing, 0 pending (26 letter pages, 1 notes page, 1 stats index, 130 season pages, 17,056 match pages, 13,364 profiles, plus robots). End-of-acquisition revalidation of letters and seasons found no change (`revalidation_changed: []`).
* Acquisition window 2026-10-01T11:46:00Z to 2026-10-02T04:44:55Z (about 17.0 h). Measured separately from compare. 174 objects were reused (re-hashed) from the pilot capture via `--seed-from`; 30,562 requests were made by the final process.
* The capture ran from an immutable code snapshot (`code-snapshots/capture-20261001T114542Z`); the six capture-relevant files have the same SHA-256 as the final worktree (`reports/*/report.json` `code.capture_files`).
* Capture manifest sha256 `83ff79f64d1354b0fb09ed100a116f54fee2d6eeeb1cbb436e959c1db11adbdd`.

## 5 Compare: four runs on one frozen capture, offline

Every run was executed inside `unshare -rn` (empty network namespace: no sockets) and measured with `scripts/measure_process_tree.py` (wall time and peak process-tree RSS sampled across all children). Host: Linux 7.0.0-34 x86_64, 12 CPUs, 62 GiB RAM, Python 3.12.3, load average about 2.6 during the runs (not an idle reference machine).

| Run | Workers | Cache | Exit | Wall | Peak tree RSS | Notes |
|---|---:|---|---:|---:|---:|---|
| cold-1 | 1 | empty | 4 | 1,008.0 s | 1,601.4 MiB | 30,577 parse misses |
| cold-4 | 4 | empty | 4 | 372.0 s | 2,150.6 MiB (6 processes) | 30,577 misses |
| warm-4 | 4 | cold-4 units | 4 | 122.5 s | 1,630.7 MiB | 260 of 260 units from cache |
| changed-since | 2 | warm + `--previous cold-4` | 4 | 122.6 s | 1,637.7 MiB | 260 of 260 units unchanged, 0 recomputed |

**Byte identity:** `report.json`, `findings.jsonl`, `players.csv`, `coverage.csv` and `summary.md` have one distinct SHA-256 each across all four runs.
* `report.json`: `58eff517a29d30b932087f919a31d36567e8be330301ecff53feb06494d8db28`
* `findings.jsonl`: `c538a53d01d182e2c063bf1372b67e0ee2c37fe79973b363f85554dc67aefaa4` (195,668 records)

Disk: capture 1.8 GB, unit cache 1.8 GB, each report directory about 199 MB.

**Targets from the design (assess, do not fabricate), measured:**
* Cold four-worker compare at most 10 minutes: **met** (6.2 minutes).
* Peak process-tree RSS at most 2 GiB: **MISSED** by cold-4 (2,150.6 MiB is 2.10 GiB, about 5% over); met by cold-1, warm and changed-since. Open; not tuned further in this task.
* Warm at most 2 minutes: **MISSED** by 2.5 s (122.5 s). Open.
* Phase split (cold-4, s): parse 144, season units 157, reduce 42, identity 18, inventory and verify 4.

## 6 Control and release

* **Control** (the retained older snapshot `sha256:aa836549e10e96a9039cc4642bbe563b94247e0bfaf95e745a0971cc0bb2b98f`, `var/finalized/data`, plan `e856cc35…`, same frozen capture, snapshot layer only): overall FAIL, 3,010,288 findings, 338 s, 2,283 MiB. It is not substituted for the primary. It differs where it should: 2,931,402 CELL_LOCAL_NULL (that vintage stored far more nulls), 24 MATCH_LINK_MISMATCH, 8 aggregate mismatches against 77 in the primary. Report hash `reports/control/report.json` in `2026-10-01-control` (findings stream gzip-compressed to save disk; the recorded manifest hash is of the uncompressed file).
* **Release:** `scvia check-integrity --scope full --as-of 2026-09-30T00:00:00Z` on the primary snapshot and release `20260929T213900Z-c8938f4ddb83`, with the B1 evidence and curated content manifest: outcome PASS, complete, 26 PASS, 0 FAIL, 0 UNKNOWN, 0 blocking, 5 warnings, 1 info; report sha256 `46d77e23a8fdb7d43f321f5065c5f435e9f4ad9ddd75fc191898622001ada089`, 75 s, 1.69 GB. Neither verdict proves the other's claim, and no statement is made about a deployed host or `var/final-site`.

## 7 Mutation probes and input immutability (D12)

`scripts/reconciliation_mutation_probes.py` makes hard-linked copies, replaces the mutated file with a NEW file, re-issues the plan, and re-runs compare with `--previous` and the warm cache.

| Probe (on a copy) | Units recomputed | Units reused | What surfaced |
|---|---:|---:|---|
| Source cell: one player's 2006 game 1 kicks 5 changed to 9, disposals unchanged at 11 **[data]** | 42 (that player's seasons 2006 to 2026, both layers) | 218 | SOURCE_CONFLICT plus CELL_UNRESOLVED for the cell (profile and match page now disagree): blocks PASS |
| Availability: notes grid, 1975 hit-outs available changed to blank | 2 (1975, both layers) | 258 | 38 unresolved hit-out cells (19 per layer) resolve to not-recorded and drop out; 58 unsupported-numeric findings are re-issued under the new rule; 1 derived-average inconsistency appears |
| Rule: Brownlow sum rule `first_season` 1984 changed to 1990 | 12 (1984 to 1989, both layers) | 248 | 16,026 new unresolved Brownlow cells |
| Local row: one legacy CSV cell, kicks 10 changed to 17 | 1 (`legacy_csv:2007`) | 259 | 1 CELL_MISMATCH and 2 AGGREGATE_MISMATCH |

Every probe's report hash differs from the unmutated report. The original capture objects, manifest, plan, cold-4 report and cache were re-hashed after the probes and are identical.

**This probe found a real defect, which was fixed before the numbers above were produced.** The first probe set showed the rule probe recomputing 0 of 260 units and finding nothing: the per-season unit digest ignored the Brownlow sum rule, so editing it would have silently reused stale cached results. The digest now includes the applicable sum rules and no-award evidence; `test_changing_a_sum_rule_invalidates_the_units_it_governs` (two cases) failed before the fix and passes now. All four runs, the control and the probes were then repeated from empty caches on the fixed code; the earlier reports are archived under `explore/superseded-2` and are not used as evidence.

**Input immutability:** before/after inventories of the candidate data, the Opus review release, `var/finalized/data` and `var/finalized/releases`, and the legacy `data/player_data` and `data/matches` list 393,772 files each (size, mtime_ns, SHA-256): 0 added, 0 removed, 0 changed (`inputs-before.json`, `inputs-after.json`). The main checkout's `git status` shows only the two Scientist memory files. No production schedule, hook, gate, data pointer or release was touched; no request was sent except to AFL Tables during the capture, and none that writes.

## 8 Implementation validation (D04 to D08)

Libraries: Python 3.12.3, pydantic 2.13.5, pytest 9.1.1, ruff 0.16.8, mypy 2.3.1, typer 0.27.2, pandas 2.3.3, pyarrow 25.0.1, lxml 6.1.3. No stochastic step exists, so there is no random seed.

| Check | Result |
|---|---|
| `ruff check src tests scripts`; `ruff format --check` on every reconciliation file | clean |
| `mypy` | clean, 98 source files |
| `tests/scvia -m "not integration"` | 1,352 passed, 1 failed; the failure is environmental and pre-existing (below) |
| legacy `tests/unit -m "not integration"` | 555 passed, 45 skipped |
| `tests/scvia/integration/test_reconciliation_real.py` with `SCVIA_RECON_RUN` set | 4 passed |
| Fast-tier cost | A/B against a detached worktree of the base commit, same machine, back to back: base 159.4 s (1,024 tests), branch 197.4 s (1,352 tests), **+38 s, +24%**. The earlier 59.6 s baseline was taken on a quieter machine and is not comparable; the reconciliation tests alone take 49.5 s |

**Unrelated failure:** `tests/scvia/unit/test_builder.py::test_browser_readers_accept_the_python_release` fails in this git worktree because `web/node_modules` does not exist there; it passes in the main checkout. Left alone, per scope.

## 9 D01 to D18

| | Status | Evidence |
|---|---|---|
| D01 Opus Surveyor design review | Met | design review record |
| D02 No blocking architecture question | Met | same |
| D03 T01 to T35 each have an owner and test | Met | `tests/scvia/unit/test_reconciliation_*.py`, `recon_e2e.py` worlds |
| D04 Commands, schemas, docs, configuration | Met | `scvia reconcile-afltables plan, capture, compare`; `schemas/reconciliation/`; `docs/afltables-reconciliation.md`; config files installed under `src/supercoach_via/config/` |
| D05 Deterministic logic only | Met | rules are executable Python plus versioned TOML; no model call in the program |
| D06 Tests, ruff, mypy, fast tier | Met, with one explicit unrelated failure | section 8 |
| D07 Resume, lock, containment, stream completeness, cache invalidation, corruptions | Met | capture, e2e and cache unit tests (interrupt, corrupt entry, hard-link output alias, failing write); probes above |
| D08 Before/after inventories | Met | section 7 |
| D09 Terminal acquisition record for every directory and in-scope page | Met | section 4 |
| D10 Every player, appearance and requested field accounted for | Met | section 2; identities hold |
| D11 Per-game cells and season/stint/career aggregates and denominators | Met | section 2; averages checked against recorded-games denominators |
| D12 Four runs byte-identical; mutations caught; caches invalidated correctly | Met | sections 5 and 7 |
| D13 Offline rerun preserves verdict; times, RSS and disk measured; missed targets stated | Met; targets for RSS and warm time **missed** | section 5 |
| D14 Latest completed final present and checked | Met | section 2: present in the snapshot, absent from the legacy CSV layer |
| D15 Opus review, QA, Gaffer acceptance | **Not met: pending** | section 11 |
| D16 Primary snapshot full-audit PASS | **Not met: FAIL** | section 3 |
| D17 Release checker PASS for that snapshot/release | Met (checker PASS); the app is **not** confirmed because D16 fails | section 6 |
| D18 Legacy layer PASS | **Not met: FAIL** | section 3 |

## 10 Deviations from the approved design (for Gaffer and Surveyor)

1. `plan_id` (covers everything) is separated from `capture_identity` (scope, source policy and capture code only) so comparison rules can be re-issued without invalidating a 17-hour capture. Resume still refuses when a capture-relevant file changed.
2. `compare` also requires `capture/receipt.json` to say `complete`. The manifest builder counts only first-pass tasks as pending, so a capture interrupted during final revalidation can show `capture_complete: true`; this defect is documented, and the capture code stayed frozen for auditability.
3. Match pages print no career games-to-date, so the profile-counter-versus-match-page check cannot exist; counters are validated against the profile's own footers.
4. Printed averages are compared on a closed rounding interval using recorded-games denominators; they are derived figures and can only produce info findings, never a layer verdict.
5. Extra cell-accounting buckets (`cell_in_missing_appearance`, `cell_in_unresolved_appearance`) so the identities close exactly.
6. Row-date gating: dates the legacy layer itself declares inferred or synthetic produce grouped ROW_DATE_UNVERIFIED info findings (about 57,400 per layer) instead of failures.
7. `capture.py` was written in one pass before its full test file existed (I did not follow strict test-first for that module); its tests were written afterwards against design cases T16, T17, T18 and T35.
8. A versioned evidence rule, `R-BR-SIX-PER-MATCH`, was added: from 1984 every home-and-away match's two team Brownlow totals sum to 6 (7,620 of 7,620 matches in the frozen capture, none otherwise). It resolves 83,565 blank Brownlow cells as proven zeros. Evidence is cited in the rules file. It cannot waive a mismatch.

## 11 Scope determination (CLAUDE.md section 6)

* The new command is an **opt-in operator audit**. It is not invoked by `weekly_refresh.sh`, `refresh_and_rank.sh`, any hook or any gate, and it adds nothing to `tests/integration/` that a cycle would run: the new integration test asserts nothing about shipped artifacts and skips its run-directory checks unless `SCVIA_RECON_RUN` is set.
* The only edits to existing code are additive (plus registering the new Typer sub-app in `src/supercoach_via/cli.py`): `retry_after_s` appended to `ingest/http.py` (default behaviour unchanged, tested) and a `rowspan` attribute defaulting to 1 on `integrity/sourcepages.Cell`. Both fail test 1 (not a harness entry point) and test 2 (no change to phase order, staging, a gate, or artifact selection). No harness smoke run is required. If a cycle is active, nothing here changes it.

## 12 Handoff to Gaffer

Do not edit the tree during the commit. Commit only through `scripts/git_commit_safe.sh`, never `--no-verify`, with `COUNCIL_PYTHON=/home/abhi/sourceCode/python/coding/.venv/bin/python`. Large run artifacts stay local under `var/reconciliations/afltables/` (they are not pipeline outputs and must not be committed).

**Allowlist** (worktree `var/worktrees/afltables-reconciliation`, branch `work/afltables-reconciliation`; exact list in `var/reconciliations/afltables/2026-10-01-full/gaffer-allowlist.txt`):
* `src/supercoach_via/reconciliation/` (all modules), `src/supercoach_via/cli.py`, `src/supercoach_via/ingest/http.py`, `src/supercoach_via/integrity/sourcepages.py`
* `config/reconciliation_*` and the identical copies under `src/supercoach_via/config/`
* `schemas/reconciliation/*.schema.json`
* `docs/afltables-reconciliation.md`, `docs/reviews/AFLTABLES_RECONCILIATION_RUN.md`
* `scripts/measure_process_tree.py`, `scripts/reconciliation_mutation_probes.py`
* `tests/scvia/unit/test_reconciliation_*.py`, `recon_*.py` helpers, `test_measure_process_tree.py`, additions to `test_http.py` and `test_integrity_sourcepages.py`, `tests/scvia/integration/test_reconciliation_real.py`, `tests/scvia/fixtures/reconciliation/*`
* Main checkout (separate commit): `.claude/agent-memory/Scientist/MEMORY.md`, `afltables_reconciliation_structure.md`, `afltables_reconciliation_run_lessons.md`

Some files are already staged in the worktree index (earlier `git add`; no commit was made). Verify with `git status` before committing.

**Exact final-acceptance command** (not yet run):

```bash
cd /home/abhi/git/SuperCoach-VIA/var/worktrees/afltables-reconciliation && claude --agent Gaffer --model claude-opus-5-5 --effort high \
  "Final acceptance for the AFL Tables reconciliation. Commission Surveyor (Opus 5.5 via the Agent tool's opus override) and QA to inspect the FINAL code and the completed reports, not this summary: docs/reviews/AFLTABLES_RECONCILIATION_RUN.md, var/reconciliations/afltables/2026-10-01-full/{completion.json,implementation-validation.json,reports/cold-4/report.json,probes/probes-summary.json} and the control in var/reconciliations/afltables/2026-10-01-control. Check D01-D18, the section 10 deviations, the missed RSS and warm-time targets, and that the data verdict (snapshot FAIL, raw CSV FAIL, release checker PASS) is stated without softening. Record accept or reject with their evidence; do not mark any unrun step passed. If accepted, commit the allowlist through scripts/git_commit_safe.sh with COUNCIL_PYTHON set, then deliver to main per the owner's standing request."
```

Reproduce any number: `RUN=var/reconciliations/afltables/2026-10-01-full`; `scvia reconcile-afltables compare --plan $RUN/plan.json --capture-manifest $RUN/capture/manifest.json --out <new dir> --workers 4 --cache <empty dir>` (offline, about 6 minutes) must reproduce `report.json` sha256 `58eff517…8db28`.

## 13 Exact commands

```bash
# plan (offline); candidate = primary snapshot; legacy = this repository's CSVs
scvia reconcile-afltables plan --data-root var/reviews/opus55/20260929T203757Z-followup/candidate-data \
  --snapshot sha256:3de6597513b5c6eaca1b102d304a2cf128d8c08f229ee1de384c1d40fe746db0 --legacy-root /home/abhi/git/SuperCoach-VIA \
  --through-date 2026-09-30 --scope all --run-dir $RUN
# capture (network), from an immutable code snapshot, resumable
PYTHONPATH=var/reconciliations/afltables/code-snapshots/capture-20261001T114542Z/src scvia reconcile-afltables capture \
  --plan $RUN/plan.json --allow-network --resume --seed-from var/reconciliations/afltables/2026-10-01-pilotA/capture
# four compare runs, each under: unshare -rn python scripts/measure_process_tree.py $RUN/measure/NAME.json -- scvia reconcile-afltables compare ...
compare --plan $RUN/plan.json --capture-manifest $RUN/capture/manifest.json --out $RUN/reports/cold-1 --workers 1 --cache $RUN/cache-1
compare ... --out $RUN/reports/cold-4  --workers 4 --cache $RUN/cache-4
compare ... --out $RUN/reports/warm-4  --workers 4 --cache $RUN/cache-4
compare ... --out $RUN/reports/changed --workers 2 --cache $RUN/cache-4 --previous $RUN/reports/cold-4/report.json
# control: same capture, retained older snapshot (plan e856cc35…)
plan --data-root var/finalized/data --snapshot sha256:aa836549e10e96a9039cc4642bbe563b94247e0bfaf95e745a0971cc0bb2b98f \
  --through-date 2026-09-30 --scope all --run-dir var/reconciliations/afltables/2026-10-01-control
compare --plan <control>/plan.json --capture-manifest $RUN/capture/manifest.json --out <control>/reports/control --workers 4 --cache <control>/cache-4
# release checker (separate)
scvia check-integrity --data-root <candidate-data> --snapshot sha256:3de6… --release-dir <release 20260929T213900Z-c8938f4ddb83> \
  --scope full --as-of 2026-09-30T00:00:00Z --evidence docs/rewrite/evidence/b1/raw --content-root . --content-manifest config/public_content.toml
# probes and inventories
unshare -rn python scripts/reconciliation_mutation_probes.py $RUN $RUN/probes
python var/reconciliations/afltables/tools/tree_inventory.py $RUN/inputs-{before,after}.json <six roots>
```

Exit codes: 0 PASS, 2 invalid, 4 FAIL, 5 locked, 8 UNKNOWN, 9 software failure. All four primary runs and the control exited 4.

## 14 What was not done, and what could change the conclusion

* **Not verified:** historical truth. AFL Tables may itself be wrong; 324 internal source inconsistencies are reported as info. Not captured: Brownlow no-award season evidence (the Brownlow career-average check therefore reports `unresolved_no_award_evidence` for 4,308 career-average checks), and the 1931 to 1934 Brownlow one-sided rule was not adopted without a located evidence page (local nulls there are reported as CELL_LOCAL_NULL).
* **UNKNOWN residue:** about 11,000 all-zero team columns (hit-outs 1966 to 1979, bounces 1990s to 2020s) cannot be proven zero or missing from the page alone; no rule was adopted without evidence. A rule backed by captured evidence could resolve them in either direction.
* **Reference mode:** pages were observed over 17 hours; a page edited during acquisition is caught only by the end-of-acquisition revalidation of letters and seasons, not of every match page.
* **Decision for the owner (not made here):** whether a snapshot null that stands for a source-proven zero is a FAIL (design default, 36,573 findings) or a representation note, and how to treat pre-1984 Brownlow season totals the local layers never held (9,258 findings). Correction proposals are a separate, authorized task.
* **Residual risk:** the data FAIL is real and reproducible, but its size is sensitive to those two classification choices, not to the comparison machinery, whose determinism and invalidation were tested directly.
