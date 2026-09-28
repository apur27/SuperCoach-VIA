# Cursor continuation progress (2026-09-27)

## Current local status (2026-09-28)

Parent update: independent review returned **PASS for local merge** after the final scratch smoke and four-core timing run. See [final verification](finalization-20260928/README.md). The merge itself follows this review; historical sections below retain their original state.

The recovery-review corrections are in this worktree. Nothing was committed, pushed, merged, deployed, or scheduled. Two genuine shadow cycles and production activation are still external decisions. This is not a merge approval.

- Snapshot `sha256:aa836549e10e96a9039cc4642bbe563b94247e0bfaf95e745a0971cc0bb2b98f` under `var/gf-season-20260928`: 218 complete 2026 matches, last date 2026-09-26, `schedule_complete` null, `source_status` null. `data/matches/matches_2026.csv` is still the pre-final file. The recovered B1 pointer is still `sha256:55e295f1…`.
- A passing refresh that writes source observations and no match rows now stamps `fixture_checked_at` for the seasons whose fixture was checked. A partial refresh still does not move the pointer.
- Match pages use the recorded provenance. A source fetch is labeled `AFL Tables source page` and keeps the captured URL. A legacy import keeps `Legacy match/player CSV import`.
- Sealed site after that label: release `20260928T111014Z-ca96603163ad`, seal `bd9a655b06795c2d83ca77a48adb9df85fb2774463fa390f814dac2ff72474fa`, site 276,821,039 bytes. Mean 2021–2025 remains 4,007,870 bytes. Projection 294,087,529 bytes (280.46 MiB). Pack verify matched the built site. The parent-verified release `20260928T104520Z-2561b2316c8a` (276,820,994 bytes, projection 280.46368 MiB) is unchanged. `gf-season-growth.json` still names the earlier release `20260928T102101Z-d2e56565da24`; the corrected attribution is `gf-label-growth.json`.
- Final scratch smoke exit 0, wall 620s, inventory differences 0, `318c37e8b2a80d6f518f89cde1427923dc4d7a1a389224d58f81400442c8ea5a`, 854 files, 18,202,333 bytes. It replays the captured pre-final corpus (snapshot `sha256:55e295f1…`). It is not a shadow cycle. Log: `var/agent-runs/cursor-grok47-recovery/scratch-smoke-final.log`.
- Hermetic after these fixes, same 695 tests: 34.15s on `taskset -c 0-3` (two physical cores, load average about 5.5; `hermetic-post-review.txt`) and 29.33s on `taskset -c 0,2,4,6` (four physical cores, wall 29.56s; `hermetic-cores-0-2-4-6.txt`). Both used four workers and native-thread caps of one. Source-check and real-import integration: 9 passed in 49.92s. Web, wheel, and mypy checks from the earlier recovery were not rerun.
- `docs/rewrite/evidence/c8-followup-2026-09-27/` is not in this tree. The historical inventory `fb871eda…` (857 files) was not forced.

The parent notes and the 2026-09-27 sections below are the record of that work. They are not the current grand-final state.

## Parent notes during recovery (2026-09-28; resolved in the section above)

- Timing note at 11:15 UTC: `lscpu -e=CPU,CORE,SOCKET,ONLINE,MAXMHZ` confirms CPU IDs 0,1 share physical core 0 and IDs 2,3 share core 1. Thus `taskset -c 0-3` uses two physical cores/four SMT threads. After the current final smoke exits, measure the unchanged full hermetic suite once on `taskset -c 0,2,4,6` (four distinct physical cores), still four workers and native-thread caps of one. Preserve/report the 34.15s result and both CPU placements; do not pretend it passed on the original affinity. CPU topology is saved in the parent's recovery directory as `cpu-topology.json`. No code or test reduction is needed for this measurement.
- The parent has independently checked all grand-final player rows against the captured source. It will repeat the check on the corrected snapshot and the sealed site's match page and player logs.
- Update at 10:51 UTC: the parent also passed the public/site verification and a real browser check on release `20260928T104520Z-2561b2316c8a`. Reports are `parent-grand-final-public-site.json`, `grand-final-browser.json`, and `parent-final-season-growth.json` under `/home/abhi/git/SuperCoach-VIA/var/recovery/cursor-grok47-20260928/`. The exact final site is 276,820,994 bytes; the growth projection is 280.46 MiB. The writer's `gf-season-growth.json` still names the earlier release/site and must not be cited for this release.
- The real GF page still labels its source "Legacy match/player CSV import" even though the row came from a verified source fetch (`publish/resources.py`, the hardcoded `Source` in `match_details`). Correct the provenance label from the recorded provenance, with a regression that preserves the legacy case. Queue this until any running scratch cycle exits, per CLAUDE.md section 6.1.
- Review the unchanged-fixture path too: `_with_season_aggregates` currently returns when there are no match upserts, so a successful refresh with only source observations leaves `fixture_checked_at` stale. Include successfully checked seasons when selecting aggregates, with a regression for a refresh that has no changed match rows. Failed or partial fetches must still leave the accepted snapshot untouched.
- The parent updated `docs/operations.md` to include sealing, base-aware preview, and the model knowledge-cutoff requirement. It also added comments linking permanent release regressions to the historical diagnostic and scratch upload-failure helper in `test_release.py`. These are intentional changes to preserve.

Model: grok-4.7-high. Workspace: `/tmp/supercoach-via-grok47`. Branch: `cursor/rewrite-grok47`. Base SHA: `2aad178730990aad2623bd06cb0abfb17c5b0987` (same as `origin/rewrite/wip` at the start of this continuation). No commit, push, deploy, or schedule activation.

Lock hashes (unchanged): `uv.lock` `ce8e1730fcae14d2e62e741659c83b9e9694d2ac0799ba339f010fdcd8b04c15`; `web/package-lock.json` `3bff66d0c3ab314fa05e8eb3d0af033349352db30fba8b8f4ede23490353ea7b`; `pyproject.toml` `369e676b5207f55f44d1ab57fdf2b0c1dfed3a3a3be49af01961bc51355aa950`.

Node: the selected pin is 22.23.3 in `.node-version` and `web/.node-version`. `/usr/bin/node` is v22.23.3, and that is the binary the browser checks used. The file previously said 22.23.2; that was not the verified runtime. CI reads `.node-version`. Nothing new was installed. `npm ci` had already completed.

## Reuse

Claude's publish, ML, compact contracts, demo corpus, B1 evidence, Phase 8 timings and the old-contract rollback path are kept. This 2026-09-27 continuation extends them. It did not replace the scaffold or recompute the published 2026 holdout metrics. The 2026-09-28 recovery, recorded above, later imported the grand final into a separate data root.

## Phase status

Latest independent review: [C8_FOLLOWUP_REVIEW_2026-09-27.md](C8_FOLLOWUP_REVIEW_2026-09-27.md). The supervising Codex agent reproduced two additional publication/validation defects and identified missing deployment/switch gates. The measured final artifact's seal still verifies. C1, C7 and C8 are not closed; the detailed findings and temporary-only reproduction are linked in that review. Cursor was not restarted for this review.

| Phase | Status | Evidence |
|---|---|---|
| C0 | Recorded here. Install was already done in this worktree (`uv sync --locked --group dev --group legacy --extra ml`, `npm --prefix web ci`). | Lock hashes above. Cycle marker `.claude/audit/last_refresh_status.json` has `exit_code` 0, so no harness cycle is active. |
| C1 | Publication binds one captured inventory. A new sealed site must contain that public data. | F1 reads checksums and the seal once, before `active_release()`, and upload uses that inventory. F2: removing `site/data` fails validation; a self-declared `retained/*.json` does not authorize another data directory. An older HTML-only PASS still rolls back without re-running validation. Evidence: `var/agent-runs/cursor-grok47/closure-write-01/`. |
| C2 | Gate held on the unit files, then the demo command was aligned. | Feature order, eligibility and persisted feature spec: 82 passed in 7.87s across release, features, predict, evaluate and resources (before the later release-test additions). `pipeline.demo` sets `holdout_end` to 2026-05-01 and replays only `r08`/`r09`. That CLI test passed in 5.39s. |
| C3 | Keys and shared facts are in the builder and the browser fixture. | Canonical public keys are `k.` plus unpadded base64url. `public_key("a__b")` and `public_key("id.a_x_b")` differ. `legacy:a___b` round-trips. Legacy `__` strings still decode. Python property tests and Vitest ids tests passed. The browser fixture is regenerated with those keys, empty shared game-log columns, `match_facts`, and non-empty era and Brownlow-proxy CSVs. A Python-built release is read by the TypeScript join (`test_browser_readers_accept_the_python_release`, passed inside the hermetic run). |
| C4 | Browser failures from the key change were fixed and rechecked. | First production e2e: 310 passed, 10 failed, 4 skipped, 2.2m (5 cases on root and subpath). Causes: player budget omitted the shared match index; game-log assertions still looked for `demo_player_a1` in the URL; live heading is now "Snapshot"; an archived snapshot does not poll. Combined recheck from `web/` of budget-map, detail-states, players, routes, states, facts and screenshots: 248 passed, 0 failed, 93.3s (`web/test-results/e2e-report.json`, start `2026-09-27T08:56:12Z`). Screenshots: 40 files, five pages at 320/375/768/1440 in light and dark. |
| C5 | Two real sites measured. Hermetic tier is under 30s on four CPUs. | Sites unchanged: `var/corpus-site` 276,794,969 bytes and `var/corpus-site-final` 276,795,445 bytes. Projection 280.44 MiB under the unchanged 300 MiB budget. Final hermetic run: 688 passed, 122 warnings, 28.11s (`closure-repair-01/hermetic-xdist-final.txt`). Command: `taskset -c 0-3` with `OMP/OPENBLAS/MKL/NUMEXPR=1` and `pytest -n 4 --dist loadscope`. |
| C6 | Re-run on the final source. Published 2026 holdout figures were not recomputed. | Evidence directory `var/agent-runs/cursor-grok47/final-local-20260927T0943Z/`. Forecast on the new release is `unavailable` / `no_valid_future_fixture`. The new sealed site validated PASS and was not published to a host. |
| C7 | Scratch opt-in smoke passed, including an explicit parity PASS. | `SCVIA_NUMERIC_ENTRY=1 scripts/weekly_refresh.sh` from a scratch checkout of this dirty tree. Compare verdict `ok: true`, all-time delta 0, 130/130 yearly lists exact. Two sealed releases, injected copy failure left the live bytes in place, then rollback. Log: `closure-repair-01/scratch-smoke-b.log`. The earlier 7:56 direct-wrapper smoke remains a fail and was not deleted. |
| C8 | Not closed. This writer did not launch the review. | Parent review command is unchanged. Evidence for the reviewer is `var/agent-runs/cursor-grok47/closure-repair-01/`. |

## Measured release (uncontended)

No competing job at the start. Threads capped at 4. Evidence: `var/release-measure.json`, `var/corpus-site-growth.json`.

| Stage | Exit | Wall | Peak process-tree RSS |
|---|---|---|---|
| import-legacy (`--repair` of archived B1) | 0 | 38.41s | 1.871 GiB |
| forecast (no `--replay-season`) | 0 | 53.54s | 1.859 GiB |
| build-release | 0 | 37.10s | 1.368 GiB |
| Astro site | 0 | 13.57s | 0.586 GiB |

Forecast timings: `load_history` 1.287s, `train` 51.515s, `forecast` 0.008s. Status `unavailable`, reason `no_valid_future_fixture`. Bundle `bundle-3362930435fbf247a167`. Prediction dir `var/corpus/predictions/prospective-20260925T000000Z-203f7931991729e1`. Release `20260927T085937Z-d71eb27f6618` (`reused: false`). Public tree 272,462,055 bytes (91,379 files).

Astro output `var/corpus-site`: 276,794,969 bytes (263.97 MiB). Base 40,851,666 bytes (38.96 MiB). Mean season 2021–2025: 4,007,870 bytes (3.82 MiB). 2026: 4,128,965 bytes (3.94 MiB) and is not a complete season. Three more seasons at the 2021–2025 mean plus 5 MiB = 294,061,459 bytes (280.44 MiB), under 282 MiB. Headroom from the measured site to 300 MiB is 36.03 MiB.

The generated validators contain JSON pointers such as `/home/behinds`. A bare `/home/` byte marker treated those as leaks. The marker is now `/home/<name>/`, so `/home/behinds` validates and `/home/abhi/secret` still fails `private_content`. After that fix, `var/corpus-site` was copied into the release `site/` and `scvia seal-site` returned PASS. `seal_sha256` `2904b23d964b7a9371aedb60b422988e764fd28020ab46ed93419e05a255e6f4`. Seal wall 33.82s. Not published.

## Commands and results

- `.venv/bin/ruff check` on the publish/ML files touched by C1–C3, and again on `release.py` plus `test_release.py` after the marker fix: all checks passed.
- `.venv/bin/python -m pytest tests/scvia/unit/test_resources_matches.py tests/scvia/unit/test_release.py tests/scvia/unit/test_ml_features.py tests/scvia/unit/test_ml_predict.py tests/scvia/unit/test_ml_evaluate.py -q`: 82 passed, 7.87s.
- `npm --prefix web exec vitest run` on ids, search, release-tree, live: 24 passed, then live+tabular 8 passed.
- Publication downgrade and private-file tests, plus the key property tests: `pytest tests/scvia/unit/test_release.py tests/scvia/unit/test_public_keys.py tests/scvia/unit/test_resources_matches.py` — 43 passed, 1.95s (earlier this continuation).
- Hermetic scvia tier after the key and publication fixes: 623 passed, 25 deselected, 55.04s. Two filename assertions that still used colon-to-`__` were fixed first; the 55.04s run is that green one. Era corpus setup was module-scoped after it.
- `pytest tests/scvia/unit/test_release.py` after the home-pointer tests and marker change: 35 passed, 3.49s.
- Full hermetic tier after the marker fix and the era module-scope: 625 passed, 25 deselected, 66 warnings, 57.42s. Still over 30s.
- Browser: `npm --prefix web run test:e2e` with `PATH="/usr/bin:$PATH"` — 310 passed, 10 failed, 4 skipped. Combined recheck of the failing files plus facts and screenshots — 248 passed, 0 failed, 93.3s.
- Scratch smoke: `SMOKE_SKIP_SCRAPE=1 scripts/smoke_harness.sh` — exit 1, phase 0, log `/tmp/supercoach-smoke/20260927T090951Z.log`. Missing legacy interpreter. Harness files were not edited to point at this worktree's `.venv`.
- C8 review wrote `var/agent-runs/cursor-grok47/c8-review.txt`. It did not re-run pytest, Vitest, Playwright, or the corpus build.
- After the review fixes: `pytest tests/scvia/unit/test_release.py tests/scvia/unit/test_ml_features.py` plus the forecast/train modules — 52 passed in the release/features/predict slice (5.19s) and 26 passed in predict+train (4.80s). Vitest `tests/unit/ids.test.ts`: 7 passed. Match-page e2e (`routes`, `detail-states`, `states`, `budget-map`, `nojs`, root and subpath): 208 passed, 1.2m.

## Final local artifact (after the publication corrections)

Evidence: `var/agent-runs/cursor-grok47/final-local-20260927T0943Z/`. The earlier measurement files were not overwritten. Preserved release `20260927T085937Z-d71eb27f6618` still has seal `2904b23d964b7a9371aedb60b422988e764fd28020ab46ed93419e05a255e6f4`.

Source HEAD `2aad178730990aad2623bd06cb0abfb17c5b0987`. Diff sha256 `9da41e06593bec9d8227f76a4ebae5c581972191687afae4f4b4d0affb1c1625`. Lock hashes unchanged (see top). cpu_count 12, thread cap 4. Commands and exits are in `release-measure-final.json`.

| Stage | Exit | Wall | Peak process-tree RSS |
|---|---|---|---|
| import-legacy (`--repair` of archived B1) | 0 | 21.49s | 1.878 GiB |
| forecast (cutoff `2026-09-25T00:00:00+00:00`, no replay) | 0 | 46.62s | 1.813 GiB |
| build-release | 0 | 34.55s | 1.354 GiB |
| Astro (`var/corpus-site-final`) | 0 | 12.90s | 0.604 GiB |
| seal-site | 0 | 32.36s | 0.364 GiB |
| validate-release | 0 | 19.93s | 0.341 GiB |
| season growth | 0 | 5.42s | 0.021 GiB |

Forecast timings: `load_history` 1.274s, `train` 44.71s, `forecast` 0.008s. Status `unavailable`, reason `no_valid_future_fixture`. Bundle `bundle-d8349be6008314f2f6ac`. Prediction dir `var/corpus/predictions/prospective-20260925T000000Z-ae524d5b7bbc306f`. Release `20260927T100844Z-bc1ed67add68` (`reused: false`). Seal `72c090cc4510668688d58ddbc270957df07967048834f3a23cbc3f1a3a30008d`. Validation PASS. Not published to a host.

Site and sealed copy are both 276,795,445 bytes (263.97 MiB). Base 40,852,142 bytes. Mean season 2021–2025: 4,007,870 bytes. 2026: 4,128,965 bytes and is not a complete season. Three more seasons at the 2021–2025 mean plus 5 MiB = 294,061,935 bytes (280.44 MiB), under 282 MiB. Headroom from this site to 300 MiB is 37,777,355 bytes (36.03 MiB).

The sealed MatchView chunk `MatchView.-QgFrEK4.js` reads the query with `parsePlayerIdParam` compiled as `p=o(u), m=p?.key??""`. It does not contain the old `includes(':') ? encodeId(raw) : raw` ternary. `do not truncate` is present in the sealed `_astro` tree.

Hermetic tier on this source: `.venv/bin/python -m pytest tests/scvia -m "not integration" -q --durations=15` — 645 passed, 25 deselected, 66 warnings, 53.59s (`hermetic-final.txt`). Budget ≤30s is missed. The slowest calls are full demo trains and release builds (parallel build 3.83s, era corpus setup 3.06s, legacy import CLI 2.73s, demo CLI 2.71s, builder fixture setup 2.54s). The top 15 are about 25s and the remaining tests are about 28s. Removing the expensive ones would delete coverage. The budget was not raised and tests were not skipped.

Other final-source checks in that directory: `ruff-final.txt` (pass), wheel build/install/help/doctor/demo (demo exit 0, release `20260501T060000Z-demo-00fe6e5d239b`, run from `/tmp`). Earlier in the same directory, before the last publisher edit: mypy 0 errors, `gen:types:check`, web check, lint, vitest, integration+performance 25 passed in 243.38s, full Playwright 320 passed / 4 skipped (`e2e-report-full.json`). Those web and integration runs predate the upload-inventory fix; that fix is Python publication code covered by `test_release.py` (51 passed). The installed wheel was rebuilt after that fix and the external demo passed.

Local rehearsal (`rehearsal-publish-restore.json`, host `/tmp/scvia-rehearsal-final/host`): all 9 steps as expected. Publish old, publish new, refuse `not-a-release`, injected upload failure leaves the new release live, rollback restores `20260927T085937Z-d71eb27f6618`. Restored forecast reused bundle `bundle-d8349be6008314f2f6ac` (`train` 4.669s) and stayed `unavailable`.

## Closure write 01

Evidence: `var/agent-runs/cursor-grok47/closure-write-01/`. No commit, push, dispatch, or schedule change. Preserved seals still match: `20260927T085937Z-d71eb27f6618` / `2904b23d…` and `20260927T100844Z-bc1ed67add68` / `72c090cc…`.

Manual deploy: `python -m supercoach_via.publish.deploy pack` on the preserved final release wrote a 479,068,160-byte tar (archive `9fc24abd…`). Verify into `/tmp/scvia-pack-roundtrip/extracted` (not the release id) matched the seal and the 276,795,445 site bytes. The workflow download limit is 629,145,600 bytes so that tar fits. The site budget in `web/scripts/budget-lib.mjs` is still 300 MiB. The workflow was not dispatched.

Smoke (`smoke-sealed.log`, exit 0, wall 7:56, peak RSS 1,985,844 KiB): releases `20260927T110622Z-373d0f76633b` and `20260927T110940Z-bd62926be694`, seals `2847659a…` and `4f21192c…`, both forecast `unavailable` / `no_valid_future_fixture`, snapshot `sha256:55e295f1…`. A missing release was rejected and the live pointer returned to the earlier release. The comparison is not a pass: `compare-verdict.json` is `ok: false` for all-time scores, 2025 and 2026. The wrapper now exits on that verdict. This smoke called the wrapper directly and did not inject an upload-copy failure; those upload failures remain the unit tests in `test_release.py`.

Hermetic: `.venv/bin/python -m pytest tests/scvia -m "not integration" -q --durations=15` — 676 passed, 25 deselected, 53.70s. Budget ≤30s is missed. mypy: 0 errors. ruff: pass. Wheel demo from `/tmp`: exit 0, release `20260501T060000Z-demo-71e1e677695e`, forecast available on the demo clock. Playwright with `/usr/bin/node` v22.23.3: 320 passed, 4 skipped, 1.8m (`e2e.txt`). Integration and `gen:types:check` were not re-run.

## Closure repair

Evidence: `var/agent-runs/cursor-grok47/closure-repair-01/`. No commit, push, dispatch, or schedule change. The two preserved seals were not rebuilt.

Parity: regenerated legacy rankings from the captured tree, excluding `green_william_08092005` and `steele_roan_19092002` from the scan, and appending all 14 archived B1 games (9 `legacy:` rows plus 5 `src:` rows written under `repair_<id>_00000000_performance_details.csv`) onto an isolated copy. Input inventory `995b138997fad3efdbc310ca95910490e0357fa5b3e2fe6d23491dd2fd7e9b5e`, 27,350 files. Against preserved release `20260927T100844Z-bc1ed67add68`: verdict `ok: true`, all-time delta 0, 130 seasons in exact order, biography content matches (`existing-release-compare-final.json`). That recheck is after the short-table and common-order fixes. It did not rebuild the site.

Scratch smoke `/tmp/scvia-scratch-smoke-20260927b` (`scratch-smoke-b.log`, exit 0, wall 550s): entered through `SCVIA_NUMERIC_ENTRY=1 scripts/weekly_refresh.sh` on a scratch rsync of this tree, with `PYTHONPATH` set to that scratch `src` and the borrowed `.venv` binary. Releases `20260927T122715Z-2fd140a62b5e` (seal `0fdca8bc…`) and `20260927T123129Z-c8e6c6428af1` (seal `37cb769d…`). Compare on the first release was `ok: true` with all-time delta 0. An injected copy failure left live `20260927T122715Z-2fd140a62b5e` and its `index.html` bytes unchanged; rollback returned to that release. An earlier scratch attempt at `/tmp/scvia-scratch-smoke-20260927` failed because the ranker import could not see the checkout root; that tree was kept. The 7:56 smoke under `/tmp/scvia-candidate-smoke` is still not a parity pass.

Hermetic, after the final tests: 688 passed, 122 warnings, 28.11s on CPUs 0–3 (`hermetic-xdist-final.txt`). `pytest-xdist` is in the locked dev group. CI and the commented command in `docs/rewrite/switch-candidate/pre-commit-python.sh` use `-n 4 --dist loadscope` with one native thread. The live hook was not changed.

mypy: 0 errors (`mypy.txt`). ruff: pass (`ruff.txt`). Wheel demo from `/tmp`: exit 0, release `20260501T060000Z-demo-4ef535204bae`, forecast `available` (`wheel-demo.txt`). Playwright: 320 passed, 4 skipped, 1.8m, Node v22.23.3 (`e2e.txt`, `node.txt`). Those three gates ran before the last short-table and common-order edits; those edits are outside `src/` and `web/`, and ruff was run again after them. Integration, performance, and `gen:types:check` were not re-run.

## Closure finish

Evidence: `var/agent-runs/cursor-grok47/closure-finish-01/`. No commit, push, dispatch, or schedule change. No application edit after the smoke below.

The independent review (`CLOSURE_REVIEW_01.md`) left two blockers: the passing scratch smoke was older than the final comparison helper, and its fingerprint omitted executed inputs. Both are closed by `/tmp/scvia-scratch-smoke-20260927c` (`scratch-smoke.log`, exit 0, wall 557s). That run used the current `rehearsal_compare.py`. Compare verdict `ok: true`, all-time delta 0, 130/130 yearly lists exact. Releases `20260927T130550Z-d424a18cd169` (seal `37854d87…`, site 276,795,037 bytes, public 272,462,041 bytes) and `20260927T131007Z-fd89198bce47` (seal `5243945f…`, site 276,795,037 bytes). Injected copy failure left the first release's bytes in place; rollback returned to it. Forecast `unavailable` / `no_valid_future_fixture`. Train 46.002s, build 36.313s. Process-tree RSS peak 1,968,336 KiB and 1,970,188 KiB. Astro and seal wall times were not separately timed. Source inventory `fb871edaad740cd19eca046142328127747d4843f552dc5eff637d8b8b0dc38e`, 857 files, 18,310,694 bytes, zero scratch differences. Captured inputs remain `995b1389…`, 27,350 files. The growth projection was not recomputed; the preserved final site of 276,795,445 bytes still projects to 280.44 MiB under the 300 MiB budget. 2026 stays incomplete.

Hermetic after the fingerprint tests, before this smoke: 690 passed, 127 warnings, 28.64s on CPUs 0–3 (`hermetic-xdist.txt`).

Parent checks, not re-run here: integration/performance 25 passed, 688 deselected, 234.56s on four CPUs (`parent-integration-01.json`); `gen:types:check` exit 0 on Node v22.23.3 (`parent-gen-types-check.json`). The parent's pre-finish source inventory was `f62471d3…` (836 files). The smoke inventory is the later tree, after the fingerprint and the 22.23.3 pin. Earlier scratch logs, including `/tmp/scvia-scratch-smoke-20260927b`, are historical.

## Pending

- C8 is not closed by this writer. The parent returns this result to the independent reviewer. This writer did not launch that review.
- The 2026 grand final is in local snapshot `sha256:aa836549…` only. `schedule_complete` is still unknown, so 2026 is not a declared-complete season. The real forecast on that snapshot is `unavailable` / `no_valid_future_fixture`. Published 2026 holdout figures were not recomputed.
- Published model-card 2026 holdout figures were not recomputed.
- Two real shadow cycles and production activation remain external. The numeric entry is opt-in only. No commit, push, deploy, or schedule change.
