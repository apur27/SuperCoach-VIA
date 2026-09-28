# Claude Code Opus 5.5 — thorough review and deterministic integrity checker

## Launch this task

Run from `/home/abhi/git/SuperCoach-VIA`:

```bash
claude --model claude-opus-5-5 --effort high "Read docs/rewrite/CLAUDE_CODE_OPUS_55_REVIEW_PAYLOAD.md and complete its review and deterministic data-integrity implementation task. Preserve the existing Claude and Grok work."
```

This uses the native Claude Code model name. [Anthropic's model configuration documentation](https://code.claude.com/docs/en/model-config) identifies `claude-opus-5-5` and requires Claude Code 2.1.280 or later; the locally checked version is 2.1.283. Confirm the session actually selects Opus 5.5. Report an access or provider restriction if it prevents that selection. This payload has not been executed.

---

## Your assignment

Perform a thorough, independent review of the merged SuperCoach VIA app. Cover data correctness, security, performance, user experience, accessibility, code quality, tests and local operations. **Also implement executable, deterministic code that checks data integrity intelligently, with regression tests and machine-readable reports.** A prose review or a script that merely repeats existing PASS records is insufficient.

The integrity checker is the implementation deliverable. Record unrelated application defects with precise remediation steps; keep those fixes out of this change unless they are necessary to implement or test the checker. Preserve the static website plus local Python pipeline architecture and the existing Claude/Grok implementation.

Use deterministic rules, explicit inputs, content hashes and documented policies. The checker must make no model calls. It must work offline with captured evidence, explain failures, and distinguish contradictions from missing evidence. Its result must not depend on an agent's judgment, the current clock, filesystem traversal order or a network response that can change between runs.

Complete both the review and checker. Do not stop after writing a plan, inventorying files, or running the happy-path tests.

## 1. Establish the starting state

At payload creation:

- Branch: local `main`, clean before this payload was added.
- Merged application: `2bcdfbeb4142c98bf120e5056c06f3f142515c3f`.
- Previous handoff: `c101a09e33abf51b10c4546536ccdb66b93f9283`.
- Claude implementation retained as ancestor: `2aad178730990aad2623bd06cb0abfb17c5b0987`.

Inspect current HEAD, branch, status and active processes. Confirm the application commit is an ancestor. Record any later changes; do not reset to the recorded revision. Use an isolated branch/worktree for checker implementation, preferably under `var/worktrees/`. Preserve unrelated local changes, existing releases, the recovery worktree, stash and backup. Main's retained data can be read from that worktree by absolute path.

Read:

1. `CLAUDE.md` and any more specific instructions that actually exist.
2. `docs/rewrite/FINALIZATION_2026-09-28.md`.
3. `docs/rewrite/finalization-20260928/README.md`, `independent-review.txt`, the data/browser reports and source inventory there.
4. `docs/operations.md`, `docs/migration.md`, `docs/model-card.md`, `docs/rewrite/SWITCH_PLAN.md`.
5. `docs/rewrite/CLAUDE_WORK_REUSE_2026-09-27.md`, the original review, rewrite plan and Cursor plan. Treat their dated findings as history to reconcile with current code.

The earlier PASS is evidence with a defined scope. Independently test important claims and look beyond the previous review's two final fixes. For every reviewed subsystem, name the files read and the behavior checked. Avoid treating a file listing or a passing suite as proof that every path is correct.

## 2. Correct data and artifact baseline

Use the following local retained inputs, which are outside the agent worktree:

| Input | Location or identity |
|---|---|
| Data, models, predictions, captures | `var/finalized/data/` |
| Release | `var/finalized/releases/20260928T111014Z-ca96603163ad/` |
| Preview alias | `var/final-site` |
| Deployment archive | `var/finalized/site.tar` |
| Metadata | `var/finalized/metadata.json` |
| Snapshot | `sha256:aa836549e10e96a9039cc4642bbe563b94247e0bfaf95e745a0971cc0bb2b98f` |
| Seal | `bd9a655b06795c2d83ca77a48adb9df85fb2774463fa390f814dac2ff72474fa` |
| Archive SHA-256 | `59496b26d421998b874ea879277b63365bc49fa1427c95902565bc4abc9599f7` |
| Reviewed source inventory | `318c37e8b2a80d6f518f89cde1427923dc4d7a1a389224d58f81400442c8ea5a` |

The new snapshot/site contains the grand final. The legacy `data/matches/matches_2026.csv` remains a pre-final migration input. Verify the current app through its selected snapshot and release, rather than assuming this CSV is its current database.

The snapshot retains historical `legacy_unverified` provenance. `schedule_complete` and `source_status` remain unknown. Forecasts are unavailable because there is no valid future fixture. Preserve these distinctions and the holdout labeling in the model card.

These large local artifacts are ignored by Git. If absent, report precisely what cannot be checked, continue with portable fixtures, and document how to transfer or reproduce the inputs. Never label an unexecuted real-data check as PASS.

Useful prior checker: `var/recovery/cursor-grok47-20260928/verify_grand_final.py`. It compares captured source cells, the snapshot, public resources and embedded site data. Read and reuse its proven comparisons, then remove its recovery-directory assumptions and generalize beyond one match. Do not copy its hardcoded fixture counts into general validation rules. The committed source-statistics and capture reports are in `docs/rewrite/finalization-20260928/`.

## 3. Thorough application review

Review the full paths below, including input parsing, error handling and consumers. Reconcile each prior finding as fixed, still open, regressed, or not reproduced.

| Area | Required investigation |
|---|---|
| Data ingestion and storage | Source revisions, bounded fetches, parsing, canonical identities, quarantines, provenance, atomic promotion, partition selection, schema evolution, empty/partial/unreachable sources, interrupted writes and concurrent readers/writers |
| Analytics | Ranking parity, ties, observed denominators, historical coverage, career/season totals, filters, duplicate identities, stale aggregates and joins that accidentally multiply rows |
| ML | Event versus knowledge time, holdout/replay labels, bundle eligibility, persisted feature order/specification, code fingerprints, cache invalidation, calibration and unavailable forecasts |
| Publication and security | Manifest closure, hashes, sealing the actual uploaded bytes, traversal/symlinks, unexpected files, conflicting JSON keys, archive extraction limits, captured-inventory races, existing destination tampering, private content, failure recovery and rollback across schema changes |
| Browser and UX | Real navigation, loading/empty/error states, missing data labels, legacy links, season membership, search, comparison, watchlist persistence, shared match facts, downloads, history/proxy labels and immutable live-snapshot behavior |
| Accessibility | Keyboard and focus, headings/labels, screen-reader table meaning, status announcements, contrast, reduced motion, mobile overflow, light/dark themes and no-JavaScript behavior |
| Performance | Release build wall time/process-tree RSS, repeated scans, memory growth, full history versus one-season work, compressed/uncompressed payloads, route-specific downloads, main-thread work, test costs and contention |
| Code and tests | Duplicated authorities, unclear boundaries, type/schema drift, generated contracts, hidden globals, unsafe defaults, exception handling, hermeticity, flaky ordering, fixture realism and meaningful negative cases |
| Operations and CI | Locked install, wheel outside the checkout, absolute release path, root/subpath builds, preview instructions, CI coverage, source mode/network opt-in, ownership locks, exit receipts, shadow-cycle requirements and rollback instructions |

Inspect real pages at widths 320, 375, 768 and 1440 in both themes, using the existing screenshot matrix where appropriate. Follow grand-final match-to-player links and back. Cover overview, search/player list, player, match, comparison, predictions, history, downloads, watchlist, live snapshot and data status. Record console/request errors, representative screenshots and any route/state you could not inspect.

For security findings, state the trust boundary and a practical local reproduction. For performance findings, record the workload and measurement. Avoid presenting hypothetical risks or stylistic preferences as confirmed defects.

## 4. Build on existing deterministic checks

Start with these implementations and their tests:

- `src/supercoach_via/ingest/reconcile.py`: `validate_dataset`, coverage policy, deterministic issue IDs, historical exceptions, arithmetic and foreign-key rules.
- `src/supercoach_via/storage/snapshots.py`: content-addressed fragments, containment, manifest identity, hash verification and promotion.
- `src/supercoach_via/storage/queries.py`: bounded DuckDB/Arrow access.
- `src/supercoach_via/publish/release.py` and `publish/deploy.py`: semantic validation, captured inventories, embedded public data, seals and archive verification.
- `src/supercoach_via/domain/schemas.py`, `config/coverage.yaml`, `config/stat_coverage_eras.yaml`, source policies and generated public schemas.
- `src/supercoach_via/publish/resources.py`, `publish/view_models.py` and the browser's stat/key/join helpers.
- Existing reconciliation, storage, release, deploy, source-check, cross-language and match-resource tests under `tests/scvia/` and `web/tests/`.

Produce a short rule map: requirement → existing implementation/test → gap → added check/test. Extend or compose the existing checks. Keep a single authoritative policy for a rule. Use ordinary Python/SQL and the locked dependencies; avoid introducing a generic plugin framework or an external data-validation service.

## 5. Required integrity-checker contract

Add a documented `scvia check-integrity` command, or extend an existing command if it can expose this contract without changing its established behavior. Report the final interface and why it was chosen. The example below is a **target interface to implement**, not an existing command:

```bash
scvia check-integrity \
  --data-root /absolute/path/to/var/finalized/data \
  --snapshot current \
  --release-dir /absolute/path/to/var/finalized/releases/20260928T111014Z-ca96603163ad \
  --scope full \
  --as-of 2026-09-28T12:00:00Z \
  --report /absolute/path/to/var/reviews/opus55/integrity-full.json \
  --json
```

### Determinism and read-only behavior

1. Resolve `current` once, pin immutable snapshot/release inventories, and bind the report to their verified identities. A moving pointer or file replacement during the scan must be detected or yield a consistent captured view.
2. Inputs include checker/rule version, policy hash, scope, pinned manifests and an explicit `as_of` when testing freshness. No implicit current time in a correctness verdict.
3. Identical bytes and options produce identical ordered findings and canonical report bytes, including across relocated directories and worker counts. Use stable sorting and semantic row keys. Reject non-finite JSON numbers. Define field-specific numeric tolerances only where the contract requires rounding; do not use a blanket approximate comparison.
4. Store elapsed time, host details, absolute local paths and timestamps in separate execution metadata so they do not change the canonical result. An evidence fingerprint is not a trust signature; a self-consistent hash alone does not prove a statistic correct.
5. Read source data, snapshots, models and retained releases without modifying them. Write reports atomically to a separate output directory; refuse report/cache locations inside immutable inputs. Reuse existing validation with non-writing options where available.
6. Every check reports PASS, FAIL, UNKNOWN or NOT_APPLICABLE, with a reason and its coverage. Map this explicitly to existing `ValidationReport`/severity conventions without weakening them. Missing optional historical evidence may be UNKNOWN; a required check that cannot run prevents an overall complete PASS.
7. Use existing CLI exit-code conventions. Violations, incomplete required verification, invalid usage and checker execution failures must be distinguishable and nonzero. Document whether policy-accepted historical warnings permit exit zero. A crash is never translated to PASS.

### Machine-readable result

Use a versioned, validated JSON contract containing:

- Checker/rules/policy identities; snapshot/release/seal and evidence digests; explicit scope and `as_of`.
- Overall outcome, scope completeness and severity counts.
- Checks requested, checks performed, rows/resources examined, unknown/not-applicable counts and reasons.
- Findings with stable `rule_id`, stable issue ID, severity, status, entity key, table/resource and field, expected/actual values, supporting evidence, and suggested operator action.
- Deterministic sample limits plus **exact total counts**. If examples are capped, say so and provide an optional complete findings stream. Never mistake truncating examples for checking only a sample.
- A report digest over canonical content. Bound any accepted exception to the exact rule, entity and reason; make accepted and stale exceptions visible. Retain the existing rule against suppressing current-season defects.

## 6. Integrity rules to implement or prove already covered

### A. Stored bytes and contracts

Verify pointers/manifests against their semantic identities; referenced fragment existence, containment, hashes, byte sizes, actual row counts, schemas and partition declarations. Detect duplicate keys, null required values, invalid enums, conflicting JSON keys, non-finite statistics and misplaced records. Integrity failures in historical bytes remain failures even when historical semantic coverage is incomplete.

For releases, check the closed public inventory, resource references, schemas, checksum binding, exact site/public agreement and final seal. Detect extra, missing, truncated, substituted and path-escaping resources. Check that validation records name the same artifact. Preserve the existing older-release rollback contract.

### B. Relationships and independent aggregates

Check foreign keys, canonical player identities, alias cycles/ambiguity, unique player-match membership, participating clubs/opponents, seasons/stages, linked dates and date quality. Recompute season match counts, first/last dates and scheduled/completed counts from accepted match rows. Reconcile player season membership and aggregate totals with contributing facts, using observed denominators and documented rounding.

An integrity audit must recompute the relevant invariant from facts; comparing two values produced by the same faulty aggregation is insufficient. Reuse contracts and identity rules while making the key checks independent of the producer being tested.

### C. Football arithmetic and coverage

Check score arithmetic and quarter/final consistency where the source supplies the required fields. Check disposals against kicks plus handballs when all three are observed, numeric ranges and applicable statistic relationships. Reconcile team/player totals only when the evidence establishes complete, compatible coverage.

Use the era policy and recorded provenance. Null is not zero. Do not enforce a modern team size across historical seasons, assume every statistic exists in every era, require player behinds to include rushed behinds, or invent a fixed season match count. Distinguish cumulative quarter scores from per-quarter scores and handle documented corrections/exceptional matches explicitly. Unusual but valid values need an explainable warning rule, not a guessed hard failure.

### D. Freshness, source completeness and revisions

Compare accepted source observations and revisions with fixture freshness and source hashes. Recheck the successful-unchanged-fixture case and verify partial failures do not promote or claim freshness. A source request timestamp alone does not prove its returned data is current or complete.

Compare a pinned source fixture inventory with accepted matches when that inventory is available. Report missing completed fixtures and unexpected changes; distinguish a future scheduled fixture from a missing result. A grand final's presence alone must not set schedule completeness. Report actual coverage instead of upgrading the whole legacy corpus to verified.

### E. Source → snapshot → release → site

Generalize the existing grand-final verification. Select source-captured matches by manifest/evidence identity, compare player identities, clubs, dates, source links, all available statistic cells and quarter scores with canonical rows, then compare the corresponding compact public arrays, shared match facts, player logs and embedded site resources.

Check array lengths, stat-column ordering, omitted all-null columns, duplicate keys, row membership and joins. Ensure a value attached to the wrong player fails even if totals match. In full mode, check all relevant published resources, not just the grand final. Identify precisely which historical facts lack independent source captures.

The grand-final capture is a required end-to-end regression fixture. Preserve all original nulls and source identities. A source parser and the verifier must not share the same untested mapping mistake: use a small independently annotated raw-source fixture and deliberately permuted columns/identities to challenge the adapter.

### F. Predictions and model artifacts

Where model/prediction inputs are supplied, validate their referenced snapshot, feature specification/order/fingerprint, relevant code hashes, knowledge cutoff and forecast cutoff. Check interval ordering, finite values and prospective/replay/holdout labels under the existing contract. An unavailable forecast with no future fixture is valid; a missing mandatory model artifact in an available forecast is not. Auditing integrity should not require retraining models.

## 7. Make repeated checks efficient without losing coverage

Implement a reliable full audit first. Use projected columns, season partitions, batched joins and streaming hashes rather than loading every dataset into Python objects or re-reading the same files for each rule.

Then add a changed-data mode if it provides a measured benefit. Require an explicit prior snapshot/report. Determine changed partitions from verified content identities and build the affected entity set, including old and new relationships for moves/deletions. Recheck dependent aggregates, neighboring records needed for chronology, relevant public resources, and global uniqueness/referential constraints. Do not validate only the rows that happened to be added.

Cache **semantic check results** only when verified input content, rule/checker version, dependencies, policy, scope and applicable `as_of` match. Filename, mtime, size or an unverified hash claim is insufficient. A file modified under the same content-addressed filename must still be detected. Always perform the byte-integrity verification needed to trust any reused result.

If the cache/baseline is missing, corrupt or incompatible, fall back to a full check or return an explicit incomplete result. In tests, merged changed-mode results must equal a fresh full audit for the same final inputs. Report checked, reused and invalidated work. Never label a narrow audit as a whole-corpus PASS.

Measure cold and warm runs, data sizes and peak process-tree RSS. Explain any worthwhile optimization omitted from the first implementation. Keep existing app/test budgets; do not invent a claim that a whole-corpus audit must meet the hermetic unit suite's time limit.

## 8. Prove the checker finds bad data

Write failing tests before implementing rules. Use disposable synthetic snapshots and copies; never corrupt retained inputs. Include independent negative cases for:

- Missing/tampered fragment, bad manifest identity, invalid row count, malformed schema, duplicate keys and orphan references.
- Swapped player identities or stat columns with unchanged totals; wrong club/opponent/date/season; moved/deleted facts leaving stale aggregates.
- Changed numeric data **with all structural hashes recalculated** so a semantic/source check, rather than just SHA-256, must detect the defect.
- Null converted to zero; modern coverage rules applied incorrectly to an older season; legitimate historical exception; invalid current-season suppression.
- Successful unchanged fixture fetch with stale metadata; partial fetch incorrectly marked fresh; absent independent source evidence.
- Wrong snapshot in release, truncated compact arrays, mismatched shared match facts, altered embedded site data, extra file and unsafe resource path.
- Wrong feature order/fingerprint, future knowledge cutoff, invalid prediction interval, and correctly unavailable forecast.
- Unknown required check, unreadable required input, checker exception and interrupted report write.
- Stale/poisoned cache, unchanged filename with changed bytes, changed policy/code, affected cross-partition relationships, and changed-mode versus full-mode equivalence.
- Repeat runs, shuffled query-result/directory iteration, a moved directory and worker-count changes producing the same canonical report for the same pinned bytes. If a fixture physically reorders rows and therefore changes content identities, compare normalized findings while requiring the new identities to be reported correctly.
- Source/canonical/public/site inputs remaining byte-identical after an audit, including a failing audit.

Use seeded or exhaustively enumerated synthetic cases as useful; no nondeterministic test failures. Assert rule IDs and entity keys, not merely a nonzero exit. Each negative case should have a valid control so a checker that always fails cannot pass the suite.

Run the checker on the retained real snapshot/release and produce an honest result. Do not alter real records, accepted exceptions or policy thresholds to obtain green output. Classify and explain any actual anomalies.

## 9. Verification and operational boundaries

Inspect existing environments first. Locked setup, when needed:

```bash
uv sync --locked --group dev --group legacy --extra ml
npm --prefix web ci
```

Run Ruff, mypy, the full hermetic scvia tier, affected integration checks and the cross-language resource tests. For the thorough browser review, run generated-contract checks, type checks, lint, Vitest and production Playwright at both supported bases. The scripts are in `web/package.json`; use their actual names.

```bash
uv run --locked ruff check
uv run --locked mypy
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 uv run --locked pytest tests/scvia -m 'not integration' -q -n 4 --dist loadscope
npm --prefix web run gen:types:check
npm --prefix web run check
npm --prefix web run lint
npm --prefix web test
npm --prefix web run test:e2e
```

Save command, working directory, environment, exit code and output for each check. Distinguish failures from missing prerequisites; do not silently skip browser execution if it cannot start. Do not delete tests or raise budgets to claim success. Serialize training, integration, audit benchmarks and release builds.

Prior measurements are comparison points: 695 hermetic tests in 29.33s on four distinct physical cores (`0,2,4,6` on this machine), versus 34.15s on CPU IDs `0–3`, which share two physical cores. The site was 276,821,039 bytes; the recorded growth projection was 280.46 MiB against a 282 MiB headroom target and a 300 MiB site budget. Release build targets are 60s and 2 GiB process-tree RSS on the reference setup. Record new measurements with hardware and contention, and distinguish test count/runtime changes caused by the new checker.

Follow `CLAUDE.md` section 6 for any harness/gate changes. Prefer an opt-in checker command initially. If changing what a current gate accepts or promotes, state the scope decision and operator recovery, freeze relevant source during the required scratch weekly smoke, and verify the final source inventory. Existing full-tier commit checks still apply. Use `scripts/git_commit_safe.sh` for any commits.

The review may write its reports, isolated implementation/tests and disposable reproduction artifacts. Preserve main, original data, sealed releases, hooks and schedules. Do not publish, deploy, change the default weekly entry, manufacture shadow-cycle evidence or merge checker changes into main as part of this review. Production still requires two genuine shadow cycles and the recorded owner decisions; preparing local checks does not satisfy those prerequisites.

## 10. Deliverables and completion criteria

Write `docs/reviews/CLAUDE_OPUS55_REVIEW.md` with:

1. Reviewed commit/artifact identities, actual model, scope and limitations.
2. Findings ordered by severity: ID, confirmed/suspected status, file and line, concrete trigger, expected/actual result, impact, evidence path, minimal remediation and regression required. Reserve blockers for consequential reproducible defects. Distinguish existing data anomalies from checker defects and missing source evidence.
3. Coverage matrix for every area in section 3, including surfaces that were inspected with no finding.
4. A separate verdict for local application use, integrity-checker readiness and production activation. Explain any blocker; do not compress distinct readiness states into one PASS.
5. Ordered remediation tasks with acceptance criteria, dependencies and estimated scope.

Write `docs/data-integrity.md` describing the actual checker command, rule catalog, policies, scopes, result schema, exit codes, cache invalidation, determinism guarantees, example failure and operator response.

Keep machine-readable reports, execution metadata, screenshots and complete logs in a distinct `var/reviews/opus55/<run-id>/` directory. Commit small sanitized representative reports/fixtures and test code needed to reproduce the checks; keep large artifacts out of Git. Give exact paths in the review.

Finish only when the checker runs without an LLM, negative controls demonstrate detection, determinism/read-only tests pass, real-data results are recorded, changed-mode equivalence is proved if implemented, and the broad app review has its findings and coverage matrix. Report any remaining failure honestly. Return the review, checker commands, test results and branch/commit state in a concise handoff.
