# Claude Code Opus 5.5 — finish integrity coverage and correct the app

## Use this payload

Give the existing Claude Code Opus 5.5 session this instruction:

> Read `/home/abhi/git/SuperCoach-VIA/docs/rewrite/CLAUDE_CODE_OPUS_55_FOLLOWUP_PAYLOAD.md` and complete its tasks on the existing review branch. Preserve the previous implementation and retained artifacts. Do not merge or deploy.

This is a follow-up task, prepared on 2026-09-29. Preparing this file did not launch an agent or change application code.

Local reproduction code and output are in the main checkout's `var/reviews/opus55-followup-20260929/probe.py` and `probe-result.json`. The probe uses temporary fixtures and leaves the retained inputs unchanged. A focused rerun of the existing CLI, cache, determinism and release checker tests passed: 44 tests in 34.20 seconds, serially with numeric-library threads limited to one. This is not a fresh run of the complete suite or a comparable measurement of the parallel hermetic budget.

## Starting point and scope

Continue the work in `/home/abhi/git/SuperCoach-VIA/var/worktrees/opus55-integrity`, branch `review/opus55-integrity`. The reviewed tip is `7f2c5ebca`, with four commits after `21dff89a6`. Check the actual tip and worktree status before editing; preserve subsequent work. Read `CLAUDE.md`, the original review payload, `docs/reviews/CLAUDE_OPUS55_REVIEW.md`, `docs/data-integrity.md`, `docs/pending-decisions.md`, and `docs/rewrite/SWITCH_PLAN.md`.

Finish the deterministic checker and implement the source-backed application corrections below in the isolated checkout. Keep the static site plus local Python architecture. Keep the checker offline, opt-in and independent of model calls. Preserve the retained inputs under the main checkout's `var/finalized/`; build corrected candidates under a new output directory. Do not overwrite the original snapshot, sealed release, archive or preview alias.

This task does not authorize merge, push, deployment, schedule changes, default harness activation, a test-budget increase, or treating rehearsals as genuine shadow cycles. Use the repository's safe commit wrapper for implementation commits. Apply `CLAUDE.md` §6 to changes that alter gates, hooks, artifact selection or harness behavior; record the scope determination and required smoke evidence.

## 1. Reproduce and fix the report-output collision

Confirmed on the reviewed tip through the actual CLI using the small `integrity_fixtures.with_sources` corpus:

```text
scvia check-integrity --data-root <fixture> --scope data --as-of <fixture-as-of> \
  --report <output>/report.json --execution <output>/report.json --json
```

The command exits zero and advertises a canonical report digest. The file at `report.json` is actually `scvia.integrity-execution/1` and has no audit outcome: execution metadata overwrote the report. `integrity/runner.py:write_outputs` checks that outputs are outside inputs but does not check whether outputs alias one another.

Write failing tests first. Validate the report, execution and findings-stream destinations together before auditing or writing. Refuse collisions with a usage error, including equivalent relative/absolute paths and aliases through symlinked parent directories. Retain the existing outside-input checks. Test distinct destinations as well as collisions, with existing output bytes preserved on rejected requests.

Also test an I/O failure while writing the output set. The CLI currently says "no report was written" for every exception, even though an earlier output may already exist. Make the receipt truthful about any completed or partial writes. Do not promise an atomic multi-file transaction unless the implementation provides one.

## 2. Complete semantic coverage and make completeness truthful

The original payload required comparisons for all relevant published resources. The review currently treats team/history comparisons as optional. Complete that requirement.

Evidence from the retained report: `scope.complete` is true, but `coverage.public_compare.not_compared_by_model` includes team-season resources, history tables, lists, downloads, overview and quality data. The operator guide also discloses missing comparisons for accuracy reports, articles and captured B1 player pages. The review's claim that "every published cell matches the snapshot" is therefore too broad.

A local diagnostic built the existing demo fixture, changed the first team-season statistic's total, adjusted its mean consistently, and used `integrity_fixtures.reseal` to recalculate checksums, embedded data, the seal and validation. The canonical snapshot was unchanged. The checker produced identical findings before and after the mutation; `release.artifact`, `release.validate` and `release.public` all returned PASS. The full demo audit remained UNKNOWN in both cases for other required checks; this diagnostic does not establish a false full-scope PASS. It establishes that the altered numeric resource receives no semantic comparison.

Required work:

1. Inventory every machine-readable published resource type and its authoritative input. Distinguish snapshot-derived values from captured live inputs, pinned history/list imports and evaluated model artifacts. Do not compare unlike vintages or interpret a proxy metric as an observed statistic.
2. Add independent comparisons for team-season facts/aggregates, history and list values, numeric downloads and their metadata, overview and quality summaries. Complete any missing comparisons for accuracy/prediction/live resources against their own pinned inputs. Reuse verified input loading and policy, without calling the producer to manufacture the expected output.
3. Add the missing B1 player-page comparison for the captured evidence already supplied. State exactly which rows and fields it establishes. Do not imply that all historical rows have independent source evidence.
4. Separate checks that executed from semantic coverage achieved. A required comparison that is unimplemented or lacks its required input must yield UNKNOWN and prevent a complete PASS. Optional absent historical captures remain an explicit source-coverage limitation. Byte/seal verification alone is not a semantic comparison.
5. For prose articles and other resources without a general numeric comparator, verify the available provenance/content contract and explicitly report what remains unaudited. Do not imply that every prose claim has been proven.
6. Add adversarial tests that change values while recalculating structural hashes and seals, including team totals, history ranks/values, download row membership and stale summary values. Require the responsible semantic check to detect the discrepancy. Exercise these mutations through cold, warm-cache and changed-since modes.

Correct the operator guide, report schema and review summary together. Preserve useful distinctions between source truth, snapshot consistency and transport integrity. Do not solve this only by weakening the meaning of full scope or deleting the coverage requirement.

## 3. Correct zero semantics using source evidence

The review identified a real denominator problem. Independent CSV arithmetic reproduces Scott Pendlebury's goals total of 209 over 442 rows, with 162 nonblank goal cells and 280 blanks **[data]**. The current non-null mean is about 1.29; dividing that total by all these games gives about 0.47 **[data]**. This confirms the arithmetic mechanism; establish the meaning of the blanks from the source before changing storage.

Do not use `recorded_from <= season` and column presence as sufficient proof that every blank means zero. Establish deterministic rules for the particular source, table, statistic and reporting state. A recorded zero, an absent column, an unknown value and a statistic that does not apply are different cases.

Before changing either importer:

- Annotate a small set of captured source rows independently of the production parser. Establish which blanks mean zero and which remain unknown or not applicable. Include modern sparse counts, genuinely absent data within an otherwise recorded era, pre-era gaps and malformed nonblank tokens.
- Explicitly establish the Brownlow applicability and publication rules from source evidence, including finals and votes not yet reported. Do not automatically use all career appearances as its denominator.
- Preserve decision 3's observed-data policy and `N of M` disclosure for genuine missing coverage. Correct source-proven zeros without turning unknown historical data into zeros. If evidence cannot distinguish a case, retain null and report the limitation.
- Encode the rules in deterministic, tested code used consistently by legacy and source imports. Independently test the checker interpretation so a shared untested mapping cannot make both sides agree on a mistake.

Build a new candidate snapshot and release. Recompute downstream aggregates, rankings and model features affected by the corrected inputs. Ensure changed semantics invalidate model caches; retrain applicable bundles and keep holdout/replay labels accurate. Do not reuse earlier evaluation figures as evidence for changed models.

Test the browser's totals, denominators, means and missing-value labels against independent expected values. Keep legitimate coverage gaps visible. Do not require the blanket absence of every blank-related warning as proof of correctness, and do not globally relabel "games with data" as "games in recorded era" when reporting remains incomplete.

This task implements source-supported corrections in a candidate build. It does not record an invented owner approval or reverse an editorial policy. Isolate any genuinely unresolved policy choice, explain its consequences, and continue work that does not depend on it.

## 4. Repair the attendance mismatch through the pipeline

Independently confirmed: snapshot match `m:2026:r17:collingwood:richmond:0` has attendance zero **[data]**, while pinned season capture `87c99c1a2eb84c4957759e5ec82e02aec97d7daf5264e1ba6db800cdd59bc83a` records 62,117 **[historical record]**. The capture exists under the retained data root's `raw/objects/87/` directory.

Use the captured source to apply a bounded correction with provenance in the candidate pipeline. Do not manually edit a sealed release, suppress the finding, or replace a known attendance with null merely to make the check disappear. Verify the new snapshot, public resource and embedded site agree with the capture. Preserve legitimate recorded zero attendances elsewhere.

## 5. Close the remaining confirmed defects

Work from the existing O55 findings, with failing regressions before changes:

- **O55-03:** malformed nonblank statistic text must fail parsing or enter an explicit quarantine path, rather than silently becoming null. Test empty and legitimate missing tokens separately.
- **O55-04:** validate each compact row against its declared statistic columns, including width, duplicates, allowed vocabulary and order. Exercise producer validation and the browser contract. Keep independent checker tests able to construct invalid releases; do not simply remove the old negative case when resealing starts rejecting it.
- **O55-05:** reject duplicate JSON keys and invalid boolean types at the appropriate production boundaries. Cover existing rollback compatibility and avoid silently coercing malformed values.
- **O55-06:** resolve drawn-final/replay links from verified dates or other sufficient source identity. A guessed legacy date is not proof. Quarantine ambiguous rows with an actionable finding instead of relinking them by outcome alone.
- **O55-08:** address the demonstrated mobile clutter and unclear match headings, then rerun affected browser states at the required widths and themes.

Do not expand this into another architecture rewrite. Preserve working code and the intentional legacy URL alias.

## 6. Restore the test performance target without hiding work

Keep the rewrite's existing 30-second hermetic-tier target. Profile the full tier on the documented environment with fixed worker/thread counts and no competing training job. Inspect repeated fixture construction, scans, model training and subprocess startup. Optimize shared immutable fixtures or test setup while retaining meaningful behavioral coverage and isolation.

Do not delete/skip tests, reclassify fast regressions as integration tests, or silently move them out of the gate to claim the target passed. Do not raise the budget. If the target remains unmet, report the measured full runtime, dominant costs and concrete alternatives, with the budget still open. Distinguish this rewrite target from any separate legacy-tier budget in `CLAUDE.md`.

## 7. Validation and final handoff

Run the affected tests while implementing, then the required complete Python and browser tiers once on final code. Include the new negative cases above; existing happy-path tests did not expose these gaps. Record commands, outcomes and skipped checks accurately.

On preserved original inputs, rerun the audit and retain the original failures as a reference. On the new candidate, run the full audit cold, warm and changed-since with pinned `as_of`, policy and evidence, and multiple worker counts. Canonical reports for identical final inputs must be byte-identical across execution modes. Changes to inputs must invalidate the relevant cached results. Measure wall time and peak process-tree RSS without overlapping training/tests.

Recheck the grand final end to end against its pinned capture and all downstream representations. Confirm its presence separately from source coverage, zero semantics and any missing source fields. Rebuild and seal the candidate site, remeasure its size and season-growth headroom, and run the required old/new rehearsal and applicable scratch harness smoke on the final source inventory. Preserve the prior artifact for local rollback. These checks do not replace genuine shadow cycles.

Keep full logs and artifacts in a new `var/reviews/opus55/<run-id>/` directory. Commit small reproducible regression fixtures and sanitized evidence with the implementation. Update `docs/reviews/CLAUDE_OPUS55_REVIEW.md` and `docs/data-integrity.md` around the final implementation, including the two additional checker findings from this follow-up.

End with: branch/commit and dirty status; fixes and remaining findings; exact source/snapshot/release identities; semantic and source coverage; test/performance results; paths and commands for preview and audit; and any real owner decisions or external prerequisites still open. Give separate verdicts for the checker, candidate app and production activation. Do not claim completion merely because every implemented check executed.
