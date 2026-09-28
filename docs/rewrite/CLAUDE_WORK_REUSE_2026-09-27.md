# Continue Claude Code's work

**Owner instruction:** use Claude Code's implementation; have Cursor Agent use Grok 4.7 to complete and improve it. This is a continuation plan.

## 1. Which code is the starting point?

| Location/revision | What it represents | Treatment |
|---|---|---|
| Current dirty `main`, HEAD `b4ce74770` | Refreshed data and an earlier untracked implementation; this was the initial review target | Preserve all changes; do not use its application files to overwrite the branch |
| Local `rewrite/wip`, `e72d49f45`, in a Claude worktree | Older local branch tip | Do not assume this is the newest implementation |
| Verification worktree at `208a54e21` | The revision in the owner's pasted independent verification | Keep it and its real releases available; do not remove it during this task |
| Fetched `origin/rewrite/wip`, **`2aad178730990aad2623bd06cb0abfb17c5b0987`** | Newest branch revision observed on 27 September 2026 | Pin this as the reviewed baseline; check for later commits before implementation |

`git ls-remote` and `git fetch origin rewrite/wip` confirmed the remote revision. The review did not switch branches, merge, commit, push or edit Claude's worktrees. Focused checks used a separate temporary export of the pinned revision.

Read the **branch versions** of these files before implementation: `docs/rewrite/IMPLEMENTATION_STATUS.md`, `REHEARSAL.md`, `SWITCH_PLAN.md`, `docs/operations.md`, `docs/migration.md`, `docs/model-card.md`, and `config/app.example.toml`. Some do not exist in the older current checkout. Inspect with `git show <revision>:<path>` or in the continuation worktree.

## 2. Work already delivered: retain it

| Claude's work | Evidence on the branch | Cursor action |
|---|---|---|
| Canonical import, identity resolution and coverage validation | `ingest/legacy.py`, `reconcile.py`, `domain/`, fixtures and tests | Extend existing services and identity decisions; preserve quarantine and provenance |
| B1 repair and reproducible accepted snapshot | `docs/rewrite/evidence/b1/`, offline hash verification and replay | Reuse archived payloads; do not repeat repair fetches or manually patch input CSVs |
| Analytics and legacy ranking parity | `analytics/`, `ranking_legacy_v1.toml`, parity suite | Preserve formulas, deterministic ties and documented identity corrections |
| Training, forecasting, evaluation and model card | `ml/`, `pipeline.py`, `docs/model-card.md` | Fix the concrete temporal/spec defects below inside this implementation |
| Full CLI and application composition | `cli.py`, `pipeline.py`, `publish/builder.py` | Keep names, flags, exit codes and working output layout; fill behavioral gaps |
| Compact public data | `StatColumns`, `PlayerGameColumns`, `BoxScoreColumns`, Python/TS stat helpers and round-trip tests | Preserve layouts and equivalent values; avoid expanding to verbose per-cell objects |
| Live row contract repair | `LivePlayerRow`, updated monitor and generated validators | Keep the repair; fix static-site update semantics separately |
| Astro/React site | Existing routes, accessibility states, watchlist, filtering, charts and export helpers | Improve specific interactions and add absent views within the existing app |
| Strict typing and generated contracts | Web fixes, generator and schema checks | Run existing gates; no weakening TypeScript or adding blanket ignores |
| Installation, demo layout and absolute release-directory fixes | `8fd0817bd`, `208a54e21`, operations and CI | Do not reimplement or re-report them as unresolved |
| Release time/memory improvements | `e68c28de3` through `26b3ebd20`: streaming, bounded processes, batched lookup, parallel validation | Preserve these optimizations; measure process-tree memory and thread contention |
| Phase 8 rehearsal | `f1d6f164e`, rerun recorded in `7d9eb0905`, JSON evidence and scripts | Review and reuse; repeat only checks affected by later changes |
| Rollback across schema changes | Integrity-only activation of a previously semantically validated artifact | Preserve this guarantee while extending validation to the exact final site |
| Safe bounded refresh controls | `95be5f61f`: `--new-matches-only`, `--max-requests`, `--proxy` | Retain explicit request accounting; coordinate with the existing source-update task |
| Phase 9 switch plan | `docs/rewrite/SWITCH_PLAN.md` | Extend its shadow/switch/hooks sequence and required smoke checks |

The user-provided verification reports reproduced model metrics and a roughly 292.8 MiB real artifact on another machine. The latest branch ledger reports three uncontended release builds at 56.9–57.8 seconds and 1.39–1.43 GiB across the process tree. These are **existing reported measurements**, not fresh full-corpus benchmarks performed in this review. Cursor must retain their evidence and remeasure after relevant changes.

The 1,575-second forecast run was contended by concurrent training in integration tests. The ledger records 105.7 seconds for uncontended **training only**; the second-machine 108 seconds included training and replay. Do not compare them as identical workloads or reopen a disproved training-speed defect.

## 3. Reconcile the original review before changing anything

The IDs below refer to [the initial review](REVIEW_2026-09-27.md). Status is scoped to `2aad17873`.

| Finding | Current status | Next action |
|---|---|---|
| R01: final site differs from validated public data | **Reproduced again.** Added `site/private.json` is uploaded with a successful receipt | Seal and validate the exact upload tree, keeping cross-contract rollback |
| R02: model from the future used in replay | **Reproduced again.** A later-trained/calibrated synthetic bundle scores 19 earlier targets | Enforce bundle knowledge eligibility at every forecast cutoff |
| R03: persisted feature configuration ignored | **Reproduced again.** Training window 2 becomes inference window 5; 20 rows differ | Restore the exact persisted `FeatureSpec` and verify its fingerprint |
| R04: late observation breaks chronology | **Reproduced again.** A late availability timestamp causes `FeatureOrderError` | Select eligible revisions, then sort by event time |
| R05: live dict/list validation crash | Fixed on branch; focused tests pass | Preserve `LivePlayerRow`; keep regression |
| R06: stale public stat contracts | Branch has regenerated compact contracts and compatible consumers | Verify generated-file drift and a Python-to-browser value test; change only remaining gaps |
| R07: web type errors | Branch records clean checks and includes the reset/type fixes | Rerun on the chosen implementation revision; do not recreate the old patch |
| R08: missing CLI/pipeline | Implemented | Exercise existing commands; add only missing behavior |
| R09: machine-local import fixture | Replaced with repeatable import and B1 evidence | Reuse fixtures; classify real-data tests correctly |
| R10: existing upload directory trusted | Still present in inspected publisher | Reject mismatched/partial existing destinations before activation |
| R11: unlisted resource copying | Relevant integration file is unchanged | Copy the closed inventory; test unexpected files and final-site closure |
| R12: inferred dates shown in game logs | Still present in inspected query/serialization | Prefer linked match date and precision, retain original evidence separately |
| R13: season-span search includes gaps | Still present in inspected filter | Export exact season membership and test a gap year |
| R14–R15: wrapping and repeated technical banners | Relevant layout/styles retain the reviewed behavior; latest real-site visual check still needed | Check current screenshots; make focused layout changes |
| R16: immutable live resource polling | Relevant component unchanged | Use archived snapshot semantics and optional new-release notification |
| R17: key collisions and duplicate authorities | Key encoding and separately authored demo generator remain | Introduce a compatible key migration and Python-produced integration fixtures |

Legacy findings L01–L13 are requirements to verify against the replacement, not instructions to rewrite legacy modules. Use the acceptance mapping in the completion specification.

### Checks newly run on the latest branch

- Focused pytest command covering `test_live.py`, `test_cli.py`, `test_release.py` and `tests/scvia/contract`: **59 passed**, 6 warnings, 14.67 seconds. [Test output](evidence/review-2026-09-27/latest-focused-tests.txt).
- R01–R03 reproduction against exported branch source: all three defects remain. [Captured output](evidence/review-2026-09-27/latest-reproductions.txt).
- R04 reproduction: `FeatureOrderError: season order disagrees with date order for a player`.
- The existing full real import/train/build and browser suites were not rerun against this later revision during planning. Distinguish the original checkout's failing tests from Claude's branch reports and these focused confirmations.

The [portable reproduction script](evidence/review-2026-09-27/reproduce_open_findings.py) accepts a checkout path and writes synthetic artifacts only to a new temporary directory. Run it with that checkout's installed Python dependencies. Its output is diagnostic; convert each case into the regression tests described by C1/C2.

The synthetic earlier replay demonstrates a missing guard. It does not by itself invalidate the branch's particular 2026 replay metrics, whose cutoffs must be inspected separately.

## 4. Open work, in order

1. Pin the continuation revision and reconcile ownership with any active Claude session. Preserve its newest commits and the dirty source checkout.
2. Verify existing rehearsal evidence, then fix publication sealing and model/time correctness with regression tests. Repeat the affected publish/rollback cases.
3. Close data/browser integration and UX gaps while preserving compact contracts and passing behaviors.
4. Preserve the build gains; reduce artifact growth and fast-tier cost. Current site headroom is about 7 MiB, so three more seasons plus new content will not fit unchanged.
5. Prepare and smoke-test Phase 9 using the existing switch plan. Keep source completeness, shadow-cycle evidence and actual activation decisions visible.

The latest ledger records a pending current-season final-match source update and an existing owner-managed check. Do not duplicate that network job, create another schedule or infer a completed season. Offline tests and application fixes can proceed using archived fixtures.

## 5. Reuse proof required from Cursor

For every completed phase, record:

```text
Base revision:
Existing modules and tests reused:
Existing evidence accepted, with input/code hashes:
Changed behavior and finding/acceptance IDs:
Files added/changed, with reason:
Commands, exit codes and evidence paths:
Remaining gaps and next phase:
```

Use the existing implementation ledger on the continuation branch. Do not erase its history or label reported results as independently measured. Never make a phase “pass” by deleting a regression test, loosening a budget, dropping data or replacing real-source acceptance with demo results.
