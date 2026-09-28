# Opus 5.5 payload — review and improve the finalized app

For the owner's later request to use **Claude Code** for a thorough review and implement a deterministic data-integrity checker, use [the Claude Code payload](CLAUDE_CODE_OPUS_55_REVIEW_PAYLOAD.md). This file preserves the earlier Cursor handoff.

Prepared after the reviewed app was merged into local `main` on 28 September 2026. This file is a ready-to-use task for the next agent; creating it did not launch Opus.

## Launch

Use Cursor Agent with **Opus 5.5 High**, model ID **`claude-opus-5-5-high`**, confirmed in the installed Cursor model list on 28 September 2026. If the model is unavailable in a later session, report that rather than silently selecting a different model.

```bash
cursor-agent --workspace /home/abhi/git/SuperCoach-VIA --model claude-opus-5-5-high --print --auto-review "Read docs/rewrite/OPUS_55_PAYLOAD_2026-09-28.md and carry out its task. Preserve the merged Claude and Grok work."
```

In the Cursor UI, select Opus 5.5 and give it that same instruction.

---

## Task for Opus

Review the finished SuperCoach VIA app, then fix concrete remaining defects in performance, usability, security and code quality. Build on the merged implementation. Prioritize failures a user can encounter, correctness of displayed football data, and reliable local operation. Deliver focused, tested improvements and an updated operator handoff. Do not start another rewrite or repeat work already proven by the recorded checks.

The architecture is a **static Astro/React/TypeScript site plus a local Python pipeline**. Keep that architecture. Keep Claude's canonical data model, identity resolution, analytics, ranking formulas, repair evidence, compact contracts, model training/evaluation, CLI, release builder and rollback semantics. Preserve Grok's publication sealing, temporal model eligibility, feature fingerprints, shared match facts, collision-safe keys, archived live view, history tables and operational fixes.

### 1. Establish the actual starting state

Repository: `/home/abhi/git/SuperCoach-VIA`.

Merged application commit: **`2bcdfbeb4142c98bf120e5056c06f3f142515c3f`**, descended from Claude's `2aad178730990aad2623bd06cb0abfb17c5b0987`. Documentation commits may follow it. Confirm that commit is an ancestor of the current branch, inspect status and recent history, and preserve unrelated work. Do not reset the checkout to the old base.

Read these in order:

1. `CLAUDE.md`, particularly tests, data verification, harness freeze, scratch smoke and guarded commits.
2. `docs/rewrite/FINALIZATION_2026-09-28.md` and `docs/rewrite/finalization-20260928/README.md`.
3. `docs/rewrite/finalization-20260928/independent-review.txt` and the current artifact metadata.
4. `docs/operations.md`, `docs/migration.md`, `docs/model-card.md`, `docs/rewrite/SWITCH_PLAN.md`.
5. `docs/rewrite/CLAUDE_WORK_REUSE_2026-09-27.md`, `CURSOR_GROK47_PLAN_2026-09-27.md`, and the original rewrite/review plans for context. Their historical open findings are not automatically current defects.

The recovery worktree is `var/worktrees/cursor-rewrite-grok47`. Its work is merged. The original `/tmp/supercoach-via-grok47` is gone. The saved pre-merge stash and backup contain older application files; do not apply them over the current app. Avoid running concurrent training, integration or performance jobs.

### 2. Use the correct real data and site

Local retained paths:

- Dataset, models and source captures: `var/finalized/data/`.
- Sealed release: `var/finalized/releases/20260928T111014Z-ca96603163ad/`.
- Preview alias: `var/final-site`.
- Verified deployment archive: `var/finalized/site.tar`.
- Artifact identity and copy checks: `var/finalized/metadata.json`.

Snapshot: `sha256:aa836549e10e96a9039cc4642bbe563b94247e0bfaf95e745a0971cc0bb2b98f`.

Seal: `bd9a655b06795c2d83ca77a48adb9df85fb2774463fa390f814dac2ff72474fa`.

Archive SHA-256: `59496b26d421998b874ea879277b63365bc49fa1427c95902565bc4abc9599f7`.

The grand final is now present in this snapshot and site. All source player rows and statistics were verified, including null values, and player/match navigation passed a real browser check. Read the committed verification reports before doing another fetch. The legacy `data/matches/matches_2026.csv` is deliberately still pre-final migration input. Do not interpret that CSV alone as the current app state or manually update statistics in generated files.

The historical corpus retains `legacy_unverified` provenance. Current grand-final verification does not certify the whole imported corpus. The season's `schedule_complete` and `source_status` remain unknown. Forecasts are legitimately unavailable because no valid future fixture exists. Published holdout metrics were not recomputed; do not relabel them as prospective forecasts.

These large artifacts are ignored by Git. If working in a fresh environment, say which local inputs are absent and rebuild or request transfer through the documented pipeline. Keep existing evidence separate from measurements you actually repeat.

### 3. Inspect the app before choosing changes

Start the sealed real site:

```bash
node web/scripts/serve.mjs --dir var/final-site --port 4321
```

Check the overview, search, player detail, match detail, comparison, predictions, history, downloads, watchlist, live snapshot and data-status pages. Use a phone width and desktop, keyboard navigation, light/dark themes, and the existing no-JavaScript checks. Follow real grand-final match/player links. Look for broken states, misleading labels, slow interactions or unnecessary downloads before proposing a redesign.

For each finding, record the affected path, a reproducible trigger, the user impact, and the smallest correction. Reproduce code defects with a failing regression test before implementation. Prefer a few important, complete fixes over unrelated cleanup. If no material defect remains, deliver that conclusion with evidence instead of manufacturing changes.

### 4. Preserve these behavioral guarantees

- Publication uses one captured inventory and a validated seal for the exact uploaded bytes. Missing, extra, changed or unsafe files must fail closed. Preserve integrity-only rollback of an already validated older release across schema changes.
- Partial/unreachable refreshes do not promote a snapshot. Successful unchanged fixture checks update freshness for checked seasons. Counts and dates come from merged matches; uncertainty is preserved.
- Fetched matches identify their actual source; legacy imports keep their label. Null statistics stay unknown. Player game logs join the shared match facts correctly.
- Canonical `k.` keys remain collision-safe and reversible; compatible legacy links still navigate. Season filters use actual season membership.
- Forecasts honor knowledge cutoffs, persisted feature order/specification and matching feature/code fingerprints. Keep honest unavailable states and holdout labeling.
- The live page shows a captured snapshot and does not poll an immutable release. Brownlow proxy data keeps its proxy label.
- Weekly execution has explicit source mode/network permission, ownership locks, stable snapshot selection, exit receipts and refusal of overlapping cycles.

Use existing modules under `src/supercoach_via/`, schemas under `schemas/`, and generated browser contracts. Generate types/validators through the existing commands; do not patch generated contracts independently.

### 5. Verify proportionately and retain evidence

Locked setup:

```bash
uv sync --locked --group dev --group legacy --extra ml
npm --prefix web ci
```

Use the pinned Node version. Run affected regressions first. Required Python gates are Ruff, mypy and the full hermetic tier:

```bash
uv run --locked ruff check
uv run --locked mypy
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 uv run --locked pytest tests/scvia -m 'not integration' -q -n 4 --dist loadscope
```

Current baseline: 695 passing hermetic tests, 29.33s on CPU IDs `0,2,4,6` (four distinct physical cores). IDs `0–3` are two physical cores with SMT and took 34.15s. Inspect CPU topology before comparing timings. The target stays 30s; the result is hardware-specific. Do not remove tests, weaken assertions or raise budgets to make a result pass.

Run affected integration tests. For browser changes, run generated-contract checks, type checks, lint, Vitest and the relevant production Playwright suites at both `/` and `/SuperCoach-VIA/`. Baseline: 320 Playwright passes with four recorded skips, and 163 browser-unit passes with two recorded skips. The Python-to-browser release bridge also runs in pytest. Keep old and new reports distinguishable.

For release/data changes, build a new release in a new output directory, attach the matching Astro site, seal and validate it, verify the pack round trip, and check real grand-final resource values and navigation. `SCVIA_RELEASE_DIR` must be an absolute path ending in the release's `public/` directory. Use the same base throughout build and preview.

Current real site is 276,821,039 bytes. Three seasons at the recorded mean plus 5 MiB project to 280.46 MiB. Keep the 300 MiB budget and the 282 MiB growth target. Release build targets remain 60s and 2 GiB process-tree RSS on the reference setup. Measure thread count, CPU placement and contention explicitly; do not run training alongside benchmarks.

Every harness/gate change must satisfy `CLAUDE.md` section 6.2: final-source scratch weekly run, captured input, local publish/failure/rollback, and source inventory comparison. Freeze relevant source while it runs. The existing final smoke was 620s with zero source differences; its input was the captured pre-final corpus. The committed reports explain the distinction.

Use `scripts/git_commit_safe.sh` for commits with the appropriate `COUNCIL_PYTHON`. Preserve all hooks and gates. Keep reviewable changes isolated from unrelated work.

### 6. Production work still has prerequisites

No remote deployment or schedule activation has occurred. `SCVIA_NUMERIC_ENTRY=1` is opt-in; the legacy default harness remains. Two genuine shadow cycles and the owner decisions in `SWITCH_PLAN.md` remain before production activation. An offline replay is not a live shadow cycle.

Prepare the commands, acceptance evidence and rollback plan locally. Do not invent elapsed cycles, silently switch defaults, dispatch a deployment, push a live publish, or alter schedules under this payload. Obtain the owner's deployment instruction after presenting the concrete candidate if that is the next requested step. Do not launch other paid agents or models merely to repeat this review.

### 7. Completion handoff

Write `docs/rewrite/OPUS_55_RESULTS.md` with:

- Starting commit and artifact IDs; Claude/Grok work retained.
- Findings resolved, their regressions and user-visible behavior.
- Exact checks run, results, artifact paths, and comparisons with the baseline.
- Reproducible launch instructions from the resulting checkout.
- Honest remaining local defects and external activation prerequisites.
- Commit/merge state, and an explicit statement of any deployment or schedule action.

Stop when the selected fixes and their required gates are complete. Give the owner the working app and a concise handoff, without claiming production activation from local checks.
