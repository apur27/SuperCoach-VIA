# Cursor Agent + Grok 4.7 implementation plan

## Start here

Continue Claude Code's **`rewrite/wip` implementation**, using **`grok-4.7-high`** in Cursor Agent. Preserve its code, tests, source repairs, compact data, completed rehearsal and build optimizations. This plan prepares the work; no implementation agent was launched while writing it.

Read these in order:

1. [Claude work reuse and revision reconciliation](CLAUDE_WORK_REUSE_2026-09-27.md).
2. [Master agent handoff](AGENT_HANDOFF_2026-09-27.md).
3. [Completion specification](REWRITE_PLAN_2026-09-27.md), including A01–A25 acceptance checks.
4. The **branch versions** of `IMPLEMENTATION_STATUS.md`, `REHEARSAL.md`, `SWITCH_PLAN.md`, `docs/migration.md`, `docs/operations.md` and `docs/model-card.md`.
5. [Initial review](REVIEW_2026-09-27.md), using the reuse document to distinguish closed findings from current gaps.

## 1. Verified tool and model

On 27 September 2026:

- Installed executable: `/home/abhi/.local/bin/cursor-agent`.
- CLI version: `2026.09.23-86fc751`.
- `cursor-agent --list-models` returned `grok-4.7-low`, `grok-4.7-medium`, **`grok-4.7-high`**, `grok-4.7-xhigh`, plus Fast variants.
- Use `grok-4.7-high` explicitly for implementation and review. Do not use `auto`, an invented `cursor-grok-4.7` identifier, or silently fall back to another model.

The exact CLI/model observation is saved in [tool evidence](evidence/review-2026-09-27/cursor-tooling.json). Cursor's [Grok 4.7 announcement](https://cursor.com/blog/grok-4-7) confirms the release; the installed model list supplies the usable identifier. Recheck CLI help and model availability at launch because flags and account availability can change.

## 2. Workspace preparation

Use one dedicated continuation worktree based on the newest reviewed `origin/rewrite/wip`. The dirty `main` checkout and Claude's verification worktree remain intact. Do not run Cursor's automatic `--worktree` option from `main`: it starts from the wrong HEAD and does not carry the untracked implementation or data changes.

Preparation commands below are for the later implementation run, not commands already executed by this plan:

```bash
cd /home/abhi/git/SuperCoach-VIA
git status --short
git worktree list
git fetch origin rewrite/wip
git log -8 --oneline origin/rewrite/wip
git rev-parse origin/rewrite/wip
# Choose a new, unused destination and branch; if either exists, inspect it and resume safely.
git worktree add -b cursor/rewrite-grok47 /tmp/supercoach-via-grok47 origin/rewrite/wip
```

If the fetched revision is later than `2aad17873`, inspect its diff and ledger first. Mark newly completed work as reused; do not apply stale patches. Record the exact chosen SHA in the continuation ledger. If Claude is still writing the same areas, establish a handoff before those edits; use the time for read-only review. Do not kill another session or run competing writers against shared snapshots/models/releases.

Copy **only these new planning documents and their review evidence** into the continuation worktree. Do not overlay the old `src/`, `web/`, schemas, config, lockfiles or implementation ledger:

```bash
SCVIA_PLAN_SOURCE=/home/abhi/git/SuperCoach-VIA/docs/rewrite
SCVIA_CURSOR_WORKSPACE=/tmp/supercoach-via-grok47
mkdir -p "$SCVIA_CURSOR_WORKSPACE/docs/rewrite"
cp "$SCVIA_PLAN_SOURCE/REVIEW_2026-09-27.md" \
   "$SCVIA_PLAN_SOURCE/REWRITE_PLAN_2026-09-27.md" \
   "$SCVIA_PLAN_SOURCE/CLAUDE_WORK_REUSE_2026-09-27.md" \
   "$SCVIA_PLAN_SOURCE/CURSOR_GROK47_PLAN_2026-09-27.md" \
   "$SCVIA_PLAN_SOURCE/AGENT_HANDOFF_2026-09-27.md" \
   "$SCVIA_CURSOR_WORKSPACE/docs/rewrite/"
mkdir -p "$SCVIA_CURSOR_WORKSPACE/docs/rewrite/evidence"
cp -R "$SCVIA_PLAN_SOURCE/evidence/review-2026-09-27" \
   "$SCVIA_CURSOR_WORKSPACE/docs/rewrite/evidence/"
```

Compare source manifests before importing. The branch's committed corpus plus archived B1 repair is a reproducible starting point. The dirty `main` corpus may have additional accepted updates: inventory and reconcile any differences into a new candidate, retaining both originals. Never silently select the older corpus, copy files over another snapshot, or fetch the same repair again.

Worktrees share Git metadata but should have separate `var/`, `dist/`, virtual environments, logs and run locks. Do not point two writers at a shared `SCVIA_DATA_ROOT` or output root. Keep the existing verification worktree until its evidence is no longer needed by the owner.

## 3. Launch and resume

### Recommended interactive launch

After workspace preparation, this starts one agent that can progress through all phases without a new prompt after each successful gate:

```bash
cursor-agent \
  --workspace /tmp/supercoach-via-grok47 \
  --model grok-4.7-high \
  'Read docs/rewrite/AGENT_HANDOFF_2026-09-27.md and execute C0 through C8 in order. Continue Claude Code’s work. Reuse completed work and verify each phase gate. Checkpoint after each phase and continue through all locally executable work. Leave a tested local result and explicit operational gates.'
```

Implementation uses the default agent mode. `--mode plan` and `--mode ask` are for planning/review and should not be used for the implementation invocation.

### Headless alternative

The installed CLI supports `--print`, `--auto-review` and structured output. Use its normal tool review controls; a permission rejection is a checkpoint, not a reason to add a bypass flag. These flags are taken from local `--help`; the launch itself has not been exercised in this planning task.

```bash
cd /tmp/supercoach-via-grok47
mkdir -p var/agent-runs/cursor-grok47/run-001
cursor-agent --workspace "$PWD" --model grok-4.7-high \
  --print --auto-review --output-format stream-json \
  'Read docs/rewrite/AGENT_HANDOFF_2026-09-27.md. Execute C0 through C8, preserving Claude Code’s work. Verify each gate and checkpoint progress. Do not commit, push, deploy or activate schedules.' \
  > var/agent-runs/cursor-grok47/run-001/events.jsonl \
  2> var/agent-runs/cursor-grok47/run-001/stderr.log
```

Use a new run directory on each launch so logs are not overwritten. A wrapper must capture the process exit status separately from any log formatting. Check the emitted session/model metadata before accepting the run; if it resolves to a different model, stop and fix selection. Do not treat a JSON “result” message as evidence that repository tests passed.

Resume the specific recorded chat, avoiding an ambiguous global `--continue`:

```bash
cursor-agent --workspace /tmp/supercoach-via-grok47 \
  --model grok-4.7-high --resume '<recorded-chat-id>' \
  'Read docs/rewrite/CURSOR_PROGRESS.md and var/agent-runs/cursor-grok47/HANDOFF.md. Verify the last completed gate, reconcile newer changes and continue with the first incomplete phase.'
```

If the chat cannot resume, start a fresh session with the same prompt and workspace. Repository checkpoints are authoritative. Use `--mode ask` for a fresh read-only review after implementation:

```bash
cursor-agent --workspace /tmp/supercoach-via-grok47 \
  --model grok-4.7-high --mode ask \
  'Review the changes since the base SHA recorded in docs/rewrite/CURSOR_PROGRESS.md against A01–A25 and the Claude reuse inventory. Read code and evidence. Check publication sealing, temporal eligibility, compact-contract parity, regressions and switch gates. Do not edit. Report concrete findings with files, triggers and severity; distinguish unrun checks.'
```

## 4. Phase instructions

The master agent executes these serially. A phase can be reopened when later evidence invalidates its gate. Reuse is the default in every phase.

### C0 — establish the continuation baseline

Read applicable repo instructions and the branch documents. Record Git status, branch SHA, source/config/lock hashes, installed runtimes and model identifier. Compare the latest ledger with the reuse inventory. Make a `reused / fix / missing / needs evidence` entry for each area. Retain the completed Phase 8 JSON, archived B1 evidence, existing metrics and build measurements.

Install with `uv sync --locked --group dev --group legacy --extra ml` and `npm --prefix web ci`. Run schema/type checks and focused smoke tests sufficient to establish the current state. Do not immediately run an expensive real train/build just to duplicate unchanged evidence. Create `docs/rewrite/CURSOR_PROGRESS.md` and ignored run logs.

**Gate:** precise baseline, safe writer ownership, input preservation, completed work identified, concrete open findings. **Next:** C1.

### C1 — seal exactly what gets published

Work mainly in `publish/release.py`, the builder composition, `web/integrations/release-tree.mjs` and their existing tests. Add failing cases for unchecked `site/`, unlisted public files, altered/partial existing destinations and changed sealed bytes. Add the final-site seal and bind validation/upload/receipt to it. Update build/preview/local publish composition so each step consumes an explicit artifact reference.

Keep the old-contract rollback test. Preserve semantic validation at build time and integrity validation at activation. Use a local destination only. Test locks, interruption, upload/activation failure and existing-directory retries. Update the rehearsal script to use sealed final sites; no extension-based recursive copying.

**Gate:** A11–A13; no unchecked tree can activate; valid old sealed release can roll back. **Next:** C2.

### C2 — correct forecast time and feature semantics

Work in existing `ml/{features,bundles,train,predict,evaluate}.py` and ML tests. Start with the R02–R04 reproductions. Reconstruct the complete persisted feature spec. Add model eligibility that includes training, calibration and model-selection knowledge. For replay choose an eligible bundle; explicitly reject a too-new supplied bundle. Select historical revisions by cutoff, then order eligible records by event time.

Preserve current feature calculations where correct, baseline/challenger structure, interval provenance, caching and metrics. Add non-default specs, same-day boundaries, late corrections and future-model cases. Determine whether any existing published evaluation needs relabelling or rerunning; do not infer invalidity solely from the guard defect. Run training-heavy tests serially.

**Gate:** A06–A09; deterministic valid-case parity, forbidden future influence rejected, truthful model card. **Next:** C3.

### C3 — integrate the existing data and browser contracts

Reuse compact types and access helpers. Make a Python-produced demo/sample part of browser integration acceptance, with value assertions. Fix linked-match date display, exact season membership and collision-free keys with compatibility for existing links/watchlists. Retain or migrate stored IDs explicitly. Regenerate schemas/types/validators together and require no drift.

Inspect existing CLI/resume/config/packaging behavior against specification §13. Extend only missing pieces. Use archived B1 sources. Add correction/partial-source and request-cap cases where not covered. Do not duplicate the owner's pending source-update job.

**Gate:** A01–A04, A10, A15, A22–A23 for touched boundaries; existing CLI commands and compact round-trip tests stay green. **Next:** C4.

### C4 — complete product flows and improve readability

Inspect the current real site before editing. Use existing components, styles and route shells. Fix ordinary-word/date wrapping, simplify repeated provenance banners, retain clear source/freshness/forecast labels, improve mobile tables and expose full data through accessible expansion/export.

Add era summary and Brownlow proxy browser views from Claude's existing exported facts. Keep the proxy label. Verify shareable filters, sorting, browser history, comparisons and watchlist storage/recovery. Change `/live/` to honest snapshot semantics with an optional new-release notification. Keep stable paths and existing accessibility behavior.

**Gate:** A14–A20; screenshots at required widths/themes plus keyboard/zoom checks, root and subpath, no console or validation errors. **Next:** C5.

### C5 — preserve speed and create size headroom

Profile the changed final release without competing jobs. Keep the existing spawned season pool, streaming player output, batched lookups and parallel validation. Measure total process-tree RSS. Do not introduce extra workers that exceed the four-thread reference profile.

Use Claude's season-growth script and budget tooling. Prioritize repeated game-log identifiers/metadata and duplicated assets; consider a per-resource dictionary or shared match metadata with bounded requests. Preserve values, offline downloads, integrity and direct links. Choose the smallest measured change that provides three additional seasons plus 5 MiB content growth under 300 MiB; at the existing growth rate the current site target is ≤282 MiB. Do not drop history or simply raise limits.

Move actual full-corpus tests out of the hermetic tier; keep fast synthetic regressions for their behavior and retain full integration coverage. Profile repeated fixture setup rather than hiding tests with skips.

**Gate:** A21; original ≤60-second/≤2-GiB release build and browser transfer budgets, documented growth projection, hermetic tier ≤30 seconds on the reference profile. **Next:** C6.

### C6 — run the full system and update evidence

Run the final locked installation, static checks, Python/browser suites, real import/validation, model evaluation if affected, final-site build, size/security checks and fresh wheel test. Ensure both URL bases work and read actual Python output. Keep expensive jobs serial and measure stages separately.

Reuse Phase 8 scripts and documented intentional differences. Repeat the affected old/new comparison and sealed local publish/rollback/restore sequence against the final candidate. Verify the preserved corpus digest before and after. Update runbook, migration registry, model card, ledger and final evidence index. Do not normalize away unexplained result changes.

**Gate:** A05, A20, A22, A24–A25 plus all earlier gates on final inputs; local review artifact ready. **Next:** C7.

### C7 — prepare the Phase 9 switch safely

Extend `SWITCH_PLAN.md` and retain its written CLAUDE.md §6.2 scope decision. Prepare thin numeric wrappers, explicit manifest selection, portable hooks and CI changes. Keep legacy scripts/data recoverable. Build and test in the continuation/scratch worktree, away from the live harness.

Use fixture-backed source operations, inert external publication and a local destination in smoke tests. Verify missing-fixture success, partial-source failure, snapshot/release preservation, local publish/rollback, root/subpath build and budgets. Write rollback instructions for the entry-point change itself. Keep the freeze check and smoke merge condition.

Record any outstanding real shadow cycles, final-source completeness and production decisions. Prepare the work before presenting an activation choice. Do not switch schedules, push or deploy during this local implementation run.

**Gate:** reviewable switch change and scratch smoke evidence; external gates clearly separated. **Next:** C8.

### C8 — review and close

Use a fresh Grok 4.7 read-only review session with the command above. Review against the pinned base and acceptance IDs, not only the implementation summary. Bring concrete findings back to the single writer; fix them and rerun affected gates. The reviewer cannot approve its own unrun tests by inference.

Produce a final summary: reused work, changed behavior, exact validation results, performance/growth numbers, local release path, remaining operational conditions and next operator command. Distinguish completed local implementation from a live production cutover.

## 5. Checkpoints, interruptions and resource use

Maintain a concise `docs/rewrite/CURSOR_PROGRESS.md` with base SHA, current phase, per-phase status, reuse decisions, acceptance IDs, evidence paths and unresolved conditions. Keep raw logs and detailed temporary handoffs under ignored `var/agent-runs/cursor-grok47/`. Update `HANDOFF.md` after each gate and before a context reset:

```text
Model and chat ID:
Workspace and base SHA:
Current phase / next concrete action:
Files changed and reason:
Tests run / exact results / logs:
Existing Claude work preserved:
Input and artifact hashes:
Running child processes owned by this run:
Outstanding failure or external condition:
```

- One writer, one phase and one resource-heavy benchmark at a time. Additional model calls are limited to the explicit review step.
- A rate limit or usage limit leaves an incomplete checkpoint. Respect retry timing; do not spawn replacement writers, busy-loop or silently switch models.
- Stop only the processes owned by this run. Do not delete locks held by another process.
- Do not load the entire corpus, generated validators or long old transcripts into the model context. Read named modules, summaries and failing tests; preserve large evidence in files.
- A denied tool action must be reported with the action and reason. Continue independent safe work; do not turn on `--force`, `--yolo`, blanket MCP approval or a disabled sandbox to bypass it.
- A changed base/input hash invalidates the relevant gate. An agent's exit code 0 without test evidence does not complete a phase.

## 6. Final acceptance

Accept the implementation only when the completion specification's §16 handoff exists and every required local acceptance check has evidence. Remaining production/source/shadow conditions must be explicit. The final diff should show focused continuation of Claude's code, including fixes and justified additions; a replacement scaffold is a failure to follow this plan.
