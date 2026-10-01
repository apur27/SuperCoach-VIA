# AFL Tables reconciliation engineering task for Scientist

Run as the existing **Scientist** agent from `.claude/agents/Scientist.md`. Use your
existing data/code ownership, methodology rules, TDD requirements and project memory.
This document supplies the task; it does not replace your registered agent definition.
Use **Sonnet 5.5** (`claude-sonnet-5-5`), the user's explicit model override for this task
only; leave the stored default unchanged. Verify the
session's selected model and record requested/resolved model information. Report provider
restrictions honestly; do not silently switch to an alias or another model.

Implement and execute `docs/rewrite/afltables-reconciliation/DESIGN.md`. The deliverable
is working deterministic software, full-source reconciliation evidence and an honest
data verdict. Reading files, writing a plan, passing a sample or starting a background
command alone does not complete the task.

## Entry checks

1. Read `CLAUDE.md`, your agent definition and memory index, applicable local instructions,
   the design and both task payloads. Read the relevant memories about AFL Tables URL
   identity, career counters, era coverage, missing player rows and blank statistics.
   Re-verify their claims against current code and captures: the older recommendation
   to compare career totals alone is insufficient for this explicit game-by-game audit.
2. Require `docs/reviews/afltables-reconciliation-design-review.json` to say APPROVED
   with no blocking findings and an actual independent Surveyor review. Recompute the
   approved design, payload and relevant agent-definition hashes. Inspect
   code changes since the reviewed HEAD; documentation-only approval commits need not
   invalidate it, but changes to relevant behavior require architect review.
3. If approval is absent or stale, report exactly what is missing. Do not invent approval
   or run the full source acquisition before the architecture is accepted.
4. Record current branch, HEAD, dirty files, interpreter versions, current pipeline
   marker and relevant active processes. Confirm the candidate snapshot and retained
   reference identities from the design. Inspect actual pointers rather than using a
   convenient old preview as the app's current data.
5. Keep at most one implementation branch/worktree, shared sequentially by these roles.
   Prefer `work/afltables-reconciliation` under `var/worktrees/afltables-reconciliation`.
   Reuse an existing compatible task worktree instead of creating numbered branches.
   Preserve unrelated work and large ignored verification artifacts.

## Implementation

Follow phases B–F and TDD. Extend the independent source reader and reuse verified storage,
transport and report primitives where appropriate. Preserve existing APIs and behavior.
Implement the proposed CLI, strict contracts, census, resumable capture, identity resolution,
independent comparisons, aggregate checks, complete findings and coverage accounting.

Do not reuse the production importer or blank resolver as an oracle. Do not select a source
or identity because it agrees with the database. Do not use model calls inside the program.
Do not turn source failures into empty successful results. Historical mismatches must stay
visible and affect the strict requested audit verdict.

Use current supported executable discovery and pinned environments. The current remote
CI already has unrelated portability/configuration failures, including a hardcoded Node
path and mobile layout assertion. Record baseline failures separately and test the new
code in a working locked environment; do not waive tests that this implementation breaks.
Keep unrelated CI/browser repairs outside this feature unless required to execute its tests.

Keep fixtures small and representative. All unit tests must be network-free. Real full
captures and large reports belong under `var/reconciliations/`, outside Git and outside
the immutable candidate inputs. Add only small sanitized fixtures and evidence summaries
to version control. Do not skip/reclassify tests or increase a budget to obtain green output.

Apply CLAUDE.md section 6 by its actual scope. A new opt-in audit that does not affect
weekly phase ordering, artifact selection, staging or gates does not activate that harness.
If shared-code edits change those behaviors, freeze them during active cycles and perform
the required full scratch smoke before merging. Record your scope determination in writing.

## Execute the real audit

After fixtures and pilot pass, run the full commands from design section 11. The user is
asking for the full all-player comparison. The network capture is authorized when this
engineering payload is launched; respect the specified host rate/access policy and normal
tool permissions. No write to AFL Tables, source bypass, proxy rotation or remote publish
is part of this task.

For a long capture:

- record exact command, PID/session, stdout/stderr logs, plan ID, checkpoint and progress;
- confirm the process is actually alive before saying it is running;
- retain checkpoints and keep monitoring until completion, an access block, or an explicit
  user stop; do not confuse the agent ending its turn with the command completing;
- on interruption, give the exact resume command and outstanding resource counts;
- measure acquisition separately from offline comparison, and avoid concurrent training.

Run cold comparisons with one and four workers, then warm cache and changed-since on the
same frozen source collection. Compare canonical files byte for byte. Run mutation probes
on copies. Demonstrate offline operation with sockets denied. Handle data exits 4/8 as
verdicts requiring reports, not as permission to stop the remaining reproducibility work.

Run the existing full integrity checker against the same candidate snapshot and sealed
release with all required evidence/content inputs. Preserve its separate report. Audit
the legacy CSV layer explicitly; a corrected candidate PASS must not hide older CSV defects.
Verify the latest completed final's presence and each participating player's source rows.

Use evidence to resolve routine identity ambiguities. Any override needs an exact source
capture/locator and reason, and changes the plan/cache identity. If a genuine conflict
cannot be resolved, finish the other comparisons and report UNKNOWN for the affected
coverage. A website restriction or unavailable reference may make full execution incomplete;
do not claim all-player integrity from the reachable subset.

## Delivery and acceptance

1. Produce every artifact in the design, including a schema-valid completion file with
   separate design, implementation, execution and per-layer data statuses.
2. Write `docs/reviews/AFLTABLES_RECONCILIATION_RUN.md` with exact commands, versions,
   relevant code/input/capture hashes, coverage, findings, runtime/memory/disk measurements,
   source scope and D01–D18 status. Any actual player/match statistics in Markdown require
   the repository's evidence tags. Avoid a prose-only claim without machine-readable evidence.
3. Hand tested source, tests, docs and an explicit file allowlist to **Gaffer** for serialized
   commits through `scripts/git_commit_safe.sh`; Scientist does not commit or push. Preserve
   large inputs and reports locally; never directly commit pipeline outputs or change
   accepted data to force a passing report. Stop editing during Gaffer's commit operation.
4. Provide the exact **Gaffer/Opus** final-acceptance command. Gaffer commissions Surveyor
   and QA; that review must inspect final code
   and completed reports. Do not mark an unrun acceptance step passed.
5. After acceptance, the owner's standing request is delivery to main and minimal branch
   clutter. Gaffer owns merge/push through normal repository gates and preserving useful
   uncommitted work, removing the merged task branch, and retaining necessary audit artifacts.
   Do not delete divergent or dirty work without preserving its contents. There is no
   authorization here to deploy, switch the numeric harness or alter production statistics.

Corrections should be a source-backed proposal with exact affected records. Applying them
to accepted inputs is outside this audit. The report must remain useful when it proves
that data is wrong.

## Handoff and final answer

Use Scientist's existing Did / Found / Caveats / Didn't / Assumed response contract and
include the following evidence. Give Gaffer a direct handoff for review and delivery.

- Requested/resolved model, implementation commit and actual branch/delivery state.
- Implementation status and execution status separately.
- Primary snapshot, raw CSV and release verdicts separately.
- Exact player/game/cell denominators, comparison counts, gaps and findings by category.
- Whether the latest completed final and all participating players were checked.
- Four-run report hashes, mutation-test evidence and input immutability evidence.
- Timings and measured process-tree memory; missed targets and existing unrelated failures.
- Links to the full report, findings, completion file and Opus acceptance, or its pending command.
- Any running process with a verified PID, or explicitly say no process remains running.
