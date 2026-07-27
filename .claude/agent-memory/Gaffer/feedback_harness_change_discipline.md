---
name: harness-change-discipline
description: Never change the harness during an active cycle, and never merge a harness diff without an end-to-end smoke run — unit tests cannot see the failures that matter
metadata:
  type: feedback
---

Two standing rules, adopted by the user 2026-07-28 and written into `CLAUDE.md` §6 so
the whole council reads them, not just me.

**1. Freeze harness/gate/hook changes while a cycle is in flight.** Queue the hardening;
let the NEXT cycle validate it. Check `.claude/audit/last_refresh_status.json` — it is
written on every exit, so "is a cycle active?" is a file read.

**2. Smoke-run any harness diff end to end before merging it**, in a scratch worktree
against the prior week's snapshot with commit/push stubbed. Green smoke run is the merge
condition, including for one-line fixes.

**Why:** 2026-07-27/28. I gated `docs/afl-backtest-2026.md` and moved the integration
tier to Phase 3d *while a cycle was running*. Both changes were correct in isolation and
both are still in the tree. Between them they caused three failed relaunches and ~7 hours,
because each one invalidated something the in-flight cycle depended on:
- gating the doc without a re-verify hop → regeneration orphaned its recorded PASS;
- the integration tier at Phase 0b → it asserted published artifacts match source data,
  which is false mid-cycle by construction, so it blocked the run that would have fixed it;
- the re-verify hop itself verified the INDEX copy while DataSentinel hashed the WORKTREE
  copy, so it reported success against bytes that were not being shipped.

**How to apply:** every one of those was a few lines and every one passed unit tests. They
are ordering and staged-vs-worktree faults, visible only when the whole harness runs. So:
if a diff touches the harness, it does not merge on a green unit suite — it merges on a
green end-to-end run. And if a cycle is running, it does not merge at all yet.

The corollary I keep relearning: a gate that blocks is usually working. Three times this
week the "failure" was a gate correctly refusing unverified content, and the defect was
always upstream in what I had just changed.

Related: [[gates-hardening-20260725]], [[long-pipeline-monitoring]], [[open-backlog]] (BL-10, BL-11).
