---
name: harness-gap-status-words
description: When reviewing harness/architecture gap-resolution claims, "closed" is almost always premature — verify the control is actually live on disk
metadata:
  type: feedback
---

When the council declares a harness gap (pre-commit hook, audit log, CI gate) "closed"/"resolved" in `docs/ai-architecture.md`, treat the status word as the primary thing to audit.

**Why:** On 2026-05-30, Gap 1 (`.githooks/` + `core.hooksPath`) was presented as closed, but on disk: `core.hooksPath` was still `.git/hooks`, `.githooks/pre-commit` was untracked AND non-executable, and `CONTRIBUTING.md` had no setup step. A gate that fails *open silently* (no error when unwired) is the worst safety-control failure mode. Gap 2's `log-agent-turn.sh` emitted only `ts/agent/action/files/verdict` — far less than the doc's own recommendation (`model_version, prompt_hash, tool_calls, retrieved_files, output_hash, latency_ms, token_count`), and was opt-in/self-reported, so it cannot answer the forensic questions it was filed to fix.

**How to apply:**
- Verify control state directly: `git config core.hooksPath`, `git ls-files`, `stat` for exec bits, `grep` the doc that's supposed to carry the setup step.
- A committed `.githooks/` dir creates the *appearance* of a repo-travelling gate while enforcement still depends on a per-clone manual `git config` nothing checks — flag this.
- Local hooks are bypassable (`--no-verify`); only CI is real enforcement. If the claim is "enforced" but there's no CI workflow, BLOCK.
- Self-reported/opt-in logging is NOT observability/accountability — it cannot distinguish "didn't happen" from "happened but wasn't logged." Reject Principle-8/Accountability claims built on it.
- Recommended honest status words: "built, not yet wired" (artifact exists, enforcement off) and "MVP-live, not production-grade" (runs but captures a subset). Gaffer is process-boss not truth-boss — record *progress*, not *enforcement that isn't on*.
- Executable bit must be committed via `git update-index --chmod=+x` or it dies on every clone.
