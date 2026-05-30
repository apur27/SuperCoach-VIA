---
name: enforcement-substrate-state
description: On-disk state of Gap 1 (pre-commit gate) and Gap 2 (audit log) substrate — both MVP-live as of the 2026-05-30 council session
metadata:
  type: project
---

State of my own enforcement substrate. Updated 2026-05-30 after the council session that marked both gaps MVP-live.

**Gap 1 (pre-commit DataSentinel gate): MVP-live.**
- A committed `.githooks/pre-commit` now exists and invokes `scripts/check-council-stamp.sh`. CONTRIBUTING.md (line 26) instructs the operator to run `git config core.hooksPath .githooks` on clone, so the wiring travels with the repo. NOT zero-friction (operator must run the one-time config) — gap-table status reads "MVP-live, not production-grade" dated 2026-05-30.
- `scripts/check-council-stamp.sh` is a DETERMINISTIC grep (no LLM). It gates ONLY `docs/news/*.md` and `docs/hall-of-fame-stat-*.md`; it SKIPS `docs/news/README.md` (the index) and everything else. On a staged council doc it requires a `<!-- council-pipeline: ... -->` stamp whose DataSentinel: and Skeptic: lines both contain PASS, else exit 1.
- STILL OPEN (Codex flagged 2026-05-30): the hook trusts the *recorded* PASS in the stamp; it does not re-run DataSentinel against the source CSVs. A hand-edited or compromised stamp passes. Promote from passive (trust recorded) to active (re-derive verdict) is the next hardening step. Backlog, not a blocker.

**Gap 2 (LLM-turn audit log): MVP-live.**
- `.claude/audit/` now exists and `scripts/log-agent-turn.sh` is available for manual/scripted per-turn logging to `.claude/audit/YYYY-MM-DD.jsonl`.
- STILL OPEN: not yet wired into the harness to fire automatically on every turn — manual/scripted only. Gap-table status reads "MVP-live, not production-grade" dated 2026-05-30. Auto-wiring is carried forward as an open item.

**Why:** My PREFLIGHT contract checks for these two before SHIP. Both are now present at the documented paths, so routine doc/presentation cycles can proceed. Do NOT over-claim "closed" — both are MVP-live with named open halves (Gap 1 re-verification, Gap 2 auto-wiring).

**How to apply:** Before a publication cycle, treat the gate as live for non-`[data]` doc work. The two open halves are backlog items, not blockers. If editing the gap table again, preserve the honest "MVP-live, not production-grade" language (Skeptic's standard) — never "closed."

See [[backlog-priorities]] once the 11-gap triage is written.
