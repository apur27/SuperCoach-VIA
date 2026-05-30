---
name: enforcement-substrate-state
description: Actual on-disk state of the Gap 1 (pre-commit gate) and Gap 2 (audit log) enforcement substrate as of 2026-05-30
metadata:
  type: project
---

State of my own enforcement substrate, verified 2026-05-30 during the team-intro task.

**Gap 1 (pre-commit DataSentinel gate): PARTIALLY present, not at the documented path.**
- A working pre-commit hook exists at `.git/hooks/pre-commit` and invokes `scripts/check-council-stamp.sh`. It ran on my intro commit and printed `council-stamp check: 0 council doc(s) checked, 3 skipped, 0 failed.` — so it is live and it does skip non-council docs.
- BUT it is NOT at the documented `.githooks/pre-commit` with `core.hooksPath` set. `core.hooksPath` resolves to `.git/hooks` (the default), and `.githooks/` does not exist. That means the hook is local-only, not version-controlled, and not portable to a fresh clone or to other agents' environments.
- Confirmed 2026-05-30 by reading `scripts/check-council-stamp.sh`: it is the council-stamp/provenance check, NOT the full DataSentinel `[data]`-tag re-verification. It is a DETERMINISTIC grep (no LLM). It gates ONLY `docs/news/*.md` and `docs/hall-of-fame-stat-*.md`; it explicitly SKIPS `docs/news/README.md` (the index) and everything else (README.md, other docs/, CSV, Python). On a staged council doc it requires a `<!-- council-pipeline: ... -->` stamp whose DataSentinel: and Skeptic: lines both contain PASS, else exit 1. So Gap 1's `[data]` re-verification half is still OPEN — the hook only enforces that a PASS was *recorded*, not that the numbers actually match the CSV.

**Why:** My PREFLIGHT contract says to refuse SHIP if `.githooks/pre-commit` is absent. The literal path IS absent; an equivalent at `.git/hooks/pre-commit` exists. Do not over-claim "Gap 1 closed" — the portability/version-control half is still open.

**How to apply:** Before any publication cycle, (1) read `scripts/check-council-stamp.sh` to confirm what it actually checks, (2) decide whether the local-only `.git/hooks/pre-commit` satisfies the gate or whether it must be moved to version-controlled `.githooks/` with `core.hooksPath`. Treat the version-control gap as a backlog item, not a blocker for non-`[data]` presentation/doc work.

**Gap 2 (LLM-turn audit log): ABSENT.** `.claude/audit/` does not exist. Per ai-architecture.md §13.4 this is the highest-leverage security investment. Top of the backlog.

See [[backlog-priorities]] once the 11-gap triage is written.
