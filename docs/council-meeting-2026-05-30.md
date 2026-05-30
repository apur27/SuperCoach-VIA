# Council Meeting Minutes — 2026-05-30

> [← Back to AI architecture](ai-architecture.md) · [← Back to main README](../README.md)

**Chair:** Gaffer (Delivery Lead / orchestrator)
**Date:** 2026-05-30
**Session focus:** Gap 1 (pre-commit gate portability) + Gap 2 (LLM-turn audit log) governance closure, working-tree cleanup, provenance hygiene.

## Attendees

| Agent | Role | Present |
|-------|------|---------|
| Gaffer | Chair — process owner, ships on PASS only | Yes |
| Scientist | Data analysis, statistical derivation | Yes |
| BriefBuilder | Data-skeleton assembly, `[data]` tagging | Yes |
| DataSentinel | Pre-commit verification gate (PASS/FAIL) | Yes |
| Skeptic | Adversarial review (PASS / PASS_WITH_CONCERNS / BLOCK) | Yes |
| FootyStrategy | Tactical interpretation, confidence tiering | Yes |
| Codex | External / provider-separated blind read | Yes — filed `docs/council-codex-notes-2026-05-30.md` |

Codex's outside-the-frame perspective was received this session (not asynchronous), filed as a standalone note rather than inline.

## Agenda items resolved

### 1. Gap 1 — pre-commit gate portability
A committed `.githooks/pre-commit` now exists, and CONTRIBUTING.md instructs operators to run `git config core.hooksPath .githooks` on clone so the wiring travels with the repo rather than living local-only at `.git/hooks/`. The hook continues to call the committed `scripts/check-council-stamp.sh`, which refuses any council doc whose `<!-- council-pipeline: ... -->` stamp is missing or shows a non-PASS DataSentinel/Skeptic verdict. **Status: MVP-live** — real, but operator must run the one-time `core.hooksPath` config; not zero-friction, declared MVP not production-grade.

### 2. Gap 2 — LLM-turn audit log
`.claude/audit/` is created and `scripts/log-agent-turn.sh` is available for manual/scripted per-turn logging to `.claude/audit/YYYY-MM-DD.jsonl`. **Status: MVP-live** — the logging surface exists but is not yet wired into the harness to fire automatically on every turn; declared MVP not production-grade.

### 3. Draft file cleanup
Three intermediate working papers under `docs/news/` — `2026-05-15-carlton-next-coach-data.md`, `2026-05-15-carlton-next-coach-footystrategy.md`, and `2026-05-15-richmond-vs-stkilda-r11-data.md` — were removed. **Rationale:** these were per-agent draft fragments (Scientist data layer and FootyStrategy interpretation) whose content has been fully merged into published, fully-councilled articles. They were untracked working products, so removal is non-destructive to history and clears clutter from the news desk.

### 4. README badge
FootyStrategy has filed a recommendation on the README status badge. **Decision: deferred** — the badge update is folded into the next `refresh_readme.py` cycle rather than hand-edited this session, to keep the badge in sync with the auto-generated front-page blocks.

## Decisions logged

- **Gap 1 and Gap 2 marked MVP-live (not "closed").** Decided by the council on Skeptic's insistence on honest language: these shift status from "absent" to "partial/live" but are not production-grade. Recorded in the `ai-architecture.md` gap table dated 2026-05-30.
- **Draft fragments removed.** Decided by Gaffer (chair authority over presentation surface); merged work products, safe to delete.
- **Badge update deferred to next refresh.** Decided by Gaffer in line with FootyStrategy's filing.

## Open items carried forward

- **Gap 2 auto-wiring.** `scripts/log-agent-turn.sh` is not yet invoked automatically by the harness on every agent turn — manual/scripted only. Carry forward to wire into the turn loop.
- **Gap 1 hardening.** The hook trusts the recorded stamp rather than re-deriving the DataSentinel verdict against source CSVs. Codex flagged this as the key remaining weakness: a hand-edited or compromised stamp passes the gate. Promote from passive (trust recorded) to active (re-run verification) in a future cycle.
- **README badge update** — next refresh cycle.

## Gaffer's "ready to ship" declaration

The Gap 1 and Gap 2 implementations are **ready to ship as declared MVP-live**. The artifacts referenced by the gap-table status text were verified present in the tree before the status was written: `.githooks/pre-commit` committed, CONTRIBUTING.md documents the `core.hooksPath` step, `.claude/audit/` exists, and `scripts/log-agent-turn.sh` exists. No DataSentinel FAIL or Skeptic BLOCK was raised on this session's documentation changes. The honest "MVP-live, not production-grade" framing is preserved per the Skeptic's standard. Ship approved for the gap-table update, the draft cleanup, and these minutes.

<!-- council-pipeline: Gaffer:SHIP@2026-05-30 — governance/minutes doc, no [data] tags; not a council-authored data article -->
