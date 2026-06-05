---
name: no-agent-dispatch-tool
description: This session env has NO Agent/Task subagent-dispatch tool — the 6 council agents cannot be spawned; Gaffer runs single-operator cycles with deterministic gates instead
metadata:
  type: project
---

There is no `Task`/Agent subagent-dispatch tool exposed in this Claude Code environment. Searching `select:Task` and "agent dispatch council" returns nothing — only cron/worktree/monitor/remote-trigger deferred tools exist.

**Why:** My contract says "commission the council chain via the Agent tool; you do not author." But with no dispatch tool, the 7-agent chain (BriefBuilder -> DataSentinel x2 -> FootyStrategy -> Skeptic -> Codex) cannot be spawned. I am the only execution surface.

**How to apply:** When asked to orchestrate the council and no Agent tool is present, run a SINGLE-OPERATOR cycle that preserves the *substance* of every gate, and be transparent about it:
- Do all data derivation DETERMINISTICALLY in Python — every `[data]` number re-read from the CSV, never in-token arithmetic. This is DataSentinel-style verification, which is mechanical/auditable, not interpretive authorship — it stays inside the ONE RULE.
- Run a hard-coded DataSentinel pass that re-reads every tagged number and compares (claimed vs csv) with a PASS/FAIL table. On any FAIL, fix the doc and re-run — never ship around it. (2026-06-05 list-management-101 cycle: this caught two rank-tie overstatements — "ranked 3rd" was really "tied 3rd-4th" / "tied 2nd-3rd". Ties are the classic trap: a looser exploratory filter said one rank, the strict full-22 filter said another.)
- Run a Skeptic-style adversarial CONTROL test (e.g. premier vs league-average) to stop the narrative overclaiming, and document concerns in a Methodology section.
- STAMP HONESTLY: the provenance footer must say "single-operator cycle (no Agent dispatch tool available)", DataSentinel-self:PASS, Skeptic-self verdict — NOT a fabricated 7-agent chain. Faking the stamp violates the ONE RULE.

If the human wants the genuine multi-agent chain, escalate: the dispatch tool must be wired into this env first. See [[council-stamp-gate-scope]] and [[enforcement-substrate-state]].
