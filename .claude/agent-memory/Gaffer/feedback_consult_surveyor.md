---
name: consult-surveyor
description: Standing user directive — consult the Surveyor agent on any complex/difficult issue BEFORE committing a solution; it is the advisory resource for the sprint and beyond
metadata:
  type: feedback
---

Before committing a non-trivial solution, consult **Surveyor** (read-only advisory diagnostician) and incorporate its risk flags. This is a standing user directive, "not optional."

**Why:** In Sprint 1 the Surveyor caught real defects before ship — the F2 vacuous Skeptic sampler (a private regex missed `**[data: spec]**` tags), the F4 working-tree-vs-staged-blob hole, and the F6 afl-insights brick risk. Its design consult on F2/F4/F6 shaped safer implementations. An adversarial read before commit repeatedly paid for itself.

**How to apply:**
- Put the specific design question to Surveyor (via SendMessage to a warmed instance, or dispatch the `Surveyor` agent type), collect the read, then implement.
- Surveyor advises/never operates; its findings are advisory but the *directive to consult* is mandatory. A FAIL/BLOCK from DataSentinel/Skeptic/QA still overrides everything (Surveyor never re-litigates a verdict).
- Availability is session-dependent (see [[agent-dispatch-is-session-dependent]]); if the `Surveyor` type isn't registered, run its read via a capable agent under the Surveyor spec (`.claude/agents/Surveyor.md`), honestly framed.
