---
name: datasentinel-gate-traps
description: Predictable ways a recap/methodology paragraph re-triggers DataSentinel's untagged-number check even after the main recap passes
metadata:
  type: feedback
---

When writing the `## Round N — Week in Review` section of `docs/afl-insights.md` (or any gated doc), a methodology/sources footnote is a common re-trigger point for DataSentinel Pass 2, even after the main recap paragraphs (disposal leaders, ladder, watch-list, tactical note) pass clean.

**The trap:** restating a figure that was already correctly `**[data]**`-tagged earlier in the doc, inside the "here's where these numbers came from" footnote — e.g. "the r = +0.74 correlation", "585 eligible players", "the 8-point gap (56 − 48)". The restatement is a *new* bare number occurrence and DataSentinel's untagged-number scan flags it independently of the first, tagged occurrence.

**Why:** DataSentinel's check is occurrence-based, not concept-based — tagging a number once in the doc does not immunize a later plain-text repetition of the same number.

**How to apply:** a methodology/sources footnote should **name source files only** — `docs/afl-stat-leaders-2026.md`, `docs/afl-finals-2026.md`, `data/matches/matches_2026.csv`, prediction CSV filenames — and describe *what kind* of figure came from where ("the contested-possessions/clearances correlation", "the top-two points gap") without repeating the value. If a number must appear in a methodology sentence, it needs its own `**[data]**` tag even if it duplicates an already-tagged figure elsewhere in the doc — but the simpler fix is to just not restate it.

Fixed 2026-07-13 in `docs/afl-insights.md` Round 20 recap (see [[coach_anonymity_lint]] for the sibling lint on the other recurring gate trap, coach naming).
