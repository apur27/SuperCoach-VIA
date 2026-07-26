---
name: skeptic-ladder-trend-overreach
description: A single-round ladder snapshot cannot support comparative ("extended their lead") or season-wide superlative ("largest gap of the season") claims — these need round-by-round history, which the weekly recap's cited source does not contain
metadata:
  type: feedback
---

Sibling trap to [[skeptic_tactical_note_overreach]], seen in the Round 21 recap's
"Ladder:" sentence in `docs/afl-insights.md` (Skeptic finding SK-R21-04, third BLOCK
on this doc before the fix landed).

**The trap:** `docs/afl-finals-2026.md` (and the `## Round N — Week in Review` ladder
paragraph that cites it) only ever holds the *current* round's ladder — one snapshot
of W-L/points/percentage per team, no history of prior rounds. Two phrasings look
harmless but are unsupported by that source:
- A **comparative**: "have extended their lead" / "have pulled away" / "the gap has
  grown" — asserts the gap at an earlier round was smaller. Requires a second
  data point (the prior round's gap) that a single snapshot doesn't have.
- A **season-wide superlative**: "the largest gap of the season to date" / "their
  biggest lead all year" — requires comparing against *every* prior round's gap,
  not just this one.

**Why it kept slipping through:** both phrasings sound like natural football
commentary and don't introduce a new bare number (no new digit to catch on a
tag-scan), so they read as safe. But Skeptic checks claims, not just numbers —
a claim of trend or record requires the same evidentiary backing as a number would.

**Fix pattern:** report the snapshot as a **position**, not a **trend**. Replace
"have extended their lead to X, the largest gap of the season" with "sit atop the
ladder at X — a commanding buffer at the top of the table" (or equivalent neutral
magnitude language). Keep every already-tagged figure exactly as-is; only strip the
comparative verb and the superlative clause. Do not hedge ("appears to be the
largest") — a hedged unsupported claim is still unsupported and will re-BLOCK.

**Route not taken (and why):** the repo's `data/matches/matches_2026.csv` could in
principle support a full round-by-round ladder recompute to substantiate the
comparative honestly, but that is a new derived-data analysis outside the existing
pipeline (nothing in the repo currently produces or gates a round-by-round ladder
history) — disproportionate to a one-sentence recap fix and better routed through
Scientist/DataSentinel as its own piece of work if the user wants trend language in
future recaps, not improvised ad hoc by FootyStrategy under a three-strikes deadline.
