---
name: feedback-methodology-paragraph-untagged-restatement
description: Methodology paragraphs often restate stat-shaped numbers (sample sizes, correlation coefficients, arithmetic gaps) in plain prose with no tag — these still count as untagged numbers and must be flagged even though the same value is correctly tagged elsewhere in the body.
metadata:
  type: feedback
---

When a doc's methodology paragraph explains *where* numbers came from, authors frequently restate the numbers themselves inline (e.g. "585 eligible players", "the r = +0.74 correlation", "the 8-point gap (56 − 48)") without a `**[data]**` tag — because the paragraph is prose/explanation, not a headline claim. These still count as player/season-stat-shaped untagged numbers per the hard rule (untagged numbers are FAIL, not opt-in via tagging), even when:
- the exact same value is correctly tagged earlier in the document body, or
- the number is a sample-size / correlation / derived-arithmetic restatement rather than a "new" claim.

**Why:** DataSentinel's job is mechanical tag-walking, not judgment about whether a restatement is "redundant" — a future edit could change the methodology number without touching the tagged instance, silently diverging. Flagging both closes that drift path.

**How to apply:** When scanning a methodology paragraph, treat it exactly like body prose for the untagged-number scan (step 6) — do not give it a pass just because it's "explaining sources." Round numbers/labels ("Round 19", "rounds 1–19") remain structural and are never flagged; specific counts, correlation coefficients, percentages, and point gaps are always flagged if untagged, wherever in the doc they appear.

Seen in: `docs/afl-insights.md` Round 20 Week-in-Review methodology paragraph (2026-07-13) — "585 eligible players", "r = +0.74" (restated), "8-point gap (56 − 48)" (restated) all untagged despite the tagged instances elsewhere passing.
