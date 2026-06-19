---
name: multiclub-completeness-check
description: For docs claiming "all N clubs"/full-ladder coverage, cross-check table-row count + classification tally against the computed ladder; a skipped ladder position is an easy, high-impact miss
metadata:
  type: feedback
---

When a multi-club strategy/preview doc claims completeness ("All 18 Clubs", "every club"), do NOT trust the prose tally. Recompute the ladder from `data/matches/matches_<year>.csv` (W/L/D, pct, margin) and check three things against the table:

1. **Row count** — does the table have one row per club, and are all ladder positions present? A skipped position (e.g. table jumps 11th → 13th) means a club was dropped.
2. **Classification tally** — does the summary count (e.g. "9 Window Now, 6 Building, 3 Rebuilding") sum to the claimed club total AND match the number of rows actually carrying each label? A phantom count (9 claimed, 8 rows) is the tell that a deleted/never-written club is still being counted.
3. **Orphan mentions** — grep the omitted club's name; a single vague aside ("Collingwood-style veteran exits") with no section/row is a placeholder that survived an edit.

**Why:** The 2026-06-19 5yr-GF-strategy doc passed DataSentinel (every [data] number correct) yet silently omitted Collingwood (12th). Title said "All 18", window count said 9+6+3=18, but the table had 17 rows / 8 Window-Now. DataSentinel verifies numbers that ARE present; it does not catch a missing entity. That gap is exactly the Skeptic's job.

**How to apply:** Run the ladder recompute as the first move on any all-clubs doc. It is cheap, idempotent, and catches a misleading-completeness BLOCK that number-verification structurally cannot.
