---
name: feedback-superlative-and-jersey-collision
description: Two failure modes that survive a DataSentinel number-by-number PASS — cross-club superlative ranking contradictions, and jersey-map collisions on duplicate surnames
metadata:
  type: feedback
---
On list-quality / draft-pipeline articles that rank clubs off a derived column (e.g. avg games per pick), two defects pass DataSentinel but are Skeptic scope:

1. **Superlative ranking contradictions.** Each tagged cell (84.6, 84.8, 81.2) verifies correct, but the prose "highest in the competition / no other club comes close / second highest behind X" is computed by eye and contradicts the table and other sections. Always re-sort the column yourself and check every "highest/second/no other club" claim against it. Real example (2026-06-17 list-quality article): GWS prose said 84.6 was "the highest... no other club comes close" while Melbourne's own row was 84.8; Hawthorn said "second behind GWS" when it was third behind Melbourne and GWS.

**Why:** DataSentinel checks each [data] tag against source; it does not cross-check interpretive ranking claims across sections.
**How to apply:** Whenever an article ranks clubs on a column, rebuild the sort from the master table and diff every comparative/superlative against it.

2. **Jersey-map collisions on duplicate surnames.** The squad-union method maps jersey numbers to a 2026 player row by name; two players sharing a surname (e.g. two Bailey Williamses — born 1997 WB half-back 187cm, born 2000 WC ruck 199cm) can collapse to one record, putting an identical (draft year, pick, grade, games) row under two clubs. The number still "reconciles" because it matches one real file. Scan for identical cross-club rows; verify against data/player_data personal_details (born_date, debut_date, height) which disambiguate.

**Why:** confirms the "jersey-to-player mapping" limitation is load-bearing, not theoretical.
**How to apply:** grep squad tables for repeated (player, pick, games) tuples across clubs; treat any duplicate-surname pair as guilty until the born/debut/height splits them.
