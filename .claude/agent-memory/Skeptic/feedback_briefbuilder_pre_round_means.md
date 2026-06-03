---
name: briefbuilder-pre-round-means
description: When auditing BriefBuilder pre-match briefs, player_data CSVs already contain the previewed round's row — verify season means exclude it AND that game-count matches
metadata:
  type: feedback
---

When spot-checking a BriefBuilder pre-match brief (docs/coaches-strategy-corner/), the player `performance_details.csv` files frequently ALREADY CONTAIN a row for the round being previewed (e.g. an R13 brief, but the CSV has an R13 row), and finals rows carry non-numeric round codes ('EF', 'QF') that break `int()` casting.

**Why:** On a WB-vs-Collingwood R13 2026 brief, a naive `df[year==2026]['disposals'].mean()` included the R13 result and produced means ~1 disposal higher than the doc's claimed pre-R13 figures — a false smoke-test scare. The doc was mostly correct; my query was wrong. After filtering to round<13 (handling 'EF'/'QF' as None), 6 of 7 players matched exactly. But the 7th (Daicos Nick) was a REAL mismatch: doc claimed "11 games / 33.8 mean" when the CSV had 10 pre-R13 games / 34.7 mean. So both error modes coexist — naive inclusion fakes a mismatch; a genuine game-count error hides among them.

**How to apply:**
- To verify a "through Round N, pre-R(N+1)" season mean, filter `round < N+1` and map non-numeric rounds to None before any int cast. Never trust a raw `year==season` mean for a pre-match brief.
- Verify BOTH the mean AND the stated game count (e.g. "11 games"). A mean that nearly matches can still sit on a wrong denominator — the count is the independent check that catches it.
- The last-5 mean can be correct while the season mean/count is wrong (Daicos case: last-5 33.4 matched, season 33.8/11 did not) — check them as separate claims.
- A failed spot-check here is CRITICAL: surface as "DataSentinel either not run or failed; full verification required before commit," do not fix the number yourself.
