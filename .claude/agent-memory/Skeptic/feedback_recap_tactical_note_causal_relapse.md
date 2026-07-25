---
name: recap-tactical-note-causal-relapse
description: Weekly-recap "Tactical note" rewrites tend to relapse into causation via the verb, plus season-aggregate figures described as "this round" — check the verb and the window, not the disclaimer
metadata:
  type: feedback
---

In `docs/afl-insights.md` weekly recaps, the **Tactical note** line is the highest-risk sentence in the doc and survives rewrites in weakened-but-still-defective form. Three tells, all seen on the Round 21 rewrite after a prior BLOCK:

1. **Disclaimer present, causation re-imported through the verb.** The rewrite added an explicit "a strong association, not evidence the two are the same skill", then closed with "...whose stoppage-contest work is **translating directly into** clearance production." The hedge and the claim are in the same sentence; the hedge does not neutralise the verb. Audit the *predicate* of the final clause, not the hedge that precedes it. Causal verb-phrases to grep for: "translating into", "driving", "producing", "converting into", "feeding", "turning X into Y".
   **Why:** a correctly-scoped population r followed by an individual causal verb is the exact drift the caveat hierarchy forbids, and it reads as compliant because the disclaimer is right there.
   **How to apply:** quote the final clause on its own, stripped of the hedge, and ask whether the cited evidence (a Pearson r + two leaderboard ranks) supports it. It almost never does.

2. **Individual-as-example-of-a-correlation.** "Player X leads both stats, making him the clearest example of the relationship" is not evidence of the relationship — co-leadership of two leaderboards is two independent facts. Counter-test that is cheap and decisive: compute the ratio for the named player vs the #2/#3 on the *dependent* leaderboard. On R21, Oliver was 7.94/15.61 = 0.51 while Newcombe (7.72 clearances, outside the contested-poss top 5, so ≤13.24) converts at ≥0.58 — the same cited source contradicts the "clearest example" superlative.

3. **Window mismatch: season aggregates narrated as "this round".** Recap figures are rounds 1–N per-game aggregates; phrases like "this round" / "this week" attach them to an unplayed round. Same defect class appears in the Ladder paragraph ("**extended** their lead", "largest gap **of the season to date**") when the cited source is a single end-of-round ladder snapshot with no round-by-round history. Both are untagged qualitative claims, so DataSentinel passes them — verify the *named methodology source* actually contains a time series before accepting any "extended / largest / first time / of the season" phrasing.

Related: [[superlative-and-jersey-collision]], [[news-caveat-prominence-and-ambiguous-players]].
