---
name: recap-tactical-note-causal-relapse
description: Weekly-recap "Tactical note" rewrites tend to relapse into causation via the verb, plus season-aggregate figures described as "this round" — check the verb and the window, not the disclaimer; the terminal failure mode is hedge-stacking
metadata:
  type: feedback
---

In `docs/afl-insights.md` weekly recaps, the **Tactical note** line is the highest-risk sentence in the doc and survives rewrites in weakened-but-still-defective form. Tells, all seen across four passes on the Round 21 recap:

1. **Disclaimer present, causation re-imported through the verb.** The rewrite added an explicit "a strong association, not evidence the two are the same skill", then closed with "...whose stoppage-contest work is **translating directly into** clearance production." The hedge and the claim are in the same sentence; the hedge does not neutralise the verb. Audit the *predicate* of the final clause, not the hedge that precedes it. Causal verb-phrases to grep for: "translating into", "driving", "producing", "converting into", "feeding", "turning X into Y".
   **Why:** a correctly-scoped population r followed by an individual causal verb is the exact drift the caveat hierarchy forbids, and it reads as compliant because the disclaimer is right there.
   **How to apply:** quote the final clause on its own, stripped of the hedge, and ask whether the cited evidence (a Pearson r + two leaderboard ranks) supports it. It almost never does.

2. **Individual-as-example-of-a-correlation.** "Player X leads both stats, making him the clearest example of the relationship" is not evidence of the relationship — co-leadership of two leaderboards is two independent facts. Counter-test that is cheap and decisive: compute the ratio for the named player vs the #2/#3 on the *dependent* leaderboard. On R21, Oliver was 7.94/15.61 = 0.51 while Newcombe (7.72 clearances, outside the contested-poss top 5, so ≤13.24) converts at ≥0.58 — the same cited source contradicts the "clearest example" superlative.

3. **Window mismatch: season aggregates narrated as "this round".** Recap figures are rounds 1–N per-game aggregates; phrases like "this round" / "this week" attach them to an unplayed round. Same defect class appears in the Ladder paragraph ("**extended** their lead", "largest gap **of the season to date**") when the cited source is a single end-of-round ladder snapshot with no round-by-round history. Both are untagged qualitative claims, so DataSentinel passes them — verify the *named methodology source* actually contains a time series before accepting any "extended / largest / first time / of the season" phrasing.

4. **"Already fixed" is a claim to verify, not accept.** On the R21 third pass the operator stated the line-28 note had been "rewritten and cleared" — the text on disk was byte-identical to the version I had BLOCKed. The Ladder fix *had* landed, so the report was half-true, which is what made it credible. **How to apply:** on any re-review, diff the actual quoted clause against the defect you logged before reading the operator's summary of what changed; one paragraph landing does not imply the others did.

5. **Terminal failure mode is hedge-stacking, not causation.** Once the causal verb is finally removed (R21 pass 4), the sentence carried *two* disclaimers — "a strong association, not evidence the two are the same skill" and "...a season-to-date pairing that shows association, not that either stat is driving the other". The second one still mis-assigns evidentiary weight: an appositive on one player's two figures cannot "show association" (n=1); only the population r can. The correct predicate for the individual is *consistent with* / *illustrates*, never *shows* / *demonstrates* / *is evidence of*.
   **Why:** the rewrite loop converges on adding hedges rather than fixing the subject of the claim, so each pass reads safer while the epistemic error migrates into the hedge itself.
   **How to apply:** after confirming no causal verb, check what the *subject* of each evidentiary verb is. Population statistic → "shows/indicates" is fine. Named individual → must be illustrative language only. Also flag the redundancy: one caveat per claim; two is a tell the paragraph has been patched rather than rewritten.

Related: [[superlative-and-jersey-collision]], [[news-caveat-prominence-and-ambiguous-players]].
