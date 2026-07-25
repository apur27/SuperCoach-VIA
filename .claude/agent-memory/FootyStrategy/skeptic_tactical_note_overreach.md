---
name: skeptic-tactical-note-overreach
description: Three recurring overreach patterns Skeptic blocks on in the weekly afl-insights.md "Tactical note" sentence — identity-from-correlation, variable substitution, unscoped superlatives
metadata:
  type: feedback
---

Skeptic blocked the Round 21 `docs/afl-insights.md` tactical note (line 28) on three patterns worth checking every time a correlation is turned into a football claim:

1. **Identity asserted from correlation.** A strong r (even +0.75) supports "strongly associated" or "strongest correlate of," never "the same skill expressed twice" / "is" / "are identical." Correlation coefficients bound the *strength* of a relationship, not its *mechanism* or *sameness*.
2. **Variable substitution.** Don't cite a correlation between variables A and B, then draw a conclusion about variable C (e.g. citing contested-possessions-vs-clearances r, then concluding something about "midfield disposal volume" — a third variable the cited stat never measured). Check that the noun phrase in the conclusion clause is literally one of the two variables in the cited r.
3. **Unscoped dataset-wide superlatives.** "The clearest predictor" / "the dataset's cleanest evidence" claims global superlative status. Before writing any superlative, check whether the source doc (`docs/afl-stat-leaders-2026.md`'s "Top per-game correlates" lists) actually contains a *higher* r elsewhere — it usually does (effective_disposals/uncontested_possessions pairs routinely hit 0.8–0.97). If the r is only the top correlate *within that one stat's own card* (e.g. contested_possessions is clearances' #1 listed correlate, ahead of handballs/disposals on that specific card), scope the superlative to that: "the strongest correlate of clearance output reported in the stat leaders' correlation breakdown," not "the dataset's cleanest evidence."

**How to apply:** before finalizing any "Tactical note" or similar interpretive sentence built on a correlation figure, re-read the source doc's correlate listings for that stat and ask (a) am I claiming identity instead of association, (b) does my conclusion clause name the same variable that's in the cited r, (c) is my superlative scoped to what the source actually supports. See [[datasentinel_gate_traps]] for the sibling DataSentinel-side trap (methodology footnotes restating tagged numbers) that fires on the same paragraph.
