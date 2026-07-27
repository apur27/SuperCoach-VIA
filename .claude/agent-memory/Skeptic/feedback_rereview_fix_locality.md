---
name: rereview-fix-locality
description: On re-reviews, fixes land only on the flagged line — re-grep every other place the same concept is stated (glossary rows, intro thesis sentences, "Read:" glosses)
metadata:
  type: feedback
---

On a re-review after a BLOCK, do **not** scope the check to the lines the previous
findings named. Fixes in this repo land on the exact flagged line and nowhere else,
so a corrected claim routinely ends up contradicting an uncorrected restatement of
the same concept elsewhere in the doc.

**Why:** observed on `docs/afl-backtest-2026.md` (2026-07-27 re-review). The bias-sign
fix was applied to the cumulative table and the team-bias preamble, but the
plain-English glossary row still read "a bias of −0.7 means we predict 0.7 too high" —
the inverse, in the row most likely to be read by the fan audience. Same pass: a new
accountability paragraph disclosed ~302 outright misses while an un-updated "Read:"
line 64 lines later still said "there is no fat tail of catastrophic mispredictions",
and a methodology fix establishing "the training set is unchanged for every round"
left an earlier sentence asserting "the model gets more data per player each round".
Three separate self-contradictions, all created *by* the fixes.

**How to apply:** for each fixed finding, grep the whole doc for the *concept*, not the
line — every definition row, every intro/thesis sentence, every "Read:"/"What to look
for:" gloss, and every plain-English restatement. Rank the glossary and the intro
highest: they are written for the least expert reader and are the least likely to be
touched by a targeted fix. A doc that contradicts itself is a BLOCK even when both
halves were individually reviewed and one half is new.

**Locality is measured in the GENERATOR, not the rendered doc — and it is tighter than
you expect.** Confirmed on the pass-3 re-review of `docs/afl-backtest-2026.md`
(2026-07-27), where the operator had correctly moved all four fixes into
`update_team_analysis.py`. The S4 fix landed at generator L4347; the sentence it
contradicts ("the model is trained on all data **before** that round", false against
`prediction.py:589`) sat at generator L4341 — *six lines above, in the immediately
preceding `parts.append()`*. Fixes are applied to the string that was quoted in the
finding and to nothing else, even when the defect is in the adjacent paragraph of the
same function. **How to apply:** once you locate a fixed string in the generator, read
±40 lines of surrounding `parts.append()` calls in full before clearing the finding.
Also: when a prior pass quoted *two* instances of one error, re-check both by name —
here instance 2 was fixed and instance 1 shipped three passes running.

**Corollary — new accountability paragraphs orphan the reassurance surfaces that
predate them.** Each honest disclosure added to that page (302 outright misses at L62,
the top-30 disaggregation at L126) left its fan-facing counterpart untouched: the intro
rule-of-thumb still graded the season "strong", the glossary still taught "a bias near 0
is ideal", the hit/miss definitions still called a miss anomalous. Pattern: **a fix that
adds a caveat rarely propagates upward to the surface that taught the reader how to
interpret the thing being caveated.** Grep the glossary/intro for the metric named in
every new caveat.

**One rendered doc can have MORE THAN ONE generator — grep every one before clearing a
finding.** Pass-4 on `docs/afl-backtest-2026.md` (2026-07-27): the S5 training-window fix
was correctly moved into `update_team_analysis.py` (root, NOT `scripts/`) "so it survives
regeneration" — but the same page's cumulative table is written by a *different* script,
`scripts/update_eval_surface.sh`, whose L372 still glossed the design in plain English in
a column literally headed "What it means". A fix's durability claim only covers the
marker block its generator owns. **How to apply:** for each rendered doc, first map every
`<!-- X-START -->` marker to the script that rewrites it (grep the marker name across
`*.py` and `*.sh`), then run the concept-grep across all of them. Also: the same-function
±40-line rule still held — the weaker restatement sat 3 lines below the fix.

**Grade "true but strictly weaker" as CONCERN, not BLOCK.** Same pass: the corrected
sentence said "never on any part of the season being scored"; the next one said "the
model is not fitted on the round it is predicting". The second implies by omission the
exact misconception the fix existed to kill, but states nothing false. BLOCK stays
reserved for a claim that is actually false against the code. Say plainly why it was not
promoted, or the operator reads the concern as a soft block.

Related: [[feedback_accountability_surface_audits]],
[[feedback_recap_tactical_note_causal_relapse]].
