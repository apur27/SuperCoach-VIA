---
name: prove-with-the-right-file
description: Before citing a file comparison as evidence, confirm the file actually contains the quantity in question — and prefer isolating the variable over inferring from inputs
metadata:
  type: feedback
---

When claiming "X did not change", verify the file you compared actually CARRIES X. A
"no fields changed" result from a file that never held the quantity is not evidence of
stability — it is evidence of nothing, and it reads as proof.

**Why:** 2026-07-28, BL-04. To show a chart's byte drift was rendering rather than data,
I diffed `all_time_top_100.csv` between the chart-era commit and now, got "changed: none"
on all ten rows, and published that as proof the data was stable. That file is the BIO
file — Serial Number, Player Name, Footy Teams, Comment. It holds no scores at all, so
the comparison could not have detected a data change under any circumstances. The scores
live in a separate file, and Surveyor found 17 of 100 rows there had in fact changed
since June. My conclusion (rendering drift) was correct; my evidence was worthless.

**The method that actually settles it:** isolate the variable instead of inferring from
inputs. Surveyor rendered the OLD data vintage in the CURRENT environment and got a
byte-identical result to today's render. That proves "renders identically regardless of
data vintage" directly, rather than arguing it from an input diff.

**How to apply:**
- Before using a file diff as evidence, print its columns and confirm the quantity is in
  there. One `list(df.columns)` would have caught this.
- Prefer re-running the transform across the two conditions over comparing the inputs.
  Inputs can differ in ways that do not reach the output, and vice versa.
- Beware evidence that is convenient. "changed: none" on ten rows looked decisive and
  matched what I already believed, which is exactly when to check it hardest.
- State the claim the evidence supports, not the claim you set out to make: here,
  "renders identically across data vintages", not "the data never changed".

Related: [[verify-routed-findings]], [[route-every-finding]].
