---
name: rereview-own-fixes
description: After folding reviewer findings into a design, send the amended text back for a confirmation pass before approving; Gaffer's own fixes introduced 2 HIGH contradictions in the 2026-10-01 AFL Tables design review
metadata:
  type: feedback
---

Never approve a design on the strength of your own edits that resolve review findings. Re-commission the reviewer (Surveyor) on the amended hash and bind approval to the exact text it approved.

**Why:** In the 2026-10-01 AFL Tables reconciliation design review, Surveyor run 1 raised S-01..S-16. Gaffer's fixes introduced two HIGH contradictions (a T14 row left contradicting the new §8 rule, and an aggregate partition with no source-unavailable bucket, which made PASS unreachable by construction), plus a wrong Brownlow career-average model. Run 2 caught all of them; run 3 confirmed. A LOW item raised at the final pass went into the review as a binding clarification, not a design edit, so the approved hash stayed the one the reviewer read.

**How to apply:** Fix findings, rehash, then run a narrow confirmation pass ("resolved/partial/not resolved plus new contradictions only"). Repeat until the reviewer says APPROVE. Late LOW items go into the review record as binding clarifications, which keeps the hash stable. Also verify the reviewer's own evidence by content: run 1 misread a colspan=3 footer. Related: [[consult-surveyor-before-committing]].
