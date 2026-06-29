---
name: llm-datasentinel-arithmetic
description: LLM DataSentinel agents mis-compute CSV sums — verify disputed numbers deterministically yourself; prefer a hard-coded pandas gate
metadata:
  type: feedback
---

The LLM-based DataSentinel sub-agent cannot be trusted for arithmetic on its own. On the 2026-06-29 HOF cycle, one re-gate run reported FAILs with CSV sums that were ALL wrong: Pendlebury disposals 11,043 (truth 11,044), Sidebottom 8,375 (truth 8,386), B.Harvey 9,166 (truth 9,213), R.Harvey 9,649 (truth 9,656). Every "mismatch" it reported was a correct figure in the doc. A separate run also used the wrong games metric (see [[canonical-games-metric]]).

**Why:** number-matching by an LLM is a suggestion, not a gate, and is itself error-prone/injectable. My own deterministic pandas re-measurement (`df['col'].sum()`, `len(df)`, `max(rows, games_played.max())` for games) is the ground truth that de-conflicted it.

**How to apply:**
1. When a DataSentinel verdict hinges on a numeric mismatch, RE-MEASURE the disputed figures yourself with a one-off pandas script before trusting the verdict — especially before routing a "fix" to the Scientist or escalating a FAIL.
2. An LLM DataSentinel run is still useful for POINTING at suspect lines (the bad run correctly located 3 genuinely-stale prose figures even while mis-stating their values) — treat it as a locator, not an arithmetic authority.
3. Do NOT parallelize a DataSentinel reader against a Scientist writer on the same CSV — that race produced one of the false sums. See [[parallel-council-commits]].
4. Backlog/root-cause: build a DETERMINISTIC DataSentinel numeric checker (hard-coded comparison of every [data] career-total tag vs the CSV sum) so the gate stops depending on LLM arithmetic. Pairs with the renderer-extension item in [[drawn-gf-dedup-defect]].
