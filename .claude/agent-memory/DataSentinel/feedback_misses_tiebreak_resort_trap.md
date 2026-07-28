---
name: misses-tiebreak-resort-trap
description: When reproducing the "top over-predictions" half of the afl-backtest-2026.md misses table, sort descending directly with mergesort — do NOT sort ascending-mergesort then .tail(5) then re-sort descending, which reverses the tie-break order and silently swaps the 5th-place player.
metadata:
  type: feedback
---

**Rule**: `df.sort_values('error', kind='mergesort', ascending=False).head(5)` reproduces the
doc's "top over-predictions" column exactly. `df.sort_values('error', kind='mergesort').tail(5)`
followed by a second `.sort_values(ascending=False)` on that 5-row slice does NOT — the second
sort re-orders tied values (e.g. two players both at error=+11) using the tail-sorted (reversed)
relative order rather than original file order, so it can name the wrong player among a tie
(confirmed: this wrongly produced "Mitch Lewis" instead of the doc's correct "Balyn Obrien" for
Round 21's 5th over-prediction, even though both rows have identical error=+11 — the doc picks
whichever appears earlier in the source CSV).

**Why**: a stable sort's guarantee (equal keys keep their relative *input* order) is not
transitive across two separate stable-sort calls with different sort directions — sorting
ascending then reversing a slice does not equal sorting descending in one pass.

**How to apply**: always do the single mergesort call in the direction you need (ascending for
unders, `ascending=False` for overs) and take `.head(5)` from that one call. Don't derive one
direction from the other via reslicing/re-sorting.

See also [[project_backtest_tiebreak_and_readback_trap]] (the original ascending-side
tie-break finding this refines).
