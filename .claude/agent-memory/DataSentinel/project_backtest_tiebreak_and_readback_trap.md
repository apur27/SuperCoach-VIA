---
name: backtest-tiebreak-and-readback-trap
description: afl-backtest-2026.md misses-table tie-break sort order, and a "Read:" callout paragraph pattern that restates tagged figures alongside a NEW untagged derived stat
metadata:
  type: project
---

Two findings from the 2026-07-27 Pass-2 re-verification of `docs/afl-backtest-2026.md`
(11-fix round, post-Skeptic-BLOCK on prose):

1. **Misses-table top-5 tie-break order requires a stable sort on file row order,
   not `sort_values('error')` default.** When two players in the same round share the
   exact same signed error (e.g. two players at -14 in Round 1: Petracca and Sinclair),
   the doc picks whichever one appears EARLIER in the source
   `prediction_vs_actual_round_<N>_2026_<ts>.csv` file. Reproducing this requires
   `sort_values('error', kind='mergesort')` (stable) — NOT the pandas default
   `kind='quicksort'`, which does not preserve original row order on ties and will
   silently pick the wrong player among a tie, even though the numeric values (predicted,
   actual, error) are all individually correct. Confirmed by testing rounds 1, 5, 15
   where the top-4 always matched under any sort but the 5th (a tie) only matched under
   `mergesort`. See also [[project_backtest_doc_verification_gotchas]].

2. **A "Read:" interpretive callout placed AFTER a tagged table (e.g. the Cumulative
   summary block) can smuggle in a brand-new derived statistic with no tag of its own,**
   even when it also restates an already-tagged figure from the table just above (e.g.
   "the population-level signed error ... population figure of −0.110" — fine, already
   tagged/verified in the table — followed immediately by "every one of the top 30
   disposal-winners is under-predicted, by roughly 3 disposals on average" — a NEW
   mean-of-avg-error computation over the top-30 table, computed correctly (verified:
   mean of the 30 `Avg error` values = -2.96, "roughly 3" is accurate) but never tagged
   anywhere in the doc). The established mid/before-sentence covering-tag convention for
   this doc (a tag governs its own sentence or the table it introduces) does NOT extend
   across a `<!-- ...-END -->` block boundary into a fresh prose paragraph below it —
   treat each post-table "Read:"/interpretive paragraph as its own tag scope requiring
   its own **[data]** tag if it introduces a new number, even if it also references an
   already-tagged number in the same breath. Restating an EXACT already-tagged value
   (same number, same precision) from the table immediately above is not itself a
   violation; a NEW aggregate computed from that table's rows is.

Also reconfirms (still true as of this pass): `Blake Hardwick` and `Shai Bolton` appearing
in the misses table are real, currently-active AFL players (verified against the backtest
CSV rows) whose surnames coincidentally collide with entries on `config/coach_names.txt`
(Damien Hardwick, [coach] Bolton) — per that file's own scope note, a full player name is
never a coach-anonymity violation. Don't flag these.
