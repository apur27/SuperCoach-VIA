---
name: backtest-reproduction-recipes
description: Exact reproduction recipes for docs/afl-backtest-2026.md derived-stat tags that are easy to get subtly wrong (rounded-vs-raw averaging, birth-year loader filter, definitional-threshold exemption)
metadata:
  type: project
---

Three reproduction recipes confirmed exact during the 2026-07-27 Pass-2 (content hash
`da1fc808…`) full re-verification of `docs/afl-backtest-2026.md`, all 18 `**[data]**` tags
verified computationally with zero failures:

1. **Top-30 "mean of Avg error" tag averages the DISPLAYED (1dp-rounded) column, not the
   raw unrounded per-player means.** Recomputing the top-30 disposal-winners' per-player
   `mean(error)` from the pooled per-game data and then averaging those 30 raw means gives
   `-2.949`. Rounding each player's `avg_error` to 1dp first (matching what the table
   displays) and then averaging gives exactly `-2.96` — which is what the doc's tag claims.
   Both are within the 0.1 tolerance band, but if you want an exact match rather than a
   "within tolerance" pass, average the rounded column. This is a real, reproducible
   generator convention (average-of-displayed-values), not noise.

2. **Birth-year loader filter (`1,808 of 13,357 player files`) is reproducible directly
   from `supercoach/prediction.py`'s `load_and_prepare_data`**: glob `data/player_data/*.csv`
   for `_performance_details` files (13,357 total as of 2026-07-27), extract DOB via
   `extract_dob_and_name`, keep only files where `dob.year > target_year - 40` (1986 for a
   2026 backtest) → 1,808 loaded. Confirmed exact via direct import and replay of that
   function's filter logic (do not approximate this by filtering the CSVs by `year` column
   — it is a DOB/filename-based per-file exclusion applied before any row is read).

3. **Definitional/methodology thresholds are exempt from the untagged-number scan, even
   though they are specific numbers.** E.g. "Bolded rows are those whose mean ABSOLUTE
   error exceeds **6** disposals" (top-30 table footnote), "Hit — within ±**5** disposals",
   "Miss — more than **10** disposals off", "more than **five** outright misses" (concerning-round
   rule), and the illustrative examples in the metric-definitions table ("if MAE = **4.1**,
   we were within ±**4** disposals", "a bias of **−0.7** means...**0.7** disposals too low")
   are all untagged in this doc and should NOT be flagged as FAILs. They define a rule or
   illustrate a formula in the abstract — they are not claims about actual measured data for
   a specific round/player, and this doc's established convention (unchanged across multiple
   verified passes) treats them as structural/definitional, same tier as "Round 11" or "Q3".
   Contrast with the genuinely-flaggable pattern in
   [[project_backtest_tiebreak_and_readback_trap]]: a NEW aggregate *computed from the data*
   in a post-table "Read:"/callout paragraph (e.g. "under-predicted by roughly 3 disposals
   on average") is not definitional — it's a derived stat and needs its own tag.

See also [[project_backtest_doc_verification_gotchas]] and
[[project_backtest_tiebreak_and_readback_trap]] for the keep-last vintage selection,
name-reversal, and stable-sort tie-break mechanics all reused here.
