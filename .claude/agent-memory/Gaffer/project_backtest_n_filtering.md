---
name: backtest-n-filtering
description: Backtest cumulative player-count (n) is late-out-filtered, ~9 below raw prediction_vs_actual row sum; metric deltas negligible
metadata:
  type: project
---

The council-gated `docs/afl-backtest-2026.md` headline "player predictions scored" (n=4,806 for R1-R13) is the **late-out-filtered** count, not the raw row count of the pooled `prediction_vs_actual_round_*` CSVs. A deterministic pool of those CSVs gives ~4,815 — the doc's per-round table n_players runs +1/+2 lower than raw rows on ~7 of 13 rounds (no duplicate `player` rows; it's `backtest.py`'s late-out filter, methodology line ~185).

**Why:** When verifying README eval figures against the doc, my raw-CSV pool disagreed with the doc's n by 9. Investigated rather than papered over; deltas on MAE/bias/within-5 were negligible (MAE 4.019 vs 4.020), confirming the doc is canonical and my pool just used a looser filter.

**How to apply:** When re-deriving cumulative backtest n to check a doc, expect raw-CSV pool to slightly EXCEED the doc's filtered n — that gap is expected, not an error. The Scientist's filtered figure in the gated doc is the canonical `[data]` value to propagate; do not substitute a raw pool count. Canonical CSV per round (per doc reproducibility note): R1-R10 from run 20260511_191837, R11 from 20260518_144551, R12 from 20260525_190033, R13 from 20260601_225644. CSV `error` col = predicted - actual (negative => under-predict). See [[feedback_gaffer_consult_scientist]].
