---
name: backtest-artifact-vintage-selection
description: Two ways backtest artifact merges silently pick the wrong vintage — orphan detail CSVs with no summary, and by_team dedup on (year,round,team) leaking stale teams. Both produce plausible-but-wrong eval numbers.
metadata:
  type: project
---

**Rule: the surviving `backtest_summary_*.csv` vintage per round is the ONLY
authoritative selector. Never pick a backtest artifact by file mtime.**

**Why:** two independent selection bugs, both found 2026-07-26 when README eval
figures disagreed with an independent re-measurement. Each produces numbers that
look reasonable and reconcile to nothing.

### Trap 1 — orphan detail CSVs (no companion summary)

`prediction_vs_actual_round_N_2026_<ts>.csv` files exist for runs that ABORTED
before writing a `backtest_summary_<ts>.csv`. As of 2026-07-26 there are two
such orphan R1 files (`20260518_131738`, `20260525_182141`) — residue of the
from-R1 re-run rule violations. Their mtimes are *newer* than the authoritative
`20260511_191837` vintage, so "latest by mtime" picks the orphan.

Effect on R1-R20 2026 (n=7,153 either way, so row count does NOT catch it):
`bias -0.1099 -> -0.1127`, `within-10 95.778% -> 95.736%`, `MAE 3.9575 -> 3.9583`.

**How to apply:** build `vint = {round: ts}` by merging all `backtest_summary_*`
(dedup `(year,round)` keep=last, files ordered by mtime), then load
`prediction_vs_actual_round_{r}_2026_{vint[r]}.csv`. Do not glob-and-take-latest.

### Trap 2 — `by_team` dedup key leaks stale teams

`drop_duplicates(subset=["year","round","team"], keep="last")` does NOT supersede
a round when the newer run covers FEWER teams. Rows for the dropped teams survive
from the stale vintage.

Live instance: R18 2026 was scored twice — `20260707_154033` (412 preds, 18 teams)
then re-scored `--from-csv` as `20260710_214217` (320 preds / 284 scored, only 14
teams, because the archived forward CSV predated the full fixture). Geelong,
Melbourne, St Kilda, Western Bulldogs survived from the stale file, adding 92
phantom player-rounds (by_team sum n = 7,245 vs summary n = 7,153) and moving
St Kilda's season bias from **-0.583 to -0.733**. The headline used one vintage
and the team table another — an incoherent hybrid, not merely stale.

**Detector (cheap, run it every time):** `by_team` deduped `n.sum()` MUST equal
`backtest_summary` deduped `n_players.sum()`. Any delta is leaked vintages.

**Correct merge:** keep only rows from the newest FILE per `(year, round)`:
```python
newest = t.groupby(["year","round"])["_mtime"].transform("max")
t = t[t["_mtime"] == newest]
```
Verified to reproduce the pooled-detail team table exactly (sum n = 7,153).

### Identity worth remembering

n-weighted mean of per-round summary metrics == pooled per-player metrics,
**exactly**, for MAE / bias / within-5 / within-10 (all are means over the same
denominator). So summary-based and detail-based methods can never disagree on
their own. If they do, it is a vintage-selection mismatch — nothing else.
See [[backtest_doc_verification]], [[feedback_backtest_rules]].

### Related coverage gap (open, not a selection bug)

R18 2026 scores only 284 of ~412 player-rounds — 4 of 18 teams absent — because
the `--from-csv` path scored an archived forward CSV written before the full R18
fixture existed. Real hole in the headline pool, not a merge artifact.
