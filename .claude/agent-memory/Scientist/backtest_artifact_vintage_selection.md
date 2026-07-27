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
before writing a `backtest_summary_<ts>.csv`. Two such orphan R1 files
(`20260518_131738`, `20260525_182141`) — residue of the from-R1 re-run rule
violations — existed on 2026-07-26; their mtimes were *newer* than the
authoritative `20260511_191837` vintage, so "latest by mtime" picked the orphan.
**Both were deleted before 2026-07-27 (verified gone).** The trap class is still
live — any aborted run recreates it — but do NOT expect to find those two files.
Their disappearance also made the doc's old frozen R1-R13 headline
(MAE 4.020 / RMSE 5.155 / bias -0.097) **unreproducible from artifacts on disk**:
no surviving vintage combination regenerates it (closest, R1=`20260430_184619`,
gives 4.0214 / 5.1542 / -0.0925). Keep-last today gives 4.0185 / 5.1497 / -0.0930.

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

### Trap 3 — `_load_top30_player_deviation` sorts by TIME-OF-DAY, not timestamp

`update_team_analysis._load_top30_player_deviation` extracts its dedup sort key as
`base.rsplit("_", 1)[-1].replace(".csv", "")`. For
`prediction_vs_actual_round_1_2026_20260511_191837.csv` that yields **`"191837"`** —
the HHMMSS only. **The date is silently discarded.** Two vintages of the same round
are then ranked by what time of day the run happened to start.

Verified 2026-07-26: all 10 multi-vintage 2026 rounds currently resolve to the
correct file anyway — pure luck, because each re-run happened to start later in the
clock-day than the run it superseded. **0 rounds mis-selected today, but it is a coin
flip.** A re-run kicked off at 09:00 that supersedes an 18:00 run selects the STALE
vintage.

**How to apply:** do not trust this function's "keeps the most-recent run" docstring.
If you are reasoning about which vintage a published surface uses, read the extraction,
don't assume full-timestamp ordering — and never verify it by computing
`sorted(full_timestamps)[-1]` yourself, which is how I initially mis-diagnosed an
orphan file as live-contaminating this table when it was not. Fix is a full-timestamp
regex (`(\d{8}_\d{6})`), same as `scripts/backtest_completeness.py` already uses.

### Trap 4 — top-30 table dedups on `(player, round)`: same leak as Trap 2

`update_team_analysis._load_top30_player_deviation` does
`drop_duplicates(subset=["player","round"], keep="last")`. Identical failure class
to Trap 2: when the newer vintage of a round covers FEWER PLAYERS, the missing
players survive from the stale vintage instead of being superseded.

Live instance (verified 2026-07-27, R1-R20): the top-30 table pools **7,419** rows
— the canonical 7,153 plus **128 phantom R18 rows** from `20260707_154033`, exactly
the four clubs (Geelong, Melbourne, St Kilda, Western Bulldogs) that the canonical
`20260710_214217` vintage lacks. So `docs/afl-backtest-2026.md` publishes a headline
computed over 14 R18 clubs sitting directly above a top-30 table computed over 18 —
an incoherent hybrid *inside the auto-generated block*, which is why it survived a
"block 1 is auto-generated, therefore fine" review. The per-round table and its
closing line in the same block ARE correct (they come from summary CSVs, not details).

Visible symptom: `rounds_tracked` inflated by 1 for players at the four affected
clubs (Bailey Smith 17 vs canonical 16, Wanganeen-Milera 14 vs 13), and Bradley Hill
occupying a top-30 slot that canonically belongs to Jake Bowey / Ryley Sanders.

**Fix + invariant:** apply supersede-by-FILE first (`newest = groupby(["year","round"])
["_mtime"].transform("max")`), then dedup. Reconciliation: pooled detail rows after
`dropna(["predicted_disposals","actual_disposals"])` MUST equal
`backtest_summary` deduped `n_players.sum()`. Same 7,153 invariant as by_team.

**STATUS: Traps 3 and 4 are FIXED in `update_team_analysis._load_top30_player_deviation`
(2026-07-27, uncommitted at time of writing).** Three points worth carrying forward:

1. The vintage key in THIS loader is the **filename** timestamp (`(\d{8}_\d{6})` regex),
   NOT `os.path.getmtime` as in `scripts/update_eval_surface.sh`. Filename beats mtime —
   mtime is what makes Trap 1 (orphan detail CSVs) bite. `update_eval_surface.sh` is still
   on mtime and is only safe because `backtest_by_team_*` files are summary-companions.
2. Supersede-by-file alone gets you to **7,291**, not 7,153. The last 138 rows are
   **unscored forward predictions** (`actual_disposals` null) that the round summaries never
   counted. Verified per-round: `nonnan == n_players` for all 20 rounds, exactly. Those rows
   were also contaminating `avg_predicted` and `rounds_tracked` (but not `avg_error`, since
   `error` is null on them) — the two halves of the published comparison were computed over
   different populations.
3. After supersede-by-file the `(player, round)` dedup is a **within-file guard only**
   (measured 0 hits on the real 2026 corpus). Keep it; do not present it as vintage selection.

**Two unrelated observations on this table, NOT actioned (presentation calls, not methodology):**
no minimum-rounds filter, so a 9-round player (Jake Bowey) ranks on `avg_actual` against
18-round players; and the group key is `(player, team)`, which splits any player appearing
under two clubs — 1 such row in 2026 (`Williams Bailey`, WB 11 rounds / WCE 7), which is
either a mid-season move or two distinct players sharing a name. Both are outside the top 30
except Bowey.

### Identity worth remembering

n-weighted mean of per-round summary metrics == pooled per-player metrics,
**exactly**, for MAE / bias / within-5 / within-10 (all are means over the same
denominator). So summary-based and detail-based methods can never disagree on
their own. If they do, it is a vintage-selection mismatch — nothing else.
See [[backtest_doc_verification]], [[feedback_backtest_rules]].

### Related coverage gap (CLOSED as ACCEPTED 2026-07-26, not a selection bug)

R18 2026 scores only 284 of 412 player-rounds — 4 of 18 teams absent — because
the `--from-csv` path scored an archived forward CSV written before the full R18
fixture existed. Real hole in the headline pool, not a merge artifact.
**Decision: accepted, do not re-score.** Full rationale and the both-vintage
comparison live in [[r18-2026-coverage-gap-accepted]]; durable note is in
`docs/afl-backtest-2026.md` under `## Methodology`.
