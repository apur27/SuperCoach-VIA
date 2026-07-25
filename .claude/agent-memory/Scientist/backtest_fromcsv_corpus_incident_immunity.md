---
name: backtest-fromcsv-corpus-incident-immunity
description: The --from-csv backtest path is structurally immune to the dedup-doubling class of corpus incident; verify via the backtest_run_*.log "actuals=" line before ordering a costly re-score
metadata:
  type: project
---

**A corpus-integrity incident during cycle N does NOT automatically taint the round-(N-1) backtest scored in that same cycle.**

Verified 2026-07-25 by re-scoring R20 2026 after the 2026-07-20 dedup-doubling
incident ([[dedup_dtype_mismatch_doubling]]): the re-score was **bit-identical**
to the tainted-run artifact (n=361, MAE 3.919668, RMSE 5.04138, w5 76.454%,
w10 95.291%). Nil delta.

**Why:** Two structural reasons, both worth checking before ordering a re-score.

1. *Prediction side is frozen.* `--from-csv` scores an **archived** forward CSV
   written in a PRIOR cycle (here `next_round_20_prediction_20260714_0730.csv`,
   6 days before the incident). No retrain, no `LeakProofPredictor`. The live
   corpus cannot reach it.
2. *Actuals side can't be doubled for the round being scored.* The doubling
   mechanism re-emits **the file's last-recorded game from the previous cycle**
   (a fixture-date-drift row, e.g. R19). Rows for the round just scraped are
   NEW — they have no int-typed counterpart already in the CSV, so there is
   nothing for the dtype-blind `drop_duplicates` to fail against. Doubled rows
   are therefore always round <= N-1, never round N.

**Why:** ordering a re-score costs real time and re-running the wrong path
(without `--from-csv`) costs ~24 min/round of Optuna. Knowing the blast radius
of a corpus incident up front avoids both.

**How to apply:** when a corpus incident lands in the same cycle as a backtest,
do NOT assume the backtest is tainted. Check first:

- `grep "actuals=" data/prediction/backtest/backtest_run_<TS>.log` — the
  `predictions=P actuals=A` line records exactly how many actuals rows
  `_gather_actuals` saw. Compare A against the current deduped corpus count for
  that (year, round). Equal => the actuals slice was clean at scoring time.
- Fan-out check: `preds.merge(actuals, on=["player","team"], how="left")` is a
  LEFT merge, so duplicated actuals inflate the detail CSV above `len(preds)`.
  If `prediction_vs_actual_*.csv` row count == forward-CSV row count with zero
  duplicate (player, team) pairs, no fan-out occurred. Definitive.

**Contrast — the FORWARD prediction IS affected.** The R21 forward CSVs written
either side of the fix (`next_round_21_prediction_20260720_1557.csv` vs
`..._2007.csv`) differ on 192 of 413 rows. The doubled corpus did reach the
training data. So: forward predictions from an incident cycle are suspect;
`--from-csv` backtests of prior rounds are not.

Re-scoring is safe to repeat — it writes a new timestamped summary, and both
`generate_backtest_section()` (merge-all, dedup by year+round keep-last) and
`refresh_and_rank.sh`'s `LAST_TS`/`LAST_ROUND` detection handle the extra file
correctly. See [[feedback_backtest_rules]].
