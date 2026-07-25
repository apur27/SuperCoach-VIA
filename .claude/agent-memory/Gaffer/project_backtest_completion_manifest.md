---
name: backtest-completion-manifest
description: Backtest rounds count as done only if a completed cycle marked them in completed_runs.json — artifact-exists is no longer proof; sweep bootstraps on a fresh clone
metadata:
  type: project
---

`scripts/backtest_completeness.py` + `data/prediction/backtest/completed_runs.json`
decide which backtest rounds are actually complete. Added 2026-07-25.

**Why:** on 2026-07-20 a cycle wrote backtest artifacts at 15:57, then FATALed at
16:04 on the phantom-row gate (mass-duplicated player corpus). The retry run took the
newest `backtest_summary_*.csv` on disk as proof round 20 was scored, skipped
recomputation, and shipped figures derived from a corpus we had just rejected.
"An artifact exists" cannot distinguish a finished cycle from an aborted one.

**How to apply:**
- A run counts only if a cycle marked it complete AFTER a successful push. The mark
  call sits next to the F04 completion sentinel in `weekly_refresh.sh`, and after the
  standalone push in `refresh_and_rank.sh`. Never move it earlier — marking before the
  push re-creates the bug.
- `sweep` runs BEFORE the start-round is computed. It MOVES orphans to
  `backtest/quarantine/`, which also hides them from `update_team_analysis.py` and
  `update_eval_surface.sh` (both pick the newest summary by mtime) — one mechanic
  closes both the harness and the publishing path.
- **Bootstrap semantics matter:** no manifest at all means "never initialised" (fresh
  clone of committed history), NOT "everything is an orphan". Sweep adopts what is on
  disk instead of quarantining it. Without this, a lost manifest would move the entire
  backtest history aside and reset the pipeline to round 1.
- `completed_runs.json` is in the Phase-1 allowlist and MUST stay committed.

**Postscript worth remembering:** the R20 figures this incident threatened turned out
to be bit-identical when re-scored on the repaired corpus — `--from-csv` reads an
immutable archived forward CSV, and the round's actuals were being scraped for the
first time so the dedup bug could not reach them. But the *forward* R21 prediction WAS
corrupted (192 of 413 rows differed). Corpus damage does not hit every artifact
equally; re-derive rather than assume in either direction.

Related: [[phantom-row-dedup-gate]], [[weekly-r20-r21-retro]].
