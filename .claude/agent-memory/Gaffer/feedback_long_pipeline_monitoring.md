---
name: long-pipeline-monitoring
description: How to monitor the long refresh_and_rank.sh pipeline — timing profile, silent-fit phases, self-match pgrep trap, Bash 10-min cap
metadata:
  type: feedback
---

# Monitoring the long refresh pipeline

**Why:** the 2026-07-07 cycle burned many turns polling a healthy-but-slow refresh, and its `refresh_and_rank.sh` got killed mid-backtest. These are the durable lessons.

**Timing profile of `refresh_and_rank.sh` (~60 min total):**
- [1/6] scrape ~10-15 min (logs "Found player link" / "Processed N/916 player pages").
- [2/6] top-100 fast.
- [3/6] prediction: runs an Optuna study (~50-100 trials, ~10-30s each) THEN a **silent CPU-bound final fit of ~10-13 min with NO log output**. Writes `next_round_<N>_prediction_*.csv` at the end.
- [4/6] backtest: **its own** Optuna study per round (~50 trials) + silent fit. This is the long, fragile leg — it was the one killed.
- [5/6] refresh_readme.py + compute_stat_leaders + charts (~4 min).
- [6/6] self-commit + push (its own allowlist: season docs + all_time_top_100.csv + data/top100/, NOT raw data/player_data|matches|lineups).

**A multi-minute log silence with the process alive at 300%+ CPU is a normal silent-fit phase, NOT a stall.** Only worry if CPU drops to ~0 or one core pins at 100% (busy loop).

**Self-match pgrep trap (cost real confusion this run):** `pgrep -f "bash refresh_and_rank.sh"` run inside a `bash -c` whose command line CONTAINS that string matches its own parent → always returns "running", never detects exit. Use the bracket trick: `pgrep -f 'refresh_and_[r]ank.sh'`, `pgrep -f '[s]upercoach.prediction'`. Same for `ps aux | grep`.

**How to WAIT:** the Bash tool caps at 600000ms (10 min), so a 25-min backtest CANNOT run in one synchronous call and long background jobs get killed here. Do NOT tight-poll every ~20s (wastes turns). Best: launch the pipeline once with `run_in_background`, then check state with self-match-proof commands + the log file. Definitive "did it finish" = grep the log for `Pipeline completed successfully` (NOT the intermediate `Pipeline completed!` that refresh_data.py prints mid-scrape) AND a bracket-pgrep showing the process gone AND a new self-commit in `git log`.

**If it dies mid-run:** it's a partial cycle. Recover by running the missing steps deterministically (refresh_readme.py, HOF pipeline, update_eval_surface.sh) and committing the union of refresh_and_rank.sh's + weekly_refresh.sh's allowlists yourself. Ship the incomplete-backtest as a documented QA warning; don't hide it. Backlog fix: Optuna `--resume` (SQLite storage) on backtest.py so a kill resumes instead of restarting.
