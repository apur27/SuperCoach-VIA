---
name: weekly-r19-retro
description: 2026-07-07 weekly refresh (R18→R19) retro — partial run recovery, backtest gap, CR-1 fix, Surveyor fix plan, S-1/S-2 escalations
metadata:
  type: project
---

# Weekly refresh 2026-07-07 (R18 settled → R19) — retro

Shipped: commits `12fa8202d` (refresh) → `eb44eafd7`/`7596f8385` (pending-tasks) → `c6e1b6b29` (run report). All on main.

**What broke:** `refresh_and_rank.sh` was KILLED mid-[4/6] backtest (Optuna ~trial 48), so steps [5/6] docs/charts and [6/6] self-commit never ran. Recovered by running the missing deterministic steps by hand (refresh_readme.py → season docs to R18; compute_stat_leaders + charts + update_hof_pages; update_eval_surface.sh) and committing the union of the two scripts' allowlists myself.
**Why:** long ML jobs get killed in this environment — see [[long-pipeline-monitoring]].

**Deferred gap shipped (documented, not hidden):** the R18 walk-forward backtest did NOT complete. `afl-backtest-2026.md` + README eval surface reflect through R17. Body honestly says "17 rounds backtested" (not a false claim), but the refresh DATE is today. QA logged it as PASS_WITH_WARNINGS. Re-run the backtest leg next cycle.

**Top Chronicler recommendation (do first):** stamp `afl-backtest-2026.md` freshness from `max(round)` in the latest `backtest_summary_*.csv`, NOT wall-clock date; add a `weekly_refresh.sh` WARN when `max_round_backtested < max_round_played`. #2: add Optuna study persistence + `--resume` to `backtest.py` (SQLite storage, load_if_exists) so a killed run resumes instead of restarting — the direct cause of this cycle's gap. Chronicler ranks these eval-honesty/backtest-resume fixes ABOVE the Surveyor's product features (S-4 fantasy, S-7 age).

**CR-1 fixed this cycle:** `weekly_refresh.sh` round detection changed lexicographic `ls|sort|tail -1` → mtime `ls -t|head -1`; regression test `tests/unit/test_prediction_selection.py` (3/3). Confirmed real-world: FootyStrategy replaced a stale "Round 9" insights heading — the bug had frozen insights on Round 9 for weeks.

**Backlog written to `docs/pending-tasks.md`:** the full Surveyor fix plan (25 findings F01–F18, S1–S7, Sprint 2–4 + backlog, dependency-ordered, decision-gated). Superseded the earlier raw-findings version. Do NOT start any item marked *Blocked by decision* until the human records Decisions 1–3 (in `pending-decisions.md`).

**ESCALATIONS TO HUMAN (unresolved, in fix plan):**
- **S-1:** published "venue effects" model claim in `docs/how-to-use-this-for-supercoach.md:29` is FALSE — venue is not a model feature; also phantom features cba_percent + percentage_time_played silently dropped (prediction.py). Owner: Scientist (code) + Gaffer (doc correction S1a).
- **S-2:** `data/conceded_stats/team_stats_conceded_2025.csv` columns scrambled — wrong numbers, not just stale. No published figure contaminated (briefs never used it). Owner: Scientist.
