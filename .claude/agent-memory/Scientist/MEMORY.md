# Memory Index

- [Backtest doc verification](backtest_doc_verification.md) - afl-backtest-2026.md has 3 sections that drift to different round windows; canonical sources + 4486/4806 (NaN-actual) reconciliation
- [All-time ranking formula constraints](all_time_formula.md) - Rank-based formula trade-offs; data supports Matthews #1 over Carey, expert lists don't
- [No position column in player data](data_no_position.md) - Player CSVs lack a `position` field; per-position analysis requires a new data source
- [Blank counting stat = zero in a played game](blank_counting_stat_means.md) - NaN tackles/marks/hit-outs means a real zero, not missing data; use fill-zero for season means within the stat's recorded era
- [Disposal predictor top-end compression](prediction_top_end_compression.md) - log1p target + L1 LGBM loss caused max-pred compression; removed both and added OOF linear calibration
- [AFL stat coverage by year](data_stat_coverage_eras.md) - Tackles only from 1987, clearances/cont-poss from 1998, hit-outs jump in 2017 is a recording change not a real shift
- [5-year team profile recipe](team_5yr_profile_recipe.md) - Methodology + markers for the auto-generated team playing-style section in README
- [Brownlow proxy weight validation](brownlow_proxy_weights.md) - Goals lifted from 5% to 15% after EDA on 145k historical games; eff_disp = disposals-clangers (no true column)
- [HOF data verification workflow](hof_methodology.md) - How to verify HOF page stats; recurring GF-result errors (1978/1991/1992) and captaincy attribution errors found in prior versions
- [Snapshot goals/behinds/clangers unreliable](snapshot_data_quality.md) - FanFooty live snapshot misindexes 3 fields; cross-check afltables for goals/behinds/clangers only
- [LightGBM CPU vs CUDA in prediction.py](prediction_lgbm_cpu.md) - prediction.py auto-probes for GPU LightGBM at module load; this env falls back to CPU (full backtest ~5-6h)
- [⚠️ CRITICAL: Backtest rules — incremental only, preserve all rounds](feedback_backtest_rules.md) - NEVER re-run from R1. Only backtest missing round. Merge ALL summary CSVs for cumulative doc. Both fixes in commits 855b6d225 + 2edbee5f9. Violated TWICE.
- [Live pipeline misfires past end-of-game](live_pipeline_glitch.md) - FanFooty polling stamps end-of-game scores into earlier-quarter docs; prune by scope + score-magnitude sanity
- [Brownlow ineligibility registry](brownlow_ineligibility_registry.md) - Suspended-player flag lives in `BROWNLOW_INELIGIBLE_2026` in update_team_analysis.py; doc is auto-regen between markers
- [refresh_data.py "0 files grew" log is unreliable](refresh_data_grew_counter_bug.md) - Don't trust the summary line; verify with `git status` on data/player_data/
- [Player CSV date column is unreliable](player_csv_date_format.md) - Performance-file `date` field off by ~1 month from real match date; cross-check via data/matches/ for exact dates. Also: games_played counter > row count (drawn GFs collapsed, some finals rows missing).
- [HoF games counter must use games_played col, not row count](hof_games_counter_gotcha.md) - compute_stat_leaders.py was patched: rank ties get `rank_label` "1=" and chart_wall_of_records joins tied co-holders.
- [GPU "no device" = kernel module missing, not CUDA](gpu_kernel_module_missing.md) - This laptop's nvidia.ko is not built for the running kernel after kernel upgrades; userspace is fine. Also: CUDA_VISIBLE_DEVICES='' is exported in this shell.
