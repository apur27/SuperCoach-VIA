# Memory Index

- [All-time ranking formula constraints](all_time_formula.md) - Rank-based formula trade-offs; data supports Matthews #1 over Carey, expert lists don't
- [No position column in player data](data_no_position.md) - Player CSVs lack a `position` field; per-position analysis requires a new data source
- [Disposal predictor top-end compression](prediction_top_end_compression.md) - log1p target + L1 LGBM loss caused max-pred compression; removed both and added OOF linear calibration
- [AFL stat coverage by year](data_stat_coverage_eras.md) - Tackles only from 1987, clearances/cont-poss from 1998, hit-outs jump in 2017 is a recording change not a real shift
- [5-year team profile recipe](team_5yr_profile_recipe.md) - Methodology + markers for the auto-generated team playing-style section in README
- [Brownlow proxy weight validation](brownlow_proxy_weights.md) - Goals lifted from 5% to 15% after EDA on 145k historical games; eff_disp = disposals-clangers (no true column)
- [HOF data verification workflow](hof_methodology.md) - How to verify HOF page stats; recurring GF-result errors (1978/1991/1992) and captaincy attribution errors found in prior versions
- [Snapshot goals/behinds/clangers unreliable](snapshot_data_quality.md) - FanFooty live snapshot misindexes 3 fields; cross-check afltables for goals/behinds/clangers only
- [LightGBM CPU vs CUDA in prediction.py](prediction_lgbm_cpu.md) - prediction.py hardcodes `device='cuda'`; this env has no GPU so flip both occurrences to `'cpu'` (full backtest ~5-6h on CPU)
