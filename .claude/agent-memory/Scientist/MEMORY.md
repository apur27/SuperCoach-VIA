# Memory Index

- [All-time ranking formula constraints](all_time_formula.md) — Rank-based formula trade-offs; data supports Matthews #1 over Carey, expert lists don't
- [No position column in player data](data_no_position.md) — Player CSVs lack a `position` field; per-position analysis requires a new data source
- [Disposal predictor top-end compression](prediction_top_end_compression.md) — log1p target + L1 LGBM loss caused max-pred compression; removed both and added OOF linear calibration
