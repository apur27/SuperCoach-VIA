---
name: AFL stat coverage by year (recording-era boundaries)
description: When each player stat first appeared in the data, plus known recording-method changes that look like real shifts but aren't
type: project
---

The player_data CSVs have inconsistent stat coverage across history. Forgetting this corrupts any era comparison.

**First year with a non-trivial mean (>0.01) for each metric** (verified by aggregating data/player_data/*_performance_details.csv across all 13,320 player files, May 2026):

- **kicks, marks, handballs, disposals, behinds**: 1965 (no AFL stats kept before that)
- **hit_outs**: 1966
- **tackles**: 1987
- **clearances, inside_50s, rebound_50s, clangers**: 1998
- **contested_possessions, uncontested_possessions, contested_marks, marks_inside_50, one_percenters, bounces**: 1999
- **goal_assist** (column is SINGULAR, not `goal_assists`): 2003
- **goals**: 1897 (the entire dataset)
- **brownlow_votes**: coverage is patchy — data present 1931-1934, then a GAP 1935-1983 (all NaN/0), then continuous 1984-2025. No pre-1931 data despite the medal starting 1924. 2026 NaN (votes counted at season end). Lives only in performance_details.csv (also career-aggregated in all_time_top_100.csv). For a hard era gate use **1984** (first continuously-populated year); the 1931-1934 fragment is real but isolated and unreliable.

**CORRECTION (2026-07, re-verified 691,687 player-game rows across 13,343 files):** behinds is 1965, NOT 1897 — behinds is entirely NaN before 1965; only **goals** goes back to 1897. Earlier version of this note wrongly grouped behinds with goals.

**Known recording-method change that's NOT a real on-field shift:**

- **Hit-outs jump from ~10/g per ruck (2016) to ~17/g (2017)** and stay there. AFL Stats appears to have changed how hit-outs are counted around the 2016/2017 boundary. Don't treat any hit-out comparison that straddles 2016/2017 as evidence of a tactical change.

**Why:** The original era_based_statistical_analysis.py bucketed every game of a player's career into the era of their *debut*, which mixed (e.g.) a 1969 debutant's 1985 stats into "pre-1970". Plus, treating missing tackles in 1970 as zero deflates means and creates fake era differences.

**How to apply:** When doing era comparisons, (1) bucket player-game rows by **match year** (the `year` column in performance_details.csv), not debut year; (2) report `n_with_metric` alongside means so readers see when the stat became reliable; (3) in any narrative, explicitly call out that tackles pre-1987, clearances/contested-poss pre-1998, and hit-outs pre/post-2017 aren't directly comparable.

The canonical era-comparison script is `era_based_statistical_analysis.py` and its outputs (`data/era_stats.csv`, `data/era_yearly_trends.csv`, `data/era_team_scoring.csv`, `data/era_significance_tests.csv`, `data/era_summary.json`).
