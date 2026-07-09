---
name: conceded-stats-corruption
description: conceded-stats file had a disposals/handballs/marks column rotation; no writer in repo; 2025 player_data too incomplete to regenerate; repair via scripts/build_conceded_stats.py
metadata:
  type: project
---

`data/conceded_stats/team_stats_conceded_2025.csv` (per-team, per-match totals a
team CONCEDED = sum of opponent's player stats) was committed as raw data with
**no generator anywhere in the repo** (grep for `conceded` across .py/.ipynb/.sh
finds only prose in update_team_analysis.py).

**Corruption fingerprint (2026-07-09):** a 3-way value rotation among
disposals/handballs/marks. The physical identity `disposals = kicks + handballs`
did NOT hold; instead the corrupt signature `marks == kicks + disposals` held on
100% of rows. Fix: `disposals<-old marks`, `handballs<-old disposals`,
`marks<-old handballs`. kicks + the last-6 block were correctly placed.

**Why the values are trustworthy despite corruption:** goals_conceded /
behinds_conceded match the opponent's final score in `data/matches/matches_2025.csv`
exactly (100%, authoritative complete source). That proves the underlying values
were computed from COMPLETE data — so the fix is a relabel, not a recompute.

**Cannot regenerate from player_data:** 2025 `data/player_data/*performance_details.csv`
is only ~55% complete (mean ~12 of 22 players per team per match; some files store
`round` as str vs int causing double-grouping). Summed team totals are useless as
ground truth — even goals correlated only 0.06 with the (correct) file values.
Do not try to rebuild conceded totals from player_data for recent seasons.

**Unverifiable residual:** tackles/hitouts/inside_50s/clearances conceded have NO
complete per-stat source in the repo. `hitouts_conceded` median ~10 looks
implausibly low for full-team hitouts but could NOT be proven wrong — left
untouched. To verify, obtain a complete per-stat team source (afltables team
match stats or a fully-populated player_data season).

Repair writer: `scripts/build_conceded_stats.py` (idempotent: no-ops if invariant
already holds, raises on rows matching neither invariant). Schema/invariant tests:
`tests/unit/test_conceded_stats.py`. See [[data_stat_coverage_eras]] for related
recording-era gaps.
