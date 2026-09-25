---
name: legacy-v1-ranking-parity
description: Legacy top-100 ranking is itself non-deterministic (glob-order ties); parity needs a sorted scan. Legacy conceded file is internally inconsistent, not reproducible.
metadata:
  type: project
---

Rewrite analytics (src/supercoach_via/analytics/rankings.py, config/ranking_legacy_v1.toml) reproduce
top_players_comprehensive.py EXACTLY (100/100 ranks, max |d| 4e-16, all 130 yearly lists) when the
legacy scan is sorted. The legacy script uses unsorted glob order, and pre-1965 season scores tie
often (goals*55+behinds*1.5, int-truncated), so two legacy runs on identical bytes differ: 91/100
same rank, max |d| 0.0021 (verified 2026-09-25). legacy_v1 breaks ties by player_id = sorted slug.

**Why:** "parity" against a committed legacy CSV can never be exact; compare against a sorted-glob
legacy run on the same bytes instead.

**How to apply:** run legacy only in var/agent-analytics/legacy-copy (or in-process with outputs
redirected to tmp). Removing the green_william/steele_roan duplicate files leaves all-time ranks
unchanged (|d| < 1e-5); only yearly 2025/2026 move. data/conceded_stats/team_stats_conceded_2025.csv
(R1-13 only) mixes own-team disposals/kicks/handballs/marks with opponent goals; other columns match
neither side — treat as archive, not a golden. 2025 player_data now has 23 player rows per team-game.
