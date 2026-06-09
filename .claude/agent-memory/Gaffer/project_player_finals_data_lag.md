---
name: player-finals-data-lag
description: Player performance CSVs lag matches CSVs for the current season's finals — matches_2025.csv has the GF result but player files have only H&A rounds
metadata:
  type: project
---

For the in-progress / most-recent season, `data/matches/matches_YYYY.csv` contains the finals series (incl. the Grand Final result), but the per-player `data/player_data/*_performance_details.csv` files contain only home-and-away rounds (no `round=="GF"/"PF"/"SF"/"EF"` rows yet).

**Confirmed 2026-06-05:** matches_2025.csv has the 2025 GF (Brisbane 122 def Geelong 75), but McCluggage/Rayner/Ah Chee player files show NO finals rounds for 2025 (their last 2024 rows include EF/SF/PF/GF; 2025 stops at H&A). Self-corrects on next data refresh.

**Why:** matters for any analysis that joins player game logs to premiership outcomes (e.g. "did player X play in the winning GF?"). A GF-participation metric will silently undercount the current season's premiers at the player level even when the result is verified at the match level.

**How to apply:** When deriving player-level finals/premiership participation, scope the metric to seasons where player finals data is complete (e.g. "2000–2024 Grand Finals") and verify the current season's premiership at the *match-result* level only, with an explicit data-currency note. The list-management-101.md article (2026-06-05 rewrite) handles it exactly this way — its 45-premiership-player / strike-rate-by-pick figures are computed over 2000–2024 GFs while flag *counts* run through 2025. See [[player-data-quirks]] and [[backtest-n-filtering]].
