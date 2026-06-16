---
name: matches-player-sync
description: matches_*.csv vs player_data sync status — R10 2026 truncation fixed + guard live; residual quarter-score and 2025 R1/R2 caveats
metadata:
  type: project
---

State of the matches-file vs player-data sync (as of 2026-06-09 check).

**RESOLVED — R10 2026 truncation.** matches_2026.csv had lost all 8 Round-10
(2026-05-03) matches; only the lone 2026-05-10 Richmond v Adelaide survived. The
8 rows were reconstructed from player CSVs (commit de9fe39ec) and now reconcile
exactly: R10 holds 9 rows == 9 distinct R10 matchups in data/player_data/.
matches_2026.csv carries all rounds R1–R14, no other count anomalies.

**LIVE GUARD.** `audit_match_rounds` + `fetch_round_fixture` in
scrapers/game_scraper.py compare each round against the afltables season fixture
by team-pair and WARN on any scheduled-but-unscraped matchup by name. Wired into
refresh_data.py (post-write self-check, ~L178–194), which refresh_and_rank.sh L12
runs every cycle. The silent-truncation class is now caught.

**Why:** A scraper gap silently dropped a whole round and went unnoticed because
nothing cross-checked match files against the known fixture or against player CSVs.

**How to apply:**
- The 8 reconstructed R10 rows are LOWER-FIDELITY: quarter scores all 0 (unknown),
  behinds are player-sum lower bounds, venue/time blank. Final goals are sound.
  Any consumer reading quarter splits or exact behinds for R10 2026 will be wrong;
  margin- and aggregate-based consumers (backtest, briefs) are safe. Backfill when
  afltables publishes full R10 box scores.
- Residual: 2025 R1/R2 shows a ±1 row delta vs player-data matchup counts —
  consistent with the AFL Opening Round labeling boundary, NOT a truncation. Low
  severity, not fully reconciled; confirm before treating as a real gap.
- Related: [[project_finals_doc_stale_2026]], [[project_player_finals_data_lag]].
