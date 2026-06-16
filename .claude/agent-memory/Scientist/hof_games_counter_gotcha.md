---
name: HoF games counter — use games_played column, not row count
description: Career games = max(row_count, max(games_played)), NOT row count alone (undercounts missing-row games like drawn GFs) NOR blind games_played.max() (283 files have trailing-NaN counters that undercount). Applies to compute_stat_leaders.py AND top_players_comprehensive.py.
type: project
---

`docs/hall-of-fame/compute_stat_leaders.py` originally computed career games as `df["team"].notna().sum()` (row count). This is wrong for any player whose career includes drawn Grand Finals or finals appearances that the FanFooty-derived per-player CSVs collapse — Pendlebury's file has 429 rows but the canonical AFL games tally is 432, and the per-row `games_played` column carries the correct running count.

**Why:** the Harvey-vs-Pendlebury joint 432 milestone (R10 2026) was hidden by the stale row count, which had Pendlebury ranked #2 at 428 in `_stat_leaders.json` even though he was actually tied for #1 at 432. The hand-edited `hall-of-fame-stat-leaders.md` already showed the tie, but the JSON and the auto-generated chart did not, so the next regen would have silently undone the fix.

**CRITICAL — use `max(row_count, max(games_played))`, NOT a blind `games_played.max()`.** A 2026-06-10 audit of all 13,334 player files found 219 files where max(games_played) > row count (the gap the fix targets, e.g. Pendlebury 432 rows → 435) BUT ALSO 283 files where max(games_played) < row count. The negative-gap files have a trailing run of NaN `games_played` (the counter stops being populated for the last few rows), so a blind `.max()` UNDERCOUNTS them — e.g. Shane Tuck (different person from Michael Tuck) 173 rows but max_gp 167, Brett O'Hanlon 9 rows but max_gp 3. Always floor with the row count. This applies to both the HoF generator and the ranking pipeline.

Largest positive gaps (fix bumps these up): mcandrew_lachlan 5→14 (sparse stat rows, Sydney+Adelaide), dunkley_josh 203→207, rayner_cam 176→180, lester_ryan 239→243. None crossed the 150-game eligibility line, so the all-time top-100 membership/order did not change.

**How to apply:**
- Counting career games from a player performance CSV: `games = max(row_count, max(games_played))` after stripping `↑`/`↓` markers and coercing to numeric. Fall back to row count if the column is missing or all-NaN.
- Two scripts now embed this: `docs/hall-of-fame/compute_stat_leaders.py` (per_player) and `top_players_comprehensive.py` (`_aggregate_one_file` career-games return + `_read_player_performance` `career_games` field). In the latter, `total_games` stays row-based as the per-game-average denominator; only the narrative count uses `career_games`.
- Per-SEASON / windowed game counts (5yr team profiles in update_team_analysis.py, weekly cheat sheet) correctly use row count — do NOT switch those to the counter.
- After regenerating `_stat_leaders.json`, verify ties surface with `rank_label` like "1=" / "1=" rather than 1 / 2. `chart_wall_of_records` joins tied co-holders with " = ".
- The pipeline is deterministic (no seeds); regenerating `data/top100/all_time_top_100.csv` should be byte-identical run-to-run. `_stat_leaders.json` / `.md` are gitignored build artifacts — the committed truth is the table in `hall-of-fame-stat-games.md`.

Related memory: [player_csv_date_format.md] documents the row-count-vs-games-played gap.
