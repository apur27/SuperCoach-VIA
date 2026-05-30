---
name: Player CSV date column is unreliable
description: In data/player_data/*_performance_details.csv, the `date` field is off by ~5-7 days compared to the actual match date - cross-check via data/matches/ when an exact date matters
type: project
---

The `date` column in `data/player_data/<player>_performance_details.csv` does not match the actual match date in `data/matches/matches_<year>.csv`.

**Concrete example (verified 2026-05-19):**
- Pendlebury performance file row 0: Round 10 2006 vs Brisbane, `date = 2006-05-03`
- Actual `data/matches/matches_2006.csv` Round 10 Collingwood vs Brisbane: **2006-06-03 19:10**
- The performance-file date is off by exactly one month.

The personal-details file (`*_personal_details.csv`) has its own `debut_date` field in `dd-mm-yyyy` format which doesn't match either - Pendlebury's says `29-05-2006` but the first row in his performance file is R10 vs Brisbane (which was actually 2006-06-03).

**Why:** Looks like a parsing/format bug in the original data ingest - month and day appear to have been swapped in some rows, or the day-of-month is shifted.

**How to apply:**
- For relative ordering within a season the `date` field is fine (rows are in correct chronological order).
- For any article/doc that cites an **exact match date**, cross-check `data/matches/matches_<year>.csv` by `round_num` + team name.
- For round-based citations (e.g., "Round 10, 2006 vs Brisbane") the performance file is trustworthy.
- The `games_played` counter is also slightly off because the 2010 drawn-GF + replay is collapsed into one row but counted as two games (gap of 2 at that row), and a couple of 2025 finals appear to be missing rows entirely (gap of 3 between end-2025 and start-2026). Total career-game number from `games_played.max()` (432 for Pendlebury) matches official record; row count (429) under-counts by 3.
