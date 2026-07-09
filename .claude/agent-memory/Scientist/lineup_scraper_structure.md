---
name: lineup-scraper-structure
description: afltables match-stats table column layout + lineup name format + backfill dedup gotcha for team_lineups CSVs
metadata:
  type: reference
---

Team lineups (`data/lineups/team_lineups_<team>.csv`) are scraped by
`MatchScraper` in `scrapers/game_scraper.py` from afltables match pages.

**afltables match-stats table structure** (stable across at least 2024→2026):
each `table.sortable` is `#`, `Player`, KI, MK, HB, DI, ... — **column 0 is the
jersey number** (may carry `↑`/`↓` sub markers), **column 1 `Player` is the name
as `"Surname, Firstname"`**. Footer rows are `Rushed` / `Totals` / `Opposition`
(label in col 0). The parser must read the `Player` column (not col 0) and
reverse `"Surname, Firstname"` → `"Firstname Surname"` to match the stored CSV
format. `_extract_player_names` raises `LineupParseError` if a populated table
yields zero valid names (guards against a future silent structure change).

**Backfill gotcha:** lineups dedup on `(year, date, round_num, team_name)` via
`processed_lineup_keys`. A plain scraper re-run **skips** rows already on disk —
so corrupt rows are NOT overwritten. To repair, first DELETE the bad rows, then
force `_process_year(year, ...)` for the affected seasons (a normal delta run
only re-scrapes from the last-processed year). Full backfill recipe is in
`docs/experiment-log.md` (2026-07-09 entry).

The 2025-R3→2026 garbage-lineups bug (jersey numbers instead of names) was a
code regression, not a site change — see [[data_no_position]] for the related
fact that player CSVs lack positions.
