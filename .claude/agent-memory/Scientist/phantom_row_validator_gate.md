---
name: phantom-row-validator-gate
description: The drawn-GF dedup bug, its fix (games_played in dedup key), and the phantom_row_validator gate that guards against silently dropped player game rows
metadata:
  type: project
---

Silently-dropped player game rows are now gated. Built 2026-07 after a drawn-GF
dedup collapse deleted Steele Sidebottom's 2010 drawn Grand Final row.

**Root cause:** `scrapers/player_scraper.py` deduped appended rows on
`(team, year, round, opponent)`. A drawn final + its replay share all four
(Collingwood/2010/GF/St Kilda), so keep='last' deleted the draw.

**Fix:** `dedup_player_performance(df)` in scrapers/player_scraper.py adds the
normalised `games_played` counter (strips ↑/↓ arrows) to the dedup key. The
counter (35 draw vs 36 replay) is the authoritative per-game distinguisher.
`date` is NOT reliable for finals (placeholder YYYY-03-01 collisions).

**The gate:** `scripts/phantom_row_validator.py`
- Layer (a) `check_counter_gaps(path)` — DETERMINISTIC, zero-FP. games_played is
  a running career counter that ticks only for games played, so a complete file
  must be {1..max} with no gaps/dups. A gap == a deleted row. Legit missed
  rounds do NOT create gaps. Returns {missing, duplicated, counter_max, ok}.
- Layer (b) `find_drawn_finals` + `check_drawn_final_consistency` — the
  counter-RENUMBER blind spot (drop + renumber leaves a contiguous counter, so
  layer (a) is blind). Scoped to drawn-final years only. Single finals row for a
  finalist club → REVIEW (needs afltables); >2 rows or repeated result →
  WARNING. A single row is NOT auto-a-bug (squads change between draw & replay).
- `gaps_in_season(path, season)` attributes a gap to the season of the next
  surviving counter (conservative toward current season at boundaries).

**Wired into** refresh_data.py refresh_players(): over grown_paths, WARNING for
historical gaps, HARD ABORT (RuntimeError) if a gap lands in the current season.

**Known REVIEW no-action case:** Tyson Goldsack shows REVIEW SINGLE_FINAL_ROW for
2010 GF — afltables confirms he only played the replay (R=W, Gm56), so his single
row is correct. Do NOT "fix" it. See [[finals_gap_rescrape_recipe]].

Tests: tests/unit/test_player_scraper_dedup.py, test_phantom_row_validator.py,
test_drawn_gf_integrity.py (pre-authored, now passing).
