---
name: phantom-row-dedup-gate
description: Player game rows silently dropped by (team,year,round,opponent) dedup collapsing drawn-final+replay; fixed with games_played key + phantom-row validator gate
metadata:
  type: project
---

A player-specific bug (Sidebottom missing 2010 drawn-GF row) was really a MISSING-GATE bug. Reframe: any surviving single-player data gap is a symptom of an absent process gate, not a one-off patch.

**Root cause:** `scrapers/player_scraper.py:273` deduped on `(team,year,round,opponent)`. A drawn final + its replay share all four keys, so `drop_duplicates(keep='last')` deleted the draw and kept the replay. Same bad key was used by one-time cleanup commit 58f1a4f20.

**Detection invariants (validated across all 13,343 player files):**
- DETERMINISTIC counter-gap check = `games_played` set must equal {1..max}, no gaps/dups. Zero false positives, and does NOT flag legitimately-missed games (counter only increments for games played). Caught Sidebottom (missing counter 35) and nothing else.
- BLIND SPOT: a dropped-then-renumbered row leaves a contiguous counter (rowcount==max) → counter-gap check can't see it. Needs a finals-specific matches cross-check scoped to drawn-final years only (rare) to avoid injured-player FP explosion. Goldsack was the candidate blind-spot case (1 GF row, contiguous 165).

**Why:** The existing audits (`audit_player_career_totals`, `audit_match_rounds` in scrapers/game_scraper.py, wired in refresh_data.py ~line 180-235) check career TOTALS and match-file round coverage, but nothing checked per-game row COMPLETENESS.

**How to apply:** The fix is `scripts/phantom_row_validator.py` wired into refresh_data.py post-scrape (WARNING historical / hard-abort current season), plus dedup key now includes games_played. If a future "one player's stat is wrong/missing" report appears, ask what GATE would have caught it before patching the player. Dedup keys on player rows must include the game-distinguishing counter, never just (team,year,round,opponent). See [[project_2024_finals_dup_rows]], [[project_drawn_gf_dedup_defect]], [[feedback_canonical_games_metric]].
