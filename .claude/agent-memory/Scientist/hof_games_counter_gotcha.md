---
name: HoF games counter — use games_played column, not row count
description: compute_stat_leaders.py historically used row count for career games; that undercounts active players because drawn GFs and some finals are collapsed in per-player CSVs. Switched to df['games_played'].max().
type: project
---

`docs/hall-of-fame/compute_stat_leaders.py` originally computed career games as `df["team"].notna().sum()` (row count). This is wrong for any player whose career includes drawn Grand Finals or finals appearances that the FanFooty-derived per-player CSVs collapse — Pendlebury's file has 429 rows but the canonical AFL games tally is 432, and the per-row `games_played` column carries the correct running count.

**Why:** the Harvey-vs-Pendlebury joint 432 milestone (R10 2026) was hidden by the stale row count, which had Pendlebury ranked #2 at 428 in `_stat_leaders.json` even though he was actually tied for #1 at 432. The hand-edited `hall-of-fame-stat-leaders.md` already showed the tie, but the JSON and the auto-generated chart did not, so the next regen would have silently undone the fix.

**How to apply:**
- Whenever counting games from a player performance CSV, prefer the max of `games_played` after stripping the `↑` / `↓` season-debut markers and coercing to numeric. Fall back to row count only if the column is missing.
- After regenerating `_stat_leaders.json`, verify ties surface with `rank_label` like "1=" / "1=" rather than 1 / 2.
- `chart_wall_of_records` was patched to join tied co-holders with " = " (e.g. "Brent Harvey = Scott Pendlebury 432") rather than silently surface only the first.

Related memory: [player_csv_date_format.md] documents the row-count-vs-games-played gap.
