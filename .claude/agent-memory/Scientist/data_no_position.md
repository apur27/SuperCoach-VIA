---
name: No position column in player data
description: SuperCoach-VIA player CSVs and personal_details have no position field; any per-position analysis is currently un-sourced
type: project
---

The `data/player_data/*_performance_details.csv` files do not contain a player `position` column. The companion `*_personal_details.csv` files only have `first_name, last_name, born_date, debut_date, height, weight` - no position either.

**Why:** Discovered while building `backtest.py` (2026-04-30). The user requested per-position aggregate metrics, but no source field exists. Backtest currently emits `position="Unknown"` for every row so the schema is stable.

**How to apply:**
- Any analysis that asks for per-position breakdowns needs a position source wired up first (likely a scrape extension to `player_scraper.py`, or an external lookup).
- Don't fabricate positions from height/weight or guess from kicks/handball ratios - flag that the source is missing and ask before inferring.
- The placeholder string used in `backtest.py` is the literal `"Unknown"`.
