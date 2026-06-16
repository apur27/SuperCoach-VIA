---
name: Draft schools CSV games column not reconciled to canonical per-player games
description: data/drafts/afl_draft_schools.csv has its own pre-computed games column that can disagree by 1 with the canonical per-player figure; cite per-player as truth
metadata:
  type: project
---

`data/drafts/afl_draft_schools.csv` carries its own pre-computed integer `games` column (and `goals`). It is NOT fully reconciled to the canonical per-player computation.

**The canonical games figure** is `max(row_count, max_numeric(games_played))` from the per-player file in `data/player_data/` — see [[hof_games_counter_gotcha]] (Scientist memory). `games_played` is stored as a string with ↑/↓ markers and must be numerically coerced; the row count is the floor.

**Known divergence (verified 2026-06-16):** Nathan Fyfe — draft CSV `games`=248, canonical per-player=247 (row_count 247, max_gp 247). Off by one. Franklin, Pendlebury, Neale, Josh Kennedy (Syd/Haw mid) all agreed (354 / 435 / 308 / 290).

**Josh Kennedy disambiguation trap:** there are two Josh Kennedys in the data. The 2006 pick-40 Sydney/Hawthorn midfielder (Xavier College, A+, 290 games) is file `kennedy_josh_20061988_*`. The `kennedy_josh_25081987_*` file is the WEST COAST/Carlton key forward (2005 pick 4, 293 games, 723 goals) — a different player. Do not conflate them; the 2006 superdraft A+ cohort uses the midfielder (290).

**Why:** the 2024 draft-analysis article (`docs/articles/afl-draft-analysis-2025.md`) cited the draft CSV for its pick/games tables, and DataSentinel flagged Fyfe as a mismatch because it row-counted while the draft CSV had a different number. The dispute was only resolved by computing the canonical figure explicitly.

**How to apply:** when an article table cites `afl_draft_schools.csv` for games, treat the per-player canonical figure as truth, not the CSV's `games` column. If they disagree, use the canonical per-player number and expect DataSentinel to flag the CSV column — that flag is the CSV being stale, not the article being wrong. A one-line methodology caveat ("career games use max(row_count, max(games_played)) from per-player files, which can differ by 1 from the draft CSV's games column") pre-empts the gate noise.
