---
name: drawn-gf-dedup-defect
description: Commit 58f1a4f20 dedup wrongly deleted drawn-and-replayed finals rows (2010 GF draw); + leaders-hub aggregates are hand-written and silently drift
metadata:
  type: project
---

Two related data-integrity findings from the 2026-06-29 HOF refresh cycle (Scientist-adjudicated).

**1. Drawn-and-replayed finals rows are vulnerable to (year,round,opponent) dedup.**
Commit `58f1a4f20` ("Fix 2024 finals duplicate rows and stale dates across 121 player files") deduped on `(year, round, opponent)` and **wrongly deleted real games**. A drawn Grand Final and its replay share identical (year, round, opponent) but are TWO distinct games. It deleted Steele Sidebottom's 2010 drawn-GF row (game#35, result=D, 19 disp), understating his career totals (disposals 8367→8386, kicks 4799→4813 after restore). Games count stayed right only because the canonical metric recovered it (see [[canonical-games-metric]]).
- **Why it matters:** Other drawn-and-replayed finals exist historically (1948, 1977 GFs, plus drawn finals in other rounds). The Scientist's corpus-wide scan (counter>rowcount + distinct-game-number deleted) found blast radius = Sidebottom ONLY this time, but the detector relies on the stored `games_played` counter being stable; any player whose counter was also truncated would be invisible. Treat drawn-replay finals as a known fragile class.
- **How to apply:** Never dedup player rows on (year,round,opponent) alone — drawn finals are legitimate same-tuple duplicates. Distinguish by result/date/game-number. Related: [[2024-finals-dup-rows]], [[player-data-quirks]].
- **Residual:** restored 2010 GF rows initially carried placeholder date 2010-08-30 (the surviving replay row's pre-existing bad date); corrected to 2010-09-25 / 2010-10-02. A bounded corpus check for similar out-of-window finals placeholder dates was requested — log scale to backlog if broad.

**2. Backlog gap — leaders-hub aggregates drift silently.**
`scripts/update_hof_pages.py` only renders rank-1 (HOF-TOP), the hub nav, and the date stamp. **Leaderboard ranks 2-20 AND the hub's per-player kicks/handballs/ratio figures are hand-written `[data]`** not driven by `_stat_leaders.json`, so they silently drift for active players (this cycle: Pendlebury handballs 5,531→5,543, file count 13,329→13,343 were stale at HEAD = a LIVE published defect). Neither the renderer nor a full `weekly_refresh.sh` heals them. Scientist recommends extending the renderer to emit full top-20 tables + hub aggregates from JSON (data exists, TDD-able). High-value backlog item.
