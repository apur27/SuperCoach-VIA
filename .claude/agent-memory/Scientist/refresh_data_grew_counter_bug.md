---
name: refresh_data.py "0 active files grew" summary is unreliable
description: The "Refresh summary: N active files grew" log line in refresh_data.py undercounts; trust git diff on data/player_data/ instead
type: project
---

The summary line `Refresh summary: N active files grew, 0 new files appeared, total active rows now X` printed at the end of `refresh_data.py` can report **0 files grew even when many player files received new rows**. Observed 2026-05-19: log said "0 active files grew" but `git diff` showed ~400 player CSVs with new R11 rows added.

The likely cause is that the `before_rows` snapshot uses one path/key scheme while the post-scrape comparison uses a different one, so no diffs register even when rows were appended.

**Why:** Without this caveat you might wrongly conclude the scraper failed when in fact it succeeded.

**How to apply:** After running `refresh_data.py`, do NOT rely on the "files grew" summary number. Verify scrape outcome by either:
1. `git status --short data/player_data/ | wc -l` (count of modified player files)
2. Spot-check a known active player who definitely played the latest round (e.g. open their CSV tail and confirm the new row is present)
3. Check Pendlebury-style players who missed a round - their file *should* remain unchanged
