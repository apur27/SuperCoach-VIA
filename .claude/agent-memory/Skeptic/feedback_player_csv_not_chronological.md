---
name: player-csv-not-chronological
description: player_data performance CSVs are NOT sorted by round; naive .tail(5) gives wrong last-5 means
type: feedback
---

Player performance_details.csv files in `data/player_data/` are NOT stored in chronological round order. Rows can appear with recent rounds (e.g. R10-13) in the middle of the file and earlier rounds (R5-9) at the end.

**Why it matters:** any "last 5" computed as `df['disposals'].tail(5)` silently grabs the wrong 5 games. On the Brisbane-vs-Fremantle R13 2026 brief, Wilmot's doc "last-5" (27.2) matched neither the true chronological last-5 (23.0, via `sort_values(['year','round']).tail(5)`) nor the naive tail-5 (25.6) — the numbers were unreproducible under ANY rule, which is the tell that the interpretation layer was written against numbers nobody can regenerate.

**How to apply:** when spot-checking a `**[data]**` last-N or season-mean tag, always `sort_values(['year','round'])` before `.tail()`. Compute BOTH the chronological last-5 and the naive tail-5; if the doc's figure matches neither, escalate as a Sentinel-failed CRITICAL, not a rounding nit. A doc whose numbers reproduce under no interpretation means Data Sentinel was not run or did not catch it.
