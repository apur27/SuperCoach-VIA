---
name: games-played-gap-detector
description: The naive games_played gap scan (to_numeric+dropna) has ~99% false positives from sub-annotation chars; strip non-digits first
metadata:
  type: feedback
---

The naive "missing finals rows" detector — `pd.to_numeric(df['games_played'], errors='coerce').dropna().astype(int)` then look for `diff() > 1` — is **massively false-positive-prone**. On 2026-06-21 it flagged **1453 players**; the true count was **17**.

**Why:** afltables writes substitute markers directly into the `games_played` cell — a down-arrow `↓` (subbed off) or up-arrow `↑` (subbed on), e.g. `19↓`. Each appears ~3,760 times across the dataset. `to_numeric(...).dropna()` silently DROPS every sub-affected row, so the surrounding `games_played` values look non-contiguous and manufacture a phantom gap of 1. The "confirmed-fixed" Andrews file (250 rows = max_gp, genuinely complete) still tripped the naive scan for exactly this reason.

**How to apply:** To detect genuinely missing game rows, extract the leading integer first, keeping the row:
`pd.to_numeric(series.astype(str).str.extract(r'(\d+)')[0], errors='coerce')`
Then a residual `diff() > 1` is a real missing row. Also note: a `+1` step across many skipped rounds is NOT a gap — the player simply didn't play those rounds; only a jump in the contiguous games_played counter signals a missing row.

**Fix recipe (worked for all 17):** delete the CSV (forces `_get_last_game_date`→None→full re-scrape, since `_process_player` is incremental and won't backfill mid-history otherwise), then `PlayerScraper()._process_player("players/{First[0]}/{First}_{Last}.html", PLAYER_DIR)`. URL folder = first-name initial. `get_soup` already sleeps 0.5s. Back up first; restore if new rows < orig. Scripts: /tmp/scan_v2.py (corrected scan), /tmp/fix_gaps.py (batch). All gaps were missing 2025 finals (jump row tagged 2026 because it's the season opener after the gap). Related: [[player_csv_date_format]].
