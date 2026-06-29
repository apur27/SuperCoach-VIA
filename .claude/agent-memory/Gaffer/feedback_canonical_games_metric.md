---
name: canonical-games-metric
description: Career games = max(row_count, games_played.max()), NOT naive len(df) — naive rowcount induces false DataSentinel FAILs
metadata:
  type: feedback
---

When commissioning DataSentinel (or verifying yourself) to confirm a player's CAREER GAMES count, the canonical metric in this repo is **`max(row_count, cleaned games_played column .max())`** as implemented in `docs/hall-of-fame/compute_stat_leaders.py` (~line 104). Do NOT instruct an agent to use naive `len(df)` row-count.

**Why:** On the 2026-06-29 cycle I re-dispatched a DataSentinel run with the prompt "career games = row count of the CSV." It returned a FAIL on Steele Sidebottom (doc 366 vs len(df) 365). The original full DataSentinel run had correctly PASSED using the canonical `max(rows, games_played.max())`. The naive metric undercounts whenever a real game's detail row is missing while the stored `games_played` counter still remembers it (finals-lag, or — as it turned out — a wrongly-deleted row; see [[drawn-gf-dedup-defect]]). The Scientist confirmed 366 is the true count and the canonical algo is correct; do NOT "fix" the compute down to row-count — that reintroduces undercounting.

**How to apply:** In any DataSentinel/verification prompt that checks games totals, specify the canonical metric explicitly. CAREER TOTALS for other stats (disposals/kicks/handballs/etc.) ARE a plain column sum over all rows — the max() rule is games-specific. Note the silver lining: the naive-rowcount FAIL, though methodologically wrong, surfaced a genuine corpus defect — a row/counter mismatch is a useful smoke signal worth investigating even when the canonical games number is right.
