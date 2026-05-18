---
name: Live pipeline misfires past end-of-game
description: The FanFooty live-polling pipeline keeps appending blocks to earlier-quarter docs after Full Time, stamping end-of-game scores onto Q1/Q2/HT/Q3 routing messages
type: project
---

The `scripts/live_match_monitor.py` polling loop (90s interval, runs until status flips to Full Time) sometimes keeps writing to earlier-quarter docs *after* the game has ended. The result is that Q1/Q2/HT/Q3 docs accumulate "Final Siren" or end-of-game "QUARTER BREAK" blocks at the top with the *final* score in the header, e.g. `### QUARTER BREAK: end of Q1 - St Kilda 16.13.109 vs Richmond 11.7.73` appearing in the Q1 doc.

**Why:** the autopipeline appears to route the latest snapshot to whichever doc was last active, and isn't strict about closing earlier-quarter docs at the quarter break. Saw this in R11 2026 STK v RIC where Q1 alone was 75KB with ~8 bogus end-of-game blocks at top.

**How to apply:** When cleaning up live docs, filter auto-blocks by both:
- header h3 keyword (Q1/Q2/HT/Q3/Q4-scoped — drop "Final Siren" or "Full Time" from Q1/Q2/HT/Q3)
- score-magnitude sanity: STK score in the header should be <= a per-scope cap (Q1: 50, Q2/HT: 80, Q3: 100, Q4: any). Catches QUARTER BREAK blocks with corrupt-scope scores in the header even though they don't carry timestamps.

Pruning recipe lives in commit 5ae20feee — Python parses on `^---$` separators, classifies chunks as `substantive` (Scientist/FootyStrategy long-form), `static-tail` (Scoreline/Disposal leaders/etc), or `autoblock`, then for autoblocks keeps: first-in-scope, last-in-scope, one instance of each distinct transition header, and margin-jump-of-6 since last kept block.
