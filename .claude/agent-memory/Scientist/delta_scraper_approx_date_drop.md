---
name: delta-scraper-approx-date-drop
description: Player delta-scraper drops a genuinely-new game when afltables labels it with a non-chronological (low) round number; fixed with counter-aware delta
metadata:
  type: project
---

The player delta scraper (`scrapers/player_scraper.py`) filters "already seen"
games by an APPROXIMATE date `datetime(header_year,3,1)+weeks(round-1)`. That
date is NOT monotonic with real chronology: afltables sometimes labels a
late-season game with a low round number (e.g. Angus Clarke's career game #14 —
a 2025-08-27 Essendon-v-Gold-Coast game shown under the `Essendon - 2025` table
with round label "1"). Its approx date `2025-03-01` fell before `since_date`
(last-recorded game), so the delta filter dropped it while the genuinely-later
2026 game survived → a `games_played` counter gap (13→15) that fail-closed the
phantom-row gate in `refresh_data.py`. Manifested in the 2026-07-13 weekly run.

**Two filters can drop such a row:**
- Gate 2 (per-game): `if since_date and game_date <= since_date: continue`
- Gate 1 (per-year): `if year_int < since_date.year: continue` — strands the row
  ONCE the file already advanced to a later season (the post-drop state).

**Fix (counter-aware delta):** the `games_played` career counter is the
authoritative "already seen" signal. `_scrape_player_performance_details` now
takes `max_counter`; a row whose counter > max_counter is kept regardless of its
approx date. `_process_player` computes it via new `_get_max_counter(file)`.
Gate 2 fixed; Gate 1 left as-is (surgical) — the counter fix keeps #14 and #15
together on the SAME run so stranding never starts. Residual: a retroactive
afltables backfill into a season < file's max-year would still hit Gate 1, but
the phantom gate catches it (fail-closed) and the repair is a full re-scrape.

**Repair recipe for an already-stranded file:** delete the `_performance_details.csv`
and re-run `_process_player` (since_date=None → both gates bypassed, fresh write,
no dedup). See [[finals_gap_rescrape_recipe]].

**Interrupted-harness side effect:** when the phantom gate aborts, the CSVs have
already grown but downstream artifacts (HOF JSON / stat leaders) have NOT been
regenerated. So `test_qa_rank1_cross_check_runs` (and similar CSV-vs-JSON cross
checks) can fail with e.g. "CSV says 437, JSON says 436" — pre-existing, resolves
on the successful full re-run. NOT a scraper bug. Do not hand-regen the JSON
(bypasses the gate chain — CLAUDE.md serialize-writes rule).
