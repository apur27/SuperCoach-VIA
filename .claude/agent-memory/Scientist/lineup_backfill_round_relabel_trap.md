---
name: lineup-backfill-round-relabel-trap
description: BL-03 lineup repair — a re-scrape can legitimately return a row under a DIFFERENT round label, so key-based "did it come back?" checks undercount; and it can add a duplicate fixture to matches_<year>.csv
metadata:
  type: project
---

Repairing `data/lineups/` (BL-03, done 2026-07-28) exposed two traps that will
recur on any delete-then-re-scrape backfill. Both look like data loss/corruption
but aren't — diagnose before "fixing".

## Trap 1 — a repaired row can come back under a different round label

Verification that matches deleted rows to restored rows on the dedup key
`(year, date, round_num, team_name)` will report false "unrecoverable" rows.
afltables relabels postponed games: the 2025 Brisbane Lions v Geelong game at
the Gabba, played **2025-03-29 18:35**, is labelled **Round 1** (the Cyclone
Alfred postponement of the R1 opener), not the Round 4 its date implies.

The old buggy scraper had written that fixture **twice** — once under R4 and
once under R1, byte-identical garbage. The re-scrape repaired the R1 copy in
place and correctly left the R4 phantom gone.

**How to apply:** when a backfill reports N unrecovered rows, before declaring
loss, re-query by `(date, team_name)` ignoring `round_num`. Only rows absent
under the *date* key are genuinely missing. Net counts will legitimately shrink
when phantom duplicates are dropped (34,017 -> 34,015 here; 2025 434 -> 432).

## Trap 2 — `_process_year` can ADD duplicate fixtures to matches_<year>.csv

FIXED 2026-07-28 in `build_match_key` (now `date` + sorted team pair, with a
legacy date+round+venue fallback) and at the `_process_year` write site.

The old key `(date, round_num, venue)` was blind to the two fields that actually
move upstream, so a re-scrape re-added games already on disk:

* **venue alias** — afltables publishes the 2025 Gather Round games as
  `Barossa Park`; the file held `Barossa Oval`. Two rows, identical on every
  other column, double-counted.
* **round relabel** — the Cyclone Alfred fixture (Brisbane v Geelong, Gabba,
  2025-03-29) is R1 upstream, was stored as the R4 its date implies.

`matches_2025.csv` went 215 -> 219 on re-scrape (1 genuinely missing R2 M.C.G.
game + 2 venue-alias dupes + 1 round-relabel dupe) and is now **216**.

**Why the team pair is the right identity:** venue can be renamed and the round
label can move, but who played cannot change. It still keeps simultaneous
same-date/same-round kickoffs distinct (the original reason venue was in the
key), and sorting the pair survives home/away column swaps.

**How to apply — do NOT verify dedup with the dedup key.** I checked for dupes
on `(date, round_num, venue)`, the very key that cannot see an alias, and
declared it clean; Gaffer caught it on `(date, teams)`. Always verify on an
*independent* key. And **`audit_match_rounds()` cannot catch this class at all**
— a duplicate makes a round look MORE complete, never short, so it reported
"complete, zero warnings" over a double-count. It only detects shortfalls.

Related: [[delta_scraper_approx_date_drop]], [[match_completeness_gate]],
[[lineup_scraper_structure]], [[dedup_dtype_mismatch_doubling]].
Guard against regression: `tests/integration/test_lineup_integrity.py`.
