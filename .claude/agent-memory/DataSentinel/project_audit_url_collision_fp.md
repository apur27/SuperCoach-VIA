---
name: audit-url-collision-fp
description: player-audit career-total WARNINGs are often false positives — URL builder discards DOB, collides same-name players against wrong afltables page
metadata:
  type: project
---

The weekly-refresh `[player-audit]` career-total reconciliation can emit WARNINGs
that are FALSE POSITIVES, not data errors. Confirmed 2026-06-22.

**Root cause:** `scrapers/game_scraper.py:_player_url_from_csv_path()` (line 249-270)
builds the afltables profile URL from name only. Filenames are
`<last>_<first>_<DDMMYYYY>_performance_details.csv`; the function uses `parts[0]`
(last) and `parts[1]` (first) but DISCARDS `parts[2]` (the DOB). Two players who
share name+initial collapse to the same URL, so the audit compares the CSV against
the WRONG person's career totals.

**Why:** known audit-side bug, not a data-file error. The player CSVs are correct.

**How to apply — triage rule for player-audit WARNINGs:**
- Open the actual CSV file the audit named and check its DOB-stamped filename vs the
  player you expect. Mismatched DOB or wildly divergent year-span = collision FP, SKIP.
- Tell: collision FPs show HUGE divergent totals (csv=175 games vs afltables=4) OR
  two files with different DOBs for the same name. Genuine lag shows delta of 1-2 on
  a single stat (a round not yet on afltables' running total).
- Confirmed collision FPs (do NOT cherry-pick re-scrape): rioli_maurice_01092002
  (vs Rioli Sr), kennedy_matthew_06041997, henry_jack_29081998 (real Geelong player
  vs a 4-game namesake), brown_callum (file is _27041998 played 2017-2022; audit hit
  _15082000 — two different Callum Browns), jones_arthur (file _12101891 played 1914
  only; audit warned _18072003).
- Tiny-delta genuine (same player, normal scrape lag, NOT corruption, no cherry-pick):
  rivers_trent (delta 1 disp), sinclair_jack (delta 1 disp), keane_mark (delta 1
  tackle / 35 disp = totals lag not a missing row).

Gap-audit script (games_played position gaps) ALSO false-positives on the
games-counter skip quirk: Sidebottom "missing position 35" is afltables' running
tally jumping 34->36 across the 2010 GF, not a missing match. See
[[project_player_data_quirks]] (counter quirk) and [[project_2024_finals_dup_rows]].
