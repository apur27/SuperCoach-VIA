---
name: draft-scraper-wikipedia-schema
description: AFL draft scraper (scrapers/draft_scraper.py) parses Wikipedia {year}_AFL_draft; era schema drift gotchas and table-selection rule
metadata:
  type: project
---

AFL draft history is scraped from Wikipedia (`{year}_AFL_draft`), NOT afltables
(404s). Code: `scrapers/draft_scraper.py`, tests `tests/unit/test_draft_scraper.py`,
output `data/drafts/afl_draft_history.csv` (year,round,pick,club,player_name,
recruited_from). 2851 rows, 1990-2025.

**Why:** these are non-obvious, recurring Wikipedia structural quirks that a
re-scrape or a sibling scraper (e.g. rookie draft) will hit again.

**How to apply / the four gotchas that caused silent data loss:**

1. TABLE SELECTION by section heading, not index. From 2018+ a page has several
   Round/Pick/Player tables (mid-season rookie, national, pre-season, rookie) in
   that order. "First match" grabs the mid-season draft -> only 2-6 picks. Rule:
   pick the table whose nearest preceding `<h2/h3>` contains "national draft".
   1997's first matching table was "1998 rookie draft" -> would mis-scrape the
   wrong draft type entirely. 1992 has NO national-draft section -> correctly
   skipped (genuine source gap, not a bug). 1989 and earlier 404.

2. ROWSPANNED Round cell (2018+). Round number written once per round via
   `rowspan`; picks 2..n of a round omit the Round `<td>`, shifting all cells
   left so the Pick column lands on a name and the whole round after pick 1 is
   dropped. Fix: expand the table to a dense grid honouring rowspan/colspan
   (`_expand_rows`) before indexing by column.

3. DRAFTING-CLUB column has 4 era-dependent header names, all meaning the same
   thing: "Club" (modern), "Recruited to" (2000), "Drafted to" (2012),
   "Recruited by" (1993). Must NOT be confused with "Recruited from" = the
   PATHWAY club (state-league/TAC team). Mapping only "Club" left 948 club cells
   NaN. `_CLUB_HEADERS` preference order.

4. ROUND/PICK header variants: round = "Round" | "Rd." | "Rd", or ABSENT in the
   1990s single-round national draft (then default round=1). Pick = "Pick" or
   "#" (1990/1991). "Priority" in the Round column -> round=0.

Validation expectation: a real national draft year is ~59-112 picks. Anything
under ~30 means table selection or rowspan parsing broke. 90 missing
recruited_from + 21 missing round in the final CSV are genuine Wikipedia
omissions, not parser bugs.
