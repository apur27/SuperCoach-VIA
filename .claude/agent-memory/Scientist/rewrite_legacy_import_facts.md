---
name: rewrite-legacy-import-facts
description: Non-obvious facts found building the supercoach_via legacy importer (2026-09-25): 97% synthetic player dates, 2026 missing-player gaps, malformed finals rows, lineup multi-part surnames
metadata:
  type: project
---

Facts from the Phase 2 legacy import (src/supercoach_via/ingest/legacy.py, reconcile.py):

- **Player-row dates are ~97% synthetic**: 675,062 of 695,331 imported rows have `date` = March 1 + 7k days
  (incl. YYYY-03-01 finals placeholders). Only 20,269 equal the linked match date. Linking must use
  season+stage+club+opponent, never the row date; date is only a replay tie-break. See [[player_csv_date_format]].
- **2026 current-season gaps (validation FAIL, blocking)**: Flynn Perez (perez_flynn_25082001 stops 2023) and
  Jack Dalton (no file at all; only an 1876 namesake) play for Hawthorn 2026; Will Brodie (brodie_will_23081998
  stops 2023) for Port Adelaide 2026. Detected via lineups + Hawthorn R17/R18 goal-total mismatches.
  The DATA_REFRESH claim of "exact agreement between new lineups and identities" did not cover these.
- 3 malformed match rows have a score string as team_2 name (1994 QF, 2007 SF, 2017 EF: "15.24.114",
  "10.14.74", "10.16.76"); quarantined, so their ~130 player rows are unlinkable.
- Legacy numbering: from 2024 the Opening Round is labelled round "1" in both match and player files.
- Lineup tokens for multi-part surnames (Nick Dal Santo, Matt de Boer...) need a within-match
  first+last-word fallback (legacy personal details split the surname); 1,918 tokens resolved that way.
- personal_details: born 01-01-1900 is the legacy parser's default (5 files) -> unknown DOB; height/weight
  0 and -1 are sentinels.

**How to apply:** the promoted-snapshot gate will stay FAIL until those 3 players' 2026 rows are
refreshed from source; never add a current-season exception (validation rejects it by design).
