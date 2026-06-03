---
name: player-file-disambiguation-parallel-reads
description: When reading multiple player files in parallel, verify each file's content by checking jersey number AND career start year to avoid misattributing stats
metadata:
  type: feedback
---

When reading 5 player performance files in parallel, the returned output blocks can be confused if:
- Two players have similar career lengths (similar last-line numbers)
- Two players at the same club have overlapping jersey numbers across eras
- Files are long and the 2026-era rows appear at the same approximate offset

**Safe approach:** After parallel reads, spot-verify each file by checking:
1. Career start year (player's debut year — must match the player's known debut from ARCHITECTURE or personal_details)
2. Jersey number consistency (should be stable within a club tenure)
3. If any file shows identical disposals sequences to another player's file, treat it as a sort-order transcription error first, not a data coincidence — re-read the file independently from scratch before flagging.

**Why:** In R13 2026 brief, Oliver Clayton's section was populated with Callaghan Finn's stats due to a sort-order error during parallel reads. The initial draft flagged this as a "data coincidence" and raised a DataSentinel issue — but the root cause was wrong-file transcription, not matching source data. Clayton's correct 2026 sequence is 26,26,33,37,31,37,26,32,26,38,37 (season mean 31.7, last-5 mean 31.8), materially different from Callaghan's 26,27,32,35,37,26,21,28,30,19,31,31 (season mean 28.6, last-5 mean 27.8).

**How to apply:** For any brief with multiple players at the same club, do targeted re-reads (offset + limit) to isolate each player's 2026 rows independently. Never trust a sequence that looks identical to another player's — re-read the source file directly. A Glob check on different DOBs is necessary but not sufficient.
