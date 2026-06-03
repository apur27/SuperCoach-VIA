---
name: matches-2026-r10-data-gap
description: R10 2026 is missing from matches_2026.csv for almost all teams — only Richmond vs Adelaide is present
metadata:
  type: project
---

Round 10 of the 2026 season is missing from `data/matches/matches_2026.csv` for all teams except Richmond vs Adelaide.

**Why:** Confirmed during WB vs Collingwood R13 brief assembly. Player `performance_details.csv` files clearly show R10 entries for both WB (vs Port Adelaide, W) and Collingwood (vs Geelong, L) and multiple other teams. The matches file does not contain those rows.

**How to apply:** When computing season records from `matches_2026.csv`, always check whether the team count of games aligns with the rounds played. If a gap is found at R10, supplement W/L from player files and explicitly caveat that For/Against/% excludes R10. Flag for data team. Do not invent R10 scores — they are not available in any verified source file in the repo.
