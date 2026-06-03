---
name: team-name-canonicalisation
description: AFL team name normalisation oddities encountered in BriefBuilder runs
metadata:
  type: feedback
---

## Confirmed canonical forms (as used in data/matches/ and data/lineups/)

- "Brisbane Lions" (not "Brisbane", not "Lions")
- "Greater Western Sydney" (not "GWS", not "Giants")
- "Fremantle" (not "Fremantle Dockers")
- "St Kilda" (not "St. Kilda")
- "Western Bulldogs" (not "Bulldogs", not "Western Dogs")
- "North Melbourne" (not "Kangaroos")
- "Gold Coast" (not "Gold Coast Suns")
- "Port Adelaide" (not "Port")

**How to apply:** Always cross-check user input against data/lineups/ filenames and matches_2026.csv team_1_team_name column before beginning any brief. Correct silently and note in confirmation line.

Related: [[h2h-window-patterns]]
