---
name: team-name-canonicalisation
description: Read canonical AFL team names LIVE from the data, never hardcode; slug rule + the recurring gotchas
metadata:
  type: feedback
---

**Rule: never hardcode team names — read the canonical form live from the data at the start of every brief.** (Consolidates three former duplicate memories, A-01.)

**Live source of truth (pick one, don't trust memory):**
- `data/matches/matches_<year>.csv` — columns `team_1_team_name` / `team_2_team_name`.
- or the `team` column of the latest `data/prediction/backtest/backtest_by_team_*.csv`.

With Bash now available, verify with an executed command, e.g.:
`python -c "import pandas as pd; d=pd.read_csv('data/matches/matches_2026.csv'); print(sorted(set(d.team_1_team_name)|set(d.team_2_team_name)))"`

**Recurring gotchas (verify live, but these are the ones that bite):**
- "Greater Western Sydney" (not "GWS", not "Giants")
- "St Kilda" (not "St. Kilda")
- "Western Bulldogs" (not "Bulldogs" / "Footscray")
- "Brisbane Lions" (not "Brisbane" alone)
- "North Melbourne" (not "Kangaroos"), "Gold Coast" (not "Suns"), "Port Adelaide" (not "Port")
- "West Coast" (not "West Coast Eagles"), "Essendon" (not "Essendon Bombers"), "Fremantle" (not "Dockers")

**Slug rule (filenames):** lowercase, hyphen-separated, drop spaces/dots — "St Kilda" → `stkilda` (no hyphen), "Greater Western Sydney" → `greater-western-sydney`, "West Coast" → `west-coast`.

**How to apply:** cross-check the user's two team inputs against the live source before assembling; correct silently and note the canonical form in the confirmation line.
