---
name: Hall of Fame data verification workflow
description: How to verify player stats in HOF pages against repo data, common errors to watch
type: feedback
---

When reviewing or updating Hall of Fame pages (`docs/hall-of-fame-*.md`), the data verification workflow is:

1. Use `/home/abhi/sourceCode/python/coding/.venv/bin/python` with pandas.
2. Player CSVs at `/home/abhi/git/SuperCoach-VIA/data/player_data/<surname>_<firstname>_<DDMMYYYY>_performance_details.csv`. Match against partial pattern (e.g. `surname_firstname`) - many surnames have multiple entries (e.g. multiple Trengoves, multiple Newmans). Pick the longest career match.
3. Match CSVs at `data/matches/matches_<year>.csv` with columns `team_1_team_name`, `team_2_team_name`, `team_1_final_goals`, `team_1_final_behinds`, etc. `round_num` can be `"Grand Final"`, `"Semi Final"`, `"Preliminary Final"`, `"Qualifying Final"`, `"Elimination Final"` or a number.

**Recurring data errors found in HOF text:**
- Hawthorn 1978 GF: text said "lost to North Melbourne" - data shows Hawthorn 18.13 def North 15.13 (Hawthorn WON).
- Hawthorn 1991 GF: text said "lost to West Coast" - data shows Hawthorn 20.19 def West Coast 13.8 (Hawthorn WON).
- West Coast 1992 GF: text said "lost to Geelong" - data shows West Coast 16.17 def Geelong 12.13 (WC WON).
- Pendlebury did NOT captain the 2010 Collingwood flag (Nick Maxwell did); he became captain in 2014.
- Wanganeen did NOT captain Port Adelaide's 2004 premiership (Warren Tredrea did).

**Why:** These factual errors had propagated through text written from training-data memory rather than from the repo. The CLAUDE.md rule requires verification from data first.

**How to apply:** Before publishing any specific stat in HOF pages, run a quick verification script reading the per-game CSV and aggregating. Tag `**[data]**` only when the number is verified; use `**[historical record]**` for facts external to the repo's data; use `**[historical record - unverified in data]**` if the claim is specific but cannot be checked.
