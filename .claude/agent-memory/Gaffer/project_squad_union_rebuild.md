---
name: squad-union-rebuild
description: 2026-06-19 senior instruction — club squads in list-quality article must be dedup union of ALL R1–latest selected-22s, not just latest round
metadata:
  type: project
---

Senior instruction (2026-06-19): each club's squad in
`docs/news/2026-06-17-afl-2026-list-quality-draft-pipeline.md` must be the
DEDUPLICATED UNION of every player in any selected-22 lineup from R1 through
latest available round (R14/R15 by club), not just the single most-recent round.

**Why:** Original tables showed only the latest round's 22, missing players
injured/rested in the latest round but featured earlier (e.g. Sam Lalor at
Richmond, 2023/2024 pick 1; Jake Lloyd at Sydney). Supersedes the earlier
Sam-Lalor point-patch.

**How to apply:** Routed as a data-derivation cycle — Scientist owns the
pipeline + the 18 club [data] tables; DataSentinel Pass 2 gates; Skeptic; then
Gaffer ships. Key data quirks for the pipeline: round_num must be
`pd.to_numeric(errors='coerce')` before sort/filter (string-sort bug);
brisbane_lions suffix; GWS perf files use 'Greater Western Sydney'; jersey map
keyed `(team, jersey_num)->name` from 2026 perf files; name key
`re.sub(r'[^a-z]','',name.lower())`; alias table (Cam/Cameron Rayner, Jordan
De Goey/Goey, Jacob van Rooyen/Rooyen). Intermediate JSON:
/tmp/squads_all_rounds_2026.json.

SHIPPED 2026-06-19 (commit 359191707, all 18 clubs). Chain on this cycle:
DataSentinel FAIL (St Kilda Sinclair wrong-number) -> fixed -> PASS pass1+pass2;
Skeptic BLOCK (jersey-collision + ranking contradiction) -> Scientist sweep
fixed 5 collisions + 5 comparatives -> re-gate PASS_WITH_NOTES -> SHIP. The
second-stage `name -> pick/games/grade` lookup must be team+era scoped, NOT the
unscoped name key — see [[feedback-collision-gate-layering]]. BACKLOG GAP: the
generator is an uncommitted ad-hoc script (only the /tmp JSON output exists), so
the collision fix has NO committed regression test; if/when the generator is
committed, add a failing-first test reproducing the cross-club same-surname
collision (CLAUDE.md TDD). Related: [[project_player_data_quirks]].
