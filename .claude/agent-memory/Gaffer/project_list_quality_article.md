---
name: project-list-quality-article
description: 2026-06-17 list-quality/draft-pipeline article — grade system, † derivation convention, concurrent squad-rebuild agent
metadata:
  type: project
---

`docs/news/2026-06-17-afl-2026-list-quality-draft-pipeline.md` has 18 club sections, each with a National Draft table (Grade from DraftGuru) and a pathway table (Grade derived from career games, tagged †). Grade scale: A+ 200+ / A 150–199 / B+ 100–149 / B 75–99 / C+ 50–74 / C 25–49 / D <25.

**Why:** Grades are filled deterministically from `data/drafts/draftguru_enrichment.csv` (verbatim `grade`, matched by `(year,pick)`) and `data/player_data/` game counts — never authored from memory. DraftGuru uses short first names (Mitch/Oli/Cam/Dan/Brad) so dash-grade Table-1 rows resolve by (year,pick), not name. Deep ND picks (>~80) and pre-2004 players are absent from DraftGuru → derive from games, tag †.

**How to apply:** A concurrent squad-rebuild agent edits this same article and can leave malformed 3-cell pathway rows (e.g. Hawthorn "Unknown #NN" → all —; a real player with pathway dropped → pathway —, derive grade from the lone game count). Always `git pull --rebase` immediately before pushing edits to this file — see [[feedback_parallel_council_commits]]. Stamp uses the multi-line DONE/PASS form and is accepted by check-council-stamp.sh.
