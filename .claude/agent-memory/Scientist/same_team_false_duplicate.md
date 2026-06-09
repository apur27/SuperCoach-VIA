---
name: same-team-false-duplicate
description: Two same-team players' performance CSVs look like duplicates because they share opponent/round/team columns; verify on STAT columns, not fixture columns
metadata:
  type: project
---

Same-team player performance CSVs (e.g. newman_nic vs cripps_patrick, both Carlton 2026) will share the exact `round`, `opponent`, `team`, and `result` columns because they play the same fixture. A glance makes them look like duplicates. They are NOT.

**Why:** A R14 brief flagged Newman's 2026 section as an "exact duplicate" of Cripps's. It wasn't — zero of 10 shared rounds had identical stat-lines. Coincidence was R14 both totalling 23 disposals, but k/h split, jersey (#24 vs #9), tackles, etc. all differed. Newman is a kick-heavy defender; Cripps a handball-heavy mid — opposite profiles.

**How to apply:** To test a suspected duplicate between two players, compare on STAT columns only (kicks, handballs, marks, tackles, clearances, jersey_num, contested_poss) after indexing on round — never on opponent/team/round. Also confirm `disposals == kicks + handballs` per row (internal consistency) and check `jersey_num` differs. Don't change a file on a duplicate flag until a full stat-line identity test returns non-empty.

Side note found while investigating: `data/matches/matches_2026.csv` was missing the Round 10 Carlton-vs-Brisbane row (jumps R9->R11). Player files correctly had R10. Gap is in the match file, not player data.
