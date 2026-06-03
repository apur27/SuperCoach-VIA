---
name: finals-doc-stale-2026
description: RESOLVED 2026-06-02 — afl-finals-2026.md stale round labels were fixed by the 1126df5b2 auto-refresh; prose now reads "After Round 13" consistently
metadata:
  type: project
---

RESOLVED as of 2026-06-02 (commit 1126df5b2, "Auto-update: refresh AFL insights,
predictions and backtest"). The finals doc prose now consistently reads "After
Round 13 ... most sides have played 12 games with roughly 10 games left" and every
team paragraph reads "sit Nth on the ladder after Round 13". The generator's
round-label interpolation appears to track the data now. Kept for history; the
original defect write-up follows.

---

`docs/afl-finals-2026.md` had a round/games-count inconsistency, found 2026-05-31.

The ladder NUMBERS (W-L-D, points, percentage) are correct — they reproduce
exactly from `data/matches/matches_2026.csv` through Round 12 (Fremantle 9-1
137.0%, Sydney 8-2 146.9%, ... Essendon 1-9 69.4%). The bug is purely in the
hard-coded prose round/games references:
- Intro says "After Round 12 ... every side has played 7 games with roughly 11 left"
  (12 rounds played ≠ 7 games; and most teams have played 10, two have 11).
- EVERY team paragraph says "sit Nth on the ladder after Round 8".
The sibling docs are internally consistent at "Round 12 / 11 games" wording
(afl-team-analysis-2026.md, afl-stat-leaders-2026.md, afl-brownlow-2026.md "after
Round 12").

**Why:** Almost certainly a template-string staleness in `update_team_analysis.py`
/ `refresh_readme.py` — the round-label in the finals-pathway generator was not
updated when the ladder data was. mtime of the file is 2026-05-25 (older than the
Round 12 data sync at 2026-05-31 commit a09cb582e).

**How to apply:** This is a SCRIPT bug, not a hand-edit fix — afl-finals-2026.md is
auto-generated, so hand-editing it will be overwritten next refresh. The fix
belongs in the generator's round/games-count interpolation. Also: matches_2026.csv
has 91 match rows; most teams = 10 games, Adelaide & Richmond = 11 (bye asymmetry),
so a single "every side has played N games" sentence is itself imprecise.
