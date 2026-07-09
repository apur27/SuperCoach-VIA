---
name: project_stat_leaders_json_gotchas
description: Two gotchas when independently verifying docs/hall-of-fame/_stat_leaders.json against source data — corrected paths/casts to use
metadata:
  type: project
---

Two traps when writing an ad-hoc verification snippet against the HOF pipeline
(discovered during the 2026-07-07 Round 19 QA gate):

1. **Schema path (CR-2).** `_stat_leaders.json` top-level keys are
   `['meta', 'categories', 'single_season']`. Career-stat leaders live at
   `categories.<stat>.leaders`, e.g. `categories.career_games.leaders[0]`. A
   snippet that reads `leaders['career_games']` at the top level will KeyError
   — that's a bug in the snippet, not a data failure. `career_games` leader
   entries do not carry a `player_id` field (it's `None`); match against
   source CSVs by name (surname_firstname glob), not player_id.

2. **`games_played` dtype in per-player CSVs.** The column is stored as
   `object` (string), not int. Calling `.max()` directly on it does a
   **lexicographic string comparison**, not numeric — so a modern 400+-game
   player's max can come back as `"99"` (since `'9' > '4'` as characters),
   which looks like a wild regression but is a QA-snippet bug. Always cast
   with `pd.to_numeric(df['games_played'], errors='coerce').max()` before
   comparing to the JSON `total`. Verified 2026-07-07: once cast correctly,
   Pendlebury/Harvey/Tuck/Burgoyne/Bartlett (top-5 career_games) all match
   the JSON exactly.

Also: `docs/hall-of-fame/_stat_leaders.json` is gitignored
(`docs/hall-of-fame/.gitignore` excludes it) — there is no git-tracked prior
version to diff against for regression detection. The authoritative
regression gate is `scripts/check_hof_numbers.py` (compares JSON rank-1 totals
against the rendered HOF-TOP sentinel row in each subpage, exit 0/1) plus
direct source-CSV verification as above; do not expect `git diff` on this file
to show anything.

**Why:** without this note, a future QA run will waste a cycle chasing a
phantom "games count dropped" regression that is actually just Python
comparing strings instead of numbers.

**How to apply:** whenever independently re-verifying HOF numbers (i.e., not
just trusting `check_hof_numbers.py`'s exit code), use the corrected path and
numeric cast above. See also [[project_council_stamp_gate]] for the related
"badge line" naming gotcha — the trust badge text is
`✓ All N stats verified against source data · council-pipeline-gated · <date>`,
not the literal word "trust", so a `grep -i trust` on HOF pages returns 0 hits
even when the badge is correctly present.
