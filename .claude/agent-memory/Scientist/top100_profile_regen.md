---
name: top100-profile-regen
description: docs/hall-of-fame-top100.md has a table (marker block) + 100 narrative profiles that drift apart; how the auto-regen + gate keeps them consistent, and the membership-swap gotcha
metadata:
  type: project
---

`docs/hall-of-fame-top100.md` = auto-gen table inside `<!-- ALL-TIME-TOP100-START/END -->`
markers PLUS 100 narrative profile blocks (`### #N Name — Club` + italic stat-line +
prose) below the markers. They drift apart every season as active players climb.

**Source of truth for BOTH table and profiles:** root `all_time_top_100.csv`
(cols: Serial Number, Player Name, Footy Teams, Comment). Stats are parsed from the
Comment prose by `_parse_top100_comment()`. Score/rank order comes from
`data/top100/all_time_top_100.csv` (score-descending; row i == serial i+1).
The Comment fields are regenerated upstream from live player CSVs, so they already
match `data/player_data/*`. **Build profiles from bio_df/_parse_top100_comment, NOT
from live player CSVs directly** — that guarantees table==profile by construction and
survives a week where the Comment CSV lags the raw data.

**Why:** the profile section used to be frozen prose. It diverged badly (stale
stat-lines, wrong rank headings, out-of-order ranks). Fixed 2026-07-11 by
`regenerate_top100_profiles()` + `check_top100_consistency()` in
update_team_analysis.py, wired into main() step [13/14], with a hard-fail gate.

**⚠️ The step-[13/14] wiring was DEAD for 2 weeks (found 2026-07-25).** No harness
script invokes `update_team_analysis.py` / `main()` at all — grep both
`refresh_and_rank.sh` and `scripts/weekly_refresh.sh` for it and you get zero hits,
and no weekly log contains an `[N/14]` marker. What the harness actually runs is
`refresh_readme.py` (refresh_and_rank.sh, after `top_players_comprehensive.py`),
which **re-implements main() step-by-step** in `_step_team_analysis()` +
`_step_top100_markdown()`. That re-implementation rewrote the TABLE only and
silently omitted the profile pass — so the table advanced weekly while profile
stat-lines froze. Drift found: 8 hard mismatches (6 stale stat-lines on active
players + a Neale/Vallence rank swap). Fixed by adding the regen+gate to
`refresh_readme.py::_step_top100_markdown()` — NOT by wiring update_team_analysis.py
into a shell script (that would re-run all 14 steps and duplicate work).
**Lesson: `refresh_readme.py` is a hand-maintained mirror of `main()`; any new
step added to `main()` must be mirrored there or it never runs in production.**
Regression test: `tests/unit/test_refresh_readme_top100_gate.py`.

**Gates here must fail CLOSED.** Step [13/14] guards the regen with
`if os.path.exists(TOP100_CSV) and os.path.exists(TOP100_SCORES_CSV):` — an absent
CSV silently skips the gate but still writes the doc. Same bug class as the dead
wiring: a gate that quietly does not run. In `refresh_readme.py` this is now an
error-and-no-write (Gaffer review, 2026-07-25). Note the state is unreachable via
`generate_top100_section()` (it returns `""` if either CSV is missing, and the
caller already errors on an empty body) — so fail-closed costs nothing.
**The twin fail-open still stands at `update_team_analysis.py:5001`** — dead path
today, but re-syncing the mirror from it would reintroduce the bug.

**How to apply:**
- `regenerate_top100_profiles(hof_text, bio_df, scores_df)` reorders the 100 blocks
  into current rank order, rewrites headings/stat-lines, preserves prose verbatim,
  and is idempotent. `check_top100_consistency()` returns (hard, warnings); pipeline
  raises on any hard mismatch for a RANKED profile.
- **Membership-swap gotcha (recurs every season):** a player can DROP OUT of the
  top-100 (profiled but not in bio) and another ENTERS (bio but no profile). On
  2026-07-11 Jeremy Cameron dropped out; Dane Swan entered at #100. Handling:
  dropped players move intact to a `## Honourable Mention — Just Outside the Top 100`
  section (heading loses its `#N` so the ranked-block regex skips it → idempotent);
  new entrants get a placeholder `<!-- FOOTYSTRATEGY INSERT: <surname> tactical read -->`
  block — Scientist does NOT invent tactical prose. Both are gate WARNINGS, not fails.
- Name-join edge cases: two "Gary Ablett" rows disambiguated by a distinguishing club
  token; "Michael O'Loughlin" vs bio "Michael OLoughlin" folded by `_norm_name`
  (strips non-alphanumerics, also folds Jr/Sr).

**Residual prose drift NOT auto-fixed (FootyStrategy prose-pass items):** embedded
career numbers inside narrative paragraphs are preserved verbatim and can go stale —
e.g. Pendlebury's opening "10,955 career disposals ... across 432 games" (now
11,069/436) and Harvey's #8 "Joint all-time games-record holder at 432 ... now tied by
Pendlebury" (Pendlebury has since passed him). The regen only owns headings +
stat-lines; embedded prose numbers need a human/FootyStrategy pass. Related:
[[data_no_position]], [[all_time_formula]].
