---
name: squad-pipeline-name-collision
description: Cross-club same-name jersey->stats collision in the list-quality squad pipeline; root cause + team+era-scoped fix + detection recipe
metadata:
  type: project
---

The squad-rebuild pipeline for `docs/news/2026-06-17-afl-2026-list-quality-draft-pipeline.md`
(ad-hoc derivation, NOT a committed script; intermediate JSON `/tmp/squads_all_rounds_2026.json`)
produced wrong-player rows. Root cause: the `(team,jersey)->name` map IS team-scoped, but the
SECOND lookup `name -> draft pick / games / grade` used an unscoped key
`re.sub(r'[^a-z]','',name.lower())` against ALL perf files + `afl_draft_history`, so a same-name
player from another club/era won the match.

**Fix (resolution logic):** scope the name->stats lookup to the player whose 2026 perf row is for
THAT club; disambiguate same-name players by club AND debut era (perf `year.min()` ≈ draft year).

**NOW A COMMITTED MODULE (2026-06-19, commit e630fc865):** the throwaway `/tmp/build_squads_2026.py`
resolution layer is promoted to `scrapers/squad_builder.py` with TDD regression tests in
`tests/unit/test_squad_builder.py` (30 cases, all green; full suite 122). The fix is
`pick_best(recs, cteam=None, club_col='club')` replacing the old always-earliest `pick_earliest`:
when `cteam` is given it prefers records where `canon_team(r[club_col])==cteam`, else falls back to
earliest-by-year. `lookup_draft(...)` threads `cteam` through exact/alias/loose layers. CLI:
`python -m scrapers.squad_builder --draft-csv <csv> --name "Bailey Williams" --team "West Coast"`.

**Collisions found & fixed (2026-06-19), all in this article:**
West Coast Bailey Williams (was Bulldogs BW 2015/48/184 -> 2018/35/B/98); Collingwood Will Hayes
(was Bulldogs WH 2018/78/13 -> 2024/56/D/8); Hawthorn Sam Butler (was WC SB 2003/20/32 -> 2021/23/C/29);
Richmond Tom Lynch (was St Kilda TL 2008/13/164 -> Gold Coast TL 2010/11/A/242); GWS Callum Brown
(was Collingwood CB 2016/35/70 ND pick -> Irish/non-ND, moved to pathway, 71 games). Plus 2 prose
errors from the same two-Tom-Lynch confusion: St Kilda "Tom Lynch moved to Richmond"->Adelaide;
Gold Coast "Lynch to West Coast"->Richmond.

**Detection recipe (drift-immune):** the PICK (draft_year,pick) column does not drift, unlike games.
For each article row whose name-key has >1 perf file (key as `first+sur`, lowercased a-z only),
compare the article (year,pick) to the correct 2026-club player's TRUE draft from `afl_draft_history`
(match name + club-in-teams_all + |year-debut| minimised). PICK mismatch = collision. NOTE: resolver
falls back to an impostor's row when the correct player's ND row is MISSING from afl_draft_history
(e.g. Gold Coast Tom Lynch 2010/11 is only in `draftguru_enrichment`, not afl_draft_history) — verify
those by hand. Same-player re-draft entries (Stringer 2012/5+2025/59, Wagner, Byrnes) are NOT collisions.

**Out-of-scope drift seen:** ~167 squad rows have games off by a small uniform per-club constant
(1-4) vs current perf `len(df)` — this is article-vs-current snapshot freshness, NOT a collision; do
NOT mass-rewrite (incremental only). The per-cell DataSentinel gate passes these because each number
is individually real.
