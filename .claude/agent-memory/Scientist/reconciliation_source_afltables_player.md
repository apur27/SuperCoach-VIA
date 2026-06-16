---
name: reconciliation-source-afltables-player
description: afltables player pages carry an all-season Totals row that maps 1:1 to our CSV columns — the second source of truth for catching missing player stat rows
metadata:
  type: project
---

afltables player page is the chosen reconciliation source for catching missing PLAYER stat rows (the Pendlebury 3-missing-games class of bug, which `audit_match_rounds` in game_scraper.py does NOT catch — that only checks missing MATCH rows).

**Why:** It is the same site we already scrape (no new dependency/auth), URLs are stable, and its column abbreviations map 1:1 to our player CSV columns.

**How to apply:**
- URL: `https://afltables.com/afl/stats/players/<Initial>/<First>_<Last>.html` (e.g. `S/Scott_Pendlebury.html`). Has a "By totals" table ending in a **Totals** row aggregating all seasons. `pandas.read_html()` parses it directly.
- Abbrev→CSV map: GM=games, KI=kicks, MK=marks, HB=handballs, DI=disposals, GL=goals, BH=behinds, HO=hit_outs, TK=tackles, RB=rebound_50s, IF=inside_50s, CL=clearances, CG=clangers, FF/FA=free kicks, BR=brownlow_votes, CP/UP=contested/uncontested poss, CM=contested_marks, MI=marks_inside_50, 1%=one_percenters, BO=bounces, GA=goal_assist, %P=pct game played.
- Reconcile at CAREER-TOTAL level (one fetch per player per refresh), NOT per-game (~400x heavier). A per-game gap changes the total, so the total catches it. Use per-season diff only as a drill-down after a total mismatch fires.
- Threshold = 0 (exact integer counting stats), but reconcile each stat only within its recorded era — see [[data_stat_coverage_eras]] (tackles pre-1987, clearances/contested pre-1998 absent on afltables).
- GM comparison must account for our within-season `games_played` column — see [[hof_games_counter_gotcha]].

**Rejected sources:** AFL.com.au (JS/Champion-Data, no public API, fragile). Squiggle API (`teams/games/sources/tips/standings/ladder/pav` only — no raw player counting stats; `pav` is a derived composite). Footywire (opaque numeric player IDs, same Champion Data feed — fallback tie-breaker only).
