---
name: match-completeness-gate
description: Match-audit gate instrument choice + why incremental scrape can't self-heal past-round gaps
metadata:
  type: project
---

Blocking match-completeness gate (`scripts/match_completeness_gate.py`, wired into
`weekly_refresh.sh` before the Phase 1 push, mirrors the phantom-row gate).

**Instrument choice — gate on `audit_match_rounds`, NOT `check_match_completeness`.**
Why: `check_match_completeness` (game_scraper.py) compares each team's game-count to
the season MODE with a `>=2` threshold. A single dropped round leaves every affected
team short by only **1**, below that threshold — so it returns **0 warnings on the
real broken file** and structurally cannot catch the "R10 2026" class of bug.
`audit_match_rounds` is exact + fixture-aware (`fetch_round_fixture` off the afltables
season page), counts distinct team-pairs per H&A round, and names the exact missing
matchups. Gate **fails OPEN** on network/fixture outage (audit skips unverifiable rounds).

**Why the incremental scrape can't self-heal a past-round gap.** `MatchScraper.scrape_all_matches`
passes `last_processed_date` (max date on file) to `_process_year`, which skips any game
dated before it. Games dropped from a PAST round (R10 in May while the cursor is at
R19/July) are never re-fetched by the normal harness. To backfill you must call
`_process_year(year, folder, last_processed_date=None)` to force a full re-scan. This is
why the gate is "force-a-human" blocking rather than self-healing: a blocked harness stays
blocked until someone runs a full re-scrape + commits the data.

**Dedup blind spot exposed by backfill.** The matches dedup key is `(date, round_num, venue)`.
A full re-scrape re-added the missing games but left 2 stale pre-existing R10 rows that the
key couldn't catch: same game under a different kickoff time / reversed home-away order /
NaN venue. Verify pair-uniqueness after any backfill: within each numeric round, every
`frozenset({team_1, team_2})` must be unique. Keep the fresh scrape (has venue + attendance),
drop the stale row, else the next scrape re-adds the fresh one and re-duplicates.

Fixes: commits `75ea7bd0f` (F4 date mint), `896b16e35` (F8/E2 gate + backfill). See also
[[player_csv_date_format]] and [[finals_gap_rescrape_recipe]].
