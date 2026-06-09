---
name: match-round-audit
description: audit_match_rounds() in scrapers/game_scraper.py guards against silently-truncated match rounds; now fixture-aware (exact), no modal/threshold heuristic
metadata:
  type: project
---

`scrapers/game_scraper.py` has `audit_match_rounds(file_path)` (module-level) that guards against the "R10 2026" bug: the match-summary scraper silently wrote only 1 of 9 Round 10 rows and it went undetected 5+ weeks.

**Why:** silent partial-round writes corrupt downstream team aggregates with no error.

**How it works now (fixture-aware, exact — replaced the modal/threshold heuristic Jun 2026):**
- For each integer H&A round in the file, it calls `fetch_round_fixture(year, round_num)` and compares scraped matchups (by team pair, order-agnostic via `frozenset`) to the scheduled fixture. WARNING names the exact missing matchups; no probabilistic threshold, no `MAX_BYE_MATCH_DROP`.
- **Ground-truth source:** the SAME site the scraper already hits — `https://afltables.com/afl/seas/<year>.html`. That season page lists the FULL fixture: completed rounds carry scores, upcoming rounds (e.g. R15+ in mid-2026) list scheduled matchups with no scores. No new dependency, no second source.
- **HTML parse model (verified against 2026 page):** each round is a `<b>Round N ...</b>` heading; the asterisk variant "Round 1* see notes" is handled by a bounded regex `^Round N(?!\d)` (plain `Round 1` would otherwise match `Round 15`). The matches are in the NEXT sibling `<table>`. Within it, every team-name cell (first cell, in `KNOWN_TEAM_NAMES`, row has >=3 cells) is collected in document order and paired sequentially. Bye rows are `[team, "Bye"]` (2 cells) and are skipped; the trailing "Rd N Ladder" block is skipped (not a team row).
- Byes need no special handling: the fixture itself omits resting teams, so `expected` already excludes them.
- `_get_season_soup(year)` caches the parsed season page per process, so auditing all rounds in a file fetches the page ONCE, not once per round.
- Returns dicts with keys `year, round_num, n_matches, expected, teams_present, severity, missing` (severity WARNING iff `missing` non-empty). refresh_data.py reads `severity` + `round_num` — both preserved.

**Where it runs:** automatically post-write in `_process_year`, at the end of `refresh_matches()` in refresh_data.py, and standalone via `python scrapers/game_scraper.py --audit [dir|files]`.

**Verified Jun 2026:** exact reconciliation across all 14 played 2026 rounds (incl. variable counts: R1=5 opening, R3/R4=7, R5=8 byes, R13=7, R14=8). Simulated R10-bug (1 of 9 rows) → WARNING naming the 8 missing matchups. Historical seasons (1925, 1990, 2015) audit clean with no crash on former team names (Footscray/Fitzroy/Brisbane Bears etc. are in `KNOWN_TEAM_NAMES`). Match CSV columns: `round_num` (int for H&A, string for finals), `team_1_team_name`, `team_2_team_name`, `year`; finals dropped by filtering numeric round_num.

**If fixture fetch fails for a round** (page layout change / round absent): that round is skipped with an `[info] ... fixture unavailable` line — no false positive when ground truth is unavailable. If afltables ever changes the round-heading/table markup, the parser to revisit is `fetch_round_fixture`.
