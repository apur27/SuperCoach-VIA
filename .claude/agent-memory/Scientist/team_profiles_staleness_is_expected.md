---
name: team-profiles-staleness-is-expected
description: docs/afl-team-profiles.md mtime lagging the README "auto-updates" label is NOT a bug — the 5-year window is frozen seasons
metadata:
  type: project
---

`docs/afl-team-profiles.md` showing an old mtime (e.g. May 11 when "today" is June) while README labels it *(auto-updates)* is EXPECTED, not a breakage.

**Why:** The doc's content is the 5-year team-style window = `range(current_year-5, current_year)` (in `update_team_analysis.generate_5year_profiles`, ~line 1560). For 2026 that window is **2021-2025 — all completed seasons, so the source data is frozen**. The refresh pipeline (`refresh_and_rank.sh` step 5 → `refresh_readme.py::_step_team_analysis` → `replace_5year_section` via `_replace_in_file`) regenerates the body every run, but `_replace_in_file` only writes when content changes (refresh_readme.py lines 162-165). Identical body → no write → mtime stays put. The doc IS staged in refresh_and_rank.sh (line 73); it just never diffs.

Confirmed: re-running `generate_5year_profiles(2026)` produces a body byte-identical to the committed doc (`new != cur` → False). Also: the scraper rewrites 1000+ player CSVs in place each run bumping their mtime, but `git status` shows 0 diffs for completed-season rows — only 2026 rows append, which don't touch the 2021-2025 window.

**How to apply:** Before treating this doc (or any frozen-window aggregate) as "stale/broken," check whether re-running the generator actually changes the bytes. The doc will only legitimately change when the 2026 season completes and the window rolls to 2022-2026, OR if a 2021-2025 player CSV's content (not just mtime) changes. Don't "fix" a pipeline that's correctly no-op-ing.

Caveat on hand-verifying team per-game stats: a naive `groupby(['year','round']).sum().mean()` under-counts vs the doc by ~2x because `build_team_game_table` aggregates per team-match differently. To verify doc numbers, use the script's own aggregation path, not an ad-hoc groupby.
