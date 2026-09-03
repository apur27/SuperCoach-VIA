---
name: brownlow-season-scale-constant
description: Brownlow proxy table's scaled column and its "(×N)" heading both read update_team_analysis.HOME_AND_AWAY_GAMES; closing BL-20 (22 -> 23) silently moves both
metadata:
  type: project
---

`update_team_analysis._build_brownlow_proxy_table` writes `season_proxy_scaled`
(= `brownlow_proxy_pg × HOME_AND_AWAY_GAMES`) and `_build_brownlow_table_md`
interpolates the SAME module constant into the column heading
`Season proxy (×N)`. Neither carries a private literal. Guarded by
`tests/unit/test_brownlow_proxy_column.py`.

The constant resolves to `config.HOME_AND_AWAY_GAMES`, which is **22 and wrong**
— the AFL H&A season is 23 games per team (verified: `data/matches/matches_*.csv`
holds 207 non-finals rows over 18 teams for 2024, 2025 and 2026, every team on
exactly 23). That off-by-one is logged as BL-20 in `docs/pending-tasks.md` and was
deferred to season end by user decision 2026-08-29.

**Why:** the column was published as "Proj. votes" showing ~+60 against a Brownlow
record of 36 — a heading naming a quantity that was not tabulated. The fix renamed
it to a proxy and removed the duplicate `HOME_AND_AWAY = 22` literal, but did NOT
change `config.HOME_AND_AWAY_GAMES` — that would have altered the finals-pathway
doc and chart mid-pause, against an explicit deferral, without BL-20's
fixture-derived regression test.

**How to apply:** when BL-20 lands, expect the Brownlow table heading to change
from `(×22)` to `(×23)` and every value in that column to rescale by 23/22 — this
is intended, not a regression, and rank order is unchanged (pure rescale). Also
expect `docs/afl-finals-2026.md` prose + the finals-pathway chart to move
(consumers at `update_team_analysis.py` lines ~1983, ~2101, ~2136 — the `# 22`
comment on the TOTAL_GAMES line goes stale at the same moment).
