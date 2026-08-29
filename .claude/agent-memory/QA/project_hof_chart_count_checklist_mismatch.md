---
name: project_hof_chart_count_checklist_mismatch
description: QA checklist says "at least 6" alltime_top20_*.png charts but the generator only ever produces 4 (plus a 5th differently-named file) — chronic, not a per-cycle regression
metadata:
  type: project
---

The QA agent's own checklist (item 2, "HOF charts" row) says the mandatory
artifact check for `assets/charts/hall/alltime_top20_*.png` requires "at
least 6 chart files exist". As of 2026-08-18 (R25 cycle) only 4 files match
that glob: `alltime_top20_disposals.png`, `alltime_top20_games.png`,
`alltime_top20_goals.png`, `alltime_top20_tackles.png`. A 5th file,
`alltime_stat_categories_leaders.png`, exists in the same directory but has
a different name and doesn't match the glob.

Verified this is not a regression: `docs/hall-of-fame/generate_records_charts.py`
(the sole generator) has hardcoded exactly these 4 `out_name=` values for the
`alltime_top20_*` family since the file was introduced in commit `e114f9c48`
("Add Hall of Fame: all-time statistical leaders"). The "6" figure in the
checklist was never grounded in what the generator produces.

**Why:** without this note, every future QA run will flag a false artifact
shortfall on a check that has never once passed at "6" and never will unless
the generator is deliberately extended to more stat categories. Treat this
like [[project_banner_aria_label_stale]] — a standing WARN with a known,
named cause, not a per-cycle FAIL signal.

**How to apply:** report "4 of 4 generator-defined alltime_top20_* charts
present (+1 alltime_stat_categories_leaders.png)" as PASS against actual
generator output, and note the checklist's "6" figure as a stale expectation
in need of correction — do not FAIL a cycle over it. The real regression
signal for chart health is `tests/integration/test_pipeline_artifacts.py::
test_every_chart_referenced_by_a_published_doc_exists`, which does pass and
does catch a genuinely missing/unreferenced chart.
