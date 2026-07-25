---
name: tests-can-write-real-artifacts
description: Unit tests that call refresh_readme._step_* or update_team_analysis doc-writers will write REAL assets/charts and docs/*.md unless every uta module-level path is monkeypatched
metadata:
  type: feedback
---

Any unit test that calls `refresh_readme._step_*()` or
`update_team_analysis.update_top100_hof_doc()` runs against the **real** repo
paths unless you monkeypatch *all* of the module-level constants it touches.
Redirecting only the CSVs is not enough.

For the top-100 path you must patch every one of:
`uta.TOP100_CSV`, `uta.TOP100_SCORES_CSV`, `uta.HALL_OF_FAME_PATH`,
`uta.generate_top100_chart` (writes `assets/charts/top10_alltime_hall.png`),
`uta.generate_top100_section` (globs `data/`).

**Why:** writing a reachability probe for the HOF gate, I patched the gate
functions but not the paths. The test silently regenerated
`assets/charts/top10_alltime_hall.png` (showed up as ` M` in git status) and came
within one unequal-text comparison of overwriting
`docs/hall-of-fame-top100.md` while another agent had uncommitted edits in it.
It only escaped because the rendered table happened to be byte-identical — that
is luck, not a safeguard. CLAUDE.md's "never touch real data files" rule covers
this; the trap is that these entrypoints look like pure functions.

**How to apply:** before adding any test that calls a `_step_*` or doc-writer
entrypoint, patch paths first, then assert. After running a new test of this
kind, run `git status --porcelain assets/ docs/ data/` and compare against the
pre-run state — mtime alone will not tell you (a `git checkout` restore updates
mtime too). Existing safe examples to copy:
`tests/unit/test_refresh_readme_top100_gate.py` and
`tests/unit/test_update_team_analysis_top100_gate.py`.

Related: [[hof_full_table_regen]], [[top100_profile_regen]]
