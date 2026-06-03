---
name: script-health-backlog
description: Known script-health issues found in the 2026-05-31 full script-run audit — charts.py Era KeyError + unsafe --help in pipeline scripts
metadata:
  type: project
---

Findings from the 2026-05-31 "run every script" verification mission (25 PASS / 1 FAIL).

**FAIL: `supercoach/charts.py` raises `KeyError: 'Era'` at import time** (module-level code, not guarded by a function/`__main__`). The loaded dataframe has no `Era` column. Sibling `supercoach/bar_chart.py` reads the same era data correctly using lowercase `era`, so charts.py is reading a CSV whose column was renamed/relocated. Pre-existing, not a restructure regression.
**Why:** column contract drift between charts.py and the era-analysis data source.
**How to apply:** this is an era-analysis data-contract change — route to Scientist sign-off before rewriting the column reference. Do NOT silently patch (ML/data logic is out of Gaffer's lane).

**Backlog items (presentation/UX, low priority):**
- `scripts/live_analysis_pipeline.py` has no `--help` guard; `--help` is parsed as a gameid and STARTS a real 90s poll-and-push loop. Add an argparse front-door so probing can't launch a live committing loop. See [[feedback_parallel_council_commits]] — this is another auto-push hazard.
- `refresh_readme.py`, `update_team_analysis.py`, `generate_readme_charts.py` do all work at module import before arg parsing, so `--help` hangs (60s timeout). Import-only checks pass; add a guarded `main()` if convenient.

**Chart re-render churn:** running generate_readme_charts / generate_records_charts / generate_strategy_charts rewrites ~11 PNGs under assets/charts/ as byte-different binaries (timestamp-only change, same data). compute_stat_leaders.py is deterministic and produces no diff. Revert chart churn with `git checkout -- assets/charts/`.
