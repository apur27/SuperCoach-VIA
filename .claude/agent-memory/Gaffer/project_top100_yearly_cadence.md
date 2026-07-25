---
name: top100-yearly-cadence
description: data/top100/yearly/year_<season>.csv is a SEASON-END artifact — its weekly uncommitted churn is expected drift, not an incident; never add it to a harness allowlist
metadata:
  type: project
---

`data/top100/yearly/year_<current-season>.csv` regenerates on every weekly run and
therefore always shows an uncommitted diff (~100 insertions / 100 deletions as ranks
churn). This is **expected drift, not an incident**, and it must stay OUT of both
harness `git add` allowlists (`refresh_and_rank.sh`, `scripts/weekly_refresh.sh`).

**Why:** user decision, 2026-07-25. The yearly top-100 only needs committing once,
after the season ends — mid-season ranks are meaningless churn and staging them
weekly would add noise to every cycle's commit. This was previously logged as open
backlog item "F3" and re-discovered twice as if it were a bug; it is not.

**How to apply:**
- Do NOT stash, reset, or force-commit this file before a recovery re-run, and do
  NOT propose adding it to an allowlist. Leave it dirty; the next cycle rewrites it.
- The weekly harness must never block on it.
- ONE end-of-season task: commit the completed season's file, then it goes stable
  like every prior year.
- Cadence is also documented in a code comment at both write sites in
  `top_players_comprehensive.py` (`generate_yearly_top_100`, `_generate_yearly_from_memory`).

Contrast with the lineup CSVs, which WERE real drift and were added to the Phase-1
allowlist on the same date — see [[weekly-r21-r22-retro]]. The distinguishing
question is whether the per-cycle rewrite is meaningful output or transient churn.
