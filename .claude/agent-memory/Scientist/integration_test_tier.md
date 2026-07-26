---
name: integration-test-tier
description: What the tests/integration tier is for, what QA checks were ported into it, and which QA checks are deliberately excluded as noisy
metadata:
  type: project
---

`tests/integration/` asserts against the REAL `data/` tree and published docs.
Marked `integration` (marker declared in `pyproject.toml`), excluded from the
fast/pre-commit tier, and run by `scripts/weekly_refresh.sh` Phase 0b, which
ABORTS the cycle on failure. Anything added here is a blocking weekly gate, so
it must be deterministic and must not trip on legitimate seasonal variation.

- `test_published_artifacts.py` — banner aria-label vs visible pills, banner
  player-file count vs the tree, HOF doc vs the ranking CSV.
- `test_pipeline_artifacts.py` — QA's data-touching checks, ported 2026-07-26:
  prediction-CSV schema + 0-80 value range + no duplicate player/team rows,
  backtest-summary schema and plausibility, `check_hof_numbers.py` exit 0,
  interior round gaps in the newest `data/matches/matches_*.csv`, and every
  chart referenced by a published doc existing on disk.

**Deliberately NOT ported — leave these to a human QA run:**
- `assets/charts/hall/` "at least 6 charts": the generator emits exactly 4 by
  design and always has. Chronic spec drift, not a regression.
- "top-5 career-games players must have current-season rows": a legitimate
  retirement fails it with no defect present.
- Round gaps at the TOP of the current season: the newest round is legitimately
  unscraped mid-week. Only INTERIOR gaps are deterministic.
- "new .py files need tests": QA.md itself says use judgement.

**Gotchas:** `round_num` in `data/matches/*.csv` is `object` once finals land
("Grand Final", …) and `int64` before — always `pd.to_numeric(errors="coerce")`.
Prediction files must be selected by MTIME, never lexicographically
(`next_round_9_*` sorts above `next_round_18_*`).
`docs/hall-of-fame/_stat_leaders.json` is gitignored, so tests that read it must
skip-if-missing rather than fail on a fresh checkout.

Related: [[golden_file_test_tier]], [[tests_can_write_real_artifacts]].
