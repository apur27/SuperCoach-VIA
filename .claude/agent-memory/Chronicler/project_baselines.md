---
name: project-baselines
description: Per-cycle baseline metrics (test count, files-per-run) so drift is visible across runs
metadata:
  type: project
---

Baseline metrics tracked across runs so drift is visible. Update the trailing value each run; do not delete history.

- **Test count:** 244 passing (2026-07-07, weekly-r19). Prior: 239 (2026-07-03 sprint 1); ~176 before. +5 this run = test_prediction_selection.py (3) plus others.
- **Files per run:** weekly-refresh (r19) = 42 files / +684 / -441 (many are chart .png binaries + regenerated HOF/season docs). Harness sprint = 33 files. Compare like-for-like by cycle type, not absolute.
- **Prediction accuracy (disposals backtest MAE):** R16 3.981 → R17 3.825 (bias -0.62 → +0.13, within-5 75.8%→77.8%). Improving trend; ~3.8 is the healthy band. Worst by-team consistently a top-6 club — R17 was Sydney (5.09), Gold Coast over-predicts (bias +1.5).
- **Prediction CSV schema:** still bare `player, team, predicted_disposals` (no intervals) as of R19 — grounds the recurring floor/ceiling recommendation (Surveyor S-5).

**Why:** a sudden test-count drop signals deleted/skipped tests; an unusually large weekly-refresh diff signals a runaway regeneration.
**How to apply:** when writing Pipeline Health, compare the current test count to the last value here and call out any drop explicitly. See [[project-council-doc-staleness]].
