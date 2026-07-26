---
name: project-baselines
description: Per-cycle baseline metrics (test count, files-per-run) so drift is visible across runs
metadata:
  type: project
---

Baseline metrics tracked across runs so drift is visible. Update the trailing value each run; do not delete history.

- **Test count:** 511 passing in 15.13s (2026-07-27, gate-integrity-hardening) — now TWO tiers: 498 fast/pre-commit (`-m "not integration"`, 14.98s) + 13 `tests/integration` (real `data/` tree, 0.66s). Budget in CLAUDE.md raised 10s → ~20s to match reality; policy is to raise deliberately, never trim tests. Prior: 384 (post-r21 ship); 352 (2026-07-13 r20); 244 (r19); 239 (sprint 1); ~176 before.
- **Test count (historical):** 352 passing (2026-07-13, weekly-r20). Prior: 244 (2026-07-07 r19); 239 (2026-07-03 sprint 1); ~176 before. +108 since r19 largely from Surveyor-audit hardening commits (0ac79a645, 967cc9359) + delta-scraper fix (+3).
- **Files per run:** weekly-refresh (r20) = 17 files in the ship commit (presentation layer) + a larger Phase-1 commit (player CSVs + 15 chart png + matches + README). r19 = 42 files. Compare like-for-like by cycle type; note r20 split ship into Phase-1 (data) + Phase-4 (presentation) commits.
- **Prediction accuracy (disposals backtest MAE):** R16 3.981 → R17 3.825 → R18 3.609 → R19 3.973 (aggregate R1–R19 3.960, bias -0.129, within-5 74.2%, within-10 95.8%). ~3.6–4.0 is the healthy band; single-round upticks within it are variance. Worst-by-team rotates among top clubs — r20 cycle: Hawthorn MAE 5.4, St Kilda bias -2.83 (most under-predicted, matches -0.69 aggregate). Best: Richmond 2.55, Port 2.71. Spread ~2x best-to-worst.
- **Prediction CSV schema:** still bare `player, team, predicted_disposals` (no intervals) as of R19 — grounds the recurring floor/ceiling recommendation (Surveyor S-5).

**Why:** a sudden test-count drop signals deleted/skipped tests; an unusually large weekly-refresh diff signals a runaway regeneration.
**How to apply:** when writing Pipeline Health, compare the current test count to the last value here and call out any drop explicitly. See [[project-council-doc-staleness]].
