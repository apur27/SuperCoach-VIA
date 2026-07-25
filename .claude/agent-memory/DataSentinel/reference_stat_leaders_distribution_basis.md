---
name: stat-leaders-distribution-basis
description: docs/afl-stat-leaders-2026.md mixes two different aggregation bases — league mean/std/percentiles are computed over PER-PLAYER SEASON AVERAGES (585 values), while the "Top per-game correlates" r-values are computed on the PER-GAME FRAME (thousands of rows). Do not recompute both from the same flattened dataframe or you'll get false mismatches.
metadata:
  type: reference
---

When verifying tags sourced from `docs/afl-stat-leaders-2026.md` (or any future stat-leaders auto-doc with the same structure), note the doc mixes two aggregation bases in the same block:

- **"League distribution (eligible players, season-to-date): mean X, std Y, p10/p50/p90, max Z"** — this is computed over the **585 per-player season averages**, not over the raw per-game rows. Reproducing it by flattening all eligible player-games into one series and taking `.mean()` gives a noticeably different number (e.g. 16.15 raw-per-game vs 15.1 per-player-average, for 2026 rounds 1-19 disposals) — the `max` value is the tell: it will exactly equal the single best player's own season average (e.g. Daicos 34.75), which can't happen on a per-game flattened series.
- **"Top per-game correlates: `stat` (r = ...)"** — this IS computed on the flattened per-game frame (confirmed by the doc's own intro sentence: "Correlations are Pearson r on the per-game frame"), and reproduces closely (within ~0.01) when you concat all eligible players' filtered per-game rows and call `.corr()`.

**How to apply:** when spot-verifying a league mean/std/percentile claim from this doc, group by player and average per player FIRST, then take distribution stats across player-level means — don't flatten to player-games. When verifying a correlation r-value, flatten to player-games directly. Mixing the two produces a false-mismatch investigation. See [[feedback-canonical-games-metric]] for a related "which aggregation is canonical" trap in this repo.
