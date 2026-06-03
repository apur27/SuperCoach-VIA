---
name: nan-counting-stats
description: Correct NaN handling for tackles/marks/hit-outs means in briefs
metadata:
  type: feedback
---

Blank/NaN in counting stats (tackles, marks, hit-outs) = data recording gap, NOT player absence.
Verified: every blank-tackle row in this repo has percentage_of_game_played > 0.

**Rule:** Use dropna (skipna=True) for per-game means. Display as "X.X (N of M games recorded)" when N < M.

**Why:** The data never stores explicit zeros. A zero-tackle game and a missing-tackle game are both NaN. Dropna gives the honest "per game played" rate. Fill-zero fabricates zeros for games the stat wasn't recorded.

**How to apply:** Always compute from raw row values. Print the list before computing. Never use a pre-aggregated figure.
