---
name: finals-cadence-pause
description: User decision 2026-08-29 — after the round-25 finals-mode cycle, run NO weekly cycles for the rest of the 2026 finals; wait for the Grand Final to settle, then one closing run
metadata:
  type: feedback
---

**After the 2026-08-29 finals-mode cycle (round 25 settled), do NOT run weekly
cycles through the rest of the finals. Wait until the Grand Final has settled,
then do ONE closing run to wrap up the 2026 season.**

**Why:** decided by the user on 2026-08-29. Round 25 is the last home-and-away
round, and it is the last round that has an archived forward prediction to score
— so once it is backtested, **the 2026 backtest lane is finished**. With no
predictions being generated during finals (see [[finals-mode-harness-gap]]),
every remaining finals cycle would carry no prediction AND no backtest: pure data
refresh plus HOF/stat-leader churn. Four cycles of that is drift and gate risk for
no reader-facing gain.

**How to apply:** if a future invocation is asked for "the weekly refresh" between
2026-08-29 and the 2026 Grand Final, do not default back to weekly cadence —
surface this decision and confirm before running anything. The trigger for the
next run is *the Grand Final result being settled*, not a calendar week elapsing.
Do not treat the absence of a recent cycle as a missed run needing catch-up.

The closing run should still go through `FINALS_MODE=1` (no prediction exists for
finals), and it is the natural moment to pick up the season-end items deliberately
deferred here: `data/top100/yearly/year_2026.csv` (a season-END artifact — weekly
churn is expected drift, see [[top100-yearly-cadence]]), BL-20 / BL-21 in
`docs/pending-tasks.md`, and the rest of BL-06..BL-19.

See [[finals-mode-harness-gap]] for why the harness cannot run un-flagged during
finals at all, and [[open-backlog]] for where deferred items are tracked.
