---
name: finals-r25-cycle-retro
description: 2026-08-29 finals-mode cycle — FINALS_MODE flag built and shipped, round 25 settled + backtested; smoke runs caught 3 real defects before ship; BL-17 root-caused to one rcParam
metadata:
  type: project
---

Shipped 2026-08-29: `6319946f4` (harness), `c276ef393` + `d67b1ffce` (cycle),
`6da4baaa4` (manifest + backlog). Round 25 settled and backtested; **no prediction
generated** — first cycle ever run without one.

**Verified on the remote, not from the harness summary:** `matches_2026.csv` at
exactly 207 H&A rows, max round 25, every team on 23 games, 0 finals rows —
matching the season shape predicted from 2024/2025 before the scrape ran.

## The smoke run paid for itself, twice over

CLAUDE.md 6.2's smoke requirement is easy to treat as ceremony. It caught three
real defects that would all have shipped:

1. **A material Skeptic regression** — the recap ranked Gulden (11 games) flat
   against Sheezel (22) and Oliver (23) with no disclosure. The Round 24 recap had
   disclosed it; Round 25 dropped it. **The fix survived exactly one cycle.**
2. **An unsupported claim I authored myself, in the prompt.** I told FootyStrategy
   round 25 "is the FINAL home-and-away round"; it faithfully repeated it as bare
   prose. True, derivable — but uncited. *Do not hand an agent a structural claim
   it cannot source: it will repeat it, and the gate will rightly object.*
3. **Phase 3d chart abort** (BL-17, now fixed).

**Lesson: run the smoke even when the diff looks obviously safe.** Every finding
was in content and behaviour, not in the shell logic the unit tests covered.

## What the design got right

The load-bearing decision was deriving the round and the backtest bound from
`matches_<year>.csv` rather than from the prediction artifact. A naive "skip the
prediction" would have printed `Scoring completed rounds 25..24` — an empty `seq`,
no error, **no backtest**. Both smoke runs and the live cycle printed `25..25`.

## BL-17: root cause, and a docstring that lied

`generate_top100_chart()` rendered differently in-pipeline because
`generate_readme_charts._apply_dark_style` leaves `font.size: 11` in process-global
`rcParams`; the x tick labels resolve `xtick.labelsize: "medium"` against it, and
`bbox_inches="tight"` then reflows the whole figure — so a one-point delta looks
exactly like a font-build change. Bisected to that single key out of 14. Fixed with
`plt.rc_context(rcParamsDefault)`; committed bytes unchanged.

The gate's own docstring blamed a font upgrade and said to regenerate and commit
the charts — which under this failure mode **bakes the bad render in**. Corrected it
to a one-step discriminator: render in a fresh interpreter; if it matches the
committed file you have state leakage and must NOT recommit. See
[[prove-with-the-right-file]] — isolating the variable (clean vs in-pipeline render)
is what settled it, not reasoning from the inputs.

## Two findings only the post-ship verification surfaced

- **BL-24**: `next_round_26_prediction_20260430_1200.csv` has sat in the live
  namespace since April — a backtest leak (unrounded floats; the forward path writes
  ints). `find_latest_prediction` picks it over the real round-25 CSV *today*. The
  harness escaped only because it passes `--csv` explicitly. Routed to Scientist —
  removal touches `data/`, which I must never edit.
- **BL-25**: `mark` runs after the Phase 4 push (correct — only a pushed cycle counts)
  but nothing commits the result, so **every cycle strands its own entry**. Normally
  self-heals next cycle; with cycles paused to the GF it would have sat for weeks.

**Both were found by checking remote blobs after the "successful" ship.** A green
harness summary is not verification.

See [[finals-mode-harness-gap]], [[finals-cadence-pause]] (no cycles until the GF),
[[backtest-completion-manifest]].
