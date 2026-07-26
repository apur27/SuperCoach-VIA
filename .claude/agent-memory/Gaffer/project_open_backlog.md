---
name: open-backlog
description: Open backlog carried out of the 2026-07-26 gates-hardening ship — legacy lineup garbage (Scientist) and the backtest live-namespace collision (F13)
metadata:
  type: project
---

Neither is a regression from the 2026-07-26 ship; both were surfaced by Surveyor's
post-ship audit and deliberately deferred.

**1. Legacy garbage rows in the lineup CSVs — owner: Scientist.**
~700 corrupt rows remain in the now-committed `data/lineups/` files (412 from 2025,
288 from early 2026): jersey numbers plus `Rushed`/`Totals`/`Opposition` tokens
instead of player names. Adding lineups to the Phase-1 allowlist stopped NEW drift
(current output is clean name-form) but did nothing about the existing garbage.
Cleanup recipe already written up at `docs/experiment-log.md:330`.
Not urgent: the rows are historical and nothing published cites them.

**2. F13 — the backtest still writes into the live `next_round_*` namespace.**
This cycle's completion manifest hides orphans from mtime-based consumers, but it
does not stop a future FATALed run from colliding in that namespace again. The
tainted-provenance incident that drove this whole session came from exactly that
collision. A proper fix is a separate namespace per run (or equivalent isolation),
not another downstream filter. Owner: whoever takes the backtest harness code —
route to Scientist for the Python, me for any harness wiring.

**3. Committed chart PNGs no longer reproduce byte-identically — owner: Scientist.**
`assets/charts/top10_alltime_hall.png` regenerates deterministically within this
environment (identical md5 across runs) but differs from the committed blob
(175,862 -> 174,837 bytes). The chart plots scores, not the goals formatting changed
on 2026-07-26, so this predates that work — most likely a matplotlib/font version
drift from whatever produced the committed version. Consequence: any agent that runs
the HOF regeneration dirties `assets/` as a side effect and has to revert it.
Related trap worth knowing: `generate_backtest_section()` also writes
`assets/charts/backtest_accuracy_2026.png` despite its read-only-sounding name, so
"just previewing" a section is not side-effect free. Check both before assuming a
dry run was clean.

**Why they were deferred:** the ship they came out of was already large and both are
pre-existing conditions with no live incorrect published claim attached. Do not let
them silently age out — F13 in particular is a repeat-incident risk, not cosmetic.

Related: [[gates-hardening-20260725]], [[backtest-completion-manifest]].
