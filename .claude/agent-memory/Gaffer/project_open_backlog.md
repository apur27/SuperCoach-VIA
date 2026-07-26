---
name: open-backlog
description: Pointer to the canonical backlog in docs/pending-tasks.md — memory is a cache, not the record; open items carry BL-nn IDs there
metadata:
  type: project
---

**The canonical backlog is `docs/pending-tasks.md`, not this file.** Read it before
acting on any backlog item; where it disagrees with memory, it wins.

Open items carried out of the 2026-07-27 gate-integrity cycle, by canonical ID:

- **BL-01** — backtest writes into the live `next_round_*` namespace. Repeat-incident
  risk: this is the collision that produced the tainted-provenance incident the cycle
  was opened to fix. The completion manifest is a downstream filter, not the fix.
- **BL-02** — top-30 deviation loader ignores the completion manifest.
- **BL-03** — ~700 legacy garbage rows in `data/lineups/` (recipe at
  `docs/experiment-log.md:330`).
- **BL-04** — committed chart PNGs no longer reproduce byte-identically; HOF
  regeneration dirties `assets/` as a side effect, and `generate_backtest_section()`
  writes a PNG despite its read-only-sounding name.
- **BL-05** — gate verdict records store the verdict but not the findings behind it.
  Mine to fix; see [[route-every-finding]] for the incident that motivated it.

**Why this file is now a pointer.** Tracking backlog only in agent memory made items
invisible to every other agent and let IDs drift: "F13" simultaneously meant a Sprint-2
Round-14-briefs item in `docs/pending-tasks.md` and the backtest-namespace item that
existed only here. Nobody could see the collision because the two records never met.

**How to apply:** when deferring something, write the entry in `docs/pending-tasks.md`
with a fresh `BL-nn` ID and add at most a one-line pointer here. Do not restate the fix
brief in memory — a second copy is a second thing to drift. The `BL-nn` namespace was
chosen so it cannot collide with the `F`/`S`/`A` IDs from the 2026-07-07 surveys.

Related: [[gates-hardening-20260725]], [[backtest-completion-manifest]].
