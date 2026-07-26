---
name: recommendation-ledger
description: Which Chronicler expansion recommendations were adopted, deferred by explicit decision, or remain open — so closed items are not re-surfaced
metadata:
  type: project
---

Ledger of expansion recommendations across run reports, so closed or deliberately-deferred items are not re-surfaced as if new.

**Adopted / shipped:**
- Skeptic gate on `docs/afl-insights.md` — shipped `a46ca60e8` (Phase 3c in `weekly_refresh.sh`, fail-closed via `scripts/skeptic_verdict.py`).
- Backtest completion manifest — shipped `a46ca60e8`, bootstrap bug fixed `71fa430b3`.
- HOF regeneration + `check_top100_consistency()` made reachable — shipped `a46ca60e8`.
- README drift-gate + regenerator for the stat block — shipped `ce6ff6163`.

**Deferred by explicit user decision — do NOT re-raise as a gap:**
- `data/top100/yearly/year_2026.csv` regeneration moved to **end-of-season cadence**, not weekly.
- **R18-2026 backtest coverage gap** (284 player-rounds / 14 clubs vs 412 / 18) accepted as a documented limitation in `docs/afl-backtest-2026.md`. Rationale: substituting the fuller vintage moves MAE 3.958→3.960 and bias −0.110→−0.105 — nothing the page claims. Team table does move (St Kilda bias −0.583→−0.733), which is why it is documented rather than silent.
- Test-suite budget raised in CLAUDE.md 10s → ~20s to match reality at 511 tests, rather than trimming coverage. Policy: raise deliberately, never delete or skip tests to hit a number.
- Skeptic SK-R21-04/05 on `afl-insights.md` — logged non-blocking stylistic concerns.

**Open, carried forward (as of 2026-07-27):**
- F13 / SV26-N3 — backtest still writes into shared `next_round_*` namespace; top-30 selector still does not consult `completed_runs.json`. Note: these IDs live in **agent memory only**, not `docs/architecture.md` §4; and "F13" collides with an unrelated item at `docs/pending-tasks.md:85`.
- ~700 garbage rows in committed lineup CSVs (re-measured 700 exactly, 18 of 24 files). Recipe written at `docs/experiment-log.md:322–357`, never executed. Now higher priority: the 18 files were allowlisted `a46ca60e8`, so the bad rows are committed weekly.
- Chart PNGs (HOF, backtest accuracy) no longer reproduce byte-identically — uninvestigated.
- Gate verdict JSON does not persist reasoning; only commit messages do.
- `docs/afl-backtest-2026.md` publishes three mutually inconsistent headline figure sets (3.96 R1–R20 table / 4.020 & −0.097 claiming R1–R13, n=4,806 / 3.958 & −0.110, n=7,153).

**How to apply:** before writing an expansion recommendation, check this ledger. Re-raising a deferred item as a defect misreads an explicit decision as an oversight. See [[unrunnable-gate-class]].
