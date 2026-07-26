---
name: unrunnable-gate-class
description: Dominant defect class in this repo — a gate that cannot run is indistinguishable from a gate that passed; check reachability before trusting any PASS
metadata:
  type: project
---

The repo's recurring defect class, named explicitly during the 2026-07-27 gate-integrity cycle: **a gate that cannot run is indistinguishable from a gate that passed.**

Confirmed instances (all found in one session, all pre-existing and silent):
- `check_top100_consistency()` / HOF profile regeneration — never invoked by either harness script, despite CLAUDE.md citing that gate as the justification for exempting `docs/hall-of-fame-top100.md` from inline `[data]` tags. 7 profile stat-lines were stale on main.
- `.githooks/pre-commit` returned exit 0 whenever no Markdown was staged — every pure-Python commit bypassed it entirely.
- Backtest completion-manifest bootstrap adopted every timestamp on disk, grandfathering in the orphan vintages it existed to catch.
- `os.path.exists` guards that spell "skip" as "pass"; regexes that match nothing and report success.
- Related but distinct: a selector that resolved correctly **by luck** (top-30 vintage sort discarded the date and compared time-of-day only; authoritative runs happened to also hold the later wall-clock time).

Earlier instance of the same family: Sprint 1 (2026-07-03) found 6/8 legacy docs FAILed a genuine re-check behind a text-only "DataSentinel: PASS" stamp. See [[council-doc-staleness]].

**Why:** every one of these reported success. Test-suite green and gate-PASS output are both compatible with the gate never executing, so neither is evidence on its own.

**How to apply:** when writing Pipeline Health, do not record a gate as PASS on the strength of its own output. Check it was actually reached — grep the harness scripts for the invocation, or look for the artifact it should have rewritten. A gate whose output diff is suspiciously empty week after week is the tell. When a gate runs genuinely for the first time, expect a large corrective diff (7 HOF profiles here) — and thereafter, a *large* diff is the leading indicator that it silently stopped running again.
