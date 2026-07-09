---
name: truth-in-badging
description: Trust badge = a FRESH-verification claim; only badge docs that earned a genuine DataSentinel PASS this cycle, never legacy/unverified ones
metadata:
  type: feedback
---

The `✓ All N stats verified against source data` trust badge is a truth claim, not decoration. Only apply it to a doc that has a **fresh** DataSentinel PASS record for its current content.

**Why:** In Sprint 1 (2026-07-03) the badge backfill + `AUDIT_ENFORCE=1` ran genuine DataSentinel on 8 legacy docs — 6 FAILed on stale/wrong numbers. Badging "verified" on a doc I didn't just verify would ship a false claim, and legacy docs proved pervasively stale (see [[legacy-doc-staleness]]). A badge that lies is worse than no badge.

**How to apply:**
- Never badge a doc solely because it has a `DataSentinel: PASS` stamp — stamp text is forgeable; only a content-hash-keyed audit record counts.
- Non-gated docs (e.g. `docs/list-management-101.md`, `docs/articles/*`) can commit a badge without a record, but still withhold the badge unless freshly verified — the "All N verified" claim must be true regardless of whether the gate forces it.
- Zero genuine `[data]` tags ⇒ no badge (an "All 0 stats" line is false). `inject_trust_badge.py` enforces this.
- Badge is hash-neutral (stripped by `council-content-hash.sh`), so adding it never invalidates a record — the gate is on the underlying stats, not the badge.
