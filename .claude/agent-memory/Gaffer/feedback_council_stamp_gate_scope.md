---
name: council-stamp-gate-scope
description: The pre-commit council-stamp gate blocks ANY docs/hall-of-fame-stat-*.md without a PASS stamp, including legacy index pages
metadata:
  type: feedback
---

The pre-commit gate (`scripts/check-council-stamp.sh`) requires a `<!-- council-pipeline -->` stamp with DataSentinel:PASS and Skeptic:PASS for: `docs/news/*.md` (except README/sentinel-log/council-meeting), `docs/hall-of-fame-stat-*.md` (by glob — unconditional), and any other `docs/hall-of-fame-*.md` that already carries a stamp (sticky opt-in).

**Why:** `docs/hall-of-fame-stat-leaders.md` matches the `hall-of-fame-stat-*` glob, so even adding a one-line navigation pointer to that legacy hub (which has never carried a stamp) trips the gate. I will not fabricate a DataSentinel/Skeptic PASS to satisfy a gate — that is the ONE RULE. The legacy page needs a real council pass before it can take any edit, OR the edit must be dropped.

**How to apply:** When a cycle needs to touch `docs/hall-of-fame-stat-*.md`, plan for it to require a full council chain (or skip the edit). Pure navigation pointers TO an already-PASSED page belong in ungated index files instead — `README.md` and `docs/news/README.md` are skipped by the gate and take such links cleanly. New council pages (e.g. forgotten-heroes) opt in by carrying a valid stamp from the start. See [[parallel-council-commits]].
