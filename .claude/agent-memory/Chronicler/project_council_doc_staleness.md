---
name: project-council-doc-staleness
description: Council docs decay after authoring — staleness is the dominant failure mode, not fraud; re-verification is the recurring recommendation
metadata:
  type: project
---

When genuine DataSentinel was run against 8 legacy stamped-but-recordless council docs (Sprint 1, 2026-07-03), **6 of 8 FAILed**. Of the 6: 3 were **staleness drift** (active-player / current-season counts advanced since authoring), 3 were frozen authoring errors (jonathan-brown 46≠49 Brownlow; dustin-martin "12 players"→16; forgotten-heroes retired-player miscounts).

**Why:** verification that runs only once (at authoring) is guaranteed to decay for any doc referencing active-player or current-season numbers. The trust badge / enforce=1 machinery surfaced this; it does not fix it.
**How to apply:**
- Do NOT re-surface "build a scheduled staleness re-verification gate" as a novel recommendation if it has shipped — check whether `scripts/reverify_stale_docs.py` (or equivalent) exists first.
- Distinguish deterministic docs (13 HOF stat pages, regenerated from `_stat_leaders.json` each refresh — self-healing) from LLM-verified prose (news docs — can rot). Only the latter needs re-verification.
- Expect the 6 legacy FAILs to burn down via Scientist; jonathan-brown + dustin-martin were prioritized first (2026-07-03). Track the open count run-over-run.

Related: [[project-baselines]].
