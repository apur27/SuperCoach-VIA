---
name: unfilled-footystrategy-layer-tell
description: How to detect a BriefBuilder doc that asserts a tier/tripwires before the FootyStrategy interpretation layer is actually authored
metadata:
  type: feedback
---

A coaches-strategy-corner brief can reach the Skeptic with a `[Tier: ...]` and tripwires in the Executive Summary but NO real FootyStrategy deliberation layer. The tells:

1. **Stale footer**: the BriefBuilder footer still reads "Interpretation pending FootyStrategy fill of `<!-- FOOTYSTRATEGY INSERT -->` markers." If that line is present, the doc is mid-pipeline regardless of how polished the prose looks.
2. **No structural sections**: grep `^#` for `## Lens reads`, `## Convergence`, `## Tensions`, `## Recommendation`. If absent, there is no deliberation structure for Audit 3 to check.
3. **Lens labels as rhetoric**: phrases like "the structural lens reads," "the conditioning lens flags," "the talent-development lens sees" woven INTO BriefBuilder body prose are NOT activated lens reads. They are narrative labels. A genuine lens read lives in a Lens-reads section and feeds Convergence/Tensions.

**Why:** Audits 2 and 3 exist to check the interpretation layer. If that layer was never written, the tier and tripwires are unsupported assertions and the audits are vacuous. Narrated lens-labels create the false impression the deliberation happened.

**How to apply:** When the footer says "pending fill" AND the structural sections are missing, BLOCK. Do not accept inline lens-labels as a substitute. Also: when a doc narrates a genuine tension in prose (e.g. "loser's disposals" — high possession volume losing the scoreboard) but never surfaces it as a structured Tension or encodes it in a tripwire, that is smoothing-by-omission even when the words are on the page. The Exec tripwire often addresses a secondary issue (e.g. player availability) while the brief's actual headline insight goes un-tripwired.

Related calibration: a Probationary/Settled tier with NO upstream Scientist data brief (only an unrelated news piece in docs/news/) is a BLOCK unless the doc explicitly names the missing-upstream and justifies the tier — see edge-case rule. Watch for an upstream filename that matches the teams by coincidence (e.g. a Hird-Essendon coaching piece) but has no data layer relevant to the brief's actual question.
