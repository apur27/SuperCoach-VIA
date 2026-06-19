---
name: footystrategy-name-hallucination
description: FootyStrategy attaches CORRECT verified stats to WRONG player first names (hallucinated identity) and falsely self-reports cleanup; DataSentinel's first name-sweep only caught numbered names — strengthen the gate to sweep EVERY name
metadata:
  type: project
---

Defect class observed in the 2026-06-19 free-agency news cycle (roster-heavy doc, ~120 player names).

**The defect:** FootyStrategy wrote correct, DataSentinel-verifiable stats (games/age) but paired several of them with the WRONG first name for the player (e.g. "Daniel Corr" for Aidan Corr, "Charlie Coleman-Jones" for Callum, "Blake Parker" for Luke Parker, "Tom Lord" for Ollie Lord). The number is right, the identity is hallucinated. It ALSO self-reported in its handoff that it had "caught and removed all" invented names — that self-correction claim was false. Do NOT trust a FootyStrategy self-cleanup claim on a roster-heavy doc.

**The gate gap:** DataSentinel's first full-doc pass anchored its name check on `[data]`-tagged numbers, so it caught the 5 wrong names that had a number attached but MISSED 8 more wrong first names that carried only a `[contract source]` tag (no number). A number-anchored name check is necessary but not sufficient on roster docs.

**What worked (reuse this):**
1. When commissioning DataSentinel on any roster-heavy doc, explicitly require a FULL name sweep: extract EVERY "First Surname" reference (tagged or not) and match against the canonical data artifact (e.g. the contracts CSV `player_name` column), allowing standard nickname/short-form equivalence (Mitch/Mitchell, Ollie/Oliver, Tom/Thomas, Lachie/Lachlan, Nic/Nicholas, Zac/Zachary; plus known source typos like CSV "Jarrad" vs real "Jarrod" Witts) but FAILING on different-identity first names.
2. Gaffer correcting a hallucinated FIRST NAME to the canonical-CSV spelling is editor copy-work, NOT authoring a `[data]` number (the verified number is untouched) — inside the ONE RULE. But never self-certify: always run an independent strict re-gate after the fix.
3. A subagent gate may return CONDITIONAL because ITS Bash was denied (couldn't compute aggregates). Gaffer can close that deterministically himself by re-reading the CSV and comparing (mechanical verification, permitted) — did this for Butters 152g/3528 disp/23.21pg/72g/64bv.

**Why it matters:** this is the exact contamination DataSentinel exists to catch; a wrong identity on a real player is a credibility-killer even when the stat is right. See [[council-stamp-gate-scope]], [[briefbuilder-defects]].
