---
name: survey-open-findings
description: Findings routed by past surveys not yet confirmed fixed — the first thing the next survey re-checks
metadata:
  type: project
---

Re-check these first on the next survey; retire each line only after verifying the
fix by content, not by claim.

## Verified fixed 2026-07-07 (Sprint 2 verification pass, HEAD 798ff2557)
Retired: F1 (BriefBuilder Bash+executed-computation), F2 (Step 1b in council-brief),
F3 (team-name trio → 1 memory), F4 (DataSentinel memories, 4 files + index),
F6/F7 (FootyStrategy HANDOFF CONTRACT + description), F10 (Skeptic BLOCK routing
table by nature), F11-schema (QA.md stat-leaders snippets), F13 (SURVEYOR
INTEGRATION in Gaffer.md), F14 (architecture.md §3 = nine agents), F16 (scope
boundary in both Scientist memories), F17 (Scientist.md hard rule 13), F21
(weekly-cycle date removed). Also verified: CR-1/F01 (ls -t), F04, F05, F07,
F12, F13-withdrawal, F14a, F02a, era yaml F11. Suite: 277 passed.

## NEW from 2026-07-07 verification pass
- D1 **QA.md §3 prediction-CSV snippet uses lexicographic sorted(glob)[-1]** —
  executed proof: selects next_round_9_prediction_20260511 (stale round 9), not
  mtime-latest next_round_19. Same CR-1 class F01 just fixed, living inside the
  QA gate itself (.claude/agents/QA.md:61-63). HIGH, gate defect → escalated to
  human, routed Gaffer.
- D2 **S1a residuals in docs/ai-architecture.md** — line 37 mermaid node B1 lists
  "venue" as a model feature; line 609 says model "uses ... (rolling form,
  opponent, venue, context)". Contradicts corrected line 129. LOW-MED, Gaffer.
- D3 **Scraped ground truth never committed by harness** — 426 data/ files
  (matches_2026.csv, all lineups, player_data) modified 12:13 2026-07-07,
  uncommitted; neither weekly_refresh.sh Phase 4 nor refresh_and_rank.sh
  allowlist stages data/matches|player_data|lineups. R17 cycle DID commit them
  (commit "Weekly refresh R17: scrape actuals..."), so a step regressed or was
  manual. origin/main docs claim R18 actuals its data lacks. MED-HIGH, Gaffer.
- OBS: 7 round-13 briefs still carry footer "Interpretation pending FootyStrategy
  fill of `<!-- FOOTYSTRATEGY INSERT -->` markers" (string, not live markers).
- OBS: F02-blocked docs (dustin-martin, 3 news, hall-of-fame-forgotten-heroes)
  sit modified-uncommitted in the tree while the decision is pending.

## Still open / not yet re-verified
- F8 FootyStrategy tripwire learning loop never fired — MEDIUM
- F19 Chronicler output commit ownership ambiguous — MEDIUM (not re-checked)
- F02 unstrand 5 stat docs — BLOCKED on human (pending-decisions.md, count 17 vs 16)
- F02a Scientist half: deterministic round-cap helper + skip as-of docs in live lanes
- From 2026-07-03: F1 skeptic_sample_tags spec-form blindness; F2 HOF stamp
  attribution; F3 AUDIT_ENFORCE=1 record-less envs; F4 commit marker doesn't
  serialize — all still unverified.
