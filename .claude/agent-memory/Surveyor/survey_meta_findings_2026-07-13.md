---
name: survey-meta-findings-2026-07-13
description: 2026-07-13 META survey of agent definitions, A2A chain, skills, CLAUDE.md, memory — 12 routed findings (M1-M12) + Surveyor.md self-edit list (S1-S8)
metadata:
  type: project
---

# META survey 2026-07-13 — open findings (all reported in chat to owner; no survey file written per harness note)

All OPEN unless marked. Re-verify before re-reporting. See [[survey-open-findings]] for the data-layer ledger.

**2026-07-25 re-verification:** RETIRED M1 (Gaffer.md chain+stamp now include QA at :59/:64/:102),
M3 (tagging exemption codified in CLAUDE.md — but the gate it cites doesn't run: see P25-F1 in
[[survey-open-findings]]), M5 (last_refresh_complete.json exists; 3 full cycles ran), M6-spec-half
(DataSentinel.md:31 now references data-tag-spec.md), M8 (consult rule codified Gaffer.md:61),
M10 ("currently empty" line gone), M2-core (harness uses `--agent FootyStrategy`, no --model
override). STILL OPEN: M4 (HOF doc owner — now folded into P25-F1), M9 (insights Skeptic
exemption, human decision, still no Skeptic in weekly_refresh.sh), M12 (dual registers), M7
residue (Chronicler skipped again for 07-20 cycle).

- M1 HIGH — Gaffer.md internal contradiction: chain at :59-61 omits QA/Chronicler vs full chain at :106; stamp template :68 omits QA:PASS vs :120 & council-brief.md:101. Owner Gaffer.
- M2 HIGH — FootyStrategy harness drift: weekly_refresh.sh:267 `--model sonnet` vs FootyStrategy.md:4 `model: opus`; allowedTools exclude Bash; FootyStrategy.md still calls Phase-3b gating "Sprint-2 work — F05" (implemented at weekly_refresh.sh:279-291). Owner Gaffer + human model decision.
- M3 HIGH — HOF top-100 tagging convention written nowhere; ~2 inline [data] tags in 100 profiles vs DataSentinel untagged-number FAIL rule. HUMAN DECISION then Gaffer codifies.
- M4 HIGH — docs/hall-of-fame-top100.md has no owning agent in any definition. Owner Gaffer (add to FootyStrategy surfaces).
- M5 MED-HIGH — harness never completed end-to-end since fixes: no last_refresh_complete.json anywhere; last weekly_refresh log 2026-06-22. First live run = R20 2026-07-14; recommend supervised dry run.
- M6 MED — DataSentinel.md:7 tools Read,Grep,Glob,Bash vs runtime registry incl. Write,Edit; DataSentinel.md never references docs/data-tag-spec.md (BriefBuilder.md:63 mandates it). Owner Gaffer.
- M7 MED — weekly_refresh.sh stale comments :15, :126-127 claim "DataSentinel agent updates HOF pages" (actual: deterministic update_hof_pages.py, :156-161); log labels [N/5] but no phase 5; Chronicler skipped since 07-07 (run-reports stop at 2026-07-07-weekly-r19.md). Owner Gaffer.
- M8 MED — "Gaffer consults Scientist before escalating" lives only in user auto-memory, not Gaffer.md. Owner Gaffer.
- M9 MED — afl-insights.md ships DataSentinel-gated but never Skeptic-gated; exception undocumented. HUMAN DECISION (add gate vs document exception).
- M10 LOW-MED — stale boilerplate in definitions: Scientist.md:288 "MEMORY.md is currently empty" (false — 37 files); per-agent memory-rules blocks of varying vintage. Owner Gaffer.
- M11 LOW — CLAUDE.md gaps: canonical top-100 CSV unnamed (root + data/top100 both exist); no pointer to docs/data-tag-spec.md; no single-writer push rule. Owner Gaffer + human.
- M12 LOW — open-findings tracked in two unsynced registers (Surveyor memory ledger + Gaffer backlog). Owner Gaffer: pick one canonical register.

Surveyor.md self-edits S1-S8 handed to Gaffer 2026-07-13: explicit tools line; description triggers (post-incident, meta-audit); new META scope; recurrence classes 9 (two-sources-of-truth drift) & 10 (unwritten conventions); class-1 extended to membership drift; class-4 extended to model-tier fitness; gate-semantics verification (prove the failure path); ownership map + QA/Chronicler; survey filename slug; delta-only re-reporting; 2 anti-pattern additions.

Watch: RTK/headroom compression proxy lossily corrupted large tool outputs during this survey (expired retrieval hashes). If gate agents run through same proxy, quoted verification output could garble. Global config, human call.
