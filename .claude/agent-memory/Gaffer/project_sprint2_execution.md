---
name: sprint2-execution
description: 2026-07-07 Sprint 2 execution — 18 harness/agent-architecture items shipped, Surveyor-verified, F02 human-blocked
metadata:
  type: project
---

# Sprint 2 execution — 2026-07-07 (after the R18 weekly refresh)

Shipped 18 Sprint-2 items in one session, each verify-claim → TDD → commit → push. Test suite 244 → 277. Surveyor verified 17/18 hold; the 2 misses + 1 regression it caught were fixed (D1/D2/D3 below).

**Done:** F01 (CR-1 verified), F03/A-02 (QA schema), F11 (era yaml), A-01 (BriefBuilder Bash + memory dedup), A-03 (FootyStrategy contract), A-04 (Scientist NaN memory scope + invariant), A-05 (DataSentinel memory relocation), A-06 (Gaffer QA-FAIL + Surveyor section + architecture 9-agent), A-07 (council-brief routing + weekly-cycle hardcode), F12 (phantom validator wired), F13 (8 R14 briefs withdrawn), F14a (pre-commit fail-closed), F04 (single entry point + sentinel), S1a (venue claim retracted), F07 (verdict vocabulary + Skeptic records), F05 (insights lane gated), F02a-Gaffer-slice (verify-asof badge + hash + cap rule).

**Blocked / handed off:**
- **F02** (unstrand 5 stat docs): BLOCKED on human editorial call — the D3 era-boundary helper yields live count **17**, not the decision's 16 (Toby Greene crossed live + 3 partial-coverage players Barassi/Skilton/Bisset). Escalated in `pending-decisions.md`. Do not ship the 5 docs until the human picks publish-17-live vs freeze-16-via-F02a.
- **F02a Scientist half:** deterministic round-cap compute helper + skipping as-of docs in any recurring live-re-verify lane. DataSentinel does the cap inline via the prompt rule for now.

**Surveyor defects caught + fixed (the value of the adversarial read):**
- **D1:** the lexicographic "latest file" bug (CR-1) also lived in QA.md §3's prediction-CSV selector (`sorted(glob)[-1]` → validated the round-9 May file). Fixed to `os.path.getmtime`. LESSON: when fixing a bug class, grep the WHOLE repo for the pattern — it recurred in 2 more places (QA §2/§4 which I got, and §3 which I missed).
- **D2:** S1a venue-claim retraction missed `ai-architecture.md` :37 (mermaid) and :609 (fairness) — a grep across `docs/` after a claim retraction is mandatory, not just the obvious file.
- **D3 (important):** the R18 scraped ground truth (438 data files) was stranded uncommitted because the killed `refresh_and_rank.sh` never self-committed AND the doc allowlists exclude `data/`. The R17 cycle DID commit matches+player_data (manual Gaffer stage). Published docs on origin cited data rows absent from origin → remote re-verification fails. Fixed: committed the ground truth, AND added `data/matches/` + `data/player_data/` to `refresh_and_rank.sh`'s allowlist so it self-commits going forward (lineups excluded pending S3 corruption fix). LESSON: a weekly ship must commit the scraped ground truth the docs cite, not just the docs.
