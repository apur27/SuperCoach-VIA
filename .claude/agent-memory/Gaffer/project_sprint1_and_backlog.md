---
name: sprint1-and-backlog
description: Sprint 1 shipped (commit e84cd41f9) — what landed, the 6-FAIL Scientist remediation buckets, and the Sprint 2 backlog with priority
metadata:
  type: project
---

**Sprint 1 shipped 2026-07-03, commit e84cd41f9 on origin/main** (33 files, TDD, 239 tests green).

## Landed
- Q1 AUDIT_ENFORCE=1 default (stamp trusted only if a content-hash record backs the STAGED content). DataSentinel records its verdict (workflow step 8); HOF lane records its `check_hof_numbers` verdict so enforce=1 doesn't brick the weekly refresh.
- F2 single tag-vocabulary source (`scripts/tag_vocabulary.py`) — sampler/badge/DataSentinel share one matcher; sampler had gone vacuous missing `**[data: spec]**`. DataSentinel enumerates via CLI + masks spans so its untagged backstop stays independent.
- F4 gate hashes the STAGED blob (`git cat-file`), symlink-in-index fails closed.
- F6 legacy news exempt by exact filename; coaches-strategy-corner opt-in-sticky; FootyStrategy recap mandates bold `**[data]**`.
- Only-Gaffer-commits enforced (auth marker in `git_commit_safe.sh` + hook block; escape `--no-verify`).
- Trust badge shipped on 15 freshly-verified docs (13 HOF + neale-daniher + greg-williams).

## Scientist remediation (routed, in flight) — see [[legacy-doc-staleness]]
- **Bucket 2 authoring/data errors (priority FIRST):** jonathan-brown (46≠49 arithmetic; 2009 rank), dustin-martin ("12 players"→16 scan; active-peer counts), forgotten-heroes (Sewell/Chad Cornes/Hewett retired-player miscounts + Hopper drift).
- **Bucket 1 pure staleness (SECOND):** list-quality-draft (R9→R15), free-agency (~2-round drift), 5yr-grand-final (Geelong/Collingwood). After fix → DataSentinel re-verify → Gaffer re-badge + commit.

## Chronicler top-3 forward recs (retro headline: "a badge is only as good as its last check")
1. **Scheduled staleness re-verification gate** `scripts/reverify_stale_docs.py` — re-run DataSentinel against badged docs citing active-player/current-season counts on a cadence; strip badge + WARNING on FAIL. HIGHEST leverage (3 of 6 FAILs were pure staleness ⇒ one-shot verification decays). ~0.5-1 day; all components live. **The thing the 6/8 rate most directly demands — promote to Sprint 2 headline.**
2. **Badge classes deterministic vs LLM-verified** in `inject_trust_badge.py` — HOF pages self-heal from JSON; news prose rots. One badge hides a real trust gap. ~2-4h.
3. **Renamed-doc gate** `--diff-filter=ACMR` in pre-commit — `git mv` currently slips a stamped doc past re-verification under a new name. ~2-3h (incremental on F4).

## Sprint 2 backlog (approved order)
Rec-1 staleness re-verification gate (Gaffer, headline); Q2a hard-abort refresh audits + DOB-collision FP guard (Scientist); Q2b pandera on CSV write (Scientist, pandera not installed); I1 executable-data-tags design spike (Scientist+Gaffer); afl-insights.md deterministic recap gate + Phase-3 record wiring (mirror HOF lane); renamed-doc gate (rec-3); badge classes (rec-2); F5 extend record to Skeptic; F8 architecture.md §5 verbatim-prompt drift; F9 weekly_refresh wiring (unreachable HOF error branch under set -e, masked git-add failures, push skips pull --rebase); HOF stamp-agent-mismatch (stamps say DataSentinel:PASS but records are check_hof_numbers).

## Loose ends
- `.claude/agents/Surveyor.md` + `.claude/surveys/` untracked, intentionally NOT committed in Sprint 1 (out of scope) — decide whether to version the Surveyor agent.
- Consult Surveyor before committing any complex solution (standing user directive).
