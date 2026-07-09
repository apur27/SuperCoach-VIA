---
name: datasentinel-nondeterminism
description: DataSentinel returns contradictory PASS/FAIL on identical content; untagged-number flags surface incrementally; how to gate safely around it
metadata:
  type: feedback
---

DataSentinel (the LLM agent) is **non-deterministic on the untagged-number rule** and on how exhaustively it sweeps. The same content hash can get both a PASS and a FAIL audit record minutes apart.

**Why:** Some runs apply the lenient reading ("a prose number that restates an already-[data]-tagged value is an acceptable reference") and PASS; other runs apply the strict hard rule ("any player-stat-shaped untagged number is a FAIL") and flag dozens. Same doc, same bytes, opposite verdict. Observed 2026-07-09 across the F02 five-doc gate set: forgotten-heroes ran PASS, PASS, then FAIL (54 untagged prose numbers itemized); dustin-martin's authoritative gate returned PASS with 0 untagged yet a same-hash FAIL record also existed.

**How to apply:**
- **Ship criterion = zero recorded FAIL at the current content hash.** Before committing a council doc, list every `.claude/audit/sentinel-*.json` whose `doc_path` matches and whose `doc_hash` == the current `council-content-hash.sh` output. Ship only if there is a PASS **and no FAIL** at that exact hash. This directly implements "never publish around a FAIL" and is deterministic. (`check-council-stamp.sh` does NOT do this — it accepts any matching PASS and ignores a co-existing same-hash FAIL. That is a gate hole to fix: make the stamp gate fail-closed on any same-hash FAIL record.)
- **Untagged-number flags are revealed incrementally.** Each pass may surface a new cluster as earlier ones get tagged (Brown skeleton: flagged 9, then 5 more; forgotten-heroes: 1 numeric, then 54 prose). When a doc fails on untagged numbers, tag the WHOLE class comprehensively in one Scientist pass (with a grep self-check), never just the sampled lines — otherwise you loop.
- **Prose restatements of verified numbers are the usual culprit** in long docs: "The 93 goals across 300 games…", "a defender averaging 21.3 disposals…". They restate a value already tagged in the same entry. Fix = add a **[data]** tag to each (Scientist verifies it equals the tagged value). Root-cause harness fix (backlog): make the gate recognise a restatement-of-a-tagged-value, OR enforce the strict rule uniformly so it stops being run-dependent.
- Escalate the non-determinism itself to the human as a gate-reliability defect — a gate whose verdict flips on identical bytes undermines its authority.
