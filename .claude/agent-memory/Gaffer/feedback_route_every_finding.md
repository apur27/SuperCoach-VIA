---
name: route-every-finding
description: When a gate BLOCKs with multiple findings, route ALL of them — never tell a downstream agent a finding is "already cleared" unless a gate actually cleared it
metadata:
  type: feedback
---

A BLOCK/FAIL verdict is a checklist, not a headline. Enumerate every finding it
contains and route each one. Never describe a finding as "already fixed and
cleared" to a downstream agent unless a gate re-ran and said so.

**Why:** 2026-07-26, `docs/afl-insights.md`. Skeptic's pass-2 BLOCK carried FOUR
findings — three on line 28 (SK-R21-01/02/03: a causal predicate, an unsourced
superlative, a wrong time window) and one on line 24 (SK-R21-04: ladder
comparative + season superlative). I read the newest finding as the whole verdict,
routed only line 24 to FootyStrategy, and explicitly told it the tactical note was
"already fixed and cleared". It was not: line 28 had been rewritten after pass 1,
and pass 2 blocked the REWRITE. Pass 3 therefore BLOCKed on findings that were
never sent to anyone — burning a full gate cycle and consuming the user's
"three strikes then escalate" budget on my own scoping error.

The trap: a rewrite landing successfully is easy to confuse with a rewrite being
accepted. Between pass 1 and pass 2 the line genuinely changed (the pass-1 phrases
were gone), which made "line 28 is done" feel true. Only the verdict decides.

**How to apply:**
- On any FAIL/BLOCK, list the finding IDs before dispatching, and map each to the
  agent that will fix it. If a finding is deliberately NOT routed, say why.
- Re-gate the whole document, not the edited line — later findings can attach to
  text an earlier pass did not reach.
- When a subagent reports "fixed", verify on disk before treating it as cleared,
  and still require the gate's verdict. Skeptic independently logged the same
  lesson this cycle as "verify the fix, don't trust the fix report".
- **Never pipe a gate invocation through `head`/`tail`.** Capture the full output to a
  file and read it from there. On 2026-07-27 a Skeptic BLOCK on
  `docs/afl-backtest-2026.md` was run with `| tail -12`, so only the closing lines
  survived: the verdict was recorded but every finding was gone, and the whole run
  had to be repeated. The audit record does not save you here — it stores the verdict
  and nothing else, which is precisely BL-05.
- Beware a reviewer's *diagnosis* of why something is unfixed. Skeptic inferred the
  edit had been "lost before staging"; it had not — it landed and was then blocked.
  Its substantive finding was right, its explanation wrong. Check the file.

Related: [[verify-routed-findings]], [[gates-hardening-20260725]].
