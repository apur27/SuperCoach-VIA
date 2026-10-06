---
name: afltables-reconciliation-run-lessons
description: Methodology lessons + measured baselines from the first full AFL Tables reconciliation run (2026-10-01/02): cache-digest omission caught by a mutation probe, byte-identity recipe, runtime/RSS, what the FAIL is made of
metadata:
  type: project
---

First full run (plan `2026-10-01-full`, snapshot sha256:3de6597513b5..., 17,056 matches, 13,364 profiles): overall FAIL (snapshot FAIL, raw CSV FAIL; existing release checker PASS separately). Four runs (cold-1/cold-4/warm/changed-since) byte-identical.

Lessons (why -> how to apply):
- A mutation probe on a COPY of a rules file caught `_task_digest` ignoring `sum_rules`: units were "reused 260/260" and findings identical after the rule changed. Any new rule/config field read by `season.py`/`cells.py` MUST also enter `compare._task_digest`'s `rules_slice`; test_changing_a_sum_rule_invalidates_the_units_it_governs is the guard. Always rerun the probes after touching digests, and re-run all four final runs (old caches/reports are void when digests change).
- plan refuses a run dir that CONTAINS an input root, so a mutated legacy copy for a probe must live outside the probe's run dir.
- Fast-tier timing: compare against a detached worktree of the base commit on the SAME machine back to back. The old 59.6 s baseline was a quiet-machine number; the same base commit measured 159 s later (load avg ~2.6). Branch delta was +38 s (+24%).
- Measured (12 cores, load ~2.6): cold-1 1,008 s / 1.60 GiB; cold-4 372 s / 2.10 GiB (misses 2 GiB target by ~5%); warm 123 s (misses 120 s by 2.5 s); control (older snapshot, 3.0M findings) 338 s / 2.23 GiB.
- Dominant FAIL drivers: snapshot stores NULL where the source proves a recorded zero (36.6k CELL_LOCAL_NULL, 14.8k Brownlow); pre-1984 season Brownlow totals the local layers do not hold (9.3k LOCAL_MISSING_SUMMARY_VALUE); legacy CSV lacks the 2026 Grand Final; 4 real cell mismatches (1976, one player). Whether null-for-zero should FAIL is a stakeholder decision, not a methodology one.
- worktree test `test_builder::test_browser_readers_accept_the_python_release` fails only because the git worktree has no web/node_modules; it passes in the main checkout.

Corrections run (2026-10-03..06; owner authorised direct CSV + snapshot edits):
- Converged in 3 applied rounds (2: 1.41M changes, 3: 2,132, 4: 4); each round's fixes expose the next layer (e.g. stint totals only after dates). Always re-audit after apply; stop when a round proposes nothing. Final: 0 fail, 63 UNKNOWN cells/layer that AFL Tables itself cannot settle (1932/34 Brownlow, 1974 hit-outs, 2 %P). Do NOT invent rules to turn them PASS.
- Verify corrections with an INDEPENDENT parser (supercoach_via.ingest.afltables shares no code with the reconciliation reader): scripts/reconciliation_spotcheck.py, stratified by rule, seed 20261005. Match ids differ between the two schemes; join fixtures on (team pair, stage label, replay).
- Extra-time finals: legacy match row "final" = LAST score line, not full time (1994 QF, 2007 SF, 2017 EF).
- Stub/duplicate players: a local record whose games are a strict subset of a same-named record of the same profile is the duplicate (harness made a 2nd file for a debutant); teammates' profiles also contain those games, so disambiguate by same-name owner.
- Probes that copy the legacy tree must copy ALL legacy dirs incl. data/awards, else 9k false findings.
- Peak RSS fell 2.10 -> 1.74 GiB by reducing one layer at a time + storing integral totals as ints (aggregate.compact).
- Acceptance-review gotchas (2026-10-06): identity-level change lines (R-ID-BIND/REPAIR/DUPLICATE, R-PLAYER-MISSING, delete_files) carry rule id + source URL but finding_id "" and body_sha256 null -- never claim "every change line carries finding id + body sha". Integrity report `report_sha256` is an EMBEDDED content hash, not the file's byte sha256; label which one. The gate prints "no legacy season changed" only if RECON_DATA_ROOT points at a real snapshot AND RECON_GATE_BASE is overridden (default base sees 130 seasons on the corrected tree) -- a smoke run with those set does not exercise the gate's audit path. Date corrections touch 175,631 legacy rows >=2005 -> days_since_last_game input break for the next backtest.
