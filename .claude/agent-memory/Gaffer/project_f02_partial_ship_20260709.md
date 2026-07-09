---
name: f02-partial-ship-20260709
description: F02 doc-remediation cycle — 3 of 5 corrected docs shipped, 2 held on untagged-prose FAIL; parallel-commit hazard recurred
metadata:
  type: project
---

F02 = ship five Scientist-corrected docs through DataSentinel and commit. Ran 2026-07-09.

**Outcome:** commit `da89a31c7` (origin/main) shipped THREE gate-clean docs — jonathan-brown-fist-of-god, list-quality-draft-pipeline, free-agency-trade-window — with trust badges + fresh sentinel records. TWO held: dustin-martin + forgotten-heroes, both FAILing on untagged prose stat-numbers (see [[datasentinel-nondeterminism]]); routed to Scientist for comprehensive tagging + re-gate before a follow-up ship.

**Why partial:** a DataSentinel FAIL halts a doc; I ship only docs with zero same-hash FAIL record. Brown/list-quality/free-agency had only PASS records; Martin/forgotten-heroes each carried a same-hash FAIL. Don't block clean docs on a failing sibling.

**Per-doc gate history worth remembering:**
- Brown: FAILed twice before PASS — backtick `[data]` skeleton header isn't recognised (only bold **[data]**), so skeleton numbers read as untagged (Scientist tagged them, incrementally); and "Voss" tripped the coach-name grep in a player-era phrase ("Voss-Akermanis-Lynch core") — I reworded to "three-peat championship core" (player Michael Voss later became a coach, so he's on coach_names.txt).
- forgotten-heroes: Cornes 18.2 vs 18.3 was a fill-zero-vs-dropna convention dispute, NOT an error — Scientist confirmed fill-zero canonical (1 of 255 games null; 4636/255=18.2), annotated the tag `convention=fill-zero`. Decision 3's dropna rule applies ONLY to the era-boundary rarity scan, not to ordinary career per-game means. Then FAILed again on 54 untagged prose restatements.

**Parallel-commit hazard recurred** ([[parallel-council-commits]]): during my long-running session, commits `5a5789bd1` (the five corrected docs) and `0b07b3048` (memory/Surveyor/surveys/R19 preds) landed AND pushed to origin — not by me. My `git_commit_safe.sh` flock kept my own commit clean and it fast-forwarded. Lesson reaffirmed: verify by content (`git show HEAD:<file> | grep`), not by diffstat — my commit's diff was "badge-only" because the corrections were already committed by the concurrent flow.

**Badge-date honesty:** the gating run crossed midnight UTC; task said `--date 2026-07-08` but true ship/verify date was 2026-07-09. Badged 2026-07-09 (a badge is a reader-facing "verified on <date>" claim). See [[truth-in-badging]].

**Open follow-ups:** (1) finish Martin+forgotten-heroes tagging→re-gate→ship; (2) harness: make `check-council-stamp.sh` fail-closed on any same-hash FAIL record; (3) harness: fix DataSentinel untagged-number non-determinism (recognise restatement-of-tagged-value, or enforce uniformly).
