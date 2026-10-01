# Survey — 2026-10-01 — scope: PULSE-depth confirmation, run 3 (AFL Tables reconciliation design)

- **Surveyor model (session metadata):** Opus 5.5, model ID `claude-opus-5-5`. This is the model the user requested; no substitution occurred.
- **Reviewed design:** `docs/rewrite/afltables-reconciliation/DESIGN.md`, sha256 `da64c4cf5974d9ae4e5a638ef218de247a7dc42508262145778b2e802150797f`.
  - It is uncommitted on top of HEAD `fec9cd3d8`: `git diff --numstat` gives 194 insertions and 37 deletions, 857 lines in total.
  - Line numbers below refer to this hash.
- **Baseline:** my run-2 survey, sha256 `a0262a3172b82710399b371d365f4bd5816da4c95aa0c31bdcecf07559f7e9d9`. It reviewed DESIGN `2e69b86d…` and raised findings N-01 to N-08.
- **Evidence:** the preserved samples still verify (`sha256sum -c SHA256SUMS` gives 54 OK). This run is a text review, so no new measurement was needed.
- **Network:** zero requests.
- **Edits:** read-only, apart from this file.

## Executive read

All eight run-2 findings are **RESOLVED** in the text. The new edits keep the four sections consistent with each other: the state table, the §9 partitions, the PASS/FAIL/UNKNOWN table, and the §13/§14 T- and D-rows. They also do not weaken complete population, deterministic comparison, input immutability or strict treatment of missing evidence.

Two LOW clarifications came out of the new text. Neither affects a verdict and neither blocks. **Surveyor recommends: APPROVE.** The two LOW items can be taken in Phase B as test-naming clarifications, and approval should be bound to hash `da64c4cf…`.

## 1. Run-2 findings

| ID | Status | Where (DESIGN @ da64c4cf) | Note |
|---|---|---|---|
| N-01 T14 contradiction | **RESOLVED** | T14 l.774; §8 l.522-538; T32 l.792 | Under T14, a disagreement between per-appearance facts is a SOURCE_CONFLICT, and a disagreement between a printed total and single-sourced cells is unresolved. Totals and averages compared against double-sourced cells move to T32. D06 can now be satisfied. |
| N-02 aggregate partition | **RESOLVED** | §9 l.560-561, l.568-571; UNKNOWN row l.586; T33 l.793 | Adds source-unavailable and not-applicable buckets. An aggregate whose sources are all `NOT_RECORDED` is source-unavailable. The UNKNOWN row now reads "No FAIL condition". |
| N-03 BR career denominator | **RESOLVED** | l.509-515; T32 l.792; Phase E l.739 | The denominator is home-and-away games in seasons where the award was conducted. The list of no-award seasons comes from a captured, versioned, hashed evidence file and is never inferred from blanks. 1942–45 is correctly labelled unverified. |
| N-04 composed references | **RESOLVED** | l.439-441 ("**all**", "**every**", "for the player's team"); l.445-450; l.528; T33 l.793 | Each club-season contributes its per-game sum, or its season value when `SOURCE_SUMMARY_ONLY`. The printed career Total is compared only as a derived figure. |
| N-05 per-appearance class | **RESOLVED** | l.522-538; T14 l.774 | Counter, result and jumper are now enumerated. Single-sourced cells under a disagreeing total are unresolved. "Derived" is limited to figures computed over several appearances. See R3-02 for the one-page carve-out. |
| N-06 stale S-04/S-09 text | **RESOLVED** | l.153; l.285-286 | The rate limiter is described as "a floor only". Retries belong to the coordinator and validators are unused. "Conditional requests" no longer appear in the spacing list. A grep found no other residue. |
| N-07 DNTF bucket and era | **RESOLVED** | state row l.416; l.420-433; §9 l.557-558; T34 | Agreeing null or zero is not-applicable, never equal, and never in `verified_numeric_fraction`. A positive value is a mismatch. The rule is scoped by its conditions, and rows outside 2021–22 also get a pilot-review count. |
| N-08 pilot evidence | **RESOLVED** (design side) | Phase E l.739 | The pilot captures four things: a 1935–83 home-and-away page, a 1933/34 Essendon page, the no-award evidence and a 2011–15 substitute. The exit evidence names the 1931–34 BR state, the DNTF scope and the evidence file. Execution remains Scientist's job in Phase E. |

## 2. Contradiction and weakening check

**Contradictions.** None were found across the state table (l.405-416), §8 (l.420-538), the §9 partitions (l.557-571), the outcome table (l.584-586), T14/T32–T35 (l.774, 792-795) and D03/D06/D16:
- `LOCAL_MISSING_SUMMARY_VALUE` is consistent across l.453, l.561, l.585 (FAIL) and T33.
- `SOURCE_DERIVED_INCONSISTENCY` is consistent across l.524, l.584 (does not block) and T32.
- `SOURCE_CONFLICT` is consistent across l.532-536, l.584, T14 and D16.
- The not-applicable handling of DNTF is consistent across l.416, l.558 and T34.

**Weakening.**

| Requirement | Verdict |
|---|---|
| Complete population | Unchanged. No diff hunk touches §3–5 except l.17 and l.130-146. |
| Deterministic comparison | Strengthened: N-03 requires an evidence file with no memory or inference, and N-04 adds a composition rule. |
| Input immutability | Unchanged. |
| Strict treatment of missing evidence | Strengthened: N-05 makes single-sourced aggregates unresolved and enumerates per-appearance conflicts. |

## 3. New findings (LOW, non-blocking)

### R3-01 — The aggregate bucket is under-specified for mixed compositions  [LOW] [class 10]
- **Evidence:**
  - l.568-571 makes the aggregate partition exclusive, but only the *all*-`NOT_RECORDED` case is assigned.
  - Two cases have no stated bucket:
    1. A career that mixes recorded and `NOT_RECORDED` club-seasons. Examples are the Reynolds 1942–45 BR seasons, and KI for a career spanning 1964–66. l.447 does not say whether a `NOT_RECORDED` club-season contributes nothing or an undefined value to the composite.
    2. An aggregate that is both a mismatch and local-missing-summary. Example: a local career BR of 31 against a composite of 154.
  - l.522-523 also defines "printed season totals" as derived, while l.443/447 makes a `SOURCE_SUMMARY_ONLY` season value the reference. Specific-over-general resolves this, but the carve-out is not written down.
- **Impact:** Counts only. The verdict is the same either way: both candidate buckets are FAIL in case 2, and the cells are already accounted in case 1. The risk is that two implementations produce different partition counts, which would break T20 determinism across reviewers' expectations.
- **Owner:** Gaffer.
- **Recommended outcome:**
  - `NOT_RECORDED` club-seasons contribute nothing to the composite and are counted as unavailable.
  - An aggregate with at least one recorded contributor is compared over the recorded scope.
  - `LOCAL_MISSING_SUMMARY_VALUE` takes precedence over mismatch for bucket assignment.
  - A `SOURCE_SUMMARY_ONLY` season value is the reference, not a derived figure.
  - These can be stated in one sentence each, or fixed by T33 test names in Phase B.
- **Effort:** S

### R3-02 — The pilot-decided case "printed on one page, blank on the other" needs a stated default and a codified rule  [LOW] [class 10]
- **Evidence:** l.536-538 leaves this state to the Phase E pilot. The text names neither of these:
  - **The interim or default state.** By l.532-533 this case is a disagreement between two facts about the same appearance, which makes it SOURCE_CONFLICT.
  - **The form the pilot's decision takes.** Elsewhere, rule IDs are hashed into the plan (l.474).
- **Impact:** Low. Without the sentence, a pilot conclusion could be applied as an informal convention rather than as a hashed rule ID, which is the "unwritten convention" class.
- **Owner:** Gaffer for the text; Scientist for the Phase E decision.
- **Recommended outcome:** Any one-page case not covered by an evidence-backed rule ID is SOURCE_CONFLICT. The pilot's decision is recorded as a versioned rule ID with captured evidence locators.
- **Effort:** S

**No BLOCKING, CRITICAL, HIGH or MEDIUM findings.** Nothing new needs escalating to the human beyond the S-01 scope-narrowing decision the design already reserves for the owner (l.453-456).

## Anti-pattern list (standing)

Carried from run 2 unchanged. No additions or retirements.

## Watch list (speculation, unranked)

- There is an asymmetry for single-sourced cells:
  - A printed *total* that disagrees is unresolved (blocks).
  - A printed *average* model miss over the same cells is a non-blocking `SOURCE_DERIVED_INCONSISTENCY` (T32).

  This is defensible, because an average miss may be a model error, but the pilot should note any case where it matters.
- Items carried from run 2:
  - rule-4 identity UNKNOWN for no-DOB players;
  - 3 locally all-blank rows in 2003–04;
  - the courtesy note on a crawl of roughly 17 hours.

## What was checked and found clean

- All N-01..N-08 text locations listed above.
- A grep of the whole document for the terms `conditional`, `validators`, `2021–22`, `No confirmed mismatch`, `T01–T30` and `exhaustive` found no stale residue.
- The cross-references between the state table, §9, the outcome table, T-rows and D-rows (§2 above).
- The diff hunk map, which confirms the population, identity-census and output-safety sections were not loosened.
- The 54 preserved sample hashes, all OK.
