# Survey — 2026-10-01 — scope: DEEP, run 2 (confirmation review of the amended AFL Tables reconciliation design)

- **Surveyor model (session metadata):** Opus 5.5, model ID `claude-opus-5-5`. This is the model the user requested; no substitution occurred.
- **Commissioned by:** Gaffer, design-review mode (`docs/rewrite/afltables-reconciliation/ARCHITECT_PROMPT.md`).
- **Reviewed design:** `docs/rewrite/afltables-reconciliation/DESIGN.md` sha256 `2e69b86d6d1abb43fc599a5adc7ef8edbddb66539c6b12390b04065d27d59ff0`. This is the uncommitted working-tree edit on top of HEAD `fec9cd3d8`, read with `git diff`. Line numbers below refer to this hash.
- **Run 1:** `.claude/surveys/2026-10-01-afltables-reconciliation-design-survey.md` sha256 `0e19d9673e37…`. It is byte-identical to `docs/reviews/afltables-reconciliation/surveyor-design-review-2026-10-01-run1.md`.
- **Review record read:** `docs/reviews/AFLTABLES_RECONCILIATION_DESIGN_REVIEW.md` sha256 `420c19da72b0fd3a…` (untracked).
- **Evidence base:** the preserved samples at `var/reconciliations/afltables/design-review-samples-20261001/`. `sha256sum -c SHA256SUMS` reports 54/54 OK, and `p_Dick_Reynolds.html` matches run-1 sample #6 (`422f66ab…`).
  - **Zero network requests** were made in run 2.
  - All numbers below come from executed Python. The tables were read with a colspan-aware parser (scratchpad `cs.py`/`prof.py`).
  - The local data was read through `SnapshotQuery` on the candidate snapshot `sha256:3de65975…`.
- **Read-only:** apart from this file and my memory directory, I made no edits.

---

## Executive read

The amendment faithfully resolves 11 of 16 findings outright, S-01's core state is right, and none of Gaffer's judgement calls let a real local defect pass.

**On S-01, FAIL is the honest category.** The source value is captured and local storage lacks it: 0 of 232,137 local player-games for 1935–83 carry `brownlow_votes`, and there is no season-award table. Both facts are resolved, so UNKNOWN would misreport a known gap as uncertainty.

**However, the amendment introduced two internal contradictions in the acceptance criteria:**
- T14 (l.745) still requires "profile disagrees with … summary → SOURCE_CONFLICT", which is the opposite of the new §8 l.503–508. D06 therefore cannot be satisfied.
- The new aggregate partition (l.541–542) has no bucket for source-unavailable or not-applicable aggregates, even though the review record (§5) says the partitions are "exhaustive".

**Two measured gaps remain:**
- The S-03 career-average model, as written, reproduces at most 2 of 5 sampled career Brownlow averages. The rule "H&A games excluding 1942–45" reproduces 5 of 5.
- The S-07 "per-appearance evidence" class is not exhaustive.

All of these are sentence-level fixes, so nothing is BLOCKING.

**Surveyor recommends: APPROVE_WITH_CHANGES.** Fix N-01 and N-02 before Phase B and N-03 to N-05 before Phase C; bind approval to the post-edit hash.

---

## 0. Correction to my run-1 evidence (Dick Reynolds career Totals BR)

**My run-1 statement "The career Totals BR cell is blank" (run-1 l.47 and l.73) was wrong.** Gaffer is right.

The raw footer of the by-totals table (`p_Dick_Reynolds.html` line 14) is:

```
<tfoot><tr><td colspan=3><b>Totals</td><td align=center><b>320</td><td nowrap align=center><b>188-5-127</td> … <td align=center><b>442</td> … <td align=center><b>154</td> …
           <tr><td colspan=3>Averages</td><td align=center>16.84</td><td align=center>59.53%</td> … 1.38 … 0.66 …
```

- Colspan-aware parse: `Totals {'GM': '320', 'W-D-L': '188-5-127', 'GL': '442', 'BR': '154'}` and `Averages {'GM': '16.84', 'W-D-L': '59.53%', 'GL': '1.38', 'BR': '0.66'}`.
- **Root cause:** my run-1 ad-hoc parser did not expand the `colspan=3` label cell. That shifted every footer column two places left, so the BR position read a blank cell. A naive re-parse in this run reproduced the error: 25 footer cells against 27 header labels.
- **Blast radius checked:**
  - **McMullin career footer values are unaffected.** Re-parsed colspan-aware: Totals GM 49, W-D-L 32-0-17, GL 55, TK 27, HO 1; Averages 6.13, 65.31%, 1.12, 0.93, 0.02. These are identical to run 1.
  - **The 1,028 season-average reproductions are unaffected.** Re-run colspan-aware: 1028 reproduced, 0 misses. Season rows have no colspan.
  - **The production reader is not exposed to this.** `integrity/sourcepages.py:162-165` already repeats each cell `colspan` times.
- **Does the correction change S-01?** The core finding stands: per-game BR blank, season rows printed, never a SOURCE_CONFLICT. The re-verified facts are:
  - per-game BR sum 31 (1933–34 only);
  - summary-only seasons 1935–1950, 12 seasons, BR sum 123;
  - composite 154, which equals the printed career Totals 154.
- **It changes three things in the design, and adds one fixture requirement:**
  - DESIGN l.433-434 already carries the corrected evidence.
  - The career Totals BR is a further *derived* figure that needs a composition rule over mixed per-game and summary-only seasons (N-04).
  - The career Averages BR of 0.66 = 154/234 exposes the Brownlow career-denominator convention, which the amended S-03 model gets wrong (N-03).
  - T12/T33 fixtures must include a `colspan=3` footer, so a reader that skips colspan expansion fails loudly. That is the mistake I made.

## 1. Per-finding resolution (S-01 to S-16)

| ID | Resolution | Where (DESIGN @ 2e69b86d) | Notes |
|---|---|---|---|
| S-01 | **RESOLVED** (follow-ons N-03, N-04, N-08) | §8 l.432-445; §9 l.535-536, 541-542, 556; T33 l.764 | The state, the three-condition rule, "never SOURCE_CONFLICT", the separate count and the owner-reserved narrowing all match run 1's recommended outcome. FAIL is defended in §2a below |
| S-02 | **RESOLVED** (wording follow-on N-07) | state table l.416; l.420-430; l.453-454; §9 l.533-534; T34 l.765 | The four-condition rule is deterministic. The null/zero "representation" category matches run 1 |
| S-03 | **PARTIAL** | l.490-501; T32 l.763 | Fixes run 1's issues: the "denominator exactly" impossibility is gone, HALF_UP is stated, and model misses are classified as source consistency. **But the career-denominator rule (l.495-496) is wrong for BR** (N-03) |
| S-04 | **RESOLVED** | l.289-298, 307-309; T35 l.766 | Feasible against today's code. With `max_attempts=1` the client never sleeps or retries (`http.py:632, 690-692`: sleep only `if attempt < policy.max_attempts`). `_Attempt.retry_after` already exists (`http.py:435`), so the additive `FetchResult` field is a plumb-through. Nit: the persisted next-eligible time must be wall-clock UTC, because `HostRateLimiter` is monotonic (`http.py:254-278`) and does not survive restart |
| S-05 | **RESOLVED** | §7 l.357-367 | Uses set equality only, global uniqueness, Jan-1 DOB as low precision, and partial overlap = UNKNOWN. Side effect (watch list): for a no-DOB player with a genuinely missing local game, rule 4 yields identity-UNKNOWN rather than FAIL. Strict, since UNKNOWN still blocks PASS |
| S-06 | **RESOLVED** | §6 step 6 l.262-266; T31 l.762 | — |
| S-07 | **PARTIAL** | l.503-516; §9 l.555 | The ambiguity is gone, but the per-appearance class is under-enumerated (N-05). **T14 was not amended and now contradicts it (N-01)** |
| S-08 | **RESOLVED** | l.456-463 | — |
| S-09 | **RESOLVED** (stale residue N-06) | l.299-302, 314-316; T17 l.748 | — |
| S-10 | **RESOLVED** | l.144-146; state table l.415; T12 l.743 | Add a colspan-footer fixture (§0) |
| S-11 | **RESOLVED** | l.385-387 | — |
| S-12 | **RESOLVED** | l.274-278 | — |
| S-13 | **RESOLVED** | l.267-268 | — |
| S-14 | **RESOLVED** | §10 l.618-621 | — |
| S-15 | **RESOLVED** | §12 phase H l.713 | The launch README contains no commit commands (`grep commit\|python\|venv` shows ownership lines only), so §12 is the right and sufficient place |
| S-16 | **RESOLVED** | T23 l.754 | — |

## 2. Gaffer's judgement calls — challenged

### a. S-01: is FAIL honest rather than UNKNOWN? Yes. Is the rule evidenced and testable? Mostly.

**FAIL vs UNKNOWN.**
- UNKNOWN (l.557) means "required evidence, identity, schema, drift or availability cannot be resolved". None of these applies:
  - the source value is captured (Reynolds 1935 BR 13);
  - the local absence is measured. `player_games.brownlow_votes` is non-null for 0 of 232,137 rows in 1935–83 and 0 of 100,147 before 1931 (1,247 of 17,024 in 1931–34).
  - `domain/schemas.py` has no table that stores season or award values (table list l.284-630: players … legacy_top100_bios).
- Both sides are resolved and they disagree on presence. UNKNOWN would launder a known completeness gap as uncertainty.
- A published value missing locally is the value-level analogue of a missing appearance, which is already FAIL (l.556).

**Two honest caveats; neither changes the call:**
1. This is a completeness FAIL, not a numeric contradiction. The design keeps it as a distinct category with its own count (l.535, 542), which is sufficient. Because the candidate will FAIL on every run, operators must read the per-category counts rather than the exit code to spot a new regression. That is a consequence to state in the operator guide, not a design defect.
2. **D16 (l.808) is now unreachable for this candidate by construction.** Only two things can change that: a local schema extension that stores season-level award values (Scientist; a data-model change), or the owner-reserved narrowing that the design already names. The design states this permits D09–D15 to finish with a genuine FAIL (l.817-821), so the outcome is consistent, not contradictory.

**Rule evidence and testability.** Each of the three conditions is a deterministic predicate over captured bytes.
- Condition 1 is observed: Reynolds 1935 season BR 13.
- Condition 2 is observed: 1935–50 per-game BR blank (parsed).
- Condition 3 is **observed in other seasons, not 1935**. BR is blank in both player cells and team Totals on the 1897, 1948 (GF replay) and 1975 R11 match pages. By contrast, the 2021 R6 page prints Totals BR 5/1 with player cells (Anderson 3, King 2).
- So the preserved samples cannot fully fixture T33. They contain no H&A match page from a summary-only season of a sampled vote-getter. They also contain no 1931–34 match page to show whether match pages carry the per-game BR the profile prints for 1933–34 (N-08).
- **The rule should state "all" per-game cells blank, and say whose matches** (the player's appearances for that club and season). Today l.436-438 says "per-game cells … blank" and "that season's matches". That ambiguity is folded into N-04.

### b. S-07/S-03: does "derived figures are non-blocking" loosen strictness or let a local defect pass?

- **No local defect can pass through it.** Local cells and aggregates are always judged against the source per-game cells and their exact sums (l.480-481, 506-507). The full run captures every match page (l.257-258), so each per-game cell is double-sourced (profile and match page), and a disagreement blocks (l.510-513).
- When a printed total disagrees while both per-appearance sources agree, the printed figure is the odd one out. Recording it as `source_consistent=false` is the honest call. Blocking would make PASS hostage to a known self-inconsistency in the source.
- **Measured exposure:** printed season totals equal game-cell sums in 1,015 of 1,015 comparisons across the 8 sampled profiles (GM included), with 0 counter discontinuities in 1,370 rows. Reclassification therefore buys almost nothing, and restoring strictness where it is cheap costs almost nothing.
- **Residual loosening:** the "per-appearance evidence" list (l.511-512) names only cell-vs-cell and membership-vs-lineup. It leaves out:
  - the profile's game counter versus the match page's career-games-to-date. These are measured equal for Sexton 2021 R6: profile `Gm` 139, match page "139 (35-0-104 25.18%)".
  - the row result (W/D/L) versus the match result.
  - the jumper number.
  - a per-game cell printed on only one page (possibly BR in 1931–34), which has no double-sourcing.

  If any of these disagree, the unenumerated case defaults to non-blocking "derived", or is unclassified. That weakens strict treatment of missing evidence (N-05).

### c. S-02: NOT_APPLICABLE_DNTF where local null or zero agrees, positive is a mismatch.

- **Sound.** Local zero is supported by the source's own averaging convention, which counts these rows as games in the denominator (run 1: Sexton KI 167/21 = "7.95"). A positive local value claims activity the source rules out, so it is correctly a mismatch.
- Local reach is measured: snapshot rows with null kicks, handballs and `%P` number 2003: 1, 2004: 2, 2021: 202, 2022: 183, 2023–25: 5 each, and **0 in 2011–2015**.
- Two wording gaps (N-07):
  - the cell partition must put agreeing DNTF cells in not-applicable, never in equal or verified, and positive ones in mismatch;
  - the rule is described both as era-scoped ("In 2021–22", "before the rule is extended to them") and as condition-scoped (four year-agnostic conditions). The 2003/2004 rows show the difference matters.

### d. S-04: client `max_attempts=1`, coordinator owns retries, additive Retry-After field.

**RESOLVED; verified against code (see table).** No contradiction remains between l.293 and l.307-309.

## 3. New findings (run 2)

### N-01 — T14 still says a profile-vs-summary disagreement is SOURCE_CONFLICT; this contradicts amended §8 and makes D06 unsatisfiable  [HIGH] [class 10, unwritten/contradictory convention]
- **Evidence:** DESIGN l.745 `| T14 | Source profile disagrees with match or summary | SOURCE_CONFLICT; no convenient source selection |` vs l.503-508 (printed totals and averages are derived, so a disagreement is `SOURCE_DERIVED_INCONSISTENCY` and does not change the verdict) and T32 l.763 (a model miss is "never a local-layer verdict"). D06 (l.784) requires T01–T35 to pass.
- **Impact:** The engineer must write one test asserting that a summary disagreement blocks PASS and another asserting it does not. Whichever is chosen, a gate dispute follows at Phase G.
- **Owner:** Gaffer.
- **Recommended outcome:** T14 covers only per-appearance disagreements (the enumerated list from N-05) → SOURCE_CONFLICT. Summary and average disagreements live only in T32.
- **Effort:** S · Rank 1

### N-02 — The new aggregate partition has no bucket for aggregates with no source reference  [HIGH] [class NEW / accounting]
- **Evidence:** l.541-542: "every requested aggregate belongs to exactly one of equal/mismatch/local-missing-summary/unresolved". The cell partition (l.540) has source-unavailable and not-applicable buckets; the aggregate partition does not.
  - Pre-1965 sampled profiles print season values only for GL (Kelly Robinson 1897–1901, Bill Ahern0 1897) or GL and BR (Reynolds 1933–51). This was executed.
  - So about 20 of the 22 statistics per pre-1965 player-season, plus BR for seasons with no H&A games (Reynolds 1951) and seasons with no award (1942–45), have no source reference.
  - Review record §5 asserts the partitions are "exhaustive and exclusive".
- **Impact:** Implemented literally, these aggregates fall into "unresolved", which means UNKNOWN. PASS (D16) then becomes unreachable by construction for any candidate or narrowed scope, which is the same class as run-1 S-01. Filed as "equal", they inflate verification. The UNKNOWN row (l.557, "No confirmed mismatch") also does not name `LOCAL_MISSING_SUMMARY_VALUE`. Precedence is clear from the FAIL row, but the wording should be "no FAIL condition".
- **Owner:** Gaffer.
- **Recommended outcome:** The aggregate partition adds source-unavailable and not-applicable buckets, mirroring cells, so an all-`NOT_RECORDED` aggregate is neither unresolved nor equal. The UNKNOWN row reads "no FAIL condition".
- **Test:** `test_reconciliation_report.py::test_aggregate_partition_exhaustive_with_source_unavailable`.
- **Effort:** S · Rank 2

### N-03 — The S-03 career-average model is wrong for Brownlow  [MEDIUM] [class 4/10]
- **Evidence:** l.495-496: "career denominator excludes games from eras or matches where the statistic is not recorded, per notes and match Totals". Notes mark BR from 1984, and pre-1984 match Totals BR is blank. Measured across all five sampled careers with a printed career BR average:

  | Profile | Printed | Amended model (H&A, BR recorded from 1984) | H&A excluding 1942–45 |
  |---|---|---|---|
  | Reynolds | 0.66 | 154/0, undefined | 154/234 = 0.66 |
  | Ablett0 | 0.43 | 100/226 = 0.44 | 100/232 = 0.43 |
  | Sidebottom | 0.29 | 16/47 = 0.34 | 16/56 = 0.29 |
  | Pendlebury | 0.56 | 230/409 = 0.56 | 230/409 = 0.56 |
  | Sexton | 0.02 | 3/186 = 0.02 | 3/186 = 0.02 |

  Zero-vote seasons, where the season BR is printed blank, *are* in the denominator: Sidebottom 1983/1987, Ablett 1991, Pendlebury 2006. Reynolds 1942–45 (season BR blank) are excluded.
  - **Inference:** the Brownlow was not awarded 1942–45 `[historical record — unverified in data]`.
  - Profile bytes alone cannot distinguish "zero votes" from "not awarded". Both print blank.
  - For other statistics the notes/era rule holds: all 80 sampled career averages are reproduced, 76 by all games and the TK cases by era exclusion.
- **Impact:** The verdict is unaffected. But every career spanning pre-1984 H&A games or 1942–45 yields a false `SOURCE_DERIVED_INCONSISTENCY`, so `source_consistent` is permanently false for the wrong reason. Meanwhile T32's "exact model reproduction" cannot pass on real fixtures without the post-hoc convention choice l.500-501 forbids.
- **Owner:** Gaffer (text). Scientist supplies a captured, versioned source for the no-award seasons in Phase B/E.
- **Recommended outcome:** The model states that the BR career denominator is all H&A games in seasons where the award was conducted, sourced from captured evidence rather than notes or match Totals. Pre-1984 seasons count.
- **Test:** `test_reconciliation_aggregate.py::test_brownlow_career_average_excludes_no_award_seasons_only` (Reynolds, Sidebottom, Ablett0 fixtures).
- **Effort:** S · Rank 3

### N-04 — Career and stint reference composition across mixed per-game and summary-only seasons is unspecified  [MEDIUM] [class 10]
- **Evidence:** l.435-439 assign `SOURCE_SUMMARY_ONLY` per (player, season, statistic). l.507-508 makes the printed summary "the only reference" for such a value, but nothing defines the career reference when seasons mix. Reynolds BR: per-game sum 31 (1933–34) plus summary sum 123 (1935–50, 12 seasons) = 154, which equals printed career Totals 154. Executed.
- **Impact:** An implementation might use per-game sums only (career reference 31), which would produce false local mismatches or a false derived inconsistency against 154. Or it might use the printed career total (derived) as the reference, which contradicts l.503-507.
- **Owner:** Gaffer.
- **Recommended outcome:** The career and stint reference for a statistic is the sum of per-season references. Each season's reference is its per-game sum, or its season value when `SOURCE_SUMMARY_ONLY`. The printed career Total is compared to that composite only as a derived figure. The rule's conditions read "all of the player's per-game cells for that club-season" and "Totals blank in every one of those matches".
- **Test:** `test_reconciliation_aggregate.py::test_career_reference_composes_pergame_and_summary_only_seasons`.
- **Effort:** S · Rank 4

### N-05 — The "per-appearance evidence" class for SOURCE_CONFLICT is not exhaustive  [MEDIUM] [class 10, strict missing-evidence]
- **Evidence:** l.510-512 lists only "profile cell vs match-page cell" and "profile membership vs match lineup". Per-appearance facts that are captured but not listed:
  - the profile `Gm` counter vs the match-page career-games-to-date (Sexton 2021 R6: 139 vs "139 (35-0-104 25.18%)", measured equal);
  - the row result vs the match result;
  - the jumper number (compared per §2 l.103-104).
  - Cells printed on one page only have no corroboration (see N-08).
- **Impact:** A disagreement among these defaults to the non-blocking "derived" class, or to no class at all. A single-sourced per-game cell can also be contradicted by a printed total without blocking. Measured exposure is near zero (0 of 1,015 total mismatches, 0 counter gaps), so restoring strictness costs no reachability.
- **Owner:** Gaffer.
- **Recommended outcome:**
  - Any disagreement between two source facts that index the same appearance is `SOURCE_CONFLICT`.
  - A printed total that disagrees with per-game cells *lacking match-page corroboration* marks that aggregate unresolved.
  - Only figures computed over multiple appearances (totals, averages) are "derived".
- **Test:** `test_reconciliation_compare.py::test_counter_vs_match_games_to_date_conflict_blocks_pass`, `::test_single_sourced_cells_with_disagreeing_total_unresolved`.
- **Effort:** S · Rank 5

### N-06 — Stale text left by the S-04 and S-09 edits  [LOW] [hygiene]
- **Evidence:**
  - l.153 still says reuse "rate limiter, retry handling and `RawArchive`", but retries move to the coordinator (l.293) and validators are unused (l.299-302).
  - l.285 still lists "conditional requests" among the spaced request types, although none are sent.
- **Owner:** Gaffer. **Outcome:** both lines agree with S-04 and S-09. **Effort:** S · Rank 6

### N-07 — DNTF partition placement and era scope stated two ways  [LOW] [class 10]
- **Evidence:**
  - l.533 "not-applicable (rule-based and did-not-take-field counted separately)" vs l.416, where a positive value is a mismatch. The cell partition (l.540) leaves open whether an agreeing DNTF cell is equal or not-applicable.
  - l.420 "In 2021–22" and l.429-430 "before the rule is extended" vs a four-condition rule (l.423-426) with no year term.
  - Local all-blank rows exist in 2003 (1) and 2004 (2), none in 2011–15.
- **Owner:** Gaffer.
- **Recommended outcome:**
  - Agreeing DNTF cells are not-applicable, never equal and never in `verified_numeric_fraction`.
  - Positive DNTF cells are mismatches.
  - The rule is condition-scoped. Rows outside 2021–22 that meet all four conditions are reported in a pilot-review count until the pilot confirms them.
- **Effort:** S · Rank 7

### N-08 — The preserved samples cannot fully fixture T33 or settle 1931–34 BR on match pages  [LOW] [pilot requirement]
- **Evidence:** The 54 preserved files include match pages for 1897, 1948 GF replay, 1975 R11, 2010 GF ×2 and 2021 R6, but no H&A match page from a summary-only season of a sampled vote-getter and no 1931–34 match page. The profile prints per-game BR for 1933–34; local holds 1,247 non-null BR cells in 1931–34.
- **Impact:** If 1931–34 match pages leave BR blank while profiles print it, the result is either a false SOURCE_CONFLICT (l.510-511) or an unclassified single-sourced cell (N-05). T33 would rest on a synthetic fixture.
- **Owner:** Scientist.
- **Recommended outcome:** The Phase E pilot captures one H&A match page from 1935–83 for a sampled vote-getter (for example Reynolds 1935), plus one 1933 or 1934 Essendon match page. The 1931–34 BR state is decided from that evidence before Phase F.
- **Effort:** S · Rank 8

**No CRITICAL or BLOCKING findings.** Nothing needs the human owner beyond the decision the design already reserves: narrowing S-01's scope. The trade-off in one sentence: keeping pre-1984 season Brownlow in scope makes the primary-snapshot PASS (D16) unreachable until the local schema stores season-level award values, while narrowing it makes PASS reachable but the attestation must name the exclusion.

## 4. Core-requirement check (did the amendment weaken anything?)

| Requirement | Verdict | Evidence |
|---|---|---|
| Complete population | Unchanged; strengthened by S-06 | l.53-64 unchanged; l.262-266 closure identity |
| Deterministic comparison | Unchanged; S-03, S-05 and S-11 add deterministic rules with no similarity scores | l.366, 386-387, 490-501 |
| Input immutability | Strengthened (S-09, S-16) | l.299-302, 314-317; T23 l.754 |
| Strict treatment of missing evidence | **Slightly loosened at one seam** (N-05). Otherwise preserved: DNTF requires all four conditions; summary-only requires all three; all-zero columns stay UNRESOLVED (l.455-456) | l.510-512 |
| Accounting identities | **Contradiction introduced** (N-02) | l.541-542 |
| Acceptance tests coherent | **Contradiction introduced** (N-01) | l.745 vs l.503-508, 763 |

T17, T23 and T31–T35 are coherent with their sections. D03/D06 correctly move to T35. ARCHITECT_PROMPT l.60 and l.81 still say T01–T30; that is historical and needs no change.

## Anti-pattern list (standing)

- Never trust an LLM sum; re-measure disputed numbers in pandas before acting.
- Never verify by exit code; re-read the file content after any write.
- Never `git add .`; stage by explicit allowlist.
- Never hand-edit anything under `data/` or a generated table body.
- Never let a Pass-1 PASS stand in for Pass-2 clearance.
- Never soften an upstream caveat when translating numbers into prose.
- Never run the refresh before round settlement.
- Never push to `main` from parallel agents; serialize through one committer.
- Never define an agent's role, model or tool scope in more than one place.
- Never leave a gate-enforced convention unwritten.
- Never treat a source's derived figure as a peer of its per-game cells (run 1).
- Never let a transport layer's internal retry decide politeness (run 1).
- **NEW:** Never read an HTML table row by zipping header labels to cells without expanding `colspan`. Evidence: my own run-1 misread of the Reynolds career Totals BR; footer label cells use `colspan=3`.
- **NEW:** When an amendment reclassifies a result, grep the acceptance-test table for the old classification. Evidence: N-01, T14 left behind.

## Watch list (speculation, unranked)

- Rule 4 identity (S-05) turns a genuinely missing local game of a no-DOB player into identity-UNKNOWN rather than FAIL. This is strict but less informative; the pilot should count how many players reach rule 4.
- Pre-1984 zero-vote seasons are blank at season level, so they become `NOT_RECORDED`, not `SOURCE_SUMMARY_ONLY`, and local null agrees. If any local layer ever stores a pre-1984 BR zero, it becomes an "unsupported local numeric claim", which means UNKNOWN (l.564-565). Today no layer does (0 non-null).
- The 2003 and 2004 locally all-blank rows (3 total) may be DNTF-like or data errors; they belong in the pilot.
- Courtesy note carried from run 1: a roughly 17 h crawl of a volunteer site.

## What was checked and found clean

- 54/54 preserved sample hashes; no network use in run 2.
- The S-04 code path (`http.py:52-73, 254-278, 341-359, 425-435, 501-503, 522-544, 606-705`): `max_attempts=1` gives no in-client sleep or retry; every redirect hop passes the limiter (l.544); `_Attempt.retry_after` exists.
- Season averages: 1,028/1,028 reproduced, colspan-aware. Career averages: 80 sampled, all explained by era/award exclusion.
- Printed season totals = game-cell sums: 1,015/1,015. Counter contiguity: 0 gaps in 1,370 rows.
- Pre-1984 match pages (1897, 1948, 1975): BR blank in player cells and Totals. 2021: BR present. 2010 GF: BR blank (finals).
- `sourcepages.py:162-165` expands colspan.
- §12 phase H S-15 text; README has no commit recipe.
- The review record's dispositions table matches the DESIGN text for S-01..S-16 except as noted (S-03, S-07, the §5 "exhaustive" claim).

## Inventory

- DESIGN.md `2e69b86d6d1abb43fc599a5adc7ef8edbddb66539c6b12390b04065d27d59ff0`; ARCHITECT_PROMPT `2135e6c3…`, ENGINEER_PROMPT `c013915c…` and README `12feee65…` are unchanged since run 1.
- `docs/reviews/AFLTABLES_RECONCILIATION_DESIGN_REVIEW.md` `420c19da72b0fd3a0bef23b62ce5b4356d3a79c8e28bb281973f507780644f9c`.
- `ingest/http.py` `58141a151c50d948…`, unchanged since run 1.
- Samples dir: `var/reconciliations/afltables/design-review-samples-20261001/` (SHA256SUMS 54 lines, all OK).
- Local: candidate snapshot `sha256:3de6597513b5c6eaca1b102d304a2cf128d8c08f229ee1de384c1d40fe746db0` via `SnapshotQuery`. Legacy CSV `data/player_data/reynolds_dick_20061915_performance_details.csv`: BR non-null 5 cells in 1933 (sum 12) and 8 in 1934 (sum 19); 0 in 1935–51.
