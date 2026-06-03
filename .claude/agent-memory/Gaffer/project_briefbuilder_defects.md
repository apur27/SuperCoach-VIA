---
name: briefbuilder-defects
description: Three recurring BriefBuilder defect classes found in the round-13 2026-06-02 cycle (sort-order, NaN-handling, win-count source) + DataSentinel arithmetic slip
metadata:
  type: project
---

Found across the round-13 brief cycle (2026-06-02). Three BriefBuilder defect classes, two systemic. Surfaced by DataSentinel re-gates; verified by Gaffer reading the CSVs directly.

**1. Sort-order transcription bug (SYSTEMIC).** BriefBuilder reads player performance files in file order, not round/date order, so stat sequences get matched to the wrong player. Hit Oliver Clayton AND Thomas Harvey in the SAME cycle (Harvey claimed 31.4/32.2 disposals vs actual ~20.6). Note: both players have a given-name-as-surname pattern (Clayton, Harvey, also `thomas_harvey_08082005`), which is likely how the mismatch evades notice.
**Why:** stats keyed by row position instead of explicit (player_id, round).
**How to apply:** require BriefBuilder to sort by round/date and key every stat to an explicit (player_id, round) tuple; add a deterministic pre-gate assert that round sequence is monotonic and player_id matches the requested player. Deterministic check, not LLM judgement. See [[feedback_parallel_council_commits]] for the "verify by content not command success" principle.

**2. NaN/blank-row inconsistency (SYSTEMIC, needs Scientist decision).** BriefBuilder inconsistently includes/excludes blank stat rows in mean calcs. Reproduced on Sinclair tackles: dropna=1.82 vs fillna(0)=1.67 (1 blank row); Impey tackles: dropna=2.33 vs fillna(0)=1.75 (3 blank rows). The "correct" value is AMBIGUOUS until the blanks=0 vs blanks=excluded convention is decided. St Kilda v Hawthorn brief failed twice on this; routing back for another retry without settling the convention just produces another wrong number.
**Why:** no repo-wide NaN convention pinned for derived means.
**How to apply:** this is a Scientist call (blank = "did not play"/excluded, or "played, scored 0"/counted). Once decided -> config + pre-gate assert that fails on unhandled NaN. Do NOT let Gaffer pick the convention (data-logic, out of lane).

**3. Win/derived-count from single gap-prone source (one-off but revealing).** BriefBuilder counted GWS wins from matches_2026.csv (4) which has a known R10 gap, missing the R10 win (actual 5). Player performance files would have caught it.
**How to apply:** derived counts (wins, finals, tallies) must cross-check matches file AND player performance files; on disagreement mark `partial` and name the gap, never silently emit the lower number. R10 absent from matches_2026.csv is a known data gap. See [[project_finals_doc_stale_2026]] for another matches_2026 data-quality issue.

**DataSentinel arithmetic slip + SAMPLING weakness (gate-quality concern).** Same cycle, DataSentinel falsely flagged Essendon points-against 1188->1178; Scientist arbitrated 1188 correct. Its St Kilda re-gate tackle numbers (Sinclair 2.09, Impey 2.91) also did not match the CSV. An LLM doing arithmetic is a suggestion, not a gate (Threat 9, injectable).
On the StK-v-Haw marquee re-gate (2026-06-03), three consecutive DataSentinel(haiku) passes returned DIFFERENT defect sets on the SAME doc: Pass2 caught Hawthorn record 7-1-1->7-2-1 + Sinclair marks-l5 6.4->7.0; Pass3 caught Flanders marks-l5 5.0->7.0 + a FALSE-POSITIVE on St Kilda points-against (claimed 841, actual 843, Gaffer Python-confirmed doc correct). This proves DataSentinel SAMPLES cells, not exhaustive — it never flagged the worst errors.
**How to apply:** backlog item (now HIGH) — make DataSentinel numeric comparisons a deterministic Python sweep of EVERY computable [data] cell against the CSV, not an LLM sample. Reserve the LLM for interpretation only.

**4. Player form-table means widely wrong, beyond NaN (StK-v-Haw marquee, 2026-06-03).** Gaffer deterministic sweep of all 10 player form tables found ~15 cell mismatches DataSentinel never flagged. Two classes worse than NaN: (a) undisclosed game-exclusion — Wilkie last-5 disposals doc=23.6 but raw R8-R12 {25,22,5,25,21} has NO NaN and means 19.6; 23.6 only achievable by silently dropping the R10=5 partial game; (b) values HIGHER than any convention supports — Sicily tackles doc=3.5/3.4 but every denominator gives 2.6/2.8 (raw {2,4,2,2,4,1,1,5}). Windhager disposals doc=20.6 vs raw 22.5 (zero NaN). These are NOT convention artifacts and NOT Gaffer-fixable — denominator/exclusion rule is a Scientist call.
**How to apply:** the StK-v-Haw brief needs a full Scientist re-derivation of the per-player form tables under a PINNED convention, then a fresh exhaustive DataSentinel(deterministic) gate. Do NOT route single-cell fixes back to BriefBuilder one at a time — that whack-a-mole produced 3 failed gates this cycle. Escalated to human 2026-06-03.
