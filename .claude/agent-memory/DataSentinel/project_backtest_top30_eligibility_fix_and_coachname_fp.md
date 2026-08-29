---
name: backtest-top30-eligibility-fix-and-coachname-fp
description: 2026-08-05 R22 Pass-2 - confirms the per-round-eligibility fix to _load_top30_player_deviation closed the stale-top30-table defect, and documents a coach-name-scan false-positive pattern (player surnames colliding with coach surnames).
metadata:
  type: project
---

## Top-30 table staleness (from [[project_backtest_partial_regen_stale_blocks]]) — CONFIRMED FIXED 2026-08-05

Root cause was: `_load_top30_player_deviation` filtered to backtest vintages a cycle had
marked complete (BL-02), but `mark` only runs after a successful push, while this doc is
generated BEFORE it — so the round being scored that same cycle could never be eligible
yet, and the table silently lagged by one round every cycle.

Fix verified: eligibility is now decided PER ROUND — prefer marked vintages, but if a
round has no marked vintage at all, fall back to its newest. Re-verified end-to-end at
content hash `9f28ec05…`: top-30 table, CUMULATIVE, TEAMBIAS, MISSES, TRAINCORPUS and
VINTAGEPATH all reproduce exactly on the SAME R1–R22 basis (n=7,898). The "Read:" callout
under the CUMULATIVE table now carries its OWN two `**[data]**` tags (one for the
top-30-mean-error figure, one for the restated pooled bias) rather than silently reusing
the tag from the table above it — this also closes the untagged-restatement trap from
[[feedback_methodology_paragraph_untagged_restatement]] for this specific callout.

Still true per [[project_backtest_partial_regen_stale_blocks]]: which block goes stale
moves cycle to cycle. Don't assume this specific fix means top-30 is now permanently safe
from staleness in general — a DIFFERENT generator could still desync next cycle. Keep
recomputing every block from the live keep-last artifact set every pass.

## Coach-name scan false positive: player surnames that collide with coach surnames

`config/coach_names.txt` entries `Hardwick` and `Bolton` (single-surname form) will
substring-match current AFL PLAYERS who happen to share that surname — e.g. **Blake
Hardwick** (St Kilda/Hawthorn defender) and **Shai Bolton** (Richmond/West Coast
forward), both of whom appear routinely in the backtest misses table. Per the config
file's own scope note ("Player names are NOT coach names and are always allowed") and
[[coach_anonymity_lint]] (FootyStrategy memory), these are NOT violations — verify by
checking whether the matched full name is a current AFL player (data/player_data/ has a
file) before flagging. Don't auto-flag on a bare surname substring hit; confirm the
surrounding context names a *coach* being discussed as an authority/tactics source, not a
player stat entry.

See also [[project_backtest_partial_regen_stale_blocks]],
[[project_backtest_traincorpus_vintagepath_blocks]].

## 2026-08-11 (R23 landing, hash `28686c6f…`) — clean full regen, zero defects; rounding-tie floating-point note

All blocks (2026-BACKTEST per-round table, top-30, CUMULATIVE, TEAMBIAS, MISSES 23×10=230
entries, TRAINCORPUS, VINTAGEPATH incl. attestation) reproduced exactly against the live
R1–R23 keep-last pool (n=8,263; MAE 3.962; RMSE 5.101; bias −0.095; three-way reconciliation
8,263 == 8,263 == 8,263) with zero staleness — the per-round-eligibility fix from the R22
pass continues to hold, and no block lagged behind another this cycle. All 23 `**[data]**`
tags PASS.

**New note — exact-half rounding ties are a floating-point artifact, not a generator
inconsistency.** A few top-30 cells landed exactly on a `.x5` boundary in the raw
unrounded mean (e.g. Nick Daicos avg_actual raw=35.25 → doc shows 35.2; Noah Anderson
avg_error raw=−1.85 → doc shows −1.9; Marcus Bontempelli avg_predicted raw=25.05 → doc
shows 25.1; Jake Bowey avg_predicted raw≈23.25-ish → doc shows 23.2). Neither Python's
`round()` (banker's/round-half-even) nor decimal `ROUND_HALF_UP` reproduces the doc's
choice consistently across all four — because these raw means are sums of integers
divided by non-power-of-2 game counts, the underlying float is never exactly `x.x5`, it's
`x.24999999...` or `x.25000000...1` depending on summation order, so the "correct"
rounding is whatever IEEE-754 order the generator's pandas call happened to produce, not
a discoverable rounding-mode rule. **Do not try to reverse-engineer which convention the
generator uses at these boundary cells** — verify via the existing ±1-last-decimal
tolerance band instead (per the Edge Cases spec: "Display value within 1 in the last
shown decimal ... is a match"), which every one of these cases satisfies (diff ≤ 0.1).
This is the same root cause as the previously-documented "average of rounded vs raw"
finding in [[project_backtest_reproduction_recipes]] item 1, now seen at the individual-cell
level rather than only in the aggregate mean-of-avg-error tag.

## 2026-08-18 (R24 landing, hash `3d0d136e…`) — clean full regen; EXACT top-30 tie needs the real function, not a guessed secondary sort key

All blocks (2026-BACKTEST per-round table, top-30, CUMULATIVE, TEAMBIAS n=8,635==8,635==8,635,
MISSES 24×10=240 entries, TRAINCORPUS, VINTAGEPATH incl. attestation, frozen R1–R20 Known
Coverage Limitation numbers) reproduced exactly against the live R1–R24 keep-last pool. All 23
`**[data]**` tags PASS, zero untagged numbers, zero coach-name violations (Hardwick/Bolton again
false positives — same players as before).

**New this pass — a genuine (not rounding-boundary) exact tie in the top-30 table.** Clayton
Oliver and Harry Sheezel both landed on `avg_actual = 30.454545...` to full float precision
(335/22 for both — not a `.x5` display-rounding artifact like the R23 note above). Doc ranks
Harry Sheezel above Clayton Oliver. Guessing a secondary sort key (avg_predicted desc? avg_error
desc? alphabetical?) from the displayed table alone is unreliable — instead, `sys.path.insert`
the repo root and directly call `update_team_analysis._load_top30_player_deviation(2026,
'data/prediction/backtest')`, which reproduced the doc's exact row order (including this tie)
and every value/bold-flag to the displayed decimal on the first try. The underlying tie-break
is just pandas' non-stable default `sort_values(..., ascending=False)` (quicksort) applied to
whatever order `groupby(["player","team"])` (default `sort=True`, i.e. alphabetical on the raw
`Surname Firstname` CSV field) produced — not a deliberately-chosen secondary key. **Lesson:
when a top-30 (or similar generated-table) reproduction hits an exact tie, import and call the
actual generator function rather than trying to reverse-engineer a tie-break rule from table
order** — it's faster and immune to guessing wrong.
