---
name: backtest-known-coverage-limitation-stale
description: docs/afl-backtest-2026.md "Known coverage limitation — Round 18" section (static prose, outside any auto-regen markers) goes stale the moment a new round lands, because only 2026-BACKTEST/CUMULATIVE/TEAMBIAS/MISSES blocks get regenerated — check it every pass, don't assume it's frozen-by-design.
metadata:
  type: project
---

## 2026-07-28 (R21 landing) — new stale-block instance, different location than the previously-known one

[[project_backtest_partial_regen_stale_blocks]] documented CUMULATIVE/TEAMBIAS/MISSES
blocks going stale on a partial regen. On this pass those three blocks (plus the
per-round table, top-30 table, and pre-registered-threshold sentence) were all
correctly refreshed to the full 21-round basis (n=7,524, MAE 3.959, RMSE 5.100, bias
-0.134 — all reproduced exactly). **But** the "### Known coverage limitation — Round 18
2026" section (static hand-written prose below `<!-- MISSES-END -->`, NOT inside any
`<!-- ...-START/END -->` marker pair — confirmed via grep) still carries its
**[data]**-tagged headline-comparison table and the St Kilda team-bias sentence computed
against the OLD 20-round basis:

- "Pooled figures as published: 7,153 / MAE 3.958 / RMSE 5.094 / Bias -0.110 / %within5
  74.36 / %within10 95.78" — reproduces EXACTLY if you pool rounds 1–20 only (excluding
  the newly-landed R21). Does NOT match the doc's own current CUMULATIVE block two
  sections above (7,524 / 3.959 / 5.100 / -0.134), which correctly includes R21.
- "St Kilda's season bias moves from −0.583 to −0.733" — also reproduces exactly under
  a rounds-1–20 team-bias pool (published vintage vs. fuller-R18-vintage), not the
  current 21-round TEAMBIAS block (which shows St Kilda at -0.65).

**Root cause**: this section is apparently maintained by hand (or a different, less
frequently-run generator) as documentation of a one-time decision ("Decision taken
2026-07-26: accept the gap, document it, do not re-run"). It reads in present tense
("Pooled figures **as published**") which is now factually false relative to the same
document's own regenerated blocks.

**Verdict**: treated as two FAILED `**[data]**` tags (the comparison-table tag and the
St Kilda sentence tag), not merely a note — the numbers are individually reproducible
against real files but contradict what this same page currently claims elsewhere,
which is exactly the internal-inconsistency failure mode DataSentinel exists to catch.
DataSentinel does not adjudicate whether the section *should* be frozen-by-design
(that's a Gaffer/Scientist call on whether to add an as-of marker or wire this section
into the regen pipeline) — only that, as written today with no as-of qualifier, it
disagrees with the rest of the doc.

**How to apply next time**: don't assume "Known coverage limitation" section numbers
track live — always recompute them fresh (rounds 1 through N-1 pool, i.e. excluding
the most-recently-landed round, plus the fuller-R18-vintage substitution) and diff
against both (a) what the section itself claims and (b) the CURRENT CUMULATIVE/TEAMBIAS
blocks elsewhere on the page. If B has moved past what this section's "as published"
column says, it's stale — flag it, regardless of whether A-vs-claimed-numbers reproduces
exactly (exact reproducibility against a stale scope is not the same as being correct).

See also [[project_backtest_partial_regen_stale_blocks]],
[[project_backtest_doc_verification_gotchas]], [[project_backtest_reproduction_recipes]].

## 2026-07-28 (R21 Pass-2, hash `28ee21d1…`) — RESOLVED via explicit "frozen as-of" framing

The prior FAIL was fixed correctly — not by re-pointing the section at live R21 data, but by
labelling it as what it always was: a dated, frozen record of the evidence a specific decision
("accept the Round-18 gap, do not re-run") was made on. The fix that landed:

- Table header now reads "Headline metric (**as at R1–R20, 2026-07-26**)" instead of a bare
  "Pooled figures as published" (present-tense, now-false framing).
- An italic note above the table states the block is deliberately NOT updated as new rounds
  land and explicitly redirects readers to the Cumulative summary above for current figures.
- The St Kilda sentence now reads "measured over **that same R1–R20 window**, St Kilda's
  season bias moved from −0.583 to −0.733" plus a parenthetical: "(St Kilda's *current* bias
  ... is in the team-bias table above and will not match those two figures.)"

Verified computationally: the R1–R20 pool (excluding R21, substituting Round 18's fuller
412-row/18-club vintage where the table says so) reproduces every cell exactly — 7,153/7,281,
3.958/3.960, −0.110/−0.105, 74.36%/74.40%, 95.78%/95.78%, and St Kilda −0.584→−0.733 (weighted
by team `n`, rounds to the doc's −0.58 published / −0.73 fuller). This now does NOT contradict
the doc's own live CUMULATIVE (7,524/3.959/5.100/−0.134) or TEAMBIAS (St Kilda −0.65) blocks,
because it no longer claims to represent current state — it explicitly time-stamps and scopes
itself and points to where current figures live.

**Verdict for future passes**: a section like this — numbers accurate for a stated historical
cutoff, explicitly labelled as such, with a live cross-reference and an explicit "will not
match" warning — is a PASS, not a contradiction. Don't flag frozen-and-labelled historical
snapshots as stale; only flag this pattern when the "as of" qualifier is missing, vague, or the
present tense implies current validity (that was the original 2026-07-27 FAIL). This is the
same resolution shape as the doc-level `<!-- verify-asof: round=N -->` mechanism, applied at
section granularity instead of whole-doc.

See also [[project_backtest_teambias_supersede_whole_file]] for the mechanics used to
reproduce the fuller-R18-vintage St Kilda figure.
