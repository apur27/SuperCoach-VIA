---
name: r18-2026-coverage-gap-accepted
description: R18 2026 scores only 284 of 412 player-rounds (14 of 18 clubs) in the canonical pool — a real sample hole, ACCEPTED 2026-07-26, do not re-run. Headline stable, team table is not.
metadata:
  type: project
---

**Round 18 2026 is under-covered in the canonical 2026 backtest pool. This is a known,
ACCEPTED limitation as of 2026-07-26. Do not re-score it. Do not "fix" it.**

### The gap

R18 2026 was played as 9 matches / 18 clubs. The canonical (keep-last) vintage
`prediction_vs_actual_round_18_2026_20260710_214217.csv` scores **284** player-rounds
across **14** clubs. Geelong, Melbourne, St Kilda and Western Bulldogs are absent
entirely. The earlier vintage `..._20260707_154033.csv` scored the full **412** across
18 clubs, so the season pool is short ~**128** player-rounds.

### Why

R18 was scored twice. The second pass used `--from-csv`, which re-scores an archived
*forward* prediction CSV instead of re-predicting. That archive was written before the
full R18 fixture existed, so four clubs were never in the input. The scorer was correct;
the input was short. **This is a genuine sample hole, not a vintage-merge artifact** —
distinct from the two selection traps in [[backtest-artifact-vintage-selection]].

### Why we accepted it (the numbers that justify the decision)

Verified 2026-07-26 by pooling both vintages directly. Headline barely moves; team
table moves a lot:

| metric | canonical (R18=20260710) | alt (R18=20260707) |
|---|---:|---:|
| n | 7,153 | 7,281 |
| MAE | 3.9575 | 3.9603 |
| bias | −0.1099 | −0.1048 |
| within-5 | 74.36% | 74.40% |
| within-10 | 95.78% | 95.78% |
| St Kilda bias | **−0.5835** | **−0.7330** |

Every headline figure is stable within its own rounding. St Kilda's club bias moves by
0.15 — material at club level. **How to apply:** if a question is about season-level
accuracy, the gap is immaterial and no caveat is needed beyond the doc note. If a
question is about Geelong / Melbourne / St Kilda / Western Bulldogs bias, say the club
figure rests on a round the pool does not cover.

**Why not fixed:** the round has already passed and the headline metric it feeds barely
moves — not worth the CPU or the risk of touching a completed round. Human decision,
2026-07-26. Re-raising it is re-litigating a closed call.

### Canonical convention this rests on

Canonical figures pool **one vintage per round, keep-last**: merge all
`backtest_summary_*.csv` oldest-first, dedup `(year, round)` keep=last, then load only
the detail CSV whose timestamp that map names. Never select by mtime. Under keep-last
R18 resolves to `20260710_214217`, which is *why* the gap is the published state — a
different convention would close it by accident, not by design.

### Durable repo note

Written up under `## Methodology` in `docs/afl-backtest-2026.md`, section
"Known coverage limitation — Round 18 2026 (accepted...)", outside the
`<!-- 2026-BACKTEST-START/END -->` auto-regenerated block so the weekly refresh cannot
clobber it. Check there first before re-deriving any of this.

Related: [[backtest-artifact-vintage-selection]], [[backtest-doc-verification]],
[[feedback-backtest-rules]].
