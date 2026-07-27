---
name: backtest-doc-verification-gotchas
description: Repo-mechanics traps hit verifying docs/afl-backtest-2026.md (backtest_summary/by_team/prediction_vs_actual keep-last dedup, from-csv scoring rounds, doc name-order)
metadata:
  type: project
---

Verifying `docs/afl-backtest-2026.md` (or any doc drawing on `data/prediction/backtest/`)
surfaces three repeatable traps:

1. **Player name order flips between CSV and doc.** `prediction_vs_actual_round_*.csv`
   stores `player` as `Surname Firstname` (e.g. `Daicos Nick`). The backtest doc's misses
   table and top-30 table render `Firstname Surname` (e.g. `Nick Daicos`). A literal
   string match will spuriously report every row as "NOT FOUND" — reverse word order
   (last two tokens swap; 3-token names like `Nasiah Wanganeen-Milera` keep the hyphenated
   surname as one token) before joining against the CSV.

2. **"Keep-last per (year, round)" must be computed from `backtest_summary_*.csv`
   filenames sorted by timestamp, not by file mtime.** Concat all summary CSVs, sort by
   source filename (which sorts chronologically because timestamps are `YYYYMMDD_HHMMSS`),
   `drop_duplicates(['year','round'], keep='last')`. Then look up the companion
   `prediction_vs_actual_round_<N>_2026_<ts>.csv` / `backtest_by_team_<ts>.csv` using the
   *same* timestamp the winning summary row came from — not the latest file on disk for
   that round. Confirmed exactly reproduces this doc's cumulative (n=7153, MAE 3.958,
   RMSE 5.094, bias -0.110), team-bias, and per-round tables to the displayed decimal.

3. **Not every backtest round retrains.** Some rounds are scored via a `--from-csv`
   path that re-scores an already-archived forward-prediction CSV instead of retraining
   with a leak-proof cutoff. Those runs' logs contain **no** `[cutoff y=2026 r=N] dropped
   X future rows` line at all (grep confirms zero hits) — the "the log line is the
   in-line audit trail" methodology claim does NOT hold universally. As of 2026-07-27,
   rounds 18, 19, and 20 (the three most recent, ~1,016 of 7,153 pooled predictions) were
   all scored this way. Check the run log's own `scoring archived prediction CSV ...
   (no retrain)` line before trusting a blanket "every round retrains with a leak-proof
   cutoff" claim in doc prose.

See also [[feedback_player_csv_not_chronological]] (same "don't trust file order" family
of trap, different artifact).

## 2026-07-27 update: methodology fix verified, dual-path audit-trail claim confirmed true

Re-verified after the doc's methodology section was rewritten to describe both scoring
paths explicitly. Confirmed by grepping every dedup-selected round's `backtest_run_<ts>.log`:
rounds 1–17 (retrain path) all contain `[cutoff y=2026 r=N] dropped X future rows`; rounds
18, 19, 20 (archive path, `--from-csv`) contain zero cutoff lines and instead log
`scoring archived prediction CSV ... (no retrain)` — exact string match to what the doc
now claims. This will drift as new rounds are added each week — always re-grep both
patterns per round rather than trusting last week's list of "which rounds are archive-path."

**Covering-tag placement convention confirmed for this doc**: a single `**[data]**` tag
per compound-claim sentence or per-table is valid here even when the tag sits *mid-sentence*
(immediately after the first value, e.g. "MAE 3.96 **[data]** disposals · 74.3%... · 95.7%...")
or even *before* the values it covers (e.g. "loads **[data]** 1,808 of 13,357 player files...
2005"). Treat one tag as covering every number in its sentence/table regardless of exact
position, matching the doc's established style — do not require a tag adjacent to every
individual number when the doc already uses this convention consistently. Full misses table
(20 rounds × 10 entries = 200 player-level claims) verified exactly against
`prediction_vs_actual_round_<N>_2026_<ts>.csv` for the keep-last vintage — format is
`Name (predicted→actual, error)`, NOT `(actual→predicted, ...)` — easy to invert by accident
when parsing.

## 2026-07-27 update: misses-table tie-break requires STABLE sort; "Read:" callout under cumulative table is an untagged-restatement trap

**Tie-break gotcha (verification-side, not a doc defect).** Reproducing the round-by-round
"5 biggest under-/over-predictions" misses table requires `df.sort_values('error',
kind='mergesort')` (or any stable sort) — NOT the pandas default `quicksort`. Many rounds have
3+ players tied on the exact same signed error (e.g. round 1 has two players tied at -14;
round 2 has seven players tied at -11). The doc's generator clearly preserves original
per-player-CSV row order as the tie-break. Using the default unstable quicksort reorders tied
rows unpredictably and produces false-looking mismatches against the doc (players present in
the data with the right values, but not the "5 biggest" the doc names) — do not mistake this
for a data-integrity problem; re-sort with `kind='mergesort'` before concluding a miss-list
entry is wrong.

**Untagged-restatement trap, confirmed again.** A `**Read:**` callout placed directly under a
tagged cumulative table (e.g. "every one of the top 30 disposal-winners is under-predicted, by
roughly 3 disposals on average against a population figure of −0.110") can be numerically
correct (verified independently against pooled per-player CSVs: 30/30 negative, mean ≈ -2.95,
range -6.3 to -0.2) and still be a FAIL, because the callout paragraph itself carries no
`**[data]**` tag of its own — restating "−0.110" from the table two lines above does not
inherit that table's tag, per [[feedback_methodology_paragraph_untagged_restatement]]. Every
new interpretive sentence needs its own tag even when it sits immediately beside an
already-tagged table it's summarizing.
