---
name: accountability-surface-audits
description: Auditing model-accountability pages (afl-backtest-2026.md): apply the doc's own stated thresholds to its own table, and demand independent attestation for publication-order leak-proofing claims
metadata:
  type: feedback
---

On model-accountability surfaces (`docs/afl-backtest-2026.md` and kin), three checks catch
what DataSentinel structurally cannot, because every individual number is true:

**1. Apply the doc's own pre-registered thresholds to the doc's own published table.**
A page that defines "a round is *good* if X and *concerning* if Y" almost never re-reads its
own table against that rule. On the 2026 backtest page, the criterion "no more than five
outright misses (errors > 10 disposals)" was failed by **20 of 20 rounds** (derivable directly
from the published within-10 column: `n × (100 − within10)/100`, min 6.0 at R9, ~302 season-wide),
and the page labelled zero rounds concerning. Compute every stated threshold from the published
columns before reading the prose.

**Why:** a self-authored standard that is never applied is worse than no standard — it reads as
passed. This is the accountability-surface analogue of the superlative/table contradiction class
in [[superlative-and-jersey-collision]].

**How to apply:** grep the doc for `if`/`≥`/`threshold`/`good`/`concerning`/`worth investigating`,
then arithmetically test each against the doc's own tables. Also check legend-to-emphasis
consistency (e.g. "players with errors above ±6 (bolded)" — 8 rows bolded, only 1 qualified).

**2. Publication-order leak-proofing needs an attestation the author does not control.**
When a round is scored from an archived *forward* prediction CSV with no retrain, the stated
guarantee ("a prediction written before the game cannot have seen it") rests entirely on *when
the file was written*. A filename timestamp and an mtime are both writer-controlled. The durable
attestation is `git log -1 --format=%aI -- <file>` compared against the round's first bounce in
`data/matches/matches_<year>.csv` (`round_num`, `date`). Check **every** archive-path round —
in the R18–R20 case, R18 and R19 were git-attested pre-bounce but R20 was **not committed at all**,
so the guarantee was unconditional in prose and unattested in fact for one round in three.

**How to apply:** enumerate archive rounds from
`grep -l 'no retrain' data/prediction/backtest/backtest_run_*.log`, then attest each.

**3. Aggregate-level "unbiased" claims mask monotone subgroup bias.** Pooled bias ≈ 0 is
compatible with every single top-30 player being under-predicted (all 30 negative, to −6.3).
Read the interpretive gloss against the disaggregated table, not just the headline.

**4. Verify the training-window sentence against the trainer, not the cutoff class.** The
2026 backtest page claimed twice ("trained on all data **before** that round", "Train the model
on every game played strictly before round R") that each round retrains on prior in-season data.
`supercoach/prediction.py:589` is `historical_data = df[df['year'] < self.target_year]` — training
excludes the **entire** target year, so zero 2026 rows are ever training targets; in-season data
enters only as lagged features for the prediction slice. The `LeakProofPredictor` cutoff in
`backtest.py:159` is belt-and-braces on top of that, which is why reading only the cutoff class
makes the doc's sentence look true. **How to apply:** always open
`prepare_features_and_target` (or the trainer) and read the training-set filter literally; the
leak-proofing class is not the training window.
Corollary: `backtest.py`'s own module docstring and the doc both say the cutoff drops rows
"at or after" the target round; the code drops **strictly after** and deliberately keeps the
cutoff round ("Keep the cutoff round itself so we have a row to predict against"). No actual
leak — training is year-filtered and all features are `.shift(1)` lagged — but the described
mechanism is wrong, so don't infer a leak from the mismatch without tracing the trainer.

**5. Glossary sign conventions invert silently.** A plain-English metric table is prose, so
DataSentinel never touches it. On the backtest page the Bias row read "a bias of −0.7 means we
tend to predict 0.7 disposals too high" while three other locations (the cumulative table, the
team-bias intro, and the pre-registered definition `mean(predicted − actual)`) all said negative
= too **low**. One inverted glossary row makes an 18-row table read backwards. **How to apply:**
for every signed metric, check the glossary row against the formal definition AND against each
narrative gloss; they drift independently.

**6. Cheap independent reconciliation beats tag-by-tag spot-checking on this page.** The
keep-last vintage map is reconstructible in ~10 lines (merge every `backtest_summary_*.csv`
oldest-first, dedupe `(year, round)` keep-last, load the named `prediction_vs_actual_*` files,
drop null `actual_disposals`). Doing so reproduces n, MAE, RMSE, bias, within-5/10, the team
table, the top-30 signs and the outright-miss count in one pass — which both discharges the
Sentinel smoke test and independently tests every *new* number a fix introduced. Do this before
reading the prose; it costs one Bash call and converts most numeric doubt into fact.
Watch the last decimal: hand-authored prose outside the generated marker blocks truncates
(−0.583548 published as −0.583, correctly −0.584) while generated blocks round.

**7. Irreducibility claims are the load-bearing half of a self-set benchmark.** "An MAE of 4–5
is competitive" is a weak self-graded bar; "the game has too many random events **for any model
to do much better**" is an unfalsifiable universal with no citation that retroactively converts
the bar into a ceiling. When auditing a self-set benchmark, quote the *scope* clause, not the
threshold — that is where the unfalsifiability lives.

**Coach-anonymity note:** `config/coach_names.txt` matches on surname, so player names collide
(Blake Hardwick, Shai Bolton). The file's own scope note settles it — "Player names are NOT coach
names and are always allowed" — and auto-generated data tables are not council structural prose.
Clear these explicitly rather than blocking; say so in the report so the next reviewer does not
re-litigate.
