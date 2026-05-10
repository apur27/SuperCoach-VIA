# Model report card

> [← Back to main README](../README.md) | [← Back to fan landing page](start-here-no-code.md)

A pre-registered, in-public accuracy log for the disposal prediction model. The point of this page is to commit to a small set of metrics, define what counts as a hit and a miss **before** seeing each round's results, and report every round - good or bad.

This is not a marketing page. If the model has a bad week, that round is in the table.

---

## Methodology - pre-registered

The metrics, definitions, and the commitment to report-every-round are fixed in this section. Changing them retroactively to flatter the model would defeat the point. If we ever change a definition, the change is dated, justified, and the previous version stays visible above it.

### The metrics

| Metric | Definition | Lower / higher = better |
|---|---|---|
| **MAE** (Mean Absolute Error) | Mean of the absolute differences between predicted and actual disposals across all players in the round. | Lower |
| **% within 5 disposals** | Share of predictions where `abs(predicted - actual) <= 5`. The headline accuracy number for fans. | Higher |
| **% within 10 disposals** | Share of predictions where `abs(predicted - actual) <= 10`. The "obvious blunder" rate is `1 - this`. | Higher |
| **Bias** | Mean signed error: `mean(predicted - actual)`. Tells us whether we are systematically over- or under-predicting. | Closer to zero |
| **n** | Number of players the model predicted in this round, after late-out filtering. | Higher = more coverage |

These are fixed. We do not invent a new metric mid-season because the existing ones look bad.

### Hit / miss definitions

For the qualitative report card:

- **Hit** - prediction was within ±5 disposals of the actual value. The model "got it right" for the average fan's expectation.
- **Near miss** - between 5 and 10 disposals off. Wrong, but the player was not a wildcard.
- **Miss** - more than 10 disposals off. The model had no business being this far off; round-level investigation justified.

A round is good if % within 5 ≥ 65% **and** there are no more than five outright misses. A round is concerning if either threshold is broken.

### What we commit to reporting

- **Every round** - the metrics table below is updated regardless of result.
- **No cherry-picked windows** - we do not start the table in the middle of a hot streak and ignore the bad rounds before it.
- **No retroactive metric changes** - if we add a metric mid-season, the row for previous rounds gets a `-` and we say so.
- **The five biggest misses each round** - players where the model was most wrong, with the model's likely explanation (role change / tag / late out / weather / pure noise).
- **Cumulative numbers** - season-to-date averages so the table cannot be read as just a slice.

### What we do not promise

- That the model will improve every round - it will plateau and dip; AFL is noisy.
- That we will explain every miss - sometimes a player just had a weird game.
- That the report card will catch every methodology error - it is one layer of accountability, not a full audit. The [backtest doc](afl-backtest-2026.md) is the formal evaluation.

### When does this kick in

The pre-registration above takes effect for rounds **from this commit onward**. Earlier rounds are reproduced from the existing [backtest doc](afl-backtest-2026.md) in the section below for context, but they were evaluated with the same metrics so the comparison is fair.

---

## 2026 season - rounds reported so far

The numbers below are pulled from the 2026 walk-forward backtest in [afl-backtest-2026.md](afl-backtest-2026.md). The backtest doc is the canonical source - this page is the fan-facing summary with the methodology pinned at the top.

### Round-by-round

| Round | Players (n) | MAE (disposals) | RMSE | % within 5 | % within 10 | Notes |
|------:|------------:|----------------:|-----:|-----------:|------------:|-------|
| 1 | 230 | 4.89 | 6.17 | 58.7% | 89.6% | Round 1 has too little 2026 history per player to be fair - included for completeness |
| 2 | 413 | 4.14 | 5.15 | 68.0% | 94.2% | First useful round once each player has at least one 2026 game |
| 3 | 320 | 4.07 | 5.28 | 69.7% | 94.7% | |
| 4 | 319 | 4.15 | 5.31 | 69.9% | 94.0% | |
| 5 | 365 | 3.74 | 4.74 | 70.4% | 97.3% | Best round so far on MAE and within 10 |
| 6 | 411 | 3.98 | 5.06 | 71.3% | 95.1% | Best round so far on % within 5 |
| 7 | 410 | 4.04 | 5.14 | 68.5% | 94.6% | |
| 8 | 411 | 4.13 | 5.25 | 67.9% | 94.4% | |

### Cumulative (mean across 8 rounds reported in the backtest)

| Metric | Value |
|---|---|
| MAE | **4.14 disposals** |
| % within 5 disposals | **68.1%** |
| % within 10 disposals | **94.2%** |

### How to read these numbers

- **MAE 4.14** - on a typical prediction, we are within ±4 disposals of the actual figure.
- **68% within 5** - just over two-thirds of predictions land within five of the real number. For SuperCoach decision-making, that is the most intuitive accuracy bound.
- **94% within 10** - "obvious blunders" (10+ disposals off) happen on about 6% of predictions. Some are unavoidable (late role changes, weather games, hard tags); some are signal that the model has not figured out a specific player yet.

Rule-of-thumb context: an MAE of 4-5 disposals is competitive for AFL prediction. The game has too many random events (injuries, umpire decisions, tactical changes) for any model to do much better. % within 5 above 65% is good; above 70% is strong.

---

## Where the model is consistently off

The [backtest results page](afl-backtest-2026.md) bolds the players where the model has been most consistently wrong, with the average error. That is the watchlist of "model has not figured this player out yet" - usually a role change, a tag, or a player whose 2026 form is genuinely different from their pre-2026 baseline.

If you see a player on that list in your team or on your watchlist, weight your eye over the model.

---

## Why this page exists

Public accuracy reporting is the cheapest form of model accountability. If the model is good, the report card shows it. If the model is having a bad month, the report card shows that too - and the operator (and the fans) can ask why before any decisions get made on a bad assumption.

The alternative - reporting only when the model wins - is what every betting tipster does, and the average tipster is not statistically significant.

---

## Related

- [Backtest results (full detail)](afl-backtest-2026.md)
- [How to use this for SuperCoach](how-to-use-this-for-supercoach.md)
- [Glossary](glossary.md)
- [Data science deep-dive](data-science.md) - the model and the backtest framework
- [Prediction model overview](prediction-model.md)
