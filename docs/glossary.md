# Glossary - footy and data terms in plain English

> [← Back to main README](../README.md)

A one-stop reference for the AFL, SuperCoach, and data-science vocabulary used throughout this repo. Footy first, data second.

---

## Footy terms

### Disposals
The total of a player's **kicks** plus **handballs** in a match. The headline ball-use stat in modern AFL and the metric this repo's main model predicts. A midfielder averaging 25+ disposals is in good form; 30+ is elite.

### Kicks
Any kick of the ball, effective or not. Counted toward disposals.

### Handballs
A punched pass with a closed fist. Counted toward disposals.

### Marks
A clean catch from a kick travelling at least 15 metres. A high mark count usually means a player is finding space and being looked for as an outlet.

### Tackles
A physical pinning of an opponent in possession - if they're held long enough that they can't legally dispose, that's a tackle. Inside-mid and small-forward roles tend to lead the count.

> **Data note:** tackles are only recorded in this repo from **1987 onward**. Pre-1987 player files have zeros in the tackle column - they are unrecorded, not zero acts.

### Clearances
Winning the ball cleanly out of a stoppage (centre bounce, ball-up, or boundary throw-in). Clearances are the engine room of contested football.

> **Data note:** clearances and contested possessions are only recorded from **1998**.

### Inside 50s
Each time a team moves the ball from the midfield into their forward 50-metre arc. A team that gets to 60+ inside 50s is generally in the contest; under 40 usually means they're getting beaten territorially.

### Contested possessions
Picking the ball up under physical pressure, in a contest, or from a hard ball-up. The opposite is an "uncontested" possession - a free mark, a kick-in, an unpressured handball receive.

### Hit-outs
A ruck's tap from a stoppage to a teammate. A high hit-out count without clearances often means the ruck is winning the tap but the midfield is losing the follow-up.

> **Data note:** there is a recording-method change in **2017** that lifts the apparent hit-out totals. Treat 2017 as a methodology break, not a sudden league-wide skill jump.

### CBA (Centre Bounce Attendance)
Each centre bounce of the quarter has four onballers per team. CBA% is the share of a player's team centre bounces they're listed for - a midfield-time proxy. SuperCoach players watch CBA% closely because midfield time drives disposals.

### Free kicks for / against
Free kicks awarded to or against the player. Heavy "frees against" can flag a player who's giving up scoring chains on tackles or holding-the-ball calls.

### Brownlow votes
Three-two-one votes given by the umpires to the best players in each match. The Brownlow Medal goes to the home-and-away leader. We have a [Brownlow predictor doc](afl-brownlow-2026.md) that estimates current standings from a stats-based proxy model.

---

## SuperCoach terms

### SuperCoach
A fantasy football competition where you pick a squad of real AFL players. They score points each week based on their on-field stats (disposals, marks, tackles, goals, etc.). Your team's score is the sum of your selected players' real-world performances.

### Lockout
The deadline before each round when you can't change your team anymore. Tools like this repo are most useful in the days before lockout.

### Cash cow
A cheap, fast-scoring player whose price will jump quickly - you pick them to make money and trade them out before they crash.

### Premium
The expensive elite players you build your team around.

---

## Data-science terms

### MAE (Mean Absolute Error)
On average, how many disposals our prediction was off by. If MAE = 4.1, the typical prediction was within ±4 disposals of the actual number. **Lower is better.**

### RMSE (Root Mean Square Error)
Like MAE, but punishes large blunders harder. If you predict 30 and the player gets 10, RMSE notices that more than MAE does. **Lower is better.**

### Bias
Whether the model systematically over- or under-predicts. A bias of −0.7 means we tend to predict 0.7 disposals too high on average. **Closer to zero is better.**

### Calibration
Whether predicted numbers match real-world frequencies. A calibrated model that says "30 disposals" should produce roughly that. The current pipeline uses **post-hoc out-of-fold linear calibration** to correct top-end compression - the raw model under-predicts elite ball-winners, calibration fixes it.

### Backtest
Pretending we're back at an earlier point in time and asking the model to predict what came next - using only the data that would have been available then. The honest test of whether a model would have actually worked.

### Walk-forward backtest
A backtest where the model is retrained as time advances - round 5 is predicted using everything up to round 4, round 6 using everything up to round 5, and so on. This is the only valid approach for time-series sports data.

### Leakage
When information from the future (or from the test set) accidentally contaminates the training data. A model trained on leaked data looks great in evaluation and falls apart in production. Avoiding leakage is a non-negotiable in this repo.

### GroupKFold
A cross-validation scheme that ensures the same group (in our case, the same player) doesn't appear in both training and validation. Without it, the model could "learn" a specific player's typical output and look more accurate than it really is - a player-level leak.

### Ensemble
A combination of multiple models (here: LightGBM + HistGradientBoosting + RandomForest) whose predictions are averaged or stacked. Ensembles tend to outperform any single model because they cancel out each model's individual quirks.

### LightGBM / HistGradientBoosting / RandomForest
Three different tree-based machine-learning algorithms. They all work by combining decisions from many simple trees but with different tricks to control overfitting. The ensemble blends them.

### Within-cohort z-score
For era-fair historical ranking: how many standard deviations a player's stat is above (or below) the average of their own season + position cohort. Lets you compare a 1985 ruckman to a 2015 ruckman without the era-difference distortion.

### Hit / miss (model report card)
Pre-registered definitions used in our weekly accuracy report. A "hit" is a prediction within ±5 disposals of actual; a "miss" is more than ±10. See [the model report card](model-report-card.md) for the full methodology.

---

## Pipeline / repo terms

### Refresh
Running `refresh_and_rank.sh` - downloads the latest match and player data, retrains the model, runs the backtest, and updates the auto-generated docs.

### Auto-update
Sections of docs marked `<!-- ...-START -->...<!-- ...-END -->` are rewritten by the refresh pipeline. Don't edit between those markers by hand - your changes will be wiped on the next refresh.

### Scientist agent
The Claude-Code-driven analysis assistant configured in this repo with strict methodology rules (no leakage, baselines first, reproducibility). See [scientist-agent.md](scientist-agent.md).

---

## Related

- [How predictions work](prediction-model.md) - the model in technical depth
- [Backtest results](afl-backtest-2026.md) - current accuracy numbers
- [Model report card](model-report-card.md) - hit/miss methodology
- [How it works: data science deep-dive](data-science.md) - full technical reference
