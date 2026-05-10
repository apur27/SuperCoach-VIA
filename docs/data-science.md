# How it works: data science deep-dive

> [← Back to main README](../README.md)

A technical reference for the modelling, validation, and ranking work in this repository. It is written in three layers: every section opens with a plain-English summary, then a methodology paragraph, then a practitioner-level paragraph for ML readers who want the full detail.

If you only have 30 seconds, read the **In plain English** lines and skip the rest. If you are a data scientist, the **The methodology** paragraphs cover the design choices. If you are reviewing this as an ML practitioner or leader, the **For the ML practitioner** paragraphs hold the rigour.

---

## Contents

1. [The dataset](#1-the-dataset)
2. [The prediction model](#2-the-prediction-model)
3. [The backtest framework](#3-the-backtest-framework)
4. [The era-normalised all-time ranking](#4-the-era-normalised-all-time-ranking)
5. [AI and LLM integration (the Scientist agent)](#5-ai-and-llm-integration-the-scientist-agent)
6. [Current accuracy](#6-current-accuracy)
7. [Roadmap - future improvements](#7-roadmap--future-improvements)

---

## 1. The dataset

**In plain English:** The repo holds a per-game record for every player who has played senior VFL/AFL, going back to 1897. Each row is a single game for a single player, with the stats recorded that day. Modern games carry a couple of dozen stat columns; old games sometimes only have goals and behinds.

**The methodology:** The dataset is split into two pieces. Match-level files (`data/matches/matches_<year>.csv`, one per year, 1897 → present) carry the fixture and result of every senior VFL/AFL match. Player-level files (`data/player_data/<slug>_performance_details.csv`, one per player) carry the per-game stat lines. There are roughly 13,300 player performance files and 130 yearly match files. Stat coverage is era-dependent: pre-1965 games typically only record `goals` and `behinds`; tackles appear from 1987; clearances and contested possessions from 1998; CBA percent and `percentage_time_played` are recent additions. Feature engineering treats this missingness explicitly rather than imputing it away - a player with no rolling tackle history simply gets no tackle-derived features rather than a synthetic zero.

**For the ML practitioner:** The CSV schema for performance files has 30 columns including `team`, `year`, `opponent`, `round`, `kicks`, `marks`, `handballs`, `disposals`, `goals`, `behinds`, `hit_outs`, `tackles`, `rebound_50s`, `inside_50s`, `clearances`, `clangers`, `frees_for`, `frees_against`, `brownlow_votes`, `contested_possessions`, `uncontested_possessions`, `contested_marks`, `marks_inside_50`, `one_percenters`, `bounces`, `goal_assists`, `percentage_time_played`, `cba_percent`, and `date`. Two features are computed during ingestion (in [`prediction.py:78-115`](../prediction.py)): `extract_dob_and_name` parses the player's name and date of birth from the filename slug `<lastname>_<firstname>_<DDMMYYYY>_performance_details.csv`, and `extract_round_number` normalises the `round` string into an integer for chronological sorting. The dtype contract (`DTYPES` in `prediction.py`) is explicit: counts as `Int16`, dates as `datetime64[ns]`, and player/team/round/opponent/venue as `category`. The full historical dataset also contains derived files under `data/top100/` (yearly top-100 rankings, one per year, plus the all-time aggregate) and `data/prediction/` (latest-round disposal forecasts and walk-forward backtest outputs).

---

## 2. The prediction model

**In plain English:** For each player who is going to play this weekend, the model produces a single number - how many disposals (kicks + handballs) we expect them to have. It learns from every senior AFL game ever played that has full stats (around 1990 onwards), looks at how each player has been going recently, who they're playing, and where, and outputs a forecast. The right way to use it: as a smart starting point, not a guarantee.

**The methodology:** The pipeline lives in [`prediction.py`](../prediction.py). Disposals is the prediction target because (a) it is the strongest single proxy for SuperCoach scoring among universally-recorded stats, (b) it is densely available across players and seasons, and (c) it is roughly normally-distributed with a manageable right tail, which keeps regression well-behaved. Features are entirely derived from the player's own past games - across-season rolling means (5-game window), within-season rolling means (3-game window), season-to-date expanding means, and an exponentially-weighted recent-form signal (span=3) - plus context features (`round_number`, `days_since_last_game`, `cba_percent`, `percentage_time_played`) and one-hot-encoded `venue` and `opponent` columns. Three model classes are trained and benchmarked: a `HistGradientBoostingRegressor` (sklearn, tuned via Optuna), a `LGBMRegressor` (LightGBM, GPU-accelerated when CUDA is available), and a `RandomForestRegressor` baseline. They are also combined into a `VotingRegressor` ensemble. The model with the lowest cross-validated MSE is selected, then a final post-hoc linear calibration is fitted on out-of-fold predictions to correct residual mean bias and top-end compression.

**For the ML practitioner:** The training protocol is configured in [`prediction.py:373-450`](../prediction.py). All four candidate models share a common `Pipeline` - `StandardScaler` → regressor - and are evaluated with `GroupKFold(n_splits=5)` grouped on `player`, which is the load-bearing leakage defence: the same player cannot appear in both the training and the validation fold of any split, so cross-validated scores are not inflated by memorising per-player tendencies. Hyperparameter tuning uses Optuna's TPE sampler (50 trials, fixed seed=42); the HGB search ranges over `max_depth ∈ [3,7]`, `learning_rate ∈ [0.01, 0.1]`, `l2_regularization ∈ [1e-6, 1.0]`, and `loss ∈ {poisson, quantile, squared_error}`. The LGBM search additionally tunes `num_leaves`, `subsample`, `colsample_bytree`, and `max_bin`. Model selection is by mean CV `neg_mean_squared_error`. Two known traps were mitigated explicitly in the current codebase, with the rationale in inline comments at [`prediction.py:383-394`](../prediction.py): (1) an earlier `np.log1p(target)` / `np.expm1(prediction)` round-trip caused severe top-end compression - max prediction was 28 versus max actual 43 - so the target is now raw `disposals`; (2) the LGBM objective was switched from `regression_l1` (predicts the median) to `regression` (predicts the mean), because L1 was systematically under-predicting 30+ disposal games by 7–15 disposals on a right-skewed target. After model selection, [`_fit_calibration`](../prediction.py) runs a final 5-fold GroupKFold pass with the chosen model, collects out-of-fold predictions, fits `actual ≈ a · pred + b` by closed-form OLS, and applies `(a, b)` at predict time. Slope is sanity-bounded to `(0.5, 2.0)` and intercept to `|b| < 20`; out-of-bounds values fall back to identity. Final predictions are clipped to `[1, 55]` for physical plausibility (lower bound = "played", upper bound = era max with a small margin). The CPU variant ([`prediction_cpu.py`](../prediction_cpu.py)) has the same structure but skips the CUDA `device='cuda'` parameter on LGBM and runs sequentially on the CPU; expect 10–30× slower training.

---

## 3. The backtest framework

**In plain English:** The model is graded by replaying past rounds as if it were predicting them at the time. For each round, the script wipes out everything that happened on or after that round, retrains on what was available, predicts that round, then scores the predictions against what actually happened. This is the single most important number - it tells you, honestly, how often the model is right.

**The methodology:** The backtest lives in [`backtest.py`](../backtest.py) and is **walk-forward**, not random k-fold. For every `(year, round)` pair in the requested window, a `LeakProofPredictor` (a subclass of the production predictor) drops every row from the dataset where `year > target_year` or `(year == target_year AND round > cutoff_round)`, then runs the full training and prediction pipeline against that truncated dataset. The cutoff is enforced both on the main `DataFrame` and on the per-player cache that the prediction loop iterates over, so feature engineering at predict time also sees only pre-cutoff data. Predictions are joined to actual disposals and scored with MAE, RMSE, median absolute error, bias, and the percentage of predictions within 5 and 10 disposals. The script also reports per-team and (when a position source is wired up) per-position slice metrics, plus the ten most over- and under-predicted players in each round.

**For the ML practitioner:** The single defensive choke point is [`LeakProofPredictor.load_and_prepare_data`](../backtest.py) - every other path through the predictor calls it, so leakage cannot be introduced by adding new features later as long as the cutoff filter remains in place. The metrics are computed in [`_round_metrics`](../backtest.py): `MAE = mean(|pred − actual|)`, `RMSE = sqrt(mean((pred − actual)²))`, `bias = mean(pred − actual)` (negative = model under-predicts on average), `pct_within_5 = mean(|pred − actual| ≤ 5) · 100`, and the special slice `top10_actual_mae` (MAE on the top-10 actual scorers in the round), which is the diagnostic for top-end compression - the systematic failure mode where the model under-predicts elite ball-winners. Round-level results are written to `data/prediction/backtest/backtest_summary_<ts>.csv`, prediction-vs-actual detail to `prediction_vs_actual_round_<N>_<year>_<ts>.csv`, and a structured log to `backtest_run_<ts>.log`. Two systematic biases the framework has surfaced: (a) **Round 1 MAE is always elevated** - there is no within-season form data, so the model leans almost entirely on prior-season features, which understates form change between seasons (Round 1 2026 MAE = 4.89 vs season-average 4.04); (b) **Top-end compression remains** - `top_10_mae` averages around 10.8 across the 2026 rounds while the season-wide MAE is 4.1, meaning the model is meaningfully more wrong on the players the model is most often used to evaluate. The post-hoc calibration step in `prediction.py` partially mitigates this by stretching the predicted range, but does not eliminate it. A position source and per-position calibration are open work; see the roadmap.

---

## 4. The era-normalised all-time ranking

**In plain English:** Comparing players across eras is hard because the game has changed - pre-1965 games barely recorded any stats, the rules have shifted multiple times, and a "good kick rate" in 1980 is not the same as a "good kick rate" in 2024. The all-time top-100 in this repo gets around it by ranking each player against their own peers in their own year, then aggregating those year-by-year ranks into a single career score. It is not the only defensible way to do it, but it is principled rather than arbitrary.

**The methodology:** The ranking pipeline lives in [`top_players_comprehensive.py`](../top_players_comprehensive.py). It runs in three stages. (1) **Per-year top-100s** are produced for every season since 1897, using a weighted sum of the stats that were actually recorded in that era (`goals` and `behinds` only pre-1965; goals + behinds + kicks + handballs from 1965; marks added from 1991; tackles, clearances, contested possessions and others from 2011). A single-stat cap (`SINGLE_STAT_CAP = 0.55`) prevents pure goal-kickers from running away in eras with only two or three stats. (2) **Within-cohort z-scores** are computed against position-stratified peer groups (the four-way classification is `key_forward / forward_mid / midfielder / backline`, derived from career goals-per-game), then capped at ±3 and shrunk by `sqrt(era_completeness)` so a pre-1965 z of +3 deflates to +1.9 - epistemic humility, not punishment. (3) **All-time aggregation** combines a player's rank within each year's top-100 with their z-score signal, weighted by era-completeness, and adds a season-count career bonus. The final score discounts still-active players by 5% to avoid over-rewarding peak careers that haven't completed their decline phase.

**For the ML practitioner:** The all-time score for player `p` is `mean_adj · (1 + career_bonus)` (with an active-player discount applied at the end). For each year `y` in which `p` appears in the yearly top-100, define `rank_score = ((101 − rank) / 100) ^ RANK_GAMMA` with `RANK_GAMMA = 0.37` - a concave mapping that compresses the gap between rank #1 and rank #25, allowing consistent top-20 finishers to accumulate comparable signal to dominant #1 finishers without forcing a quota. The within-cohort z-score is mapped to `[0, 1]` via `z_signal = (z + Z_CAP) / (2 · Z_CAP)`, capped at the bounds, and blended into `year_score = (1 − Z_BLEND) · rank_score + Z_BLEND · z_signal` with `Z_BLEND = 0.20`. The blend exists because raw rank rewards consistency (top-100 is top-100) but a pure-rank signal cannot separate a generationally dominant season (Martin 2017, z ≈ +2.88) from a merely top-30 season; the z signal does. `mean_adj = mean(top_11 adj_y values)` rewards sustained excellence - long careers get more candidate seasons but also dilute their average vs short-peak players. `career_bonus = 2.0 · min(seasons / 18, 1.0)` is **seasons-based, not games-based**: pre-1990 players had 16–18 game seasons, modern players play 22–24, so a games-based bonus systematically rewarded modern longevity. Eligibility is `career_games ≥ 150` (computed from the full per-game aggregate, not just top-100 seasons, which fixes a bug where injury-affected seasons were silently dropped). `ERA_COMPLETENESS` is calibrated to give a credible decade distribution in the top-30 without quotas; values are 0.84 (pre-1965), 0.92 (1965–1990, the anchor), 0.89 (1991–2010), and 0.89 (post-2010). Limitations: (a) pre-1965 incomplete data is the load-bearing weakness - no disposal/mark/tackle data means the algorithm is essentially ranking on goals and behinds alone, which favours forwards; (b) positional evolution is captured only crudely by the four-way classification; (c) the formula has no concept of finals performance, premiership impact, or peer recognition.

---

## 5. AI and LLM integration (the Scientist agent)

**In plain English:** Claude is not just answering questions about the data. It has access to the working directory and can write its own Python, run it, look at the output, and decide what to do next. When the weekly refresh runs, Claude regenerates charts, writes the team analysis section of the README from the latest stats, and commits the result. This is what "AI applied to sport" looks like when you let it actually do the work rather than just narrate it.

**The methodology:** The Scientist agent (configured in `.claude/agents/Scientist.md`) is a Claude Code subagent specialised for analytical work in this repo - its system prompt holds it to inspect-before-transform, baseline-first, leakage-aware practice, and a structured `Did / Found / Caveats / Didn't / Assumed` response contract. The agent reads the actual code (`prediction.py`, `backtest.py`, `top_players_comprehensive.py`) before writing any analysis. The weekly refresh script (`refresh_and_rank.sh`) calls the agent at step 5 to regenerate auto-updating sections of the documentation: 5-year team profiles, finals pathway analysis, Brownlow votes prediction, stat-leader summaries. Charts are produced by direct Python under the agent's control rather than embedded into the agent prompt. The pattern that emerges is "natural language as a thin wrapper over structured data" - the user asks a question, the agent reads the relevant CSV, runs the necessary aggregation, produces the answer with uncertainty quantified, and (where appropriate) commits the result.

**For the ML practitioner:** The agent is operating against ~13,300 player CSVs and ~130 match files; full-dataset scans are I/O-bound, so the agent caches per-player parsed DataFrames where possible (the same pattern is used inside `prediction.py` via `self._player_cache`). Memory is project-scoped at `.claude/agent-memory/Scientist/` and shared via version control - methodological gotchas (e.g. "tackles only recorded from 1987", "Brownlow proxy weights validated on 145k historical games", "no `position` column in the player data - per-position analysis requires a new data source") are persisted across sessions. The agent's prediction-pipeline contract is simple: it does not edit the model code without a Plan acknowledged by the user, but it is allowed to run `prediction.py`, `backtest.py`, and the analysis scripts directly, and to update the auto-generated documentation sections from their output. The agent has identified and fixed two production-relevant model issues in the current codebase: the log1p target compression and the LGBM L1 median bias; both fixes are documented in commit history and in the inline comments of [`prediction.py`](../prediction.py).

---

## 6. Current accuracy

**In plain English:** As of the latest backtest run (rounds 1–8 of the 2026 season, 2,879 player-round predictions), the model is on average off by about 4.1 disposals, beats within-5 disposals two-thirds of the time, and within-10 disposals nineteen times out of twenty. That is good, but the model is meaningfully worse on the elite ball-winners - exactly the players you most care about for SuperCoach.

**The methodology:** Numbers below are weighted by the number of player-round predictions each round and pulled from `data/prediction/backtest/backtest_summary_*.csv` (latest run timestamp `20260430_184619`):

| Metric | Value |
|---|---|
| Rounds backtested | 8 (Round 1–8, 2026) |
| Total player-round predictions scored | 2,879 |
| Mean Absolute Error (MAE), weighted | **4.11 disposals** |
| Root Mean Squared Error (RMSE), weighted | **5.21 disposals** |
| % within 5 disposals | **68.5%** |
| % within 10 disposals | **94.5%** |
| Mean bias (pred − actual), weighted | **−0.06 disposals** (well-calibrated globally) |
| Round 1 MAE | 4.89 (~20% above the rest-of-season average - no in-season form data) |
| Round 2–8 mean MAE | 4.04 |
| Top-10 actual scorers - mean per-round MAE | **~10.8 disposals** (top-end compression is the dominant residual error mode) |

**For the ML practitioner:** Context for "good": a naive baseline of "predict the player's across-season rolling-5 disposal mean" gives MAE in the high-5s on the same backtest window, so the model captures roughly 1.5 disposals of additional signal per prediction over the strongest non-model baseline. The mean bias near zero is the result of the post-hoc OOF calibration - without it, raw model output had a bias of around −1.3 disposals and a max prediction of ~28 vs max actual ~43. The top-10 MAE of ~10.8 is genuinely large and reflects an unsolved problem: on the players generating 35+ disposal performances, the model still under-predicts by 7–15 disposals on a meaningful fraction of games. Two structural causes: (a) the right tail of the disposals distribution is sparse, so trees with `min_samples_leaf ≥ 10` smooth elite outputs toward the cohort mean; (b) midfield-rotation and tagging signals - the things that actually drive 35+ vs 25 - are not in the feature set yet (no team-sheet data, no opposition-specific tagging history). Round 1 elevation is structural: the within-season rolling features are NaN for the first round and feed only across-season features, which the calibration cannot recover from. Per-team bias spreads from approximately +2.6 (Brisbane Lions, Round 2) to −3.1 (Collingwood, Round 1); these are not consistent across rounds, suggesting they are noise from small per-round samples (n=23 per team) rather than systematic team-level bias.

---

## 7. Roadmap - future improvements

**In plain English:** The current model is gradient-boosted trees trained per-player with rolling features. There are several genuine upgrades on the horizon - most aim to either close the top-end compression gap, give the user uncertainty intervals instead of point predictions, or bring in signals the model currently does not see (selected team-sheets, position-specific behaviour, opposition-specific patterns). Below is the technical ambition; this is what the model could plausibly become over the next twelve months.

**The methodology - high-impact upgrades, ordered by expected lift per unit of complexity:**

1. **Uncertainty quantification** - Move from point predictions to prediction intervals. The model should be reporting "Daicos: 32 ± 6 (80% interval)" rather than a single number. Two complementary candidates: **conformal prediction** via [`mapie`](https://mapie.readthedocs.io/) gives distribution-free intervals around the existing model with a one-line wrapper at fit time, and **quantile regression** via LightGBM's `objective='quantile'` (already in the HGB search space, just not currently selected) gives per-quantile predictions natively. Conformal first, because it does not require retraining the underlying model.

2. **Player injury and team-selection signals** - The single biggest unforced error in the current model is treating every player as if they will play. Once a team-sheet ingestion source is wired up, the model should consume `is_named`, `last_emergency`, `last_late_out`, and a team-rotation flag - these collectively explain a meaningful share of the top-end compression (a player named on-bench gets treated identically to one named centre, which is wrong).

3. **Position-aware modelling** - The dataset currently has no `position` column, so backtest per-position metrics report a single "Unknown" bucket. Wiring up a position source (either scraped or derived from career stat shape, similar to the four-way classification used in the all-time ranking) unlocks (a) per-position bias correction in the post-hoc calibration step, (b) position embeddings as model features, and (c) potentially a position-stratified ensemble where each position gets its own LGBM head sharing a common feature pipeline.

4. **Opposition modelling** - Today, `opponent` is just a one-hot column. Replacing it with a learned opponent embedding (or a lookup table of opponent-specific defensive strength against high-disposal players) should help the residuals on games where a tagger is deployed. The cleanest implementation is a small entity-embedding layer, trained jointly with the disposals head, which is where a deep model starts to earn its complexity.

5. **Transformer-based sequence models** - The current rolling features (5-game across-season, 3-game within-season, EWMA span-3) are fixed and hand-tuned. A **Temporal Fusion Transformer** ([`pytorch-forecasting`](https://pytorch-forecasting.readthedocs.io/), Lim et al. 2021) would replace the entire feature engineering layer with a learned attention over each player's full game history, with native support for static covariates (player), known-future covariates (opponent, venue, days_since_last_game), and observed past covariates (every stat). The honest expectation is that TFT will not beat tuned LightGBM on aggregate MAE (LightGBM is hard to dethrone on tabular per-game data) but will close the top-end gap by giving more weight to the most recent 2–3 games of context, which is where the elite-ball-winner form-spike signal lives.

6. **Hyperparameter search at scale** - Current Optuna search is 50 trials with TPE. Moving to [`optuna`](https://optuna.org/) ≥ 4.0 with a multivariate TPE sampler and ASHA pruning, plus an Optuna dashboard, would make the search itself reproducible and inspectable. Combined with a `polars`-backed feature pipeline ([`polars`](https://pola.rs/) is 5–20× faster than pandas on the rolling/groupby operations that dominate this codebase's feature step), the per-round backtest should drop from ~30 minutes to under 10, which is what the walk-forward 5-year backtest actually needs to be feasible.

**For the ML practitioner - concrete package candidates and what each unlocks:**

| Capability | Library | What it gives you |
|---|---|---|
| Conformal prediction intervals | `mapie >= 0.9` | Distribution-free `[lower, upper]` with marginal coverage guarantee; wraps any sklearn-compatible regressor |
| Quantile regression | `lightgbm >= 4.0` (already in repo) | Native `objective='quantile'` with per-fold quantile selection; no extra dep |
| Sequence / attention models | `pytorch-forecasting`, `torch >= 2.4` | TFT, N-BEATS, DeepAR - all support categorical embeddings out of the box |
| Hyperparameter search | `optuna >= 4.0` | Multivariate TPE, ASHA pruning, persistent study DB, dashboard |
| Faster feature engineering | `polars >= 1.0` | Native lazy execution; rolling/groupby are 5–20× faster than pandas equivalents |
| Position / opponent embeddings | `pytorch >= 2.4` (or `keras >= 3.0`) | Entity-embedding layers learned jointly with the regression head |
| Tracking & reproducibility | `mlflow >= 2.16` | Model registry, parameter/metric logging, artefact versioning across backtest runs |

The roadmap is intentionally ordered by **expected lift per unit of complexity**: uncertainty quantification is one library import, team-sheet integration is mostly a data-engineering problem, position-aware modelling is a moderate refactor, and the transformer step is the only one that meaningfully changes the training infrastructure. Each can be added without invalidating the layers below it; the load-bearing leak-proof backtest framework is model-agnostic and will continue to grade whatever model is plugged into it.

---

> [← Back to main README](../README.md)
