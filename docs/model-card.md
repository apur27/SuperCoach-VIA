# Model card: disposal forecast (`features_v1`, champion `lgbm`)

*Engineering document for the rewrite's forecasting package (spec: [rewrite/PLAN.md](rewrite/PLAN.md) §7). The numbers here are model-evaluation metrics, not player statistics. Each one is read from the artifacts named below, which a rerun of the same command reproduces.*

## What it predicts

The model predicts **disposals** (kicks + handballs) for one player in one specific future match. It does not predict SuperCoach points, and it does not give selection or injury probabilities. Every forecast row carries `player_id`, `match_id`, `forecast_cutoff`, `origin` (`prospective`, `replay` or `legacy_unknown`), the model or baseline identity and `history_games`. If the source has no valid future fixture, the output is `forecast_status=unavailable`. No "next round" is ever manufactured.

## Artifacts this card describes

| Item | Value |
|---|---|
| Dataset snapshot | `sha256:55e295f184c49096c4c8c783e6d7d59f416ca70e108ce0ca7f186f16157fb09c`: checked-in corpus plus the B1 repair evidence |
| Model bundle | `bundle-ac063296460d42d54d9c` (`var/real/models/`; manifest sha256 `f8769368…`) |
| Replay evaluation | `eval-72a34903a1427939cfa3` (2026 season, origin `replay`) |
| Command | `scvia forecast --train-cutoff 2025-01-01 --calibration-end 2026-01-01 --cutoff 2026-09-25T10:00:00+00:00 --replay-season 2026` |
| Environment | CPU only, 4 logical CPUs (Xeon 2.80 GHz), Python 3.12, scikit-learn 1.9.1, LightGBM 4.7.0, seed 20260925 |

## Data and features

- Training population: player-games from 2010 onward with a verified date, joined to a complete, dated match and a non-null disposal label. Blocks: 134,971 train rows before 2025-01-01, 9,907 calibration rows (2025), and 9,971 holdout rows (2026).
- Features (`features_v1`) are built by one engine for both training and serving. For six base stats they are: prior-5 mean, within-season prior-3 mean, season-to-date mean and EWM(span 3). The rest are prior time on ground, days since the last game, career counter, age where the DOB is known, history and stage context, plus club, opponent and venue categories. Missingness indicators are derived before imputation. Encoders and imputers are fitted inside each training fold.
- Leakage controls: only games whose match day precedes the target's cutoff are used. Ambiguous same-day games are excluded. Tests mutate future outcomes and assert the features do not change (M02), and assert that replay and prospective feature construction agree (M05).
- Known limitation: legacy rows have no archived `available_at`. Replay therefore assumes each game was available the following UTC day, and it cannot prove what an operator knew at the time.

## Selection and promotion

Four candidates were compared on four expanding chronological folds (validation starting 2016-04-09): the prior-5 mean baseline, a cohort baseline, HistGradientBoosting and LightGBM. Out-of-fold MAE was 4.005, 4.076, 3.892 and 3.889 respectively. The selection used out-of-fold MAE only; the holdout was not used. Recalibration choice: identity (OOF MAE 3.899) over linear (3.908).

Champion gate: the candidate's MAE must beat the prior-5 baseline by at least 1% on the matched holdout, and no cohort with at least 100 outcomes may be more than 5% worse. Result: **passed**. The champion's holdout MAE was 3.753 against 3.900 for the baseline, a 3.79% improvement on 9,971 matched rows, and no cohort failed.

## Holdout results (2026, matched rows)

| Metric | Champion `lgbm` | Baseline prior-5 |
|---|---:|---:|
| Rows | 9,971 | 9,971 |
| MAE | 3.753 | 3.900 |
| RMSE | 4.837 | 5.037 |
| Bias (prediction − actual) | −0.214 | −0.114 |
| Median absolute error | 3.051 | 3.200 |
| Within 5 | 72.1% | 71.6% |
| Within 10 | 95.8% | 95.2% |
| Mean of round MAEs (separately labelled) | 3.742 | 3.905 |

The 2026 replay evaluation scores the same bundle over all 30 stages. It covers 9,811 scored rows: 22,259 intended candidates, of which 17,759 were predicted and 9,826 joined to an actual outcome. 15 rows were excluded: 11 with an unknown actual and 4 with a club mismatch. Its pooled MAE is 3.755 against 3.899 for the baseline. The six finals stages have 92 rows each, below the 100-outcome threshold, so they are reported descriptively only.

## Uncertainty interval

The interval is an 80% split-conformal interval built from absolute residuals. Its quantile comes from the 9,907-row 2025 calibration block, which sits after training and before the holdout, using the finite-sample rank `ceil((n+1)·0.8)`. The lower bound is clipped at zero. On the holdout it covered **81.1%** with a median width of 12.3 disposals, inside the 75–85% acceptance band, so it is labelled calibrated. Temporal drift can break the exchangeability assumption, so the band is re-checked on every training run.

## Intended use and limits

- The forecasts are research estimates of disposal counts. They are not betting advice and not selection predictions. Without a verified, timestamped team announcement, `selection_status` stays `unconfirmed`.
- Players with no prior game get the named `cold_start_prior_v1` value, not an individual model prediction.
- Prospective accuracy starts accruing only from archived forecasts made before each match. Replay metrics are a retrospective reconstruction. Legacy forecast archives stay `legacy_unknown` and are never pooled with either.
- Training time on the reference box was about 23 minutes, almost all of it out-of-fold fitting. It ran alongside another heavy job, so this is an upper bound. A refresh reuses a cached bundle whose cache key (snapshot, cutoff, features, folds, versions, seed) is unchanged, and it never retunes implicitly.
