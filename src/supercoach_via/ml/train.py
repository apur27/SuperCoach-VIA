"""Training at an explicit cutoff, chronological model selection and promotion (PLAN 7.3).

Flow of ``train_model``:

1. Targets = completed, time-verified player-games with ``season >= target_seasons_from``
   and date before ``holdout_end``; each target's cutoff is 00:00Z on its match day.
   Features come from ``ml.features.build_features`` (the only feature code). Labels are
   joined separately; rows with an unknown actual are excluded with a count.
2. Rows are assigned to train / calibration / holdout blocks by date
   (``ml.splits``). Inner expanding folds live wholly inside the train block.
3. Candidates (prior-5 baseline, cohort baseline, HistGradientBoosting, LightGBM if
   installed, RandomForest if requested) are fitted per fold on fold-train rows only
   (preprocessing is inside each pipeline) -> out-of-fold MAE. The ML candidate with the
   best OOF MAE is *selected on OOF only*; identity-vs-linear recalibration is chosen by
   leave-one-fold-out OOF MAE. Optional bounded tuning also uses only inner folds.
4. Every candidate is refitted on the whole train block. The calibration block (later
   than all training rows, before the holdout) provides split-conformal residuals.
5. The holdout block is scored once for every candidate on identical rows. The champion
   gate compares the selected ML candidate with the prior-5 baseline; if it fails, the
   baseline is the shipped model and the report says so.

Cold starts: targets with zero eligible history get the named ``cold_start_prior_v1``
value (mean outcome of zero-history training targets) under every candidate.
"""

from __future__ import annotations

import hashlib
import os
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from supercoach_via.ml import models as M
from supercoach_via.ml.bundles import (
    BundleManifest,
    ModelBundle,
    cache_key,
    dependency_versions,
    json_safe,
    load_bundle,
    make_bundle_id,
    save_bundle,
)
from supercoach_via.ml.evaluate import (
    GateDecision,
    IntervalCalibration,
    apply_interval,
    champion_gate,
    cohort_metrics,
    conformal_quantile,
    evaluate_interval,
    mean_of_groups_mae,
    metric_block,
)
from supercoach_via.ml.features import FeatureFrame, FeatureSpec, History, build_features, historical_targets
from supercoach_via.ml.splits import Fold, SplitPlan, assign_blocks, plan_splits, season_folds

MAX_TUNE_TRIALS = 30
MAX_TUNE_SECONDS = 1200.0
TARGET = "disposals"
POPULATION = (
    "player_games with date_quality in spec.verified_date_qualities, joined to a complete, "
    "dated match; season >= target_seasons_from; date < holdout_end; label disposals not null"
)
GATE_COHORT_DIMS = {
    "club": "club_id",
    "stage_type": "stage_type",
    "history_band": "history_band",
    "predicted_volume_band": "volume_band",
}
REPORT_COHORT_DIMS = {
    **GATE_COHORT_DIMS,
    "season_stage": "season_stage",
    "season": "season",
}


@dataclass(frozen=True)
class TrainingConfig:
    train_cutoff: date
    calibration_end: date
    holdout_end: date | None = None
    target_seasons_from: int = 2010
    feature_spec: FeatureSpec = field(default_factory=FeatureSpec)
    candidates: tuple[str, ...] = ("hgb", "lgbm")
    include_rf: bool = False
    n_folds: int = 4
    validation_seasons: tuple[int, ...] | None = None
    seed: int = 20260925
    threads: int = 4
    tune: bool = False
    tune_trials: int = MAX_TUNE_TRIALS
    tune_seconds: float = MAX_TUNE_SECONDS
    interval_level: float = 0.8
    min_calibration: int = 200
    gate_min_rel_improvement: float = 0.01
    cohort_min_n: int = 100
    cohort_max_rel_worse: float = 0.05
    params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tune_trials > MAX_TUNE_TRIALS or self.tune_seconds > MAX_TUNE_SECONDS:
            raise ValueError(f"tuning is bounded to {MAX_TUNE_TRIALS} trials / {MAX_TUNE_SECONDS}s")
        if self.calibration_end < self.train_cutoff:
            raise ValueError("calibration_end precedes train_cutoff")

    def ml_candidates(self) -> tuple[str, ...]:
        names = [c for c in self.candidates if c != "lgbm" or M.lightgbm_available()]
        if self.include_rf and "rf" not in names:
            names.append("rf")
        return tuple(names)

    def as_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["feature_spec"] = {"fingerprint": self.feature_spec.fingerprint(), **asdict(self.feature_spec)}
        d["ml_candidates_effective"] = list(self.ml_candidates())
        return json_safe(d)  # type: ignore[no-any-return]


@dataclass
class Dataset:
    ff: FeatureFrame
    y: np.ndarray
    dates: np.ndarray
    blocks: np.ndarray
    plan: SplitPlan
    excluded: dict[str, int]


@dataclass
class TrainingResult:
    bundle: ModelBundle
    report: dict[str, Any]
    holdout_rows: pd.DataFrame
    reused: bool


def _code_hash() -> str:
    h = hashlib.sha256()
    here = Path(__file__).parent
    for name in ("features.py", "models.py", "train.py", "splits.py"):
        h.update((here / name).read_bytes())
    return h.hexdigest()


def cpu_info() -> dict[str, Any]:
    model = platform.processor() or ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu_model": model, "logical_cpus": os.cpu_count(), "python": platform.python_version(),
            "platform": platform.platform()}


def labels_for(history: History, keys: pd.DataFrame, target: str = TARGET) -> np.ndarray:
    pg = history.player_games[["match_id", "player_id", "club_id", target]]
    j = keys[["match_id", "player_id", "club_id"]].merge(
        pg, on=["match_id", "player_id", "club_id"], how="left", validate="one_to_one"
    )
    return j[target].to_numpy(dtype=float, na_value=np.nan)


def prepare_dataset(history: History, config: TrainingConfig) -> Dataset:
    spec = config.feature_spec
    t = historical_targets(history, spec=spec)
    n0 = len(t)
    t = t[t["season"] >= config.target_seasons_from]
    if config.holdout_end is not None:
        t = t[t["match_date"] < config.holdout_end]
    t = t.reset_index(drop=True)
    y = labels_for(history, t)
    known = ~np.isnan(y)
    excluded = {"outside_season_or_holdout_window": n0 - len(t), "unknown_actual": int((~known).sum())}
    t = t[known].reset_index(drop=True)
    y = y[known]
    ff = build_features(history, t, spec)
    dates = t["match_date"].to_numpy(dtype=object)
    train_mask = np.array([d < config.train_cutoff for d in dates], dtype=bool)
    folds: tuple[Fold, ...] | None = None
    if config.validation_seasons:
        folds = season_folds(dates[train_mask], config.validation_seasons)
    plan = plan_splits(
        dates, t["match_id"].to_numpy(dtype=object), train_cutoff=config.train_cutoff,
        calibration_end=config.calibration_end, holdout_end=config.holdout_end,
        n_folds=config.n_folds, folds=folds,
    )
    return Dataset(ff, y, dates, assign_blocks(dates, plan), plan, excluded)


def subset(ff: FeatureFrame, mask: np.ndarray) -> FeatureFrame:
    idx = np.flatnonzero(mask)
    return FeatureFrame(
        keys=ff.keys.iloc[idx].reset_index(drop=True),
        X=ff.X.iloc[idx].reset_index(drop=True),
        numeric_features=ff.numeric_features,
        categorical_features=ff.categorical_features,
        history_games=ff.history_games[idx],
        spec=ff.spec,
        snapshot_id=ff.snapshot_id,
        diagnostics=ff.diagnostics,
    )


def _fit_predictor(name: str, ff: FeatureFrame, y: np.ndarray, config: TrainingConfig,
                   params: dict[str, Any] | None, recal: M.Recalibration | None = None) -> M.FittedPredictor:
    cold = M.fit_cold_start(ff, y)
    fit_mask = ff.history_games >= 1
    cand = M.make_candidate(name, seed=config.seed, threads=config.threads, params=params)
    cand.fit(subset(ff, fit_mask), y[fit_mask])
    return M.FittedPredictor(cand, cold, recalibration=recal or M.Recalibration())


def _oof(
    name: str, ds: Dataset, config: TrainingConfig, params: dict[str, Any] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(final OOF predictions, raw candidate OOF predictions, fold id) over fold-valid rows."""
    n = len(ds.y)
    final = np.full(n, np.nan)
    raw = np.full(n, np.nan)
    fold_id = np.full(n, -1)
    in_train = ds.blocks == "train"
    for fold in ds.plan.folds:
        tr, va = fold.masks(ds.dates)
        tr &= in_train
        va &= in_train
        if not tr.any() or not va.any():
            continue
        pred = _fit_predictor(name, subset(ds.ff, tr), ds.y[tr], config, params)
        ffv = subset(ds.ff, va)
        p, _ = pred.predict(ffv)
        final[va] = p
        raw[va] = pred.candidate.predict(ffv)
        fold_id[va] = fold.index
    return final, raw, fold_id


def _tune(name: str, ds: Dataset, config: TrainingConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from sklearn.model_selection import ParameterSampler

    spaces: dict[str, dict[str, list[Any]]] = {
        "hgb": {"max_iter": [150, 300, 500], "learning_rate": [0.03, 0.05, 0.1],
                "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [20, 40, 100],
                "l2_regularization": [0.0, 1.0, 5.0]},
        "lgbm": {"n_estimators": [150, 300, 500], "learning_rate": [0.03, 0.05, 0.1],
                 "num_leaves": [15, 31, 63], "min_child_samples": [20, 40, 100],
                 "colsample_bytree": [0.7, 0.9, 1.0], "reg_lambda": [0.0, 1.0, 5.0]},
        "rf": {"n_estimators": [100, 200], "min_samples_leaf": [10, 20, 50], "max_features": [0.3, 0.5]},
    }
    trials: list[dict[str, Any]] = []
    best = dict(M.DEFAULT_PARAMS[name])
    best_mae = float("inf")
    t0 = time.monotonic()
    sampler = ParameterSampler(spaces[name], n_iter=config.tune_trials, random_state=config.seed)
    for params in sampler:
        if time.monotonic() - t0 > config.tune_seconds:
            trials.append({"stopped": "time_budget"})
            break
        merged = {**M.DEFAULT_PARAMS[name], **params}
        final, _, _ = _oof(name, ds, config, merged)
        ok = ~np.isnan(final)
        mae = float(np.mean(np.abs(final[ok] - ds.y[ok])))
        trials.append({"params": merged, "oof_mae": mae})
        if mae < best_mae:
            best, best_mae = merged, mae
    return best, trials


def _frame(ds: Dataset, mask: np.ndarray) -> pd.DataFrame:
    k = ds.ff.keys.iloc[np.flatnonzero(mask)].reset_index(drop=True)
    out = k[["player_id", "club_id", "match_id", "season", "stage_id", "stage_type", "match_date"]].copy()
    out["season_stage"] = out["season"].astype(str) + ":" + out["stage_id"].astype(str)
    out["history_games"] = ds.ff.history_games[mask]
    out["history_band"] = M.history_band(out["history_games"].to_numpy())
    out["actual"] = ds.y[mask]
    return out


def _metrics_by_candidate(df: pd.DataFrame, names: list[str]) -> dict[str, Any]:
    return {n: metric_block(df[f"pred_{n}"], df["actual"]).as_dict() for n in names}


def train_model(history: History, config: TrainingConfig, *, bundle_root: Path,
                clock: Callable[[], datetime]) -> TrainingResult:
    """Train, select, gate and persist (or reuse) a model bundle at an explicit cutoff."""
    from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    ds = prepare_dataset(history, config)
    timings["features_s"] = time.perf_counter() - t0
    key_inputs = {
        "snapshot_id": history.snapshot_id,
        "config": config.as_json(),
        "population": POPULATION,
        "feature_names": list(ds.ff.feature_names),
        "feature_dtypes": ds.ff.dtypes(),
        "folds": ds.plan.fingerprint(),
        "code": _code_hash(),
        "versions": dependency_versions(),
        "device": "cpu",
    }
    key = cache_key(key_inputs)
    bundle_id = make_bundle_id("bundle", key)
    if (bundle_root / bundle_id / "manifest.json").is_file():
        b = load_bundle(bundle_root, bundle_id)
        return TrainingResult(b, dict(b.manifest.holdout), pd.DataFrame(), reused=True)

    with threadpool_limits(limits=config.threads):
        baselines = [M.BASELINE_PRIOR5, M.BASELINE_COHORT]
        ml = list(config.ml_candidates())
        params: dict[str, dict[str, Any]] = {n: dict(config.params.get(n, M.DEFAULT_PARAMS[n])) for n in ml}
        tuning: dict[str, Any] = {}
        if config.tune:
            t1 = time.perf_counter()
            for n in ml:
                params[n], tuning[n] = _tune(n, ds, config)
            timings["tune_s"] = time.perf_counter() - t1
        # --- inner chronological OOF ------------------------------------------------
        t1 = time.perf_counter()
        oof: dict[str, dict[str, Any]] = {}
        raw_oof: dict[str, np.ndarray] = {}
        fold_ids = np.full(len(ds.y), -1)
        for n in baselines + ml:
            final, raw, fid = _oof(n, ds, config, params.get(n))
            ok = ~np.isnan(final)
            oof[n] = metric_block(final[ok], ds.y[ok]).as_dict()
            raw_oof[n] = raw
            fold_ids = fid
        timings["oof_s"] = time.perf_counter() - t1
        selected = min(ml, key=lambda n: oof[n]["mae"]) if ml else None
        recal = M.Recalibration()
        recal_scores: dict[str, float] = {}
        if selected is not None:
            recal, recal_scores = M.fit_recalibration(raw_oof[selected], ds.y, fold_ids)
        # --- final fits on the whole train block --------------------------------------
        t1 = time.perf_counter()
        tr = ds.blocks == "train"
        fitted: dict[str, M.FittedPredictor] = {}
        for n in baselines + ml:
            fitted[n] = _fit_predictor(n, subset(ds.ff, tr), ds.y[tr], config, params.get(n),
                                       recal if n == selected else None)
        timings["final_fit_s"] = time.perf_counter() - t1
        # --- calibration block: conformal residuals -----------------------------------
        cal_mask = ds.blocks == "calibration"
        hold_mask = ds.blocks == "holdout"
        cal_ff, hold_ff = subset(ds.ff, cal_mask), subset(ds.ff, hold_mask)
        intervals: dict[str, IntervalCalibration] = {}
        for n in {M.BASELINE_PRIOR5, *( [selected] if selected else [])}:
            p, _ = fitted[n].predict(cal_ff)
            intervals[n] = conformal_quantile(np.abs(p - ds.y[cal_mask]), config.interval_level,
                                              config.min_calibration)
        # --- holdout: score every candidate on identical rows ------------------------
        t1 = time.perf_counter()
        hold = _frame(ds, hold_mask)
        for n in baselines + ml:
            p, basis = fitted[n].predict(hold_ff)
            hold[f"pred_{n}"] = p
            hold[f"basis_{n}"] = basis
        timings["holdout_predict_s"] = time.perf_counter() - t1
        timings["holdout_rows_per_s"] = len(hold) / max(timings["holdout_predict_s"], 1e-9)
    hold["volume_band"] = M.volume_band(hold[f"pred_{M.BASELINE_PRIOR5}"].to_numpy())
    gate: GateDecision | None = None
    if selected is not None and len(hold):
        gate = champion_gate(
            hold, candidate=selected, baseline=M.BASELINE_PRIOR5, cand_col=f"pred_{selected}",
            base_col=f"pred_{M.BASELINE_PRIOR5}", actual_col="actual", cohort_dims=GATE_COHORT_DIMS,
            min_rel_improvement=config.gate_min_rel_improvement, cohort_min_n=config.cohort_min_n,
            cohort_max_rel_worse=config.cohort_max_rel_worse,
        )
    champion = selected if (gate is not None and gate.passed and selected) else M.BASELINE_PRIOR5
    if gate is None:
        note = "no ML candidate or no holdout rows; transparent prior-5 baseline ships"
    elif gate.passed:
        note = f"{selected} promoted: {gate.reason}"
    else:
        note = f"{selected} not promoted ({gate.reason}); transparent prior-5 baseline ships"
    ship_int = intervals[champion]
    lo, hi = apply_interval(hold[f"pred_{champion}"].to_numpy(), ship_int)
    int_eval = evaluate_interval(lo, hi, hold["actual"].to_numpy(), config.min_calibration)
    names = baselines + ml
    cohorts = cohort_metrics(
        hold, pred_col=f"pred_{champion}", actual_col="actual", dims=REPORT_COHORT_DIMS,
        baseline_col=f"pred_{M.BASELINE_PRIOR5}", min_n=config.cohort_min_n,
    ) if len(hold) else []
    sel_cohorts = cohort_metrics(
        hold, pred_col=f"pred_{selected}", actual_col="actual", dims=REPORT_COHORT_DIMS,
        baseline_col=f"pred_{M.BASELINE_PRIOR5}", min_n=config.cohort_min_n,
    ) if (len(hold) and selected) else []
    blocks = {b: int((ds.blocks == b).sum()) for b in ("train", "calibration", "holdout", "unused")}
    holdout_report: dict[str, Any] = {
        "label": "matched chronological holdout (model frozen at train_cutoff; features use "
                 "history before each target's match day)",
        "rows": len(hold),
        "metrics": _metrics_by_candidate(hold, names) if len(hold) else {},
        "mean_of_rounds_mae": {
            n: mean_of_groups_mae(hold[f"pred_{n}"].to_numpy(), hold["actual"].to_numpy(),
                                  hold["season_stage"].to_numpy()) for n in names
        } if len(hold) else {},
        "cold_start_rows": int((hold[f"basis_{M.BASELINE_PRIOR5}"] == M.COLD_START_NAME).sum()) if len(hold) else 0,
        "selected_candidate": selected,
        "selection_basis": "lowest inner-fold OOF MAE (holdout not used for selection)",
        "gate": asdict(gate) if gate else None,
        "champion": champion,
        "cohorts_champion_vs_baseline": [_cohort_json(c) for c in cohorts],
        "cohorts_selected_vs_baseline": [_cohort_json(c) for c in sel_cohorts],
        "interval": {**asdict(ship_int), "holdout": asdict(int_eval)},
        "interval_baseline_vs_selected": {k: asdict(v) for k, v in intervals.items()},
    }
    training_report: dict[str, Any] = {
        "blocks": blocks,
        "excluded": ds.excluded,
        "feature_diagnostics": ds.ff.diagnostics,
        "folds": [asdict(f) for f in ds.plan.folds],
        "oof_metrics": oof,
        "recalibration_scores": recal_scores,
        "tuning": tuning,
        "timings": timings,
        "cpu": cpu_info(),
    }
    predictors = {"champion": fitted[champion], "baseline": fitted[M.BASELINE_PRIOR5]}
    manifest = BundleManifest(
        bundle_id=bundle_id,
        kind="baseline" if champion.startswith("baseline") else "model",
        name=champion,
        description=_describe(champion),
        created_at=clock(),
        cache_key=key,
        cache_inputs=json_safe(key_inputs),
        snapshot_id=history.snapshot_id,
        train_cutoff=config.train_cutoff.isoformat(),
        calibration_end=config.calibration_end.isoformat(),
        holdout_end=config.holdout_end.isoformat() if config.holdout_end else None,
        feature_version=config.feature_spec.version,
        feature_names=list(ds.ff.feature_names),
        feature_dtypes=ds.ff.dtypes(),
        params=json_safe(params),
        seed=config.seed,
        threads=config.threads,
        versions=dependency_versions(),
        cold_start={"name": M.COLD_START_NAME, "value": fitted[champion].cold_start_value,
                    "min_history": fitted[champion].min_history},
        recalibration=json_safe(asdict(fitted[champion].recalibration)),
        interval=json_safe({**asdict(ship_int), "holdout": asdict(int_eval)}),
        training=json_safe(training_report),
        holdout=json_safe(holdout_report),
        promoted=champion != M.BASELINE_PRIOR5,
        promotion_note=note,
        payload_sha256="0" * 64,
    )
    bundle = save_bundle(bundle_root, manifest, predictors)
    return TrainingResult(bundle, holdout_report, hold, reused=False)


def _cohort_json(c: Any) -> dict[str, Any]:
    return {"dimension": c.dimension, "cohort": c.cohort, "model": c.model.as_dict(),
            "baseline": c.baseline.as_dict() if c.baseline else None, "sufficient": c.sufficient,
            "post_hoc": c.post_hoc}


def _describe(name: str) -> str:
    return {
        M.BASELINE_PRIOR5: "Mean disposals over the player's previous 5 eligible games "
                           "(cold starts: mean of zero-history training targets).",
        M.BASELINE_COHORT: "Training-period mean disposals by history band x prior-5 band.",
        "hgb": "HistGradientBoostingRegressor on features_v1 (fold-local imputer/encoder).",
        "lgbm": "LightGBM regressor (CPU) on features_v1 (fold-local imputer/encoder).",
        "rf": "RandomForestRegressor challenger on features_v1.",
    }.get(name, name)
