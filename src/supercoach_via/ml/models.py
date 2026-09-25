"""Candidate predictors: baselines, cold-start prior and pipeline estimators.

Every candidate exposes ``fit(ff, y) -> self`` and ``predict(ff) -> ndarray`` over a
``FeatureFrame``. Preprocessing (imputation + one fitted categorical encoder with
unknown-category handling, fixed output order) lives *inside* the estimator pipeline, so
fitting a candidate on a fold's training rows fits its preprocessing on those rows only.

``FittedPredictor`` composes a candidate with the named cold-start prior (targets with no
eligible history get the prior, never an individual model prediction), an optional
recalibration map and the nonnegative output constraint (no 1-55 clipping).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from supercoach_via.ml.features import FeatureFrame

COLD_START_NAME = "cold_start_prior_v1"
BASELINE_PRIOR5 = "baseline_prior5"
BASELINE_COHORT = "baseline_cohort"
HISTORY_BANDS = (0, 1, 5, 20, 50, 100, 200)
PRIOR5_EDGES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)


def history_band(h: np.ndarray) -> np.ndarray:
    labels = ["0", "1-4", "5-19", "20-49", "50-99", "100-199", "200+"]
    idx = np.searchsorted(np.asarray(HISTORY_BANDS), np.asarray(h), "right") - 1
    return np.array([labels[max(i, 0)] for i in idx], dtype=object)


def volume_band(pred: np.ndarray) -> np.ndarray:
    labels = ["<5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"]
    idx = np.searchsorted(np.asarray(PRIOR5_EDGES), np.asarray(pred), "right") - 1
    return np.array([labels[max(i, 0)] for i in idx], dtype=object)


class Candidate(Protocol):
    name: str

    def fit(self, ff: FeatureFrame, y: np.ndarray) -> Candidate: ...

    def predict(self, ff: FeatureFrame) -> np.ndarray: ...


@dataclass
class Prior5Baseline:
    """Prior 5-game mean of disposals; NaN where no observed prior value."""

    name: str = BASELINE_PRIOR5

    def fit(self, ff: FeatureFrame, y: np.ndarray) -> Prior5Baseline:
        return self

    def predict(self, ff: FeatureFrame) -> np.ndarray:
        return ff.X["disposals_prior5_mean"].to_numpy(dtype=float)


@dataclass
class CohortBaseline:
    """Past-data cohort mean by (history band x prior-5 band); unseen -> training mean."""

    name: str = BASELINE_COHORT
    table: dict[str, float] = field(default_factory=dict)
    global_mean: float = float("nan")

    @staticmethod
    def _keys(ff: FeatureFrame) -> np.ndarray:
        p5 = ff.X["disposals_prior5_mean"].to_numpy(dtype=float)
        vb = np.where(np.isnan(p5), "na", volume_band(np.nan_to_num(p5)))
        hb = history_band(ff.history_games)
        return np.array([f"{a}|{b}" for a, b in zip(hb, vb, strict=True)], dtype=object)

    def fit(self, ff: FeatureFrame, y: np.ndarray) -> CohortBaseline:
        k = self._keys(ff)
        s = pd.Series(y).groupby(k).mean()
        self.table = {str(a): float(b) for a, b in s.items()}
        self.global_mean = float(np.mean(y))
        return self

    def predict(self, ff: FeatureFrame) -> np.ndarray:
        return np.array([self.table.get(k, self.global_mean) for k in self._keys(ff)], dtype=float)


def _preprocessor(ff: FeatureFrame) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                 sparse_output=False, dtype=np.float64)),
    ])
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median", keep_empty_features=True), list(ff.numeric_features)),
            ("cat", cat, list(ff.categorical_features)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _categorical_frame(ff: FeatureFrame) -> pd.DataFrame:
    X = ff.X.copy()
    for c in ff.categorical_features:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


@dataclass
class PipelineCandidate:
    """sklearn Pipeline(preprocess -> estimator); estimator built from ``kind``/``params``."""

    name: str
    kind: str  # hgb | lgbm | rf
    params: dict[str, Any]
    seed: int
    threads: int
    pipeline: Any = None
    feature_names_in: tuple[str, ...] = ()

    def _estimator(self) -> Any:
        if self.kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(random_state=self.seed, early_stopping=False, **self.params)
        if self.kind == "rf":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(random_state=self.seed, n_jobs=self.threads, **self.params)
        if self.kind == "lgbm":
            import lightgbm as lgb

            return lgb.LGBMRegressor(random_state=self.seed, n_jobs=self.threads, device_type="cpu",
                                     verbose=-1, deterministic=True, force_row_wise=True, **self.params)
        raise ValueError(f"unknown estimator kind {self.kind!r}")

    def fit(self, ff: FeatureFrame, y: np.ndarray) -> PipelineCandidate:
        from sklearn.pipeline import Pipeline

        self.feature_names_in = ff.feature_names
        self.pipeline = Pipeline([("pre", _preprocessor(ff)), ("model", self._estimator())])
        self.pipeline.fit(_categorical_frame(ff), y)
        return self

    def predict(self, ff: FeatureFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("candidate not fitted")
        if ff.feature_names != self.feature_names_in:
            raise ValueError("feature schema/order differs from the one the model was fitted on")
        return np.asarray(self.pipeline.predict(_categorical_frame(ff)), dtype=float)


DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "hgb": {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 40,
            "l2_regularization": 1.0},
    "lgbm": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 40,
             "subsample": 1.0, "colsample_bytree": 0.9, "reg_lambda": 1.0},
    "rf": {"n_estimators": 200, "min_samples_leaf": 20, "max_features": 0.5},
}


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        return False
    return True


def make_candidate(name: str, *, seed: int, threads: int, params: dict[str, Any] | None = None) -> Candidate:
    if name == BASELINE_PRIOR5:
        return Prior5Baseline()
    if name == BASELINE_COHORT:
        return CohortBaseline()
    if name in DEFAULT_PARAMS:
        return PipelineCandidate(name=name, kind=name, params=dict(params or DEFAULT_PARAMS[name]),
                                 seed=seed, threads=threads)
    raise ValueError(f"unknown candidate {name!r}")


@dataclass
class Recalibration:
    method: str = "identity"  # identity | linear
    a: float = 0.0
    b: float = 1.0

    def apply(self, p: np.ndarray) -> np.ndarray:
        return p if self.method == "identity" else self.a + self.b * p


@dataclass
class FittedPredictor:
    """A candidate plus the named cold-start prior, recalibration and >=0 constraint."""

    candidate: Candidate
    cold_start_value: float
    min_history: int = 1
    recalibration: Recalibration = field(default_factory=Recalibration)

    @property
    def name(self) -> str:
        return self.candidate.name

    def predict(self, ff: FeatureFrame) -> tuple[np.ndarray, np.ndarray]:
        """(predictions >= 0, eligibility basis per row)."""
        raw = self.recalibration.apply(self.candidate.predict(ff))
        cold = (ff.history_games < self.min_history) | np.isnan(raw)
        pred = np.where(cold, self.cold_start_value, raw)
        basis = np.where(cold, COLD_START_NAME, self.candidate.name).astype(object)
        return np.maximum(pred, 0.0), basis


def fit_cold_start(ff: FeatureFrame, y: np.ndarray) -> float:
    """Mean outcome of zero-history training targets (debuts); global mean if none."""
    zero = ff.history_games == 0
    if zero.sum() >= 1:
        return float(np.mean(y[zero]))
    return float(np.mean(y))


def fit_recalibration(
    oof_pred: np.ndarray, y: np.ndarray, fold_ids: np.ndarray
) -> tuple[Recalibration, dict[str, float]]:
    """Choose identity vs linear recalibration by leave-one-fold-out MAE on OOF rows only."""
    ok = ~np.isnan(oof_pred)
    p, t, f = oof_pred[ok], y[ok], fold_ids[ok]
    folds = np.unique(f)
    if len(folds) < 2 or len(p) < 50:
        return Recalibration(), {"identity": float("nan"), "linear": float("nan")}
    lin_pred = np.empty_like(p)
    for k in folds:
        tr = f != k
        b, a = np.polyfit(p[tr], t[tr], 1)
        lin_pred[~tr] = a + b * p[~tr]
    mae_id = float(np.mean(np.abs(np.maximum(p, 0) - t)))
    mae_lin = float(np.mean(np.abs(np.maximum(lin_pred, 0) - t)))
    scores = {"identity": mae_id, "linear": mae_lin}
    if mae_lin < mae_id:
        b, a = np.polyfit(p, t, 1)
        return Recalibration("linear", float(a), float(b)), scores
    return Recalibration(), scores
