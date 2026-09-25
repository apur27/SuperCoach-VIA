"""Training contracts: M04 fold-local preprocessing, cold start, gate, caching."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from supercoach_via.ml import features as F
from supercoach_via.ml import models as M
from supercoach_via.ml import train as T
from supercoach_via.ml.evaluate import champion_gate
from tests.scvia.fixtures.ml.synthetic import write_corpus

SMALL = {"hgb": {"max_iter": 20, "learning_rate": 0.1, "max_leaf_nodes": 7, "min_samples_leaf": 5}}


def clock() -> datetime:
    return datetime(2026, 9, 25, tzinfo=UTC)


@pytest.fixture(scope="module")
def history(tmp_path_factory: pytest.TempPathFactory) -> F.History:
    root = tmp_path_factory.mktemp("snap")
    write_corpus(root)
    return F.load_history(root)


def _config(**kw: Any) -> T.TrainingConfig:
    base: dict[str, Any] = dict(
        train_cutoff=date(2025, 1, 1), calibration_end=date(2025, 4, 1), holdout_end=None,
        target_seasons_from=2023, candidates=("hgb",), n_folds=2, params=SMALL, threads=1,
    )
    base.update(kw)
    return T.TrainingConfig(**base)


class TestFoldLocalPreprocessing:
    def test_M04_imputer_and_encoder_fit_on_fold_training_rows_only(self, history: F.History) -> None:
        ds = T.prepare_dataset(history, _config())
        fold = ds.plan.folds[0]
        tr, _va = fold.masks(ds.dates)
        cand = M.make_candidate("hgb", seed=1, threads=1, params=SMALL["hgb"])
        fftr = T.subset(ds.ff, tr)
        cand.fit(fftr, ds.y[tr])
        assert isinstance(cand, M.PipelineCandidate)
        pre = cand.pipeline.named_steps["pre"]
        imp = pre.named_transformers_["num"]
        expected = fftr.X[list(fftr.numeric_features)].median(skipna=True).to_numpy()
        got = imp.statistics_
        both = ~np.isnan(expected)
        np.testing.assert_allclose(got[both], expected[both])
        full_med = ds.ff.X[list(ds.ff.numeric_features)].median().to_numpy()
        assert not np.allclose(got[both], full_med[both])  # would equal if fit on all rows
        enc = pre.named_transformers_["cat"].named_steps["onehot"]
        seen = set(fftr.X["opponent_club_id"].dropna())
        assert set(enc.categories_[1]) - {"__missing__"} <= seen

    def test_M04_unseen_category_and_schema_mismatch(self, history: F.History) -> None:
        ds = T.prepare_dataset(history, _config())
        cand = M.make_candidate("hgb", seed=1, threads=1, params=SMALL["hgb"])
        cand.fit(ds.ff, ds.y)
        weird = T.subset(ds.ff, np.arange(len(ds.y)) < 5)
        X = weird.X.copy()
        X["opponent_club_id"] = "brand_new_club"
        X["venue_id"] = None
        out = cand.predict(F.FeatureFrame(weird.keys, X, weird.numeric_features,
                                          weird.categorical_features, weird.history_games,
                                          weird.spec, weird.snapshot_id, {}))
        assert np.isfinite(out).all()
        dropped = F.FeatureFrame(weird.keys, X.drop(columns=["tog_last"]),
                                 tuple(c for c in weird.numeric_features if c != "tog_last"),
                                 weird.categorical_features, weird.history_games, weird.spec,
                                 weird.snapshot_id, {})
        with pytest.raises(ValueError):
            cand.predict(dropped)

    def test_M04_oof_fits_never_see_validation_or_later_rows(
        self, history: F.History, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[tuple[date, int]] = []
        real = M.make_candidate

        def spy(name: str, **kw: Any) -> Any:
            c = real(name, **kw)
            orig = c.fit

            def fit(ff: F.FeatureFrame, y: np.ndarray) -> Any:
                seen.append((max(ff.keys["match_date"]), len(y)))
                return orig(ff, y)

            c.fit = fit  # type: ignore[method-assign]
            return c

        monkeypatch.setattr(M, "make_candidate", spy)
        cfg = _config()
        ds = T.prepare_dataset(history, cfg)
        T._oof("hgb", ds, cfg, SMALL["hgb"])
        assert len(seen) == len(ds.plan.folds)
        for (mx, _), fold in zip(seen, ds.plan.folds, strict=True):
            assert mx < fold.valid_start


class TestPredictor:
    def test_nonnegative_only_no_1_to_55_clip(self) -> None:
        class Neg:
            name = "neg"

            def fit(self, ff: Any, y: Any) -> Any:
                return self

            def predict(self, ff: Any) -> np.ndarray:
                return np.array([-3.0, 0.4, 80.0])

        ff = type("FF", (), {"history_games": np.array([5, 5, 5])})()
        p, basis = M.FittedPredictor(Neg(), cold_start_value=12.0).predict(ff)  # type: ignore[arg-type]
        assert list(p) == [0.0, 0.4, 80.0]
        assert list(basis) == ["neg"] * 3

    def test_zero_history_uses_named_cold_start(self, history: F.History) -> None:
        ds = T.prepare_dataset(history, _config())
        pred = T._fit_predictor(M.BASELINE_PRIOR5, ds.ff, ds.y, _config(), None)
        p, basis = pred.predict(ds.ff)
        zero = ds.ff.history_games == 0
        assert zero.any()
        assert set(basis[zero]) == {M.COLD_START_NAME}
        assert np.allclose(p[zero], ds.y[zero].mean())


class TestGate:
    @staticmethod
    def _frame(n: int, cand_err: float, base_err: float, club_b_err: float | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        actual = rng.integers(0, 35, n).astype(float)
        club = np.where(np.arange(n) % 2 == 0, "a", "b")
        sign = np.where(rng.random(n) < 0.5, -1, 1)
        cand = actual + sign * cand_err
        if club_b_err is not None:
            cand = np.where(club == "b", actual + sign * club_b_err, cand)
        return pd.DataFrame({"actual": actual, "c": cand, "b": actual + sign * base_err, "club": club})

    def _gate(self, df: pd.DataFrame) -> Any:
        return champion_gate(df, candidate="c", baseline="b", cand_col="c", base_col="b",
                             actual_col="actual", cohort_dims={"club": "club"})

    def test_passes_when_better_everywhere(self) -> None:
        assert self._gate(self._frame(400, 4.0, 5.0)).passed

    def test_fails_below_one_percent(self) -> None:
        g = self._gate(self._frame(400, 4.97, 5.0))
        assert not g.passed and "< required" in g.reason

    def test_fails_when_a_sufficient_cohort_is_worse(self) -> None:
        g = self._gate(self._frame(400, 1.0, 5.0, club_b_err=5.5))
        assert g.relative_improvement is not None and g.relative_improvement > 0.01
        assert not g.passed and g.failing_cohorts == ["club=b"]

    def test_small_cohorts_are_descriptive_only(self) -> None:
        g = self._gate(self._frame(150, 1.0, 5.0, club_b_err=5.5))  # 75 rows per club < 100
        assert g.passed


class TestTrainModel:
    def test_end_to_end_reports_gate_and_reuses_cache(self, history: F.History, tmp_path: Path) -> None:
        cfg = _config()
        r = T.train_model(history, cfg, bundle_root=tmp_path / "models", clock=clock)
        assert not r.reused
        man = r.bundle.manifest
        assert man.snapshot_id == history.snapshot_id
        assert man.name in (M.BASELINE_PRIOR5, "hgb")
        rep = r.report
        assert rep["selected_candidate"] == "hgb"
        assert set(rep["metrics"]) == {M.BASELINE_PRIOR5, M.BASELINE_COHORT, "hgb"}
        assert rep["gate"]["n"] == rep["rows"] == len(r.holdout_rows) > 0
        # conformal needs >=200 calibration outcomes; the tiny fixture has fewer -> null + reason
        assert rep["interval"]["available"] is False
        assert rep["interval"]["reason"].startswith("insufficient_calibration")
        if not rep["gate"]["passed"]:
            assert man.name == M.BASELINE_PRIOR5 and "baseline ships" in man.promotion_note
        files = {p.name: p.read_bytes() for p in r.bundle.directory.iterdir()}  # type: ignore[union-attr]
        again = T.train_model(history, cfg, bundle_root=tmp_path / "models", clock=clock)
        assert again.reused and again.bundle.bundle_id == r.bundle.bundle_id
        assert {p.name: p.read_bytes() for p in again.bundle.directory.iterdir()} == files  # type: ignore[union-attr]
        other = T.train_model(history, _config(seed=99), bundle_root=tmp_path / "models", clock=clock)
        assert other.bundle.bundle_id != r.bundle.bundle_id

    def test_holdout_rows_are_after_calibration_and_training_rows_before_cutoff(
        self, history: F.History
    ) -> None:
        cfg = _config()
        ds = T.prepare_dataset(history, cfg)
        assert max(ds.dates[ds.blocks == "train"]) < cfg.train_cutoff
        assert min(ds.dates[ds.blocks == "holdout"]) >= cfg.calibration_end
        cal = ds.dates[ds.blocks == "calibration"]
        assert all(cfg.train_cutoff <= d < cfg.calibration_end for d in cal)

    def test_tuning_is_bounded(self) -> None:
        with pytest.raises(ValueError):
            _config(tune=True, tune_trials=31)
        with pytest.raises(ValueError):
            _config(tune=True, tune_seconds=1201)

    def test_tuning_records_trials(self, history: F.History, tmp_path: Path) -> None:
        r = T.train_model(history, _config(tune=True, tune_trials=2), bundle_root=tmp_path, clock=clock)
        trials = r.bundle.manifest.training["tuning"]["hgb"]
        assert len(trials) == 2 and all("oof_mae" in t for t in trials)

    def test_lightgbm_is_optional(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(M, "lightgbm_available", lambda: False)
        assert _config(candidates=("hgb", "lgbm")).ml_candidates() == ("hgb",)


def test_model_card_facts_come_from_manifest(history: F.History, tmp_path: Path) -> None:
    from supercoach_via.ml.bundles import model_card_facts

    r = T.train_model(history, _config(), bundle_root=tmp_path, clock=clock)
    facts = model_card_facts(r.bundle.manifest)
    text = "\n".join(facts)
    assert r.bundle.manifest.promotion_note in text
    assert "Interval: unavailable (insufficient_calibration" in text
    assert f"n={r.report['rows']}" in text
