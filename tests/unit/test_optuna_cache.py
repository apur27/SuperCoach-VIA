"""Tests for Optuna best-params caching in supercoach/prediction.py (F6).

The prediction pipeline runs two Optuna studies (HGB: 50 trials; LGBM: 50
trials / 600s) from scratch on every invocation. The training corpus grows
~0.5%/week and best params are stable week to week, so re-tuning 100 trials
weekly is ~20 min of wasted CPU. These tests pin the cache contract:

  * cache hit (row count within 5% AND age < 28 days) → skip Optuna entirely
  * cache miss (absent / stale / >5% row growth) → run Optuna, then save

All tests mock the Optuna study so no real tuning runs (no network, <10s),
and use tmp_path for the cache file so real data/prediction/ is untouched.
"""
import importlib.util
import json
import pathlib
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MODULE_PATH = REPO_ROOT / "supercoach" / "prediction.py"

# Real HGB constructor param names so the cache-miss path can build a Pipeline
# from the mocked study's best_params without sklearn rejecting them.
HGB_PARAMS = {"max_depth": 5, "learning_rate": 0.05, "loss": "squared_error"}


def _load_module():
    spec = importlib.util.spec_from_file_location("prediction_cache_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


@pytest.fixture
def predictor(mod, tmp_path):
    """Predictor with an explicit target_year (no data-dir scan) and a
    tmp_path cache file so the real cache is never touched."""
    p = mod.AFLDisposalPredictor(data_dir=str(tmp_path), target_year=2026)
    p.optuna_cache_path = tmp_path / "optuna_best_params.json"
    p._train_groups = None
    return p


def _tiny_xy(n=200):
    X = pd.DataFrame({"a": range(n), "b": range(n)})
    y = pd.Series(range(n), dtype=float)
    return X, y


def _write_cache(path, key, params, n_rows, tuned_at):
    payload = {
        key: {
            "params": params,
            "n_training_rows": n_rows,
            "tuned_at": tuned_at,
            "optuna_version": "4.2.1",
        }
    }
    pathlib.Path(path).write_text(json.dumps(payload))


def _fake_study():
    study = MagicMock()
    study.best_params = HGB_PARAMS
    study.best_value = -10.0
    return study


def test_cache_hit_skips_optuna(mod, predictor):
    """Valid cache (matching rows, fresh) → Optuna study is never created."""
    X, y = _tiny_xy(200)
    _write_cache(predictor.optuna_cache_path, "hgb", HGB_PARAMS, 200,
                 datetime.now().isoformat())
    study = _fake_study()
    with patch.object(mod.optuna, "create_study", return_value=study) as mock_create:
        predictor.tune_model(X, y)
    mock_create.assert_not_called()
    study.optimize.assert_not_called()
    assert "hgb_tuned" in predictor.models


def test_cache_miss_on_stale_age(mod, predictor):
    """Cache older than 28 days → Optuna runs despite matching row count."""
    X, y = _tiny_xy(200)
    stale = (datetime.now() - timedelta(days=29)).isoformat()
    _write_cache(predictor.optuna_cache_path, "hgb", HGB_PARAMS, 200, stale)
    study = _fake_study()
    with patch.object(mod.optuna, "create_study", return_value=study):
        predictor.tune_model(X, y)
    study.optimize.assert_called_once()


def test_cache_miss_on_row_growth(mod, predictor):
    """Cached row count differs >5% from current → Optuna runs."""
    X, y = _tiny_xy(200)  # current 200 vs cached 100 → 100% growth
    _write_cache(predictor.optuna_cache_path, "hgb", HGB_PARAMS, 100,
                 datetime.now().isoformat())
    study = _fake_study()
    with patch.object(mod.optuna, "create_study", return_value=study):
        predictor.tune_model(X, y)
    study.optimize.assert_called_once()


def test_cache_saved_after_tune(mod, predictor):
    """After a (mocked) tune run, the cache file is written with the schema."""
    X, y = _tiny_xy(200)
    study = _fake_study()
    with patch.object(mod.optuna, "create_study", return_value=study):
        predictor.tune_model(X, y)
    assert predictor.optuna_cache_path.exists()
    cache = json.loads(predictor.optuna_cache_path.read_text())
    assert "hgb" in cache
    entry = cache["hgb"]
    assert entry["params"] == HGB_PARAMS
    assert entry["n_training_rows"] == 200
    assert "tuned_at" in entry
    assert "optuna_version" in entry


def test_cache_miss_when_file_absent(mod, predictor):
    """No cache file → Optuna runs normally and the cache is written."""
    X, y = _tiny_xy(200)
    assert not predictor.optuna_cache_path.exists()
    study = _fake_study()
    with patch.object(mod.optuna, "create_study", return_value=study):
        predictor.tune_model(X, y)
    study.optimize.assert_called_once()
    assert predictor.optuna_cache_path.exists()
