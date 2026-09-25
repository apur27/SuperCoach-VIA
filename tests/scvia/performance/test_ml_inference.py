"""Batch inference budget: <=2 s per 1,000 candidates with a loaded model (PLAN 12)."""

from __future__ import annotations

import os
import time
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from supercoach_via.ml import bundles as B
from supercoach_via.ml import features as F
from supercoach_via.ml import train as T

REPO = Path(__file__).resolve().parents[3]
SMALL = {"hgb": {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 40}}


def test_batch_inference_1000_under_2s(tmp_path: Path) -> None:
    root = Path(os.environ.get("SCVIA_ML_DATA_ROOT", REPO / "var" / "agent-import"))
    h = F.load_history(root, os.environ.get("SCVIA_ML_SNAPSHOT", "current"))
    cfg = T.TrainingConfig(train_cutoff=date(2025, 1, 1), calibration_end=date(2025, 7, 1),
                           holdout_end=date(2026, 1, 1), target_seasons_from=2022,
                           candidates=("hgb",), n_folds=2, params=SMALL, threads=4)
    res = T.train_model(h, cfg, bundle_root=tmp_path, clock=lambda: datetime(2026, 9, 25, tzinfo=UTC))
    bundle = B.load_bundle(tmp_path, res.bundle.bundle_id)
    targets = F.historical_targets(h, seasons=(2026,)).head(1000)
    F.build_features(h, targets)  # prepared-history warm state, as for a loaded serving process
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        ff = F.build_features(h, targets)
        bundle.predictor["champion"].predict(ff)
        bundle.predictor["baseline"].predict(ff)
        times.append(time.perf_counter() - t0)
    assert float(np.median(times)) <= 2.0, times
