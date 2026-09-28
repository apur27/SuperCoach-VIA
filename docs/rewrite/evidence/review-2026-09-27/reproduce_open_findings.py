"""Reproduce R01-R04 on the reviewed revision using only synthetic/temp data.

Usage: /path/to/venv/bin/python reproduce_open_findings.py /path/to/checkout
Pinned review revision: 2aad178730990aad2623bd06cb0abfb17c5b0987.
This diagnostic imports the checkout's existing test fixtures and implementation.
It is expected to change behavior after the fixes; use permanent regression tests
for acceptance, rather than preserving vulnerable behavior to satisfy this script.
"""

from dataclasses import replace
from datetime import UTC, date, datetime
import json
from pathlib import Path
import sys
import tempfile


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: reproduce_open_findings.py /absolute/checkout")
    checkout = Path(sys.argv[1]).resolve(strict=True)
    sys.path.insert(0, str(checkout))
    sys.path.insert(0, str(checkout / "src"))

    from tests.scvia.unit.test_release import _write_minimal, NOW
    from tests.scvia.fixtures.ml.synthetic import build_corpus
    from supercoach_via.domain.schemas import Origin
    from supercoach_via.ml import features as F, train as T, predict as P, evaluate as E
    from supercoach_via.publish import release as R

    root = Path(tempfile.mkdtemp(prefix="scvia-review-findings-"))
    print("TEMP_ROOT", root)

    release = _write_minimal(root / "output", "review-r1")
    assert R.validate_release(release).ok
    site = release / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>Unchecked synthetic replacement</h1>")
    (site / "private.json").write_text('{"synthetic_private_marker":true}')
    assert R.validate_release(release, write=False).ok
    receipt = R.publish_release(
        release, R.LocalDirectoryDestination("local", root / "host"), clock=lambda: NOW
    )
    print("R01", json.dumps({
        "status": receipt.status,
        "unchecked_file_published": (root / "host/live/private.json").exists(),
    }))

    history = F.load_history(build_corpus(future_rounds=1).write(root / "history"))
    spec = F.FeatureSpec(window_long=2)
    config = T.TrainingConfig(
        train_cutoff=date(2025, 1, 1), calibration_end=date(2025, 4, 1),
        holdout_end=date(2026, 1, 1), target_seasons_from=2023,
        candidates=(), n_folds=2, threads=1, feature_spec=spec,
    )
    bundle = T.train_model(history, config, bundle_root=root / "models", clock=lambda: NOW).bundle
    early = P.ForecastRequest(
        forecast_cutoff=datetime(2024, 3, 1, tzinfo=UTC), generated_at=NOW,
        origin=Origin.REPLAY, season=2024, stage_id="r01",
    )
    replay = P.forecast(history, bundle, early)
    evaluation = E.evaluate(replay, history)
    print("R02", json.dumps({
        "forecast_cutoff": early.forecast_cutoff.isoformat(),
        "train_cutoff": bundle.manifest.train_cutoff,
        "calibration_end": bundle.manifest.calibration_end,
        "status": replay.manifest.status, "scored_rows": evaluation.headline.n,
    }))

    later = P.ForecastRequest(forecast_cutoff=datetime(2026, 3, 10, tzinfo=UTC), generated_at=NOW)
    artifact = P.forecast(history, bundle, later)
    expected = F.build_features(history, artifact.features.keys, spec)
    actual_values = artifact.features.X["disposals_prior5_mean"]
    expected_values = expected.X["disposals_prior5_mean"]
    mismatch = (actual_values - expected_values).abs().fillna(0) > 1e-10
    print("R03", json.dumps({
        "training_window": spec.window_long,
        "inference_window": artifact.features.spec.window_long,
        "differing_rows": int(mismatch.sum()),
    }))

    games = history.player_games.copy()
    games.loc[games.index[0], "available_at"] = datetime(2026, 2, 1, tzinfo=UTC)
    revised = replace(history, player_games=games)
    try:
        F.build_features(revised, F.historical_targets(revised, seasons=(2025,)), F.FeatureSpec())
    except F.FeatureOrderError as exc:
        print("R04", type(exc).__name__, str(exc))
    else:
        print("R04", "No FeatureOrderError; inspect cutoff correctness with the regression oracle.")


if __name__ == "__main__":
    main()
