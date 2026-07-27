"""
Tests for backtest.py --from-csv (backtest-by-archive) mode.

--from-csv scores an already-published forward prediction CSV against the
recorded actuals for a single (year, round) instead of re-running the
predictor. This eliminates the namespace pollution where backtest's internal
predictor.run() wrote a next_round_*.csv into the live prediction directory
(mtime-newest, so downstream consumers shipped the backtest artifact, not the
forward prediction).

All tests are hermetic: synthetic CSV fixtures under tmp_path, no network, no
real data files, and the module-level BACKTEST_DIR / config.PREDICTION_DIR are
monkeypatched to tmp so nothing touches the repo's data/ tree.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import backtest


def _write_perf_csv(data_dir, slug, year, round_num, disposals, team="Test Team"):
    """Write a minimal *_performance_details.csv the actuals gatherer can read.

    slug is the leading name portion; the file name gains the
    `_performance_details.csv` suffix _gather_actuals globs for. The player
    name it derives is `parts[:-3]` title-cased, so slug must carry a trailing
    DOB-like token (e.g. `smith_john_01011990`) to name-resolve to "Smith John".
    """
    pd.DataFrame(
        {
            "team": [team],
            "year": [year],
            "round": [round_num],
            "disposals": [disposals],
        }
    ).to_csv(data_dir / f"{slug}_performance_details.csv", index=False)


def _write_pred_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def synthetic_env(tmp_path, monkeypatch):
    """A tmp player-data dir with one played round-16 actual (Smith John = 20
    disposals) and a monkeypatched BACKTEST_DIR + live PREDICTION_DIR.
    """
    data_dir = tmp_path / "player_data"
    data_dir.mkdir()
    bt_dir = tmp_path / "backtest"
    bt_dir.mkdir()
    live_dir = tmp_path / "prediction"
    live_dir.mkdir()
    monkeypatch.setattr(backtest, "BACKTEST_DIR", bt_dir)
    monkeypatch.setattr(backtest.config, "PREDICTION_DIR", str(live_dir))
    _write_perf_csv(data_dir, "smith_john_01011990", 2026, 16, 20, team="Test Team")
    return data_dir, bt_dir, live_dir


def test_from_csv_skips_predictor_run(synthetic_env, tmp_path):
    """With --from-csv, the LeakProofPredictor is never constructed or run."""
    data_dir, _bt_dir, _live_dir = synthetic_env
    pred_csv = tmp_path / "next_round_16_prediction_x.csv"
    _write_pred_csv(
        pred_csv,
        [{"player": "Smith John", "team": "Test Team", "predicted_disposals": 22}],
    )

    with patch.object(backtest, "LeakProofPredictor") as mock_pred:
        result, _detail = backtest.run_round_backtest(
            year=2026,
            round_num=16,
            data_dir=data_dir,
            timestamp="testts",
            log=MagicMock(),
            from_csv=pred_csv,
        )

    mock_pred.assert_not_called()
    assert result.n_with_actual == 1


def test_from_csv_loads_predictions_from_file(synthetic_env, tmp_path):
    """Predictions are read from the supplied CSV, not regenerated."""
    data_dir, _bt_dir, _live_dir = synthetic_env
    pred_csv = tmp_path / "archived.csv"
    _write_pred_csv(
        pred_csv,
        [{"player": "Smith John", "team": "Test Team", "predicted_disposals": 22}],
    )

    with patch.object(backtest, "LeakProofPredictor"):
        _result, detail = backtest.run_round_backtest(
            year=2026,
            round_num=16,
            data_dir=data_dir,
            timestamp="testts",
            log=MagicMock(),
            from_csv=pred_csv,
        )

    row = detail[detail["player"] == "Smith John"].iloc[0]
    assert row["predicted_disposals"] == 22  # from the supplied CSV
    assert row["actual_disposals"] == 20  # from the perf CSV
    assert row["error"] == pytest.approx(2.0)


def test_from_csv_does_not_write_to_live_namespace(synthetic_env, tmp_path):
    """--from-csv must not write any next_round_* file into the live prediction
    directory — that namespace belongs to the forward run only."""
    data_dir, _bt_dir, live_dir = synthetic_env
    pred_csv = tmp_path / "archived.csv"
    _write_pred_csv(
        pred_csv,
        [{"player": "Smith John", "team": "Test Team", "predicted_disposals": 22}],
    )

    with patch.object(backtest, "LeakProofPredictor") as mock_pred:
        backtest.run_round_backtest(
            year=2026,
            round_num=16,
            data_dir=data_dir,
            timestamp="testts",
            log=MagicMock(),
            from_csv=pred_csv,
        )

    assert list(live_dir.glob("next_round_*")) == []
    mock_pred.assert_not_called()


def test_without_from_csv_uses_predictor(synthetic_env, tmp_path):
    """Existing behaviour is unchanged: absent --from-csv, the predictor is
    constructed and run, and backtest scores the CSV it wrote."""
    data_dir, _bt_dir, _live_dir = synthetic_env

    mock_instance = MagicMock()

    def fake_run():
        # BL-01: the predictor writes into the isolated directory it was given,
        # not the live namespace.
        out = Path(mock_instance.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_pred_csv(
            out / "next_round_16_prediction_20260101_0000.csv",
            [{"player": "Smith John", "team": "Test Team", "predicted_disposals": 25}],
        )

    mock_instance.run.side_effect = fake_run

    def _ctor(*a, **kw):
        mock_instance.output_dir = kw.get("output_dir")
        return mock_instance

    with patch.object(
        backtest, "LeakProofPredictor", side_effect=_ctor
    ) as mock_cls:
        result, _detail = backtest.run_round_backtest(
            year=2026,
            round_num=16,
            data_dir=data_dir,
            timestamp="testts",
            log=MagicMock(),
        )

    mock_cls.assert_called_once()
    mock_instance.run.assert_called_once()
    assert result.n_with_actual == 1


def test_retrain_mode_ignores_stale_csv_with_newer_mtime(synthetic_env, tmp_path):
    """The predictor's output must be selected by identity (a file that did not
    exist before ``run()``), not by mtime-newest.

    ``data/prediction/`` accumulates one ``next_round_*`` CSV per round per run,
    so there is always a stale candidate. Selecting mtime-newest scores whatever
    file happens to be freshest on disk, which is not necessarily the one this
    round's predictor just wrote.
    """
    import os

    data_dir, _bt_dir, _live_dir = synthetic_env

    # Since BL-01 the predictor writes into a per-run directory, so a stale file
    # from ANOTHER round can no longer sit beside it. Identity selection still
    # matters though: plant a stale CSV inside that same run directory (a reused
    # or resumed run) and give it the newest mtime. The run dir name is
    # deterministic from the timestamp argument.
    run_dir = backtest.BACKTEST_DIR / "_runs" / "2026_r16_testts"
    run_dir.mkdir(parents=True, exist_ok=True)
    stale = run_dir / "next_round_15_prediction_20260101_0000.csv"
    _write_pred_csv(
        stale,
        [{"player": "Smith John", "team": "Test Team", "predicted_disposals": 99}],
    )
    fresh = run_dir / "next_round_17_prediction_20260101_0001.csv"

    def fake_run():
        _write_pred_csv(
            fresh,
            [{"player": "Smith John", "team": "Test Team", "predicted_disposals": 25}],
        )
        # Stale file now has the NEWEST mtime — mtime-newest selection picks it.
        future = os.stat(fresh).st_mtime + 10_000
        os.utime(stale, (future, future))

    mock_instance = MagicMock()
    mock_instance.run.side_effect = fake_run

    with patch.object(backtest, "LeakProofPredictor", return_value=mock_instance):
        _result, detail = backtest.run_round_backtest(
            year=2026,
            round_num=16,
            data_dir=data_dir,
            timestamp="testts",
            log=MagicMock(),
        )

    row = detail[detail["player"] == "Smith John"].iloc[0]
    assert row["predicted_disposals"] == 25, "scored the stale CSV, not the new one"


def test_retrain_mode_raises_when_predictor_writes_nothing(synthetic_env, tmp_path):
    """``run()`` warns-and-returns (does not raise) when it generates no
    predictions. Falling back to the newest pre-existing CSV would silently
    score another round's predictions against this round's actuals, so the
    backtest must fail loudly instead."""
    data_dir, _bt_dir, live_dir = synthetic_env

    _write_pred_csv(
        live_dir / "next_round_15_prediction_20260101_0000.csv",
        [{"player": "Smith John", "team": "Test Team", "predicted_disposals": 99}],
    )

    mock_instance = MagicMock()
    mock_instance.run.side_effect = lambda: None  # writes nothing

    with patch.object(backtest, "LeakProofPredictor", return_value=mock_instance):
        with pytest.raises(RuntimeError, match="wrote no"):
            backtest.run_round_backtest(
                year=2026,
                round_num=16,
                data_dir=data_dir,
                timestamp="testts",
                log=MagicMock(),
            )


def test_retrain_mode_never_writes_into_the_live_namespace(synthetic_env, tmp_path):
    """BL-01: a backtest run must not write into the directory forward-prediction
    consumers read.

    `predictor.run()` wrote its output straight into `config.PREDICTION_DIR` — the
    same namespace `data/prediction/next_round_*.csv` is served from. Downstream
    consumers resolve that directory by mtime-newest, so a backtest artifact could
    be shipped in place of the real forward prediction. That collision produced the
    tainted-provenance incident this whole cycle was opened to fix; the completion
    manifest added later only hides the symptom downstream.

    Identity-based selection (the earlier fix) stops the backtest from READING the
    wrong file. This asserts the stronger property: it never writes there at all.
    """
    data_dir, _bt_dir, live_dir = synthetic_env

    before = set(live_dir.iterdir())

    captured = {}
    mock_instance = MagicMock()

    def fake_run():
        # Write wherever the predictor was told to write — the whole point of the
        # fix is that backtest hands it an isolated directory.
        out = Path(mock_instance.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        captured["out"] = out
        _write_pred_csv(
            out / "next_round_17_prediction_20260101_0001.csv",
            [{"player": "Smith John", "team": "Test Team", "predicted_disposals": 25}],
        )

    mock_instance.run.side_effect = fake_run

    def _capture_ctor(*a, **kw):
        mock_instance.output_dir = kw.get("output_dir")
        return mock_instance

    with patch.object(backtest, "LeakProofPredictor", side_effect=_capture_ctor):
        backtest.run_round_backtest(
            year=2026,
            round_num=16,
            data_dir=data_dir,
            timestamp="testts",
            log=MagicMock(),
        )

    assert set(live_dir.iterdir()) == before, (
        "backtest wrote into the live prediction namespace: "
        f"{sorted(p.name for p in set(live_dir.iterdir()) - before)}"
    )
    assert captured.get("out") is not None, "predictor was never given an output dir"
    assert Path(captured["out"]).resolve() != Path(live_dir).resolve(), (
        "predictor was pointed at the live namespace"
    )
