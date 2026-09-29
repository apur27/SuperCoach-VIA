"""Integrity checker, model bundles and prediction artifacts (family F). No model is loaded or trained."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from supercoach_via.integrity.runner import AuditOptions, AuditResult, run_audit
from supercoach_via.ml.bundles import BundleManifest
from tests.scvia.unit import integrity_fixtures as fx

AS_OF = "2026-05-03T00:00:00Z"


@pytest.fixture
def demo(integrity_demo: Any) -> Any:
    return integrity_demo.env


@pytest.fixture
def env(demo: Any, tmp_path: Path) -> dict[str, Path]:
    out = {"data": tmp_path / "var", "models": tmp_path / "models", "predictions": tmp_path / "predictions"}
    shutil.copytree(demo.data_root, out["data"])
    shutil.copytree(demo.root / "models", out["models"])
    shutil.copytree(demo.root / "predictions", out["predictions"])
    for tree in ("models", "predictions"):  # copies of write-once artifacts; tests edit them
        for p in out[tree].rglob("*"):
            p.chmod(0o755 if p.is_dir() else 0o644)
    return out


def audit(env: dict[str, Path], **kw: Any) -> AuditResult:
    opts = dict(
        data_root=env["data"],
        models_root=env["models"],
        predictions_root=env["predictions"],
        scope="data",
        as_of=AS_OF,
        families=("models",),
        keep_all=True,
    )
    opts.update(kw)
    return run_audit(AuditOptions(**opts))  # type: ignore[arg-type]


def found(res: AuditResult, rule_id: str) -> list[Any]:
    return [f for f in res.findings if f.rule_id == rule_id and f.status == "open"]


def _bundle(env: dict[str, Path]) -> Path:
    return next(p for p in sorted(env["models"].iterdir()) if (p / "manifest.json").is_file())


def _rewrite_bundle(env: dict[str, Path], edit: Any) -> None:
    path = _bundle(env) / "manifest.json"
    m = BundleManifest.model_validate_json(path.read_bytes())
    data = m.model_dump(mode="json")
    edit(data)
    m2 = BundleManifest.model_validate(data)
    m2 = m2.model_copy(update={"manifest_sha256": m2.self_hash()})  # a valid self-hash: only semantics can catch it
    path.write_text(m2.model_dump_json(indent=1))


def _prediction(env: dict[str, Path]) -> Path:
    return next(p for p in sorted(env["predictions"].iterdir()) if (p / "manifest.json").is_file())


def _rewrite_rows(env: dict[str, Path], edit: Any) -> None:
    d = _prediction(env)
    t = pq.read_table(d / "rows.parquet")
    rows = t.to_pylist()
    edit(rows)
    pq.write_table(pa.Table.from_pylist(rows, schema=t.schema), d / "rows.parquet")
    man = json.loads((d / "manifest.json").read_text())
    man["rows_sha256"] = hashlib.sha256((d / "rows.parquet").read_bytes()).hexdigest()
    (d / "manifest.json").write_text(json.dumps(man, indent=1))


def test_clean_available_forecast_passes(env: dict[str, Path]) -> None:
    res = audit(env)
    assert res.outcome.value == "PASS", [f.as_dict() for f in res.findings]
    assert res.findings == []
    assert res.report["inputs"]["models"]["predictions"][0]["status"] == "available"


def test_wrong_feature_order(env: dict[str, Path]) -> None:
    def edit(d: dict[str, Any]) -> None:
        d["feature_names"][0], d["feature_names"][1] = d["feature_names"][1], d["feature_names"][0]

    _rewrite_bundle(env, edit)
    res = audit(env)
    assert found(res, "models.feature_order")
    assert res.outcome.value == "FAIL"


def test_wrong_feature_fingerprint(env: dict[str, Path]) -> None:
    _rewrite_bundle(env, lambda d: d["cache_inputs"]["config"]["feature_spec"].update(window_long=2))
    res = audit(env)
    assert found(res, "models.feature_spec")


def test_future_knowledge_cutoff(env: dict[str, Path]) -> None:
    _rewrite_bundle(env, lambda d: d["training"].update(knowledge_cutoff="2026-06-01T00:00:00+00:00"))
    res = audit(env)
    assert found(res, "models.ineligible_bundle")
    assert found(res, "models.knowledge_after_as_of")
    assert res.outcome.value == "FAIL"


def test_invalid_prediction_interval(env: dict[str, Path]) -> None:
    def edit(rows: list[dict[str, Any]]) -> None:
        rows[0].update(interval_low=20.0, interval_high=10.0)

    _rewrite_rows(env, edit)
    res = audit(env)
    assert found(res, "models.interval_order")


def test_non_finite_prediction(env: dict[str, Path]) -> None:
    _rewrite_rows(env, lambda rows: rows[0].update(predicted_disposals=float("inf")))
    res = audit(env)
    assert found(res, "models.prediction_value")


def test_tampered_rows_file(env: dict[str, Path]) -> None:
    d = _prediction(env)
    (d / "rows.parquet").write_bytes((d / "rows.parquet").read_bytes() + b"x")
    res = audit(env)
    assert found(res, "models.prediction_file_hash")


def test_row_names_another_model(env: dict[str, Path]) -> None:
    _rewrite_rows(env, lambda rows: rows[0].update(model_id="bundle-other"))
    res = audit(env)
    assert found(res, "models.row_identity")


def test_missing_mandatory_payload_for_an_available_forecast(env: dict[str, Path]) -> None:
    (_bundle(env) / "predictor.joblib").unlink()
    res = audit(env)
    assert found(res, "models.payload_missing")
    assert res.outcome.value == "FAIL"


def test_missing_bundle_for_an_available_forecast(env: dict[str, Path]) -> None:
    shutil.rmtree(_bundle(env))
    res = audit(env)
    assert found(res, "models.bundle_missing")


def _unavailable(root: Path, snapshot_id: str) -> None:
    d = root / "prospective-20260928T000000Z-0000000000000000"
    d.mkdir(parents=True)
    pq.write_table(pa.table({"prediction_id": pa.array([], pa.string())}), d / "rows.parquet")
    (d / "omissions.json").write_text("[]")
    man = {
        "schema_version": 1,
        "kind": "disposal_forecast",
        "prediction_run_id": d.name,
        "origin": "prospective",
        "status": "unavailable",
        "reason": "no_valid_future_fixture",
        "snapshot_id": snapshot_id,
        "model_id": "bundle-absent",
        "model_name": "lgbm",
        "model_promoted": True,
        "baseline_model_id": "bundle-absent:baseline_prior5",
        "feature_version": "features_v1",
        "forecast_cutoff": "2026-09-28T00:00:00Z",
        "generated_at": "2026-09-28T01:00:00Z",
        "season": None,
        "stage_id": None,
        "stage_label": None,
        "target_matches": {},
        "intended": 0,
        "predicted": 0,
        "omissions": {},
        "interval": {"available": False},
        "rows_file": "rows.parquet",
        "rows_sha256": hashlib.sha256((d / "rows.parquet").read_bytes()).hexdigest(),
        "omissions_file": "omissions.json",
        "omissions_sha256": hashlib.sha256(b"[]").hexdigest(),
        "warnings": [],
    }
    (d / "manifest.json").write_text(json.dumps(man))


def test_correctly_unavailable_forecast_passes(tmp_path: Path) -> None:
    root = tmp_path / "var"
    m = fx.build(root)
    _unavailable(root / "predictions", m.snapshot_id)
    res = run_audit(AuditOptions(data_root=root, scope="data", as_of=fx.AS_OF, families=("models",), keep_all=True))
    assert res.outcome.value == "PASS", [f.as_dict() for f in res.findings]


def test_unavailable_forecast_while_a_future_fixture_exists(tmp_path: Path) -> None:
    def add_future(r: dict[str, list[dict[str, Any]]]) -> None:
        future = fx._match("m:2026:r03:alpha:beta:0", 2026, "3", 3, date(2026, 10, 3), "alpha", "beta", (0, 0), (0, 0))
        for k in list(future):
            if k.endswith(("_goals", "_behinds", "_score")):
                future[k] = None
        future["status"] = "scheduled"
        r["matches"].append(future)
        r["seasons"][1]["matches_scheduled"] = 1

    root = tmp_path / "var"
    m = fx.rehash(root, add_future)
    _unavailable(root / "predictions", m.snapshot_id)
    res = run_audit(AuditOptions(data_root=root, scope="data", as_of=fx.AS_OF, families=("models",), keep_all=True))
    assert found(res, "models.forecast_unavailable_with_fixture")


def test_no_model_inputs_is_not_applicable(tmp_path: Path) -> None:
    root = tmp_path / "var"
    fx.build(root)
    res = run_audit(AuditOptions(data_root=root, scope="data", as_of=fx.AS_OF, families=("models",)))
    assert {c["status"] for c in res.report["checks"] if c["family"] == "models"} == {"NOT_APPLICABLE"}
    assert res.outcome.value == "PASS"
