"""CLI contract: fast side-effect-free help, doctor, exit codes, JSON result (P01-P04, U05)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from typer.testing import CliRunner

from supercoach_via.cli import EXIT, app

runner = CliRunner()


def test_help_is_fast_and_imports_no_ml_or_plotting(tmp_path: Path) -> None:
    code = (
        "import sys, time; t=time.perf_counter();"
        "from supercoach_via.cli import main;"
        "sys.argv=['scvia','--help'];\n"
        "try:\n main()\nexcept SystemExit: pass\n"
        "heavy=[m for m in ('sklearn','lightgbm','matplotlib','pandas','duckdb','pyarrow') if m in sys.modules];"
        "print('HEAVY', heavy, time.perf_counter()-t, file=sys.stderr)"
    )
    t0 = time.perf_counter()
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path, timeout=30)
    elapsed = time.perf_counter() - t0
    assert out.returncode == 0, out.stderr
    assert "HEAVY []" in out.stderr
    assert elapsed < 3.0  # budget is 1 s on the reference machine; generous for CI noise
    assert list(tmp_path.iterdir()) == []  # help touches no data/files


def test_doctor_json_reports_checks(tmp_path: Path) -> None:
    res = runner.invoke(
        app, ["doctor", "--json", "--data-root", str(tmp_path / "var"), "--output-root", str(tmp_path / "dist")]
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    names = {c["name"] for c in payload["checks"]}
    assert {"python", "writable_data_root", "public_base", "source_policy", "ml_extra"} <= names


def test_invalid_config_exit_code(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("nope = 1\n")
    res = runner.invoke(app, ["doctor", "--config", str(bad)])
    assert res.exit_code == EXIT["invalid_input"] == 2


def test_validate_release_unknown_is_validation_failure(tmp_path: Path) -> None:
    res = runner.invoke(app, ["validate-release", "--release", "nope", "--output-root", str(tmp_path)])
    assert res.exit_code == EXIT["invalid_input"]


def test_schemas_command_writes_files(tmp_path: Path) -> None:
    res = runner.invoke(app, ["schemas", "--out", str(tmp_path)])
    assert res.exit_code == 0
    assert (tmp_path / "release.schema.json").exists()


def _demo_src(tmp_path: Path) -> Path:
    from supercoach_via.demo import write_demo_corpus

    write_demo_corpus(tmp_path / "src")
    return tmp_path / "src"


def test_import_legacy_promotes_and_status_reports_it(tmp_path: Path) -> None:
    src = _demo_src(tmp_path)
    res = runner.invoke(app, ["import-legacy", "--source", str(src), "--data-root", str(tmp_path / "var"), "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    assert payload["ok"] and payload["promoted"] and payload["snapshot_id"].startswith("sha256:")
    st = runner.invoke(app, ["status", "--data-root", str(tmp_path / "var"), "--json"])
    assert st.exit_code == 0, st.output
    status = json.loads(st.stdout.strip().splitlines()[-1])
    assert status["current_snapshot"] == payload["snapshot_id"]
    assert status["last_run"]["state"] == "dataset_promoted"


def test_validate_command_on_current_snapshot(tmp_path: Path) -> None:
    src = _demo_src(tmp_path)
    runner.invoke(app, ["import-legacy", "--source", str(src), "--data-root", str(tmp_path / "var")])
    res = runner.invoke(app, ["validate", "--snapshot", "current", "--data-root", str(tmp_path / "var"), "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.stdout.strip().splitlines()[-1])["outcome"] == "PASS"


def test_refresh_plan_is_offline_and_write_free(tmp_path: Path) -> None:
    src = _demo_src(tmp_path)
    runner.invoke(app, ["import-legacy", "--source", str(src), "--data-root", str(tmp_path / "var")])
    before = sorted(p.relative_to(tmp_path) for p in (tmp_path / "var").rglob("*"))
    res = runner.invoke(app, ["refresh", "--season", "2026", "--plan", "--data-root", str(tmp_path / "var"), "--json"])
    assert res.exit_code == 0, res.output
    plan = json.loads(res.stdout.strip().splitlines()[-1])
    assert plan["network_during_plan"] is False and plan["estimated_requests"]["min"] >= 1
    assert sorted(p.relative_to(tmp_path) for p in (tmp_path / "var").rglob("*")) == before


def test_real_refresh_requires_explicit_network_opt_in(tmp_path: Path) -> None:
    res = runner.invoke(app, ["refresh", "--season", "2026", "--data-only", "--data-root", str(tmp_path / "var")])
    assert res.exit_code == EXIT["invalid_input"]


def test_build_release_without_forecast_inputs_is_honestly_unavailable(tmp_path: Path) -> None:
    src = _demo_src(tmp_path)
    var, dist = str(tmp_path / "var"), str(tmp_path / "dist")
    assert runner.invoke(app, ["import-legacy", "--source", str(src), "--data-root", var]).exit_code == 0
    res = runner.invoke(app, ["build-release", "--snapshot", "current", "--editorial", "off", "--demo",
                              "--data-root", var, "--output-root", dist, "--json"])  # fmt: skip
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["outputs"]["forecast_status"] == "unavailable"
    rid = out["outputs"]["release_id"]
    val = runner.invoke(app, ["validate-release", "--release", rid, "--output-root", dist, "--json"])
    assert val.exit_code == 0, val.output


def test_build_release_rejects_unknown_bundle_and_editorial_on(tmp_path: Path) -> None:
    var = str(tmp_path / "var")
    bad = runner.invoke(app, ["build-release", "--bundle", "../escape", "--data-root", var, "--json"])
    assert bad.exit_code == EXIT["invalid_input"]
    ed = runner.invoke(app, ["build-release", "--editorial", "on", "--data-root", var, "--json"])
    assert ed.exit_code == EXIT["invalid_input"]
