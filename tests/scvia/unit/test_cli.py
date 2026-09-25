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
