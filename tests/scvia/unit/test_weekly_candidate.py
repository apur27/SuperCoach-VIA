"""Command order of the numeric weekly candidate. No network and no site build."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "scvia_weekly.sh"
SOURCE_ID = "sha256:" + "b" * 64
CORRECTED_ID = "sha256:" + "c" * 64


def _fake(tmp_path: Path) -> tuple[Path, Path]:
    log = tmp_path / "calls.log"
    fake = tmp_path / "scvia"
    source = json.dumps({"ok": True, "outputs": {}, "snapshot_id": SOURCE_ID})
    corrected = json.dumps({"ok": True, "outputs": {}, "snapshot_id": CORRECTED_ID})
    fake.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$CALL_LOG"\n'
        'case "$1" in\n'
        f"  apply-corrections) echo '{corrected}' ;;\n"
        "  forecast) echo '{\"outputs\":{\"bundle_id\":\"bundle-test\",\"prediction_dir\":\"/tmp/pred\"}}' ;;\n"
        "  build-release) echo '{\"outputs\":{\"release_id\":\"r-test\"}}' ;;\n"
        f"  *) echo '{source}' ;;\n"
        "esac\n"
    )
    fake.chmod(0o755)
    return fake, log


def _env(tmp_path: Path, fake: Path, log: Path, **extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "SCVIA_BIN": str(fake),
            "SCVIA_VAR_DIR": str(tmp_path / "var"),
            "SCVIA_DATA_ROOT": str(tmp_path / "data"),
            "SCVIA_OUTPUT_ROOT": str(tmp_path / "out"),
            "SCVIA_SKIP_SITE": "1",
            "CALL_LOG": str(log),
            "SCVIA_SOURCE_MODE": "rehearsal",
            "SCVIA_CORRECTION_EVIDENCE_ROOT": str(tmp_path / "evidence"),
            "SCVIA_CORRECTION_EVIDENCE_SNAPSHOT": "sha256:" + "a" * 64,
        }
    )
    marker = tmp_path / "cycle.json"
    marker.write_text('{"phase": "4", "exit_code": 0}\n')
    env.pop("SCVIA_ALLOW_NETWORK", None)
    env.pop("SCVIA_CAPTURED_SOURCE", None)
    env.pop("SCVIA_LEGACY_ROOT", None)
    env.pop("SCVIA_LOCAL_DEST", None)
    env["SCVIA_CYCLE_MARKER"] = str(marker)
    env["SCVIA_FORECAST_CUTOFF"] = "2026-09-25T00:00:00Z"
    env["SCVIA_LOCK_DIR"] = str(tmp_path / "locks")
    env.update(extra)
    return env


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, env=env, text=True, capture_output=True)


def test_rehearsal_imports_the_captured_source_and_does_not_refresh(tmp_path: Path) -> None:
    fake, log = _fake(tmp_path)
    captured = tmp_path / "captured"
    (captured / "data").mkdir(parents=True)
    result = _run(_env(tmp_path, fake, log, SCVIA_CAPTURED_SOURCE=str(captured)))
    assert result.returncode == 0, result.stderr
    calls = log.read_text()
    assert calls.splitlines()[0].startswith("import-legacy ")
    assert calls.splitlines()[1].startswith("apply-corrections ")
    assert f"--evidence-root {tmp_path / 'evidence'}" in calls.splitlines()[1]
    assert "--evidence-snapshot sha256:" + "a" * 64 in calls.splitlines()[1]
    assert "--season 2026" in calls.splitlines()[1]
    assert f"--snapshot {CORRECTED_ID}" in calls
    assert f"--snapshot {SOURCE_ID}" not in calls
    assert "--snapshot current" not in calls
    assert "--allow-network" not in calls
    assert "refresh " not in calls
    assert f"--source {captured}" in calls
    status = json.loads((tmp_path / "var" / "scvia-weekly-status.json").read_text())
    assert status["mode"] == "rehearsal" and status["phase"] == "complete" and status["exit_code"] == 0
    assert status["forecast_cutoff"] == "2026-09-25T00:00:00Z" and status["run_id"] and status["started"]
    assert status["snapshot_id"] == CORRECTED_ID
    assert status["source_snapshot_id"] == SOURCE_ID
    assert (Path(status["evidence_dir"]) / "corrections.json").is_file()


def test_rehearsal_without_pinned_correction_evidence_stops_before_import(tmp_path: Path) -> None:
    fake, log = _fake(tmp_path)
    captured = tmp_path / "captured"
    (captured / "data").mkdir(parents=True)
    env = _env(tmp_path, fake, log, SCVIA_CAPTURED_SOURCE=str(captured))
    env.pop("SCVIA_CORRECTION_EVIDENCE_ROOT")
    env.pop("SCVIA_CORRECTION_EVIDENCE_SNAPSHOT")
    result = _run(env)
    assert result.returncode == 2
    assert "SCVIA_CORRECTION_EVIDENCE" in result.stderr
    assert not log.exists()


def test_comparison_alignment_uses_source_and_corrected_snapshot_ids(tmp_path: Path) -> None:
    fake, log = _fake(tmp_path)
    python = tmp_path / "python"
    python.write_text('#!/bin/sh\nprintf "python %s\\n" "$*" >> "$CALL_LOG"\necho "{}"\n')
    python.chmod(0o755)
    captured = tmp_path / "captured"
    (captured / "data").mkdir(parents=True)
    result = _run(_env(tmp_path, fake, log, SCVIA_CAPTURED_SOURCE=str(captured), SCVIA_LEGACY_ROOT="1"))
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    [regenerate] = [call for call in calls if " --regenerate " in call]
    [compare] = [call for call in calls if " --promoted " in call]
    assert f"--correction-data-root {tmp_path / 'data'}" in regenerate
    assert f"--source-snapshot {SOURCE_ID}" in regenerate
    assert f"--corrected-snapshot {CORRECTED_ID}" in regenerate
    assert f"--snapshot {CORRECTED_ID}" in compare


@pytest.mark.parametrize("failed_output, command_exit", [
    ('{"ok":false,"exit_code":3,"snapshot_id":"sha256:wrong"}', 3),
    ('{"ok":false,"exit_code":3,"snapshot_id":"sha256:wrong"}', 0),
    ('{"ok":true,"snapshot_id":null}', 0),
    ('{"ok":true,"snapshot_id":"current"}', 0),
    ('{"ok":true,"snapshot_id":"sha256:wrong"}', 0),
    (json.dumps({"ok": True, "snapshot_id": "sha256:" + "z" * 64}), 0),
    ('not json', 0),
])
def test_failed_corrections_stop_before_forecast_with_truthful_status(
    tmp_path: Path, failed_output: str, command_exit: int,
) -> None:
    fake, log = _fake(tmp_path)
    source = json.dumps({"ok": True, "snapshot_id": SOURCE_ID})
    fake.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$CALL_LOG"\n'
        'if [ "$1" = "apply-corrections" ]; then\n'
        f"  printf '%s\\n' '{failed_output}'\n"
        f"  exit {command_exit}\n"
        "fi\n"
        f"echo '{source}'\n"
    )
    captured = tmp_path / "captured"
    (captured / "data").mkdir(parents=True)
    result = _run(_env(tmp_path, fake, log, SCVIA_CAPTURED_SOURCE=str(captured)))
    assert result.returncode != 0
    calls = log.read_text().splitlines()
    assert [c.split()[0] for c in calls] == ["import-legacy", "apply-corrections"]
    status = json.loads((tmp_path / "var" / "scvia-weekly-status.json").read_text())
    assert status["phase"] == "corrections" and status["exit_code"] == result.returncode
    assert status["snapshot_id"] == SOURCE_ID and status["source_snapshot_id"] == SOURCE_ID
    assert (Path(status["evidence_dir"]) / "corrections.json").read_text().strip() == failed_output


def test_rehearsal_without_a_captured_source_does_not_fetch(tmp_path: Path) -> None:
    fake, log = _fake(tmp_path)
    result = _run(_env(tmp_path, fake, log))
    assert result.returncode == 2
    assert "SCVIA_CAPTURED_SOURCE" in result.stderr
    assert not log.exists()


def test_rehearsal_without_a_cutoff_records_the_exit(tmp_path: Path) -> None:
    fake, log = _fake(tmp_path)
    captured = tmp_path / "captured"
    (captured / "data").mkdir(parents=True)
    env = _env(tmp_path, fake, log, SCVIA_CAPTURED_SOURCE=str(captured))
    del env["SCVIA_FORECAST_CUTOFF"]
    result = _run(env)
    assert result.returncode == 2
    assert "SCVIA_FORECAST_CUTOFF" in result.stderr
    assert not log.exists()
    status = json.loads((tmp_path / "var" / "scvia-weekly-status.json").read_text())
    assert status["exit_code"] == 2 and status["phase"] == "init"


def test_active_cycle_marker_is_a_recorded_refusal(tmp_path: Path) -> None:
    fake, log = _fake(tmp_path)
    env = _env(tmp_path, fake, log, SCVIA_CAPTURED_SOURCE=str(tmp_path))
    Path(env["SCVIA_CYCLE_MARKER"]).write_text('{"phase": "1"}\n')
    result = _run(env)
    assert result.returncode != 0
    status = json.loads((tmp_path / "var" / "scvia-weekly-status.json").read_text())
    assert status["exit_code"] == result.returncode
    assert status["exit_code"] is not None
    assert not log.exists()


def test_production_refresh_requires_an_explicit_network_opt_in(tmp_path: Path) -> None:
    fake, log = _fake(tmp_path)
    result = _run(_env(tmp_path, fake, log, SCVIA_SOURCE_MODE="production"))
    assert result.returncode == 2
    assert "SCVIA_ALLOW_NETWORK" in result.stderr
    assert not log.exists()


def _stop_tree(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
    else:
        proc.wait(timeout=5)


def _wait_phase(status_path: Path, phase: str) -> str:
    last = ""
    for _ in range(100):
        if status_path.exists():
            last = status_path.read_text()
            # The owner writes this marker in place; polling may catch a partial write.
            with contextlib.suppress(json.JSONDecodeError):
                if json.loads(last).get("phase") == phase:
                    return last
        time.sleep(0.05)
    raise AssertionError(f"owner status did not reach {phase}: {last}")


def test_a_second_run_leaves_the_locked_status_in_place(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    fake = tmp_path / "scvia"
    fake.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$CALL_LOG"\n'
        'if [ "$1" = "import-legacy" ]; then sleep 30; fi\n'
        'echo \'{"outputs":{},"snapshot_id":"sha256:pinned"}\'\n'
    )
    fake.chmod(0o755)
    var = tmp_path / "var"
    env = _env(tmp_path, fake, log, SCVIA_CAPTURED_SOURCE=str(tmp_path / "captured"))
    (tmp_path / "captured" / "data").mkdir(parents=True)
    env["SCVIA_VAR_DIR"] = str(var)
    first: subprocess.Popen[str] | None = None
    try:
        first = subprocess.Popen(
            ["bash", str(SCRIPT)], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        status_path = var / "scvia-weekly-status.json"
        original = _wait_phase(status_path, "import")
        second = _run(env)
        assert second.returncode == 5
        assert "locked" in second.stderr
        assert status_path.read_text() == original
    finally:
        _stop_tree(first)


def test_a_different_log_dir_cannot_use_a_locked_data_root(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    fake = tmp_path / "scvia"
    fake.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$CALL_LOG"\n'
        'if [ "$1" = "import-legacy" ]; then sleep 30; fi\n'
        'echo \'{"ok": true, "outputs":{},"snapshot_id":"sha256:pinned"}\'\n'
    )
    fake.chmod(0o755)
    captured = tmp_path / "captured"
    (captured / "data").mkdir(parents=True)
    env = _env(tmp_path, fake, log, SCVIA_CAPTURED_SOURCE=str(captured))
    first: subprocess.Popen[str] | None = None
    try:
        first = subprocess.Popen(
            ["bash", str(SCRIPT)], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,
        )
        status_path = tmp_path / "var" / "scvia-weekly-status.json"
        original = _wait_phase(status_path, "import")
        other = env.copy()
        other["SCVIA_VAR_DIR"] = str(tmp_path / "other-var")
        other["SCVIA_OUTPUT_ROOT"] = str(tmp_path / "other-out")
        second = subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, env=other, text=True, capture_output=True)
        assert second.returncode == 5
        assert "data root" in second.stderr
        assert status_path.read_text() == original
        assert not (tmp_path / "other-var" / "scvia-weekly-status.json").exists()
    finally:
        _stop_tree(first)


def test_a_failed_source_step_does_not_forecast(tmp_path: Path) -> None:
    log = tmp_path / "calls.log"
    fake = tmp_path / "scvia"
    fake.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$CALL_LOG"\n'
        'echo \'{"ok": false, "outputs":{}, "snapshot_id": null}\'\n'
    )
    fake.chmod(0o755)
    captured = tmp_path / "captured"
    (captured / "data").mkdir(parents=True)
    result = _run(_env(tmp_path, fake, log, SCVIA_CAPTURED_SOURCE=str(captured)))
    assert result.returncode == 1
    assert "snapshot id" in result.stderr
    assert "forecast" not in log.read_text()


def test_smoke_refuses_an_existing_root(tmp_path: Path) -> None:
    existing = tmp_path / "already"
    existing.mkdir()
    env = os.environ.copy()
    env["SCVIA_SMOKE_ROOT"] = str(existing)
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "smoke_scvia_candidate.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "already exists" in result.stderr
    assert existing.is_dir()


def test_opt_in_weekly_refresh_reaches_the_candidate(tmp_path: Path) -> None:
    fake, log = _fake(tmp_path)
    captured = tmp_path / "captured"
    (captured / "data").mkdir(parents=True)
    env = _env(tmp_path, fake, log, SCVIA_CAPTURED_SOURCE=str(captured), SCVIA_NUMERIC_ENTRY="1")
    result = subprocess.run(["bash", str(ROOT / "scripts" / "weekly_refresh.sh")], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "import-legacy" in log.read_text()
    assert "/home/abhi/sourceCode/python/coding/.venv/bin/python" not in result.stderr


def test_production_refresh_is_the_only_source_step(tmp_path: Path) -> None:
    fake, log = _fake(tmp_path)
    result = _run(_env(tmp_path, fake, log, SCVIA_SOURCE_MODE="production", SCVIA_ALLOW_NETWORK="1"))
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    assert calls[0].startswith("refresh ") and "--allow-network" in calls[0] and "--data-only" in calls[0]
    assert f"--data-root {tmp_path / 'data'}" in calls[0]
    assert calls[1].startswith("apply-corrections ") and "--season 2026" in calls[1]
    assert "--evidence-root" not in calls[1] and "--allow-network" not in calls[1]
    assert all(not line.startswith("import-legacy") for line in calls)
    status = json.loads((tmp_path / "var" / "scvia-weekly-status.json").read_text())
    assert status["mode"] == "production" and status["exit_code"] == 0


def test_archived_b1_repair_is_retired_by_default_and_passed_only_when_named(tmp_path: Path) -> None:
    """The B1 repair (2026 games of Perez, Dalton, Brodie) is fully contained in the legacy CSVs since the AFL Tables
    reconciliation corrections (commit 67217df40); replaying it on the corrected source fails its exact-reproduction
    check and would add Dalton under a second id. It is no longer the default; a rehearsal of an older,
    pre-correction capture names it explicitly with SCVIA_REPAIR."""
    fake, log = _fake(tmp_path)
    captured = tmp_path / "captured"
    (captured / "data").mkdir(parents=True)
    env = _env(tmp_path, fake, log, SCVIA_CAPTURED_SOURCE=str(captured))
    env.pop("SCVIA_REPAIR", None)
    result = _run(env)
    assert result.returncode == 0, result.stderr
    first = log.read_text().splitlines()[0]
    assert first.startswith("import-legacy ") and "--repair" not in first
    log.unlink()
    explicit = f"{ROOT}/docs/rewrite/evidence/b1:2026"
    (tmp_path / "x").mkdir()
    result = _run(_env(tmp_path / "x", fake, log, SCVIA_CAPTURED_SOURCE=str(captured), SCVIA_REPAIR=explicit))
    assert result.returncode == 0, result.stderr
    assert f"--repair {explicit}" in log.read_text().splitlines()[0]
