"""The committed completion writer (scripts/reconciliation_completion.py, Surveyor B5): every step status is computed
from a real return code or real pytest JUnit counts, and the implementation status from those steps, never typed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "reconciliation_completion.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("recon_completion", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _junit(*cases: str) -> str:
    return '<?xml version="1.0"?><testsuites><testsuite name="pytest">' + "".join(cases) + "</testsuite></testsuites>"


OK = '<testcase classname="t.a" name="ok"/>'
FAIL = '<testcase classname="t.a" name="bad"><failure message="boom"/></testcase>'
ERR = '<testcase classname="t.b" name="err"><error message="fixture"/></testcase>'
SKIP_VENV = '<testcase classname="t.c" name="s{}"><skipped message="repo venv python not available"/></testcase>'


def test_counts_and_skip_reasons_come_from_the_junit_report() -> None:
    w = _load()
    xml = _junit(OK, OK, SKIP_VENV.format(1), SKIP_VENV.format(2))
    step = w.pytest_step("tier", "pytest x", xml, "2 passed, 2 skipped, 7 deselected in 1.0s")
    assert step.status == "PASS"
    assert step.counts.model_dump() == {"passed": 2, "failed": 0, "errors": 0, "skipped": 2, "deselected": 7}
    assert step.skip_reasons == {"repo venv python not available": 2}


def test_a_failure_or_an_error_is_never_recorded_as_pass() -> None:
    w = _load()
    assert w.pytest_step("tier", "pytest x", _junit(OK, FAIL), "").status == "FAIL"
    assert w.pytest_step("tier", "pytest x", _junit(OK, ERR), "").status == "FAIL"


def test_a_failure_is_baseline_only_with_evidence_for_exactly_those_tests() -> None:
    w = _load()
    base = {"t.a::bad": "fails identically at base commit 21ca71797 in this worktree (no web/node_modules)"}
    step = w.pytest_step("tier", "pytest x", _junit(OK, FAIL), "", baseline=base)
    assert step.status == "BASELINE_FAIL" and "21ca71797" in (step.baseline_evidence or "")
    # an additional, unexplained failure keeps the step FAIL
    assert w.pytest_step("tier", "pytest x", _junit(FAIL, ERR), "", baseline=base).status == "FAIL"


def test_a_command_step_is_pass_only_on_exit_zero() -> None:
    w = _load()
    assert w.command_step("ruff", "ruff check", 0, "All checks passed!\n").status == "PASS"
    bad = w.command_step("mypy", "mypy", 1, "x.py:1: error\nFound 1 error in 1 file\n")
    assert bad.status == "FAIL" and bad.detail == "Found 1 error in 1 file"


def test_the_implementation_status_and_data_verdicts_are_derived_not_typed(tmp_path: Path) -> None:
    w = _load()
    steps = [w.command_step("ruff", "ruff", 0, "ok"), w.pytest_step("tier", "pytest", _junit(FAIL), "")]
    iv = w.validation(steps, branch="b", base_commit="c")
    report = {"layers": {"snapshot": {"verdict": "UNKNOWN"}, "legacy_csv": {"verdict": "PASS"}}}
    impl, data, overall = w.derived_statuses(iv, report)
    assert impl == "INCOMPLETE"
    assert data == {"snapshot": "UNKNOWN", "legacy_csv": "PASS"} and overall == "UNKNOWN"
    report["layers"]["legacy_csv"]["verdict"] = "FAIL"
    assert w.derived_statuses(iv, report)[2] == "FAIL"


def test_a_malformed_junit_report_raises_instead_of_recording_zero_tests() -> None:
    w = _load()
    with pytest.raises(ValueError):
        w.pytest_step("tier", "pytest", "<html>not junit</html>", "")
    with pytest.raises(ValueError):
        w.pytest_step("tier", "pytest", _junit(), "")  # a tier that ran nothing is not a pass


def test_written_records_validate_against_the_repository_schema(tmp_path: Path) -> None:
    from supercoach_via.reconciliation import schema as S

    w = _load()
    iv = w.validation([w.command_step("ruff", "ruff", 0, "ok")], branch="b", base_commit="c")
    path = tmp_path / "implementation-validation.json"
    path.write_bytes(S.canonical_dump(iv))
    S.ImplementationValidation.model_validate(json.loads(path.read_text()))
