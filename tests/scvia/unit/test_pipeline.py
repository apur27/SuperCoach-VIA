"""Staged pipeline (PLAN 5.3, 13 R01-R04, R08-R10): state machine, lock, pointer safety, resume.

Hermetic: the DEMO corpus under tmp_path; no network, no Git.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from supercoach_via import pipeline
from supercoach_via.demo import write_demo_corpus
from supercoach_via.domain.schemas import CheckOutcome, RunState, ValidationReport
from supercoach_via.settings import RunContext, Settings
from supercoach_via.storage import runs
from supercoach_via.storage.snapshots import read_current

NOW = datetime(2026, 5, 1, 6, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    src = tmp_path_factory.mktemp("demo-src")
    write_demo_corpus(src)
    return src


def _ctx(tmp_path: Path, corpus: Path) -> RunContext:
    return RunContext(
        settings=Settings(data_root=tmp_path / "var", output_root=tmp_path / "dist", source_root=corpus),
        clock=lambda: NOW,
    )


def test_ingest_imports_validates_and_promotes(tmp_path: Path, corpus: Path) -> None:
    ctx = _ctx(tmp_path, corpus)
    res = pipeline.ingest(ctx, source_root=corpus)
    assert res.exit_code == 0 and res.state is RunState.DATASET_PROMOTED
    cur = read_current(tmp_path / "var")
    assert cur is not None and cur.snapshot_id == res.snapshot_id
    run = json.loads((tmp_path / "var" / "runs" / res.run_id / "run.json").read_text())
    assert [h[0] for h in run["history"]] == ["created", "planned", "parsed", "validated", "dataset_promoted"]
    assert set(run["steps"]) >= {"import", "validate", "promote"}
    assert (tmp_path / "var" / "runs" / res.run_id / "import-report.json").is_file()
    assert (tmp_path / "var" / "runs" / res.run_id / "validation-report.json").is_file()


def test_validation_failure_keeps_previous_pointer(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx(tmp_path, corpus)
    first = pipeline.ingest(ctx, source_root=corpus)
    before = (tmp_path / "var" / "current.json").read_bytes()
    monkeypatch.setattr(
        pipeline, "_validate", lambda *_a, **_k: ValidationReport(outcome=CheckOutcome.FAIL, issues=[{"x": 1}])
    )
    res = pipeline.ingest(ctx, source_root=corpus)
    assert res.exit_code == 4 and res.state is RunState.FAILED
    assert (tmp_path / "var" / "current.json").read_bytes() == before
    assert res.snapshot_id != first.snapshot_id or res.promoted is False
    run = json.loads((tmp_path / "var" / "runs" / res.run_id / "run.json").read_text())
    assert run["error_code"] == "validation_failed" and run["recovery"]


def test_import_crash_is_a_failed_run_not_a_success(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx(tmp_path, corpus)

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(pipeline, "_import", boom)
    res = pipeline.ingest(ctx, source_root=corpus)
    assert res.exit_code != 0 and res.state is RunState.FAILED
    assert read_current(tmp_path / "var") is None


def test_competing_writer_is_locked_out(tmp_path: Path, corpus: Path) -> None:
    ctx = _ctx(tmp_path, corpus)
    (tmp_path / "var").mkdir(parents=True)
    with runs.WriterLock(tmp_path / "var"):
        res = pipeline.ingest(ctx, source_root=corpus)
    assert res.exit_code == 5 and res.state is None


def test_rerun_is_idempotent(tmp_path: Path, corpus: Path) -> None:
    ctx = _ctx(tmp_path, corpus)
    a = pipeline.ingest(ctx, source_root=corpus)
    b = pipeline.ingest(ctx, source_root=corpus)
    assert a.snapshot_id == b.snapshot_id and a.run_id != b.run_id


def test_pipeline_never_touches_git(tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def spy(*a: object, **_k: object) -> None:
        calls.append(a)
        raise AssertionError("pipeline spawned a subprocess")

    monkeypatch.setattr(subprocess, "run", spy)
    monkeypatch.setattr(subprocess, "Popen", spy)
    pipeline.ingest(_ctx(tmp_path, corpus), source_root=corpus)
    assert not calls
