"""Identical bytes and options -> identical canonical report bytes (repeat, relocation, workers, order, TZ)."""

from __future__ import annotations

import random
import shutil
import time
from pathlib import Path
from typing import Any

import pytest

from supercoach_via.integrity import capture
from supercoach_via.integrity.report import canonical_bytes
from supercoach_via.integrity.runner import AuditOptions, run_audit
from tests.scvia.unit import integrity_fixtures as fx


def _defective(r: dict[str, list[dict[str, Any]]]) -> None:
    # findings in several families so the comparison is not of an empty report
    r["player_games"][0].update(kicks=40, handballs=30, disposals=70)
    next(m for m in r["matches"] if m["match_id"] == "m:2026:r02:alpha:beta:0").update(attendance=5)


def _report(root: Path, **kw: Any) -> bytes:
    opts = dict(data_root=root, scope="data", as_of=fx.AS_OF)
    opts.update(kw)
    return canonical_bytes(run_audit(AuditOptions(**opts)).report)  # type: ignore[arg-type]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    fx.with_sources(tmp_path / "a" / "var", mutate=_defective)
    return tmp_path / "a" / "var"


def test_repeat_runs_are_byte_identical(corpus: Path) -> None:
    first = _report(corpus)
    assert b'"outcome":"FAIL"' in first
    assert _report(corpus) == first


def test_relocated_directory_gives_the_same_report(corpus: Path, tmp_path: Path) -> None:
    moved = tmp_path / "elsewhere" / "deeper" / "var"
    shutil.copytree(corpus, moved)
    assert _report(moved) == _report(corpus)


@pytest.fixture
def demo(integrity_demo: Any) -> tuple[Path, dict[str, Any]]:
    cand_dir = integrity_demo.release_dir
    return integrity_demo.data_root, dict(
        release_dir=cand_dir, scope="full", as_of="2026-05-03T00:00:00Z", families=("release",)
    )


def test_directory_iteration_order_does_not_matter(
    demo: tuple[Path, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    data, kw = demo
    want = _report(data, **kw)
    real = capture._list_dir
    calls = []

    def shuffled(path: Path) -> list[Any]:
        calls.append(1)
        entries = real(path)
        random.Random(7).shuffle(entries)
        return entries

    monkeypatch.setattr(capture, "_list_dir", shuffled)
    assert _report(data, **kw) == want
    assert calls  # the release walk really went through the shuffled iterator


def test_host_time_zone_does_not_matter(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    want = _report(corpus)
    monkeypatch.setenv("TZ", "Australia/Melbourne")
    time.tzset()
    try:
        assert _report(corpus) == want
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


def test_physically_reordered_rows_change_identity_but_not_findings(tmp_path: Path) -> None:
    def shuffled(r: dict[str, list[dict[str, Any]]]) -> None:
        _defective(r)
        for rows in r.values():
            random.Random(3).shuffle(rows)

    fx.with_sources(tmp_path / "x", mutate=_defective)
    fx.with_sources(tmp_path / "y", mutate=shuffled)
    a = run_audit(AuditOptions(data_root=tmp_path / "x", scope="data", as_of=fx.AS_OF)).report
    b = run_audit(AuditOptions(data_root=tmp_path / "y", scope="data", as_of=fx.AS_OF)).report
    assert a["inputs"]["snapshot"]["snapshot_id"] != b["inputs"]["snapshot"]["snapshot_id"]
    assert b["inputs"]["snapshot"]["snapshot_id"] == capture.SnapshotCapture.open(tmp_path / "y", "current").snapshot_id

    def normal(rep: dict[str, Any]) -> Any:
        return [{k: v for k, v in f.items() if k != "evidence"} for f in rep["findings"]], rep["outcome"], rep["rules"]

    assert normal(a) == normal(b)


def test_worker_count_does_not_change_the_report(demo: tuple[Path, dict[str, Any]]) -> None:
    data, kw = demo
    assert _report(data, workers=1, **kw) == _report(data, workers=3, **kw)
