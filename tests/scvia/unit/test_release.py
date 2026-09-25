"""Release staging, validation, publication and rollback (R05, R08-R10, S04)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from supercoach_via.domain.schemas import CheckOutcome
from supercoach_via.publish import release as rel
from supercoach_via.publish.view_models import (
    ArticleIndex,
    Downloads,
    QualityReport,
)

NOW = datetime(2026, 9, 23, 11, 0, tzinfo=UTC)
SNAP = "sha256:" + "a" * 64


def _write_minimal(out: Path, release_id: str = "20260923T110000Z-demo") -> Path:
    w = rel.ReleaseWriter(out, release_id)
    w.put_json("articles/index.json", ArticleIndex(articles=[]))
    w.put_json(
        "quality.json",
        QualityReport(
            dataset_status="demo", table_counts={}, quarantined_rows=0, issues=[], limitations=["demo"], sources=[]
        ),
    )
    w.put_bytes("downloads/players.csv", b"id,name\n1,Demo\n")
    items = [w.download_item("players_csv", "Players (all rows)", "downloads/players.csv", as_of="demo", rows=1)]
    w.put_json("downloads.json", Downloads(release_id=release_id, items=items, retained=[]))
    return w.finish(
        snapshot_id=SNAP,
        generated_at=NOW,
        season=2026,
        demo=True,
        coverage_status="demo",
        coverage_through=None,
        forecast_status="unavailable",
        forecast_reason="no_valid_future_fixture",
        forecast_artifact=None,
        index_resources={
            "article_index": "articles/index.json",
            "quality": "quality.json",
            "downloads": "downloads.json",
        },
    )


def test_release_validates_when_complete(tmp_path: Path) -> None:
    rdir = _write_minimal(tmp_path)
    report = rel.validate_release(rdir)
    assert report.outcome is CheckOutcome.PASS, report.issues
    manifest = json.loads((rdir / "public" / "release.json").read_text())
    assert set(manifest["resources"]) == {"article_index", "quality", "downloads"}
    assert (rdir / "validation.json").exists()


def test_staging_is_invisible_until_finished(tmp_path: Path) -> None:
    w = rel.ReleaseWriter(tmp_path, "r-partial")
    w.put_bytes("downloads/x.csv", b"a\n")
    assert not (tmp_path / "releases" / "r-partial").exists()
    assert rel.list_releases(tmp_path) == []


@pytest.mark.parametrize(
    "mutation",
    ["tamper", "extra", "missing", "nan", "schema", "forbidden", "private_path"],
)
def test_validation_rejects_bad_release(tmp_path: Path, mutation: str) -> None:
    rdir = _write_minimal(tmp_path)
    pub = rdir / "public"
    if mutation == "tamper":
        (pub / "downloads" / "players.csv").write_bytes(b"id,name\n1,Evil\n")
    elif mutation == "extra":
        (pub / "stray.json").write_text("{}")
    elif mutation == "missing":
        (pub / "quality.json").unlink()
    elif mutation == "nan":
        _rewrite_with_checksum(
            rdir,
            "quality.json",
            b'{"dataset_status":"demo","issues":[],"limitations":[],"quarantined_rows":NaN,"sources":[],"table_counts":{}}\n',
        )
    elif mutation == "schema":
        _rewrite_with_checksum(rdir, "quality.json", b'{"dataset_status":"bogus"}\n')
    elif mutation == "forbidden":
        _rewrite_with_checksum(rdir, ".env", b"TOKEN=x\n")
    elif mutation == "private_path":
        _rewrite_with_checksum(rdir, "downloads/players.csv", b"id,name\n1,/home/abhi/secret\n")
    assert rel.validate_release(rdir).outcome is CheckOutcome.FAIL


def _rewrite_with_checksum(rdir: Path, relpath: str, data: bytes) -> None:
    """Simulate an attacker/bug that also updates checksums (so only content rules catch it)."""
    import hashlib

    target = rdir / "public" / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    sums = json.loads((rdir / "checksums.json").read_text())
    sums["files"][relpath] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    (rdir / "checksums.json").write_text(json.dumps(sums))


def test_publish_refuses_unvalidated_release(tmp_path: Path) -> None:
    rdir = _write_minimal(tmp_path)
    dest = rel.LocalDirectoryDestination("local", tmp_path / "site")
    with pytest.raises(rel.PublishError):
        rel.publish_release(rdir, dest, clock=lambda: NOW)


def test_publish_then_failed_publish_keeps_previous_and_rollback(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    r1 = _write_minimal(out, "r1")
    r2 = _write_minimal(out, "r2")
    for r in (r1, r2):
        assert rel.validate_release(r).ok
    dest = rel.LocalDirectoryDestination("local", tmp_path / "site")
    receipt1 = rel.publish_release(r1, dest, clock=lambda: NOW)
    assert receipt1.status == "published" and dest.active_release() == "r1"

    class Boom(rel.LocalDirectoryDestination):
        def _activate(self, release_id: str) -> None:
            raise OSError("host rejected upload")

    bad = Boom("local", tmp_path / "site")
    with pytest.raises(rel.PublishError):
        rel.publish_release(r2, bad, clock=lambda: NOW)
    assert dest.active_release() == "r1"
    receipts = sorted((out / "receipts").glob("*.json"))
    assert any(json.loads(p.read_text())["status"] == "failed" for p in receipts)

    rel.publish_release(r2, dest, clock=lambda: NOW)
    assert dest.active_release() == "r2"
    back = rel.rollback(out, dest, "r1", clock=lambda: NOW)
    assert back.status == "published" and back.kind == "rollback" and dest.active_release() == "r1"


def test_publish_never_runs_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    calls: list[object] = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: calls.append(a))
    r = _write_minimal(tmp_path / "dist", "r1")
    rel.validate_release(r)
    rel.publish_release(r, rel.LocalDirectoryDestination("local", tmp_path / "site"), clock=lambda: NOW)
    assert calls == []


@pytest.mark.parametrize("bad", ["../x", "/abs", "a/../../b", "x/.claude/y", "models/m.pkl"])
def test_writer_rejects_unsafe_paths(tmp_path: Path, bad: str) -> None:
    w = rel.ReleaseWriter(tmp_path, "r")
    with pytest.raises(ValueError):
        w.put_bytes(bad, b"x")


@pytest.mark.parametrize("bad", ["", "../r", "r/1", "a b"])
def test_release_id_is_validated(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        rel.ReleaseWriter(tmp_path, bad)
