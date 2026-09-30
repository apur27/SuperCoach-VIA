"""Auxiliary input selection, evidence stability and byte-backed configuration."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import pytest

from supercoach_via.integrity import checks_source
from supercoach_via.integrity.capture import EvidenceStore, ExternalCapture, ReleaseCapture, SnapshotCapture
from supercoach_via.integrity.runner import AuditOptions, run_audit
from supercoach_via.settings import default_config_dir
from tests.scvia.unit import integrity_fixtures as fx


def test_captured_missing_file_appearing_is_drift(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    capture = ExternalCapture()
    capture.pin(path, "input", path.name)
    before = capture.identity()
    path.write_text("{}")
    assert capture.get(path) is None
    assert capture.identity() == before
    assert capture.drift()


@pytest.mark.parametrize("name", ["../outside.json", "/outside.json", "linked/outside.json"])
def test_manifest_paths_are_contained_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    from supercoach_via.integrity import capture as module

    root = tmp_path / "input"
    root.mkdir()
    (tmp_path / "outside.json").write_text("private input")
    (root / "linked").symlink_to(tmp_path, target_is_directory=True)
    calls = []
    original = module._read_regular

    def record(path: Path) -> tuple[bytes | None, str | None]:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(module, "_read_regular", record)
    capture = ExternalCapture()
    path = root / name
    capture.pin(path, "input", name, root=root)
    assert capture.get(path) is None
    assert capture.digest(path) is None
    assert capture.drift() == []
    assert calls == []


def test_added_release_resource_is_drift(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    (public / "existing.json").write_text("{}")
    capture = ReleaseCapture.open(tmp_path)
    (public / "added.json").write_text("{}")
    assert capture.drift()


def test_missing_snapshot_manifest_appearing_is_drift(tmp_path: Path) -> None:
    selector = "sha256:" + "a" * 64
    capture = SnapshotCapture.open(tmp_path, selector)
    manifests = tmp_path / "snapshots"
    manifests.mkdir()
    (manifests / ("a" * 64 + ".json")).write_text("{}")
    assert capture.drift()


def test_evidence_bytes_are_verified_against_capture_and_missing_candidates_tracked(tmp_path: Path) -> None:
    raw = b"archived source"
    digest = hashlib.sha256(raw).hexdigest()
    path = tmp_path / digest
    path.write_bytes(raw)
    evidence = EvidenceStore([tmp_path])
    evidence.pin({digest})
    assert evidence.get(digest) == raw
    path.write_bytes(b"replaced")
    assert evidence.get(digest) is None
    assert evidence.drift()


@pytest.mark.parametrize("change", ["replace", "delete", "add_candidate"])
def test_source_evidence_drift_makes_audit_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    fx.with_sources(tmp_path)
    original = next(s.fn for s in checks_source.CHECKS if s.check_id == "source.match_pages")

    def mutate(ctx: Any) -> list[str]:
        result = original(ctx)
        path = next(p for p in ctx.evidence.capture.files if p.is_file())
        if change == "delete":
            path.unlink()
        elif change == "replace":
            path.write_bytes(b"replaced source")
        else:
            digest = next(iter(ctx.evidence.requested))
            (tmp_path / "raw" / f"{digest}.html").write_bytes(b"new candidate")
        return result

    monkeypatch.setattr(checks_source, "CHECKS", [
        dataclasses.replace(s, fn=mutate) if s.check_id == "source.match_pages" else s
        for s in checks_source.CHECKS
    ])
    result = run_audit(AuditOptions(
        data_root=tmp_path, scope="data", as_of=fx.AS_OF, checks=("source.match_pages",),
    ))
    assert next(c["status"] for c in result.report["checks"] if c["check_id"] == "inputs.stable") == "UNKNOWN"
    assert result.outcome.value == "UNKNOWN"
    assert result.execution["input_drift"]


def test_config_loaders_parse_supplied_bytes_without_rereading_path(tmp_path: Path) -> None:
    from supercoach_via.analytics.rankings import RankingConfig
    from supercoach_via.ingest.reconcile import load_policy

    config = default_config_dir()
    missing = tmp_path / "missing"
    assert load_policy(missing, captured_bytes=(config / "coverage.yaml").read_bytes()) == load_policy(config)
    assert RankingConfig.load(missing, captured_bytes=(config / "ranking_legacy_v1.toml").read_bytes()) == RankingConfig.load(config / "ranking_legacy_v1.toml")
