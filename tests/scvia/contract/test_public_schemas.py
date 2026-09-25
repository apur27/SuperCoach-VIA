"""Public JSON schema generation contract (Phase 1 gate: schema roundtrip)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from supercoach_via.publish.view_models import PUBLIC_MODELS, ReleaseManifest, ResourceRef
from supercoach_via.publish.web_data import export_json_schemas


def test_export_writes_one_schema_per_public_model(tmp_path: Path) -> None:
    written = export_json_schemas(tmp_path)
    assert set(written) == set(PUBLIC_MODELS)
    for key, path in written.items():
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["title"] == PUBLIC_MODELS[key].__name__
        assert doc.get("$id", "").endswith(f"{key}.schema.json")


def test_export_is_deterministic(tmp_path: Path) -> None:
    a = export_json_schemas(tmp_path / "a")
    b = export_json_schemas(tmp_path / "b")
    for key in a:
        assert a[key].read_bytes() == b[key].read_bytes()


def test_checked_in_schemas_are_current(repo_root: Path, tmp_path: Path) -> None:
    """schemas/ must be regenerated whenever a view model changes (drift gate)."""
    fresh = export_json_schemas(tmp_path)
    for path in fresh.values():
        committed = repo_root / "schemas" / path.name
        assert committed.exists(), f"missing generated schema {committed}"
        assert committed.read_bytes() == path.read_bytes(), f"stale schema {path.name}"


@pytest.mark.parametrize(
    "bad",
    ["../x.json", "/abs.json", "https://evil.example/x.json", "a\\b.json", "//host/x", ""],
)
def test_resource_ref_rejects_uncontained_paths(bad: str) -> None:
    with pytest.raises(ValidationError):
        ResourceRef(path=bad, sha256="0" * 64, bytes=1)


def test_release_manifest_rejects_nan_and_unsafe_id() -> None:
    base = {
        "release_id": "r1",
        "snapshot_id": "sha256:" + "0" * 64,
        "generated_at": "2026-09-23T11:00:00Z",
        "season": 2026,
        "demo": True,
        "base_label": "DEMO",
        "coverage": {"status": "demo", "through": None},
        "forecast": {"status": "unavailable", "reason": "no_valid_future_fixture", "artifact": None},
        "resources": {},
    }
    ReleaseManifest.model_validate(base)
    with pytest.raises(ValidationError):
        ReleaseManifest.model_validate({**base, "release_id": "../etc"})
