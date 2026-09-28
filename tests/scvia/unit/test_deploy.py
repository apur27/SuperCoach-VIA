"""Manual sealed-site deploy packs. No network and no host upload."""

from __future__ import annotations

import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from supercoach_via.publish import release as rel
from supercoach_via.publish.deploy import (
    DOWNLOAD_MAX_BYTES,
    MEASURED_PACK_BYTES,
    DeployError,
    pack_sealed_release,
    upload_verified,
    verify_sealed_bundle,
)
from tests.scvia.unit.test_release import _write_minimal


def _sealed(tmp_path: Path, release_id: str = "r-deploy") -> Path:
    rdir = _write_minimal(tmp_path / "dist", release_id)
    site = rdir / "site"
    shutil.copytree(rdir / "public", site / "data" / rdir.name)
    (site / "index.html").write_text("<h1>sealed</h1>")
    rel.write_seal(rdir)
    assert rel.validate_release(rdir).ok
    return rdir


def test_valid_bundle_reaches_a_stub_uploader_unchanged(tmp_path: Path) -> None:
    rdir = _sealed(tmp_path)
    bundle = tmp_path / "site.tar"
    seal = pack_sealed_release(rdir, bundle)
    seen: list[bytes] = []

    def upload(site: Path) -> None:
        seen.append((site / "index.html").read_bytes())

    site = verify_sealed_bundle(bundle, tmp_path / "out", expect_seal=seal, expect_archive=_digest(bundle))
    upload_verified(site, upload)
    assert seen == [b"<h1>sealed</h1>"]
    assert json.loads((rdir / "validation.json").read_text())["seal_sha256"] == seal


@pytest.mark.parametrize(
    "mutate",
    ["no_validation", "fail_outcome", "no_seal", "wrong_seal", "wrong_release", "changed", "extra", "missing"],
)
def test_verify_rejects_a_bundle_that_is_not_the_sealed_site(tmp_path: Path, mutate: str) -> None:
    rdir = _sealed(tmp_path, "r-bad")
    bundle = tmp_path / "site.tar"
    seal = pack_sealed_release(rdir, bundle)
    dest = tmp_path / "stage"
    if mutate == "no_validation":
        _repack(bundle, drop={"validation.json"})
    elif mutate == "fail_outcome":
        _rewrite_json(bundle, "validation.json", lambda doc: {**doc, "outcome": "FAIL"})
    elif mutate == "no_seal":
        _repack(bundle, drop={"seal.json"})
    elif mutate == "wrong_seal":
        _rewrite_json(bundle, "validation.json", lambda doc: {**doc, "seal_sha256": "0" * 64})
    elif mutate == "wrong_release":
        _rewrite_json(bundle, "validation.json", lambda doc: {**doc, "release_id": "other"})
    elif mutate == "changed":
        _replace_member(bundle, "site/index.html", b"<h1>changed</h1>")
    elif mutate == "extra":
        _replace_member(bundle, "site/extra.html", b"<h1>extra</h1>", add=True)
    else:
        _repack(bundle, drop={"site/index.html"})
    with pytest.raises(DeployError):
        verify_sealed_bundle(bundle, dest, expect_seal=seal, expect_archive=_digest(bundle))


def test_extract_refuses_symlink_and_traversal(tmp_path: Path) -> None:
    for name in ("../outside.txt", "site/link.html"):
        bundle = tmp_path / f"{name.replace('/', '_')}.tar"
        with tarfile.open(bundle, "w") as tar:
            if name.startswith(".."):
                info = tarfile.TarInfo(name)
                payload = b"x"
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            else:
                info = tarfile.TarInfo(name)
                info.type = tarfile.SYMTYPE
                info.linkname = "/tmp/secret"
                tar.addfile(info)
        with pytest.raises(DeployError):
            verify_sealed_bundle(
                bundle, tmp_path / "out" / name.replace("/", "_"), expect_seal="0" * 64, expect_archive=_digest(bundle)
            )


def _digest(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_download_bound_covers_the_measured_pack_and_the_site_budget_stays_300_mib() -> None:
    workflow = (Path(__file__).resolve().parents[3] / ".github/workflows/scvia-pages.yml").read_text()
    budget = (Path(__file__).resolve().parents[3] / "web/scripts/budget-lib.mjs").read_text()
    assert f"--max-filesize {DOWNLOAD_MAX_BYTES}" in workflow
    assert DOWNLOAD_MAX_BYTES >= MEASURED_PACK_BYTES
    assert "artifact_bytes: 300 * 1024 * KiB" in budget


def test_cli_verifies_into_a_directory_that_is_not_the_release_id(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    rdir = _sealed(tmp_path)
    bundle = tmp_path / "bundle.tar"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    packed = subprocess.run(
        [sys.executable, "-m", "supercoach_via.publish.deploy", "pack", str(rdir), str(bundle)],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    body = json.loads(packed.stdout)
    dest = tmp_path / "site"
    verified = subprocess.run(
        [
            sys.executable,
            "-m",
            "supercoach_via.publish.deploy",
            "verify",
            str(bundle),
            str(dest),
            "--expect-seal",
            body["seal_sha256"],
            "--expect-archive",
            body["archive_sha256"],
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert verified.returncode == 0
    assert dest.name == "site" and dest.name != rdir.name
    assert (dest / "site" / "index.html").read_bytes() == b"<h1>sealed</h1>"


def test_a_different_valid_bundle_does_not_match_the_selected_pack(tmp_path: Path) -> None:
    first = _sealed(tmp_path / "first", "r-first")
    second = _sealed(tmp_path / "second", "r-second")
    first_tar, second_tar = tmp_path / "first.tar", tmp_path / "second.tar"
    first_seal = pack_sealed_release(first, first_tar)
    pack_sealed_release(second, second_tar)
    with pytest.raises(DeployError, match="selected artifact"):
        verify_sealed_bundle(
            second_tar, tmp_path / "out", expect_seal=first_seal, expect_archive=_digest(second_tar)
        )
    with pytest.raises(DeployError, match="archive digest"):
        verify_sealed_bundle(
            second_tar, tmp_path / "other", expect_seal=first_seal, expect_archive=_digest(first_tar)
        )


def _repack(bundle: Path, *, drop: set[str]) -> None:
    kept: list[tuple[tarfile.TarInfo, bytes]] = []
    with tarfile.open(bundle, "r") as tar:
        for member in tar.getmembers():
            if member.name in drop or not member.isfile():
                continue
            kept.append((member, tar.extractfile(member).read()))
    with tarfile.open(bundle, "w") as tar:
        for member, payload in kept:
            info = tarfile.TarInfo(member.name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))


def _rewrite_json(bundle: Path, name: str, edit) -> None:
    def rewrite(member: tarfile.TarInfo, payload: bytes):
        if member.name != name:
            return None
        return json.dumps(edit(json.loads(payload))).encode()

    _repack_map(bundle, rewrite)


def _replace_member(bundle: Path, name: str, payload: bytes, *, add: bool = False) -> None:
    def edit(member: tarfile.TarInfo, current: bytes):
        if member.name == name:
            return payload
        return None

    _repack_map(bundle, edit, add=(name, payload) if add else None)


def _repack_map(bundle: Path, edit, add: tuple[str, bytes] | None = None) -> None:
    items: list[tuple[str, bytes]] = []
    with tarfile.open(bundle, "r") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            payload = tar.extractfile(member).read()
            replacement = edit(member, payload)
            items.append((member.name, payload if replacement is None else replacement))
    if add:
        items.append(add)
    with tarfile.open(bundle, "w") as tar:
        for name, payload in items:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
