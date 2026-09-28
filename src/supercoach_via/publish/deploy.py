"""Pack and verify a sealed site for a manual deploy.

A clean checkout cannot see ``var/`` or ``/tmp``. The operator packs a validated
release and publishes that archive as an HTTPS URL (for example a GitHub release
asset). The deploy job downloads that URL and uploads the verified ``site/``
bytes. It does not build the site again.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from supercoach_via.publish.release import (
    _seal_problems_from_bytes,
    _strict_json,
    embedded_data_problems,
)

MAX_MEMBERS = 200_000
MAX_BYTES = 400 * 1024 * 1024
# The 2026-09-27 final site is 276,795,445 content bytes and packed to
# 479,068,160 tar bytes (headers and 512-byte blocks). The download bound
# covers that pack. The site budget stays 300 MiB.
MEASURED_PACK_BYTES = 479_068_160
DOWNLOAD_MAX_BYTES = 600 * 1024 * 1024


class DeployError(ValueError):
    """The bundle is not the sealed site the validation record names."""


def pack_sealed_release(release_dir: Path, dest: Path) -> str:
    """Write a tar of ``site/``, ``seal.json`` and ``validation.json``. Return the seal."""
    validation = _read_validation(release_dir / "validation.json")
    seal_path = release_dir / "seal.json"
    if not seal_path.is_file() or seal_path.is_symlink():
        raise DeployError("seal.json is missing")
    raw_seal = seal_path.read_bytes()
    problems, seal_hash = _seal_problems_from_bytes(release_dir, raw_seal)
    if problems or seal_hash != validation["seal_sha256"]:
        detail = problems[0] if problems else "validation is not bound to the seal"
        raise DeployError(detail)
    if validation["release_id"] != release_dir.name:
        raise DeployError("validation release_id does not match the release directory")
    try:
        sums = _strict_json((release_dir / "checksums.json").read_bytes())["files"]
    except (OSError, ValueError, KeyError) as exc:
        raise DeployError(f"checksums unreadable: {exc}") from exc
    embedded = embedded_data_problems(release_dir, sums)
    if embedded:
        raise DeployError(embedded[0])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dest, "w") as tar:
        tar.add(release_dir / "validation.json", arcname="validation.json")
        tar.add(seal_path, arcname="seal.json")
        tar.add(release_dir / "site", arcname="site")
    return str(seal_hash)


def verify_sealed_bundle(
    bundle: Path, dest: Path, *, expect_seal: str, expect_archive: str | None = None
) -> Path:
    """Extract a pack into ``dest`` and return ``dest/site`` when every byte matches.

    ``expect_seal`` and ``expect_archive`` come from the operator's pack result.
    A different bundle that is internally consistent still fails.
    """
    if expect_archive is not None and _sha256_file(bundle) != expect_archive:
        raise DeployError("archive digest does not match the packed artifact")
    dest.mkdir(parents=True, exist_ok=True)
    _extract_bounded(bundle, dest)
    validation_path = dest / "validation.json"
    seal_path = dest / "seal.json"
    site = dest / "site"
    validation = _read_validation(validation_path)
    if not seal_path.is_file() or seal_path.is_symlink():
        raise DeployError("seal.json is missing")
    if not site.is_dir():
        raise DeployError("sealed site is missing")
    raw_seal = seal_path.read_bytes()
    problems, seal_hash = _seal_problems_from_bytes(dest, raw_seal, release_id=str(validation["release_id"]))
    if seal_hash != expect_seal:
        raise DeployError("seal does not match the selected artifact")
    if problems or seal_hash != validation["seal_sha256"]:
        detail = problems[0] if problems else "validation is not bound to the seal"
        raise DeployError(detail)
    seal = _strict_json(raw_seal)
    if seal.get("release_id") != validation["release_id"]:
        raise DeployError("seal release_id does not match validation")
    return site


def upload_verified(site: Path, uploader: Callable[[Path], None]) -> None:
    """Hand the verified site directory to the caller. This function does not upload."""
    if not site.is_dir():
        raise DeployError("sealed site is missing")
    uploader(site)


def _read_validation(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise DeployError("validation.json is missing")
    try:
        record = _strict_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise DeployError(f"validation.json unreadable: {exc}") from exc
    if not isinstance(record, dict):
        raise DeployError("validation.json unreadable: not an object")
    if record.get("outcome") != "PASS":
        raise DeployError(f"release validation outcome is {record.get('outcome')}")
    if not record.get("seal_sha256"):
        raise DeployError("validation has no seal")
    if not record.get("release_id"):
        raise DeployError("validation has no release_id")
    return record


def _extract_bounded(bundle: Path, dest: Path) -> None:
    try:
        with tarfile.open(bundle, "r:*") as tar:
            _extract_members(tar, dest)
    except DeployError:
        raise
    except (tarfile.TarError, OSError) as exc:
        raise DeployError(f"bundle unreadable: {exc}") from exc


def _extract_members(tar: tarfile.TarFile, dest: Path) -> None:
    members = tar.getmembers()
    if len(members) > MAX_MEMBERS:
        raise DeployError("bundle has too many members")
    total = 0
    safe: list[tarfile.TarInfo] = []
    root = dest.resolve()
    allowed = {"site", "validation.json", "seal.json"}
    for member in members:
        name = member.name
        if member.issym() or member.islnk() or member.isdev():
            raise DeployError(f"refused archive member {name}")
        if not member.isdir() and not member.isfile():
            raise DeployError(f"refused archive member {name}")
        parts = Path(name).parts
        if not parts or name.startswith("/") or ".." in parts or parts[0] not in allowed:
            raise DeployError(f"refused archive path {name}")
        if parts[0] == "site" and any(part.startswith(".") for part in parts):
            raise DeployError(f"refused archive path {name}")
        target = (dest / name).resolve()
        if target != root and root not in target.parents:
            raise DeployError(f"refused archive path {name}")
        if member.isfile():
            total += member.size
            if total > MAX_BYTES:
                raise DeployError("bundle exceeds the size bound")
        safe.append(member)
    tar.extractall(dest, members=safe, filter="data")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str]) -> None:
    usage = (
        "usage: python -m supercoach_via.publish.deploy pack RELEASE DEST\n"
        "       python -m supercoach_via.publish.deploy verify BUNDLE DEST "
        "--expect-seal HEX --expect-archive HEX"
    )
    if len(argv) >= 2 and argv[1] == "pack" and len(argv) == 4:
        dest = Path(argv[3])
        seal = pack_sealed_release(Path(argv[2]), dest)
        print(json.dumps({"seal_sha256": seal, "archive_sha256": _sha256_file(dest)}))
        return
    if len(argv) == 8 and argv[1] == "verify" and argv[4] == "--expect-seal" and argv[6] == "--expect-archive":
        verify_sealed_bundle(Path(argv[2]), Path(argv[3]), expect_seal=argv[5], expect_archive=argv[7])
        return
    raise SystemExit(usage)


if __name__ == "__main__":
    main(sys.argv)
