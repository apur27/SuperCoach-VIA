"""Immutable Parquet fragments, snapshot manifests and atomic promotion.

Layout under a data root::

    fragments/<table>/<sha[:2]>/<sha>.parquet   content-addressed, write-once
    snapshots/<hex>.json                         SnapshotManifest (immutable)
    current.json                                 CurrentPointer (atomically replaced)

A snapshot ID is ``sha256:`` over the manifest's semantic content (tables, status,
sources, quality, parent) — not its creation time — so re-importing identical inputs
yields the identical snapshot. Unchanged partitions produce byte-identical fragments and
are reused rather than rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from supercoach_via.domain.schemas import (
    CurrentPointer,
    DatasetStatus,
    FragmentRef,
    SnapshotManifest,
    TableEntry,
    ValidationReport,
)

if TYPE_CHECKING:  # pragma: no cover
    import pyarrow as pa

Clock = Callable[[], datetime]


class ContainmentError(ValueError):
    """A path escaped its configured root."""


class IntegrityError(RuntimeError):
    """A manifest, fragment or pointer failed hash/shape verification."""


class PromotionError(RuntimeError):
    """A candidate could not be promoted (e.g. failed validation)."""


# ---------------------------------------------------------------------------
# Filesystem primitives
# ---------------------------------------------------------------------------


def contained_path(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root``; reject absolute, traversal and symlink escape."""
    if not relative or relative.startswith(("/", "\\")) or "\\" in relative:
        raise ContainmentError(f"not a contained relative path: {relative!r}")
    parts = relative.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise ContainmentError(f"not a contained relative path: {relative!r}")
    base = root.resolve()
    candidate = (base / relative).resolve()
    if candidate != base and base not in candidate.parents:
        raise ContainmentError(f"path escapes root: {relative!r}")
    return candidate


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform without directory fds
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(target: Path, data: bytes, *, mode: int = 0o644) -> None:
    """Write ``data`` to ``target`` via same-directory temp file + fsync + os.replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.chmod(mode)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(target.parent)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


# ---------------------------------------------------------------------------
# Fragment store
# ---------------------------------------------------------------------------


def write_fragment(data_root: Path, table_name: str, table: pa.Table, partition: str | None) -> FragmentRef:
    """Write ``table`` as a content-addressed Parquet fragment (reused if it exists)."""
    import pyarrow.parquet as pq

    staging = data_root / "fragments" / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".parquet", dir=staging)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        pq.write_table(
            table,
            tmp,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
            store_schema=True,
        )
        digest = sha256_file(tmp)
        rel = f"{table_name}/{digest[:2]}/{digest}.parquet"
        final = data_root / "fragments" / rel
        size = tmp.stat().st_size
        if final.exists():
            if sha256_file(final) != digest:
                raise IntegrityError(f"existing fragment {rel} does not match its name")
            tmp.unlink()
        else:
            final.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp, final)
            final.chmod(0o444)
            _fsync_dir(final.parent)
    finally:
        tmp.unlink(missing_ok=True)
    return FragmentRef(path=rel, sha256=digest, rows=table.num_rows, bytes=size, partition=partition)


# ---------------------------------------------------------------------------
# Snapshot building / promotion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotCandidate:
    manifest: SnapshotManifest
    manifest_path: Path


@dataclass(frozen=True)
class SnapshotRef:
    snapshot_id: str
    manifest_path: Path


def snapshot_hex(snapshot_id: str) -> str:
    if not snapshot_id.startswith("sha256:") or len(snapshot_id) != 71:
        raise IntegrityError(f"malformed snapshot id {snapshot_id!r}")
    return snapshot_id.split(":", 1)[1]


class SnapshotBuilder:
    """Accumulates table fragments and writes an immutable snapshot manifest."""

    def __init__(self, data_root: Path, *, clock: Clock, code_version: str, run_id: str | None = None):
        self.data_root = data_root
        self.clock = clock
        self.code_version = code_version
        self.run_id = run_id
        self._tables: dict[str, list[FragmentRef]] = {}

    def add(self, table_name: str, table: pa.Table, partition: str | None = None) -> FragmentRef:
        ref = write_fragment(self.data_root, table_name, table, partition)
        frags = self._tables.setdefault(table_name, [])
        if any(f.partition == partition for f in frags):
            raise ValueError(f"duplicate partition {partition!r} for table {table_name}")
        frags.append(ref)
        return ref

    def add_partitioned(self, table_name: str, table: pa.Table, column: str) -> list[FragmentRef]:
        """Split ``table`` by distinct values of ``column`` into one fragment each."""
        import pyarrow.compute as pc

        refs: list[FragmentRef] = []
        values = sorted(v for v in pc.unique(table[column]).to_pylist() if v is not None)
        for value in values:
            part = table.filter(pc.equal(table[column], value))
            refs.append(self.add(table_name, part, partition=str(value)))
        if table.num_rows and pc.sum(pc.is_null(table[column])).as_py():
            raise ValueError(f"{table_name}.{column} has null partition values")
        return refs

    def finish(
        self,
        *,
        status: DatasetStatus,
        parent: str | None = None,
        source_revisions: dict[str, str] | None = None,
        quality: dict[str, int] | None = None,
        notes: Iterable[str] = (),
    ) -> SnapshotCandidate:
        tables = {
            name: TableEntry(
                row_count=sum(f.rows for f in frags),
                fragments=tuple(sorted(frags, key=lambda f: (f.partition or "", f.path))),
            )
            for name, frags in sorted(self._tables.items())
        }
        semantic = {
            "tables": {k: v.model_dump(mode="json") for k, v in tables.items()},
            "status": status.value,
            "parent": parent,
            "source_revisions": source_revisions or {},
            "quality": quality or {},
            "notes": list(notes),
        }
        digest = hashlib.sha256(_canonical(semantic)).hexdigest()
        manifest = SnapshotManifest(
            snapshot_id=f"sha256:{digest}",
            created_at=self.clock(),
            parent=parent,
            status=status,
            run_id=self.run_id,
            code_version=self.code_version,
            tables=tables,
            source_revisions=source_revisions or {},
            quality=quality or {},
            notes=tuple(notes),
        )
        path = self.data_root / "snapshots" / f"{digest}.json"
        if not path.exists():
            atomic_write_bytes(path, manifest.model_dump_json(indent=1).encode())
        return SnapshotCandidate(manifest=manifest, manifest_path=path)


def promote(
    data_root: Path,
    candidate: SnapshotCandidate,
    report: ValidationReport,
    *,
    promoted_at: datetime | None = None,
) -> SnapshotRef:
    """Atomically point ``current.json`` at ``candidate`` after a passing validation."""
    if not report.ok:
        raise PromotionError(f"validation outcome {report.outcome.value}; refusing promotion")
    verify_manifest(data_root, candidate.manifest)
    pointer = CurrentPointer(
        snapshot_id=candidate.manifest.snapshot_id,
        manifest_path=f"snapshots/{snapshot_hex(candidate.manifest.snapshot_id)}.json",
        promoted_at=promoted_at or candidate.manifest.created_at,
        run_id=candidate.manifest.run_id,
    )
    atomic_write_bytes(data_root / "current.json", pointer.model_dump_json(indent=1).encode())
    return SnapshotRef(candidate.manifest.snapshot_id, candidate.manifest_path)


def read_current(data_root: Path) -> CurrentPointer | None:
    path = data_root / "current.json"
    if not path.exists():
        return None
    return CurrentPointer.model_validate_json(path.read_bytes())


def verify_manifest(data_root: Path, manifest: SnapshotManifest, *, hashes: bool = True) -> None:
    for name, entry in manifest.tables.items():
        if sum(f.rows for f in entry.fragments) != entry.row_count:
            raise IntegrityError(f"{name}: fragment rows do not sum to row_count")
        for frag in entry.fragments:
            path = contained_path(data_root / "fragments", frag.path)
            if not path.is_file():
                raise IntegrityError(f"{name}: missing fragment {frag.path}")
            if path.stat().st_size != frag.bytes:
                raise IntegrityError(f"{name}: size mismatch for {frag.path}")
            if hashes and sha256_file(path) != frag.sha256:
                raise IntegrityError(f"{name}: hash mismatch for {frag.path}")


def load_snapshot(data_root: Path, selector: str = "current", *, verify: bool = False) -> SnapshotManifest:
    """Load a manifest by ``current`` or ``sha256:<hex>``; optionally verify every fragment."""
    if selector == "current":
        pointer = read_current(data_root)
        if pointer is None:
            raise FileNotFoundError(f"no current snapshot under {data_root}")
        selector = pointer.snapshot_id
    path = data_root / "snapshots" / f"{snapshot_hex(selector)}.json"
    if not path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {selector}")
    manifest = SnapshotManifest.model_validate_json(path.read_bytes())
    if manifest.snapshot_id != selector:
        raise IntegrityError("manifest snapshot_id does not match its selector")
    semantic = {
        "tables": {k: v.model_dump(mode="json") for k, v in sorted(manifest.tables.items())},
        "status": manifest.status.value,
        "parent": manifest.parent,
        "source_revisions": manifest.source_revisions,
        "quality": manifest.quality,
        "notes": list(manifest.notes),
    }
    if f"sha256:{hashlib.sha256(_canonical(semantic)).hexdigest()}" != manifest.snapshot_id:
        raise IntegrityError("manifest content does not hash to its snapshot id")
    verify_manifest(data_root, manifest, hashes=verify)
    return manifest
