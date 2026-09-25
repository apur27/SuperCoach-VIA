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

    def reuse(self, table_name: str, ref: FragmentRef) -> None:
        """Reference an existing fragment unchanged (unchanged partitions are not rewritten)."""
        frags = self._tables.setdefault(table_name, [])
        if any(f.partition == ref.partition for f in frags):
            raise ValueError(f"duplicate partition {ref.partition!r} for table {table_name}")
        frags.append(ref)

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


def apply_upserts(
    data_root: Path,
    base: SnapshotManifest,
    upserts: dict[str, list[dict[str, Any]]],
    *,
    clock: Clock,
    code_version: str,
    status: DatasetStatus,
    run_id: str | None = None,
    source_revisions: dict[str, str] | None = None,
    notes: Iterable[str] = (),
) -> SnapshotCandidate:
    """New candidate = ``base`` with ``upserts`` replacing rows by table key.

    Only partitions (``TableSpec.partition_by``) that receive rows are read and rewritten;
    all other fragments are reused by reference. The base snapshot is never modified.
    Raises ``KeyError`` for an unknown table/column and ``ValueError`` for empty input or
    duplicate keys within the upserts.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from supercoach_via.domain.schemas import TABLES

    if not any(upserts.values()):
        raise ValueError("no upserts to apply")
    builder = SnapshotBuilder(data_root, clock=clock, code_version=code_version, run_id=run_id)
    touched = {name for name, rows in upserts.items() if rows}
    for name in sorted(touched):
        spec = TABLES[name]
        extra = {c for r in upserts[name] for c in r} - set(spec.column_names)
        if extra:
            raise KeyError(f"{name}: unknown columns {sorted(extra)}")
    for name, entry in sorted(base.tables.items()):
        if name not in touched:
            for frag in entry.fragments:
                builder.reuse(name, frag)
    for name in sorted(touched):
        spec = TABLES[name]
        schema = spec.arrow_schema()
        rows = upserts[name]
        keys = [tuple(r.get(k) for k in spec.key) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{name}: duplicate key within upserts")
        part_col = spec.partition_by

        def part_of(row: dict[str, Any], col: str | None = part_col) -> str | None:
            return None if col is None else str(row[col])

        groups: dict[str | None, list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault(part_of(r), []).append(r)
        base_entry = base.tables.get(name)
        existing: dict[str | None, list[FragmentRef]] = {}
        for frag in base_entry.fragments if base_entry else ():
            existing.setdefault(frag.partition if part_col else None, []).append(frag)
        for part, frags in sorted(existing.items(), key=lambda kv: kv[0] or ""):
            if part not in groups:
                for frag in frags:
                    builder.reuse(name, frag)
        for part, new_rows in sorted(groups.items(), key=lambda kv: kv[0] or ""):
            new_keys = {tuple(r.get(k) for k in spec.key) for r in new_rows}
            kept: list[dict[str, Any]] = []
            for frag in existing.get(part, []):
                table = pq.read_table(contained_path(data_root / "fragments", frag.path)).cast(schema)
                kept += [r for r in table.to_pylist() if tuple(r[k] for k in spec.key) not in new_keys]
            merged = kept + [{c: r.get(c) for c in spec.column_names} for r in new_rows]
            table = pa.Table.from_pylist(merged, schema=schema).sort_by([(k, "ascending") for k in spec.key])
            if part_col is not None and pc.sum(pc.is_null(table[part_col])).as_py():
                raise ValueError(f"{name}.{part_col} has null partition values")
            builder.add(name, table, partition=part)
    revisions = dict(base.source_revisions)
    revisions.update(source_revisions or {})
    return builder.finish(
        status=status, parent=base.snapshot_id, source_revisions=revisions, quality=dict(base.quality), notes=notes
    )


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
