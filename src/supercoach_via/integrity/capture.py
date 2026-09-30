"""Pinned, read-only views of the checker's inputs.

Every input file is read once into memory, hashed, and parsed from those same bytes, so
the checks see one consistent view even if a file is replaced during the scan. After the
checks, :meth:`SnapshotCapture.drift` and :meth:`ReleaseCapture.drift` re-read the files
and report anything that changed; that goes to execution metadata and marks the result.

Nothing here writes. Paths are resolved against their roots and never followed out of
them (symlinks are recorded as problems, not followed).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from supercoach_via.domain.schemas import TABLES, CurrentPointer, FragmentRef, SnapshotManifest
from supercoach_via.storage.snapshots import ContainmentError, contained_path, semantic_snapshot_id, snapshot_hex

if TYPE_CHECKING:  # pragma: no cover
    import pyarrow as pa

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_regular(path: Path) -> tuple[bytes | None, str | None]:
    """Bytes of a regular, non-symlink file, or (None, reason)."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None, "missing"
    except OSError as exc:
        return None, f"unreadable: {exc.strerror}"
    if stat.S_ISLNK(st.st_mode):
        return None, "symlink refused"
    if not stat.S_ISREG(st.st_mode):
        return None, "not a regular file"
    try:
        with path.open("rb") as fh:
            return fh.read(), None
    except OSError as exc:
        return None, f"unreadable: {exc.strerror}"


def _list_dir(path: Path) -> list[os.DirEntry[str]]:
    """Directory entries in whatever order the filesystem returns them (callers sort)."""
    with os.scandir(path) as it:
        return list(it)


def strict_json(data: bytes) -> Any:
    """JSON with non-finite constants and duplicate object keys refused."""

    def bad_constant(name: str) -> Any:
        raise ValueError(f"non-finite JSON constant {name}")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in items:
            if k in out:
                raise ValueError(f"duplicate JSON key {k!r}")
            out[k] = v
        return out

    return json.loads(data.decode("utf-8"), parse_constant=bad_constant, object_pairs_hook=pairs)


class CapturedInputError(OSError):
    """A selected input was unavailable at capture time (logical name only)."""


class ExternalCapture:
    """Selected auxiliary files, with parsing and identity bound to the same bytes.

    Opaque model payloads and content-addressed evidence need only a streamed hash;
    JSON, TOML, YAML and Parquet inputs retain their bytes for semantic checks.
    """

    def __init__(self) -> None:
        self.files: dict[Path, tuple[bytes | None, str | None, str | None]] = {}
        self.groups: dict[str, dict[str, Path]] = {}
        self.selections: dict[tuple[Path, str, bool], tuple[Path, ...]] = {}
        self.boundaries: dict[Path, Path] = {}

    @staticmethod
    def _hash(path: Path) -> tuple[str | None, str | None]:
        try:
            if path.is_symlink() or not path.is_file():
                return None, "missing or non-regular file"
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest(), None
        except OSError as exc:
            return None, f"unreadable: {exc.strerror}"

    def _contained(self, path: Path) -> bool:
        root = self.boundaries.get(path)
        if root is None:
            return True
        try:
            contained_path(root, path.relative_to(root).as_posix())
        except (ContainmentError, ValueError):
            return False
        return True

    def pin(self, path: Path, group: str, name: str, *, retain: bool = True, root: Path | None = None) -> None:
        self.groups.setdefault(group, {})[name] = path
        if root is not None:
            self.boundaries[path] = root
        if path not in self.files:
            if not self._contained(path):
                self.files[path] = (None, None, "path escapes input root")
            elif retain:
                raw, why = _read_regular(path)
                self.files[path] = (raw, sha256_hex(raw) if raw is not None else None, why)
            else:
                digest, why = self._hash(path)
                self.files[path] = (None, digest, why)

    def read(self, path: Path) -> bytes:
        raw, _digest, why = self.files[path]
        if raw is None:
            label = next(
                f"{group}/{name}" for group, paths in sorted(self.groups.items())
                for name, selected in sorted(paths.items()) if selected == path
            )
            raise CapturedInputError(f"captured input {label}: {why or 'bytes not retained'}")
        return raw

    def get(self, path: Path) -> bytes | None:
        return self.files[path][0]

    def digest(self, path: Path) -> str | None:
        return self.files[path][1]

    @staticmethod
    def _select(root: Path, pattern: str, directories: bool) -> tuple[Path, ...]:
        if not root.is_dir() or root.is_symlink():
            return ()
        paths = root.glob(pattern)
        if directories:
            paths = (p for p in paths if p.is_dir() and not p.is_symlink() and not p.name.startswith("."))
        return tuple(sorted(paths))

    def select(self, root: Path, pattern: str, *, directories: bool = False) -> tuple[Path, ...]:
        key = (root, pattern, directories)
        if key not in self.selections:
            self.selections[key] = self._select(*key)
        return self.selections[key]

    def identity(self) -> dict[str, str]:
        # Logical names only: relocating identical inputs preserves report identity.
        return {
            group: sha256_hex(json.dumps(
                {name: self.files[path][1:] for name, path in sorted(paths.items())},
                sort_keys=True, separators=(",", ":"),
            ).encode())
            for group, paths in sorted(self.groups.items())
        }

    def drift(self) -> list[str]:
        out = []
        for (root, pattern, directories), before in self.selections.items():
            if self._select(root, pattern, directories) != before:
                out.append(f"selected inputs in {root} ({pattern}) changed during the audit")
        for path, (_raw, digest, _why) in self.files.items():
            if not self._contained(path):
                if digest is not None:
                    out.append(f"input {path} escaped its root during the audit")
            elif self._hash(path)[0] != digest:
                out.append(f"input {path} changed during the audit")
        return sorted(out)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass
class CapturedFragment:
    table: str
    ref: FragmentRef
    data: bytes | None
    problem: str | None  # missing / symlink / escape / size / hash
    num_rows: int | None = None
    schema: pa.Schema | None = None
    parquet_error: str | None = None
    partition_values: list[Any] | None = None

    @property
    def entity(self) -> str:
        return f"fragment:{self.table}" + (f"/{self.ref.partition}" if self.ref.partition is not None else "")

    @property
    def verified(self) -> bool:
        return self.data is not None and self.problem is None and self.parquet_error is None


@dataclass
class SnapshotCapture:
    data_root: Path
    selector: str
    pointer_raw: bytes | None = None
    pointer: CurrentPointer | None = None
    manifest_raw: bytes | None = None
    manifest_path: Path | None = None
    manifest: SnapshotManifest | None = None
    snapshot_id: str | None = None
    #: (rule_id, entity, message) problems found while pinning the pointer and manifest
    problems: list[tuple[str, str, str]] = field(default_factory=list)
    fragments: list[CapturedFragment] = field(default_factory=list)

    @classmethod
    def open(cls, data_root: Path, selector: str) -> SnapshotCapture:
        cap = cls(data_root=data_root, selector=selector)
        wanted = selector
        if selector == "current":
            raw, why = _read_regular(data_root / "current.json")
            if raw is None:
                cap.problems.append(("storage.pointer", "pointer:current", f"current.json {why}"))
                return cap
            cap.pointer_raw = raw
            try:
                cap.pointer = CurrentPointer.model_validate(strict_json(raw))
            except ValueError as exc:
                cap.problems.append(("storage.pointer", "pointer:current", f"current.json invalid: {exc}"[:300]))
                return cap
            wanted = cap.pointer.snapshot_id
        try:
            hexid = snapshot_hex(wanted)
        except Exception as exc:  # noqa: BLE001 - reported as a finding, never a crash
            cap.problems.append(("storage.pointer", "pointer:current", f"malformed snapshot id: {exc}"[:300]))
            return cap
        rel = f"snapshots/{hexid}.json"
        cap.manifest_path = data_root / rel
        if cap.pointer is not None and cap.pointer.manifest_path != rel:
            cap.problems.append(
                (
                    "storage.pointer",
                    "pointer:current",
                    f"manifest_path {cap.pointer.manifest_path!r} is not {rel!r} for {wanted}",
                )
            )
        raw, why = _read_regular(data_root / rel)
        if raw is None:
            cap.problems.append(("storage.manifest_missing", f"manifest:{wanted}", f"{rel} {why}"))
            return cap
        cap.manifest_raw = raw
        try:
            manifest = SnapshotManifest.model_validate(strict_json(raw))
        except ValueError as exc:
            cap.problems.append(("storage.manifest_invalid", f"manifest:{wanted}", str(exc)[:300]))
            return cap
        if manifest.snapshot_id != wanted:
            cap.problems.append(
                ("storage.manifest_identity", f"manifest:{wanted}", f"manifest names {manifest.snapshot_id}")
            )
        if semantic_snapshot_id(manifest) != manifest.snapshot_id:
            cap.problems.append(
                ("storage.manifest_identity", f"manifest:{wanted}", "manifest content does not hash to its snapshot id")
            )
        cap.manifest = manifest
        cap.snapshot_id = manifest.snapshot_id
        for name, entry in sorted(manifest.tables.items()):
            for ref in sorted(entry.fragments, key=lambda f: (f.partition or "", f.path)):
                cap.fragments.append(cap._capture_fragment(name, ref))
        return cap

    def _capture_fragment(self, table: str, ref: FragmentRef) -> CapturedFragment:
        import pyarrow as pa
        import pyarrow.parquet as pq

        base = self.data_root / "fragments"
        try:
            contained_path(base, ref.path)
        except ContainmentError as exc:
            return CapturedFragment(table, ref, None, f"path escapes the fragment store: {exc}")
        data, why = _read_regular(base / ref.path)
        if data is None:
            return CapturedFragment(table, ref, None, why)
        frag = CapturedFragment(table, ref, data, None)
        if len(data) != ref.bytes:
            frag.problem = f"size {len(data)} != declared {ref.bytes}"
        elif sha256_hex(data) != ref.sha256:
            frag.problem = "content does not hash to the declared sha256"
        try:
            pf = pq.ParquetFile(pa.BufferReader(data))
            frag.num_rows = pf.metadata.num_rows
            frag.schema = pf.schema_arrow
            spec = TABLES.get(table)
            if spec is not None and spec.partition_by and spec.partition_by in frag.schema.names:
                col = pq.read_table(pa.BufferReader(data), columns=[spec.partition_by]).column(0)
                frag.partition_values = sorted({v for v in col.to_pylist()}, key=lambda v: (v is None, str(v)))
        except Exception as exc:  # noqa: BLE001 - any parse failure is a finding
            frag.parquet_error = f"{type(exc).__name__}: {exc}"[:300]
        return frag

    # -- data access --------------------------------------------------------

    def table(self, name: str, columns: list[str] | None = None) -> pa.Table | None:
        """The verified fragments of ``name`` as one Arrow table cast to the contract schema.

        ``None`` when the table is absent, a fragment failed verification, or the stored
        schema cannot be cast (the storage checks report why).
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        spec = TABLES.get(name)
        frags = [f for f in self.fragments if f.table == name]
        if spec is None or not frags or not all(f.verified for f in frags):
            return None
        # nullability is checked from the data (contract.null_required), so cast leniently;
        # strings stay dictionary-encoded (a repeated id or path is stored once in memory)
        def compact(field: pa.Field) -> pa.Field:
            kind = pa.dictionary(pa.int32(), pa.string()) if pa.types.is_string(field.type) else field.type
            return pa.field(field.name, kind, nullable=True)

        want = pa.schema([compact(f) for f in spec.arrow_schema()])
        if columns is not None:
            want = pa.schema([want.field(c) for c in columns])
        strings = [f.name for f in want if pa.types.is_dictionary(f.type)]
        parts = []
        for f in frags:
            assert f.data is not None
            try:
                t = pq.read_table(pa.BufferReader(f.data), columns=columns, read_dictionary=strings)
                parts.append(t.select(want.names).cast(want))
            except Exception:  # noqa: BLE001 - schema problems are reported by the storage checks
                return None
        return pa.concat_tables(parts) if len(parts) > 1 else parts[0]

    def partition_bytes(self, name: str, partitions: set[str] | None) -> list[tuple[str, bytes]]:
        """Verified (sha256, bytes) of ``name``'s fragments, optionally limited to partitions."""
        out = []
        for f in self.fragments:
            if f.table == name and f.verified and (partitions is None or f.ref.partition in partitions):
                assert f.data is not None
                out.append((f.ref.sha256, f.data))
        return out

    def fragment_digest(self, name: str, partitions: set[str] | None = None) -> str:
        """Content identity of a table (or some partitions): hash over its fragment hashes."""
        refs = sorted(
            f"{f.ref.partition or ''}:{f.ref.sha256}"
            for f in self.fragments
            if f.table == name and (partitions is None or f.ref.partition in partitions)
        )
        return sha256_hex("\n".join(refs).encode())

    def identity(self) -> dict[str, Any]:
        tables = sorted({f.table for f in self.fragments})
        return {
            "selector": self.selector,
            "snapshot_id": self.snapshot_id,
            "manifest_sha256": sha256_hex(self.manifest_raw) if self.manifest_raw is not None else None,
            "pointer_sha256": sha256_hex(self.pointer_raw) if self.pointer_raw is not None else None,
            "status": self.manifest.status.value if self.manifest else None,
            "tables": len(tables),
            "fragments": len(self.fragments),
            "bytes": sum(f.ref.bytes for f in self.fragments),
            "rows": sum(e.row_count for e in self.manifest.tables.values()) if self.manifest else 0,
            # content identity per partition: what changed-data mode compares
            "partitions": {
                t: {f.ref.partition or "": f.ref.sha256 for f in self.fragments if f.table == t} for t in tables
            },
        }

    def drift(self) -> list[str]:
        """Inputs whose bytes changed on disk since they were pinned (execution metadata)."""
        out = []
        if self.selector == "current":
            raw, _ = _read_regular(self.data_root / "current.json")
            if raw != self.pointer_raw:
                out.append("current.json changed during the audit")
        if self.manifest_path is not None:
            raw, _ = _read_regular(self.manifest_path)
            if raw != self.manifest_raw:
                out.append("snapshot manifest changed during the audit")
        for f in self.fragments:
            base = self.data_root / "fragments"
            try:
                contained_path(base, f.ref.path)
            except ContainmentError:
                if f.data is not None:
                    out.append(f"fragment {f.ref.path} escaped its root during the audit")
                continue
            raw, _ = _read_regular(base / f.ref.path)
            if raw != f.data:
                out.append(f"fragment {f.ref.path} changed during the audit")
        return out


# ---------------------------------------------------------------------------
# Content-addressed source evidence
# ---------------------------------------------------------------------------


class EvidenceStore:
    """Archived source payloads addressed by SHA-256 (``raw/objects`` plus extra directories).

    A payload counts only if its (decompressed) bytes hash to the requested digest. The
    set of digests requested and found is part of the report's input identity.
    """

    def __init__(self, roots: list[Path]):
        self.roots = roots
        self.requested: dict[str, bool] = {}
        self.capture = ExternalCapture()

    def pin(self, digests: set[str]) -> None:
        for digest in sorted(digests):
            if _HEX64.match(digest or ""):
                for i, path in enumerate(self._candidates(digest)):
                    self.capture.pin(path, "evidence", f"{digest}:{i}", retain=False, root=self.roots[i // 5])

    def _candidates(self, digest: str) -> Iterator[Path]:
        for root in self.roots:
            yield root / "objects" / digest[:2] / digest
            for name in (digest, f"{digest}.html", f"{digest}.html.gz", f"{digest}.gz"):
                yield root / name

    def get(self, digest: str) -> bytes | None:
        if not _HEX64.match(digest or ""):
            return None
        self.pin({digest})
        for path in self._candidates(digest):
            if not self.capture._contained(path):
                continue
            raw, _ = _read_regular(path)
            if raw is None or sha256_hex(raw) != self.capture.digest(path):
                continue
            if path.suffix == ".gz":
                try:
                    raw = gzip.decompress(raw)
                except (OSError, EOFError):
                    continue
            if sha256_hex(raw) == digest:
                self.requested[digest] = True
                return raw
        self.requested[digest] = False
        return None

    def drift(self) -> list[str]:
        return self.capture.drift()

    def identity(self) -> dict[str, Any]:
        found = sorted(d for d, ok in self.requested.items() if ok)
        missing = sorted(d for d, ok in self.requested.items() if not ok)
        return {
            "found": len(found),
            "missing": len(missing),
            "captured": self.capture.identity(),
            "digest": sha256_hex(
                ("\n".join(f"+{d}" for d in found) + "\n" + "\n".join(f"-{d}" for d in missing)).encode()
            ),
        }


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------


@dataclass
class TreeFile:
    sha256: str
    bytes: int


class DriftError(RuntimeError):
    """A release file changed between the inventory pass and a semantic read."""


@dataclass
class ReleaseCapture:
    release_dir: Path
    release_id: str
    checksums_raw: bytes | None = None
    seal_raw: bytes | None = None
    validation_raw: bytes | None = None
    checksums: dict[str, Any] | None = None
    seal: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    public: dict[str, TreeFile] = field(default_factory=dict)
    site: dict[str, TreeFile] = field(default_factory=dict)
    #: (tree, relative path, problem) for symlinks and non-regular files
    tree_problems: list[tuple[str, str, str]] = field(default_factory=list)
    meta_problems: list[tuple[str, str, str]] = field(default_factory=list)
    has_site: bool = False

    @classmethod
    def open(cls, release_dir: Path) -> ReleaseCapture:
        cap = cls(release_dir=release_dir, release_id=release_dir.name)
        for attr, name in (("checksums", "checksums.json"), ("seal", "seal.json"), ("validation", "validation.json")):
            raw, why = _read_regular(release_dir / name)
            setattr(cap, f"{attr}_raw", raw)
            if raw is None:
                if not (attr == "seal" and why == "missing"):
                    cap.meta_problems.append((attr, name, why or "unreadable"))
                continue
            try:
                doc = strict_json(raw)
                if not isinstance(doc, dict):
                    raise ValueError("not a JSON object")
                setattr(cap, attr, doc)
            except ValueError as exc:
                cap.meta_problems.append((attr, name, f"invalid JSON: {exc}"[:300]))
        cap.public = cap._walk("public")
        site = release_dir / "site"
        cap.has_site = os.path.lexists(site)
        if cap.has_site:
            cap.site = cap._walk("site")
        return cap

    def _walk(self, tree: str) -> dict[str, TreeFile]:
        base = self.release_dir / tree
        out: dict[str, TreeFile] = {}
        try:
            st = os.lstat(base)
        except FileNotFoundError:
            self.tree_problems.append((tree, "", "directory is missing"))
            return out
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            self.tree_problems.append((tree, "", "not a real directory"))
            return out
        stack = [""]
        while stack:
            rel_dir = stack.pop()
            entries = sorted(_list_dir(base / rel_dir), key=lambda e: e.name)
            for e in entries:
                rel = f"{rel_dir}/{e.name}" if rel_dir else e.name
                if e.is_symlink():
                    self.tree_problems.append((tree, rel, "symlink refused"))
                elif e.is_dir(follow_symlinks=False):
                    stack.append(rel)
                elif e.is_file(follow_symlinks=False):
                    data, why = _read_regular(base / rel)
                    if data is None:
                        self.tree_problems.append((tree, rel, why or "unreadable"))
                    else:
                        out[rel] = TreeFile(sha256_hex(data), len(data))
                else:
                    self.tree_problems.append((tree, rel, "non-regular file refused"))
        return dict(sorted(out.items()))

    def read_public(self, rel: str) -> bytes:
        """Bytes of a public file, verified against the inventory pass (consistent view)."""
        info = self.public.get(rel)
        if info is None:
            raise KeyError(rel)
        data, why = _read_regular(self.release_dir / "public" / rel)
        if data is None or len(data) != info.bytes or sha256_hex(data) != info.sha256:
            raise DriftError(f"public/{rel} changed during the audit ({why or 'bytes differ'})")
        return data

    def identity(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "checksums_sha256": sha256_hex(self.checksums_raw) if self.checksums_raw is not None else None,
            "seal_sha256": (self.seal or {}).get("seal_sha256"),
            "validation_sha256": sha256_hex(self.validation_raw) if self.validation_raw is not None else None,
            "public_files": len(self.public),
            "site_files": len(self.site),
            "bytes": sum(f.bytes for f in self.public.values()) + sum(f.bytes for f in self.site.values()),
            "inventory_sha256": sha256_hex(
                "\n".join(
                    f"{t}/{p}:{f.sha256}:{f.bytes}"
                    for t, tree in (("public", self.public), ("site", self.site))
                    for p, f in tree.items()
                ).encode()
            ),
        }

    def drift(self) -> list[str]:
        out = []
        for name, raw in (
            ("checksums.json", self.checksums_raw),
            ("seal.json", self.seal_raw),
            ("validation.json", self.validation_raw),
        ):
            now, _ = _read_regular(self.release_dir / name)
            if now != raw:
                out.append(f"{name} changed during the audit")
        after = ReleaseCapture(self.release_dir, self.release_id)
        if os.path.lexists(self.release_dir / "site") != self.has_site:
            out.append("site directory changed during the audit")
        for tree, files in (("public", self.public), ("site", self.site)):
            if tree == "site" and not self.has_site:
                continue
            tree_now = after._walk(tree)
            for rel in sorted(set(files) | set(tree_now)):
                if files.get(rel) != tree_now.get(rel):
                    out.append(f"{tree}/{rel} changed during the audit")
                    if len(out) > 50:
                        return out
        if after.tree_problems != self.tree_problems:
            out.append("release tree file types changed during the audit")
        return out
