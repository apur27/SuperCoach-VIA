"""Content-keyed cache of parsed source facts (DESIGN section 10).

An entry is keyed by the SHA-256 of the page body AND a parser identity (hash of the reader code,
label map and notes aliases), so a changed parser or rule invalidates exactly what depends on it.
Each file starts with a JSON line carrying the key parts and the digest of its payload; a corrupt,
truncated or mismatching file is simply a miss (recomputed), and no cache hit can authorise a PASS:
a hit is reused evidence derived from a frozen body, never a fresh source fetch.

Parsed facts are cached here; whole season results are cached by ``compare.UnitCache`` under the unit's
input digest plus a salt over the import closure of the comparison code (``code_digest``).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from functools import cache
from pathlib import Path

from supercoach_via.integrity.report import canonical_bytes
from supercoach_via.reconciliation.schema import LABEL_TO_FIELD

FACTS_VERSION = 1
_PKG = Path(__file__).resolve().parents[1]  # .../supercoach_via
#: entry points whose code (and everything they import from this package) defines what a parsed fact means
PARSER_ENTRY = ("reconciliation/facts.py",)


def _file_bytes(rel: str) -> bytes:
    return (_PKG / rel).read_bytes()


@cache
def _imports(rel: str) -> tuple[str, ...]:
    """Package-relative files that ``rel`` imports from ``supercoach_via`` (top level and inside functions)."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(_file_bytes(rel))):
        names: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("supercoach_via"):
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names if a.name.startswith("supercoach_via")]
        for name in names:
            parts = name.split(".")[1:]
            for cand in ("/".join(parts) + ".py", "/".join(parts) + "/__init__.py"):
                if parts and (_PKG / cand).is_file():
                    out.add(cand)
    return tuple(sorted(out))


def code_closure(entries: tuple[str, ...]) -> tuple[str, ...]:
    """Every package file reachable from ``entries`` through imports, sorted (DESIGN section 10, F-M2).

    A cache key that hashes this closure is invalidated by an edit to ANY module that can shape the cached
    payload, so a hand-maintained file list can never silently fall behind the code."""
    seen: set[str] = set()
    todo = list(entries)
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        todo.extend(_imports(rel))
    return tuple(sorted(seen))


def code_digest(entries: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for rel in code_closure(entries):
        h.update(rel.encode() + b"\0" + hashlib.sha256(_file_bytes(rel)).digest())
    return h.hexdigest()


def parser_files() -> tuple[str, ...]:
    return code_closure(PARSER_ENTRY)


def parser_identity(extra: str = "") -> str:
    """Twelve hex digits identifying the readers, the label map and ``extra`` (e.g. notes aliases)."""
    h = hashlib.sha256()
    h.update(str(FACTS_VERSION).encode())
    for rel in parser_files():
        h.update(rel.encode() + b"\0" + hashlib.sha256(_file_bytes(rel)).digest())
    h.update(json.dumps(sorted(LABEL_TO_FIELD.items())).encode())
    h.update(extra.encode())
    return h.hexdigest()[:12]


class FactsCache:
    def __init__(self, root: Path, phash: str) -> None:
        self.root = root
        self.phash = phash
        self.hits = 0
        self.misses = 0
        self.corrupt = 0

    def path(self, kind: str, sha: str, part: str = "main") -> Path:
        return self.root / "facts" / kind / sha[:2] / f"{sha}.{self.phash}.{part}.json"

    def read(self, kind: str, sha: str, part: str = "main") -> bytes | None:
        p = self.path(kind, sha, part)
        try:
            raw = p.read_bytes()
        except FileNotFoundError:
            self.misses += 1
            return None
        head, sep, payload = raw.partition(b"\n")
        try:
            meta = json.loads(head)
        except ValueError:
            meta = None
        ok = (
            sep == b"\n"
            and isinstance(meta, dict)
            and meta.get("body") == sha
            and meta.get("phash") == self.phash
            and meta.get("part") == part
            and meta.get("digest") == hashlib.sha256(payload).hexdigest()
        )
        if not ok:
            self.corrupt += 1
            self.misses += 1
            return None
        self.hits += 1
        return payload

    def write(self, kind: str, sha: str, payload: bytes, part: str = "main") -> None:
        p = self.path(kind, sha, part)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = canonical_bytes(
            {"body": sha, "phash": self.phash, "part": part, "digest": hashlib.sha256(payload).hexdigest()}
        ).rstrip(b"\n")
        tmp = p.with_name(p.name + f".{os.getpid()}.tmp")
        tmp.write_bytes(meta + b"\n" + payload)
        os.replace(tmp, p)

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "corrupt": self.corrupt}
