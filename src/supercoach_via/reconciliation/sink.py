"""Findings are streamed to disk, never held for the whole run (DESIGN section 10).

Each unit (a season of a layer, the source facts, identity, matches, aggregates) writes one chunk:
its findings sorted by the canonical key, as canonical JSON lines. The final stream is a k-way merge
of the chunks in the same key order, so the output is identical for any worker count or traversal
order and memory stays bounded however many findings a real audit produces.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from supercoach_via.integrity.report import canonical_bytes
from supercoach_via.reconciliation.findings import sort_key


@dataclass
class ChunkInfo:
    path: str
    count: int
    #: (layer, category, severity) -> n
    by_cat: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    #: (layer, source url, severity) -> n
    by_player: Counter[tuple[str, str, str]] = field(default_factory=Counter)


def write_chunk(directory: Path, name: str, findings: list[dict[str, Any]]) -> ChunkInfo:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.jsonl"
    info = ChunkInfo(str(path), len(findings))
    with path.open("wb") as fh:
        for f in sorted(findings, key=sort_key):
            fh.write(canonical_bytes(f))
            info.by_cat[(f["layer"], f["category"], f["severity"])] += 1
            url = f["player"].get("source_url")
            if url:
                info.by_player[(f["layer"], url, f["severity"])] += 1
    return info


def _records(path: str) -> Iterator[tuple[tuple[Any, ...], bytes]]:
    with open(path, "rb") as fh:  # noqa: PTH123 - plain streaming read
        for line in fh:
            yield sort_key(json.loads(line)), line


def merge_chunks(chunks: list[ChunkInfo]) -> Iterator[bytes]:
    for _key, line in heapq.merge(*[_records(c.path) for c in chunks if c.count], key=lambda kv: kv[0]):
        yield line


def merge_to_file(chunks: list[ChunkInfo], target: Path) -> tuple[int, str]:
    """Write the merged stream to ``target``; returns (count, sha256)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    n = 0
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("wb") as fh:
        for line in merge_chunks(chunks):
            fh.write(line)
            h.update(line)
            n += 1
    tmp.replace(target)
    return n, h.hexdigest()
