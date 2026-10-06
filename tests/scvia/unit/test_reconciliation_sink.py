"""Reconciliation findings sink: per-unit sorted chunks and the deterministic k-way merge."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from supercoach_via.integrity.report import canonical_bytes
from supercoach_via.reconciliation.findings import sort_key
from supercoach_via.reconciliation.sink import ChunkInfo, merge_to_file, write_chunk


def _finding(
    fid: str,
    *,
    layer: str = "player",
    category: str = "value_mismatch",
    severity: str = "blocking",
    player_url: str | None = "https://afltables.com/p/a.html",
    season: int | None = 2020,
    field: str | None = "goals",
) -> dict[str, Any]:
    player: dict[str, Any] = {"local_id": f"local-{fid}"}
    if player_url is not None:
        player["source_url"] = player_url
    return {
        "id": fid,
        "layer": layer,
        "category": category,
        "severity": severity,
        "player": player,
        "season": season,
        "match": {},
        "field": field,
    }


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def test_write_chunk_sorts_findings_and_counts_categories(tmp_path: Path) -> None:
    findings = [
        _finding("c", season=2021),
        _finding("a", layer="match", category="missing", severity="warning", player_url=None),
        _finding("b", season=2019),
    ]
    info = write_chunk(tmp_path / "chunks", "season-2020", findings)

    assert isinstance(info, ChunkInfo)
    path = tmp_path / "chunks" / "season-2020.jsonl"
    assert info.path == str(path)
    assert info.count == 3
    # Canonical bytes, in canonical key order.
    assert path.read_bytes() == b"".join(canonical_bytes(f) for f in sorted(findings, key=sort_key))
    assert [f["id"] for f in _lines(path)] == ["a", "b", "c"]
    assert info.by_cat == Counter({("player", "value_mismatch", "blocking"): 2, ("match", "missing", "warning"): 1})
    # The finding without a player source url is not counted per player.
    assert info.by_player == Counter({("player", "https://afltables.com/p/a.html", "blocking"): 2})


def test_merge_to_file_interleaves_chunks_in_key_order(tmp_path: Path) -> None:
    all_findings = [_finding(f"f{i}", season=2000 + i) for i in range(9)]
    chunks = [
        write_chunk(tmp_path / "c", "one", all_findings[0::3]),
        write_chunk(tmp_path / "c", "two", all_findings[1::3]),
        write_chunk(tmp_path / "c", "three", all_findings[2::3]),
    ]
    target = tmp_path / "out" / "findings.jsonl"
    n, digest = merge_to_file(chunks, target)

    expected = b"".join(canonical_bytes(f) for f in sorted(all_findings, key=sort_key))
    assert n == 9
    assert target.read_bytes() == expected
    assert digest == hashlib.sha256(expected).hexdigest()
    assert [f["season"] for f in _lines(target)] == list(range(2000, 2009))
    assert not target.with_name(target.name + ".tmp").exists()


def test_merge_output_is_independent_of_chunking_and_chunk_order(tmp_path: Path) -> None:
    findings = [_finding(f"f{i}", season=1990 + (i * 7) % 11, field=f"x{i % 3}") for i in range(20)]
    single = write_chunk(tmp_path / "a", "all", findings)
    parts = [write_chunk(tmp_path / "b", f"p{k}", findings[k::4]) for k in range(4)]

    _, d1 = merge_to_file([single], tmp_path / "one.jsonl")
    _, d2 = merge_to_file(parts, tmp_path / "parts.jsonl")
    _, d3 = merge_to_file(list(reversed(parts)), tmp_path / "rev.jsonl")
    assert d1 == d2 == d3


def test_empty_chunk_and_empty_merge(tmp_path: Path) -> None:
    info = write_chunk(tmp_path, "empty", [])
    assert info.count == 0
    assert info.by_cat == Counter()
    assert info.by_player == Counter()
    assert Path(info.path).read_bytes() == b""

    target = tmp_path / "merged.jsonl"
    assert merge_to_file([info], target) == (0, hashlib.sha256(b"").hexdigest())
    assert target.read_bytes() == b""

    assert merge_to_file([], tmp_path / "none.jsonl") == (0, hashlib.sha256(b"").hexdigest())
    assert (tmp_path / "none.jsonl").read_bytes() == b""


def test_empty_chunks_are_skipped_alongside_populated_ones(tmp_path: Path) -> None:
    full = write_chunk(tmp_path, "full", [_finding("a"), _finding("b")])
    empty = write_chunk(tmp_path, "empty", [])
    n, _ = merge_to_file([empty, full, empty], tmp_path / "out.jsonl")
    assert n == 2


def test_merge_to_file_overwrites_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "out.jsonl"
    target.write_bytes(b"stale\n")
    n, _ = merge_to_file([write_chunk(tmp_path, "c", [_finding("a")])], target)
    assert n == 1
    assert _lines(target)[0]["id"] == "a"


def test_write_chunk_rejects_non_finite_values(tmp_path: Path) -> None:
    bad = _finding("a")
    bad["delta"] = float("nan")
    with pytest.raises(ValueError):
        write_chunk(tmp_path, "bad", [bad])


def test_write_chunk_rejects_finding_missing_key_fields(tmp_path: Path) -> None:
    bad = _finding("a")
    del bad["layer"]
    with pytest.raises(KeyError):
        write_chunk(tmp_path, "bad", [bad])
