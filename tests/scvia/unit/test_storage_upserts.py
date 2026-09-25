"""Merging refresh/repair upserts into a new immutable snapshot (PLAN 4.2, 5.3, 12 P01).

Contract of ``snapshots.apply_upserts``:
- rows replace base rows with the same table key, other rows are kept;
- only partitions that receive upserts are rewritten; every other fragment is reused
  by reference (same path/hash), so unchanged seasons are never re-read or re-written;
- the base snapshot is untouched and the result is a new candidate whose parent is the base;
- unknown tables/columns fail loudly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from supercoach_via.domain.schemas import TABLES, DatasetStatus
from supercoach_via.ingest.legacy import _to_table
from supercoach_via.storage import snapshots
from supercoach_via.storage.queries import SnapshotQuery

FIXED = datetime(2026, 9, 25, 1, 0, tzinfo=UTC)


def clock() -> datetime:
    return FIXED


def _pg(match_id: str, player_id: str, season: int, disposals: int | None) -> dict[str, Any]:
    row: dict[str, Any] = {c: None for c in TABLES["player_games"].column_names}
    row.update(
        match_id=match_id, player_id=player_id, club_id="hawthorn", season=season, stage_label="1",
        stage_id="r01", club_source_name="Hawthorn", link_method="key", date_quality="inferred",
        revision_id="rev:legacy", provenance="legacy_import", disposals=disposals,
    )  # fmt: skip
    return row


def _issue(issue_id: str, text: str) -> dict[str, Any]:
    return {"issue_id": issue_id, "severity": "warning", "status": "open", "table_name": None, "row_key": None,
            "source_path": None, "rule_id": "r", "explanation": text, "remediation": None,
            "acceptance_basis": None, "season": None}  # fmt: skip


def _base(root: Path) -> snapshots.SnapshotCandidate:
    b = snapshots.SnapshotBuilder(root, clock=clock, code_version="test")
    rows = [_pg("m:2025:a", "legacy:x", 2025, 10), _pg("m:2026:a", "legacy:x", 2026, 20)]
    b.add_partitioned("player_games", _to_table("player_games", rows), "season")
    b.add("quality_issues", _to_table("quality_issues", [_issue("i1", "old")]))
    return b.finish(status=DatasetStatus.LEGACY_UNVERIFIED)


def _rows(root: Path, cand: snapshots.SnapshotCandidate, sql: str) -> list[tuple[Any, ...]]:
    with SnapshotQuery(root, cand.manifest) as q:
        return q.rows(sql)


def test_inserts_and_replaces_by_key_and_reuses_untouched_partitions(tmp_path: Path) -> None:
    base = _base(tmp_path)
    new = snapshots.apply_upserts(
        tmp_path, base.manifest,
        {"player_games": [_pg("m:2026:b", "legacy:y", 2026, 7), _pg("m:2026:a", "legacy:x", 2026, 21)],
         "quality_issues": [_issue("i2", "new")]},
        clock=clock, code_version="test", status=DatasetStatus.PARTIAL, notes=["repair"],
    )  # fmt: skip
    assert new.manifest.parent == base.manifest.snapshot_id
    assert new.manifest.snapshot_id != base.manifest.snapshot_id
    assert new.manifest.status is DatasetStatus.PARTIAL
    got = _rows(tmp_path, new, "SELECT match_id, player_id, disposals FROM player_games ORDER BY 1, 2")
    assert got == [("m:2025:a", "legacy:x", 10), ("m:2026:a", "legacy:x", 21), ("m:2026:b", "legacy:y", 7)]
    assert _rows(tmp_path, new, "SELECT issue_id FROM quality_issues ORDER BY 1") == [("i1",), ("i2",)]
    old_frags = {f.partition: f for f in base.manifest.tables["player_games"].fragments}
    new_frags = {f.partition: f for f in new.manifest.tables["player_games"].fragments}
    assert new_frags["2025"] == old_frags["2025"]  # reused by reference, not rewritten
    assert new_frags["2026"].sha256 != old_frags["2026"].sha256
    assert new.manifest.tables["player_games"].row_count == 3
    # base unchanged and still loadable
    assert _rows(tmp_path, base, "SELECT count(*) FROM player_games") == [(2,)]


def test_new_partition_is_added(tmp_path: Path) -> None:
    base = _base(tmp_path)
    new = snapshots.apply_upserts(
        tmp_path, base.manifest, {"player_games": [_pg("m:2024:a", "legacy:z", 2024, 1)]},
        clock=clock, code_version="test", status=DatasetStatus.PARTIAL,
    )  # fmt: skip
    parts = sorted(f.partition or "" for f in new.manifest.tables["player_games"].fragments)
    assert parts == ["2024", "2025", "2026"]


def test_empty_upserts_rejected_and_unknown_table_or_column_fails(tmp_path: Path) -> None:
    base = _base(tmp_path)
    kw: dict[str, Any] = {"clock": clock, "code_version": "test", "status": DatasetStatus.PARTIAL}
    with pytest.raises(ValueError, match="no upserts"):
        snapshots.apply_upserts(tmp_path, base.manifest, {"player_games": []}, **kw)
    with pytest.raises(KeyError):
        snapshots.apply_upserts(tmp_path, base.manifest, {"nope": [{"a": 1}]}, **kw)
    bad = _pg("m:2026:c", "legacy:q", 2026, 1) | {"surprise": 1}
    with pytest.raises(KeyError):
        snapshots.apply_upserts(tmp_path, base.manifest, {"player_games": [bad]}, **kw)


def test_duplicate_keys_within_upserts_rejected(tmp_path: Path) -> None:
    base = _base(tmp_path)
    dup = [_pg("m:2026:c", "legacy:q", 2026, 1), _pg("m:2026:c", "legacy:q", 2026, 2)]
    with pytest.raises(ValueError, match="duplicate key"):
        snapshots.apply_upserts(tmp_path, base.manifest, {"player_games": dup}, clock=clock,
                                code_version="test", status=DatasetStatus.PARTIAL)  # fmt: skip
