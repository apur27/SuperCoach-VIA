"""Storage contracts: atomic promotion, containment, manifests, locks, run states (R01-R07)."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pytest

from supercoach_via.domain.schemas import (
    CheckOutcome,
    DatasetStatus,
    RunState,
    ValidationReport,
)
from supercoach_via.storage import runs, snapshots
from supercoach_via.storage.queries import SnapshotQuery

FIXED = datetime(2026, 9, 23, 11, 0, tzinfo=UTC)


def clock() -> datetime:
    return FIXED


def _players(n: int = 3) -> pa.Table:
    return pa.table({"player_id": [f"legacy:p{i}" for i in range(n)], "x": list(range(n))})


def _build(root: Path, n: int = 3) -> snapshots.SnapshotCandidate:
    builder = snapshots.SnapshotBuilder(root, clock=clock, code_version="test")
    builder.add("players", _players(n))
    builder.add("matches", pa.table({"match_id": ["m1"], "season": [2026]}), partition="2026")
    builder.add("matches", pa.table({"match_id": ["m0"], "season": [2025]}), partition="2025")
    return builder.finish(status=DatasetStatus.LEGACY_UNVERIFIED)


PASS = ValidationReport(outcome=CheckOutcome.PASS)
FAIL = ValidationReport(outcome=CheckOutcome.FAIL)


class TestContainment:
    def test_rejects_traversal_and_absolute(self, tmp_path: Path) -> None:
        for bad in ["../x", "/etc/passwd", "a/../../x", ""]:
            with pytest.raises(snapshots.ContainmentError):
                snapshots.contained_path(tmp_path, bad)

    def test_rejects_symlink_escape(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        root = tmp_path / "root"
        root.mkdir()
        (root / "link").symlink_to(outside, target_is_directory=True)
        with pytest.raises(snapshots.ContainmentError):
            snapshots.contained_path(root, "link/file.json")

    def test_accepts_nested(self, tmp_path: Path) -> None:
        p = snapshots.contained_path(tmp_path, "a/b/c.json")
        assert p == tmp_path.resolve() / "a" / "b" / "c.json"


class TestAtomicWrite:
    def test_replaces_whole_file(self, tmp_path: Path) -> None:
        target = tmp_path / "f.json"
        snapshots.atomic_write_bytes(target, b"one")
        snapshots.atomic_write_bytes(target, b"two")
        assert target.read_bytes() == b"two"
        assert [p.name for p in tmp_path.iterdir()] == ["f.json"]

    def test_failure_leaves_previous_bytes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / "f.json"
        snapshots.atomic_write_bytes(target, b"good")

        def boom(*_a: object, **_k: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            snapshots.atomic_write_bytes(target, b"bad")
        assert target.read_bytes() == b"good"
        assert [p.name for p in tmp_path.iterdir()] == ["f.json"]


class TestSnapshots:
    def test_snapshot_id_is_content_derived_and_rerun_idempotent(self, tmp_path: Path) -> None:
        a = _build(tmp_path)
        b = _build(tmp_path)
        assert a.manifest.snapshot_id == b.manifest.snapshot_id
        assert a.manifest.snapshot_id.startswith("sha256:")
        assert a.manifest.tables["matches"].row_count == 2
        assert len(a.manifest.tables["matches"].fragments) == 2

    def test_unchanged_partition_reuses_fragment(self, tmp_path: Path) -> None:
        a = _build(tmp_path, n=3)
        b = _build(tmp_path, n=4)
        assert a.manifest.snapshot_id != b.manifest.snapshot_id
        assert a.manifest.tables["matches"].fragments == b.manifest.tables["matches"].fragments
        assert a.manifest.tables["players"].fragments != b.manifest.tables["players"].fragments

    def test_promote_requires_passing_report(self, tmp_path: Path) -> None:
        cand = _build(tmp_path)
        with pytest.raises(snapshots.PromotionError):
            snapshots.promote(tmp_path, cand, FAIL)
        assert snapshots.read_current(tmp_path) is None

    def test_promote_then_load_verifies_hashes(self, tmp_path: Path) -> None:
        cand = _build(tmp_path)
        ref = snapshots.promote(tmp_path, cand, PASS, promoted_at=FIXED)
        loaded = snapshots.load_snapshot(tmp_path, "current", verify=True)
        assert loaded.snapshot_id == ref.snapshot_id

    def test_tampered_fragment_is_rejected(self, tmp_path: Path) -> None:
        cand = _build(tmp_path)
        snapshots.promote(tmp_path, cand, PASS, promoted_at=FIXED)
        frag = cand.manifest.tables["players"].fragments[0]
        path = tmp_path / "fragments" / frag.path
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b"x")
        with pytest.raises(snapshots.IntegrityError):
            snapshots.load_snapshot(tmp_path, "current", verify=True)

    def test_failed_promotion_keeps_previous_pointer(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        first = _build(tmp_path, n=3)
        snapshots.promote(tmp_path, first, PASS, promoted_at=FIXED)
        second = _build(tmp_path, n=5)

        real_replace = os.replace

        def failing_replace(src: str | Path, dst: str | Path) -> None:
            if str(dst).endswith("current.json"):
                raise OSError("power loss")
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", failing_replace)
        with pytest.raises(OSError):
            snapshots.promote(tmp_path, second, PASS, promoted_at=FIXED)
        pointer = snapshots.read_current(tmp_path)
        assert pointer is not None and pointer.snapshot_id == first.manifest.snapshot_id

    def test_manifest_with_missing_fragment_is_rejected(self, tmp_path: Path) -> None:
        cand = _build(tmp_path)
        snapshots.promote(tmp_path, cand, PASS, promoted_at=FIXED)
        frag = cand.manifest.tables["matches"].fragments[0]
        (tmp_path / "fragments" / frag.path).unlink()
        with pytest.raises(snapshots.IntegrityError):
            snapshots.load_snapshot(tmp_path, "current", verify=True)


class TestQueries:
    def test_query_registers_manifest_tables_only(self, tmp_path: Path) -> None:
        cand = _build(tmp_path)
        snapshots.promote(tmp_path, cand, PASS, promoted_at=FIXED)
        snap = snapshots.load_snapshot(tmp_path, "current")
        with SnapshotQuery(tmp_path, snap) as q:
            assert q.scalar("select count(*) from matches") == 2
            assert q.scalar("select count(*) from matches where season = 2026") == 1
            rows = q.rows("select player_id from players order by player_id")
            assert [r[0] for r in rows] == ["legacy:p0", "legacy:p1", "legacy:p2"]

    def test_partition_selection_reads_only_requested_fragments(self, tmp_path: Path) -> None:
        cand = _build(tmp_path)
        snapshots.promote(tmp_path, cand, PASS, promoted_at=FIXED)
        snap = snapshots.load_snapshot(tmp_path, "current")
        with SnapshotQuery(tmp_path, snap, partitions={"matches": {"2026"}}) as q:
            assert q.scalar("select count(*) from matches") == 1
            assert q.fragments_registered["matches"] == 1


class TestRuns:
    def test_run_id_format_has_microseconds_and_suffix(self) -> None:
        rid = runs.new_run_id(clock)
        assert re.match(r"^20260923T110000\.000000Z-[0-9a-f]{8}$", rid)
        assert runs.new_run_id(clock) != rid

    def test_state_machine_rejects_invalid_transition(self, tmp_path: Path) -> None:
        run = runs.RunStore.create(tmp_path, "import-legacy", clock=clock, code_version="t")
        run.transition(RunState.PLANNED)
        with pytest.raises(runs.InvalidTransitionError):
            run.transition(RunState.PUBLISHED)
        run.transition(RunState.FAILED, error_code="E_TEST")
        with pytest.raises(runs.InvalidTransitionError):
            run.transition(RunState.PLANNED)
        data = json.loads((run.directory / "run.json").read_text())
        assert data["state"] == "failed"
        assert [h[0] for h in data["history"]] == ["created", "planned", "failed"]

    def test_events_are_jsonl_and_redacted(self, tmp_path: Path) -> None:
        run = runs.RunStore.create(tmp_path, "x", clock=clock, code_version="t")
        run.event("fetch", url="https://user:secret@example.com/a", token="abc", note="n" * 5000)
        line = (run.directory / "events.jsonl").read_text().splitlines()[-1]
        rec = json.loads(line)
        assert "secret" not in line and rec["token"] == "[REDACTED]"
        assert len(rec["note"]) <= 1100

    def test_second_writer_is_locked_out(self, tmp_path: Path) -> None:
        with runs.WriterLock(tmp_path), pytest.raises(runs.LockedError), runs.WriterLock(tmp_path):
            pass
        with runs.WriterLock(tmp_path):
            pass

    def test_step_reuse_requires_matching_inputs(self, tmp_path: Path) -> None:
        run = runs.RunStore.create(tmp_path, "x", clock=clock, code_version="v1")
        run.record_step("analyze", input_hashes={"snap": "a"}, outputs={"o": "h"}, state="succeeded")
        assert run.reusable_step("analyze", {"snap": "a"}, code_version="v1") is not None
        assert run.reusable_step("analyze", {"snap": "b"}, code_version="v1") is None
        assert run.reusable_step("analyze", {"snap": "a"}, code_version="v2") is None
