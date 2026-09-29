"""Semantic cache and changed-data mode: reuse only on identical verified inputs; results equal a fresh full audit."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from supercoach_via.integrity.report import canonical_bytes
from supercoach_via.integrity.runner import AuditOptions, AuditResult, UsageError, run_audit
from supercoach_via.settings import default_config_dir
from tests.scvia.unit import integrity_fixtures as fx

AS_OF = "2026-05-03T00:00:00Z"


@pytest.fixture
def pristine(integrity_demo: Any) -> tuple[Path, Path]:
    return integrity_demo.data_root, integrity_demo.release_dir


@pytest.fixture
def rel(pristine: tuple[Path, Path], tmp_path: Path) -> tuple[Path, Path]:
    data, release = pristine
    d2, r2 = tmp_path / "var", tmp_path / "releases" / release.name
    shutil.copytree(data, d2)
    shutil.copytree(release, r2)
    return d2, r2


def run(data: Path, release: Path, **kw: Any) -> AuditResult:
    opts = dict(data_root=data, release_dir=release, scope="full", as_of=AS_OF, families=("release",), checks=("release.validate", "release.public"))
    opts.update(kw)
    return run_audit(AuditOptions(**opts))  # type: ignore[arg-type]


def body(res: AuditResult) -> bytes:
    return canonical_bytes(res.report)


def test_cold_and_warm_runs_give_identical_reports(rel: tuple[Path, Path], tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    fresh = run(*rel)
    cold = run(*rel, cache_dir=cache)
    warm = run(*rel, cache_dir=cache)
    assert body(fresh) == body(cold) == body(warm)
    assert cold.execution["cache"]["reused"] == 0 and cold.execution["cache"]["stored"] > 0
    assert warm.execution["cache"]["computed"] == 0 and warm.execution["cache"]["reused"] > 0


def test_poisoned_entries_are_discarded(rel: tuple[Path, Path], tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    want = body(run(*rel, cache_dir=cache))
    entries = sorted((cache / "units").rglob("*.json"))
    # 1) body edited without fixing its digest
    doc = json.loads(entries[0].read_text())
    doc["body"]["findings"] = []
    doc["body"]["examined"] = {"cells": 1}
    entries[0].write_text(json.dumps(doc))
    # 2) a self-consistent entry moved under another key
    shutil.copy(entries[2], entries[1])
    # 3) truncated
    entries[3].write_bytes(entries[3].read_bytes()[:20])
    again = run(*rel, cache_dir=cache)
    assert again.execution["cache"]["invalidated"] == 3
    assert body(again) == want


def test_same_filename_with_changed_bytes_is_rechecked(rel: tuple[Path, Path], tmp_path: Path) -> None:
    data, release = rel
    cache = tmp_path / "cache"
    run(data, release, cache_dir=cache)
    detail = next(
        p
        for p in sorted((release / "public" / "matches" / "detail").glob("*.json"))
        if len(json.loads(p.read_text())["home_players"]["player_id"]) >= 2
    )

    def swap(d: Any) -> None:
        s = d["home_players"]["stats"]
        s[0], s[1] = s[1], s[0]

    fx.edit_json(detail, swap)
    fx.reseal(release)
    warm = run(data, release, cache_dir=cache)
    assert any(f.rule_id == "release.cell_mismatch" for f in warm.findings)
    assert body(warm) == body(run(data, release))
    assert warm.execution["cache"]["reused"] > 0 and warm.execution["cache"]["computed"] > 0


def test_changed_policy_invalidates_everything(rel: tuple[Path, Path], tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    run(*rel, cache_dir=cache)
    cfg = tmp_path / "cfg"
    shutil.copytree(default_config_dir(), cfg)
    p = cfg / "integrity_policy.yaml"
    p.write_text(p.read_text().replace("disposals: 60", "disposals: 61"))
    res = run(*rel, cache_dir=cache, config_dir=cfg)
    assert res.execution["cache"]["reused"] == 0
    assert body(res) == body(run(*rel, config_dir=cfg))


def test_changed_since_equals_a_fresh_full_audit_across_partitions(rel: tuple[Path, Path], tmp_path: Path) -> None:
    """A 2026 fact changes: the 2026 season, and every player page whose career includes it, are rechecked."""
    from supercoach_via.domain.schemas import CheckOutcome, ValidationReport
    from supercoach_via.storage import snapshots
    from supercoach_via.storage.queries import SnapshotQuery

    data, release = rel
    cache, prior = tmp_path / "cache", tmp_path / "prior.json"
    first = run(data, release, cache_dir=cache)
    prior.write_bytes(canonical_bytes(first.report))
    m = snapshots.load_snapshot(data)
    with SnapshotQuery(data, m, tables={"player_games"}) as q:
        row = q.arrow(
            "SELECT * FROM player_games WHERE season = 2026 ORDER BY match_id, player_id LIMIT 1"
        ).to_pylist()[0]
    row["kicks"] += 1
    row["disposals"] += 1
    cand = snapshots.apply_upserts(
        data, m, {"player_games": [row]}, clock=lambda: m.created_at, code_version="t", status=m.status
    )
    snapshots.promote(data, cand, ValidationReport(outcome=CheckOutcome.PASS))
    changed = run(data, release, cache_dir=cache, changed_since=prior)
    fresh = run(data, release)
    assert body(changed) == body(fresh)
    info = changed.execution["changed"]
    assert info["compatible"] is True and info["changed_partitions"] == ["player_games/2026"]
    stats = changed.execution["cache"]
    assert stats["reused"] > 0 and stats["computed"] > 0
    # the change surfaces in the 2026 logs/details and in that player's career page
    fields = {(f.rule_id, f.entity.split("|")[0]) for f in changed.findings}
    assert ("release.aggregate_mismatch", f"player:{row['player_id']}") in fields


def test_corrupt_or_missing_baseline_falls_back_to_full(rel: tuple[Path, Path], tmp_path: Path) -> None:
    cache, prior = tmp_path / "cache", tmp_path / "prior.json"
    first = run(*rel, cache_dir=cache)
    doc = dict(first.report)
    doc["outcome"] = "PASS" if doc["outcome"] == "FAIL" else "FAIL"  # digest no longer verifies
    prior.write_bytes(canonical_bytes(doc))
    res = run(*rel, cache_dir=cache, changed_since=prior)
    assert res.execution["changed"]["compatible"] is False
    assert res.execution["cache"] is None  # nothing reused: a full recomputation
    assert body(res) == body(first)
    missing = run(*rel, cache_dir=cache, changed_since=tmp_path / "absent.json")
    assert missing.execution["changed"]["compatible"] is False and body(missing) == body(first)


def test_changed_since_needs_a_cache(rel: tuple[Path, Path], tmp_path: Path) -> None:
    with pytest.raises(UsageError):
        run(*rel, changed_since=tmp_path / "prior.json")
