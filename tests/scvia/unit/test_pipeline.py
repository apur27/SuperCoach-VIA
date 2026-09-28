"""Staged pipeline (PLAN 5.3, 13 R01-R04, R08-R10): state machine, lock, pointer safety, resume.

Hermetic: the DEMO corpus under tmp_path; no network, no Git.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from supercoach_via import pipeline
from supercoach_via.demo import write_demo_corpus
from supercoach_via.domain.schemas import CheckOutcome, RunState, ValidationReport
from supercoach_via.settings import RunContext, Settings
from supercoach_via.storage import runs
from supercoach_via.storage.snapshots import read_current

NOW = datetime(2026, 5, 1, 6, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    src = tmp_path_factory.mktemp("demo-src")
    write_demo_corpus(src)
    return src


def _ctx(tmp_path: Path, corpus: Path) -> RunContext:
    return RunContext(
        settings=Settings(data_root=tmp_path / "var", output_root=tmp_path / "dist", source_root=corpus),
        clock=lambda: NOW,
    )


def test_ingest_imports_validates_and_promotes(tmp_path: Path, corpus: Path) -> None:
    ctx = _ctx(tmp_path, corpus)
    res = pipeline.ingest(ctx, source_root=corpus)
    assert res.exit_code == 0 and res.state is RunState.DATASET_PROMOTED
    cur = read_current(tmp_path / "var")
    assert cur is not None and cur.snapshot_id == res.snapshot_id
    run = json.loads((tmp_path / "var" / "runs" / res.run_id / "run.json").read_text())
    assert [h[0] for h in run["history"]] == ["created", "planned", "parsed", "validated", "dataset_promoted"]
    assert set(run["steps"]) >= {"import", "validate", "promote"}
    assert (tmp_path / "var" / "runs" / res.run_id / "import-report.json").is_file()
    assert (tmp_path / "var" / "runs" / res.run_id / "validation-report.json").is_file()


def test_validation_failure_keeps_previous_pointer(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx(tmp_path, corpus)
    first = pipeline.ingest(ctx, source_root=corpus)
    before = (tmp_path / "var" / "current.json").read_bytes()
    monkeypatch.setattr(
        pipeline, "_validate", lambda *_a, **_k: ValidationReport(outcome=CheckOutcome.FAIL, issues=[{"x": 1}])
    )
    res = pipeline.ingest(ctx, source_root=corpus)
    assert res.exit_code == 4 and res.state is RunState.FAILED
    assert (tmp_path / "var" / "current.json").read_bytes() == before
    assert res.snapshot_id != first.snapshot_id or res.promoted is False
    run = json.loads((tmp_path / "var" / "runs" / res.run_id / "run.json").read_text())
    assert run["error_code"] == "validation_failed" and run["recovery"]


def test_import_crash_is_a_failed_run_not_a_success(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx(tmp_path, corpus)

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(pipeline, "_import", boom)
    res = pipeline.ingest(ctx, source_root=corpus)
    assert res.exit_code != 0 and res.state is RunState.FAILED
    assert read_current(tmp_path / "var") is None


def test_competing_writer_is_locked_out(tmp_path: Path, corpus: Path) -> None:
    ctx = _ctx(tmp_path, corpus)
    (tmp_path / "var").mkdir(parents=True)
    with runs.WriterLock(tmp_path / "var"):
        res = pipeline.ingest(ctx, source_root=corpus)
    assert res.exit_code == 5 and res.state is None


def test_rerun_is_idempotent(tmp_path: Path, corpus: Path) -> None:
    ctx = _ctx(tmp_path, corpus)
    a = pipeline.ingest(ctx, source_root=corpus)
    b = pipeline.ingest(ctx, source_root=corpus)
    assert a.snapshot_id == b.snapshot_id and a.run_id != b.run_id


def test_pipeline_never_touches_git(tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def spy(*a: object, **_k: object) -> None:
        calls.append(a)
        raise AssertionError("pipeline spawned a subprocess")

    monkeypatch.setattr(subprocess, "run", spy)
    monkeypatch.setattr(subprocess, "Popen", spy)
    pipeline.ingest(_ctx(tmp_path, corpus), source_root=corpus)
    assert not calls


def _http(tmp_path: Path, status: int) -> object:
    import httpx

    from supercoach_via.ingest.http import HttpClient, RawArchive, load_policies

    repo = Path(__file__).resolve().parents[3]
    return HttpClient(
        load_policies(repo / "config" / "source_policies.toml"), user_agent="scvia-test",
        archive=RawArchive(tmp_path / "raw"), transport=httpx.MockTransport(lambda _r: httpx.Response(status)),
        resolver=lambda _h: ["93.184.215.14"], sleep=lambda _s: None,
    )  # fmt: skip


def test_refresh_merge_recomputes_season_aggregates_from_the_added_match(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding one completed match through refresh updates that season from the merged table."""
    from datetime import timedelta

    from supercoach_via.domain.schemas import CheckOutcome, DatasetStatus
    from supercoach_via.ingest import refresh as rf
    from supercoach_via.storage.queries import SnapshotQuery
    from supercoach_via.storage.snapshots import load_snapshot

    ctx = _ctx(tmp_path, corpus)
    pipeline.ingest(ctx, source_root=corpus)
    root = tmp_path / "var"
    manifest = load_snapshot(root)
    with SnapshotQuery(root, manifest, tables={"matches", "player_games", "seasons"}) as q:
        sample = q.arrow(
            "SELECT * FROM matches WHERE season = 2026 AND status = 'complete' ORDER BY match_date DESC LIMIT 1"
        ).to_pylist()[0]
        games = q.arrow("SELECT * FROM player_games WHERE match_id = ?", [sample["match_id"]]).to_pylist()
        before = q.arrow("SELECT * FROM seasons WHERE season = 2026").to_pylist()[0]
        latest = q.scalar("SELECT max(match_date) FROM matches WHERE season = 2026")
    assert games, "demo complete match has no player rows to clone"
    latest_day = latest.date() if isinstance(latest, datetime) else latest
    added_day = latest_day + timedelta(days=7)
    added = dict(sample)
    added["match_id"] = f"{sample['match_id']}:added"
    added["match_date"] = added_day
    added["status"] = "complete"
    added_games = []
    for game in games:
        row = dict(game)
        row["match_id"] = added["match_id"]
        row["match_date"] = added_day
        added_games.append(row)
    checked = datetime(2026, 9, 28, tzinfo=UTC)

    def fake_refresh(base: object, plan: rf.RefreshPlan, context: object, **_kwargs: object) -> rf.RefreshResult:
        return rf.RefreshResult(
            plan=plan,
            outcome=CheckOutcome.PASS,
            dataset_status=DatasetStatus.VERIFIED,
            exit_code=0,
            source_checked_at=checked,
            counts={},
            upserts={"matches": [added], "player_games": added_games},
            revisions={},
            corrections=[],
            superseded=[],
            interior_gaps=[],
            stale_fixtures=[],
            work_log=[],
            issues=[],
            latest_completed_match_date=added_day,
            request_counts={},
            bytes_received=0,
        )

    monkeypatch.setattr(rf, "refresh_sources", fake_refresh)
    res = pipeline.refresh(ctx, season=2026)
    assert res.exit_code == 0 and res.promoted, res.message
    promoted = load_snapshot(root)
    with SnapshotQuery(root, promoted, tables={"matches", "seasons"}) as q:
        season = q.arrow("SELECT * FROM seasons WHERE season = 2026").to_pylist()[0]
        n_complete = q.scalar("SELECT count(*) FROM matches WHERE season = 2026 AND status = 'complete'")
        n_scheduled = q.scalar("SELECT count(*) FROM matches WHERE season = 2026 AND status = 'scheduled'")
        present = q.scalar("SELECT count(*) FROM matches WHERE match_id = ?", [added["match_id"]])
    last = season["last_match_date"]
    last = last.date() if isinstance(last, datetime) else last
    assert present == 1
    assert season["matches_complete"] == n_complete == before["matches_complete"] + 1
    assert season["matches_scheduled"] == n_scheduled == before["matches_scheduled"]
    assert last == added_day
    assert season["fixture_checked_at"] == checked
    assert season["schedule_complete"] is None and season["source_status"] is None


def test_unchanged_fixture_still_records_freshness_for_checked_seasons(
    tmp_path: Path, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A passing refresh whose only writes are source observations still stamps checked seasons."""
    from supercoach_via.domain.schemas import CheckOutcome, DatasetStatus
    from supercoach_via.ingest import refresh as rf
    from supercoach_via.storage.queries import SnapshotQuery
    from supercoach_via.storage.snapshots import load_snapshot

    ctx = _ctx(tmp_path, corpus)
    pipeline.ingest(ctx, source_root=corpus)
    root = tmp_path / "var"
    before = load_snapshot(root)
    with SnapshotQuery(root, before, tables={"seasons"}) as q:
        prior = {int(r["season"]): r for r in q.arrow("SELECT * FROM seasons").to_pylist()}
    checked = datetime(2026, 9, 28, 10, 42, tzinfo=UTC)

    def fake_refresh(base: object, plan: rf.RefreshPlan, context: object, **_kwargs: object) -> rf.RefreshResult:
        return rf.RefreshResult(
            plan=plan,
            outcome=CheckOutcome.PASS,
            dataset_status=DatasetStatus.VERIFIED,
            exit_code=0,
            source_checked_at=checked,
            counts={},
            upserts={
                "source_observations": [
                    {
                        "source_ref": "afltables:season:2026",
                        "adapter": "afltables.season_fixture",
                        "adapter_version": "test",
                        "url": "https://afltables.com/afl/seas/2026.html",
                        "fetched_at": checked,
                        "content_sha256": "ab" * 32,
                        "http_status": 200,
                        "etag": None,
                        "last_modified": None,
                        "bytes": 12,
                        "source_mode": "live",
                        "outcome": "PASS",
                    }
                ]
            },
            revisions={},
            corrections=[],
            superseded=[],
            interior_gaps=[],
            stale_fixtures=[],
            work_log=[],
            issues=[],
            latest_completed_match_date=None,
            request_counts={},
            bytes_received=0,
        )

    monkeypatch.setattr(rf, "refresh_sources", fake_refresh)
    res = pipeline.refresh(ctx, season=2026)
    assert res.exit_code == 0 and res.promoted, res.message
    promoted = load_snapshot(root)
    assert promoted.snapshot_id != before.snapshot_id
    with SnapshotQuery(root, promoted, tables={"seasons", "matches"}) as q:
        rows = {int(r["season"]): r for r in q.arrow("SELECT * FROM seasons").to_pylist()}
        n_matches = q.scalar("SELECT count(*) FROM matches")
    assert n_matches == sum(r.rows for r in before.tables["matches"].fragments)
    for season in (2025, 2026):
        got = rows[season]
        old = prior[season]
        assert got["fixture_checked_at"] == checked
        assert got["matches_complete"] == old["matches_complete"]
        assert got["matches_scheduled"] == old["matches_scheduled"]
        assert got["first_match_date"] == old["first_match_date"]
        assert got["last_match_date"] == old["last_match_date"]
        assert got["schedule_complete"] is None and got["source_status"] is None
    assert rows[2024]["fixture_checked_at"] is None


def test_unreachable_source_refresh_is_partial_and_never_moves_the_pointer(tmp_path: Path, corpus: Path) -> None:
    ctx = _ctx(tmp_path, corpus)
    first = pipeline.ingest(ctx, source_root=corpus)
    before = (tmp_path / "var" / "current.json").read_bytes()
    ctx.http = _http(tmp_path, 503)
    res = pipeline.refresh(ctx, season=2026)
    assert res.exit_code == 3 and res.state is RunState.PARTIAL and not res.promoted
    assert (tmp_path / "var" / "current.json").read_bytes() == before
    assert res.outputs["summary"]["outcome"] in ("UNKNOWN", "FAIL")
    assert first.snapshot_id == read_current(tmp_path / "var").snapshot_id  # type: ignore[union-attr]
