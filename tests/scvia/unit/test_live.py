"""L01-L04: generic per-match live monitor + deterministic commentary (PLAN 10)."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from supercoach_via.domain.schemas import TABLES
from supercoach_via.ingest import fanfooty as ff
from supercoach_via.ingest.http import HttpClient, RawArchive, load_policies
from supercoach_via.live import commentary
from supercoach_via.live.monitor import LiveMonitor, build_live_index
from supercoach_via.publish.view_models import LiveIndex, LiveSnapshot

RAW = Path(__file__).resolve().parents[1] / "fixtures" / "raw" / "fanfooty"
REPO = Path(__file__).resolve().parents[3]
SCHEMA = ff.load_schema(REPO / "config" / "fanfooty_schema.yaml")


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 5, 17, 5, 0, tzinfo=UTC)
        self.mono = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def monotonic(self) -> float:
        return self.mono

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.mono += s
        self.now += timedelta(seconds=s)


class Feed:
    """Serves a scripted sequence of payloads / statuses for one game URL."""

    def __init__(self, script: list[bytes | int]) -> None:
        self.script = list(script)
        self.calls = 0

    def handler(self, req: httpx.Request) -> httpx.Response:
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, int):
            return httpx.Response(item)
        return httpx.Response(200, content=item)


def _client(feed: Feed, clock: Clock, tmp: Path) -> HttpClient:
    return HttpClient(
        load_policies(REPO / "config" / "source_policies.toml"),
        user_agent="scvia-test",
        archive=RawArchive(tmp / "raw"),
        transport=httpx.MockTransport(feed.handler),
        resolver=lambda h: ["93.184.215.14"],
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        now=clock,
        jitter=lambda: 0.5,
    )


def _monitor(feed: Feed, clock: Clock, tmp: Path, gid: str = "9789", mid: str = "m:2026:r10:a:b:0") -> LiveMonitor:
    return LiveMonitor(
        gid, mid, root=tmp / "live", http=_client(feed, clock, tmp), schema=SCHEMA, clock=clock, sleep=clock.sleep
    )


def raw(name: str) -> bytes:
    return (RAW / name).read_bytes()


# ---------------------------------------------------------------------------
# Commentary
# ---------------------------------------------------------------------------


def test_reads_are_deterministic_labelled_and_reliable_only() -> None:
    feed = ff.parse_feed(raw("9781_final.txt"), SCHEMA)
    a = commentary.reads_for(feed)
    assert a == commentary.reads_for(feed)
    assert a[0].startswith("[scoreboard] Adelaide won by 37 points")
    assert any(r.startswith("[reliable: kicks+handballs]") for r in a)
    assert not any("goal" in r.lower() and "scoreboard" not in r for r in a)  # no per-player goals


def test_reads_mark_partial_quarter() -> None:
    feed = ff.parse_feed(raw("9789_q2.txt"), SCHEMA)
    reads = commentary.reads_for(feed)
    assert "lead by" in reads[0] and "(Q2 in progress" in reads[0]


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------


def test_default_poll_interval_from_policy(tmp_path: Path) -> None:
    m = _monitor(Feed([raw("9789_q1.txt")]), Clock(), tmp_path)
    assert m.poll_interval_s == 90.0


def test_l01_transitions_write_only_changed_and_breaks_once_across_restart(tmp_path: Path) -> None:
    clock = Clock()
    feed = Feed([raw("9789_q1.txt"), raw("9789_q1.txt"), raw("9789_qtr_time.txt"), raw("9789_q2.txt")])
    mon = _monitor(feed, clock, tmp_path)
    outs = mon.run(max_polls=3)
    assert [o.kind for o in outs] == ["new", "unchanged", "new"]
    assert outs[2].events == ["QT"]
    assert clock.sleeps.count(90.0) == 2  # polls spaced by the policy interval
    snaps = sorted((tmp_path / "live" / "9789" / "snapshots").glob("*.json"))
    assert len(snaps) == 2  # unchanged payload not rewritten

    # restart: a new process re-reads persisted state; the same break is not re-emitted
    feed2 = Feed([raw("9789_qtr_time.txt").replace(b"Qtr Time", b"Qtr Time ", 1), raw("9789_q2.txt")])
    mon2 = _monitor(feed2, clock, tmp_path)
    o1 = mon2.poll_once()
    assert o1.events == []
    o2 = mon2.poll_once()
    assert o2.kind == "new" and o2.events == []
    assert mon2.state.breaks_emitted == ["QT"]


def test_l02_per_match_isolation(tmp_path: Path) -> None:
    clock = Clock()
    a = _monitor(Feed([raw("9789_q2.txt")]), clock, tmp_path, "9789", "m:2026:r10:a:b:0")
    b = _monitor(Feed([raw("9781_half_time.txt")]), clock, tmp_path, "9781", "m:2026:r09:c:d:0")
    assert a.poll_once().kind == "new" and b.poll_once().kind == "new"
    la, lb = a.latest(), b.latest()
    assert la and lb and la.match_id != lb.match_id
    assert la.home.name == "Richmond" or la.home.name != lb.home.name
    assert (tmp_path / "live" / "9789" / "state.json").is_file()
    assert (tmp_path / "live" / "9781" / "state.json").is_file()
    index = build_live_index(tmp_path / "live")
    assert isinstance(index, LiveIndex) and {e.source_game_id for e in index.matches} == {"9789", "9781"}


def test_l03_invalid_snapshot_preserved_not_promoted(tmp_path: Path) -> None:
    clock = Clock()
    broken = raw("9789_q2.txt").replace(b"\n", b",extra\n", 6)  # column shift
    backwards = raw("9789_q1.txt")  # Q1 after Q2 = invalid transition
    mon = _monitor(Feed([raw("9789_q2.txt"), broken, backwards]), clock, tmp_path)
    first = mon.poll_once()
    assert first.kind == "new"
    bad = mon.poll_once()
    assert bad.kind == "anomalous" and bad.snapshot is None
    back = mon.poll_once()
    assert back.kind == "rejected_transition" and back.snapshot is None
    anomalies = list((tmp_path / "live" / "9789" / "anomalies").iterdir())
    assert len(anomalies) >= 2  # preserved for diagnosis
    latest = mon.latest()
    assert latest is not None and latest.quarter == "Q2"
    assert mon.state.accepted == [first.payload_hash]


def test_l04_fetch_failure_never_returns_stale_as_new(tmp_path: Path) -> None:
    clock = Clock()
    mon = _monitor(Feed([raw("9789_q1.txt"), raw("9789_qtr_time.txt"), 503]), clock, tmp_path)
    mon.poll_once()
    mon.poll_once()
    # make the OLDER snapshot file look newest on disk: selection must use state, not mtime
    snaps = tmp_path / "live" / "9789" / "snapshots"
    first_file = snaps / f"{mon.state.accepted[0]}.json"
    future = datetime.now().timestamp() + 10_000
    os.utime(first_file, (future, future))
    failed = mon.poll_once()
    assert failed.kind == "fetch_failed" and failed.snapshot is None and failed.stale
    latest = mon.latest()
    assert latest is not None and latest.quarter == "QT"
    assert mon.state.last_error


def test_final_stops_run_and_snapshot_view_model(tmp_path: Path) -> None:
    clock = Clock()
    mon = _monitor(Feed([raw("9781_q4.txt"), raw("9781_final.txt")]), clock, tmp_path, "9781")
    outs = mon.run(max_polls=10)
    assert outs[-1].kind == "new" and outs[-1].events == ["final"] and len(outs) == 2
    snap = mon.latest()
    assert isinstance(snap, LiveSnapshot)
    assert snap.final and snap.home.score == 61 and snap.away.score == 98
    assert "goals" in snap.unavailable_fields and "kicks" in snap.reliable_fields
    assert all("goals" not in p.stats for p in snap.players)
    assert snap.reads and snap.reads[0].startswith("[scoreboard]")
    row = mon.snapshot_row()
    assert row is not None and set(row) == set(TABLES["live_snapshots"].column_names)
    assert row["home_score"] == 61


def test_rejects_unsafe_ids(tmp_path: Path) -> None:
    clock = Clock()
    with pytest.raises(ValueError):
        _monitor(Feed([b""]), clock, tmp_path, gid="../9789")
    with pytest.raises(ValueError):
        _monitor(Feed([b""]), clock, tmp_path, mid="../../etc")
