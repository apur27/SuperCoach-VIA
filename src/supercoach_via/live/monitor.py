"""Generic per-match live monitor (PLAN 10; AUDIT P07/P08).

One monitor per (FanFooty source game id, resolved match id), started by an explicit
operator command. Each poll fetches through the shared HTTP policy, hashes the payload
and writes a snapshot only when content changed (atomic writes). Per-match state is
persisted so a restart neither duplicates quarter-break events nor loses the accepted
history. Only forward phase transitions with non-decreasing scoreboards are accepted;
anything else (and any schema/sentry failure) is preserved under ``anomalies/`` and not
promoted. A failed fetch returns ``fetch_failed`` with no snapshot: the latest accepted
snapshot is chosen from persisted state, never by file age, and is flagged stale.

Layout: ``<root>/<source_game_id>/{state.json, raw/<hash>.txt, snapshots/<hash>.json,
anomalies/<hash>.txt}``. No Git, no publishing: export is a separate step.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from supercoach_via.domain.schemas import Provenance, is_safe_id
from supercoach_via.ingest import fanfooty as ff
from supercoach_via.ingest.http import HttpClient, fit_table_row
from supercoach_via.live.commentary import reads_for
from supercoach_via.publish.view_models import BoxScoreRow, LiveIndex, LiveIndexEntry, LiveSnapshot, TeamScore
from supercoach_via.storage.snapshots import atomic_write_bytes

STATE_VERSION = 1


@dataclass
class MonitorState:
    source_game_id: str
    match_id: str
    version: int = STATE_VERSION
    accepted: list[str] = field(default_factory=list)  # payload hashes, in acceptance order
    accepted_at: list[str] = field(default_factory=list)
    last_phase_index: int = -1
    last_home_score: int | None = None
    last_away_score: int | None = None
    breaks_emitted: list[str] = field(default_factory=list)
    final: bool = False
    last_poll_at: str | None = None
    last_error: str | None = None
    anomalies: list[str] = field(default_factory=list)
    label: str = ""


@dataclass
class PollOutcome:
    kind: str  # new | unchanged | fetch_failed | anomalous | rejected_transition
    payload_hash: str | None = None
    snapshot: LiveSnapshot | None = None
    events: list[str] = field(default_factory=list)
    stale: bool = False
    detail: str = ""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "x"


class LiveMonitor:
    def __init__(
        self,
        source_game_id: str,
        match_id: str,
        *,
        root: Path,
        http: HttpClient,
        schema: ff.FanFootySchema,
        clock: Callable[[], datetime],
        poll_interval_s: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = ff.feed_url(source_game_id)  # validates the id
        if not is_safe_id(match_id):
            raise ValueError(f"unsafe match id {match_id!r}")
        self.source_game_id = source_game_id
        self.match_id = match_id
        self.dir = root / source_game_id
        self.http = http
        self.schema = schema
        self.clock = clock
        self.sleep = sleep
        self.poll_interval_s = (
            float(poll_interval_s) if poll_interval_s is not None else http.policies.live_poll_interval_s
        )
        self.state = self._load_state()

    # -- persistence ------------------------------------------------------------

    def _load_state(self) -> MonitorState:
        path = self.dir / "state.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            state = MonitorState(**data)
            if state.source_game_id != self.source_game_id or state.match_id != self.match_id:
                raise ValueError("persisted state belongs to a different game/match mapping")
            return state
        return MonitorState(self.source_game_id, self.match_id)

    def _save_state(self) -> None:
        atomic_write_bytes(self.dir / "state.json", json.dumps(asdict(self.state), indent=1).encode())

    # -- polling ------------------------------------------------------------------

    def poll_once(self) -> PollOutcome:
        now = self.clock()
        self.state.last_poll_at = now.isoformat()
        res = self.http.fetch(self.url, conditional=False)
        if not res.ok or res.content is None or res.sha256 is None:
            self.state.last_error = f"{now.isoformat()} {res.outcome.value}: {res.error}"
            self._save_state()
            return PollOutcome("fetch_failed", stale=bool(self.state.accepted), detail=res.error or "")
        digest = res.sha256
        if self.state.accepted and digest == self.state.accepted[-1]:
            self.state.last_error = None
            self._save_state()
            return PollOutcome("unchanged", digest)
        feed = ff.parse_feed(res.content, self.schema)
        reason = None
        if feed.outcome.value != "PASS" or feed.header is None:
            reason = "; ".join(feed.anomalies[:5]) or "parse failed"
            kind = "anomalous"
        elif feed.phase.index < self.state.last_phase_index or (self.state.final and not feed.phase.final):
            reason = f"phase went backwards to {feed.phase.label}"
            kind = "rejected_transition"
        elif (
            self.state.last_home_score is not None and (feed.header.home_score or 0) < self.state.last_home_score
        ) or (self.state.last_away_score is not None and (feed.header.away_score or 0) < self.state.last_away_score):
            reason = "scoreboard decreased"
            kind = "rejected_transition"
        if reason is not None:
            atomic_write_bytes(self.dir / "anomalies" / f"{digest}.txt", res.content)
            self.state.anomalies.append(f"{now.isoformat()} {digest[:12]} {reason}"[:300])
            self._save_state()
            return PollOutcome(kind, digest, detail=reason)

        assert feed.header is not None
        events: list[str] = []
        if (feed.phase.is_break or feed.phase.final) and feed.phase.label not in self.state.breaks_emitted:
            self.state.breaks_emitted.append(feed.phase.label)
            events.append(feed.phase.label)
        snapshot = self._view_model(feed, now)
        atomic_write_bytes(self.dir / "raw" / f"{digest}.txt", res.content)
        atomic_write_bytes(self.dir / "snapshots" / f"{digest}.json", snapshot.model_dump_json(indent=1).encode())
        self.state.accepted.append(digest)
        self.state.accepted_at.append(now.isoformat())
        self.state.last_phase_index = feed.phase.index
        self.state.last_home_score = feed.header.home_score
        self.state.last_away_score = feed.header.away_score
        self.state.final = feed.phase.final
        self.state.last_error = None
        self.state.label = f"{feed.header.home_name} v {feed.header.away_name}"
        self._save_state()
        return PollOutcome("new", digest, snapshot, events)

    def run(
        self, *, max_polls: int | None = None, should_stop: Callable[[], bool] = lambda: False
    ) -> list[PollOutcome]:
        """Poll until final, ``max_polls`` or ``should_stop``; sleeps the policy interval between polls."""
        outcomes: list[PollOutcome] = []
        while not self.state.final and not should_stop():
            if max_polls is not None and len(outcomes) >= max_polls:
                break
            if outcomes:
                self.sleep(self.poll_interval_s)
            outcomes.append(self.poll_once())
        return outcomes

    # -- views ---------------------------------------------------------------------

    def _view_model(self, feed: ff.FeedParse, fetched_at: datetime) -> LiveSnapshot:
        h = feed.header
        assert h is not None
        numeric = sorted(f for f in self.schema.reliable if f not in ff.TEXT_FIELDS)
        players = [
            BoxScoreRow(
                player_id=f"fanfooty:{p.source_player_id or 'unknown'}",
                name=p.name,
                stats={k: float(v) for k in numeric if isinstance((v := p.reliable.get(k)), int | float)},
            )
            for p in feed.players
        ]
        timeline: list[dict[str, str | int | None]] = [
            {"fetched_at": at, "payload": hsh[:12]}
            for hsh, at in zip(self.state.accepted, self.state.accepted_at, strict=True)
        ]
        timeline.append({"fetched_at": fetched_at.isoformat(), "phase": feed.phase.label})
        return LiveSnapshot(
            source_game_id=self.source_game_id,
            match_id=self.match_id,
            fetched_at=fetched_at,
            status=h.status,
            quarter=feed.phase.label,
            home=TeamScore(
                club_id=f"src:{_slug(h.home_name)}",
                name=h.home_name,
                goals=h.home_goals,
                behinds=h.home_behinds,
                score=h.home_score,
            ),
            away=TeamScore(
                club_id=f"src:{_slug(h.away_name)}",
                name=h.away_name,
                goals=h.away_goals,
                behinds=h.away_behinds,
                score=h.away_score,
            ),
            reliable_fields=sorted(self.schema.reliable),
            unavailable_fields=feed.unavailable_fields,
            players=players,
            timeline=timeline,
            reads=reads_for(feed),
            anomalies=list(feed.notes),
            final=feed.phase.final,
        )

    def latest(self) -> LiveSnapshot | None:
        """Latest *accepted* snapshot, selected from persisted state (never by mtime)."""
        if not self.state.accepted:
            return None
        path = self.dir / "snapshots" / f"{self.state.accepted[-1]}.json"
        return LiveSnapshot.model_validate_json(path.read_bytes())

    def snapshot_row(self) -> dict[str, Any] | None:
        """``live_snapshots`` table row for the latest accepted snapshot."""
        snap = self.latest()
        if snap is None:
            return None
        digest = self.state.accepted[-1]
        return fit_table_row(
            "live_snapshots",
            {
                "source_game_id": self.source_game_id, "payload_hash": digest, "match_id": self.match_id,
                "fetched_at": snap.fetched_at, "status": snap.status, "quarter": snap.quarter,
                "home_score": snap.home.score, "away_score": snap.away.score,
                "schema_version": 1, "anomalies": json.dumps(snap.anomalies), "payload_kind": "fanfooty_txt",
                "home_source_name": snap.home.name, "away_source_name": snap.away.name,
                "source_round_label": None, "provenance": Provenance.SOURCE_FETCH.value,
                "source_path": self.url, "source_sha256": digest, "source_row": None,
            },
        )  # fmt: skip


def build_live_index(root: Path, *, delivery: str = "snapshot_archive") -> LiveIndex:
    """Index of monitored matches from persisted state files (not directory mtimes)."""
    entries: list[LiveIndexEntry] = []
    for state_path in sorted(root.glob("*/state.json")):
        state = MonitorState(**json.loads(state_path.read_text(encoding="utf-8")))
        if not state.accepted:
            continue
        entries.append(
            LiveIndexEntry(
                source_game_id=state.source_game_id,
                match_id=state.match_id,
                label=state.label or state.source_game_id,
                last_fetched_at=datetime.fromisoformat(state.accepted_at[-1]),
                final=state.final,
                resource=f"live/{state.source_game_id}/{state.accepted[-1]}.json",
            )
        )
    return LiveIndex(delivery=delivery, matches=entries)


def run_live_monitor(
    source_game_id: str,
    match_id: str,
    *,
    data_root: Path,
    http: HttpClient,
    schema_path: Path,
    clock: Callable[[], datetime],
    max_polls: int | None = None,
    poll_interval_s: float | None = None,
) -> list[PollOutcome]:
    """Operator entry point (``scvia live watch``): monitor one match under ``<data_root>/live``."""
    monitor = LiveMonitor(
        source_game_id,
        match_id,
        root=data_root / "live",
        http=http,
        schema=ff.load_schema(schema_path),
        clock=clock,
        poll_interval_s=poll_interval_s,
    )
    return monitor.run(max_polls=max_polls)
