"""Refresh planning and data-only source refresh (PLAN 5.2; AUDIT C06, C07, C09, P05).

``plan_refresh`` is offline and write-free: it reads only the accepted base state and
returns the seasons, work and request estimates a refresh would perform.

``refresh_sources`` executes a plan through the shared :class:`HttpClient`:

1. fetch each in-scope season page first (mandatory work);
2. diff the parsed fixture against the base and *then* plan match-detail work:
   new/changed/rescheduled matches always, every completed match in correction-overlap
   seasons (current + prior by default; conditional requests keep this cheap), every
   match in a ``repair_season``;
3. skip reparsing any payload whose hash equals the accepted source revision (except in
   repair mode), and read base player rows only for matches whose payload changed;
4. fetch player pages only for unknown source player URLs (new debuts) and validate that
   the page shows the game;
5. return upserts keyed by source identity with new revision IDs, the superseded prior
   revisions, cell-level corrections, interior gaps, stale fixtures and per-kind
   required/attempted/succeeded/unchanged/failed/quarantined counts.

Completion is the set of required work items. Any mandatory failure or quarantine makes
the result ``PARTIAL`` with outcome ``UNKNOWN``/``FAIL`` and exit code 3; such a result is
never promotable as a verified-fresh dataset. No files are written here besides the
HTTP client's raw payload archive; promotion is the caller's job.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from supercoach_via.domain.schemas import (
    PLAYER_STAT_COLUMNS,
    CheckOutcome,
    DatasetStatus,
    DateQuality,
    IdentityStatus,
    MatchStatus,
    Provenance,
    Severity,
)
from supercoach_via.ingest import afltables as at
from supercoach_via.ingest.http import FetchResult, HttpClient, fit_table_row
from supercoach_via.settings import RunContext

REFRESH_VERSION = "1"
#: Honest upper bound on source matches in one season, for request estimates only.
SEASON_MATCH_UPPER_BOUND = 250
OUTPUT_TABLES = ("matches", "player_games", "players", "source_observations", "quality_issues", "quarantine")

ClubResolver = Callable[[str, int], str | None]
PlayerRowLoader = Callable[[int, str], list[dict[str, Any]]]


class RefreshConfigError(ValueError):
    """Invalid refresh configuration (CLI exit code 2)."""


# ---------------------------------------------------------------------------
# Base state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseMatch:
    match_id: str
    season: int
    status: str
    stage_id: str
    home_source_name: str
    away_source_name: str
    home_score: int | None
    away_score: int | None
    local_start: str | None


@dataclass
class BaseState:
    """What a refresh needs from the accepted snapshot (no fact tables loaded eagerly)."""

    snapshot_id: str | None
    matches: dict[str, BaseMatch]
    source_revisions: dict[str, str] = field(default_factory=dict)
    player_urls: dict[str, str] = field(default_factory=dict)  # source URL -> player_id
    load_player_rows: PlayerRowLoader | None = None

    @property
    def latest_season(self) -> int | None:
        seasons = [m.season for m in self.matches.values() if m.status == MatchStatus.COMPLETE.value]
        return max(seasons) if seasons else None

    def season_matches(self, season: int) -> list[BaseMatch]:
        return [m for m in self.matches.values() if m.season == season]


def base_state_from_snapshot(data_root: Path, selector: str = "current") -> BaseState:
    """Load the accepted snapshot's match index, source revisions and player source URLs.

    Player-game rows are *not* loaded; ``load_player_rows`` reads one season partition
    on demand (DuckDB over the manifest-listed fragment), so unchanged partitions are
    never read.
    """
    from supercoach_via.storage.queries import SnapshotQuery
    from supercoach_via.storage.snapshots import load_snapshot

    manifest = load_snapshot(data_root, selector)
    matches: dict[str, BaseMatch] = {}
    if "matches" in manifest.tables:
        with SnapshotQuery(data_root, manifest, tables={"matches"}) as q:
            for row in q.rows(
                "SELECT match_id, season, status, stage_id, home_source_name, away_source_name, "
                "home_score, away_score, local_start FROM matches"
            ):
                matches[str(row[0])] = BaseMatch(
                    str(row[0]), int(row[1]), str(row[2]), str(row[3]), str(row[4]), str(row[5]),
                    None if row[6] is None else int(row[6]), None if row[7] is None else int(row[7]),
                    None if row[8] is None else str(row[8]),
                )  # fmt: skip
    urls: dict[str, str] = {}
    if "players" in manifest.tables:
        with SnapshotQuery(data_root, manifest, tables={"players"}) as q:
            for pid, raw in q.rows("SELECT player_id, source_urls FROM players WHERE source_urls IS NOT NULL"):
                try:
                    for url in json.loads(str(raw)):
                        urls[str(url)] = str(pid)
                except ValueError:
                    continue

    def loader(season: int, match_id: str) -> list[dict[str, Any]]:
        entry = manifest.tables.get("player_games")
        if entry is None or not any(f.partition == str(season) for f in entry.fragments):
            return []
        with SnapshotQuery(
            data_root, manifest, tables={"player_games"}, partitions={"player_games": {str(season)}}
        ) as q:
            frame = q.arrow("SELECT * FROM player_games WHERE match_id = ?", [match_id])
        return list(frame.to_pylist())

    return BaseState(
        snapshot_id=manifest.snapshot_id,
        matches=matches,
        source_revisions=dict(manifest.source_revisions),
        player_urls=urls,
        load_player_rows=loader,
    )


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefreshRequest:
    current_season: int | None = None  # default: settings.season, else the clock's year
    overlap_seasons: int = 2  # current + prior
    repair_season: int | None = None
    seasons: tuple[int, ...] = ()  # extra explicit seasons
    include_new_player_pages: bool = True


@dataclass(frozen=True)
class WorkItem:
    kind: str  # season_fixture | match_detail | player_page
    key: str
    url: str
    season: int
    mandatory: bool
    reason: str


@dataclass(frozen=True)
class RefreshPlan:
    base_snapshot_id: str | None
    created_at: datetime
    current_season: int
    seasons: tuple[int, ...]
    overlap_seasons: tuple[int, ...]
    catch_up_seasons: tuple[int, ...]
    repair_seasons: tuple[int, ...]
    work: tuple[WorkItem, ...]
    estimated_requests: dict[str, int]
    outputs: tuple[str, ...]
    include_new_player_pages: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_snapshot_id": self.base_snapshot_id,
            "created_at": self.created_at.isoformat(),
            "current_season": self.current_season,
            "seasons": list(self.seasons),
            "overlap_seasons": list(self.overlap_seasons),
            "catch_up_seasons": list(self.catch_up_seasons),
            "repair_seasons": list(self.repair_seasons),
            "work": [w.__dict__ for w in self.work],
            "estimated_requests": dict(self.estimated_requests),
            "outputs": list(self.outputs),
            "network_during_plan": False,
        }

    def describe(self) -> str:
        lines = [
            f"base snapshot: {self.base_snapshot_id or '(none)'}",
            f"current season: {self.current_season}",
            f"seasons: {', '.join(map(str, self.seasons))}",
            f"correction overlap: {', '.join(map(str, self.overlap_seasons)) or '-'}",
            f"catch-up: {', '.join(map(str, self.catch_up_seasons)) or '-'}",
            f"repair: {', '.join(map(str, self.repair_seasons)) or '-'}",
            "sources: afltables.com season pages first, then match details planned from the "
            "fixture diff" + (", then player pages for new debuts" if self.include_new_player_pages else ""),
            f"estimated requests: {self.estimated_requests['min']}..{self.estimated_requests['max']} "
            "(conditional requests where validators exist)",
            f"outputs: {', '.join(self.outputs)} (candidate only; promotion is separate)",
            "network: none during planning",
        ]
        lines += [f"  {w.kind} {w.key} [{w.reason}] {w.url}" for w in self.work]
        return "\n".join(lines)


def _current_season(request: RefreshRequest, context: RunContext) -> int:
    season = request.current_season or context.settings.season or context.clock().year
    if not 1897 <= season <= 2100:
        raise RefreshConfigError(f"current season out of range: {season}")
    return season


def plan_refresh(base: BaseState, request: RefreshRequest, context: RunContext) -> RefreshPlan:
    """Offline plan: no network, no writes."""
    current = _current_season(request, context)
    if request.overlap_seasons < 1:
        raise RefreshConfigError("overlap_seasons must be >= 1")
    for s in (*request.seasons, *((request.repair_season,) if request.repair_season else ())):
        if not 1897 <= s <= current:
            raise RefreshConfigError(f"season {s} outside 1897..{current}")
    overlap = tuple(range(current - request.overlap_seasons + 1, current + 1))
    latest = base.latest_season
    catch_up = tuple(range(latest, overlap[0])) if latest is not None and latest < overlap[0] else ()
    repair = (request.repair_season,) if request.repair_season else ()
    seasons = tuple(sorted({*overlap, *catch_up, *repair, *request.seasons}))
    work = tuple(
        WorkItem(
            "season_fixture",
            f"afltables:season:{s}",
            at.season_url(s),
            s,
            True,
            "repair" if s in repair else "overlap" if s in overlap else "catch_up" if s in catch_up else "requested",
        )
        for s in seasons
    )
    detail_max = sum(max(len(base.season_matches(s)), SEASON_MATCH_UPPER_BOUND) for s in seasons)
    return RefreshPlan(
        base_snapshot_id=base.snapshot_id,
        created_at=context.clock(),
        current_season=current,
        seasons=seasons,
        overlap_seasons=overlap,
        catch_up_seasons=catch_up,
        repair_seasons=repair,
        work=work,
        estimated_requests={"min": len(work), "max": len(work) + detail_max},
        outputs=OUTPUT_TABLES,
        include_new_player_pages=request.include_new_player_pages,
    )


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class WorkCounts:
    required: int = 0
    attempted: int = 0
    succeeded: int = 0
    unchanged: int = 0
    failed: int = 0
    quarantined: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass
class RefreshResult:
    plan: RefreshPlan
    outcome: CheckOutcome
    dataset_status: DatasetStatus
    exit_code: int
    source_checked_at: datetime
    counts: dict[str, WorkCounts]
    upserts: dict[str, list[dict[str, Any]]]
    revisions: dict[str, str]
    corrections: list[dict[str, Any]]
    superseded: list[dict[str, Any]]
    interior_gaps: list[dict[str, Any]]
    stale_fixtures: list[dict[str, Any]]
    work_log: list[dict[str, Any]]
    issues: list[str]
    latest_completed_match_date: date | None
    request_counts: dict[str, int]
    bytes_received: int

    @property
    def promotable_as_verified(self) -> bool:
        return self.outcome is CheckOutcome.PASS and self.dataset_status is DatasetStatus.VERIFIED

    def summary(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "dataset_status": self.dataset_status.value,
            "exit_code": self.exit_code,
            "base_snapshot_id": self.plan.base_snapshot_id,
            "source_checked_at": self.source_checked_at.isoformat(),
            "latest_completed_match_date": (
                self.latest_completed_match_date.isoformat() if self.latest_completed_match_date else None
            ),
            "counts": {k: v.as_dict() for k, v in self.counts.items()},
            "upserts": {k: len(v) for k, v in self.upserts.items()},
            "corrections": len(self.corrections),
            "superseded": len(self.superseded),
            "interior_gaps": len(self.interior_gaps),
            "stale_fixtures": len(self.stale_fixtures),
            "requests": dict(self.request_counts),
            "bytes": self.bytes_received,
            "issues": self.issues[:50],
        }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return bool(a == b)


def _result_token(own: int | None, other: int | None) -> str | None:
    if own is None or other is None:
        return None
    return "W" if own > other else "L" if own < other else "D"


def _jersey(token: str) -> int | None:
    digits = "".join(ch for ch in token if ch.isdigit())
    return int(digits) if digits else None


def _new_player_id(url: str) -> str:
    token = url.rsplit("/players/", 1)[1].removesuffix(".html")
    return "src:afltables:" + token.replace("/", ".")


class _Run:
    def __init__(self, base: BaseState, plan: RefreshPlan, http: HttpClient, context: RunContext,
                 club_resolver: ClubResolver | None) -> None:  # fmt: skip
        self.base, self.plan, self.http, self.ctx = base, plan, http, context
        self.club_resolver = club_resolver
        self.counts = {k: WorkCounts() for k in ("season_fixture", "match_detail", "player_page")}
        self.upserts: dict[str, list[dict[str, Any]]] = {t: [] for t in OUTPUT_TABLES}
        self.revisions: dict[str, str] = {}
        self.corrections: list[dict[str, Any]] = []
        self.superseded: list[dict[str, Any]] = []
        self.gaps: list[dict[str, Any]] = []
        self.stale: list[dict[str, Any]] = []
        self.log: list[dict[str, Any]] = []
        self.issues: list[str] = []
        self.blocking = False
        self.unreachable = False
        self.latest: date | None = None
        self.fixtures: dict[int, at.SeasonFixture] = {}

    # -- bookkeeping --------------------------------------------------------

    def fetch(self, item: WorkItem, *, conditional: bool) -> FetchResult:
        c = self.counts[item.kind]
        c.required += 1
        c.attempted += 1
        res = self.http.fetch(item.url, conditional=conditional)
        self.upserts["source_observations"].append(
            fit_table_row("source_observations", res.observation(f"afltables.{item.kind}", at.ADAPTER_VERSION))
        )
        if not res.ok:
            c.failed += 1
            self.blocking = True
            if res.outcome is CheckOutcome.UNKNOWN:
                self.unreachable = True
            self.note(item, "failed", res.outcome, res.error or "fetch failed")
        return res

    def note(self, item: WorkItem, status: str, outcome: CheckOutcome, detail: str = "") -> None:
        self.log.append(
            {"kind": item.kind, "key": item.key, "url": item.url, "status": status,
             "outcome": outcome.value, "detail": detail[:300]}
        )  # fmt: skip
        if status in ("failed", "quarantined"):
            self.issues.append(f"{item.kind} {item.key}: {status}: {detail[:200]}")

    def issue(self, rule: str, severity: Severity, row_key: str, season: int, text: str, source: str | None) -> None:
        iid = hashlib.sha256(f"{rule}|{row_key}|{text}".encode()).hexdigest()[:20]
        self.upserts["quality_issues"].append(
            fit_table_row(
                "quality_issues",
                {"issue_id": f"issue:{iid}", "severity": severity.value, "status": "open", "table_name": "matches",
                 "row_key": row_key, "source_path": source, "rule_id": rule, "explanation": text[:500],
                 "remediation": None, "acceptance_basis": None, "season": season},
            )
        )  # fmt: skip
        if severity is Severity.BLOCKING:
            self.blocking = True
            self.issues.append(f"{rule} {row_key}: {text[:200]}")

    def quarantine(self, item: WorkItem, reason: str, res: FetchResult, raw: Mapping[str, Any]) -> None:
        self.counts[item.kind].quarantined += 1
        self.blocking = True
        qid = hashlib.sha256(f"{item.key}|{res.sha256}|{reason}".encode()).hexdigest()[:20]
        self.upserts["quarantine"].append(
            fit_table_row(
                "quarantine",
                {"quarantine_id": f"q:{qid}", "table_name": "player_games" if item.kind != "season_fixture"
                 else "matches", "reason": reason[:500], "candidates": None,
                 "raw": json.dumps(dict(raw), default=str, sort_keys=True)[:4000], "season": item.season,
                 "provenance": Provenance.SOURCE_FETCH.value, "source_path": item.url,
                 "source_sha256": res.sha256, "source_row": None},
            )
        )  # fmt: skip
        self.note(item, "quarantined", CheckOutcome.FAIL, reason)

    # -- stages ---------------------------------------------------------------

    def season(self, item: WorkItem) -> None:
        s = item.season
        repair = s in self.plan.repair_seasons
        res = self.fetch(item, conditional=not repair)
        if not res.ok or res.content is None:
            return
        base_rev = self.base.source_revisions.get(item.key)
        if res.sha256 == base_rev and not repair and s not in self.plan.overlap_seasons:
            self.counts[item.kind].unchanged += 1
            self.note(item, "unchanged", CheckOutcome.PASS)
            return
        fx = at.parse_season_page(res.content, season=s, club_resolver=self.club_resolver)
        if fx.outcome is not CheckOutcome.PASS:
            self.quarantine(item, "season page parse: " + "; ".join(fx.issues[:5]), res, {"issues": fx.issues[:20]})
            return
        self.counts[item.kind].succeeded += 1
        self.revisions[item.key] = res.sha256 or ""
        self.note(item, "succeeded", CheckOutcome.PASS)
        self.fixtures[s] = fx
        self.diff_fixture(fx, res, repair=repair)

    def diff_fixture(self, fx: at.SeasonFixture, res: FetchResult, *, repair: bool) -> None:
        s = fx.season
        in_base = {m.match_id: m for m in self.base.season_matches(s)}
        base_latest = max(
            (m.local_start or "" for m in in_base.values() if m.status == MatchStatus.COMPLETE.value), default=""
        )
        source_ids = set()
        detail_items: list[tuple[at.FixtureMatch, WorkItem, bool]] = []
        for m in fx.matches:
            source_ids.add(m.match_id)
            b = in_base.get(m.match_id)
            complete = m.status is MatchStatus.COMPLETE
            if complete and m.match_date and (self.latest is None or m.match_date > self.latest):
                self.latest = m.match_date
            changed = (
                b is None
                or b.status != m.status.value
                or not _same(b.home_score, m.home_score)
                or not _same(b.away_score, m.away_score)
                or b.local_start != m.local_start
            )
            if changed:
                self.upserts["matches"].append(m.to_row(source_ref=res.source_ref, source_sha256=res.sha256))
            if b is None and complete and base_latest and (m.local_start or "") < base_latest:
                self.gaps.append({"match_id": m.match_id, "season": s, "stage_id": m.stage.stage_id,
                                  "local_start": m.local_start})  # fmt: skip
                self.issue("interior_gap", Severity.WARNING, m.match_id, s,
                           "completed source match missing inside the accepted range", m.detail_url)  # fmt: skip
            if b is not None and b.status == MatchStatus.COMPLETE.value and not complete:
                self.issue("status_regression", Severity.BLOCKING, m.match_id, s,
                           f"accepted as complete but source now reports {m.status.value}", m.detail_url)  # fmt: skip
            if not complete or m.source_game_id is None or m.detail_url is None:
                continue
            if changed:
                reason = "new" if b is None else "changed"
            elif repair:
                reason = "repair"
            elif s in self.plan.overlap_seasons:
                reason = "overlap_recheck"
            else:
                continue
            key = f"afltables:game:{m.source_game_id}"
            detail_items.append((m, WorkItem("match_detail", key, m.detail_url, s, True, reason), b is not None))
        for mid, b in in_base.items():
            if mid not in source_ids:
                sev = Severity.BLOCKING if b.status == MatchStatus.COMPLETE.value else Severity.WARNING
                self.stale.append({"match_id": mid, "season": s, "base_status": b.status})
                self.issue("stale_fixture", sev, mid, s, "accepted match not present in the source fixture", None)
        for m, item, existed in detail_items:
            self.detail(m, item, existed=existed, repair=repair)

    def detail(self, m: at.FixtureMatch, item: WorkItem, *, existed: bool, repair: bool) -> None:
        res = self.fetch(item, conditional=not repair)
        if not res.ok or res.content is None:
            return
        base_rev = self.base.source_revisions.get(item.key)
        if res.sha256 == base_rev and not repair:
            self.counts[item.kind].unchanged += 1
            self.note(item, "unchanged", CheckOutcome.PASS)
            return
        assert m.source_game_id is not None
        d = at.parse_match_detail(res.content, season=m.season, game_id=m.source_game_id)
        if d.outcome is not CheckOutcome.PASS:
            self.quarantine(item, "match detail parse: " + "; ".join(d.issues[:5]), res, {"issues": d.issues[:20]})
            return
        if (d.home_name, d.away_name, d.home_score, d.away_score) != (
            m.home_name,
            m.away_name,
            m.home_score,
            m.away_score,
        ):
            self.quarantine(item, "match detail disagrees with the season fixture", res,
                            {"detail": [d.home_name, d.away_name, d.home_score, d.away_score],
                             "fixture": [m.home_name, m.away_name, m.home_score, m.away_score]})  # fmt: skip
            return
        revision = f"rev:{(res.sha256 or '')[:16]}"
        base_rows: dict[str, dict[str, Any]] = {}
        if existed and self.base.load_player_rows is not None:
            base_rows = {str(r["player_id"]): r for r in self.base.load_player_rows(m.season, m.match_id)}
        rows: list[dict[str, Any]] = []
        pending_debuts: list[tuple[str, dict[str, Any]]] = []
        for idx, p in enumerate(d.players, start=1):
            if p.player_url is None:
                self.quarantine(item, f"player without source URL: {p.source_name}", res, {"name": p.source_name})
                return
            home = p.team == m.home_name
            own, other = (m.home_score, m.away_score) if home else (m.away_score, m.home_score)
            known = self.base.player_urls.get(p.player_url)
            pid = known or _new_player_id(p.player_url)
            slug = m.home_team_slug if home else m.away_team_slug
            row = {
                "match_id": m.match_id, "player_id": pid,
                "club_id": (m.home_club_id if home else m.away_club_id) or f"src:{slug}",
                "season": m.season,
                "opponent_club_id": (m.away_club_id if home else m.home_club_id),
                "stage_label": m.stage.label, "stage_id": m.stage.stage_id,
                "club_source_name": p.team, "opponent_source_name": p.opponent, "link_method": "source_url",
                "match_date": m.match_date, "date_quality": DateQuality.FIXTURE_VERIFIED.value,
                "career_game_counter": None, "career_game_counter_token": None,
                "result": _result_token(own, other), "jersey_number": _jersey(p.jersey_token),
                **p.stats,
                "available_at": res.fetched_at, "revision_id": revision,
                "provenance": Provenance.SOURCE_FETCH.value, "source_path": item.url,
                "source_sha256": res.sha256, "source_row": idx,
            }  # fmt: skip
            prev = base_rows.get(pid)
            if prev is not None:
                diffs = [c for c in PLAYER_STAT_COLUMNS if not _same(prev.get(c), p.stats.get(c))]
                for c in diffs:
                    self.corrections.append({"match_id": m.match_id, "player_id": pid, "column": c,
                                             "old": prev.get(c), "new": p.stats.get(c), "revision_id": revision,
                                             "source_sha256": res.sha256})  # fmt: skip
                if diffs or prev.get("revision_id") != revision:
                    self.superseded.append({"match_id": m.match_id, "player_id": pid,
                                            "old_revision_id": prev.get("revision_id"),
                                            "new_revision_id": revision})  # fmt: skip
            if known is None:
                pending_debuts.append((p.player_url, row))
            else:
                rows.append(fit_table_row("player_games", row))
        if existed and base_rows:
            missing = set(base_rows) - {r["player_id"] for r in rows} - {r["player_id"] for _u, r in pending_debuts}
            for pid in sorted(missing):
                self.issue("player_row_missing_at_source", Severity.BLOCKING, f"{m.match_id}|{pid}", m.season,
                           "accepted player-game row absent from the source match page", item.url)  # fmt: skip
        for url, row in pending_debuts:
            if self.validate_debut(url, row, m):
                rows.append(fit_table_row("player_games", row))
        self.upserts["player_games"].extend(rows)
        self.revisions[item.key] = res.sha256 or ""
        self.counts[item.kind].succeeded += 1
        self.note(item, "succeeded", CheckOutcome.PASS, item.reason)

    def validate_debut(self, url: str, row: dict[str, Any], m: at.FixtureMatch) -> bool:
        item = WorkItem("player_page", f"afltables:player:{url.rsplit('/players/', 1)[1]}", url, m.season, True,
                        "new_player")  # fmt: skip
        if not self.plan.include_new_player_pages:
            self.counts[item.kind].required += 1
            self.counts[item.kind].failed += 1
            self.blocking = True
            self.note(item, "failed", CheckOutcome.UNKNOWN, "new player page not fetched (disabled)")
            return False
        res = self.fetch(item, conditional=True)
        if not res.ok or res.content is None:
            return False
        page = at.parse_player_page(res.content, page_url=url)
        fx = self.fixtures.get(m.season)
        resolved = at.resolve_player_games(page.games, fx) if fx else []
        if page.outcome is not CheckOutcome.PASS or not any(r.match_id == m.match_id for r in resolved):
            self.quarantine(item, "new player's page does not confirm this game", res,
                            {"player_url": url, "match_id": m.match_id, "issues": page.issues[:10]})  # fmt: skip
            return False
        for r in resolved:
            if r.match_id == m.match_id:
                row["career_game_counter"] = r.game.career_game_counter
                row["career_game_counter_token"] = r.game.career_game_counter_token
        name = page.name or ""
        first, _, last = name.partition(" ")
        self.upserts["players"].append(
            fit_table_row(
                "players",
                {"player_id": row["player_id"], "legacy_slug": None, "display_name": name, "first_name": first or None,
                 "last_name": last or None, "birth_date": page.birth_date,
                 "birth_date_quality": "source" if page.birth_date else "unknown",
                 "debut_date": m.match_date, "height_cm": None, "weight_kg": None,
                 "identity_status": IdentityStatus.CANONICAL.value, "canonical_player_id": None,
                 "source_urls": json.dumps([url]), "provenance": Provenance.SOURCE_FETCH.value,
                 "source_path": url, "source_sha256": res.sha256, "source_row": None},
            )
        )  # fmt: skip
        self.revisions[item.key] = res.sha256 or ""
        self.counts[item.kind].succeeded += 1
        self.note(item, "succeeded", CheckOutcome.PASS, "debut validated")
        return True


def refresh_sources(
    base: BaseState,
    plan: RefreshPlan,
    context: RunContext,
    *,
    club_resolver: ClubResolver | None = None,
) -> RefreshResult:
    """Execute ``plan`` (network through ``context.http``). Returns a candidate, never promotes."""
    http = context.http
    if not isinstance(http, HttpClient):
        raise RefreshConfigError("refresh_sources needs context.http (an ingest.http.HttpClient)")
    if club_resolver is None:
        club_resolver = _default_club_resolver(context)
    run = _Run(base, plan, http, context, club_resolver)
    for item in plan.work:
        run.season(item)
    if run.blocking:
        outcome = CheckOutcome.UNKNOWN if run.unreachable else CheckOutcome.FAIL
        status, code = DatasetStatus.PARTIAL, 3
    else:
        outcome, status, code = CheckOutcome.PASS, DatasetStatus.VERIFIED, 0
    return RefreshResult(
        plan=plan,
        outcome=outcome,
        dataset_status=status,
        exit_code=code,
        source_checked_at=context.clock(),
        counts=run.counts,
        upserts=run.upserts,
        revisions=run.revisions,
        corrections=run.corrections,
        superseded=run.superseded,
        interior_gaps=run.gaps,
        stale_fixtures=run.stale,
        work_log=run.log,
        issues=run.issues,
        latest_completed_match_date=run.latest,
        request_counts=dict(Counter(http.request_counts)),
        bytes_received=http.bytes_received,
    )


def _default_club_resolver(context: RunContext) -> ClubResolver | None:
    """Use the shared club alias registry when present (``config/team_aliases.csv``)."""
    path = context.settings.source_root / "config" / "team_aliases.csv"
    if not path.is_file():
        return None
    from supercoach_via.domain.ids import ClubRegistry

    return ClubRegistry.from_csv(path).resolve
