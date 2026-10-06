"""A fake afltables.com for capture tests: ``httpx.MockTransport`` plus a controllable clock."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

from supercoach_via.ingest.http import HttpClient, PolicySet
from supercoach_via.reconciliation import inventory as inv
from supercoach_via.reconciliation import schema as S
from supercoach_via.reconciliation.urls import load_reconciliation_policies
from tests.scvia.unit import integrity_fixtures as fx
from tests.scvia.unit import recon_world as rw

ERROR_PAGE = b"<html><head><title>Broked!</title></head><body><h2>This page has been sent off</h2></body></html>"


class FakeClock:
    def __init__(self, start: float = 1_790_000_000.0) -> None:
        self.t = start
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def time(self) -> float:
        return self.t

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.t += max(seconds, 0.0)

    def now(self) -> datetime:
        return datetime.fromtimestamp(self.t, UTC)


Responder = Callable[[httpx.Request], httpx.Response]


class FakeSite:
    """Serves ``pages`` (path -> body); ``override`` maps a path to a response factory or a list consumed in order."""

    def __init__(self, clock: FakeClock, pages: dict[str, bytes]) -> None:
        self.clock = clock
        self.pages = dict(pages)
        self.override: dict[str, Responder | list[httpx.Response]] = {}
        self.log: list[tuple[float, str, dict[str, str]]] = []
        self.latency = 0.3

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.log.append((self.clock.time(), path, dict(request.headers)))
        self.clock.t += self.latency
        ov = self.override.get(path)
        if callable(ov):
            return ov(request)
        if isinstance(ov, list) and ov:
            return ov.pop(0)
        if path == "/robots.txt" and path not in self.pages:
            return httpx.Response(404, content=ERROR_PAGE)
        body = self.pages.get(path)
        if body is None:
            return httpx.Response(404, content=ERROR_PAGE)
        return httpx.Response(200, content=body)

    def paths(self) -> list[str]:
        return [p for _, p, _ in self.log]

    def times(self) -> list[float]:
        return [t for t, _, _ in self.log]

    def client(self, policies: PolicySet | None = None, user_agent: str = "supercoach-via-test/1.0") -> HttpClient:
        return HttpClient(
            policies or load_reconciliation_policies(),
            user_agent=user_agent,
            archive=None,
            transport=httpx.MockTransport(self.handler),
            resolver=lambda _h: ["93.184.215.14"],
            monotonic=self.clock.monotonic,
            sleep=self.clock.sleep,
            now=self.clock.now,
            jitter=lambda: 0.5,
        )


def small_world() -> rw.World:
    """Three players, two seasons, three matches (one in 2025, one in 2026, one after the boundary)."""
    players = {
        "a": rw.P("Ann_Able", "Ann", "Able", "01-Jul-1990"),
        "b": rw.P("Bob_Baker", "Bob", "Baker", "02-Aug-1991"),
        "c": rw.P("Cy_Cole", "Cy", "Cole", "03-Sep-1992"),
    }

    def apps(k: int) -> tuple[rw.A, ...]:
        cells = {"kicks": 5 + k, "handballs": 3, "disposals": 8 + k, "goals": 1, "time_on_ground_pct": 80.0}
        return (
            rw.A("a", "Alpha", "1", dict(cells)),
            rw.A("b", "Alpha", "2", dict(cells)),
            rw.A("c", "Beta", "3", dict(cells)),
        )

    ms = (
        rw.M("041520250315", 2025, "1", "Alpha", "Beta", date(2025, 3, 15), apps=apps(0)),
        rw.M("041520260305", 2026, "1", "Alpha", "Beta", date(2026, 3, 5), apps=apps(1)),
        rw.M("041520261105", 2026, "2", "Beta", "Alpha", date(2026, 11, 5), apps=apps(2)),
    )
    return rw.World(players, ms, (2025, 2026))


def site_pages(w: rw.World, *, seasons: tuple[int, ...] = (2025, 2026)) -> dict[str, bytes]:
    pages: dict[str, bytes] = {
        "/afl/stats/stats_idx.html": rw.stats_index(list(seasons)),
        "/afl/stats/notes.html": rw.notes_page(),
    }
    by_letter: dict[str, list[rw.P]] = {chr(c): [] for c in range(ord("A"), ord("Z") + 1)}
    for p in w.players.values():
        by_letter[p.last[0].upper()].append(p)
    for letter, ps in by_letter.items():
        pages[f"/afl/stats/players{letter}_idx.html"] = rw.census_page(letter, ps)
    for y in seasons:
        pages[f"/afl/seas/{y}.html"] = rw.season_page(y, [m for m in w.matches if m.year == y])
    to_date = w.career_to_date()
    for m in w.matches:
        career = {a.pid: to_date[(m.gid, a.pid)] for a in m.apps}
        pages[f"/afl/stats/games/{m.year}/{m.gid}.html"] = rw.match_page(m, w.players, career=career)
    for pid, p in w.players.items():
        pages[f"/afl/stats/players/{p.slug[0].upper()}/{p.slug}.html"] = rw.profile_page(w, pid)
    return pages


_BASE: dict[str, tuple[Path, S.Plan]] = {}


def _base_plan(through: str) -> tuple[Path, S.Plan]:
    """One tiny snapshot per interpreter (building and hashing it dominates a capture test)."""
    if through not in _BASE:
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="recon-base-"))
        fx.build(root / "data")
        plan = inv.build_plan(
            data_root=root / "data",
            snapshot="current",
            legacy_root=None,
            through_date=through,
            scope="all",
            run_dir=root / "run",
        )
        _BASE[through] = (root, plan)
    return _BASE[through]


def make_plan(tmp_path: Path, *, first_season: int = 2025, through: str = "2026-09-30", run: str = "run") -> S.Plan:
    _root, plan = _base_plan(through)
    scope = plan.scope.model_copy(update={"first_season": first_season})
    oper = plan.operational.model_copy(update={"run_dir": str(tmp_path / run)})
    return plan.model_copy(update={"scope": scope, "operational": oper})
