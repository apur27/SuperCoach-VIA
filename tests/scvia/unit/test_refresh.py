"""Refresh planning and data-only refresh (PLAN 5.2; D13-D15, missed-week + finals +
interior-correction scenario, partial/UNKNOWN outcomes, no reparse of unchanged work).

Everything is hermetic: a synthetic AFLTables site served through httpx.MockTransport.
"""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from supercoach_via.domain.schemas import CheckOutcome, DatasetStatus, MatchStatus
from supercoach_via.ingest import afltables as at
from supercoach_via.ingest import refresh as rf
from supercoach_via.ingest.http import HttpClient, RawArchive, load_policies
from supercoach_via.settings import RunContext, Settings
from tests.scvia.unit.test_afltables_support import G, M, P, Stage, match_html, player_html, season_html

REPO = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)
QA = [(2, 1), (4, 3), (6, 5), (10, 8)]  # 68
QB = [(1, 1), (3, 2), (5, 4), (8, 6)]  # 54
_REAL_PARSE_DETAIL = at.parse_match_detail  # fixture setup must not count as refresh parses


def _resolver(name: str, season: int) -> str | None:
    return name.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Synthetic source site
# ---------------------------------------------------------------------------


@dataclass
class Site:
    pages: dict[str, bytes] = field(default_factory=dict)
    status: dict[str, int] = field(default_factory=dict)
    hits: Counter[str] = field(default_factory=Counter)

    def handler(self, req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        self.hits[url] += 1
        if url in self.status:
            return httpx.Response(self.status[url])
        if url not in self.pages:
            return httpx.Response(404)
        return httpx.Response(200, content=self.pages[url])


def _players(team: str, extra_bv: str = "") -> list[P]:
    base = "Sydney" if team == "Sydney" else "Carlton"
    return [
        P(f"{base[0]}/{base}_One", f"One, {base}", "1", {"KI": "10", "HB": "5", "DI": "15", "BR": extra_bv}),
        P(f"{base[0]}/{base}_Two", f"Two, {base}", "2", {"KI": "8", "HB": "8", "DI": "16"}),
    ]


def _detail(
    label: str,
    home: str = "Sydney",
    away: str = "Carlton",
    *,
    bv: str = "",
    date_s: str = "Thu, 05-Mar-2026 7:30 PM",
    debut: bool = False,
) -> bytes:
    home_players = _players("Sydney", bv)
    if debut:
        home_players.append(P("N/New_Debut", "Debut, New", "44", {"KI": "3", "HB": "1", "DI": "4"}))
    return match_html(
        season=2026, round_label=label, home=home, away=away, home_q=QA, away_q=QB, date=date_s,
        home_players=home_players, away_players=_players("Carlton"),
    ).encode()  # fmt: skip


def _gid(n: int, d: str) -> str:
    return f"{n:04d}2026{d}"


# Source state "today": rounds 1-4, a rescheduled Round 1 match played in August,
# Wildcard + Qualifying finals complete, Grand Final scheduled (future).
SOURCE_MATCHES = [
    ("Round 1", M("Sydney", "Carlton", "Thu 05-Mar-2026 7:30 PM", "S.C.G.", QA, QB, _gid(1, "0305"))),
    ("Round 1", M("Geelong", "Hawthorn", "Sat 15-Aug-2026 1:45 PM", "Kardinia Park", QA, QB, _gid(2, "0815"))),
    ("Round 2", M("Sydney", "Carlton", "Sat 14-Mar-2026 7:30 PM", "M.C.G.", QA, QB, _gid(3, "0314"))),
    ("Round 3", M("Sydney", "Carlton", "Sat 21-Mar-2026 7:30 PM", "M.C.G.", QA, QB, _gid(4, "0321"))),
    ("Round 4", M("Sydney", "Carlton", "Sat 28-Mar-2026 7:30 PM", "M.C.G.", QA, QB, _gid(5, "0328"))),
    ("Wildcard Final", M("Sydney", "Carlton", "Fri 28-Aug-2026 7:40 PM", "M.C.G.", QA, QB, _gid(6, "0828"))),
    ("Qualifying Final", M("Geelong", "Hawthorn", "Thu 03-Sep-2026 6:10 PM", "Perth Stadium", QA, QB, _gid(7, "0903"))),
    ("Grand Final", M("Sydney", "Geelong", "Sat 26-Sep-2026 2:30 PM", "M.C.G.")),
]


def build_site(season_2025_changed: bool = False) -> Site:
    site = Site()
    stages: list[Stage] = []
    for heading, m in SOURCE_MATCHES:
        if stages and stages[-1].heading == heading:
            stages[-1].matches.append(m)
        else:
            stages.append(Stage(heading, [m]))
    site.pages[at.season_url(2026)] = season_html(2026, stages).encode()
    labels = {"Round 1": "1", "Round 2": "2", "Round 3": "3", "Round 4": "4"}
    for heading, m in SOURCE_MATCHES:
        if m.game_id is None:
            continue
        label = labels.get(heading, heading)
        date_s = m.date.replace(" ", ", ", 1)
        site.pages[at.match_url(2026, m.game_id)] = _detail(
            label, m.home, m.away, bv="2" if m.game_id == _gid(1, "0305") else "", date_s=date_s,
            debut=m.game_id == _gid(5, "0328"),
        )  # fmt: skip
    site.pages[at.player_url("N/New_Debut")] = player_html(
        "New Debut",
        "01-Feb-2007",
        [("Sydney", 2026, [G("1", "Carlton", "4", "W", "44", {"KI": "3", "HB": "1", "DI": "4"})])],
    ).encode()
    q = [(1, 1), (2, 2), (3, 3), (9, 9)]
    s25 = season_html(
        2025, [Stage("Round 1", [M("Sydney", "Carlton", "Thu 06-Mar-2025 7:30 PM", "S.C.G.", q, q, "000120250306")])]
    )
    if season_2025_changed:
        s25 = s25.replace("S.C.G.", "SCG")
    site.pages[at.season_url(2025)] = s25.encode()
    site.pages[at.match_url(2025, "000120250306")] = match_html(
        season=2025, round_label="1", home="Sydney", away="Carlton", home_q=q, away_q=q,
        date="Thu, 06-Mar-2025 7:30 PM", home_players=_players("Sydney"), away_players=_players("Carlton"),
    ).encode()  # fmt: skip
    return site


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build_base(site: Site) -> rf.BaseState:
    """Accepted snapshot taken before the missed weeks: R1 (Syd v Car), R3; R2 missing
    (interior gap); rescheduled R1 Geelong v Hawthorn still scheduled; 2025 complete; 2020 old."""
    fx26 = at.parse_season_page(site.pages[at.season_url(2026)], season=2026, club_resolver=_resolver)
    by_gid = {m.source_game_id: m for m in fx26.matches}
    matches: dict[str, rf.BaseMatch] = {}
    revisions: dict[str, str] = {}

    def add(m: at.FixtureMatch, status: str = "complete") -> None:
        matches[m.match_id] = rf.BaseMatch(
            match_id=m.match_id, season=m.season, status=status, stage_id=m.stage.stage_id,
            home_source_name=m.home_name, away_source_name=m.away_name,
            home_score=m.home_score if status == "complete" else None,
            away_score=m.away_score if status == "complete" else None,
            local_start=m.local_start,
        )  # fmt: skip

    r1 = by_gid[_gid(1, "0305")]
    add(r1)
    add(by_gid[_gid(4, "0321")])
    add(by_gid[_gid(2, "0815")], status="scheduled")
    # R1 detail as accepted before: Brownlow votes not yet populated (later correction)
    old_r1 = _detail("1", bv="")
    revisions[f"afltables:game:{_gid(1, '0305')}"] = _sha(old_r1)
    revisions[f"afltables:game:{_gid(4, '0321')}"] = _sha(site.pages[at.match_url(2026, _gid(4, "0321"))])
    fx25 = at.parse_season_page(site.pages[at.season_url(2025)], season=2025, club_resolver=_resolver)
    add(fx25.matches[0])
    revisions["afltables:season:2025"] = _sha(site.pages[at.season_url(2025)])
    revisions["afltables:game:000120250306"] = _sha(site.pages[at.match_url(2025, "000120250306")])
    matches["m:2020:r01:carlton:sydney:0"] = rf.BaseMatch(
        "m:2020:r01:carlton:sydney:0", 2020, "complete", "r01", "Sydney", "Carlton", 50, 40, "2020-03-19 19:30"
    )
    urls = {
        at.player_url(p.path): f"legacy:{p.path.split('/')[1].lower()}"
        for team in ("Sydney", "Carlton")
        for p in _players(team)
    }
    old_rows = _REAL_PARSE_DETAIL(old_r1, season=2026, game_id=_gid(1, "0305"))
    base_rows = {
        r1.match_id: [
            {"player_id": urls[p.player_url or ""], "revision_id": "rev:old", **p.stats} for p in old_rows.players
        ]
    }
    reads: Counter[str] = Counter()

    def loader(season: int, match_id: str) -> list[dict[str, object]]:
        reads[f"{season}:{match_id}"] += 1
        return base_rows.get(match_id, [])

    state = rf.BaseState(
        snapshot_id="sha256:" + "0" * 64, matches=matches, source_revisions=revisions,
        player_urls=urls, load_player_rows=loader,
    )  # fmt: skip
    state.partition_reads = reads  # type: ignore[attr-defined]
    return state


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def make_context(site: Site, tmp_path: Path) -> RunContext:
    clock = FakeClock()
    client = HttpClient(
        load_policies(REPO / "config" / "source_policies.toml"),
        user_agent="scvia-test",
        archive=RawArchive(tmp_path / "raw"),
        transport=httpx.MockTransport(site.handler),
        resolver=lambda h: ["93.184.215.14"],
        monotonic=clock,
        sleep=clock.sleep,
        now=lambda: NOW,
        jitter=lambda: 0.5,
    )
    ctx = RunContext(settings=Settings(data_root=tmp_path, season=2026), clock=lambda: NOW)
    ctx.http = client
    return ctx


def run(site: Site, tmp_path: Path, **req: object) -> tuple[rf.RefreshResult, rf.BaseState]:
    base = build_base(site)
    ctx = make_context(site, tmp_path)
    plan = rf.plan_refresh(base, rf.RefreshRequest(**req), ctx)  # type: ignore[arg-type]
    return rf.refresh_sources(base, plan, ctx, club_resolver=_resolver), base


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def test_plan_is_offline_fast_and_describes_work(tmp_path: Path) -> None:
    site = build_site()
    base = build_base(site)
    ctx = RunContext(settings=Settings(data_root=tmp_path, season=2026), clock=lambda: NOW)  # no http client
    t0 = time.perf_counter()
    plan = rf.plan_refresh(base, rf.RefreshRequest(), ctx)
    assert time.perf_counter() - t0 < 3.0
    assert site.hits == Counter()
    assert plan.seasons == (2025, 2026) and plan.overlap_seasons == (2025, 2026)
    assert [w.url for w in plan.work] == [at.season_url(2025), at.season_url(2026)]
    assert plan.estimated_requests["min"] == 2 and plan.estimated_requests["max"] >= 2 + 3
    assert "matches" in plan.outputs and "player_games" in plan.outputs
    text = plan.describe()
    assert "2025" in text and "network: none during planning" in text
    assert plan.to_dict()["base_snapshot_id"] == base.snapshot_id


def test_plan_catch_up_spans_every_season_since_base(tmp_path: Path) -> None:
    site = build_site()
    base = build_base(site)
    base.matches = {k: v for k, v in base.matches.items() if v.season <= 2020}
    ctx = RunContext(settings=Settings(data_root=tmp_path, season=2026), clock=lambda: NOW)
    plan = rf.plan_refresh(base, rf.RefreshRequest(), ctx)
    assert plan.seasons == tuple(range(2020, 2027))
    assert plan.catch_up_seasons == tuple(range(2020, 2025))


def test_plan_repair_season_and_validation(tmp_path: Path) -> None:
    base = build_base(build_site())
    ctx = RunContext(settings=Settings(data_root=tmp_path, season=2026), clock=lambda: NOW)
    plan = rf.plan_refresh(base, rf.RefreshRequest(repair_season=2020), ctx)
    assert 2020 in plan.seasons and plan.repair_seasons == (2020,)
    with pytest.raises(ValueError):
        rf.plan_refresh(base, rf.RefreshRequest(repair_season=1700), ctx)


# ---------------------------------------------------------------------------
# Data-only refresh: missed weeks + finals + interior gap + correction
# ---------------------------------------------------------------------------


def test_missed_week_finals_interior_gap_and_correction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site = build_site()
    parse_calls: Counter[str] = Counter()
    real_parse = at.parse_match_detail

    def counting(content: bytes | str, *, season: int, game_id: str) -> at.MatchDetail:
        parse_calls[game_id] += 1
        return real_parse(content, season=season, game_id=game_id)

    monkeypatch.setattr(rf.at, "parse_match_detail", counting)
    result, base = run(site, tmp_path)

    assert result.outcome is CheckOutcome.PASS, result.issues
    assert result.exit_code == 0 and result.dataset_status is DatasetStatus.VERIFIED
    assert result.promotable_as_verified

    # never touched a season outside the plan
    assert not any("/2020" in u for u in site.hits)

    upserted = {r["match_id"]: r for r in result.upserts["matches"]}
    new_ids = set(upserted) - set(base.matches)
    stages = {upserted[m]["stage_id"] for m in new_ids}
    assert {"r02", "r04", "wf", "qf", "gf"} <= stages
    # rescheduled early-numbered round discovered (was scheduled in base, now complete)
    resched = next(r for r in upserted.values() if r["home_source_name"] == "Geelong" and r["stage_id"] == "r01")
    assert resched["status"] == "complete" and resched["home_score"] == 68
    # future fixture stays scheduled with no fake score
    gf = next(r for r in upserted.values() if r["stage_id"] == "gf")
    assert gf["status"] == "scheduled" and gf["home_score"] is None and gf["away_score"] is None
    # interior gap (R2 missing between accepted R1 and R3)
    assert any(g["stage_id"] == "r02" for g in result.interior_gaps)
    assert result.latest_completed_match_date == date(2026, 9, 3)

    # D15: unchanged-count stat correction (Brownlow votes populated later) detected
    corr = [c for c in result.corrections if c["column"] == "brownlow_votes"]
    assert corr and corr[0]["old"] is None and corr[0]["new"] == 2
    # D13: corrected row gets a new revision; previous revision retained as superseded
    assert any(s["old_revision_id"] == "rev:old" for s in result.superseded)
    r1_rows = [
        r for r in result.upserts["player_games"] if r["match_id"] == resched["match_id"] or r["stage_id"] == "r01"
    ]
    assert all(r["revision_id"].startswith("rev:") and r["revision_id"] != "rev:old" for r in r1_rows)

    # D14: new debut discovered from the match page, validated against the player page
    debut = [p for p in result.upserts["players"] if p["display_name"] == "New Debut"]
    assert debut and debut[0]["birth_date"] == date(2007, 2, 1)
    assert site.hits[at.player_url("N/New_Debut")] == 1

    # unchanged work is not reparsed: R3 and 2025 details were re-checked but identical
    assert parse_calls[_gid(4, "0321")] == 0 and parse_calls["000120250306"] == 0
    assert parse_calls[_gid(1, "0305")] == 1
    # base partitions read only for matches whose payload changed
    reads = base.partition_reads  # type: ignore[attr-defined]
    changed_payload = {r["match_id"] for r in result.upserts["player_games"]}
    assert set(reads) <= {f"2026:{m}" for m in changed_payload} and all(v == 1 for v in reads.values())
    assert not any(k.startswith("2025:") for k in reads)

    c = result.counts["match_detail"]
    assert c.required == c.attempted == c.succeeded + c.unchanged
    assert c.failed == 0 and c.quarantined == 0 and c.unchanged == 2
    assert result.counts["season_fixture"].succeeded == 2
    assert {o["outcome"] for o in result.upserts["source_observations"]} == {"PASS"}


def test_mandatory_season_failure_is_partial_unknown_exit_3(tmp_path: Path) -> None:
    site = build_site()
    site.status[at.season_url(2026)] = 503
    result, _ = run(site, tmp_path)
    assert result.outcome is CheckOutcome.UNKNOWN and result.exit_code == 3
    assert result.dataset_status is DatasetStatus.PARTIAL and not result.promotable_as_verified
    assert result.counts["season_fixture"].failed == 1
    assert result.latest_completed_match_date is None or result.latest_completed_match_date < date(2026, 1, 1)


def test_detail_failure_is_partial_and_recorded(tmp_path: Path) -> None:
    site = build_site()
    site.status[at.match_url(2026, _gid(6, "0828"))] = 404
    result, _ = run(site, tmp_path)
    assert result.exit_code == 3 and not result.promotable_as_verified
    assert result.counts["match_detail"].failed == 1
    failed = [w for w in result.work_log if w["outcome"] != "PASS" and w["status"] == "failed"]
    assert any(_gid(6, "0828") in w["key"] for w in failed)
    assert not any(r["match_id"].endswith("wf") for r in result.upserts["player_games"])


def test_parser_drift_in_detail_is_quarantined_not_success(tmp_path: Path) -> None:
    site = build_site()
    site.pages[at.match_url(2026, _gid(5, "0328"))] = b"<html><body>layout changed</body></html>"
    result, _ = run(site, tmp_path)
    assert result.exit_code == 3 and result.counts["match_detail"].quarantined == 1
    assert result.upserts["quarantine"]


def test_detail_disagreeing_with_fixture_is_quarantined(tmp_path: Path) -> None:
    site = build_site()
    url = at.match_url(2026, _gid(3, "0314"))
    site.pages[url] = site.pages[url].replace(b"<td>10.8.68</td>", b"<td>11.8.74</td>")
    result, _ = run(site, tmp_path)
    assert result.counts["match_detail"].quarantined == 1 and result.exit_code == 3


def test_debut_not_confirmed_by_player_page_is_quarantined(tmp_path: Path) -> None:
    site = build_site()
    site.status[at.player_url("N/New_Debut")] = 503
    result, _ = run(site, tmp_path)
    assert result.exit_code == 3
    assert result.counts["player_page"].failed == 1
    assert not any(p["display_name"] == "New Debut" for p in result.upserts["players"])


def test_repair_season_forces_full_reparse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site = build_site()
    calls: Counter[str] = Counter()
    real_parse = at.parse_match_detail

    def counting(content: bytes | str, *, season: int, game_id: str) -> at.MatchDetail:
        calls[game_id] += 1
        return real_parse(content, season=season, game_id=game_id)

    monkeypatch.setattr(rf.at, "parse_match_detail", counting)
    result, _ = run(site, tmp_path, repair_season=2026)
    assert calls[_gid(4, "0321")] == 1  # unchanged payload still reconciled in repair mode
    assert result.plan.repair_seasons == (2026,)


def test_no_http_client_is_config_error(tmp_path: Path) -> None:
    site = build_site()
    base = build_base(site)
    ctx = RunContext(settings=Settings(data_root=tmp_path, season=2026), clock=lambda: NOW)
    plan = rf.plan_refresh(base, rf.RefreshRequest(), ctx)
    with pytest.raises(rf.RefreshConfigError):
        rf.refresh_sources(base, plan, ctx)


def test_base_state_from_real_snapshot(tmp_path: Path) -> None:
    import pyarrow as pa

    from supercoach_via.domain.schemas import TABLES, DatasetStatus, ValidationReport
    from supercoach_via.storage.snapshots import SnapshotBuilder, promote

    fx = at.parse_season_page(build_site().pages[at.season_url(2026)], season=2026, club_resolver=_resolver)
    rows = [m.to_row(source_ref="s", source_sha256=None) for m in fx.matches[:3]]
    b = SnapshotBuilder(tmp_path, clock=lambda: NOW, code_version="t")
    b.add_partitioned("matches", pa.Table.from_pylist(rows, schema=TABLES["matches"].arrow_schema()), "season")
    cand = b.finish(status=DatasetStatus.LEGACY_UNVERIFIED, source_revisions={"afltables:season:2026": "a" * 64})
    promote(tmp_path, cand, ValidationReport(outcome=CheckOutcome.PASS))
    state = rf.base_state_from_snapshot(tmp_path)
    assert set(state.matches) == {r["match_id"] for r in rows}
    assert state.source_revisions["afltables:season:2026"] == "a" * 64
    assert state.matches[rows[0]["match_id"]].status == MatchStatus.COMPLETE.value
    assert state.load_player_rows is not None and state.load_player_rows(2026, rows[0]["match_id"]) == []


# ---------------------------------------------------------------------------
# Owner-bounded refresh: new matches only + a hard request budget
# ---------------------------------------------------------------------------


def test_new_matches_only_fetches_current_season_and_only_new_or_changed_matches(tmp_path: Path) -> None:
    site = build_site()
    result, _base = run(site, tmp_path, overlap_seasons=1, recheck_unchanged=False)
    assert result.plan.work and [w.url for w in result.plan.work] == [at.season_url(2026)]
    fetched = set(site.hits)
    new_or_changed = {at.match_url(2026, _gid(n, d)) for n, d in
                      ((2, "0815"), (3, "0314"), (5, "0328"), (6, "0828"), (7, "0903"))}  # fmt: skip
    assert fetched == {at.season_url(2026), *new_or_changed, at.player_url("N/New_Debut")}
    # unchanged accepted matches are not re-checked (their later corrections wait for a full refresh)
    assert at.match_url(2026, _gid(1, "0305")) not in fetched and at.match_url(2026, _gid(4, "0321")) not in fetched
    assert result.outcome is CheckOutcome.PASS and result.exit_code == 0, result.issues
    assert sum(site.hits.values()) == len(fetched)


def test_request_budget_is_hard_and_fails_closed(tmp_path: Path) -> None:
    site = build_site()
    result, _base = run(site, tmp_path, overlap_seasons=1, recheck_unchanged=False, max_requests=3)
    assert sum(site.hits.values()) == 3
    assert result.exit_code == 3 and result.dataset_status is DatasetStatus.PARTIAL
    assert not result.promotable_as_verified
    assert any("request budget" in i for i in result.issues)


def test_request_budget_must_be_positive(tmp_path: Path) -> None:
    base = build_base(build_site())
    ctx = RunContext(settings=Settings(data_root=tmp_path, season=2026), clock=lambda: NOW)
    with pytest.raises(rf.RefreshConfigError):
        rf.plan_refresh(base, rf.RefreshRequest(max_requests=0), ctx)
