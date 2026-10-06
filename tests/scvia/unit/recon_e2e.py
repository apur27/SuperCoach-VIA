"""End-to-end fixture: one world -> captured source pages + snapshot + legacy CSVs -> plan -> compare."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from supercoach_via.reconciliation import compare as CP
from supercoach_via.reconciliation import inventory as inv
from supercoach_via.reconciliation import report as RP
from supercoach_via.reconciliation import schema as S
from supercoach_via.reconciliation.capture import Capture
from tests.scvia.unit import integrity_fixtures as fx
from tests.scvia.unit import recon_world as rw
from tests.scvia.unit import snapshot_factory
from tests.scvia.unit.recon_site import FakeClock, FakeSite, site_pages

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "reconciliation"
STAGE_ID = {
    "Qualifying Final": "qf",
    "Elimination Final": "ef",
    "Semi Final": "sf",
    "Preliminary Final": "pf",
    "Grand Final": "gf",
}
LEGACY_HEADER = (
    "team,year,games_played,opponent,round,result,jersey_num,kicks,marks,handballs,disposals,goals,behinds,hit_outs,"
    "tackles,rebound_50s,inside_50s,clearances,clangers,free_kicks_for,free_kicks_against,brownlow_votes,"
    "contested_possessions,uncontested_possessions,contested_marks,marks_inside_50,one_percenters,bounces,"
    "goal_assist,percentage_of_game_played,date\n"
)
LEGACY_FIELDS = [
    "kicks", "marks", "handballs", "disposals", "goals", "behinds", "hitouts", "tackles", "rebound_50s", "inside_50s",
    "clearances", "clangers", "frees_for", "frees_against", "brownlow_votes", "contested_possessions",
    "uncontested_possessions", "contested_marks", "marks_inside_50", "one_percenters", "bounces", "goal_assists",
    "time_on_ground_pct",
]  # fmt: skip


def _club_id(name: str) -> str:
    return name.lower().replace(" ", "_")


def match_row(m: rw.M, order: int) -> dict[str, Any]:
    final = m.is_final
    row: dict[str, Any] = {
        "match_id": f"m:{m.year}:{STAGE_ID.get(m.stage, 'r' + m.stage)}:{_club_id(m.home)}:{_club_id(m.away)}:0",
        "season": m.year,
        "stage_label": m.stage,
        "stage_type": "final" if final else "regular",
        "round_number": None if final else int(m.stage),
        "stage_order": order,
        "stage_id": STAGE_ID.get(m.stage, f"r{int(m.stage):02d}" if not final else "gf"),
        "replay_occurrence": 0,
        "home_club_id": _club_id(m.home),
        "away_club_id": _club_id(m.away),
        "home_source_name": m.home,
        "away_source_name": m.away,
        "venue_id": "oval",
        "venue_source_name": m.venue,
        "local_start": f"{m.when.isoformat()} 14:10",
        "match_date": m.when,
        "date_precision": "minute",
        "status": "complete",
        "attendance": 1000,
        "provenance": "legacy_import",
        "source_path": f"data/matches/matches_{m.year}.csv",
        "source_sha256": "0" * 64,
        "source_row": order,
    }
    for side, qs in (("home", m.hq), ("away", m.aq)):
        for q, (g, b) in zip(("q1", "q2", "q3", "final"), qs, strict=True):
            row[f"{side}_{q}_goals"], row[f"{side}_{q}_behinds"] = g, b
        row[f"{side}_score"] = m.score(side)
    return row


def snapshot_tables(
    w: rw.World,
    rows_by_season: dict[int, list[Any]],
    *,
    replay_of: dict[str, int] | None = None,
    drop_players: frozenset[str] = frozenset(),
    drop_matches: frozenset[str] = frozenset(),
) -> dict[str, list[dict[str, Any]]]:
    mrows = []
    ids: dict[str, str] = {}
    for i, m in enumerate(sorted(w.matches, key=lambda x: (x.when, x.gid)), 1):
        row = match_row(m, i)
        if m.gid in drop_matches:
            continue
        if replay_of and m.gid in replay_of:
            row["replay_occurrence"] = replay_of[m.gid]
            row["match_id"] = row["match_id"][:-1] + str(replay_of[m.gid])
        mrows.append(row)
        ids[m.gid] = row["match_id"]
    players = []
    for pid, p in w.players.items():
        if pid in drop_players:
            continue
        d = datetime.strptime(p.born, "%d-%b-%Y").date() if p.born else None
        players.append(
            {
                "player_id": f"legacy:{pid}", "legacy_slug": pid, "display_name": p.display, "first_name": p.first,
                "last_name": p.last, "birth_date": d, "birth_date_quality": "source" if d else "unknown",
                "identity_status": "canonical", "provenance": "legacy_import", "source_urls": "[]",
            }
        )  # fmt: skip
    games: list[dict[str, Any]] = []
    for season, rows in sorted(rows_by_season.items()):
        for r in rows:
            gid = r.match_key
            games.append(game_row(r, ids.get((gid or "").removeprefix("m:"), gid), season))
    return {
        "players": players,
        "clubs": [
            {"club_id": _club_id(c), "name": c, "lineage_id": _club_id(c), "first_season": 1900, "active": True}
            for c in ("Alpha", "Beta")
        ],
        "venues": [{"venue_id": "oval", "name": "Oval", "source_names": '["Oval"]'}],
        "seasons": [{"season": y, "matches_complete": 1, "matches_scheduled": 0} for y in w.seasons],
        "matches": mrows,
        "player_games": games,
        "quality_issues": [],
        "quarantine": [],
        "player_aliases": [],
        "source_files": [],
        "source_observations": [],
    }


def game_row(r: Any, match_id: str, season: int) -> dict[str, Any]:
    from supercoach_via.reconciliation.schema import STAT_FIELDS

    row = {
        "match_id": match_id, "player_id": f"legacy:{r.player_key}", "club_id": _club_id(r.club), "season": season,
        "opponent_club_id": _club_id(r.opponent or ""), "stage_label": r.stage, "stage_id": r.stage.lower(),
        "club_source_name": r.club, "opponent_source_name": r.opponent, "link_method": "key", "match_date": date.fromisoformat(r.match_date),
        "date_quality": "fixture_verified", "career_game_counter": r.counter, "career_game_counter_token": r.counter_token,
        "result": r.result, "jersey_number": int(r.jersey), "revision_id": "rev:x", "provenance": "legacy_import",
        "source_path": "data/player_data/x.csv", "source_sha256": "1" * 64, "source_row": r.counter,
    }  # fmt: skip
    for f, v in zip(STAT_FIELDS, r.cells, strict=True):
        row[f] = v
    return row


def legacy_files(
    root: Path,
    w: rw.World,
    rows_by_season: dict[int, list[Any]],
    drop_players: frozenset[str] = frozenset(),
    drop_matches: frozenset[str] = frozenset(),
) -> None:
    pdir = root / "data" / "player_data"
    mdir = root / "data" / "matches"
    pdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    per_player: dict[str, list[Any]] = {}
    for rows in rows_by_season.values():
        for r in rows:
            per_player.setdefault(r.player_key.removeprefix("slug_"), []).append(r)
    for pid, p in w.players.items():
        if pid in drop_players:
            continue
        born = datetime.strptime(p.born, "%d-%b-%Y")
        slug = f"{p.last.lower()}_{p.first.lower()}_{born.strftime('%d%m%Y')}"
        lines = [LEGACY_HEADER]
        for r in sorted(per_player.get(pid, []), key=lambda x: x.counter or 0):
            cells = ",".join(c for c in r.raw_cells)
            lines.append(
                f"{r.club},{r.season},{r.counter},{r.opponent},{r.stage},{r.result},{r.jersey},{cells},{r.match_date}\n"
            )
        (pdir / f"{slug}_performance_details.csv").write_text("".join(lines))
        (pdir / f"{slug}_personal_details.csv").write_text(
            f"first_name,last_name,born_date,debut_date,height,weight\n{p.first},{p.last},{born.strftime('%d-%m-%Y')},01-01-2010,180,80\n"
        )
    by_year: dict[int, list[rw.M]] = {}
    for m in w.matches:
        by_year.setdefault(m.year, []).append(m)
    for y, ms in by_year.items():
        out = [
            "round_num,venue,date,year,attendance,team_1_team_name,team_1_q1_goals,team_1_q1_behinds,team_1_q2_goals,team_1_q2_behinds,team_1_q3_goals,team_1_q3_behinds,team_1_final_goals,team_1_final_behinds,team_2_team_name,team_2_q1_goals,team_2_q1_behinds,team_2_q2_goals,team_2_q2_behinds,team_2_q3_goals,team_2_q3_behinds,team_2_final_goals,team_2_final_behinds\n"
        ]
        for m in sorted(ms, key=lambda x: x.when):
            if m.gid in drop_matches:
                continue
            h = ",".join(f"{g},{b}" for g, b in m.hq)
            a = ",".join(f"{g},{b}" for g, b in m.aq)
            out.append(f"{m.stage},{m.venue},{m.when.isoformat()} 14:10,{y},1000,{m.home},{h},{m.away},{a}\n")
        (mdir / f"matches_{y}.csv").write_text("".join(out))


@dataclass
class E2E:
    root: Path
    world: rw.World
    plan: S.Plan
    plan_path: Path
    manifest_path: Path
    data_root: Path
    legacy_root: Path | None
    site: FakeSite
    capture_result: Any

    def options(
        self, out: str = "reports/r1", *, workers: int = 1, cache: str = "cache", **kw: Any
    ) -> CP.CompareOptions:
        return CP.CompareOptions(
            plan=self.plan_path, capture_manifest=self.manifest_path, out=self.root / "run" / out, workers=workers,
            cache=self.root / "run" / cache, **kw,
        )  # fmt: skip

    def compare(self, out: str = "reports/r1", **kw: Any) -> tuple[CP.AuditResult, int, Path]:
        opts = self.options(out, **kw)
        result, audit = CP.run_audit(opts)
        RP.write_report_dir(opts.out, result, CP.input_roots_of(audit.plan, audit.capture_dir))
        CP.cleanup(audit)
        return result, result.exit_code, opts.out


def build(
    root: Path,
    world: rw.World,
    *,
    local_rows: Callable[[str, int], list[Any]] | None = None,
    through: str = "2026-09-30",
    legacy: bool = True,
    page_edit: Callable[[dict[str, bytes]], None] | None = None,
    monkeypatch: Any = None,
    capture: bool = True,
    replay_of: dict[str, int] | None = None,
    drop_players: frozenset[str] = frozenset(),
    drop_matches: frozenset[str] = frozenset(),
) -> E2E:
    from tests.scvia.unit import recon_inputs as RI

    first = min(w.year for w in world.matches)
    if monkeypatch is not None:
        monkeypatch.setattr("supercoach_via.reconciliation.schema.FIRST_SEASON", first)
    seasons = tuple(sorted({m.year for m in world.matches}))
    rows_by_season = {y: (local_rows("snapshot", y) if local_rows else RI.local_rows(world, y)) for y in seasons}
    legacy_rows = {
        y: (local_rows("legacy_csv", y) if local_rows else RI.local_rows(world, y, "legacy_csv")) for y in seasons
    }
    data_root = root / "data"
    snapshot_factory.build(
        data_root,
        lambda: fx.CLOCK,
        snapshot_tables(
            world, rows_by_season, replay_of=replay_of, drop_players=drop_players, drop_matches=drop_matches
        ),
    )
    legacy_root = root / "legacy" if legacy else None
    if legacy_root is not None:
        legacy_files(legacy_root, world, legacy_rows, drop_players, drop_matches)
    plan = inv.build_plan(
        data_root=data_root, snapshot="current", legacy_root=legacy_root, through_date=through, scope="all",
        run_dir=root / "run",
    )  # fmt: skip
    plan_path = inv.write_plan(plan)
    # the pinned Brownlow award evidence (DESIGN section 15, A9) lives in the run's own evidence archive
    from supercoach_via.ingest.http import RawArchive

    RawArchive(root / "run" / "evidence").put((FIXTURES / "brownlow_idx.html").read_bytes())
    clock = FakeClock()
    pages = site_pages(world, seasons=seasons)
    pages["/afl/stats/notes.html"] = (FIXTURES / "notes.html").read_bytes()
    if page_edit is not None:
        page_edit(pages)
    site = FakeSite(clock, pages)
    cap_result = None
    if capture:
        cap = Capture(plan, root / "run", site.client(), clock=clock, durable=False)
        cap_result = cap.run()
    return E2E(
        root,
        world,
        plan,
        plan_path,
        root / "run" / "capture" / "manifest.json",
        data_root,
        legacy_root,
        site,
        cap_result,
    )
