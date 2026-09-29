"""Small, fully consistent canonical corpus for integrity-checker tests (test helper).

Two seasons (1970, legacy era without tackles/%P; 2026, the current season), two clubs,
four players. Every aggregate row agrees with the facts, so the clean corpus must audit
PASS and each negative test changes exactly one thing. ``rehash`` rewrites fragments and
manifests with recalculated hashes and a recalculated snapshot id, so a semantic check
rather than SHA-256 has to catch the change.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from supercoach_via.domain.schemas import PLAYER_STAT_COLUMNS, TABLES, CheckOutcome, DatasetStatus, ValidationReport
from supercoach_via.storage import snapshots
from tests.scvia.unit import snapshot_factory

CLOCK = datetime(2026, 9, 28, 1, 0, tzinfo=UTC)
AS_OF = "2026-09-28T12:00:00Z"
SEASON_URL = "https://afltables.com/afl/seas/2026.html"

# every reported count is 1; Brownlow votes are not yet reported (null), pre-era cells are null
MODERN_STATS = {s: 1 for s in PLAYER_STAT_COLUMNS} | {"time_on_ground_pct": 80.0, "brownlow_votes": None}
# 1970: only the stats config/coverage.yaml records from 1965/1966 exist (no tackles, %P...)
RECORDED_BY_1970 = ("kicks", "marks", "handballs", "goals", "behinds", "hitouts", "frees_for", "frees_against")
OLD_STATS = {s: None for s in PLAYER_STAT_COLUMNS} | dict.fromkeys(RECORDED_BY_1970, 1) | {"disposals": 2}


def _match(
    mid: str,
    season: int,
    stage: str,
    order: int,
    d: date,
    home: str,
    away: str,
    hq: tuple[int, int],
    aq: tuple[int, int],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "match_id": mid,
        "season": season,
        "stage_label": stage,
        "stage_type": "regular",
        "round_number": order,
        "stage_order": order,
        "stage_id": f"r{order:02d}",
        "replay_occurrence": 0,
        "home_club_id": home,
        "away_club_id": away,
        "home_source_name": home.title(),
        "away_source_name": away.title(),
        "venue_id": "oval",
        "venue_source_name": "Oval",
        "local_start": f"{d.isoformat()} 14:10",
        "match_date": d,
        "date_precision": "minute",
        "status": "complete",
        "attendance": 1000,
        "provenance": "legacy_import",
        "source_path": f"data/matches/matches_{season}.csv",
        "source_sha256": "0" * 64,
        "source_row": order,
    }
    for side, (g, b) in (("home", hq), ("away", aq)):
        for i, q in enumerate(("q1", "q2", "q3", "final"), 1):
            row[f"{side}_{q}_goals"] = g * i // 4
            row[f"{side}_{q}_behinds"] = b * i // 4
        row[f"{side}_score"] = 6 * g + b
    return row


def _game(match: dict[str, Any], player: str, club: str, counter: int, stats: dict[str, Any]) -> dict[str, Any]:
    home = match["home_club_id"] == club
    opp = match["away_club_id"] if home else match["home_club_id"]
    own, other = (match["home_score"], match["away_score"]) if home else (match["away_score"], match["home_score"])
    return {
        "match_id": match["match_id"],
        "player_id": player,
        "club_id": club,
        "season": match["season"],
        "opponent_club_id": opp,
        "stage_label": match["stage_label"],
        "stage_id": match["stage_id"],
        "club_source_name": club.title(),
        "opponent_source_name": opp.title(),
        "link_method": "key",
        "match_date": match["match_date"],
        "date_quality": "fixture_verified",
        "career_game_counter": counter,
        "career_game_counter_token": str(counter),
        "result": "W" if own > other else "L" if own < other else "D",
        "jersey_number": 1,
        "revision_id": "rev:legacy",
        "provenance": "legacy_import",
        "source_path": f"data/player_data/{player}.csv",
        "source_sha256": "1" * 64,
        "source_row": counter,
        **stats,
    }


def tables() -> dict[str, list[dict[str, Any]]]:
    # two goals per side: each club's two players kick one each
    m1 = _match("m:1970:r01:alpha:beta:0", 1970, "1", 1, date(1970, 4, 4), "alpha", "beta", (2, 5), (2, 7))
    m2 = _match("m:1970:r02:alpha:beta:0", 1970, "2", 2, date(1970, 4, 11), "beta", "alpha", (2, 9), (2, 9))
    m3 = _match("m:2026:r01:alpha:beta:0", 2026, "1", 1, date(2026, 3, 5), "alpha", "beta", (2, 10), (2, 4))
    m4 = _match("m:2026:r02:alpha:beta:0", 2026, "2", 2, date(2026, 3, 12), "beta", "alpha", (2, 7), (2, 6))
    games = []
    for m, stats, base in ((m1, OLD_STATS, 1), (m2, OLD_STATS, 2), (m3, MODERN_STATS, 3), (m4, MODERN_STATS, 4)):
        for player, club in (
            ("legacy:p1", "alpha"),
            ("legacy:p2", "alpha"),
            ("legacy:p3", "beta"),
            ("legacy:p4", "beta"),
        ):
            games.append(_game(m, player, club, base, dict(stats)))
    for g in games:
        if g["season"] == 2026:
            g["disposals"] = g["kicks"] + g["handballs"]
    players = [
        {
            "player_id": f"legacy:p{i}",
            "legacy_slug": f"p{i}",
            "display_name": f"Player {i}",
            "first_name": "Player",
            "last_name": str(i),
            "birth_date_quality": "unknown",
            "identity_status": "canonical",
            "provenance": "legacy_import",
            "source_urls": "[]",
        }
        for i in range(1, 5)
    ]
    clubs = [
        {"club_id": c, "name": c.title(), "lineage_id": c, "first_season": 1900, "last_season": None, "active": True}
        for c in ("alpha", "beta")
    ]
    seasons = [
        {
            "season": 1970,
            "first_match_date": date(1970, 4, 4),
            "last_match_date": date(1970, 4, 11),
            "matches_complete": 2,
            "matches_scheduled": 0,
        },
        {
            "season": 2026,
            "first_match_date": date(2026, 3, 5),
            "last_match_date": date(2026, 3, 12),
            "matches_complete": 2,
            "matches_scheduled": 0,
            "fixture_checked_at": datetime(2026, 3, 13, 0, 0, tzinfo=UTC),
        },
    ]
    observations = [
        {
            "source_ref": "src:afltables:season2026",
            "adapter": "afltables.season_fixture",
            "adapter_version": "1",
            "url": SEASON_URL,
            "fetched_at": datetime(2026, 3, 13, 0, 0, tzinfo=UTC),
            "content_sha256": "a" * 64,
            "http_status": 200,
            "bytes": 10,
            "source_mode": "live",
            "outcome": "PASS",
        },
    ]
    return {
        "players": players,
        "clubs": clubs,
        "venues": [{"venue_id": "oval", "name": "Oval", "source_names": '["Oval"]'}],
        "seasons": seasons,
        "matches": [m1, m2, m3, m4],
        "player_games": games,
        "quality_issues": [],
        "quarantine": [],
        "source_files": [
            {
                "path": "data/matches/matches_2026.csv",
                "family": "matches",
                "bytes": 1,
                "sha256": "0" * 64,
                "disposition": "imported",
            }
        ],
        "source_observations": observations,
    }


def build(root: Path, rows: dict[str, list[dict[str, Any]]] | None = None) -> snapshots.SnapshotManifest:
    rows = tables() if rows is None else rows
    manifest = snapshot_factory.build(root, lambda: CLOCK, rows)
    return manifest


def build_with_revisions(root: Path, revisions: dict[str, str]) -> snapshots.SnapshotManifest:
    """Build the clean corpus with explicit ``source_revisions`` (freshness tests)."""
    b = snapshots.SnapshotBuilder(root, clock=lambda: CLOCK, code_version="test")
    for name, rows in tables().items():
        t = snapshot_factory.table(name, rows)
        if TABLES[name].partition_by:
            b.add_partitioned(name, t, TABLES[name].partition_by)
        else:
            b.add(name, t)
    cand = b.finish(status=DatasetStatus.LEGACY_UNVERIFIED, source_revisions=revisions)
    snapshots.promote(root, cand, ValidationReport(outcome=CheckOutcome.PASS))
    return cand.manifest


def rehash(root: Path, mutate: Callable[[dict[str, list[dict[str, Any]]]], None]) -> snapshots.SnapshotManifest:
    """Rebuild the corpus after ``mutate`` with every hash and the snapshot id recalculated."""
    rows = copy.deepcopy(tables())
    mutate(rows)
    return build(root, rows)


def rewrite_manifest(root: Path, edit: Callable[[dict[str, Any]], None], *, fix_id: bool = True) -> str:
    """Edit the current manifest JSON; optionally recompute its semantic id and repoint current."""
    pointer = json.loads((root / "current.json").read_text())
    path = root / pointer["manifest_path"]
    data = json.loads(path.read_text())
    edit(data)
    if fix_id:
        m = snapshots.SnapshotManifest.model_validate({**data, "snapshot_id": "sha256:" + "0" * 64})
        new_id = snapshots.semantic_snapshot_id(m)
        data["snapshot_id"] = new_id
        new_path = root / "snapshots" / f"{snapshots.snapshot_hex(new_id)}.json"
        new_path.write_text(json.dumps(data, indent=1))
        pointer.update(snapshot_id=new_id, manifest_path=f"snapshots/{snapshots.snapshot_hex(new_id)}.json")
        (root / "current.json").write_text(json.dumps(pointer, indent=1))
        return new_id
    path.chmod(0o644)
    path.write_text(json.dumps(data, indent=1))
    return str(data["snapshot_id"])


def fragment_path(root: Path, manifest: snapshots.SnapshotManifest, table: str, partition: str | None = None) -> Path:
    frag = next(f for f in manifest.tables[table].fragments if f.partition == partition)
    return root / "fragments" / frag.path


def tree_digest(*roots: Path) -> dict[str, bytes]:
    """Every file's bytes under ``roots`` (read-only assertions)."""
    out = {}
    for r in roots:
        for p in sorted(r.rglob("*")):
            if p.is_file():
                out[str(p)] = p.read_bytes()
    return out


# ---------------------------------------------------------------------------
# Source pages rendered in the AFL Tables layout (for the independent reader)
# ---------------------------------------------------------------------------

LABELS = {
    "kicks": "KI",
    "marks": "MK",
    "handballs": "HB",
    "disposals": "DI",
    "goals": "GL",
    "behinds": "BH",
    "hitouts": "HO",
    "tackles": "TK",
    "rebound_50s": "RB",
    "inside_50s": "IF",
    "clearances": "CL",
    "clangers": "CG",
    "frees_for": "FF",
    "frees_against": "FA",
    "brownlow_votes": "BR",
    "contested_possessions": "CP",
    "uncontested_possessions": "UP",
    "contested_marks": "CM",
    "marks_inside_50": "MI",
    "one_percenters": "1%",
    "bounces": "BO",
    "goal_assists": "GA",
    "time_on_ground_pct": "%P",
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _when(local_start: str) -> str:
    d, t = local_start.split()
    y, m, day = d.split("-")
    hh, mm = (int(x) for x in t.split(":"))
    ampm = "PM" if hh >= 12 else "AM"
    return f"Sat, {day}-{MONTHS[int(m) - 1]}-{y} {hh % 12 or 12}:{mm:02d} {ampm}"


def _cell(v: Any, blank_zero: bool = False) -> str:
    if v is None or (blank_zero and v == 0):
        return "<td>&nbsp;</td>"
    return f"<td align=center>{int(v) if float(v).is_integer() else v}</td>"


def player_url(player_id: str) -> str:
    slug = player_id.split(":", 1)[1]
    return f"https://afltables.com/afl/stats/players/{slug[0].upper()}/{slug}.html"


def render_match_page(
    rows: dict[str, list[dict[str, Any]]], match_id: str, *, stats: list[str] | None = None, blank_zeros: bool = False
) -> bytes:
    """AFL Tables layout. ``blank_zeros`` prints 0 as a blank cell, as the real site does."""
    stats = list(PLAYER_STAT_COLUMNS if stats is None else stats)
    m = next(x for x in rows["matches"] if x["match_id"] == match_id)
    names = {p["player_id"]: p["display_name"] for p in rows["players"]}
    head = [
        f'<table border=2><tr><td rowspan=3><a href="../2026/0.html">&larr;</a></td><td colspan=5><b>Round: </b>'
        f'{m["stage_label"]} <b>Venue: </b><a href="../../../venues/v.html">{m["venue_source_name"]}</a> '
        f"<b>Date: </b>{_when(m['local_start'])} <b>Attendance:</b> {m['attendance']}</td>"
        f'<td rowspan=3><a href="../2026/2.html">&rarr;</a></td></tr>'
    ]
    for side in ("home", "away"):
        q = "".join(
            f"<td align=center>{m[f'{side}_{x}_goals']}.{m[f'{side}_{x}_behinds']}."
            f"<b>{6 * m[f'{side}_{x}_goals'] + m[f'{side}_{x}_behinds']}</b></td>"
            for x in ("q1", "q2", "q3", "final")
        )
        head.append(
            f'<tr><td><a href="../../../teams/{m[f"{side}_club_id"]}_idx.html">{m[f"{side}_source_name"]}</a></td>{q}</tr>'
        )
    head.append("</table>")
    body = ["<html><body>", *head]
    for side in ("home", "away"):
        club = m[f"{side}_club_id"]
        players = sorted(
            (g for g in rows["player_games"] if g["match_id"] == match_id and g["club_id"] == club),
            key=lambda g: names[g["player_id"]],
        )
        hdr = "".join(f"<th>{LABELS[s]}</th>" for s in stats)
        body.append(
            f'<table class="sortable"><thead><tr><th colspan={len(stats) + 2}>{m[f"{side}_source_name"]} '
            f"Match Statistics [Season]</th></tr><tr><th>#</th><th>Player</th>{hdr}</tr></thead><tbody>"
        )
        for g in players:
            first, last = names[g["player_id"]].split(" ", 1)
            slug = g["player_id"].split(":", 1)[1]
            body.append(
                f'<tr><td align=center>{g["jersey_number"]}</td><td><a href="../../players/{slug[0].upper()}/'
                f'{slug}.html">{last}, {first}</a></td>' + "".join(_cell(g[s], blank_zeros and s != "time_on_ground_pct") for s in stats) + "</tr>"
            )
        body.append("</tbody><tfoot>")
        team_bh = m[f"{side}_final_behinds"]
        player_bh = sum(g["behinds"] or 0 for g in players)
        if "behinds" in stats and team_bh > player_bh:
            i = stats.index("behinds")
            body.append(
                f"<tr><td colspan={i + 2}>Rushed</td><td>{team_bh - player_bh}</td>"
                f"<td colspan={len(stats) - i - 1}>&nbsp;</td></tr>"
            )
        totals = []
        for s_ in stats:
            vals = [g[s_] for g in players if g[s_] is not None]
            total = sum(vals) + (team_bh - player_bh if s_ == "behinds" else 0) if vals else None
            totals.append(
                "<td>&nbsp;</td>" if s_ == "time_on_ground_pct" or total is None else f"<td><b>{int(total)}</td>"
            )
        body.append("<tr><td colspan=2><b>Totals</td>" + "".join(totals) + "</tr></tfoot></table>")
    body.append("</body></html>")
    return "\n".join(body).encode()


def render_player_page(rows: dict[str, list[dict[str, Any]]], player_id: str, *, blank_zeros: bool = False) -> bytes:
    """AFL Tables player page: one game table per club-season (``Club - YYYY``), career counter first."""
    names = {p["player_id"]: p["display_name"] for p in rows["players"]}
    clubs = {c["club_id"]: c["name"] for c in rows["clubs"]}
    matches = {m["match_id"]: m for m in rows["matches"]}
    games = sorted((g for g in rows["player_games"] if g["player_id"] == player_id),
                   key=lambda g: (g["season"], g["career_game_counter"]))  # fmt: skip
    out = [f"<html><body><h1>{names[player_id]}</h1>"]
    hdr = "".join(f"<th>{LABELS[s]}</th>" for s in PLAYER_STAT_COLUMNS)
    blocks: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for g in games:
        blocks.setdefault((g["season"], g["club_id"]), []).append(g)
    for (season, club), gs in blocks.items():
        out.append(f"<table class=sortable><thead><tr><th colspan=28>{clubs[club]} - {season}</th></tr>"
                   f"<tr><th>Gm</th><th>Opponent</th><th>Rd</th><th>R</th><th>#</th>{hdr}</tr></thead><tbody>")  # fmt: skip
        for g in gs:
            m = matches[g["match_id"]]
            rd = m["stage_label"] if m["stage_type"] == "regular" else FINAL_TOKENS.get(m["stage_label"], "?")
            cells = "".join(_cell(g[s], blank_zeros and s != "time_on_ground_pct") for s in PLAYER_STAT_COLUMNS)
            out.append(f"<tr><td align=center>{g['career_game_counter']}</td><td nowrap>{clubs[g['opponent_club_id']]}"
                       f"</td><td><a href=\"../../games/{season}/x.html\">{rd}</a></td><td>{g['result']}</td>"
                       f"<td>{g['jersey_number']}</td>{cells}</tr>")  # fmt: skip
        out.append("</tbody></table>")
    out.append("</body></html>")
    return "\n".join(out).encode()


FINAL_TOKENS = {"Qualifying Final": "QF", "Elimination Final": "EF", "Semi Final": "SF", "Preliminary Final": "PF",
                "Grand Final": "GF", "Wildcard Final": "WF"}  # fmt: skip


def render_season_page(
    rows: dict[str, list[dict[str, Any]]], season: int, extra: list[dict[str, Any]] | None = None
) -> bytes:
    out = [f"<html><body><h1>{season} Season Scores and Results</h1>"]
    ms = sorted(
        (m for m in rows["matches"] if m["season"] == season), key=lambda m: (m["stage_order"], m["match_date"])
    )
    stage = None
    for m in [*ms, *(extra or [])]:
        if m["stage_label"] != stage:
            stage = m["stage_label"]
            heading = f"Round {stage}" if str(stage).isdigit() else stage
            out.append(f"<table border=2><tr><td align=center width=60%><b>{heading}</td></tr></table>")
        rows_html = []
        for side in ("home", "away"):
            if m.get("home_score") is None:
                q, pts = "", ""
            else:
                q = " ".join(
                    f"{m[f'{side}_{x}_goals']}.{m[f'{side}_{x}_behinds']}" for x in ("q1", "q2", "q3", "final")
                )
                pts = str(m[f"{side}_score"])
            if side == "home":
                info = f'{_when(m["local_start"])} <b>Att: </b>{m["attendance"]:,} <b>Venue:</b> <a href="../venues/v.html">{m["venue_source_name"]}</a>'
            else:
                gid = m.get("game_id")
                info = f'[<a href="../stats/games/{season}/{gid}.html">Match stats</a>]' if gid else ""
            rows_html.append(
                f'<tr><td><a href="../teams/{m[f"{side}_club_id"]}_idx.html">{m[f"{side}_source_name"]}</a></td>'
                f"<td><tt>{q}</tt></td><td> {pts}</td><td>{info}</td></tr>"
            )
        out.append("<table border=1>" + "".join(rows_html) + "</table>")
    out.append("</body></html>")
    return "\n".join(out).encode()


def archive(root: Path, payload: bytes) -> str:
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    path = root / "raw" / "objects" / digest[:2] / digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return digest


GF_URL = "https://afltables.com/afl/stats/games/2026/000120260305.html"
#: the pinned season page was fetched the day before AS_OF (fresh by policy)
CHECKED = datetime(2026, 9, 27, 0, 0, tzinfo=UTC)


def with_sources(
    root: Path,
    mutate: Callable[[dict[str, list[dict[str, Any]]]], None] | None = None,
    *,
    page_rows: dict[str, list[dict[str, Any]]] | None = None,
    season_extra: list[dict[str, Any]] | None = None,
    checked_at: datetime | None = None,
    observations: list[dict[str, Any]] | None = None,
    schedule: dict[str, Any] | None = None,
    columns: list[str] | None = None,
    blank_zeros: bool = False,
    player_pages: tuple[str, ...] = (),
) -> dict[str, str]:
    """Clean corpus + archived season page and one source-fetched match page, all pinned.

    ``page_rows`` renders the pages from different rows than the snapshot holds (a source
    that disagrees); ``mutate`` changes the snapshot rows after the pages were rendered.
    """
    rows = copy.deepcopy(tables())
    for m in rows["matches"]:
        if m["season"] == 2026:
            m["game_id"] = "000120260305" if m["stage_label"] == "1" else "000120260312"
    src_rows = copy.deepcopy(page_rows or rows)
    for m in src_rows["matches"]:
        if m["season"] == 2026 and "game_id" not in m:
            m["game_id"] = "000120260305" if m["stage_label"] == "1" else "000120260312"
    match_page = render_match_page(src_rows, "m:2026:r01:alpha:beta:0", stats=columns, blank_zeros=blank_zeros)
    season_page = render_season_page(src_rows, 2026, season_extra)
    gf_sha = archive(root, match_page)
    season_sha = archive(root, season_page)
    pages = {pid: render_player_page(src_rows, pid, blank_zeros=blank_zeros) for pid in player_pages}
    page_sha = {pid: archive(root, page) for pid, page in pages.items()}
    for m in rows["matches"]:
        m.pop("game_id", None)
        if m["match_id"] == "m:2026:r01:alpha:beta:0":
            m.update(provenance="source_fetch", source_path=GF_URL, source_sha256=gf_sha, source_row=None)
    for g_ in rows["player_games"]:
        if g_["match_id"] == "m:2026:r01:alpha:beta:0":
            g_.update(provenance="source_fetch", source_path=GF_URL, source_sha256=gf_sha, link_method="source_url")
    for p in rows["players"]:
        p["source_urls"] = json.dumps([player_url(p["player_id"])])
    checked = checked_at or CHECKED
    rows["seasons"][1]["fixture_checked_at"] = checked
    rows["seasons"][1].update(schedule or {})
    rows["source_observations"] = (
        observations
        if observations is not None
        else [
            {
                "source_ref": "src:afltables:season",
                "adapter": "afltables.season_fixture",
                "adapter_version": "1",
                "url": SEASON_URL,
                "fetched_at": checked,
                "content_sha256": season_sha,
                "http_status": 200,
                "bytes": len(season_page),
                "source_mode": "live",
                "outcome": "PASS",
            },
            {
                "source_ref": "src:afltables:gf",
                "adapter": "afltables.match_detail",
                "adapter_version": "1",
                "url": GF_URL,
                "fetched_at": checked,
                "content_sha256": gf_sha,
                "http_status": 200,
                "bytes": len(match_page),
                "source_mode": "live",
                "outcome": "PASS",
            },
            *(
                {
                    "source_ref": f"src:afltables:player:{pid}",
                    "adapter": "afltables.player_page",
                    "adapter_version": "1",
                    "url": player_url(pid),
                    "fetched_at": checked,
                    "content_sha256": page_sha[pid],
                    "http_status": 200,
                    "bytes": len(pages[pid]),
                    "source_mode": "live",
                    "outcome": "PASS",
                }
                for pid in player_pages
            ),
        ]
    )
    if mutate is not None:
        mutate(rows)
    b = snapshots.SnapshotBuilder(root, clock=lambda: CLOCK, code_version="test")
    for name, rs in rows.items():
        t = snapshot_factory.table(name, rs)
        if TABLES[name].partition_by:
            b.add_partitioned(name, t, TABLES[name].partition_by)
        else:
            b.add(name, t)
    revisions = {"afltables:season:2026": season_sha, "afltables:game:000120260305": gf_sha}
    cand = b.finish(status=DatasetStatus.LEGACY_UNVERIFIED, source_revisions=revisions)
    snapshots.promote(root, cand, ValidationReport(outcome=CheckOutcome.PASS))
    return {"season": season_sha, "match": gf_sha, **{f"player:{p}": h for p, h in page_sha.items()}}


# ---------------------------------------------------------------------------
# Releases
# ---------------------------------------------------------------------------


def add_site(release_dir: Path) -> None:
    """A minimal sealed site whose data/<id>/ is the public tree (what the Astro build produces)."""
    import shutil

    from supercoach_via.publish.release import validate_release, write_seal

    site = release_dir / "site"
    site.mkdir()
    (site / "index.html").write_text("<!doctype html><title>demo</title>\n")
    shutil.copytree(release_dir / "public", site / "data" / release_dir.name)
    write_seal(release_dir, build_inputs={"command": "test"})
    report = validate_release(release_dir)
    assert report.ok, report.issues[:3]


def reseal(release_dir: Path, *, expect_valid: bool = True) -> None:
    """Recompute every structural hash after editing public/ (checksums, refs, embedded copy, seal, validation).

    ``expect_valid=False`` builds a release that validate_release itself refuses (its record is
    then FAIL); the checker must still report the defect by its own rules.
    """
    import shutil

    from supercoach_via.publish.release import validate_release, write_seal
    from supercoach_via.publish.web_data import canonical_json_bytes, sha256_bytes

    public = release_dir / "public"
    rel = json.loads((public / "release.json").read_text())
    for ref in rel["resources"].values():
        data = (public / ref["path"]).read_bytes()
        ref.update(sha256=sha256_bytes(data), bytes=len(data))
    (public / "release.json").write_bytes(canonical_json_bytes(rel))
    files = {}
    for p in sorted(public.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            files[p.relative_to(public).as_posix()] = {"sha256": sha256_bytes(data), "bytes": len(data)}
    (release_dir / "checksums.json").write_bytes(canonical_json_bytes({"release_id": release_dir.name, "files": files}))
    site_data = release_dir / "site" / "data" / release_dir.name
    if site_data.exists():
        shutil.rmtree(site_data)
        shutil.copytree(public, site_data)
        write_seal(release_dir, build_inputs={"command": "test"})
    report = validate_release(release_dir)
    assert report.ok == expect_valid, report.issues[:3]


def edit_json(path: Path, edit: Callable[[Any], None]) -> None:
    from supercoach_via.publish.web_data import canonical_json_bytes

    doc = json.loads(path.read_text())
    edit(doc)
    path.write_bytes(canonical_json_bytes(doc))
