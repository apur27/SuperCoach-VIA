"""Build ``SeasonInput`` objects from a ``recon_world.World`` for the comparison tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Any

from supercoach_via.reconciliation import schema as S
from supercoach_via.reconciliation import source as R
from supercoach_via.reconciliation.evidence import BrownlowFacts, VoteEra
from supercoach_via.reconciliation.local import LocalGame
from supercoach_via.reconciliation.rules import Rules
from supercoach_via.reconciliation.season import ProfileView, SeasonInput, season_rows_of
from tests.scvia.unit import recon_world as rw

NOTES = R.NotesFacts(availability={2010: S.STAT_FIELDS}, exceptions=(), problems=())
#: award facts as the captured Brownlow page states them: six votes per game from 1931 (12 in 1976-77), no medal 1942-45
BROWNLOW = BrownlowFacts(
    frozenset({1942, 1943, 1944, 1945}),
    (VoteEra(1924, 1930, 1), VoteEra(1931, None, 6)),
    (VoteEra(1976, 1977, 12),),
    frozenset(),
)
RULES = Rules(version=1, brownlow=BROWNLOW)


def full_cells(k: int = 1, **over: int | float | None) -> dict[str, int | float | None]:
    cells: dict[str, int | float | None] = {f: k + i % 3 for i, f in enumerate(S.COUNT_FIELDS)}
    cells["time_on_ground_pct"] = 80.0
    cells.update(over)
    return cells


def modern_world() -> rw.World:
    players = {
        "a": rw.P("Ann_Able", "Ann", "Able", "01-Jul-1990"),
        "b": rw.P("Bob_Baker", "Bob", "Baker", "02-Aug-1991"),
        "c": rw.P("Cy_Cole", "Cy", "Cole", "03-Sep-1992"),
    }

    def apps(k: int, final: bool = False) -> tuple[rw.A, ...]:
        br_a = None if final else 3
        br_c = None if final else 3
        return (
            rw.A("a", "Alpha", "1", full_cells(k, brownlow_votes=br_a)),
            rw.A("b", "Alpha", "2", full_cells(k + 1, tackles=0, brownlow_votes=None)),
            rw.A("c", "Beta", "3", full_cells(k + 2, brownlow_votes=br_c)),
        )

    ms = (
        rw.M("041520260305", 2026, "1", "Alpha", "Beta", date(2026, 3, 5), apps=apps(1)),
        rw.M("041520260312", 2026, "2", "Beta", "Alpha", date(2026, 3, 12), apps=apps(2)),
        rw.M("041520260926", 2026, "Grand Final", "Alpha", "Beta", date(2026, 9, 26), apps=apps(3, final=True)),
    )
    return rw.World(players, ms, (2026,))


def source_side(w: rw.World, season: int) -> tuple[dict[str, R.MatchFacts], dict[str, str], dict[str, ProfileView]]:
    matches: dict[str, R.MatchFacts] = {}
    shas: dict[str, str] = {}
    to_date = w.career_to_date()
    for m in w.matches:
        if m.year == season:
            career = {a.pid: to_date[(m.gid, a.pid)] for a in m.apps}
            matches[m.url] = R.read_match(rw.match_page(m, w.players, career=career), m.url)
            shas[m.url] = f"sha-{m.gid}"
    profiles: dict[str, ProfileView] = {}
    for pid, p in w.players.items():
        facts = R.read_profile(rw.profile_page(w, pid), p.url)
        games = [g for g in facts.games if g.season == season]
        if games:
            bad = {cs.club: frozenset(cs.bad_fields) for cs in facts.club_seasons if cs.season == season}
            profiles[p.url] = ProfileView(
                p.url, facts.h1, games, bad, f"sha-{pid}", season_rows=season_rows_of(facts, season)
            )
    return matches, shas, profiles


def local_rows(w: rw.World, season: int, layer: str = "snapshot") -> list[LocalGame]:
    rows: list[LocalGame] = []
    counters: dict[str, int] = {}
    for m in sorted(w.matches, key=lambda x: (x.when, x.gid)):
        for a in m.apps:
            counters[a.pid] = counters.get(a.pid, 0) + 1
            if m.year != season:
                continue
            opp = m.away if a.club == m.home else m.home
            own, other = (m.score("home"), m.score("away")) if a.club == m.home else (m.score("away"), m.score("home"))
            res = "W" if own > other else "L" if own < other else "D"
            # a correct local layer stores a zero where the source's team total proves the statistic was recorded
            team_has = {f: any(x.cells.get(f) is not None for x in m.apps if x.club == a.club) for f in S.STAT_FIELDS}
            if layer == "snapshot":
                vals = tuple(
                    a.cells.get(f)
                    if a.cells.get(f) is not None
                    else (0 if team_has[f] and f != "time_on_ground_pct" else None)
                    for f in S.STAT_FIELDS
                )
            else:  # the CSV keeps what the site printed: a zero is a blank
                vals = tuple(None if a.cells.get(f) in (None, 0) else a.cells.get(f) for f in S.STAT_FIELDS)
            raw = tuple("" if v is None else (str(int(v)) if float(v).is_integer() else str(v)) for v in vals)
            rows.append(
                LocalGame(
                    layer=layer,
                    player_key=a.pid if layer == "snapshot" else f"slug_{a.pid}",
                    season=season,
                    club=a.club,
                    opponent=opp,
                    stage=m.rd_token,
                    match_key=f"m:{m.gid}" if layer == "snapshot" else None,
                    result=res,
                    jersey=a.jersey,
                    counter=counters[a.pid],
                    counter_token=str(counters[a.pid]),
                    match_date=m.when.isoformat(),
                    cells=vals if layer == "snapshot" else tuple(None if v is None else int(v) for v in vals),
                    raw_cells=() if layer == "snapshot" else raw,
                    origin=f"{layer}:{a.pid}:{m.gid}",
                )
            )
    return rows


def season_input(
    w: rw.World, season: int = 2026, *, layer: str = "snapshot", rows: list[LocalGame] | None = None, **kw: Any
) -> SeasonInput:
    matches, shas, profiles = source_side(w, season)
    rows = local_rows(w, season, layer) if rows is None else rows
    inp = SeasonInput(
        layer=layer, season=season, matches=matches, match_sha=shas, absent_matches={}, profiles=profiles,
        notes=NOTES, rules=RULES, emit_source=True, captured=frozenset(p.url for p in w.players.values()),
    )  # fmt: skip
    url_of = {pid: p.url for pid, p in w.players.items()}
    mkey = {f"m:{m.gid}": m.url for m in w.matches}
    if layer == "snapshot":
        for r in rows:
            inp.local_by_pair.setdefault((url_of[r.player_key], mkey[r.match_key or ""]), []).append(r)
    else:
        slug_url = {f"slug_{pid}": u for pid, u in url_of.items()}
        for r in rows:
            inp.local_by_profile.setdefault(slug_url[r.player_key], []).append(r)
    inp.local_ids = {u: (pid if layer == "snapshot" else f"slug_{pid}") for pid, u in url_of.items()}
    inp.profile_status = dict.fromkeys(url_of.values(), "mapped")
    for k, v in kw.items():
        setattr(inp, k, v)
    return inp


def with_cell(rows: list[LocalGame], pid: str, gid: str, field: str, value: object) -> list[LocalGame]:
    idx = S.STAT_FIELDS.index(field)
    out = []
    for r in rows:
        if r.origin.endswith(f":{pid}:{gid}"):
            cells = list(r.cells)
            cells[idx] = value
            raw = list(r.raw_cells)
            if raw:
                raw[idx] = "" if value is None else str(value)
            r = replace(r, cells=tuple(cells), raw_cells=tuple(raw))
        out.append(r)
    return out
