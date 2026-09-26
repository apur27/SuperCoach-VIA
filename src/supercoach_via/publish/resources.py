"""Public resources derived directly from canonical tables (matches, box scores, game logs).

Statistics are copied as observed: NULL stays ``None`` (not recorded), zero stays zero.
Ordering is chronological by local date/start then stable ``match_id`` — never by
source round number.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import Any

from supercoach_via.domain.schemas import PLAYER_STAT_COLUMNS
from supercoach_via.publish.view_models import (
    BoxScoreColumns,
    MatchDetail,
    MatchIndex,
    MatchSummary,
    PlayerGameColumns,
    PlayerIndex,
    PlayerIndexEntry,
    PlayerSeasonGames,
    QuarterScore,
    Source,
    TeamScore,
)
from supercoach_via.storage.queries import SnapshotQuery

_ORDER = "match_date NULLS LAST, local_start NULLS LAST, stage_order, match_id"


def public_key(identifier: str) -> str:
    """Encode an ID for use in a resource path (':' -> '__')."""
    return identifier.replace(":", "__")


def match_key(match_id: str) -> str:
    return public_key(match_id)


def _club_names(q: SnapshotQuery) -> dict[str, str]:
    return {cid: name for cid, name in q.rows("SELECT club_id, name FROM clubs")}


def _summary(r: dict[str, Any], names: dict[str, str]) -> MatchSummary:
    def team(side: str) -> TeamScore:
        cid = r[f"{side}_club_id"]
        return TeamScore(
            club_id=cid,
            name=names.get(cid, r[f"{side}_source_name"]),
            goals=r[f"{side}_final_goals"],
            behinds=r[f"{side}_final_behinds"],
            score=r[f"{side}_score"],
        )

    home, away = team("home"), team("away")
    winner = None
    if r["status"] == "complete" and home.score is not None and away.score is not None and home.score != away.score:
        winner = home.club_id if home.score > away.score else away.club_id
    md = r["match_date"]
    return MatchSummary(
        match_id=r["match_id"],
        season=r["season"],
        stage_id=r["stage_id"],
        stage_label=r["stage_label"],
        stage_type=r["stage_type"],
        round_number=r["round_number"],
        stage_order=r["stage_order"],
        replay_occurrence=r["replay_occurrence"],
        local_start=r["local_start"],
        match_date=md if isinstance(md, date) else None,
        date_precision=r["date_precision"],
        status=r["status"],
        venue=r.get("venue_source_name"),
        home=home,
        away=away,
        winner_club_id=winner,
    )


def _records(q: SnapshotQuery, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    cur = q.con.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def match_rows(q: SnapshotQuery, season: int) -> list[dict[str, Any]]:
    return _records(q, f"SELECT * FROM matches WHERE season = ? ORDER BY {_ORDER}", [season])  # noqa: S608 - interpolates a module constant only; values are bound parameters


def match_index(q: SnapshotQuery, season: int) -> MatchIndex:
    names = _club_names(q)
    return MatchIndex(season=season, matches=[_summary(r, names) for r in match_rows(q, season)])


def seasons(q: SnapshotQuery) -> list[int]:
    return [int(s) for (s,) in q.rows("SELECT DISTINCT season FROM matches ORDER BY season")]


def compact_stats(rows: list[dict[str, Any]]) -> tuple[list[str], list[list[float | None]]]:
    """Columns observed at least once (canonical order) and positional value arrays."""
    raw = [[r.get(c) for c in PLAYER_STAT_COLUMNS] for r in rows]
    keep = [i for i in range(len(PLAYER_STAT_COLUMNS)) if any(row[i] is not None for row in raw)]
    values = [[None if (x := row[i]) is None else float(x) for i in keep] for row in raw]
    return [PLAYER_STAT_COLUMNS[i] for i in keep], values


def match_details(q: SnapshotQuery, season: int) -> Iterator[MatchDetail]:
    names = _club_names(q)
    games: dict[str, list[dict[str, Any]]] = {}
    for g in _records(
        q,
        "SELECT g.*, p.display_name FROM player_games g JOIN players p USING (player_id) "
        "WHERE g.season = ? ORDER BY g.match_id, g.club_id, p.display_name, g.player_id",
        [season],
    ):
        games.setdefault(g["match_id"], []).append(g)
    for r in match_rows(q, season):
        rows = games.get(r["match_id"], [])
        cols, values = compact_stats(rows)

        def box(
            club: str, rows: list[dict[str, Any]] = rows, values: list[list[float | None]] = values
        ) -> BoxScoreColumns:
            side = [(g, v) for g, v in zip(rows, values, strict=True) if g["club_id"] == club]
            return BoxScoreColumns(player_id=[g["player_id"] for g, _ in side],
                                   name=[g["display_name"] for g, _ in side], stats=[v for _, v in side])  # fmt: skip

        yield MatchDetail(
            summary=_summary(r, names),
            quarters=[
                QuarterScore(
                    quarter=qn,
                    home_goals=r[f"home_{qn}_goals"],
                    home_behinds=r[f"home_{qn}_behinds"],
                    away_goals=r[f"away_{qn}_goals"],
                    away_behinds=r[f"away_{qn}_behinds"],
                )
                for qn in ("q1", "q2", "q3", "final")
            ],
            attendance=r["attendance"],
            home_players=box(r["home_club_id"]),
            away_players=box(r["away_club_id"]),
            stat_columns=cols,
            sources=[Source(label="Legacy match/player CSV import", url=None, note=r.get("source_path"))],
        )


def player_season_games(q: SnapshotQuery, season: int) -> Iterator[PlayerSeasonGames]:
    names = _club_names(q)
    rows = _records(
        q,
        "SELECT g.*, m.local_start FROM player_games g LEFT JOIN matches m USING (match_id) "
        "WHERE g.season = ? ORDER BY g.player_id, g.match_date NULLS LAST, m.local_start NULLS LAST, g.match_id",
        [season],
    )
    by_player: dict[str, list[dict[str, Any]]] = {}
    for g in rows:
        by_player.setdefault(g["player_id"], []).append(g)
    for player_id, prows in by_player.items():
        cols, values = compact_stats(prows)
        opps = [g["opponent_club_id"] for g in prows]
        games = PlayerGameColumns(
            match_id=[g["match_id"] for g in prows],
            match_date=[g["match_date"] if isinstance(g["match_date"], date) else None for g in prows],
            date_quality=[g["date_quality"] for g in prows],
            stage_label=[g["stage_label"] for g in prows],
            club_id=[g["club_id"] for g in prows],
            opponent_club_id=opps,
            opponent_name=[names.get(o) if o else None for o in opps],
            result=[g["result"] for g in prows],
            career_game_counter=[g["career_game_counter"] for g in prows],
            stats=values,
        )
        yield PlayerSeasonGames(player_id=player_id, season=season, stat_columns=cols, games=games)


def normalise_search(text: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(stripped.lower().split())


def player_index(q: SnapshotQuery) -> PlayerIndex:
    latest = q.scalar("SELECT max(season) FROM player_games")
    rows = q.rows(
        """
        SELECT p.player_id, p.display_name,
               list(DISTINCT c.name ORDER BY c.name) FILTER (WHERE c.name IS NOT NULL) AS clubs,
               min(g.season), max(g.season), count(g.match_id)
        FROM players p
        LEFT JOIN player_games g USING (player_id)
        LEFT JOIN clubs c ON c.club_id = g.club_id
        WHERE p.identity_status = 'canonical'
        GROUP BY p.player_id, p.display_name
        ORDER BY p.display_name, p.player_id
        """
    )
    players = [
        PlayerIndexEntry(
            id=pid,
            key=public_key(pid),
            name=name,
            clubs=list(clubs or []),
            first_season=first,
            last_season=last,
            games=int(n),
            active=last is not None and last == latest,
            search=normalise_search(f"{name} {' '.join(clubs or [])}"),
        )
        for pid, name, clubs, first, last, n in rows
    ]
    return PlayerIndex(count=len(players), players=players)
