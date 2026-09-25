"""Coverage-aware player statistics and leaders (PLAN 4.3/section 6; test A03).

Everything is batch SQL over the snapshot's ``player_games`` (one DuckDB pass per frame),
never a per-player loop. Aggregates follow ``domain.metrics``: blank = unknown, sums of no
observed games are null, means divide by observed games, and every figure carries its
``observed_games`` of ``career_games`` coverage.

Games metric: ``career_games = max(observed row count, max career_game_counter)``. Both
inputs are exposed (``row_games``, ``counter_max``) because the source counter can lead the
rows (missing drawn-final/finals rows) — the two are never forced to agree.
"""

from __future__ import annotations

# SQL is composed only from whitelisted identifiers (canonical stat/column names) and int-cast
# literals; every caller-supplied value is a bound parameter.
# ruff: noqa: S608
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from supercoach_via.domain.metrics import CoverageEras, sql_aggregate_columns
from supercoach_via.domain.schemas import PLAYER_STAT_COLUMNS
from supercoach_via.storage.queries import SnapshotQuery

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from supercoach_via.publish.view_models import HistoryTable, LeaderRow, SeasonLine, StatValue

METHOD_VERSION = "coverage_v1"
COVERAGE_NOTE = (
    "Totals sum observed games only (a blank cell is 'not recorded', never zero). Means divide "
    "by observed games (recorded-games denominator, pending-decision 3); coverage = observed "
    "games of career games, where career games = max(rows, source career counter)."
)


def _check_stat(stat: str) -> str:
    if stat not in PLAYER_STAT_COLUMNS:
        raise ValueError(f"unknown stat {stat!r}")
    return stat


def _has(q: SnapshotQuery, name: str) -> bool:
    return name in q.manifest.tables and (q.tables is None or name in q.tables)


# ---------------------------------------------------------------------------
# Whole-corpus bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerStatsBundle:
    """All-player frames for per-player JSON.

    * ``games``: player_id, row_games, counter_max, career_games, first_season, last_season
    * ``careers``: player_id, stat, total, observed_games, eligible_games, career_games,
      coverage, mean  (one row per player x stat)
    * ``seasons``: player_id, season, stat, total, observed_games, eligible_games, games,
      coverage, mean  (one row per player x season x stat; ``games`` = season rows)
    * ``season_clubs``: player_id, season, clubs (list of club_id in first-appearance order)
    """

    games: pd.DataFrame
    careers: pd.DataFrame
    seasons: pd.DataFrame
    season_clubs: pd.DataFrame
    stats: tuple[str, ...]


def _melt(wide: pd.DataFrame, keys: list[str], stats: Sequence[str], denom: str) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    frames = []
    for s in stats:
        f = wide[[*keys, denom]].copy()
        f["stat"] = s
        f["total"] = wide[f"{s}_total"].astype("float64")
        f["observed_games"] = wide[f"{s}_observed"].astype("int64")
        f["eligible_games"] = wide[f"{s}_eligible"].astype("int64")
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    obs = out["observed_games"].to_numpy()
    den = out[denom].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mean"] = np.where(obs > 0, out["total"].to_numpy() / np.maximum(obs, 1), np.nan)
        out["coverage"] = np.where(den > 0, np.minimum(1.0, obs / np.maximum(den, 1)), np.nan)
    return out.sort_values([*keys, "stat"], kind="stable").reset_index(drop=True)


def player_stats_bundle(
    q: SnapshotQuery, eras: CoverageEras, stats: Sequence[str] = PLAYER_STAT_COLUMNS
) -> PlayerStatsBundle:
    stats = tuple(_check_stat(s) for s in stats)
    cols = sql_aggregate_columns(stats, eras)
    games = q.df(
        """SELECT player_id, COUNT(*)::BIGINT AS row_games, MAX(career_game_counter) AS counter_max,
                  GREATEST(COUNT(*), COALESCE(MAX(career_game_counter), 0))::BIGINT AS career_games,
                  MIN(season) AS first_season, MAX(season) AS last_season
           FROM player_games GROUP BY player_id ORDER BY player_id"""
    )
    career_wide = q.df(f"SELECT player_id, {cols} FROM player_games GROUP BY player_id")
    career_wide = career_wide.merge(games[["player_id", "career_games"]], on="player_id")
    season_wide = q.df(
        f"SELECT player_id, season, COUNT(*)::BIGINT AS games, {cols} FROM player_games GROUP BY ALL"
    )
    clubs = q.df(
        """SELECT player_id, season, list(club_id ORDER BY first_seen, club_id) AS clubs
           FROM (SELECT player_id, season, club_id,
                        MIN(COALESCE(match_date, DATE '9999-12-31')) AS first_seen
                 FROM player_games GROUP BY ALL)
           GROUP BY ALL ORDER BY player_id, season"""
    )
    return PlayerStatsBundle(
        games=games,
        careers=_melt(career_wide, ["player_id"], stats, "career_games"),
        seasons=_melt(season_wide, ["player_id", "season"], stats, "games"),
        season_clubs=clubs,
        stats=stats,
    )


def _none(v: Any) -> float | None:
    import math

    if v is None:
        return None
    f = float(v)
    return None if math.isnan(f) else f


def _stat_values(frame: pd.DataFrame) -> list[StatValue]:
    from supercoach_via.publish.view_models import StatValue

    return [
        StatValue(
            stat=str(r["stat"]),
            total=_none(r["total"]),
            mean=_none(r["mean"]),
            observed_games=int(r["observed_games"]),
            eligible_games=int(r["eligible_games"]),
            coverage=_none(r["coverage"]),
        )
        for r in frame.to_dict("records")
    ]


def career_stat_values(bundle: PlayerStatsBundle, player_id: str) -> list[StatValue]:
    """Career ``StatValue`` list for one player (bundle order = stat name order)."""
    return _stat_values(bundle.careers[bundle.careers["player_id"] == player_id])


def season_lines(
    bundle: PlayerStatsBundle, player_id: str, games_resource: Callable[[str, int], str]
) -> list[SeasonLine]:
    """Per-season ``SeasonLine`` list; ``clubs`` are club_ids in first-appearance order."""
    from supercoach_via.publish.view_models import SeasonLine

    sub = bundle.seasons[bundle.seasons["player_id"] == player_id]
    clubs = bundle.season_clubs[bundle.season_clubs["player_id"] == player_id].set_index("season")["clubs"]
    out = []
    for season, frame in sub.groupby("season", sort=True):
        s = int(str(season))
        out.append(
            SeasonLine(
                season=s,
                clubs=[str(c) for c in clubs.get(s, [])],
                games=int(frame["games"].to_numpy()[0]),
                stats=_stat_values(frame),
                games_resource=games_resource(player_id, s),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Leaders
# ---------------------------------------------------------------------------


def _names_and_clubs(q: SnapshotQuery, ids: Sequence[str]) -> dict[str, tuple[str, list[str]]]:
    if not ids:
        return {}
    import pyarrow as pa

    q.con.register("_ids", pa.table({"player_id": list(ids)}))
    try:
        names: dict[str, str] = {}
        if _has(q, "players"):
            names = dict(q.rows("SELECT p.player_id, p.display_name FROM players p JOIN _ids USING (player_id)"))
        club_expr, club_join = "g.club_id", ""
        if _has(q, "clubs"):
            club_expr, club_join = "COALESCE(c.name, g.club_id)", "LEFT JOIN clubs c ON c.club_id = g.club_id"
        rows = q.rows(
            f"""SELECT g.player_id, {club_expr} AS club, MIN(COALESCE(g.match_date, DATE '9999-12-31')) AS f,
                       MIN(g.season) AS s
                FROM player_games g JOIN _ids USING (player_id) {club_join}
                GROUP BY ALL ORDER BY g.player_id, s, f, club"""
        )
    finally:
        q.con.unregister("_ids")
    clubs: dict[str, list[str]] = {}
    for pid, club, _f, _s in rows:
        clubs.setdefault(pid, []).append(club)
    return {pid: (names.get(pid, pid), clubs.get(pid, [])) for pid in ids}


def _competition_ranks(values: Sequence[float]) -> list[int]:
    ranks: list[int] = []
    for i, v in enumerate(values):
        ranks.append(ranks[-1] if i and values[i - 1] == v else i + 1)
    return ranks


def career_leaders(
    q: SnapshotQuery,
    stat: str,
    *,
    eras: CoverageEras,
    basis: Literal["total", "mean"] = "total",
    n: int = 100,
    min_observed_games: int = 0,
    seasons: tuple[int, int] | None = None,
) -> HistoryTable:
    """Career leaders on observed totals or observed-denominator means.

    ``min_observed_games`` is a disclosed qualification threshold (a choice, not a fact) and
    applies to the stat's observed games. ``seasons`` restricts rows to an inclusive range.
    """
    from supercoach_via.publish.view_models import HistoryRow, HistoryTable

    stat = _check_stat(stat)
    cols = sql_aggregate_columns([stat], eras)
    where = "" if seasons is None else f"WHERE season BETWEEN {int(seasons[0])} AND {int(seasons[1])}"
    order_col = f'"{stat}_total"' if basis == "total" else f'"{stat}_total" / "{stat}_observed"'
    rows = q.rows(
        f"""WITH a AS (
              SELECT player_id, {cols}, MIN(season) AS lo, MAX(season) AS hi FROM player_games {where}
              GROUP BY player_id),
            c AS (SELECT player_id, GREATEST(COUNT(*), COALESCE(MAX(career_game_counter), 0)) AS cg
                  FROM player_games {where} GROUP BY player_id)
            SELECT a.player_id, {order_col} AS value, "{stat}_observed", "{stat}_eligible", c.cg, lo, hi
            FROM a JOIN c USING (player_id)
            WHERE "{stat}_observed" > 0 AND "{stat}_observed" >= ?
            ORDER BY value DESC, a.player_id LIMIT ?""",
        [int(min_observed_games), int(n)],
    )
    meta = _names_and_clubs(q, [r[0] for r in rows])
    ranks = _competition_ranks([float(r[1]) for r in rows])
    out = [
        HistoryRow(
            rank=rank,
            player_id=pid,
            name=meta[pid][0],
            clubs=meta[pid][1],
            value=float(value),
            value_label=f"career {stat} ({basis})",
            observed_games=int(obs),
            eligible_games=int(elig),
            coverage=min(1.0, obs / cg) if cg else None,
            seasons=f"{lo}-{hi}",
        )
        for rank, (pid, value, obs, elig, cg, lo, hi) in zip(ranks, rows, strict=True)
    ]
    partial = sum(1 for r in out if r.coverage is not None and r.coverage < 0.9)
    return HistoryTable(
        category=f"career_{stat}_{basis}",
        title=f"Career {stat.replace('_', ' ')} ({basis})",
        scope="career",
        era="all" if seasons is None else f"{seasons[0]}-{seasons[1]}",
        method=(
            f"{basis} over observed games"
            + (f"; qualification: at least {min_observed_games} observed games" if min_observed_games else "")
        ),
        method_version=METHOD_VERSION,
        coverage_note=COVERAGE_NOTE,
        warning=(f"{partial} of {len(out)} rows rest on less than 90% of career games" if partial else None),
        rows=out,
    )


def games_leaders(q: SnapshotQuery, *, n: int = 100) -> HistoryTable:
    """Career games leaders: value = max(rows, counter); observed_games = observed rows."""
    from supercoach_via.publish.view_models import HistoryRow, HistoryTable

    rows = q.rows(
        """SELECT player_id, GREATEST(COUNT(*), COALESCE(MAX(career_game_counter), 0)) AS cg,
                  COUNT(*) AS rows_, MAX(career_game_counter) AS ctr, MIN(season), MAX(season)
           FROM player_games GROUP BY player_id ORDER BY cg DESC, player_id LIMIT ?""",
        [int(n)],
    )
    meta = _names_and_clubs(q, [r[0] for r in rows])
    ranks = _competition_ranks([float(r[1]) for r in rows])
    out = [
        HistoryRow(
            rank=rank,
            player_id=pid,
            name=meta[pid][0],
            clubs=meta[pid][1],
            value=float(cg),
            value_label="career games",
            observed_games=int(nrows),
            eligible_games=None,
            coverage=min(1.0, nrows / cg) if cg else None,
            seasons=f"{lo}-{hi}",
        )
        for rank, (pid, cg, nrows, _ctr, lo, hi) in zip(ranks, rows, strict=True)
    ]
    diff = sum(1 for r in out if r.observed_games is not None and r.observed_games != r.value)
    return HistoryTable(
        category="career_games",
        title="Career games",
        scope="career",
        era="all",
        method=(
            "career games = max(observed stat rows, source career game counter); observed_games "
            "shows the row count so counter-versus-row differences stay visible"
        ),
        method_version=METHOD_VERSION,
        coverage_note=COVERAGE_NOTE,
        warning=(f"{diff} of {len(out)} players have counter and row totals that differ" if diff else None),
        rows=out,
    )


def single_season_leaders(
    q: SnapshotQuery, stat: str, *, eras: CoverageEras, n: int = 100
) -> HistoryTable:
    """Best single-season observed totals (one row per player-season)."""
    from supercoach_via.publish.view_models import HistoryRow, HistoryTable

    stat = _check_stat(stat)
    cols = sql_aggregate_columns([stat], eras)
    rows = q.rows(
        f"""SELECT player_id, season, "{stat}_total", "{stat}_observed", "{stat}_eligible", games FROM (
              SELECT player_id, season, COUNT(*) AS games, {cols} FROM player_games GROUP BY ALL)
            WHERE "{stat}_observed" > 0
            ORDER BY "{stat}_total" DESC, player_id, season LIMIT ?""",
        [int(n)],
    )
    meta = _names_and_clubs(q, list(dict.fromkeys(r[0] for r in rows)))
    ranks = _competition_ranks([float(r[2]) for r in rows])
    out = [
        HistoryRow(
            rank=rank,
            player_id=pid,
            name=meta[pid][0],
            clubs=meta[pid][1],
            value=float(total),
            value_label=f"season {stat}",
            observed_games=int(obs),
            eligible_games=int(elig),
            coverage=min(1.0, obs / games) if games else None,
            seasons=str(season),
        )
        for rank, (pid, season, total, obs, elig, games) in zip(ranks, rows, strict=True)
    ]
    return HistoryTable(
        category=f"season_{stat}_total",
        title=f"Single-season {stat.replace('_', ' ')}",
        scope="single_season",
        era="all",
        method="season total over observed games",
        method_version=METHOD_VERSION,
        coverage_note=COVERAGE_NOTE,
        warning=None,
        rows=out,
    )


def season_leaders(
    q: SnapshotQuery, stat: str, season: int, *, n: int = 10, club_id: str | None = None
) -> list[LeaderRow]:
    """Season-total leaders (optionally for one club), observed games disclosed."""
    from supercoach_via.publish.view_models import LeaderRow

    stat = _check_stat(stat)
    params: list[object] = [int(season)]
    club_filter = ""
    if club_id is not None:
        club_filter = "AND club_id = ?"
        params.append(club_id)
    params.append(int(n))
    rows = q.rows(
        f"""SELECT player_id, SUM("{stat}") AS v, COUNT("{stat}") AS obs FROM player_games
            WHERE season = ? {club_filter} GROUP BY player_id HAVING COUNT("{stat}") > 0
            ORDER BY v DESC, player_id LIMIT ?""",
        params,
    )
    meta = _names_and_clubs(q, [r[0] for r in rows])
    return [
        LeaderRow(player_id=pid, name=meta[pid][0], stat=stat, value=float(v), observed_games=int(obs))
        for pid, v, obs in rows
    ]
