"""What team, history, list, summary and download resources must say, recomputed independently.

Every function here reads the VERIFIED snapshot tables the checker registered in DuckDB (plus,
for rankings, the pinned ``ranking_legacy_v1.toml`` method file) and returns the documents the
publication contract requires, keyed by public path. None of it calls the release builder or
the analytics that produce those resources: ladders, form, team-game means, leader tables,
the legacy_v1 ranking, list rows and summaries are re-derived from their definitions.

Expected documents are JSON values. A few marker objects say how a node is compared
(``public_compare.compare_value``):

* ``{"$skip": reason}``: presentation text that is not compared (reported as unaudited);
* ``{"$contains": [...]}``: prose must contain every phrase (numeric claims inside text);
* ``{"$one_of": [...]}``: any alternative matches;
* ``{"$instant": "..."}``: the same UTC instant, in any offset.

Floats compare with a relative tolerance of 1e-9 (summation order); everything else exactly.
"""

from __future__ import annotations

# SQL is composed only from canonical stat names and int-cast literals.
# ruff: noqa: S608
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import duckdb

    from supercoach_via.analytics.rankings import RankingConfig

#: publication contract (release builder ``TEAM_STATS`` etc.): which stats a page carries
TEAM_STATS = (
    "disposals",
    "kicks",
    "handballs",
    "marks",
    "goals",
    "tackles",
    "clearances",
    "inside_50s",
    "contested_possessions",
)
TEAM_LEADER_STATS = ("disposals", "goals", "tackles")
CAREER_STATS = ("disposals", "goals", "kicks", "handballs", "marks", "tackles", "brownlow_votes")
SEASON_STATS = ("disposals", "goals", "tackles")
LEADER_STATS = ("disposals", "goals", "tackles", "marks")
PLAYER_CSV_STATS = ("disposals", "kicks", "handballs", "marks", "goals", "tackles")
FORM_WINDOW = 5
FIVE_YEARS = 5
HISTORY_N = 100
LEADER_N = 10
STALE_AFTER = timedelta(days=8)
#: the legacy era table (``analytics.eras``); the last era extends to the latest season
ERA_BOUNDS = (
    ("pre-1965", 1897, 1964),
    ("1965-1990", 1965, 1990),
    ("1991-2010", 1991, 2010),
    ("2011-present", 2011, 2026),
)
HEURISTIC_LABEL = "Finals race (heuristic)"
UNAUDITED_LABEL = "presentation label"


def skip(reason: str = UNAUDITED_LABEL) -> dict[str, str]:
    return {"$skip": reason}


def contains(*phrases: str) -> dict[str, list[str]]:
    return {"$contains": list(phrases)}


def iso(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat().replace("+00:00", "Z")
    return v.isoformat() if isinstance(v, date) else v


def _rows(con: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    cur = con.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def competition_ranks(values: Sequence[Any]) -> list[int]:
    out: list[int] = []
    for i, v in enumerate(values):
        out.append(out[-1] if i and values[i - 1] == v else i + 1)
    return out


def chrono(m: Mapping[str, Any]) -> tuple[Any, ...]:
    """Match chronology: date, local start (nulls last), stage order, replay, id."""
    d, ls = m["match_date"], m["local_start"]
    return (d is None, d or date.min, ls is None, ls or "", m["stage_order"], m["replay_occurrence"], m["match_id"])


def summary(m: Mapping[str, Any], names: Mapping[str, str]) -> dict[str, Any]:
    from supercoach_via.integrity.public_compare import expected_summary

    return expected_summary(dict(m), dict(names))


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------


def ladder(matches: Iterable[Mapping[str, Any]], names: Mapping[str, str]) -> list[dict[str, Any]]:
    """Regular-season ladder: 4 points a win, 2 a draw; completed matches with both scores only."""
    regular = [m for m in matches if m["stage_type"] == "regular"]
    clubs = sorted({m["home_club_id"] for m in regular} | {m["away_club_id"] for m in regular})
    t = {c: {"p": 0, "w": 0, "l": 0, "d": 0, "pf": 0, "pa": 0} for c in clubs}
    for m in regular:
        hs, as_ = m["home_score"], m["away_score"]
        if m["status"] != "complete" or hs is None or as_ is None:
            continue
        for club, pf, pa in ((m["home_club_id"], hs, as_), (m["away_club_id"], as_, hs)):
            r = t[club]
            r["p"] += 1
            r["pf"] += int(pf)
            r["pa"] += int(pa)
            r["w" if pf > pa else "l" if pf < pa else "d"] += 1

    def pts(c: str) -> int:
        return 4 * t[c]["w"] + 2 * t[c]["d"]

    def pct(c: str) -> float | None:
        return None if t[c]["pa"] == 0 else 100.0 * t[c]["pf"] / t[c]["pa"]

    def order(c: str) -> tuple[Any, ...]:
        p = pct(c)
        sortable = (math.inf if t[c]["pf"] > 0 else 0.0) if p is None else p
        return (-pts(c), -sortable, -t[c]["pf"], c)

    return [
        {
            "position": i,
            "club_id": c,
            "name": names.get(c, c),
            "played": t[c]["p"],
            "won": t[c]["w"],
            "lost": t[c]["l"],
            "drawn": t[c]["d"],
            "points_for": t[c]["pf"],
            "points_against": t[c]["pa"],
            "percentage": pct(c),
            "premiership_points": pts(c),
        }
        for i, c in enumerate(sorted(clubs, key=order), start=1)
    ]


def team_index(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    rows = _rows(
        con,
        """WITH s AS (SELECT home_club_id AS club_id, season FROM matches
                      UNION SELECT away_club_id, season FROM matches)
           SELECT c.club_id, c.name, c.lineage_id, c.first_season, c.last_season, c.active,
                  list(DISTINCT s.season ORDER BY s.season) AS seasons
           FROM clubs c JOIN s USING (club_id) GROUP BY ALL ORDER BY c.name, c.club_id""",
    )
    return {
        "teams": [
            {
                "club_id": r["club_id"],
                "name": r["name"],
                "lineage_id": r["lineage_id"],
                "first_season": r["first_season"] if r["first_season"] is not None else min(r["seasons"]),
                "last_season": r["last_season"] if r["last_season"] is not None else max(r["seasons"]),
                "active": bool(r["active"]),
                "seasons": [int(s) for s in r["seasons"]],
            }
            for r in rows
        ]
    }


def team_season_docs(
    con: duckdb.DuckDBPyConnection,
    season: int,
    names: Mapping[str, str],
    ladders: Mapping[int, list[dict[str, Any]]],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    """Every ``teams/<club>/<season>.json`` of one season.

    ``matches``: every match of seasons ``season - 5 .. season`` (form reaches back across
    season boundaries); ``ladders``: this season's and the five previous seasons' ladders.
    """
    from supercoach_via.analytics.teams import LADDER_METHOD, PATHWAY_METHOD

    this = sorted((m for m in matches if m["season"] == season), key=chrono)
    clubs = sorted({m["home_club_id"] for m in this} | {m["away_club_id"] for m in this})
    lad = ladders.get(season, [])
    by_club = {r["club_id"]: r for r in lad}
    done = sorted(
        (
            m
            for m in matches
            if m["status"] == "complete" and m["home_score"] is not None and m["away_score"] is not None
        ),
        key=chrono,
    )
    stats = _team_stats(con, season, clubs)
    leaders = _team_leaders(con, season)
    docs: dict[str, Any] = {}
    for club in clubs:
        form = []
        for m in [m for m in done if club in (m["home_club_id"], m["away_club_id"])][-FORM_WINDOW:]:
            home = m["home_club_id"] == club
            own, opp = (m["home_score"], m["away_score"]) if home else (m["away_score"], m["home_score"])
            opp_id = m["away_club_id"] if home else m["home_club_id"]
            margin = int(own) - int(opp)
            form.append(
                {
                    "match_id": m["match_id"],
                    "match_date": iso(m["match_date"]),
                    "opponent": names.get(opp_id, opp_id),
                    "result": "W" if margin > 0 else "L" if margin < 0 else "D",
                    "margin": margin,
                }
            )
        row = by_club.get(club)
        heuristics = []
        if row is not None:
            heuristics.append(
                {
                    "label": HEURISTIC_LABEL,
                    "text": {
                        "$one_of": [
                            contains(f"{row['name']} are {row['position']} on {row['premiership_points']} points"),
                            contains(f"{row['name']}: finals system for {season} is unknown"),
                        ]
                    },
                    "method": PATHWAY_METHOD,
                }
            )
        docs[f"teams/{club}/{season}.json"] = {
            "club": {"club_id": club, "name": names.get(club, club)},
            "season": season,
            "ladder": lad,
            "ladder_note": LADDER_METHOD,
            "position": row["position"] if row else None,
            "form_window": FORM_WINDOW,
            "form": form,
            "fixtures": [summary(m, names) for m in this if club in (m["home_club_id"], m["away_club_id"])],
            "team_stats": stats.get(club, [_stat_value(s, [], 0) for s in TEAM_STATS]),
            "leaders": leaders.get(club, []),
            "five_year": [
                r for y in range(season - FIVE_YEARS, season) for r in ladders.get(y, []) if r["club_id"] == club
            ],
            "heuristics": heuristics,
        }
    return docs


def _stat_value(stat: str, values: list[float], team_games: int) -> dict[str, Any]:
    total = sum(values) if values else None
    n = len(values)
    return {
        "stat": stat,
        "total": total,
        "mean": None if total is None else total / n,
        "observed_games": n,
        "eligible_games": team_games,
        "coverage": None if team_games <= 0 else min(1.0, n / team_games),
    }


def _team_stats(con: duckdb.DuckDBPyConnection, season: int, clubs: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Per team-game sums (null when no row observed the stat), then observed-denominator means."""
    sums = ", ".join(f'CASE WHEN count("{s}") > 0 THEN sum("{s}") END AS "{s}"' for s in TEAM_STATS)
    games: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in _rows(con, f"SELECT club_id, match_id, {sums} FROM player_games WHERE season = ? GROUP BY ALL", [season]):
        games[r["club_id"]].append(r)
    out = {}
    for club in clubs:
        g = games.get(club, [])
        out[club] = [_stat_value(s, [float(x[s]) for x in g if x[s] is not None], len(g)) for s in TEAM_STATS]
    return out


def _team_leaders(con: duckdb.DuckDBPyConnection, season: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stat in TEAM_LEADER_STATS:
        best: dict[str, tuple[Any, ...]] = {}
        for club, pid, name, v, obs in con.execute(
            f"""SELECT g.club_id, g.player_id, coalesce(p.display_name, g.player_id), sum("{stat}"), count("{stat}")
                FROM player_games g LEFT JOIN players p USING (player_id)
                WHERE g.season = ? GROUP BY ALL HAVING count("{stat}") > 0""",
            [season],
        ).fetchall():
            cur = best.get(club)
            if cur is None or (-v, pid) < (-cur[2], cur[0]):
                best[club] = (pid, name, v, obs)
        for club, (pid, name, v, obs) in best.items():
            out[club].append(
                {"player_id": pid, "name": name, "stat": stat, "value": float(v), "observed_games": int(obs)}
            )
    return out


# ---------------------------------------------------------------------------
# History: career / games / single-season leaders and eras
# ---------------------------------------------------------------------------


def eras(con: duckdb.DuckDBPyConnection) -> list[tuple[str, int, int]]:
    lo, hi = con.execute("SELECT min(season), max(season) FROM player_games").fetchone() or (None, None)
    if lo is None:
        return []
    bounds = [*ERA_BOUNDS[:-1], (ERA_BOUNDS[-1][0], ERA_BOUNDS[-1][1], max(ERA_BOUNDS[-1][2], int(hi)))]
    return [(n, s, e) for n, s, e in bounds if s <= hi and e >= lo]


def career_clubs(con: duckdb.DuckDBPyConnection, ids: Iterable[str]) -> dict[str, list[str]]:
    """Club names a player appeared for, by first season, then first date, then name."""
    ids = sorted(set(ids))
    if not ids:
        return {}
    rows = con.execute(
        """SELECT g.player_id, coalesce(c.name, g.club_id) AS club, min(g.season) AS s,
                  min(coalesce(g.match_date, DATE '9999-12-31')) AS f
           FROM player_games g LEFT JOIN clubs c USING (club_id) WHERE list_contains(?, g.player_id)
           GROUP BY ALL ORDER BY g.player_id, s, f, club""",
        [ids],
    ).fetchall()
    out: dict[str, list[str]] = defaultdict(list)
    for pid, club, _s, _f in rows:
        out[pid].append(club)
    return out


def _display(con: duckdb.DuckDBPyConnection, ids: Iterable[str]) -> dict[str, str]:
    ids = sorted(set(ids))
    rows = con.execute(
        "SELECT player_id, display_name FROM players WHERE list_contains(?, player_id)", [ids]
    ).fetchall()
    return {p: n for p, n in rows}


def _table(
    category: str,
    scope: str,
    era: str,
    method: Any,
    warning: Any,
    rows: list[dict[str, Any]],
    method_version: Any = None,
) -> dict[str, Any]:
    return {
        "category": category,
        "title": skip(),
        "scope": scope,
        "era": era,
        "method": method,
        "method_version": method_version if method_version is not None else skip(),
        "coverage_note": skip(),
        "warning": warning,
        "rows": rows,
    }


def history_docs(
    con: duckdb.DuckDBPyConnection, recorded_from: Mapping[str, int], ranking: dict[str, Any]
) -> dict[str, Any]:
    """Every ``history/**`` resource, including the ``legacy_v1`` ranking tables."""
    docs: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []
    era_list = eras(con)

    def publish(category: str, scope: str, tables: list[tuple[str, dict[str, Any]]]) -> None:
        res = {}
        for era, doc in tables:
            path = f"history/{category}/{era}.json"
            docs[path] = doc
            res[era] = path
        entries.append({"category": category, "title": skip(), "scope": scope, "eras": list(res), "resources": res})

    for stat in CAREER_STATS:
        tables = [("all", _career(con, stat, recorded_from, None))]
        for name, s, e in era_list:
            t = _career(con, stat, recorded_from, (s, e))
            if t["rows"]:
                tables.append((name, t))
        publish(f"career_{stat}_total", "career", tables)
    publish("career_games", "career", [("all", _games(con))])
    for stat in SEASON_STATS:
        publish(f"season_{stat}_total", "single_season", [("all", _single_season(con, stat, recorded_from))])
    publish("all_time_top_100", "ranking", [("all", ranking["all_time_table"])])
    if ranking["yearly_tables"]:
        publish("yearly_top_100", "ranking", sorted(ranking["yearly_tables"].items()))
    summary_rows = []
    for name, s, e in era_list:
        players, games, matches = con.execute(
            "SELECT count(DISTINCT player_id), count(*), count(DISTINCT match_id) FROM player_games "
            "WHERE season BETWEEN ? AND ?",
            [s, e],
        ).fetchone() or (0, 0, 0)
        summary_rows.append(
            {
                "era": name,
                "first_season": s,
                "last_season": e,
                "players": int(players),
                "player_games": int(games),
                "matches": int(matches),
            }
        )
    docs["history/index.json"] = {"tables": entries, "era_summary": summary_rows}
    return docs


def _history_rows(con: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    clubs = career_clubs(con, [r["player_id"] for r in rows])
    names = _display(con, [r["player_id"] for r in rows])
    ranks = competition_ranks([float(r["value"]) for r in rows])
    return [
        {
            "rank": rank,
            "player_id": r["player_id"],
            "name": names.get(r["player_id"], r["player_id"]),
            "clubs": clubs.get(r["player_id"], []),
            "value": float(r["value"]),
            "value_label": label,
            "observed_games": r["observed_games"],
            "eligible_games": r["eligible_games"],
            "coverage": r["coverage"],
            "seasons": r["seasons"],
        }
        for rank, r in zip(ranks, rows, strict=True)
    ]


def _eligible_sql(stat: str, recorded_from: Mapping[str, int]) -> str:
    start = recorded_from.get(stat)
    return "count(*)" if start is None else f"count(*) FILTER (WHERE season >= {int(start)})"


def _career(
    con: duckdb.DuckDBPyConnection, stat: str, recorded_from: Mapping[str, int], seasons: tuple[int, int] | None
) -> dict[str, Any]:
    where = "" if seasons is None else f"WHERE season BETWEEN {int(seasons[0])} AND {int(seasons[1])}"
    # season-level award values (pre-1984 Brownlow votes are printed only per season): added to the career total
    awards = stat == "brownlow_votes" and _registered(con, "player_season_awards")
    if awards:
        rng = "" if seasons is None else f"AND season BETWEEN {int(seasons[0])} AND {int(seasons[1])}"
        award_sql = (
            f"(SELECT player_id, sum(value) AS aw FROM player_season_awards WHERE award = '{stat}' {rng} GROUP BY 1)"
        )
    else:
        award_sql = "(SELECT NULL::VARCHAR AS player_id, 0 AS aw WHERE false)"
    raw = _rows(
        con,
        f"""WITH g AS (
                SELECT player_id, sum("{stat}") AS pg, count("{stat}") AS o, {_eligible_sql(stat, recorded_from)} AS e,
                       greatest(count(*), coalesce(max(career_game_counter), 0)) AS cg,
                       min(season) AS lo, max(season) AS hi
                FROM player_games {where} GROUP BY player_id)
            SELECT g.player_id, coalesce(g.pg, 0) + coalesce(w.aw, 0) AS value, o, e, cg, lo, hi
            FROM g LEFT JOIN {award_sql} w USING (player_id)
            WHERE o > 0 OR coalesce(w.aw, 0) > 0
            ORDER BY value DESC, g.player_id LIMIT {HISTORY_N}""",
    )
    rows = [
        {
            "player_id": r["player_id"],
            "value": r["value"],
            "observed_games": int(r["o"]),
            "eligible_games": int(r["e"]),
            "coverage": min(1.0, r["o"] / r["cg"]) if r["cg"] else None,
            "seasons": f"{r['lo']}-{r['hi']}",
        }
        for r in raw
    ]
    out = _history_rows(con, rows, f"career {stat} (total)")
    partial = sum(1 for r in out if r["coverage"] is not None and r["coverage"] < 0.9)
    return _table(
        f"career_{stat}_total",
        "career",
        "all" if seasons is None else f"{seasons[0]}-{seasons[1]}",
        "total over observed games"
        + ("; plus season-level award votes the source prints only per season (pre-1984)" if awards else ""),
        f"{partial} of {len(out)} rows rest on less than 90% of career games" if partial else None,
        out,
    )


def _games(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    raw = _rows(
        con,
        f"""SELECT player_id, greatest(count(*), coalesce(max(career_game_counter), 0)) AS value,
                   count(*) AS n, min(season) AS lo, max(season) AS hi
            FROM player_games GROUP BY player_id ORDER BY value DESC, player_id LIMIT {HISTORY_N}""",
    )
    rows = [
        {
            "player_id": r["player_id"],
            "value": r["value"],
            "observed_games": int(r["n"]),
            "eligible_games": None,
            "coverage": min(1.0, r["n"] / r["value"]) if r["value"] else None,
            "seasons": f"{r['lo']}-{r['hi']}",
        }
        for r in raw
    ]
    out = _history_rows(con, rows, "career games")
    diff = sum(1 for r in out if r["observed_games"] != r["value"])
    return _table(
        "career_games",
        "career",
        "all",
        skip("method description"),
        f"{diff} of {len(out)} players have counter and row totals that differ" if diff else None,
        out,
    )


def _single_season(con: duckdb.DuckDBPyConnection, stat: str, recorded_from: Mapping[str, int]) -> dict[str, Any]:
    raw = _rows(
        con,
        f"""SELECT player_id, season, sum("{stat}") AS value, count("{stat}") AS o,
                   {_eligible_sql(stat, recorded_from)} AS e, count(*) AS games
            FROM player_games GROUP BY player_id, season HAVING count("{stat}") > 0
            ORDER BY value DESC, player_id, season LIMIT {HISTORY_N}""",
    )
    rows = [
        {
            "player_id": r["player_id"],
            "value": r["value"],
            "observed_games": int(r["o"]),
            "eligible_games": int(r["e"]),
            "coverage": min(1.0, r["o"] / r["games"]) if r["games"] else None,
            "seasons": str(r["season"]),
        }
        for r in raw
    ]
    return _table(
        f"season_{stat}_total",
        "single_season",
        "all",
        "season total over observed games",
        None,
        _history_rows(con, rows, f"season {stat}"),
    )


# ---------------------------------------------------------------------------
# legacy_v1 rankings (independent implementation of the documented formula)
# ---------------------------------------------------------------------------


def ranking_config(config_dir: Path | None, *, captured_bytes: bytes | None = None) -> RankingConfig:
    """The pinned method file (loaded, hashed; the formula below is the checker's own)."""
    from supercoach_via.analytics.rankings import RankingConfig
    from supercoach_via.settings import default_config_dir

    return RankingConfig.load(
        (config_dir or default_config_dir()) / "ranking_legacy_v1.toml", captured_bytes=captured_bytes
    )


def _era_of(cfg: RankingConfig, season: int) -> Any:
    return next((e for e in cfg.eras if e.start <= season <= e.end), None)


def rank_season(
    season: int, rows: Sequence[tuple[str, str, int, int, Mapping[str, float]]], cfg: RankingConfig
) -> list[dict[str, Any]]:
    """Yearly list: rows are (player_id, player_key, season, games, zero-filled totals)."""
    era = _era_of(cfg, season)
    if era is None:
        return []
    weights, completeness = dict(cfg.weights), dict(cfg.era_completeness)
    shrink = math.sqrt(completeness.get(era.name, cfg.missing_era_shrinkage))
    cohort = []
    for pid, key, _s, games, totals in rows:
        parts = [totals[s] * weights.get(s, 0.0) for s in era.stats if s in totals]
        raw = sum(parts)
        if not parts or raw <= 0:
            continue
        over = sum(max(0.0, c - cfg.single_stat_cap * raw) for c in parts)
        cohort.append((pid, key, games, max(0, int(raw - over))))
    if not cohort:
        return []
    scores = [c[3] for c in cohort]
    n = len(scores)
    below: dict[int, int] = {}
    equal: dict[int, int] = defaultdict(int)
    for v in scores:
        equal[v] += 1
    running = 0
    for v in sorted(equal):
        below[v] = running
        running += equal[v]
    mean = sum(scores) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in scores) / (n - cfg.z_ddof)) if n - cfg.z_ddof > 0 else 0.0
    ordered = sorted(cohort, key=lambda c: (-c[3], -c[2], c[0]))[: cfg.yearly_top_n]
    out = []
    for rank, (pid, key, games, score) in enumerate(ordered, start=1):
        z = 0.0 if sd == 0 or not math.isfinite(sd) else max(-cfg.z_cap, min(cfg.z_cap, (score - mean) / sd))
        pct = (below[score] + (equal[score] + 1) / 2) / n * 100
        out.append(
            {
                "rank": rank,
                "player_id": pid,
                "player_key": key,
                "score": score,
                "games": games,
                "percentile_rank": pct,
                "z_adj": z * shrink,
            }
        )
    return out


def rankings(con: duckdb.DuckDBPyConnection, cfg: RankingConfig, snapshot_id: str) -> dict[str, Any]:
    stats = sorted({s for e in cfg.eras for s in e.stats})
    sums = ", ".join(f'sum(coalesce("{s}", 0))::DOUBLE AS "{s}"' for s in stats)
    by_season: dict[int, list[tuple[str, str, int, int, dict[str, float]]]] = defaultdict(list)
    keys: dict[str, str] = {}
    for r in con.execute(
        f"""SELECT g.player_id, coalesce(p.legacy_slug, g.player_id), g.season, count(*), {sums}
            FROM player_games g LEFT JOIN players p USING (player_id) GROUP BY ALL"""
    ).fetchall():
        pid, key, season = r[0], r[1], int(r[2])
        by_season[season].append((pid, key, season, int(r[3]), dict(zip(stats, r[4:], strict=True))))
        keys[pid] = key
    career = {
        p: max(int(n), int(c or 0))
        for p, n, c in con.execute(
            "SELECT player_id, count(*), max(career_game_counter) FROM player_games GROUP BY player_id"
        ).fetchall()
    }
    seasons = sorted(by_season)
    complete = season_complete(con, seasons)
    yearly = {s: rank_season(s, by_season[s], cfg) for s in seasons}
    yearly = {s: v for s, v in yearly.items() if v}
    provisional = [s for s in yearly if not complete[s]]
    completeness = dict(cfg.era_completeness)
    recent = set(cfg.recent_years)
    active = {e["player_id"] for s, es in yearly.items() if s in recent for e in es}
    adj: dict[str, list[float]] = defaultdict(list)
    for s in sorted(yearly):
        era = _era_of(cfg, s)
        ec = completeness.get(era.name if era else "unknown", completeness["unknown"])
        for e in yearly[s]:
            rank_score = ((101 - e["rank"]) / 100.0) ** cfg.rank_gamma
            z_signal = max(0.0, min(1.0, (e["z_adj"] + cfg.z_cap) / (2.0 * cfg.z_cap)))
            adj[e["player_id"]].append(((1.0 - cfg.z_blend) * rank_score + cfg.z_blend * z_signal) * ec)
    scored = []
    for pid, values in adj.items():
        cg = career.get(pid, 0)
        if cg < cfg.min_career_games:
            continue
        best = sorted(values, reverse=True)[: cfg.top_n_seasons]
        mean_adj = sum(best) / len(best)
        score = mean_adj * (1.0 + cfg.career_bonus_factor * min(len(values) / cfg.career_bonus_seasons_cap, 1.0))
        if pid in active:
            score *= cfg.active_player_discount
        scored.append((pid, score, mean_adj, cg))
    top = sorted(scored, key=lambda x: (-x[1], -x[2], -x[3], x[0]))[: cfg.all_time_top_n]
    method = contains(f"{cfg.formula_version} ", f"config sha256 {cfg.config_hash()[:12]}", f"snapshot {snapshot_id}")
    ids = [p for p, *_ in top] + [e["player_id"] for es in yearly.values() for e in es]
    names = _display(con, ids)
    clubs, spans = _ranking_meta(con, ids)
    all_rows = [
        {
            "rank": i,
            "player_id": pid,
            "name": names.get(pid, pid),
            "clubs": clubs.get(pid, []),
            "value": score,
            "value_label": "all-time score",
            "observed_games": cg,
            "eligible_games": None,
            "coverage": None,
            "seasons": spans.get(pid),
        }
        for i, (pid, score, _m, cg) in enumerate(top, start=1)
    ]
    warning = (
        "Includes provisional (in-progress) seasons: " + ", ".join(str(s) for s in provisional) if provisional else None
    )
    yearly_tables = {}
    for s, es in yearly.items():
        rows = [
            {
                "rank": e["rank"],
                "player_id": e["player_id"],
                "name": names.get(e["player_id"], e["player_id"]),
                "clubs": clubs.get(e["player_id"], []),
                "value": float(e["score"]),
                "value_label": "season score",
                "observed_games": e["games"],
                "eligible_games": None,
                "coverage": None,
                "seasons": str(s),
            }
            for e in es
        ]
        yearly_tables[str(s)] = _table(
            f"yearly_top_100_{s}",
            "ranking",
            str(s),
            method,
            f"Provisional: season {s} is in progress; the yearly top 100 is published at season end."
            if s in provisional
            else None,
            rows,
            cfg.formula_version,
        )
    return {
        "all_time": [(pid, keys[pid], score) for pid, score, _m, _c in top],
        "yearly": yearly,
        "provisional": provisional,
        "all_time_table": _table("all_time_top_100", "ranking", "all", method, warning, all_rows, cfg.formula_version),
        "yearly_tables": yearly_tables,
    }


def _ranking_meta(con: duckdb.DuckDBPyConnection, ids: Iterable[str]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Ranking rows: club names by first season, then name; the season span."""
    ids = sorted(set(ids))
    clubs: dict[str, list[str]] = defaultdict(list)
    lo_hi: dict[str, tuple[int, int]] = {}
    for pid, club, lo, hi in con.execute(
        """SELECT g.player_id, coalesce(c.name, g.club_id) AS club, min(g.season) AS lo, max(g.season)
           FROM player_games g LEFT JOIN clubs c USING (club_id) WHERE list_contains(?, g.player_id)
           GROUP BY ALL ORDER BY g.player_id, lo, club""",
        [ids],
    ).fetchall():
        clubs[pid].append(club)
        a, b = lo_hi.get(pid, (lo, hi))
        lo_hi[pid] = (min(a, lo), max(b, hi))
    return clubs, {p: f"{a}-{b}" for p, (a, b) in lo_hi.items()}


def season_complete(con: duckdb.DuckDBPyConnection, seasons: Sequence[int]) -> dict[int, bool]:
    """Declared completion; else a decided Grand Final; else a later season exists."""
    declared: dict[int, bool] = {}
    if _registered(con, "seasons"):
        declared = {
            int(s): bool(c)
            for s, c in con.execute(
                "SELECT season, schedule_complete FROM seasons WHERE schedule_complete IS NOT NULL"
            ).fetchall()
        }
    gf = {
        int(s)
        for (s,) in con.execute(
            """SELECT DISTINCT season FROM matches WHERE status = 'complete'
             AND (lower(stage_label) IN ('gf', 'grand final') OR lower(stage_id) = 'gf')
             AND home_score IS NOT NULL AND away_score IS NOT NULL AND home_score <> away_score"""
        ).fetchall()
    }
    latest = max(seasons) if seasons else None
    return {s: declared.get(s, s in gf or (latest is not None and s < latest)) for s in seasons}


def _registered(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return bool(
        con.execute("SELECT count(*) FROM duckdb_views() WHERE view_name = ? AND NOT internal", [name]).fetchone()[0]  # type: ignore[index]
    ) or bool(con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = ?", [name]).fetchone()[0])  # type: ignore[index]


# ---------------------------------------------------------------------------
# Lists (pinned list imports: drafts, contracts, schools)
# ---------------------------------------------------------------------------


def lists_expected(con: duckdb.DuckDBPyConnection, tables: set[str]) -> dict[str, Any]:
    names = dict(con.execute("SELECT club_id, name FROM clubs").fetchall()) if "clubs" in tables else {}
    parts = [
        f"SELECT {c} AS s FROM {t} WHERE {c} IS NOT NULL"
        for t, c in (
            ("draft_events", "season"),
            ("contract_observations", "contract_end"),
            ("school_observations", "draft_year"),
        )
        if t in tables
    ]
    seasons = (
        [int(s) for (s,) in con.execute(f"SELECT DISTINCT s FROM ({' UNION '.join(parts)}) ORDER BY s").fetchall()]
        if parts
        else []
    )
    seasons = [s for s in seasons if 1000 <= s <= 9999]
    docs: dict[str, Any] = {}
    for s in seasons:
        drafts = (
            []
            if "draft_events" not in tables
            else [
                {
                    "season": r["season"],
                    "event_type": r["event_type"],
                    "draft_round": r["draft_round"],
                    "pick": r["pick"],
                    "club": names.get(r["club_id"], r["club_source_name"]) if r["club_id"] else r["club_source_name"],
                    "player_name": r["player_name"],
                    "player_id": r["player_id"],
                    "recruited_from": r["recruited_from"],
                    "grade": r["grade"],
                }
                for r in _rows(
                    con,
                    """SELECT * FROM draft_events WHERE season = ?
                                   ORDER BY season, event_type, draft_round NULLS LAST, pick NULLS LAST,
                                            draft_event_id""",
                    [s],
                )
            ]
        )
        contracts = (
            []
            if "contract_observations" not in tables
            else [
                {
                    "player_name": r["player_name"],
                    "player_id": r["player_id"],
                    "club": names.get(r["club_id"], r["club_id"]) if r["club_id"] else None,
                    "contract_end": r["contract_end"],
                    "fa_category": r["fa_category"],
                    "observed_at": iso(r["observed_at"]),
                    "source_type": r["source_type"],
                    "notes": r["notes"],
                }
                for r in _rows(
                    con,
                    """SELECT * FROM contract_observations WHERE contract_end = ?
                                   ORDER BY contract_end NULLS LAST, club_id NULLS LAST, player_name,
                                            observation_id""",
                    [s],
                )
            ]
        )
        schools = (
            []
            if "school_observations" not in tables
            else [
                {
                    "draft_year": r["draft_year"],
                    "pick": r["pick"],
                    "player_name": r["player_name"],
                    "player_id": r["player_id"],
                    "school": r["school"],
                    "school_type": r["school_type"],
                    "confidence": r["confidence"],
                }
                for r in _rows(
                    con,
                    """SELECT * FROM school_observations WHERE draft_year = ?
                                   ORDER BY draft_year NULLS LAST, pick NULLS LAST, observation_id""",
                    [s],
                )
            ]
        )
        families: dict[str, int] = {}
        if "draft_events" in tables:
            families = {
                str(f): int(n)
                for f, n in con.execute(
                    "SELECT source_family, count(*) FROM draft_events WHERE season = ? GROUP BY 1", [s]
                ).fetchall()
            }
        modes: dict[str, int] = defaultdict(int)
        for c in contracts:
            modes[c["source_type"]] += 1
        unresolved = sum(1 for rows in (drafts, contracts, schools) for r in rows if r["player_id"] is None)
        phrases = [
            f"Drafts: {len(drafts)} events",
            f"schools: {len(schools)} observations",
            f"{unresolved} names are not resolved",
        ]
        phrases += [f"{n} {m}" for m, n in sorted(modes.items())] or [
            f"No contract observations with a {s} contract end"
        ]
        phrases += [f"{f}: {n}" for f, n in sorted(families.items())]
        docs[f"lists/{s}.json"] = {
            "season": s,
            "drafts": drafts,
            "contracts": contracts,
            "schools": schools,
            "source_note": contains(*phrases),
            "sources": [
                {"label": f"Draft source: {f}", "url": None, "note": f"{n} rows"} for f, n in sorted(families.items())
            ]
            + [
                {"label": f"Contract observations ({m})", "url": None, "note": skip("disclosure text")}
                for m in sorted(modes)
            ],
        }
    docs["lists/index.json"] = {"seasons": seasons, "resources": {str(s): f"lists/{s}.json" for s in seasons}}
    return docs


# ---------------------------------------------------------------------------
# Season leaders, overview and quality
# ---------------------------------------------------------------------------


def season_leaders(con: duckdb.DuckDBPyConnection, stat: str, season: int, n: int = LEADER_N) -> list[dict[str, Any]]:
    return [
        {"player_id": pid, "name": name, "stat": stat, "value": float(v), "observed_games": int(obs)}
        for pid, name, v, obs in con.execute(
            f"""SELECT g.player_id, coalesce(any_value(p.display_name), g.player_id), sum("{stat}") AS v,
                       count("{stat}") FROM player_games g LEFT JOIN players p USING (player_id)
                WHERE g.season = ? GROUP BY g.player_id HAVING count("{stat}") > 0
                ORDER BY v DESC, g.player_id LIMIT {int(n)}""",
            [season],
        ).fetchall()
    ]


def stage_text(label: str, stage_type: str) -> str:
    return f"Round {label}" if stage_type == "regular" and label.isdigit() else label


def _latest_complete(matches: list[dict[str, Any]]) -> dict[str, Any] | None:
    done = [m for m in matches if m["status"] == "complete"]
    return max(done, key=_release_chrono) if done else None


def _release_chrono(m: Mapping[str, Any]) -> tuple[Any, ...]:
    """The release's season-view order (missing dates first, as the overview contract states)."""
    return (
        m["match_date"] or date.min,
        m["local_start"] or "",
        m["stage_order"],
        m["replay_occurrence"],
        m["match_id"],
    )


def coverage_through(matches: list[dict[str, Any]], season: int) -> str | None:
    last = _latest_complete([m for m in matches if m["season"] == season])
    return None if last is None else f"{season} {stage_text(last['stage_label'], last['stage_type'])}"


def overview_expected(
    con: duckdb.DuckDBPyConnection,
    *,
    release: Mapping[str, Any],
    season_matches: list[dict[str, Any]],
    names: Mapping[str, str],
    dataset_status: str,
    has_seasons: bool,
    forecast: Mapping[str, Any],
    current_set: Mapping[str, Any] | None,
    articles: list[Any],
) -> dict[str, Any]:
    """``overview.json`` from the facts and the release's own verified forecast/article resources."""
    season, demo = int(release["season"]), bool(release["demo"])
    generated = datetime.fromisoformat(str(release["generated_at"]).replace("Z", "+00:00"))
    checked = season_active = None
    if has_seasons:
        checked = con.execute("SELECT max(fixture_checked_at) FROM seasons").fetchone()[0]  # type: ignore[index]
        sc = con.execute("SELECT schedule_complete FROM seasons WHERE season = ?", [season]).fetchone()
        season_active = None if sc is None or sc[0] is None else not bool(sc[0])
    if isinstance(checked, datetime) and checked.tzinfo is None:
        from datetime import UTC

        checked = checked.replace(tzinfo=UTC)
    stale = False
    reason: str | None = "source check time is not recorded in this snapshot"
    if isinstance(checked, datetime):
        age = generated - checked
        stale = age > STALE_AFTER
        reason = f"sources last checked {age.days} days before this build" if stale else None
    ordered = sorted(season_matches, key=_release_chrono)
    last = _latest_complete(ordered)
    warnings = []
    if demo:
        warnings.append("DEMO release: synthetic data for testing the product; not AFL statistics.")
    if dataset_status == "legacy_unverified" and not demo:
        warnings.append("Imported from legacy CSV files; not yet independently re-verified against the sources.")
    if forecast["status"] != "available":
        warnings.append(f"No current forecast: {forecast['status']} ({forecast['reason']}).")
    rows = list((current_set or {}).get("rows") or [])
    top = sorted(rows, key=lambda r: (-r["predicted_disposals"], r["prediction_id"]))[:5]
    form = [
        {
            "player_id": pid,
            "name": name,
            "stat": "disposals (mean of last 3 games)",
            "value": float(v),
            "observed_games": int(n),
        }
        for pid, name, v, n in con.execute(
            """WITH g AS (SELECT player_id, disposals, row_number() OVER (PARTITION BY player_id
                            ORDER BY match_date DESC NULLS LAST, match_id DESC) AS k
                          FROM player_games WHERE season = ?)
               SELECT g.player_id, p.display_name, avg(disposals) AS v, count(disposals) FROM g
               JOIN players p USING (player_id) WHERE k <= 3 GROUP BY ALL HAVING count(disposals) = 3
               ORDER BY v DESC, g.player_id LIMIT 5""",
            [season],
        ).fetchall()
    ]
    return {
        "release_id": release["release_id"],
        "snapshot_id": release["snapshot_id"],
        "season": season,
        "demo": demo,
        "freshness": {
            # the instant, in whatever offset the build host wrote it (O55-F2: host-zone rendering)
            "source_checked_at": {"$instant": iso(checked)} if isinstance(checked, datetime) else None,
            "latest_completed_match_at": None,
            "latest_completed_match_date": iso(last["match_date"]) if last else None,
            "coverage_through": coverage_through(ordered, season),
            "generated_at": release["generated_at"],
            "published_at": None,
            "validation_state": skip("the snapshot pointer at build time is not an audit input"),
            "dataset_status": "demo" if demo else dataset_status,
            "season_active": season_active,
            "stale": stale,
            "stale_reason": reason,
        },
        "next_fixture_status": forecast["status"],
        "next_fixture_reason": forecast["reason"],
        "upcoming": [summary(m, names) for m in ordered if m["status"] == "scheduled"][:9],
        "prediction_highlights": top,
        "form_highlights": form,
        "recent_results": [summary(m, names) for m in reversed(ordered) if m["status"] == "complete"][:6],
        "latest_articles": articles[:2],
        "leaders": [r for stat in LEADER_STATS for r in season_leaders(con, stat, season)],
        "model_status": {
            "forecast_status": forecast["status"],
            "reason": forecast["reason"],
            "model": (current_set or {}).get("model") if current_set else None,
        },
        "warnings": warnings,
    }


def quality_expected(
    con: duckdb.DuckDBPyConnection,
    *,
    dataset_status: str,
    demo: bool,
    table_counts: Mapping[str, int],
    has_issues: bool,
    snapshot_id: str,
    forecast: Mapping[str, Any],
) -> dict[str, Any]:
    status = "demo" if demo else dataset_status
    issues = []
    if has_issues:
        for rule, sev, st, n in con.execute(
            "SELECT rule_id, severity, status, count(*) FROM quality_issues GROUP BY ALL ORDER BY 1, 2, 3"
        ).fetchall():
            issues.append(
                {
                    "rule_id": rule,
                    "severity": sev,
                    "count": int(n),
                    "description": f"{n} row(s) flagged by rule {rule} (status: {st}).",
                }
            )
    limitations = [
        "Missing statistics are shown as not recorded, never as zero; aggregates disclose observed games.",
        f"Forecast status for this release: {forecast['status']}"
        + (f" ({forecast['reason']})." if forecast["reason"] else "."),
    ]
    if demo:
        limitations.insert(0, "DEMO: this release is synthetic and exists only to test the product.")
    if status == "legacy_unverified":
        limitations.append("Dataset imported from legacy CSV files; dates without fixture evidence are inferred.")
    return {
        "dataset_status": status,
        "table_counts": dict(sorted(table_counts.items())),
        "quarantined_rows": int(table_counts.get("quarantine", 0)),
        "issues": issues,
        "limitations": limitations,
        "sources": [{"label": "Canonical snapshot", "url": None, "note": snapshot_id}],
    }


# ---------------------------------------------------------------------------
# Download tables (CSV rows as lists; the comparator parses the published CSV)
# ---------------------------------------------------------------------------


def players_csv(con: duckdb.DuckDBPyConnection, index_players: list[dict[str, Any]]) -> dict[str, Any]:
    """Rows in player-index order; totals and observed games over each player's career rows."""
    header = ["player_id", "name", "clubs", "first_season", "last_season", "career_games", "active"]
    for s in PLAYER_CSV_STATS:
        header += [f"{s}_total", f"{s}_observed_games"]
    cols = ", ".join(f'sum("{s}") AS "{s}__t", count("{s}") AS "{s}__o"' for s in PLAYER_CSV_STATS)
    agg = {
        r["player_id"]: r
        for r in _rows(
            con,
            f"""SELECT player_id, greatest(count(*), coalesce(max(career_game_counter), 0)) AS cg, {cols}
                 FROM player_games GROUP BY player_id""",
        )
    }
    rows = []
    for e in index_players:
        a = agg.get(e["id"])
        line: list[Any] = [
            e["id"],
            e["name"],
            "; ".join(e["clubs"]),
            e["first_season"],
            e["last_season"],
            int(a["cg"]) if a else 0,
            "true" if e["active"] else "false",
        ]
        for s in PLAYER_CSV_STATS:
            line += [a[f"{s}__t"], int(a[f"{s}__o"])] if a else [None, 0]
        rows.append(line)
    return {"header": header, "rows": rows, "key": 0}


def era_summary_csv(con: duckdb.DuckDBPyConnection, recorded_from: Mapping[str, int]) -> dict[str, Any]:
    from supercoach_via.domain.schemas import LEGACY_PLAYER_COLUMN_MAP, PLAYER_STAT_COLUMNS

    legacy = {v: k for k, v in LEGACY_PLAYER_COLUMN_MAP.items()}
    bounds = eras(con)
    case = " ".join(f"WHEN season BETWEEN {lo} AND {hi} THEN '{n}'" for n, lo, hi in bounds)
    header = [
        "era",
        "metric",
        "legacy_metric",
        "n_player_games",
        "n_with_metric",
        "mean_per_game",
        "std_per_game",
        "median_per_game",
        "mean_per_100pct_played",
        "recorded_from",
        "recording_status",
    ]
    rows = []
    for name, lo, hi in bounds:
        for m in PLAYER_STAT_COLUMNS:
            per100 = (
                f'avg("{m}")'
                if m == "time_on_ground_pct"
                else f'avg("{m}" * (100.0 / time_on_ground_pct)) FILTER (WHERE time_on_ground_pct >= 25.0)'
            )
            n, k, mean, sd, med, p100 = con.execute(
                f"""SELECT count(*), count("{m}"), avg("{m}"), stddev_samp("{m}"), median("{m}"), {per100}
                    FROM (SELECT *, CASE {case} END AS era FROM player_games) WHERE era = ?""",
                [name],
            ).fetchone()  # type: ignore[misc]
            start = recorded_from.get(m)
            status = "recorded" if start is None or start <= lo else "not_recorded" if start > hi else "partial_era"
            rows.append(
                [name, m, legacy.get(m, m), n, k, mean, sd if k else None, med if k else None, p100, start, status]
            )
    return {"header": header, "rows": rows, "key": (0, 1)}


def brownlow_proxy_csv(con: duckdb.DuckDBPyConnection, season: int) -> dict[str, Any] | None:
    """The labelled stat-profile index (not votes): recomputed from its published method."""
    from supercoach_via.analytics import awards

    rows = _rows(
        con,
        """WITH g AS (SELECT * FROM player_games WHERE season = ?),
           a AS (SELECT player_id, count(*) AS games,
                        avg(coalesce(disposals, 0)) AS disposals_pg, avg(coalesce(clearances, 0)) AS clearances_pg,
                        avg(coalesce(contested_possessions, 0)) AS contested_possessions_pg,
                        avg(coalesce(goals, 0)) AS goals_pg,
                        avg(greatest(coalesce(disposals, 0) - coalesce(clangers, 0), 0)) AS effective_disposals_pg,
                        avg(coalesce(tackles, 0)) AS tackles_pg,
                        sum(brownlow_votes) AS votes, count(brownlow_votes) AS votes_n FROM g GROUP BY player_id),
           c AS (SELECT g.player_id, coalesce(cl.name, g.club_id) AS club, count(*) AS n FROM g
                 LEFT JOIN clubs cl USING (club_id) GROUP BY ALL),
           top AS (SELECT player_id, arg_min(club, (-n, club)) AS club FROM c GROUP BY player_id)
           SELECT a.*, coalesce(p.display_name, a.player_id) AS name, top.club FROM a JOIN top USING (player_id)
           LEFT JOIN players p USING (player_id) WHERE games >= ? ORDER BY player_id""",
        [season, awards.MIN_GAMES],
    )
    if not rows:
        return None
    n = con.execute(
        """SELECT max(k) FROM (SELECT club, count(*) AS k FROM (
               SELECT home_club_id AS club FROM matches WHERE season = ? AND stage_type = 'regular'
                      AND status <> 'cancelled'
               UNION ALL SELECT away_club_id FROM matches WHERE season = ? AND stage_type = 'regular'
                      AND status <> 'cancelled') GROUP BY club)""",
        [season, season],
    ).fetchone()[0]  # type: ignore[index]
    scale = int(n) if n else awards.LEGACY_SEASON_GAMES

    def z(col: str) -> list[float]:
        xs = [float(r[col]) for r in rows]
        mu = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / len(xs))
        return [0.0] * len(xs) if sd == 0 or not math.isfinite(sd) else [(x - mu) / sd for x in xs]

    cols = {
        "disposals": "disposals_pg",
        "clearances": "clearances_pg",
        "contested_possessions": "contested_possessions_pg",
        "effective_disposals": "effective_disposals_pg",
        "goals": "goals_pg",
    }
    zs = {k: z(c) for k, c in cols.items()}
    for i, r in enumerate(rows):
        r["proxy"] = sum(w * zs[k][i] for k, w in awards.BROWNLOW_WEIGHTS)
    rows.sort(key=lambda r: (-r["proxy"], r["player_id"]))
    registry = awards.default_ineligibility(season)
    header = [
        "rank",
        "player_id",
        "name",
        "club",
        "games",
        "disposals_pg",
        "clearances_pg",
        "contested_possessions_pg",
        "goals_pg",
        "effective_disposals_pg",
        "tackles_pg",
        "proxy_per_game",
        "season_proxy_scaled",
        "observed_votes",
        "votes_observed_games",
        "ineligible",
        "ineligible_reason",
        "ineligible_source",
        "label",
        "version",
        "scale",
    ]
    out = []
    for i, r in enumerate(rows, start=1):
        inel = registry.get(str(r["player_id"]))
        out.append(
            [
                i,
                r["player_id"],
                r["name"],
                r["club"],
                int(r["games"]),
                r["disposals_pg"],
                r["clearances_pg"],
                r["contested_possessions_pg"],
                r["goals_pg"],
                r["effective_disposals_pg"],
                r["tackles_pg"],
                r["proxy"],
                r["proxy"] * scale,
                None if r["votes_n"] == 0 else int(r["votes"]),
                int(r["votes_n"]),
                inel is not None,
                inel.reason if inel else None,
                inel.source if inel else None,
                awards.LABEL,
                awards.BROWNLOW_PROXY_VERSION,
                f"Season-scaled proxy (x{scale}; index, not votes)",
            ]
        )
    return {"header": header, "rows": out, "key": 1}


def biography_csv(con: duckdb.DuckDBPyConnection, all_time: list[tuple[str, str, float]]) -> dict[str, Any]:
    """Legacy biography shape: serial, name, teams; numeric claims in the comment are checked as phrases."""
    ids = [p for p, _k, _s in all_time]
    people = {
        r["player_id"]: r
        for r in _rows(
            con, "SELECT player_id, first_name, last_name FROM players WHERE list_contains(?, player_id)", [ids]
        )
    }
    facts = {
        r["player_id"]: r
        for r in _rows(
            con,
            """SELECT player_id, count(*) AS n, greatest(count(*), coalesce(max(career_game_counter), 0)) AS cg,
                  count(DISTINCT season) AS seasons,
                  sum(coalesce(kicks, 0) + coalesce(handballs, 0))::BIGINT AS disp,
                  sum(coalesce(goals, 0))::BIGINT AS goals, sum(coalesce(brownlow_votes, 0))::BIGINT AS votes
           FROM player_games WHERE list_contains(?, player_id) GROUP BY player_id""",
            [ids],
        )
    }
    teams: dict[str, list[str]] = defaultdict(list)
    for pid, team in con.execute(
        """SELECT g.player_id, coalesce(c.name, g.club_id) FROM player_games g LEFT JOIN clubs c USING (club_id)
           WHERE list_contains(?, g.player_id) ORDER BY g.player_id, g.season, g.match_date NULLS LAST, g.match_id""",
        [ids],
    ).fetchall():
        if team not in teams[pid]:
            teams[pid].append(team)
    rows = []
    for serial, pid in enumerate(ids, start=1):
        p, f = people.get(pid), facts.get(pid)
        name = f"{p['first_name']} {p['last_name']}" if p and p["first_name"] and p["last_name"] else "Unknown"
        if f is None:
            rows.append([serial, name, "Unknown", contains("No performance data available.")])
            continue
        phrases = [f"over {f['seasons']} seasons and {f['cg']} games"]
        if f["disp"] > 0:
            phrases.append(f"{f['disp']} total disposals")
        if f["goals"] > 0:
            phrases.append(f"{f['goals']} goals")
        if f["votes"] > 0:
            phrases.append(f"He earned {f['votes']} Brownlow votes")
        rows.append([serial, name, " - ".join(teams[pid]), contains(*phrases)])
    return {"header": ["Serial Number", "Player Name", "Footy Teams", "Comment"], "rows": rows, "key": None}
