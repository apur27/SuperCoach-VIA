"""``legacy_v1`` all-time and yearly top-100 rankings (PLAN section 6; tests A01/A02).

A faithful port of ``top_players_comprehensive.py`` (``_aggregate_one_file``,
``_generate_yearly_from_memory``, ``compile_all_time_top_100``, ``format_top_100``) driven
by the complete, hashed config in ``config/ranking_legacy_v1.toml``.

Methodology notes that are part of ``legacy_v1`` and deliberately NOT coverage-aware:

* A blank stat cell counts as zero (legacy zero fill; ``metrics.legacy_zero_fill_sum``).
* Players are scored per season on their era's stats with a single-stat cap, integer
  truncated; non-positive scores leave the cohort; full-cohort z-scores (population SD)
  are clipped to +/- ``z_cap`` and shrunk by ``sqrt(era_completeness)``.
* All-time: per yearly appearance, ``year_score = (1-b)*((101-rank)/100)**gamma +
  b*z_signal``, times era completeness; mean of the best ``top_n_seasons``; times
  ``1 + factor*min(seasons/cap, 1)``; active players (in a yearly list for a
  ``recent_years`` season) discounted; eligibility ``career_games >= min_career_games``
  with ``career_games = max(rows, max career counter)``.

Differences from the legacy script, all deliberate and documented:

* Exact ties are broken by ``player_id`` ascending (legacy: filesystem glob order, which is
  not reproducible across machines).
* Inputs come from the canonical snapshot, where duplicate legacy files (e.g. the Will /
  William Green and Roan Steele duplicates) are quarantined instead of being counted as two
  players.
* The yearly top-100 keeps the season-END publication cadence: a season that is not known to
  be complete is returned with ``provisional=True`` / ``lifecycle='provisional'``.
"""

from __future__ import annotations

# SQL is composed only from whitelisted identifiers (canonical stat/column names) and int-cast
# literals; every caller-supplied value is a bound parameter.
# ruff: noqa: S608
import hashlib
import json
import math
import tomllib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from supercoach_via.storage.queries import SnapshotQuery

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "ranking_legacy_v1.toml"

NUMERIC_EXPORT_COLUMNS: tuple[str, ...] = ("player", "all_time_score")
BIOGRAPHY_EXPORT_COLUMNS: tuple[str, ...] = ("Serial Number", "Player Name", "Footy Teams", "Comment")
YEARLY_EXPORT_COLUMNS: tuple[str, ...] = ("player", "score", "percentile_rank", "games_played")

Lifecycle = Literal["final", "provisional"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Era:
    name: str
    start: int
    end: int
    stats: tuple[str, ...]


@dataclass(frozen=True)
class RankingConfig:
    formula_version: str
    z_cap: float
    top_n_seasons: int
    rank_gamma: float
    z_blend: float
    single_stat_cap: float
    min_position_group: int
    active_player_discount: float
    recent_years: tuple[int, ...]
    yearly_top_n: int
    all_time_top_n: int
    min_career_games: int
    career_bonus_factor: float
    career_bonus_seasons_cap: float
    missing_era_shrinkage: float
    z_ddof: int
    era_completeness: tuple[tuple[str, float], ...]
    eras: tuple[Era, ...]
    weights: tuple[tuple[str, float], ...]
    ordering: tuple[tuple[str, tuple[str, ...]], ...] = ()
    baseline: str = ""
    imputation: str = ""

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> RankingConfig:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
        c = raw["constants"]
        return cls(
            formula_version=str(raw["formula_version"]),
            z_cap=float(c["z_cap"]),
            top_n_seasons=int(c["top_n_seasons"]),
            rank_gamma=float(c["rank_gamma"]),
            z_blend=float(c["z_blend"]),
            single_stat_cap=float(c["single_stat_cap"]),
            min_position_group=int(c["min_position_group"]),
            active_player_discount=float(c["active_player_discount"]),
            recent_years=tuple(int(y) for y in c["recent_years"]),
            yearly_top_n=int(c["yearly_top_n"]),
            all_time_top_n=int(c["all_time_top_n"]),
            min_career_games=int(c["min_career_games"]),
            career_bonus_factor=float(c["career_bonus_factor"]),
            career_bonus_seasons_cap=float(c["career_bonus_seasons_cap"]),
            missing_era_shrinkage=float(c["missing_era_shrinkage"]),
            z_ddof=int(c["z_ddof"]),
            era_completeness=tuple((k, float(v)) for k, v in raw["era_completeness"].items()),
            eras=tuple(
                Era(e["name"], int(e["start"]), int(e["end"]), tuple(e["stats"])) for e in raw["eras"]
            ),
            weights=tuple((k, float(v)) for k, v in raw["weights"].items()),
            ordering=tuple((k, tuple(v)) for k, v in raw.get("ordering", {}).items()),
            baseline=str(raw.get("baseline", "")),
            imputation=str(raw.get("imputation", "")),
        )

    # Mapping views (the dataclass stores tuples so it is hashable/frozen).
    @property
    def completeness(self) -> dict[str, float]:
        return dict(self.era_completeness)

    @property
    def weight_map(self) -> dict[str, float]:
        return dict(self.weights)

    @property
    def all_stats(self) -> tuple[str, ...]:
        return tuple(sorted({s for e in self.eras for s in e.stats}))

    def era_for(self, season: int) -> Era | None:
        for era in self.eras:
            if era.start <= season <= era.end:
                return era
        return None

    def config_hash(self) -> str:
        """sha256 over the complete semantic config (canonical JSON)."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=list)
        return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeasonTotals:
    player_id: str
    player_key: str  # legacy slug when known (legacy export key), else player_id
    season: int
    totals: Mapping[str, float]
    games: int  # rows in the season


@dataclass(frozen=True)
class RankingInputs:
    by_season: dict[int, list[SeasonTotals]]
    career_games: dict[str, int]  # canonical: max(rows, max career counter)
    player_keys: dict[str, str]


def has_table(q: SnapshotQuery, name: str) -> bool:
    return name in q.manifest.tables and (q.tables is None or name in q.tables)


def load_ranking_inputs(q: SnapshotQuery, cfg: RankingConfig) -> RankingInputs:
    """One DuckDB pass: per (player, season) zero-filled totals + canonical career games."""
    stats = cfg.all_stats
    sums = ", ".join(f'SUM(COALESCE("{s}", 0))::DOUBLE AS "{s}"' for s in stats)
    key_expr = "g.player_id"
    join = ""
    if has_table(q, "players"):
        key_expr = "COALESCE(p.legacy_slug, g.player_id)"
        join = "LEFT JOIN players p USING (player_id)"
    rows = q.rows(
        f"""
        SELECT g.player_id, {key_expr} AS player_key, g.season, COUNT(*) AS games, {sums}
        FROM player_games g {join}
        GROUP BY ALL
        ORDER BY player_key, g.player_id, g.season
        """
    )
    by_season: dict[int, list[SeasonTotals]] = defaultdict(list)
    keys: dict[str, str] = {}
    for r in rows:
        pid, key, season, games = r[0], r[1], int(r[2]), int(r[3])
        totals = {s: float(v) for s, v in zip(stats, r[4:], strict=True)}
        by_season[season].append(SeasonTotals(pid, key, season, totals, games))
        keys[pid] = key
    career = {
        pid: max(int(n), int(mx or 0))
        for pid, n, mx in q.rows(
            "SELECT player_id, COUNT(*), MAX(career_game_counter) FROM player_games GROUP BY player_id"
        )
    }
    return RankingInputs(dict(by_season), career, keys)


# ---------------------------------------------------------------------------
# Yearly ranking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YearlyEntry:
    season: int
    rank: int
    player_id: str
    player_key: str
    score: int
    games: int
    percentile_rank: float
    z_adj: float
    totals: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class YearlyRanking:
    season: int
    entries: tuple[YearlyEntry, ...]
    cohort_size: int
    provisional: bool

    @property
    def lifecycle(self) -> Lifecycle:
        return "provisional" if self.provisional else "final"


def _percentile_ranks(scores: Sequence[int]) -> list[float]:
    import pandas as pd

    return [float(x) * 100 for x in pd.Series(scores).rank(pct=True).to_numpy()]


def _z_scores(scores: Sequence[int], cfg: RankingConfig) -> list[float]:
    import numpy as np

    if not scores:
        return []
    arr = np.asarray(scores, dtype=float)
    sigma = arr.std(ddof=cfg.z_ddof)
    if sigma == 0 or not np.isfinite(sigma):
        return [0.0] * len(scores)
    z = np.clip((arr - arr.mean()) / sigma, -cfg.z_cap, cfg.z_cap)
    return [float(v) for v in z]


def rank_season(season: int, rows: Sequence[SeasonTotals], cfg: RankingConfig) -> list[YearlyEntry]:
    """Port of ``_generate_yearly_from_memory`` for one season."""
    era = cfg.era_for(season)
    if era is None:
        return []
    weights = cfg.weight_map
    completeness = cfg.completeness
    shrink = math.sqrt(completeness.get(era.name, cfg.missing_era_shrinkage))
    scored: list[tuple[SeasonTotals, int, dict[str, int]]] = []
    for row in rows:
        contrib = {s: row.totals.get(s, 0.0) * weights.get(s, 0.0) for s in era.stats if s in row.totals}
        if not contrib:
            continue
        uncapped = sum(contrib.values())
        if uncapped <= 0:
            continue
        excess = sum(max(0.0, c - cfg.single_stat_cap * uncapped) for c in contrib.values())
        score = max(0, int(uncapped - excess))
        era_totals = {s: int(row.totals.get(s, 0)) for s in era.stats if s in row.totals}
        scored.append((row, score, era_totals))
    if not scored:
        return []
    scores = [s for _, s, _ in scored]
    pct = _percentile_ranks(scores)
    z = [v * shrink for v in _z_scores(scores, cfg)]
    order = sorted(range(len(scored)), key=lambda i: scored[i][0].player_id)
    order.sort(key=lambda i: (scored[i][1], scored[i][0].games), reverse=True)
    out = []
    for rank, i in enumerate(order[: cfg.yearly_top_n], start=1):
        row, score, era_totals = scored[i]
        out.append(
            YearlyEntry(season, rank, row.player_id, row.player_key, score, row.games, pct[i], z[i], era_totals)
        )
    return out


# ---------------------------------------------------------------------------
# All-time ranking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AllTimeEntry:
    rank: int
    player_id: str
    player_key: str
    all_time_score: float
    mean_adj: float
    seasons_in_top: int
    career_games: int
    active: bool
    best_rank: int
    best_season: int


def compile_all_time(
    yearly: Mapping[int, Sequence[YearlyEntry]],
    career_games: Mapping[str, int],
    cfg: RankingConfig,
) -> list[AllTimeEntry]:
    """Port of ``compile_all_time_top_100``."""
    completeness = cfg.completeness
    recent = set(cfg.recent_years)
    active = {e.player_id for season, es in yearly.items() if season in recent for e in es}
    adj_scores: dict[str, list[float]] = defaultdict(list)
    keys: dict[str, str] = {}
    best: dict[str, tuple[int, int]] = {}
    for season in sorted(yearly):
        era = cfg.era_for(season)
        ec = completeness.get(era.name if era else "unknown", completeness["unknown"])
        for e in sorted(yearly[season], key=lambda x: x.rank):
            rank_score = ((101 - e.rank) / 100.0) ** cfg.rank_gamma
            z_signal = max(0.0, min(1.0, (e.z_adj + cfg.z_cap) / (2.0 * cfg.z_cap)))
            year_score = (1.0 - cfg.z_blend) * rank_score + cfg.z_blend * z_signal
            adj_scores[e.player_id].append(year_score * ec)
            keys[e.player_id] = e.player_key
            if e.player_id not in best or e.rank < best[e.player_id][0]:
                best[e.player_id] = (e.rank, season)
    scored: list[tuple[str, float, float, int, int]] = []
    for pid, adj in adj_scores.items():
        cg = career_games.get(pid, 0)
        if cg < cfg.min_career_games:
            continue
        top_n = sorted(adj, reverse=True)[: cfg.top_n_seasons]
        mean_adj = sum(top_n) / len(top_n)
        seasons = len(adj)
        bonus = cfg.career_bonus_factor * min(seasons / cfg.career_bonus_seasons_cap, 1.0)
        score = mean_adj * (1.0 + bonus)
        if pid in active:
            score *= cfg.active_player_discount
        scored.append((pid, score, mean_adj, cg, seasons))
    scored.sort(key=lambda x: x[0])
    scored.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    return [
        AllTimeEntry(
            rank=i,
            player_id=pid,
            player_key=keys[pid],
            all_time_score=score,
            mean_adj=mean_adj,
            seasons_in_top=seasons,
            career_games=cg,
            active=pid in active,
            best_rank=best[pid][0],
            best_season=best[pid][1],
        )
        for i, (pid, score, mean_adj, cg, seasons) in enumerate(scored[: cfg.all_time_top_n], start=1)
    ]


# ---------------------------------------------------------------------------
# Season lifecycle (season-end cadence)
# ---------------------------------------------------------------------------


def season_completion(q: SnapshotQuery, seasons: Sequence[int]) -> dict[int, bool]:
    """True when a season is known complete.

    Order of evidence: ``seasons.schedule_complete``; a completed, non-drawn Grand Final;
    a later season present in ``player_games`` (the earlier one must have finished).
    Anything else is not known complete (=> provisional).
    """
    declared: dict[int, bool] = {}
    if has_table(q, "seasons"):
        for season, complete in q.rows("SELECT season, schedule_complete FROM seasons"):
            if complete is not None:
                declared[int(season)] = bool(complete)
    gf_done: set[int] = set()
    if has_table(q, "matches"):
        gf_done = {
            int(r[0])
            for r in q.rows(
                """SELECT DISTINCT season FROM matches
                   WHERE status = 'complete' AND (lower(stage_label) IN ('gf', 'grand final')
                         OR lower(stage_id) = 'gf')
                     AND home_score IS NOT NULL AND away_score IS NOT NULL
                     AND home_score <> away_score"""
            )
        }
    latest = max(seasons) if seasons else None
    out: dict[int, bool] = {}
    for s in seasons:
        if s in declared:
            out[s] = declared[s]
        elif s in gf_done:
            out[s] = True
        else:
            out[s] = latest is not None and s < latest
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankingResult:
    formula_version: str
    config_hash: str
    snapshot_id: str
    yearly: dict[int, YearlyRanking]
    all_time: tuple[AllTimeEntry, ...]
    career_games: dict[str, int]
    player_keys: dict[str, str]
    provisional_seasons: tuple[int, ...]


def run_legacy_v1(q: SnapshotQuery, cfg: RankingConfig | None = None) -> RankingResult:
    """Yearly top-100 for every season + the all-time top-100 (``legacy_v1``)."""
    cfg = cfg or RankingConfig.load()
    inputs = load_ranking_inputs(q, cfg)
    seasons = sorted(inputs.by_season)
    complete = season_completion(q, seasons)
    yearly: dict[int, YearlyRanking] = {}
    for season in seasons:
        rows = inputs.by_season[season]
        entries = rank_season(season, rows, cfg)
        if entries:
            yearly[season] = YearlyRanking(season, tuple(entries), len(rows), not complete[season])
    all_time = compile_all_time({s: y.entries for s, y in yearly.items()}, inputs.career_games, cfg)
    return RankingResult(
        formula_version=cfg.formula_version,
        config_hash=cfg.config_hash(),
        snapshot_id=q.manifest.snapshot_id,
        yearly=yearly,
        all_time=tuple(all_time),
        career_games=inputs.career_games,
        player_keys=inputs.player_keys,
        provisional_seasons=tuple(s for s, y in yearly.items() if y.provisional),
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def numeric_export_rows(result: RankingResult) -> list[tuple[str, float]]:
    """``data/top100/all_time_top_100.csv`` shape: (player, all_time_score)."""
    return [(e.player_key, e.all_time_score) for e in result.all_time]


def yearly_export_rows(yearly: YearlyRanking) -> list[tuple[str, int, float, int]]:
    """``data/top100/yearly/year_<season>.csv`` shape."""
    return [(e.player_key, e.score, e.percentile_rank, e.games) for e in yearly.entries]


def _fmt_measure(v: float | None) -> str:
    if v is None:
        return ""
    return str(int(v)) if float(v).is_integer() else str(v)


def _fmt_date(v: Any) -> str:
    return "" if v is None else v.strftime("%d-%m-%Y")


def biography_export_rows(q: SnapshotQuery, result: RankingResult) -> list[tuple[int, str, str, str]]:
    """Root ``all_time_top_100.csv`` shape (presentation/biography, NOT a numeric table).

    Port of ``format_top_100`` text. Team names use canonical club names.
    """
    import pandas as pd

    ids = [e.player_id for e in result.all_time]
    if not ids:
        return []
    id_list = ", ".join("'" + i.replace("'", "''") + "'" for i in ids)
    club_name = "g.club_id"
    club_join = ""
    if has_table(q, "clubs"):
        club_name = "COALESCE(c.name, g.club_id)"
        club_join = "LEFT JOIN clubs c ON c.club_id = g.club_id"
    games = q.df(
        f"""SELECT g.player_id, g.season, {club_name} AS team, g.career_game_counter,
                   g.kicks, g.handballs, g.goals, g.brownlow_votes
            FROM player_games g {club_join}
            WHERE g.player_id IN ({id_list})
            ORDER BY g.player_id, g.season, g.match_date NULLS LAST, g.match_id"""
    )
    people: dict[str, tuple[Any, ...]] = {}
    if has_table(q, "players"):
        for r in q.rows(
            f"""SELECT player_id, first_name, last_name, birth_date, debut_date, height_cm, weight_kg
                FROM players WHERE player_id IN ({id_list})"""
        ):
            people[r[0]] = r[1:]
    out: list[tuple[int, str, str, str]] = []
    for serial, entry in enumerate(result.all_time, start=1):
        pers = people.get(entry.player_id)
        if pers and pers[0] and pers[1]:
            full_name = f"{pers[0]} {pers[1]}"
            base = (
                f"{full_name}, born on {_fmt_date(pers[2])}, debuted on {_fmt_date(pers[3])}, "
                f"height {_fmt_measure(pers[4])} cm, weight {_fmt_measure(pers[5])} kg"
            )
        else:
            full_name, base = "Unknown", "Unknown player"
        g = games[games["player_id"] == entry.player_id]
        if g.empty:
            out.append((serial, full_name, "Unknown", f"{base}. No performance data available."))
            continue
        teams = list(dict.fromkeys(g["team"].tolist()))
        disp = (g["kicks"].fillna(0) + g["handballs"].fillna(0)).astype(int)
        goals = g["goals"].fillna(0).astype(int)
        total_games = len(g)
        career = max(total_games, int(pd.to_numeric(g["career_game_counter"]).max() or 0))
        votes = int(g["brownlow_votes"].fillna(0).sum())
        tot_disp, tot_goals = int(disp.sum()), int(goals.sum())
        g20, g3 = int((disp >= 20).sum()), int((goals >= 3).sum())
        n_seasons = g["season"].nunique()
        impact = (votes / total_games) * 100 if votes > 0 else (tot_disp / total_games if tot_disp else 0.0)
        consistency = (g20 / total_games) * 100 if g20 else 0.0
        teams_str = " - ".join(teams)
        s = f"{base}. A true legend of the game, {full_name} played for {teams_str}"
        if n_seasons > 0 and career > 0:
            s += f" over {n_seasons} seasons and {career} games"
        s += "."
        if tot_disp > 0 or tot_goals > 0:
            s += " He recorded"
            if tot_disp > 0:
                s += f" {tot_disp} total disposals"
            if tot_goals > 0:
                s += (" and" if tot_disp > 0 else "") + f" {tot_goals} goals"
            s += "."
        if impact > 0:
            basis = "Brownlow votes" if votes > 0 else "disposals"
            s += f" His impact is shown by a {impact:.1f} Impact Score (based on {basis})."
        if consistency > 0:
            s += f" He maintained a {consistency:.1f}% Consistency Score."
        pk_d, pk_g = int(disp.max()), int(goals.max())
        if pk_d > 0 or pk_g > 0:
            s += " Peak performances include"
            if pk_d > 0:
                s += f" {pk_d} disposals"
            if pk_g > 0:
                s += (" and" if pk_d > 0 else "") + f" {pk_g} goals"
            s += " in a single game."
        if g20 > 0 or g3 > 0:
            s += " His consistency shines with"
            if g20 > 0:
                s += f" {g20} games of 20+ disposals"
            if g3 > 0:
                s += (" and" if g20 > 0 else "") + f" {g3} games with 3+ goals"
            s += "."
        if votes > 0:
            s += f" He earned {votes} Brownlow votes, cementing his greatness."
        else:
            s += " His legacy as a great is undeniable."
        out.append((serial, full_name, teams_str, s))
    return out


# ---------------------------------------------------------------------------
# View models
# ---------------------------------------------------------------------------


def player_meta(q: SnapshotQuery, ids: Sequence[str]) -> dict[str, tuple[str, list[str], str | None]]:
    """(display name, clubs, season span) per player; per-player values do not depend on ``ids``."""
    return _player_meta(q, ids)


def _player_meta(q: SnapshotQuery, ids: Sequence[str]) -> dict[str, tuple[str, list[str], str | None]]:
    if not ids:
        return {}
    id_list = ", ".join("'" + i.replace("'", "''") + "'" for i in ids)
    names: dict[str, str] = {}
    if has_table(q, "players"):
        names = dict(q.rows(f"SELECT player_id, display_name FROM players WHERE player_id IN ({id_list})"))
    club_expr, club_join = "g.club_id", ""
    if has_table(q, "clubs"):
        club_expr, club_join = "COALESCE(c.name, g.club_id)", "LEFT JOIN clubs c ON c.club_id = g.club_id"
    rows = q.rows(
        f"""SELECT g.player_id, {club_expr}, MIN(g.season), MAX(g.season)
            FROM player_games g {club_join} WHERE g.player_id IN ({id_list})
            GROUP BY ALL ORDER BY g.player_id, MIN(g.season), 2"""
    )
    clubs: dict[str, list[str]] = defaultdict(list)
    span: dict[str, tuple[int, int]] = {}
    for pid, club, lo, hi in rows:
        clubs[pid].append(club)
        a, b = span.get(pid, (lo, hi))
        span[pid] = (min(a, lo), max(b, hi))
    return {
        pid: (names.get(pid, pid), clubs.get(pid, []), f"{span[pid][0]}-{span[pid][1]}" if pid in span else None)
        for pid in ids
    }


LEGACY_COVERAGE_NOTE = (
    "legacy_v1 counts a blank stat cell as zero (documented historical imputation); scores use "
    "era-specific stat sets, so pre-1965 seasons are scored on goals and behinds only."
)


def _method_text(result: RankingResult) -> str:
    return (
        f"{result.formula_version} rank-based all-time formula (config sha256 "
        f"{result.config_hash[:12]}, snapshot {result.snapshot_id})"
    )


def all_time_history_table(q: SnapshotQuery, result: RankingResult) -> Any:
    from supercoach_via.publish.view_models import HistoryRow, HistoryTable

    meta = _player_meta(q, [e.player_id for e in result.all_time])
    rows = [
        HistoryRow(
            rank=e.rank,
            player_id=e.player_id,
            name=meta[e.player_id][0],
            clubs=meta[e.player_id][1],
            value=e.all_time_score,
            value_label="all-time score",
            observed_games=e.career_games,
            eligible_games=None,
            coverage=None,
            seasons=meta[e.player_id][2],
        )
        for e in result.all_time
    ]
    warning = None
    if result.provisional_seasons:
        warning = (
            "Includes provisional (in-progress) seasons: "
            + ", ".join(str(s) for s in result.provisional_seasons)
        )
    return HistoryTable(
        category="all_time_top_100",
        title="All-time top 100",
        scope="ranking",
        era="all",
        method=_method_text(result),
        method_version=result.formula_version,
        coverage_note=LEGACY_COVERAGE_NOTE,
        warning=warning,
        rows=rows,
    )


def yearly_history_table(
    q: SnapshotQuery,
    result: RankingResult,
    season: int,
    *,
    meta: Mapping[str, tuple[str, list[str], str | None]] | None = None,
) -> Any:
    """``meta``: optional ``player_meta`` over a superset of this season's players (batched builds)."""
    from supercoach_via.publish.view_models import HistoryRow, HistoryTable

    yr = result.yearly[season]
    if meta is None:
        meta = _player_meta(q, [e.player_id for e in yr.entries])
    rows = [
        HistoryRow(
            rank=e.rank,
            player_id=e.player_id,
            name=meta[e.player_id][0],
            clubs=meta[e.player_id][1],
            value=float(e.score),
            value_label="season score",
            observed_games=e.games,
            eligible_games=None,
            coverage=None,
            seasons=str(season),
        )
        for e in yr.entries
    ]
    warning = (
        f"Provisional: season {season} is in progress; the yearly top 100 is published at season end."
        if yr.provisional
        else None
    )
    return HistoryTable(
        category=f"yearly_top_100_{season}",
        title=f"Top 100 of {season}",
        scope="ranking",
        era=str(season),
        method=_method_text(result),
        method_version=result.formula_version,
        coverage_note=LEGACY_COVERAGE_NOTE,
        warning=warning,
        rows=rows,
    )
