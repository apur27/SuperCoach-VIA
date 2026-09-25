"""Era summaries and adjacent-era significance tests.

Port of ``supercoach/era_based_statistical_analysis.py`` with the SAME defined statistics
(computed in DuckDB instead of a per-file pandas loop):

* rows are bucketed by match season into the legacy eras (pre-1965, 1965-1990, 1991-2010,
  2011-present); the last era now extends to the latest season in the data (legacy hard-coded
  2026 and would silently drop later seasons into 'unknown');
* per era x metric: ``n_player_games`` (all rows), ``n_with_metric`` (observed rows), mean /
  sample SD / median over observed rows (missing is never zero), and the mean per 100%
  time-on-ground over rows with time_on_ground_pct >= 25;
* adjacent eras: Welch's t-test on observed rows (skipped when either side has < 30), raw delta
  and Cohen's d with the legacy pooled SD ``sqrt((sa^2 + sb^2)/2)``;
* team scoring per era from match final goals/behinds.

Additions: explicit recording/methodology boundaries from ``config/stat_coverage_eras.yaml``
(``recording_status`` per era x metric; ``boundary_note`` on each test). P-values treat
player-game rows as independent (they are not: players and matches repeat) — rough indicators
only, as the legacy report said; read effect sizes.
"""

from __future__ import annotations

# SQL is composed only from whitelisted identifiers (canonical stat/column names) and int-cast
# literals; every caller-supplied value is a bound parameter.
# ruff: noqa: S608
import itertools
import math
from typing import TYPE_CHECKING, Any

from supercoach_via.domain.metrics import CoverageEras
from supercoach_via.domain.schemas import LEGACY_PLAYER_COLUMN_MAP, PLAYER_STAT_COLUMNS
from supercoach_via.storage.queries import SnapshotQuery

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

LEGACY_ERA_BOUNDS: tuple[tuple[str, int, int], ...] = (
    ("pre-1965", 1897, 1964),
    ("1965-1990", 1965, 1990),
    ("1991-2010", 1991, 2010),
    ("2011-present", 2011, 2026),
)
PLAYER_METRICS: tuple[str, ...] = PLAYER_STAT_COLUMNS  # same 23 metrics, canonical names
_TO_LEGACY = {v: k for k, v in LEGACY_PLAYER_COLUMN_MAP.items()}
MIN_TEST_N = 30
TOG_MIN = 25.0

#: Methodology (not availability) boundaries: comparisons across them are not like-for-like.
METHOD_BOUNDARIES: dict[str, tuple[int, str]] = {
    "hitouts": (2017, "hit-out counting method changed between 2016 and 2017"),
}


def legacy_name(metric: str) -> str:
    return _TO_LEGACY.get(metric, metric)


def era_bounds(q: SnapshotQuery) -> tuple[tuple[str, int, int], ...]:
    latest = q.scalar("SELECT MAX(season) FROM player_games")
    last = LEGACY_ERA_BOUNDS[-1]
    end = max(last[2], int(latest)) if latest is not None else last[2]
    return (*LEGACY_ERA_BOUNDS[:-1], (last[0], last[1], end))


def _era_case(bounds: tuple[tuple[str, int, int], ...], col: str = "season") -> str:
    whens = " ".join(f"WHEN {col} BETWEEN {lo} AND {hi} THEN '{name}'" for name, lo, hi in bounds)
    return f"CASE {whens} ELSE 'unknown' END"


def recording_status(metric: str, lo: int, hi: int, eras: CoverageEras | None) -> str:
    start = eras.recorded_from(metric) if eras else None
    if start is None or start <= lo:
        return "recorded"
    if start > hi:
        return "not_recorded"
    return "partial_era"


def _default_eras() -> CoverageEras:
    from pathlib import Path

    return CoverageEras.load(Path(__file__).resolve().parents[3] / "config" / "stat_coverage_eras.yaml")


def era_stats(q: SnapshotQuery, *, eras: CoverageEras | None = None) -> pd.DataFrame:
    import pandas as pd

    eras = eras if eras is not None else _default_eras()
    bounds = era_bounds(q)
    era = _era_case(bounds)
    parts = []
    for m in PLAYER_METRICS:
        norm = (
            f'AVG("{m}")'
            if m == "time_on_ground_pct"
            else f'AVG("{m}" * (100.0 / time_on_ground_pct)) FILTER (WHERE time_on_ground_pct >= {TOG_MIN})'
        )
        parts.append(
            f"""SELECT era, '{m}' AS metric, COUNT(*) AS n_player_games, COUNT("{m}") AS n_with_metric,
                       AVG("{m}") AS mean_per_game, STDDEV_SAMP("{m}") AS std_per_game,
                       MEDIAN("{m}") AS median_per_game, {norm} AS mean_per_100pct_played
                FROM g GROUP BY era"""
        )
    df = q.df(
        f"WITH g AS (SELECT *, {era} AS era FROM player_games) " + " UNION ALL ".join(parts)
    )
    order = {name: i for i, (name, _, _) in enumerate(bounds)}
    df = df[df["era"].isin(order)].copy()
    df["legacy_metric"] = df["metric"].map(legacy_name)
    rng = {name: (lo, hi) for name, lo, hi in bounds}
    df["recorded_from"] = [eras.recorded_from(m) for m in df["metric"]]
    df["recording_status"] = [
        recording_status(m, *rng[e], eras) for m, e in zip(df["metric"], df["era"], strict=True)
    ]
    df.loc[df["n_with_metric"] == 0, ["std_per_game", "median_per_game"]] = float("nan")
    df["_e"] = df["era"].map(order)
    df["_m"] = df["metric"].map({m: i for i, m in enumerate(PLAYER_METRICS)})
    out = df.sort_values(["_e", "_m"]).drop(columns=["_e", "_m"]).reset_index(drop=True)
    return pd.DataFrame(out)


def yearly_trends(q: SnapshotQuery) -> pd.DataFrame:
    """Export shape of ``data/era_yearly_trends.csv`` (legacy column names)."""
    cols = ", ".join(f'AVG("{m}") AS "{legacy_name(m)}"' for m in PLAYER_METRICS)
    return q.df(
        f"SELECT season AS year, {cols}, COUNT(*) AS n_player_games FROM player_games GROUP BY season ORDER BY season"
    )


def _boundary_note(metric: str, bounds: list[tuple[str, int, int]], eras: CoverageEras | None) -> str:
    notes = []
    for name, lo, hi in bounds:
        if recording_status(metric, lo, hi, eras) == "partial_era":
            notes.append(f"recording starts {eras.recorded_from(metric) if eras else '?'} inside {name}")
    if metric in METHOD_BOUNDARIES:
        year, why = METHOD_BOUNDARIES[metric]
        if any(lo < year <= hi for _, lo, hi in bounds):
            notes.append(f"methodology boundary {year}: {why}")
    return "; ".join(notes)


def adjacent_era_tests(q: SnapshotQuery, *, eras: CoverageEras | None = None) -> pd.DataFrame:
    import pandas as pd
    from scipy import special  # type: ignore[import-untyped]  # scipy: sklearn runtime dependency

    eras = eras if eras is not None else _default_eras()
    bounds = era_bounds(q)
    era = _era_case(bounds)
    parts = [
        f"""SELECT era, '{m}' AS metric, COUNT("{m}") AS n, AVG("{m}") AS mean, VAR_SAMP("{m}") AS var
            FROM g GROUP BY era"""
        for m in PLAYER_METRICS
    ]
    mom = q.df(f"WITH g AS (SELECT *, {era} AS era FROM player_games) " + " UNION ALL ".join(parts))
    idx: dict[tuple[str, str], dict[str, Any]] = {
        (str(r["era"]), str(r["metric"])): r for r in mom.to_dict("records")  # type: ignore[misc]
    }
    rows = []
    for (a, alo, ahi), (b, blo, bhi) in itertools.pairwise(bounds):
        for m in PLAYER_METRICS:
            ra, rb = idx.get((a, m)), idx.get((b, m))
            na = int(ra["n"]) if ra is not None else 0
            nb = int(rb["n"]) if rb is not None else 0
            ma = float(ra["mean"]) if ra is not None and na else math.nan
            mb = float(rb["mean"]) if rb is not None and nb else math.nan
            note = _boundary_note(m, [(a, alo, ahi), (b, blo, bhi)], eras)
            base = {"era_a": a, "era_b": b, "metric": m, "legacy_metric": legacy_name(m),
                    "n_a": na, "n_b": nb, "mean_a": ma, "mean_b": mb, "boundary_note": note}
            if na < MIN_TEST_N or nb < MIN_TEST_N:
                rows.append({**base, "delta": math.nan, "cohens_d": math.nan, "welch_t": math.nan,
                             "p_value": math.nan, "note": "insufficient data"})
                continue
            assert ra is not None and rb is not None
            va, vb = float(ra["var"]), float(rb["var"])
            sa, sb = math.sqrt(va), math.sqrt(vb)
            pooled = math.sqrt((va + vb) / 2.0) if (sa + sb) > 0 else math.nan
            d = (mb - ma) / pooled if pooled and pooled > 0 else math.nan
            se2 = va / na + vb / nb
            t = (ma - mb) / math.sqrt(se2) if se2 > 0 else math.nan
            dof = se2**2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)) if se2 > 0 else math.nan
            p = float(2.0 * special.stdtr(dof, -abs(t))) if not math.isnan(t) else math.nan
            rows.append({**base, "delta": mb - ma, "cohens_d": d, "welch_t": t, "p_value": p, "note": ""})
    return pd.DataFrame(rows)


def team_scoring(q: SnapshotQuery) -> pd.DataFrame:
    """Per-era team-game scoring from match final goals/behinds (legacy definitions)."""
    bounds = era_bounds(q)
    era = _era_case(bounds)
    order = ", ".join(f"('{n}', {i})" for i, (n, _, _) in enumerate(bounds))
    return q.df(
        f"""WITH m AS (SELECT *, {era} AS era FROM matches),
            tg AS (
              SELECT era, home_final_goals AS goals, home_final_behinds AS behinds FROM m
              UNION ALL SELECT era, away_final_goals, away_final_behinds FROM m),
            t AS (SELECT era, goals, behinds, goals * 6 + behinds AS points, goals + behinds AS shots
                  FROM tg WHERE goals IS NOT NULL AND behinds IS NOT NULL),
            a AS (SELECT era, COUNT(points) AS n_team_games, AVG(goals) AS mean_goals,
                         AVG(behinds) AS mean_behinds, AVG(points) AS mean_points,
                         AVG(shots) AS mean_scoring_shots,
                         AVG(goals / NULLIF(shots, 0)) AS mean_goal_accuracy,
                         STDDEV_SAMP(points) AS std_points FROM t GROUP BY era),
            mt AS (SELECT era, AVG(home_final_goals + away_final_goals) AS mean_match_total_goals,
                          AVG((home_final_goals + away_final_goals) * 6 + home_final_behinds
                              + away_final_behinds) AS mean_match_total_points,
                          COUNT((home_final_goals + away_final_goals) * 6 + home_final_behinds
                                + away_final_behinds) AS n_matches
                   FROM m GROUP BY era),
            o(era, k) AS (VALUES {order})
            SELECT a.*, mt.mean_match_total_goals, mt.mean_match_total_points, mt.n_matches
            FROM a LEFT JOIN mt USING (era) JOIN o USING (era) ORDER BY o.k"""
    )
