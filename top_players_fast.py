"""
Fast single-pass equivalent of top_players_comprehensive.py.

The original iterates over years (1897..current) and for each year reads ALL
~13k player CSVs — that's 1.7M file reads on a fresh refresh. This rewrite
makes ONE pass over each player file, builds (player, year) -> stat-totals,
then computes yearly top-100 and all-time top-100 entirely from in-memory
aggregates.

Outputs are bit-for-bit comparable to the original:
  - data/top100/yearly/year_<YYYY>.csv  with columns
        player, score, percentile_rank, games_played
  - data/top100/all_time_top_100.csv with columns
        player, all_time_score

Run:
    /home/abhi/sourceCode/python/coding/.venv/bin/python top_players_fast.py
"""
from __future__ import annotations

import glob
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


# Eras and the stats each era can score on. Identical to top_players_comprehensive.py.
ERAS: Dict[str, Tuple[int, int, List[str]]] = {
    "pre_1965": (1897, 1964, ["goals", "behinds"]),
    "1965_1990": (1965, 1990, ["goals", "behinds", "kicks", "handballs"]),
    "1990_2010": (1991, 2010, ["goals", "behinds", "kicks", "handballs", "marks", "disposals"]),
    "post_2010": (
        2011,
        2030,
        [
            "goals",
            "behinds",
            "kicks",
            "handballs",
            "marks",
            "disposals",
            "tackles",
            "one_percenters",
            "clearances",
            "contested_possessions",
            "contested_marks",
            "goal_assist",
        ],
    ),
}

WEIGHTS: Dict[str, float] = {
    "goals": 55.0,
    "behinds": 1.5,
    "disposals": 14.0,
    "goal_assist": 4.0,
    "contested_marks": 7.0,
    "contested_possessions": 5.5,
    "marks": 2.5,
    "kicks": 4.5,
    "tackles": 3.5,
    "one_percenters": 3.0,
    "clearances": 5.5,
}

ALL_STATS: List[str] = sorted({s for _, _, stats in ERAS.values() for s in stats})


def _era_for_year(year: int) -> Tuple[str, List[str]]:
    for era, (start, end, stats) in ERAS.items():
        if start <= year <= end:
            return era, stats
    return "unknown", []


def _player_id_from_filename(path: str) -> str:
    """`bartlett_kevin_06031947_performance_details.csv` → `bartlett_kevin_06031947`."""
    base = os.path.basename(path)
    return base.replace("_performance_details.csv", "")


def _aggregate_one_file(path: str) -> List[Tuple[str, int, Dict[str, float], int]]:
    """Read one player perf CSV and return a list of (player_id, year, totals, games_played)."""
    try:
        # Only load the columns we need for scoring + the year column.
        # The CSVs vary across eras (some columns absent in pre-1965 files), so we
        # use a tolerant read and just take whatever columns are there.
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:  # pragma: no cover
        log.warning("could not read %s: %s", path, e)
        return []

    if df.empty or "year" not in df.columns:
        return []

    # Coerce year to int (some rows have weird strings)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    if df.empty:
        return []

    # Coerce all stat columns we care about to numeric (missing → NaN → 0 at sum time)
    available_stats = [s for s in ALL_STATS if s in df.columns]
    if not available_stats:
        return []
    for s in available_stats:
        df[s] = pd.to_numeric(df[s], errors="coerce").fillna(0)

    pid = _player_id_from_filename(path)
    out: List[Tuple[str, int, Dict[str, float], int]] = []
    grouped = df.groupby("year", sort=False)
    for year, sub in grouped:
        totals = {s: float(sub[s].sum()) for s in available_stats}
        games_played = len(sub)
        out.append((pid, int(year), totals, games_played))
    return out


def aggregate_all_players(data_dir: str) -> List[Tuple[str, int, Dict[str, float], int]]:
    files = sorted(glob.glob(os.path.join(data_dir, "*_performance_details.csv")))
    log.info("Aggregating across %d player perf files", len(files))
    rows: List[Tuple[str, int, Dict[str, float], int]] = []
    t0 = time.time()
    for i, path in enumerate(files, 1):
        rows.extend(_aggregate_one_file(path))
        if i % 1000 == 0:
            log.info("  ... aggregated %d / %d files (%.1fs)", i, len(files), time.time() - t0)
    log.info("Aggregation complete in %.1fs — %d (player, year) rows", time.time() - t0, len(rows))
    return rows


def score_player_year(year: int, totals: Dict[str, float]) -> int:
    """Apply era-appropriate weights to per-year stat totals to produce an integer score."""
    _, era_stats = _era_for_year(year)
    score = 0.0
    for stat in era_stats:
        if stat in totals:
            score += totals[stat] * WEIGHTS.get(stat, 0)
    return int(score)


def build_yearly_top_100(
    aggregates: List[Tuple[str, int, Dict[str, float], int]],
    output_dir: str,
) -> Dict[int, List[Tuple[str, int, float, int]]]:
    """For each year present, write yearly top-100 CSV and return {year: top100 list}."""
    os.makedirs(os.path.join(output_dir, "yearly"), exist_ok=True)

    # Bucket aggregates by year, computing score on the fly.
    by_year: Dict[int, List[Tuple[str, int, int]]] = {}
    for pid, year, totals, games in aggregates:
        score = score_player_year(year, totals)
        by_year.setdefault(year, []).append((pid, score, games))

    yearly_top: Dict[int, List[Tuple[str, int, float, int]]] = {}
    for year, rows in sorted(by_year.items()):
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["player", "score", "games_played"])
        # Percentile rank over ALL players in this year (matches the original algorithm)
        df["percentile_rank"] = df["score"].rank(pct=True) * 100.0
        # Sort: higher score first, then more games
        df = df.sort_values(by=["score", "games_played"], ascending=[False, False]).reset_index(drop=True)
        top = df.head(100).copy()
        top = top[["player", "score", "percentile_rank", "games_played"]]
        out_path = os.path.join(output_dir, "yearly", f"year_{year}.csv")
        top.to_csv(out_path, index=False)
        yearly_top[year] = list(
            zip(top["player"].tolist(), top["score"].tolist(), top["percentile_rank"].tolist(), top["games_played"].tolist())
        )
    log.info("Wrote yearly top-100 for %d years (%d..%d)", len(yearly_top), min(yearly_top), max(yearly_top))
    return yearly_top


def build_all_time_top_100(
    yearly_top: Dict[int, List[Tuple[str, int, float, int]]],
    output_dir: str,
) -> pd.DataFrame:
    """all_time_score = mean(percentile_rank across top-100 appearances) * seasons_in_top_100."""
    rec: Dict[str, List[float]] = {}
    for year, rows in yearly_top.items():
        for player, _score, percentile, _games in rows:
            rec.setdefault(player, []).append(percentile)

    out_rows: List[Tuple[str, float]] = []
    for player, percentiles in rec.items():
        if not percentiles:
            continue
        seasons = len(percentiles)
        avg_pct = float(np.mean(percentiles))
        out_rows.append((player, avg_pct * seasons))

    out_rows.sort(key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(out_rows[:100], columns=["player", "all_time_score"])
    out_path = os.path.join(output_dir, "all_time_top_100.csv")
    df.to_csv(out_path, index=False)
    log.info("Wrote all-time top-100 → %s", out_path)
    return df


def main() -> None:
    started = datetime.now()
    log.info("=== top_players_fast started at %s ===", started.isoformat(timespec="seconds"))
    data_dir = "./data/player_data/"
    output_dir = "./data/top100/"

    aggregates = aggregate_all_players(data_dir)
    yearly_top = build_yearly_top_100(aggregates, output_dir)
    df_all = build_all_time_top_100(yearly_top, output_dir)

    log.info("=== top_players_fast complete in %s ===", datetime.now() - started)
    log.info("Top 5 all-time:\n%s", df_all.head().to_string(index=False))


if __name__ == "__main__":
    main()
