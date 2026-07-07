"""Deterministic era-boundary threshold scan (Decision 3, 2026-07-07).

Threshold queries like "players with 200+ games AND 20+ disposals/game AND 300+
goals" cross the AFL stat-recording boundary: for players whose career predates
a stat's `recorded_from` year (see config/stat_coverage_eras.yaml), a per-game
rate is only computable over the games where the stat was actually recorded.

The editorial call (docs/pending-decisions.md item 3, RESOLVED 2026-07-07 →
INCLUDE) is to qualify such players on a **dropna** rate — the average over their
*recorded* games only — rather than excluding them or diluting the rate with
fill-zero over unrecorded games. This is a script, not a prompt rule, because
pandas' `skipna=True` default makes the convention silent and fragile in prose.

To keep the INCLUDE rule honest, every qualifying player is emitted with its
**coverage** (`rate_recorded_n of games`) so a reader can see when a rate is
computed over a small fraction of a career — e.g. a player at 22.6 disp/g over
only 50 of 254 games is a partial-coverage figure, not a full-career rate. The
scan also emits the aggregate `N of M` (players qualifying of players considered).

This deliberately does NOT flip to fill-zero for the qualifying decision (that
contradicts DataSentinel.md:82 and the coverage-era memo, which reserve fill-zero
for *reconcilable career headlines*, not cross-player rarity qualification). Both
rates are reported per player so the divergence is always visible.

Reproducibility: deterministic; no RNG. pandas aggregation only.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

_LEADING_DIGITS = re.compile(r"^(\d+)")

# Non-stat identity/fixture columns (mirrors PLAYER_COL_TITLES metadata set).
_META_COLUMNS = {
    "team", "year", "games_played", "opponent", "round", "result",
    "jersey_num", "date", "pid",
}


def canonical_games(group: pd.DataFrame) -> int:
    """Career games for a player = max(row count, max leading-digit of the
    afltables `games_played` career counter). The counter can exceed the row
    count when drawn grand finals collapse or finals rows are missing; the row
    count can exceed the counter when the counter carries sub markers (↓/↑).
    Taking the max is robust to both. See the games_played_gap_detector memo.
    """
    row_count = len(group)
    gp = group["games_played"].dropna().astype(str).str.extract(_LEADING_DIGITS)[0]
    gp = pd.to_numeric(gp, errors="coerce")
    counter = int(gp.max()) if gp.notna().any() else 0
    return max(row_count, counter)


@dataclass
class Qualifier:
    pid: str
    games: int
    rate: float                 # dropna rate over recorded games (qualifying rate)
    rate_fillzero: float        # rate over ALL games (for transparency)
    rate_recorded_n: int        # games where the rate stat was recorded
    count: float                # the count stat (e.g. goals) total

    @property
    def coverage(self) -> str:
        return f"{self.rate_recorded_n} of {self.games}"

    @property
    def partial_coverage(self) -> bool:
        # Flag when the qualifying rate rests on < 90% of career games — the
        # dropna rate then diverges materially from the true full-career rate.
        return self.rate_recorded_n < 0.90 * self.games


@dataclass
class ScanResult:
    qualifiers: list  # list[Qualifier], sorted by rate desc
    n_considered: int

    @property
    def n_qualifying(self) -> int:
        return len(self.qualifiers)

    @property
    def n_of_m(self) -> str:
        return f"{self.n_qualifying} of {self.n_considered}"


def threshold_scan(
    df: pd.DataFrame,
    *,
    min_games: int,
    rate_col: str,
    min_rate: float,
    count_col: str,
    min_count: float,
) -> ScanResult:
    """Run the era-boundary threshold scan over a long player-game DataFrame.

    `df` must carry columns: pid, games_played, and `rate_col`/`count_col`.
    A player qualifies when:
      canonical_games >= min_games
      AND (rate_col summed / count of recorded rate_col games) >= min_rate   # dropna
      AND count_col summed >= min_count
    Players with no recorded `rate_col` game are considered but cannot qualify.
    """
    qualifiers: list = []
    n_considered = 0
    for pid, g in df.groupby("pid", sort=False):
        n_considered += 1
        games = canonical_games(g)
        if games < min_games:
            continue
        count_total = g[count_col].sum()
        if count_total < min_count:
            continue
        recorded = int(g[rate_col].notna().sum())
        if recorded == 0:
            continue
        rate_sum = g[rate_col].sum()
        rate_dropna = rate_sum / recorded
        rate_fillzero = rate_sum / games
        if rate_dropna < min_rate:
            continue
        qualifiers.append(
            Qualifier(
                pid=str(pid),
                games=games,
                rate=round(rate_dropna, 4),
                rate_fillzero=round(rate_fillzero, 4),
                rate_recorded_n=recorded,
                count=float(count_total),
            )
        )
    qualifiers.sort(key=lambda q: q.rate, reverse=True)
    return ScanResult(qualifiers=qualifiers, n_considered=n_considered)


def load_player_games(player_dir: str) -> pd.DataFrame:
    """Load every *_performance_details.csv into one long frame with a `pid`
    column (the filename stem). games_played/round kept as strings.
    """
    files = sorted(glob.glob(os.path.join(player_dir, "*_performance_details.csv")))
    if not files:
        raise FileNotFoundError(f"no performance_details CSVs under {player_dir}")
    parts = []
    for f in files:
        d = pd.read_csv(f, dtype={"games_played": str, "round": str})
        d["pid"] = os.path.basename(f).replace("_performance_details.csv", "")
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _main() -> None:  # pragma: no cover - CLI convenience only
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--player-dir", default="data/player_data")
    ap.add_argument("--min-games", type=int, default=200)
    ap.add_argument("--rate-col", default="disposals")
    ap.add_argument("--min-rate", type=float, default=20.0)
    ap.add_argument("--count-col", default="goals")
    ap.add_argument("--min-count", type=float, default=300.0)
    args = ap.parse_args()

    df = load_player_games(args.player_dir)
    res = threshold_scan(
        df,
        min_games=args.min_games,
        rate_col=args.rate_col,
        min_rate=args.min_rate,
        count_col=args.count_col,
        min_count=args.min_count,
    )
    print(
        f"Threshold: {args.min_games}+ games, {args.min_rate}+ {args.rate_col}/g "
        f"(dropna), {args.min_count}+ {args.count_col}"
    )
    print(f"N of M: {res.n_of_m} qualifying (players considered)")
    for q in res.qualifiers:
        flag = "  [PARTIAL COVERAGE]" if q.partial_coverage else ""
        print(
            f"  {q.pid:28s} games={q.games:3d} {args.rate_col}/g(dropna)={q.rate:5.1f} "
            f"(fill-zero={q.rate_fillzero:5.2f}) coverage={q.coverage:>10s} "
            f"{args.count_col}={q.count:.0f}{flag}"
        )


if __name__ == "__main__":  # pragma: no cover
    _main()
