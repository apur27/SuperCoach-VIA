"""
backtest.py
-----------

Walk-forward backtest of the AFLDisposalPredictor.

For every round N of year Y in the requested window, the script:

    1. Strips every row from the player-data DataFrame that occurs at or after
       (year=Y, round=N) — i.e. all rows from year>Y, plus same-year rows with
       round >= N. This enforces a strict temporal cutoff: the model trains
       and engineers features using only games played BEFORE round N of year Y.
    2. Runs `AFLDisposalPredictor` with target_year=Y.
    3. Extracts predictions for round N of year Y.
    4. Joins predictions to the actual disposals recorded for round N of year Y.
    5. Writes a per-round prediction_vs_actual CSV plus rich diagnostic logs.

Aggregate outputs:
    - backtest_summary_{ts}.csv         (one row per round)
    - backtest_by_position_{ts}.csv     (per round x position)
    - backtest_by_team_{ts}.csv         (per round x team)
    - backtest_run_{ts}.log             (full diagnostic log)

NOTE on "position": the SuperCoach-VIA dataset does not currently store a
position column. This script populates `position` with the literal string
"Unknown" so the schema is stable; the by-position aggregate will therefore
contain a single "Unknown" bucket until a position source is wired up.

CLI:
    python backtest.py [--start-year 2025] [--start-round 22] [--end-year 2026]
                       [--end-round auto] [--data-dir ./data/player_data/]

Defaults match the SuperCoach-VIA convention: start at round 22 of 2025
(games played around 2025-08-26) through to the last played round of 2026
(auto-detected from the data).
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the production predictor end-to-end.
from prediction import AFLDisposalPredictor, extract_round_number


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path("/home/abhi/git/SuperCoach-VIA/data/player_data")
BACKTEST_DIR = Path("/home/abhi/git/SuperCoach-VIA/data/prediction/backtest")

EXACT_TOL = 1.0  # |pred - actual| <= 1 is treated as "exact" for over_under


# ---------------------------------------------------------------------------
# Helpers: round / season discovery
# ---------------------------------------------------------------------------


def _discover_max_played_round(
    data_dir: Path, year: int, played_majority_threshold: float = 0.5
) -> int | None:
    """Return the largest round_number in `year` for which a majority of the
    rows recorded in player CSVs have a non-null `disposals` value.

    A round is only "played" for backtesting purposes once most players who
    have a row for that round actually have disposal counts recorded — partial
    rounds (in-progress weekends, or rows that exist only as fixtures) are
    excluded. This is intentionally conservative; we'd rather skip a partial
    round than score against half-empty actuals.
    """
    counts: dict[int, dict[str, int]] = {}
    for fp in data_dir.glob("*_performance_details.csv"):
        try:
            df = pd.read_csv(fp, usecols=["year", "round", "disposals"])
        except Exception:
            continue
        df = df[df["year"] == year]
        if df.empty:
            continue
        df = df.assign(
            rn=df["round"].apply(extract_round_number),
            has_disp=df["disposals"].notnull(),
        )
        df = df.dropna(subset=["rn"])
        for rn, g in df.groupby("rn"):
            rn_int = int(rn)
            slot = counts.setdefault(rn_int, {"rows": 0, "played": 0})
            slot["rows"] += len(g)
            slot["played"] += int(g["has_disp"].sum())

    played_rounds = [
        rn
        for rn, c in counts.items()
        if c["rows"] > 0 and c["played"] / c["rows"] >= played_majority_threshold
    ]
    if not played_rounds:
        return None
    return max(played_rounds)


def _enumerate_rounds(
    start_year: int, start_round: int, end_year: int, end_round: int, data_dir: Path
) -> list[tuple[int, int]]:
    """Enumerate (year, round) pairs in chronological order across the window.

    For each year between `start_year` and `end_year` we use the year's
    discovered max played round as the upper bound (capped at `end_round` for
    the final year). Years strictly between start and end are walked from
    round 1 to their max played round.
    """
    pairs: list[tuple[int, int]] = []
    for y in range(start_year, end_year + 1):
        y_max = _discover_max_played_round(data_dir, y)
        if y_max is None:
            continue
        lo = start_round if y == start_year else 1
        hi = end_round if y == end_year else y_max
        hi = min(hi, y_max)
        for r in range(lo, hi + 1):
            pairs.append((y, r))
    return pairs


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------


@dataclass
class RoundResult:
    year: int
    round: int
    n_predicted: int
    n_with_actual: int
    mae: float | None
    rmse: float | None
    median_abs_error: float | None
    bias: float | None
    pct_within_5: float | None
    pct_within_10: float | None
    top10_actual_mae: float | None
    bottom10_actual_mae: float | None
    detail_path: Path


class LeakProofPredictor(AFLDisposalPredictor):
    """AFLDisposalPredictor that enforces a hard temporal cutoff at
    (target_year, cutoff_round): every row at or after that point is
    discarded immediately after load, before any feature engineering or
    model training touches it.

    This is the single defensive choke point that prevents the backtest from
    leaking the round being scored (or any future round) into the training
    set or the prediction's lagged features.
    """

    def __init__(self, *args, cutoff_round: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.cutoff_round = int(cutoff_round)

    def load_and_prepare_data(self) -> pd.DataFrame:
        df = super().load_and_prepare_data()
        if df.empty:
            return df

        rn = pd.to_numeric(df["round"].apply(extract_round_number), errors="coerce")
        keep_mask = (
            (df["year"] < self.target_year)
            | ((df["year"] == self.target_year) & (rn < self.cutoff_round))
            # Keep the cutoff round itself so we have a row to predict against.
            | ((df["year"] == self.target_year) & (rn == self.cutoff_round))
        )
        # Drop strictly-future material:
        # - any year > target_year
        # - target_year rounds > cutoff_round
        future_mask = (df["year"] > self.target_year) | (
            (df["year"] == self.target_year) & (rn > self.cutoff_round)
        )
        rows_in = len(df)
        df = df[~future_mask].copy()
        rows_out = len(df)

        # Same filter applied to per-player cache so the prediction loop, which
        # iterates over self._player_cache directly, sees the same temporal
        # truncation.
        for fp, pdf in list(self._player_cache.items()):
            prn = pd.to_numeric(pdf["round"].apply(extract_round_number), errors="coerce")
            fmask = (pdf["year"] > self.target_year) | (
                (pdf["year"] == self.target_year) & (prn > self.cutoff_round)
            )
            self._player_cache[fp] = pdf[~fmask].copy()

        logging.getLogger("backtest").info(
            "[cutoff y=%d r=%d] dropped %d future rows (%d -> %d)",
            self.target_year,
            self.cutoff_round,
            rows_in - rows_out,
            rows_in,
            rows_out,
        )
        return df


def _gather_actuals(data_dir: Path, year: int, round_num: int) -> pd.DataFrame:
    """Return DataFrame with [player, team, round, year, actual_disposals] for
    the requested (year, round). Players are matched by file slug (the same
    title-cased convention `extract_dob_and_name` produces in prediction.py).
    """
    actuals = []
    for fp in data_dir.glob("*_performance_details.csv"):
        try:
            df = pd.read_csv(fp, usecols=["team", "year", "round", "disposals"])
        except Exception:
            continue
        df = df[df["year"] == year]
        if df.empty:
            continue
        df["rn"] = df["round"].apply(extract_round_number)
        sub = df[df["rn"] == round_num]
        if sub.empty:
            continue
        # Same name normalisation as prediction.py (drops _DDMMYYYY suffix and
        # title-cases the rest).
        parts = fp.stem.split("_")
        player_name = " ".join(parts[:-3]).title() if len(parts) >= 3 else fp.stem
        for _, row in sub.iterrows():
            actuals.append(
                {
                    "player": player_name,
                    "team": row["team"],
                    "round": round_num,
                    "year": year,
                    "actual_disposals": row["disposals"],
                }
            )
    return pd.DataFrame(actuals)


def _classify_over_under(pred: float, actual: float) -> str:
    if pd.isna(actual) or pd.isna(pred):
        return ""
    diff = pred - actual
    if abs(diff) <= EXACT_TOL:
        return "exact"
    return "over" if diff > 0 else "under"


def _round_metrics(detail: pd.DataFrame) -> dict:
    """Compute per-round error metrics from the prediction-vs-actual frame.

    Returns NaN-valued metrics when no players in the round had an actual
    recorded — that's a real signal (full DNP set) and should propagate, not
    crash.
    """
    scored = detail[detail["actual_disposals"].notnull()].copy()
    if scored.empty:
        return {
            "n_with_actual": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "median_abs_error": np.nan,
            "bias": np.nan,
            "pct_within_5": np.nan,
            "pct_within_10": np.nan,
            "top10_actual_mae": np.nan,
            "bottom10_actual_mae": np.nan,
        }
    err = scored["predicted_disposals"] - scored["actual_disposals"]
    abs_err = err.abs()
    top10 = scored.nlargest(min(10, len(scored)), "actual_disposals")
    bot10 = scored.nsmallest(min(10, len(scored)), "actual_disposals")
    return {
        "n_with_actual": int(len(scored)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "median_abs_error": float(abs_err.median()),
        "bias": float(err.mean()),
        "pct_within_5": float((abs_err <= 5).mean() * 100),
        "pct_within_10": float((abs_err <= 10).mean() * 100),
        "top10_actual_mae": float(
            (top10["predicted_disposals"] - top10["actual_disposals"]).abs().mean()
        ),
        "bottom10_actual_mae": float(
            (bot10["predicted_disposals"] - bot10["actual_disposals"]).abs().mean()
        ),
    }


def _slice_metrics(detail: pd.DataFrame, group_col: str, year: int, round_num: int) -> pd.DataFrame:
    """Return per-group metrics (one row per group within the round)."""
    scored = detail[detail["actual_disposals"].notnull()].copy()
    rows: list[dict] = []
    if scored.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "round",
                group_col,
                "n",
                "mae",
                "rmse",
                "bias",
                "median_abs_error",
                "pct_within_5",
                "pct_within_10",
            ]
        )
    for grp_val, g in scored.groupby(group_col):
        err = g["predicted_disposals"] - g["actual_disposals"]
        abs_err = err.abs()
        rows.append(
            {
                "year": year,
                "round": round_num,
                group_col: grp_val,
                "n": int(len(g)),
                "mae": float(abs_err.mean()),
                "rmse": float(np.sqrt((err**2).mean())),
                "bias": float(err.mean()),
                "median_abs_error": float(abs_err.median()),
                "pct_within_5": float((abs_err <= 5).mean() * 100),
                "pct_within_10": float((abs_err <= 10).mean() * 100),
            }
        )
    return pd.DataFrame(rows)


def run_round_backtest(
    *,
    year: int,
    round_num: int,
    data_dir: Path,
    timestamp: str,
    log: logging.Logger,
) -> tuple[RoundResult, pd.DataFrame]:
    """Run the leak-proof predictor for one (year, round_num), persist the
    detail CSV, and return (RoundResult, detail_df).
    """
    log.info("=" * 78)
    log.info("BACKTESTING round=%d year=%d", round_num, year)
    log.info("=" * 78)

    # The predictor prints a lot of progress to stdout; we let those show
    # because they're useful "I'm alive" signals during a multi-hour run.
    predictor = LeakProofPredictor(
        data_dir=str(data_dir),
        target_year=year,
        cutoff_round=round_num,
    )
    predictor.run()

    pred_path = sorted(
        Path("/home/abhi/git/SuperCoach-VIA/data/prediction").glob(
            f"next_round_*_prediction_*.csv"
        ),
        key=lambda p: p.stat().st_mtime,
    )[-1]
    log.info("predictor wrote %s", pred_path.name)
    preds = pd.read_csv(pred_path)

    # The predictor's saved CSV has [player, team, predicted_disposals] for
    # the next-round per-player slice. That "next round" should equal
    # round_num because we capped the data at round_num.
    actuals = _gather_actuals(data_dir, year, round_num)
    log.info(
        "predictions=%d  actuals=%d  for r=%d y=%d",
        len(preds),
        len(actuals),
        round_num,
        year,
    )

    detail = preds.merge(actuals, on=["player", "team"], how="left")
    detail["round"] = round_num
    detail["year"] = year
    detail["position"] = "Unknown"  # No position source; documented in module docstring.

    detail["error"] = detail["predicted_disposals"] - detail["actual_disposals"]
    detail["abs_error"] = detail["error"].abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        detail["pct_error"] = np.where(
            detail["actual_disposals"].fillna(0) > 0,
            detail["error"] / detail["actual_disposals"] * 100,
            np.nan,
        )
    detail["over_under"] = [
        _classify_over_under(p, a)
        for p, a in zip(detail["predicted_disposals"], detail["actual_disposals"])
    ]

    detail_cols = [
        "player",
        "team",
        "position",
        "round",
        "year",
        "predicted_disposals",
        "actual_disposals",
        "error",
        "abs_error",
        "pct_error",
        "over_under",
    ]
    detail = detail[detail_cols].copy()

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = BACKTEST_DIR / (
        f"prediction_vs_actual_round_{round_num}_{year}_{timestamp}.csv"
    )
    detail.to_csv(detail_path, index=False)
    log.info("wrote %s (%d rows)", detail_path.name, len(detail))

    # Diagnostics
    n_dnp = int(detail["actual_disposals"].isna().sum())
    if n_dnp:
        log.info(
            "%d players had a prediction but no actual disposals (DNP / fixture row); excluded from error metrics",
            n_dnp,
        )

    metrics = _round_metrics(detail)
    log.info(
        "round=%d year=%d  n_scored=%d  MAE=%.3f  RMSE=%.3f  bias=%+.3f  pct<=5=%.1f%%  pct<=10=%.1f%%",
        round_num,
        year,
        metrics["n_with_actual"],
        metrics["mae"] if not np.isnan(metrics["mae"]) else float("nan"),
        metrics["rmse"] if not np.isnan(metrics["rmse"]) else float("nan"),
        metrics["bias"] if not np.isnan(metrics["bias"]) else float("nan"),
        metrics["pct_within_5"] if not np.isnan(metrics["pct_within_5"]) else float("nan"),
        metrics["pct_within_10"] if not np.isnan(metrics["pct_within_10"]) else float("nan"),
    )

    scored = detail[detail["actual_disposals"].notnull()].copy()
    if not scored.empty:
        over = scored.nlargest(min(5, len(scored)), "error")
        under = scored.nsmallest(min(5, len(scored)), "error")
        log.info("top 5 over-predicted (model thought too high):")
        for _, r in over.iterrows():
            log.info(
                "  %-30s pred=%5.1f actual=%5.1f err=%+5.1f",
                r["player"],
                r["predicted_disposals"],
                r["actual_disposals"],
                r["error"],
            )
        log.info("top 5 under-predicted (model thought too low):")
        for _, r in under.iterrows():
            log.info(
                "  %-30s pred=%5.1f actual=%5.1f err=%+5.1f",
                r["player"],
                r["predicted_disposals"],
                r["actual_disposals"],
                r["error"],
            )

    result = RoundResult(
        year=year,
        round=round_num,
        n_predicted=len(detail),
        detail_path=detail_path,
        **metrics,
    )
    return result, detail


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("backtest")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    log.propagate = False
    return log


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Walk-forward backtest of the AFLDisposalPredictor. For each round "
            "in the window, trains on data strictly before the round and scores "
            "predictions against actual disposals."
        )
    )
    p.add_argument("--start-year", type=int, default=2025)
    p.add_argument("--start-round", type=int, default=22)
    p.add_argument("--end-year", type=int, default=2026)
    p.add_argument(
        "--end-round",
        default="auto",
        help='Last round to backtest in --end-year (use "auto" to pick the last played round).',
    )
    p.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    warnings.filterwarnings("ignore")
    args = _parse_args(argv)
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"data-dir does not exist: {data_dir}", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    log_path = BACKTEST_DIR / f"backtest_run_{timestamp}.log"
    log = _setup_logging(log_path)

    if str(args.end_round).lower() == "auto":
        end_round = _discover_max_played_round(data_dir, args.end_year)
        if end_round is None:
            log.error("could not auto-detect end round for year %d", args.end_year)
            return 3
    else:
        end_round = int(args.end_round)

    pairs = _enumerate_rounds(
        args.start_year, args.start_round, args.end_year, end_round, data_dir
    )
    log.info(
        "BACKTEST WINDOW: %d-R%d -> %d-R%d  (%d rounds)",
        args.start_year,
        args.start_round,
        args.end_year,
        end_round,
        len(pairs),
    )
    log.info("data-dir: %s", data_dir)
    log.info("timestamp: %s", timestamp)
    log.info("note: position column is populated as 'Unknown' (no source dataset)")

    if not pairs:
        log.error("no (year, round) pairs in window — nothing to do")
        return 0

    summary_rows: list[dict] = []
    by_position_frames: list[pd.DataFrame] = []
    by_team_frames: list[pd.DataFrame] = []
    all_details: list[pd.DataFrame] = []

    for year, round_num in pairs:
        try:
            result, detail = run_round_backtest(
                year=year,
                round_num=round_num,
                data_dir=data_dir,
                timestamp=timestamp,
                log=log,
            )
        except Exception as e:
            log.exception("round=%d year=%d failed: %s", round_num, year, e)
            continue

        summary_rows.append(
            {
                "round": result.round,
                "year": result.year,
                "n_players": result.n_with_actual,
                "mae": result.mae,
                "rmse": result.rmse,
                "median_abs_error": result.median_abs_error,
                "bias": result.bias,
                "pct_within_5": result.pct_within_5,
                "pct_within_10": result.pct_within_10,
                "top_10_mae": result.top10_actual_mae,
                "bottom_10_mae": result.bottom10_actual_mae,
            }
        )
        by_position_frames.append(_slice_metrics(detail, "position", year, round_num))
        by_team_frames.append(_slice_metrics(detail, "team", year, round_num))
        all_details.append(detail)

    if not summary_rows:
        log.error("every round failed — see error logs above")
        return 4

    summary = pd.DataFrame(summary_rows)
    summary_path = BACKTEST_DIR / f"backtest_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)
    log.info("wrote %s", summary_path.name)

    by_pos = pd.concat(by_position_frames, ignore_index=True) if by_position_frames else pd.DataFrame()
    by_team = pd.concat(by_team_frames, ignore_index=True) if by_team_frames else pd.DataFrame()
    by_pos_path = BACKTEST_DIR / f"backtest_by_position_{timestamp}.csv"
    by_team_path = BACKTEST_DIR / f"backtest_by_team_{timestamp}.csv"
    by_pos.to_csv(by_pos_path, index=False)
    by_team.to_csv(by_team_path, index=False)
    log.info("wrote %s and %s", by_pos_path.name, by_team_path.name)

    # ----- Cumulative diagnostics -----
    log.info("=" * 78)
    log.info("CUMULATIVE BACKTEST SUMMARY")
    log.info("=" * 78)
    median_mae = float(np.nanmedian(summary["mae"]))
    log.info(
        "rounds=%d  cumulative MAE (mean of round MAE)=%.3f  median round MAE=%.3f",
        len(summary),
        float(np.nanmean(summary["mae"])),
        median_mae,
    )

    # All players combined → true overall MAE/RMSE.
    full = pd.concat(all_details, ignore_index=True)
    full_scored = full[full["actual_disposals"].notnull()]
    if not full_scored.empty:
        e = full_scored["predicted_disposals"] - full_scored["actual_disposals"]
        log.info(
            "overall (all rounds, all players): n=%d  MAE=%.3f  RMSE=%.3f  bias=%+.3f",
            len(full_scored),
            float(e.abs().mean()),
            float(np.sqrt((e**2).mean())),
            float(e.mean()),
        )

    # Worst rounds (>1.5x median round MAE)
    worst = summary[summary["mae"] > 1.5 * median_mae].sort_values("mae", ascending=False)
    if not worst.empty:
        log.info("rounds where the model performed >1.5x worse than median MAE:")
        for _, r in worst.iterrows():
            log.info(
                "  y=%d r=%d  MAE=%.3f  bias=%+.3f  n=%d",
                int(r["year"]),
                int(r["round"]),
                r["mae"],
                r["bias"],
                int(r["n_players"]),
            )
    else:
        log.info("no rounds exceeded 1.5x the median MAE")

    # Position-group bias summary (currently single bucket — see docstring).
    if not by_pos.empty:
        agg = by_pos.groupby("position").agg(
            n=("n", "sum"), mae=("mae", "mean"), bias=("bias", "mean")
        )
        log.info("position-group bias summary (mean of per-round bias):")
        for pos, row in agg.iterrows():
            sign = "under-predicts" if row["bias"] < 0 else "over-predicts"
            log.info(
                "  %-12s n=%d  mae=%.3f  model %s by avg %.2f disposals",
                pos,
                int(row["n"]),
                row["mae"],
                sign,
                abs(row["bias"]),
            )

    # Team-level bias summary
    if not by_team.empty:
        agg = (
            by_team.groupby("team")
            .agg(n=("n", "sum"), mae=("mae", "mean"), bias=("bias", "mean"))
            .sort_values("bias")
        )
        log.info("team-level bias summary (sorted by bias, most under-predicted first):")
        for tm, row in agg.iterrows():
            sign = "under" if row["bias"] < 0 else "over"
            log.info(
                "  %-25s n=%d  mae=%.3f  bias=%+.2f (%s)",
                tm,
                int(row["n"]),
                row["mae"],
                row["bias"],
                sign,
            )

    log.info("DONE. log: %s", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
