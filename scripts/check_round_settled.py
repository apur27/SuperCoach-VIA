#!/usr/bin/env python3
"""Deterministic round-settlement probe (F5).

Replaces the day-of-week timing heuristic. Reads the current season's
matches_<year>.csv, finds the current home-and-away round (the highest integer
round_num present), and confirms every game present for that round has a
non-zero final score. A game with 0 goals AND 0 behinds on both sides is treated
as unplayed / mid-play -> the round is UNSETTLED.

Exit 0 = settled, safe to run the weekly cycle. Exit 1 = unsettled (or the file
is missing/unreadable) -> the harness aborts with the offending matchups named.
Fail-closed: an absent or malformed file is treated as unsettled, never as a
silent pass.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import List, Tuple

import pandas as pd


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def check_round_settled(matches_csv: str) -> Tuple[int, List[str]]:
    """Return (exit_code, unsettled_matchups) for the current H&A round.

    exit_code 0 iff the highest integer round in the file has every present game
    scored (any side non-zero on goals or behinds). Missing/unreadable/empty
    file -> (1, []) (fail-closed).
    """
    if not os.path.exists(matches_csv):
        return 1, []
    try:
        df = pd.read_csv(matches_csv)
    except Exception:
        return 1, []

    required = {
        "round_num", "team_1_team_name", "team_2_team_name",
        "team_1_final_goals", "team_1_final_behinds",
        "team_2_final_goals", "team_2_final_behinds",
    }
    if df.empty or not required.issubset(df.columns):
        return 1, []

    rn = pd.to_numeric(df["round_num"], errors="coerce")
    ha = df[rn.notna()].copy()
    if ha.empty:
        return 1, []
    ha["_rn"] = rn[rn.notna()].astype(int)

    current = int(ha["_rn"].max())
    games = ha[ha["_rn"] == current]

    g1 = _num(games["team_1_final_goals"]); b1 = _num(games["team_1_final_behinds"])
    g2 = _num(games["team_2_final_goals"]); b2 = _num(games["team_2_final_behinds"])
    unplayed_mask = (g1 == 0) & (b1 == 0) & (g2 == 0) & (b2 == 0)

    unsettled: List[str] = [
        f"{r['team_1_team_name']} v {r['team_2_team_name']}"
        for _, r in games[unplayed_mask].iterrows()
    ]
    return (1 if unsettled else 0), unsettled


def _resolve_matches_path(year: int | None) -> str:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import config  # noqa: E402

    yr = year or datetime.now().year
    return os.path.join(config.MATCHES_DIR, f"matches_{yr}.csv")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=None, help="season year (default: current)")
    parser.add_argument("--file", default=None, help="explicit matches CSV (overrides --year)")
    args = parser.parse_args(argv)

    path = args.file or _resolve_matches_path(args.year)
    code, unsettled = check_round_settled(path)

    if code == 0:
        print(f"[round-settled] Current round in {os.path.basename(path)} is settled — proceeding.")
        return 0

    if unsettled:
        # Current round exists but has unplayed games.
        try:
            df = pd.read_csv(path)
            rn = pd.to_numeric(df["round_num"], errors="coerce")
            current = int(rn.dropna().astype(int).max())
        except Exception:
            current = "?"
        print(
            f"[round-settled] Round {current} has unsettled games: "
            f"{'; '.join(unsettled)}. Run after scores are confirmed."
        )
    else:
        print(f"[round-settled] {os.path.basename(path)} missing/unreadable — cannot confirm settlement. Aborting (fail-closed).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
