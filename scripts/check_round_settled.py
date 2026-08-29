#!/usr/bin/env python3
"""Deterministic round-settlement probe (F5, finals-literate per Surveyor F2).

Replaces the day-of-week timing heuristic. Reads the current season's
matches_<year>.csv, finds the current STAGE (the latest round present, with
home-and-away rounds ordered before finals and finals in their real sequence),
and confirms every game present for that stage has a non-zero final score. A
game with 0 goals AND 0 behinds on both sides is treated as unplayed / mid-play
-> the stage is UNSETTLED.

Finals are stored as strings, never integers: 'Qualifying Final' ... 'Grand
Final' in matches_<year>.csv and QF/EF/SF/PF/GF in the player corpus. Both
spellings are accepted. Qualifying and Elimination Finals are the SAME week, so
when week 1 is the current stage EVERY qualifying and elimination final must be
settled. Coercing round_num to a number (the pre-F2 behaviour) dropped every
finals row and pinned the probe to the last H&A round for the whole finals
series -- a run mid-Preliminary-Final passed without inspecting a single game.

Exit 0 = settled, safe to run the weekly cycle. Exit 1 = unsettled (or the file
is missing/unreadable) -> the harness aborts with the offending matchups named.
Fail-closed: an absent or malformed file is treated as unsettled, never as a
silent pass.

--print-last-ha-round is a separate question, for the harness's finals mode:
"what is the last COMPLETED numbered round?" (used to derive the round label and
the backtest upper bound from data rather than from a prediction artifact). It
prints that integer alone and exits 0, or prints nothing and exits 1.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

# Finals sequence. Qualifying and Elimination Finals share week 1, so they carry
# the same order value and are validated together.
_FINALS_ORDER = {
    "qualifying final": 1, "qf": 1,
    "elimination final": 1, "ef": 1,
    "semi final": 2, "sf": 2,
    "preliminary final": 3, "pf": 3,
    "grand final": 4, "gf": 4,
}

REQUIRED_COLUMNS = {
    "round_num", "team_1_team_name", "team_2_team_name",
    "team_1_final_goals", "team_1_final_behinds",
    "team_2_final_goals", "team_2_final_behinds",
}


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def stage_key(label) -> Optional[Tuple[int, int]]:
    """Order one round_num label. H&A -> (0, round); finals -> (1, week).

    Returns None for labels that are neither (junk / unknown), so they can be
    dropped without ever becoming the "current" stage.
    """
    try:
        return (0, int(float(str(label).strip())))
    except (TypeError, ValueError):
        pass
    order = _FINALS_ORDER.get(str(label).strip().lower())
    return (1, order) if order is not None else None


def _load(matches_csv: str) -> Optional[pd.DataFrame]:
    """Read the matches file and attach a `_key` stage-order column.

    Returns None (caller fails closed) when the file is missing, unreadable,
    empty, missing required columns, or has no recognisable round label.
    """
    if not os.path.exists(matches_csv):
        return None
    try:
        df = pd.read_csv(matches_csv)
    except Exception:
        return None
    if df.empty or not REQUIRED_COLUMNS.issubset(df.columns):
        return None

    df = df.copy()
    df["_key"] = df["round_num"].map(stage_key)
    df = df[df["_key"].notna()]
    return None if df.empty else df


def _unplayed(games: pd.DataFrame) -> List[str]:
    """Matchups in `games` with 0 goals AND 0 behinds on BOTH sides."""
    g1 = _num(games["team_1_final_goals"]); b1 = _num(games["team_1_final_behinds"])
    g2 = _num(games["team_2_final_goals"]); b2 = _num(games["team_2_final_behinds"])
    mask = (g1 == 0) & (b1 == 0) & (g2 == 0) & (b2 == 0)
    return [
        f"{r['team_1_team_name']} v {r['team_2_team_name']}"
        for _, r in games[mask].iterrows()
    ]


def current_stage_label(matches_csv: str) -> Optional[str]:
    """Human-readable label for the latest stage present (for messaging)."""
    df = _load(matches_csv)
    if df is None:
        return None
    key = max(df["_key"])
    labels = sorted({str(v) for v in df.loc[df["_key"] == key, "round_num"]})
    return " / ".join(labels)


def check_round_settled(matches_csv: str) -> Tuple[int, List[str]]:
    """Return (exit_code, unsettled_matchups) for the current stage.

    exit_code 0 iff every game present at the latest stage (highest H&A round,
    or the latest finals week if any finals row is present) is scored. Finals
    week 1 covers qualifying AND elimination finals together.
    Missing/unreadable/empty file -> (1, []) (fail-closed).
    """
    df = _load(matches_csv)
    if df is None:
        return 1, []

    key = max(df["_key"])
    unsettled = _unplayed(df[df["_key"] == key])
    return (1 if unsettled else 0), unsettled


def last_settled_ha_round(matches_csv: str) -> Optional[int]:
    """Highest INTEGER home-and-away round present that is fully settled.

    Finals rows are excluded by construction. Returns None when no numbered
    round is complete (or the file is unusable).
    """
    df = _load(matches_csv)
    if df is None:
        return None
    ha = df[df["_key"].map(lambda k: k[0] == 0)]
    if ha.empty:
        return None
    for rnd in sorted({k[1] for k in ha["_key"]}, reverse=True):
        if not _unplayed(ha[ha["_key"] == (0, rnd)]):
            return int(rnd)
    return None


def _resolve_matches_path(year: int | None) -> str:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import config  # noqa: E402

    yr = year or datetime.now().year
    return os.path.join(config.MATCHES_DIR, f"matches_{yr}.csv")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=None, help="season year (default: current)")
    parser.add_argument("--file", default=None, help="explicit matches CSV (overrides --year)")
    parser.add_argument(
        "--print-last-ha-round", action="store_true",
        help="print ONLY the highest fully-settled integer home-and-away round "
             "(nothing + exit 1 if there is none); does not check settlement of "
             "the current stage",
    )
    args = parser.parse_args(argv)

    path = args.file or _resolve_matches_path(args.year)

    if args.print_last_ha_round:
        rnd = last_settled_ha_round(path)
        if rnd is None:
            return 1
        print(rnd)
        return 0

    code, unsettled = check_round_settled(path)

    if code == 0:
        print(f"[round-settled] Current round in {os.path.basename(path)} is settled — proceeding.")
        return 0

    if unsettled:
        # Current stage exists but has unplayed games.
        current = current_stage_label(path) or "?"
        print(
            f"[round-settled] Round {current} has unsettled games: "
            f"{'; '.join(unsettled)}. Run after scores are confirmed."
        )
    else:
        print(f"[round-settled] {os.path.basename(path)} missing/unreadable — cannot confirm settlement. Aborting (fail-closed).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
