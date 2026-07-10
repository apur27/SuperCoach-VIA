#!/usr/bin/env python3
"""Generate data/player_status.csv — active/retired status for every player.

Scans every data/player_data/*_performance_details.csv, and for each player
emits one row: player_slug, first_name, last_name, dob_key, last_year,
last_game_date, career_games, status.

status == "active"  iff  max(year) == current_season
status == "retired" otherwise

`current_season` is DERIVED from the data — the global maximum `year` across all
player files — never hardcoded. This mirrors the target-year auto-detection used
by supercoach/prediction.py.

The reader only pulls the columns it needs (year, date, games_played) so a full
scan of ~13k files stays well under a minute. Safe to re-run (idempotent).
"""
from __future__ import annotations

import glob
import os
import re
import warnings

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYER_DIR = os.path.join(REPO_ROOT, "data", "player_data")
OUTPUT_CSV = os.path.join(REPO_ROOT, "data", "player_status.csv")

# A currently-active player older than this is almost certainly a bad-DOB
# scrape (a current-season row landing in the wrong player's file), not a real
# record — the historical data maxes out at age 43. We warn rather than filter
# so genuine historical rows are never dropped.
MAX_PLAUSIBLE_ACTIVE_AGE = 45

SCHEMA = ["player_slug", "first_name", "last_name", "dob_key",
          "last_year", "last_game_date", "career_games", "status"]


def _parse_slug(filename: str) -> tuple[str, str, str, str]:
    """(player_slug, first_name, last_name, dob_key) from a performance filename.

    Filenames are `<surname>_<firstname>_<DDMMYYYY>_performance_details.csv`.
    The trailing 8-digit token is the DOB key; the first token is the surname;
    anything between is the given name(s).
    """
    slug = os.path.basename(filename).replace("_performance_details.csv", "")
    parts = slug.split("_")
    dob_key = parts[-1]
    last_name = parts[0] if len(parts) > 1 else slug
    first_name = "_".join(parts[1:-1]) if len(parts) > 2 else ""
    return slug, first_name, last_name, dob_key


def _read_player_info(filepath: str) -> dict | None:
    """Per-player summary read (only year/date/games_played columns)."""
    slug, first_name, last_name, dob_key = _parse_slug(filepath)
    try:
        df = pd.read_csv(filepath, usecols=lambda c: c in ("year", "date", "games_played"))
    except (ValueError, pd.errors.EmptyDataError):
        return None
    if "year" not in df.columns or df.empty:
        return None

    years = pd.to_numeric(df["year"], errors="coerce").dropna()
    if years.empty:
        return None
    last_year = int(years.max())

    # games_played is the afltables career counter; take the max leading-digit
    # value and fall back to the row count when the column is unusable.
    career_games = len(df)
    if "games_played" in df.columns:
        gp = pd.to_numeric(
            df["games_played"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        ).dropna()
        if not gp.empty:
            career_games = int(gp.max())

    last_game_date = ""
    if "date" in df.columns:
        dates = df["date"].dropna().astype(str)
        if not dates.empty:
            last_game_date = dates.max()

    return {
        "player_slug": slug,
        "first_name": first_name,
        "last_name": last_name,
        "dob_key": dob_key,
        "last_year": last_year,
        "last_game_date": last_game_date,
        "career_games": career_games,
    }


def _birth_year(dob_key: str) -> int | None:
    """Birth year from a DDMMYYYY dob key, or None if malformed."""
    if re.fullmatch(r"\d{8}", dob_key or ""):
        return int(dob_key[4:])
    return None


def build_status_table(player_dir: str = PLAYER_DIR) -> pd.DataFrame:
    """Build the player-status frame, sorted by player_slug.

    `current_season` is derived as the global max `last_year`.
    """
    files = sorted(glob.glob(os.path.join(player_dir, "*_performance_details.csv")))
    records = [r for r in (_read_player_info(f) for f in files) if r is not None]

    df = pd.DataFrame(records, columns=[c for c in SCHEMA if c != "status"])
    if df.empty:
        return pd.DataFrame(columns=SCHEMA)

    current_season = int(df["last_year"].max())
    df["status"] = df["last_year"].apply(
        lambda y: "active" if int(y) == current_season else "retired"
    )

    # Soft age guard: flag implausibly old active players for human review.
    for _, row in df[df["status"] == "active"].iterrows():
        by = _birth_year(row["dob_key"])
        if by is not None and (current_season - by) > MAX_PLAUSIBLE_ACTIVE_AGE:
            warnings.warn(
                f"Active player {row['player_slug']} would be age "
                f"{current_season - by} in {current_season} — likely a bad-DOB "
                f"scrape; review this file.",
                stacklevel=2,
            )

    df = df.sort_values("player_slug", kind="stable").reset_index(drop=True)
    return df[SCHEMA]


def main() -> None:
    df = build_status_table()
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    current_season = int(df["last_year"].max()) if not df.empty else "n/a"
    n_active = int((df["status"] == "active").sum())
    print(f"Wrote {OUTPUT_CSV}: {len(df)} players, current_season={current_season}, "
          f"{n_active} active")


if __name__ == "__main__":
    main()
