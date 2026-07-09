"""Repair writer for the conceded-stats file.

Background
----------
`data/conceded_stats/team_stats_conceded_2025.csv` was committed as raw data
with no generator in the repo. Its values are complete and real (goals/behinds
match the official scores in data/matches/matches_2025.csv exactly), but three
columns are mislabelled: the buggy producer emitted a 3-way value rotation
among disposals / handballs / marks. The fingerprint is that the physical
identity `disposals = kicks + handballs` does NOT hold, while the corrupt
identity `marks == kicks + disposals` holds on every row.

Corrupt layout (header -> value it actually holds):
    disposals_conceded  <- handballs
    kicks_conceded      <- kicks       (correct)
    handballs_conceded  <- marks
    marks_conceded      <- disposals
    (goals/behinds/tackles/hitouts/inside_50s/clearances untouched)

`fix_conceded_columns` undoes the rotation. It is idempotent: if the correct
invariant already holds it is a no-op, so re-running the writer is safe.

The full-team totals cannot be regenerated from data/player_data/ because the
2025 per-player files are only ~55% complete (mean ~12 of 22 players per team
per match). The measured values in the existing file are therefore the
authoritative source; this writer repairs their column mapping rather than
recomputing them.

Usage:
    python scripts/build_conceded_stats.py
"""

import os
import sys

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONCEDED_FILE = os.path.join(
    REPO, "data", "conceded_stats", "team_stats_conceded_2025.csv"
)

EXPECTED_COLUMNS = [
    "year", "round", "team", "opponent",
    "disposals_conceded", "kicks_conceded", "handballs_conceded",
    "marks_conceded", "goals_conceded", "behinds_conceded",
    "tackles_conceded", "hitouts_conceded", "inside_50s_conceded",
    "clearances_conceded",
]


def _has_correct_invariant(df: pd.DataFrame) -> pd.Series:
    return df["disposals_conceded"] == df["kicks_conceded"] + df["handballs_conceded"]


def _has_corrupt_signature(df: pd.DataFrame) -> pd.Series:
    return df["marks_conceded"] == df["kicks_conceded"] + df["disposals_conceded"]


def fix_conceded_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Undo the disposals/handballs/marks value rotation, per row.

    Idempotent: rows already satisfying disposals = kicks + handballs are left
    unchanged. Rows bearing the corrupt signature are rotated back:
        disposals <- old marks
        handballs <- old disposals
        marks     <- old handballs

    Raises if a row is neither already-correct nor corrupt-signature, so silent
    partial corruption cannot slip through.
    """
    out = df.copy()
    correct = _has_correct_invariant(out)
    corrupt = _has_corrupt_signature(out)
    unexplained = ~correct & ~corrupt
    if unexplained.any():
        raise ValueError(
            f"{int(unexplained.sum())} row(s) match neither the correct nor the "
            "corrupt disposal invariant; refusing to guess. Rows: "
            f"{out.index[unexplained].tolist()}"
        )

    to_fix = corrupt & ~correct
    old_disp = out.loc[to_fix, "disposals_conceded"].copy()
    old_hb = out.loc[to_fix, "handballs_conceded"].copy()
    old_marks = out.loc[to_fix, "marks_conceded"].copy()
    out.loc[to_fix, "disposals_conceded"] = old_marks
    out.loc[to_fix, "handballs_conceded"] = old_disp
    out.loc[to_fix, "marks_conceded"] = old_hb

    return out[EXPECTED_COLUMNS]


def main(path: str = CONCEDED_FILE) -> int:
    df = pd.read_csv(path)
    n_corrupt = int((_has_corrupt_signature(df) & ~_has_correct_invariant(df)).sum())
    fixed = fix_conceded_columns(df)

    # Post-condition: the physical invariant must now hold everywhere.
    inv = _has_correct_invariant(fixed)
    if not inv.all():
        raise AssertionError(
            f"invariant still violated on {int((~inv).sum())} rows after fix"
        )

    fixed.to_csv(path, index=False)
    print(f"Wrote {path}: {len(fixed)} rows, repaired {n_corrupt} corrupt rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
