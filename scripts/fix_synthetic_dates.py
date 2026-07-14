#!/usr/bin/env python3
"""Repair the corrupt `2025-03-01` synthetic date on round-1 / year-2025 rows.

Background
----------
176 player performance CSVs carry a scraper artifact: a `round=1, year=2025`
game row stamped with the fabricated date `2025-03-01`. That synthetic date
poisons the `days_since_last_game` feature in supercoach/prediction.py -- both
for the bad row and for the player's *next* game (the diff() straddles it).

The real match dates live in data/matches/matches_2025.csv. This script looks
each bad row up by team + opponent and writes back the true date (time
component stripped).

Two corruption classes
----------------------
1. SIMPLE (137 files) -- the `round=1` label is CORRECT. The three real Opening
   Round 2025 games (Sydney-Hawthorn, GWS-Collingwood, Gold Coast-Essendon)
   exist in matches with `round_num == 1`. A direct team+opponent+round==1
   lookup resolves these.

2. COMPOUND (39 files) -- Geelong-vs-Brisbane. matches has NO round_num==1
   Geelong-Brisbane game; the `round=1` label is itself part of the artifact.
   Empirically (verified across all 39 files) the bad row sits chronologically
   between the player's round-3 and round-5 games, so the true game is Round 4
   (Brisbane Lions vs Geelong, 2025-03-29). We resolve this deterministically
   via the chronological neighbour window: among the team+opponent matches, the
   unique one whose numeric round_num falls strictly between the numeric rounds
   of the immediately-preceding and immediately-following 2025 games in the
   player's own (chronologically-ordered) file. This is a *derivation*, not a
   guess -- for every affected file it yields exactly one match.

If a bad row cannot be resolved to exactly one match it is SKIPPED and logged;
nothing is guessed.

Idempotent: once dates are corrected the selector no longer matches, so a
re-run is a no-op.
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYER_DIR = os.path.join(REPO_ROOT, "data", "player_data")
MATCHES_2025 = os.path.join(REPO_ROOT, "data", "matches", "matches_2025.csv")

ARTIFACT_CUTOFF = "2025-03-05"   # any 2025 round-1 date before this is the artifact
TARGET_YEAR = 2025


def _norm_round(x) -> str:
    """Normalise a round label to a bare string: 1 -> '1', '1' -> '1', '16'."""
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():   # 1.0 -> '1' (float coercion)
        s = s[:-2]
    return s


def _pair(a, b) -> frozenset:
    return frozenset({str(a).strip(), str(b).strip()})


def _candidate_matches(matches_df: pd.DataFrame, team: str, opp: str) -> pd.DataFrame:
    """All 2025 matches between `team` and `opp` (order-agnostic)."""
    want = _pair(team, opp)
    yr = pd.to_numeric(matches_df["year"], errors="coerce")
    mask = (yr == TARGET_YEAR) & matches_df.apply(
        lambda m: _pair(m["team_1_team_name"], m["team_2_team_name"]) == want,
        axis=1,
    )
    return matches_df[mask]


def _resolve_by_neighbour_window(player_df: pd.DataFrame, bad_idx,
                                 cands: pd.DataFrame):
    """Pick the candidate match whose numeric round sits between the player's
    neighbouring 2025 games (file order == chronological order).

    Returns the chosen match row (Series) or None if not uniquely resolvable.
    """
    yr = pd.to_numeric(player_df["year"], errors="coerce")
    order = list(player_df.index[yr == TARGET_YEAR])
    if bad_idx not in order:
        return None
    pos = order.index(bad_idx)

    def _num_neighbour(offset, default):
        j = pos + offset
        if 0 <= j < len(order):
            r = _norm_round(player_df.loc[order[j], "round"])
            if r.isdigit():
                return int(r)
        return default

    lo = _num_neighbour(-1, 0)          # preceding game's round (exclusive lower)
    hi = _num_neighbour(+1, 10 ** 6)    # following game's round (exclusive upper)

    def _in_window(r):
        rn = _norm_round(r)
        return rn.isdigit() and lo < int(rn) < hi

    in_win = cands[cands["round_num"].apply(_in_window)]
    if len(in_win) == 1:
        return in_win.iloc[0]
    return None


def fix_synthetic_dates(player_df: pd.DataFrame, matches_df: pd.DataFrame):
    """Correct artifact dates in one player DataFrame.

    Returns (fixed_df, actions) where actions is a list of dicts:
      {index, status: 'fixed'|'skipped', old_date, new_date, resolved_round,
       tier, reason, team, opponent}
    """
    df = player_df.copy()
    date_str = df["date"].astype(str)
    yr = pd.to_numeric(df["year"], errors="coerce")
    is_round1 = df["round"].apply(_norm_round) == "1"
    # Non-empty, non-NaN 2025 date strictly before the cutoff == the artifact.
    # (keep_default_na=False turns missing cells into '', which we must exclude.)
    is_artifact = (date_str.str.strip() != "") & date_str.ne("nan") \
        & (date_str < ARTIFACT_CUTOFF)
    bad_idx = list(df.index[(yr == TARGET_YEAR) & is_round1 & is_artifact])

    actions = []
    for idx in bad_idx:
        team = df.loc[idx, "team"]
        opp = df.loc[idx, "opponent"]
        old_date = str(df.loc[idx, "date"])
        cands = _candidate_matches(matches_df, team, opp)

        chosen = None
        tier = None
        if len(cands) == 0:
            reason = "no team+opponent match in matches_2025"
        else:
            exact = cands[cands["round_num"].apply(_norm_round) == "1"]
            if len(exact) == 1:
                chosen, tier = exact.iloc[0], "exact-round1"
            elif len(exact) > 1:
                reason = f"ambiguous: {len(exact)} round-1 matches"
            else:
                # round label itself corrupt -> chronological neighbour window
                chosen = _resolve_by_neighbour_window(df, idx, cands)
                if chosen is not None:
                    tier = "neighbour-window"
                else:
                    reason = ("no round-1 match and neighbour window did not "
                              "yield a unique candidate")

        if chosen is not None:
            new_date = str(chosen["date"]).split(" ")[0]
            df.at[idx, "date"] = new_date
            actions.append({
                "index": idx, "status": "fixed", "old_date": old_date,
                "new_date": new_date, "resolved_round": chosen["round_num"],
                "tier": tier, "team": team, "opponent": opp,
            })
        else:
            actions.append({
                "index": idx, "status": "skipped", "old_date": old_date,
                "new_date": None, "resolved_round": None, "tier": None,
                "reason": reason, "team": team, "opponent": opp,
            })
    return df, actions


def main():
    matches_df = pd.read_csv(MATCHES_2025)
    files = sorted(glob.glob(os.path.join(PLAYER_DIR, "*performance_details.csv")))

    n_files_scanned = 0
    n_files_fixed = 0
    n_rows_fixed = 0
    tier_counts = {"exact-round1": 0, "neighbour-window": 0}
    skipped = []

    for path in files:
        n_files_scanned += 1
        # cheap pre-filter: only read files that actually contain the artifact
        try:
            with open(path, "r", encoding="utf-8") as fh:
                if "2025-03-01" not in fh.read():
                    continue
        except OSError as exc:
            print(f"WARN: could not read {os.path.basename(path)}: {exc}",
                  file=sys.stderr)
            continue

        # Read every cell as a verbatim string (keep_default_na=False) so the
        # round-trip does NOT reformat untouched cells -- e.g. pandas would
        # otherwise coerce a bare-int '10' to '10.0' on other rows. Only the
        # single target date cell must change.
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        fixed, actions = fix_synthetic_dates(df, matches_df)
        file_fixes = [a for a in actions if a["status"] == "fixed"]
        file_skips = [a for a in actions if a["status"] == "skipped"]

        for a in file_skips:
            skipped.append((os.path.basename(path), a))

        if file_fixes:
            fixed.to_csv(path, index=False)
            n_files_fixed += 1
            n_rows_fixed += len(file_fixes)
            for a in file_fixes:
                tier_counts[a["tier"]] = tier_counts.get(a["tier"], 0) + 1

    print("=" * 68)
    print("fix_synthetic_dates summary")
    print("=" * 68)
    print(f"files scanned (containing artifact): "
          f"{n_files_fixed + len({s[0] for s in skipped})}")
    print(f"files fixed          : {n_files_fixed}")
    print(f"rows fixed           : {n_rows_fixed}")
    print(f"  via exact round-1  : {tier_counts.get('exact-round1', 0)}")
    print(f"  via neighbour-window (Geelong-Brisbane R1->R4): "
          f"{tier_counts.get('neighbour-window', 0)}")
    print(f"rows skipped         : {len(skipped)}")
    for fname, a in skipped:
        print(f"  SKIP {fname}: {a['team']} vs {a['opponent']} "
              f"-- {a.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
