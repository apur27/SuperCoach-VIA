"""
Era-based statistical analysis of AFL.

Improvements over the original:
- Buckets player-game rows by MATCH year (the year the game was played),
  not by debut year. The original mixed every game of a player's career
  into the era of their debut, which corrupts every comparison.
- Uses canonical era cuts:
    pre-1965        (goals/behinds only — no disposals/marks/etc tracked)
    1965-1990       (kicks/handballs/marks tracked, but contested poss / clearances etc not until later)
    1991-2010       (full possession & pressure stats become reliable)
    2011-present    (full modern stat suite, GPS-era pace)
- Normalises per-game (mean per player-game) AND per-100%-time-played
  using the percentage_of_game_played column where available.
- Aggregates the matches/ files for team-level pace-of-play metrics:
  goals, behinds, total points, scoring shots, goal accuracy per team-game.
- Runs Welch's t-test between adjacent eras for each metric (with caveats
  about non-independence — see report).
- Outputs:
    data/era_stats.csv                  per-era player metric means + stds + n
    data/era_team_scoring.csv           per-era team scoring (matches data)
    data/era_significance_tests.csv     pairwise Welch t-tests between adjacent eras
    data/era_yearly_trends.csv          yearly mean for each metric (for plotting)
- Prints a tight human-readable summary at the end.

Honesty notes:
- Stats coverage is uneven. tackles/clearances/contested_possessions are
  effectively zero before ~1965-1970 because the AFL didn't track them.
  The script reports n>0 counts so readers can see when a stat became
  reliable.
- Welch t-tests across player-game rows treat each row as independent.
  They are not strictly independent (same player appears many times,
  same match appears 22+ times). Effect sizes (Cohen's d) and the raw
  delta are reported alongside p-values; the p-values should be read as
  "rough indicator", not gospel.
- 2026 is partial (rounds 1-8 only as of analysis date).
"""

import csv
import glob
import json
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

warnings.simplefilter("ignore", category=RuntimeWarning)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

REPO_ROOT = config.REPO_ROOT
PLAYER_DATA_DIR = config.PLAYER_DATA_DIR
MATCHES_DIR = config.MATCHES_DIR
OUT_ERA_STATS = os.path.join(config.DATA_DIR, "era_stats.csv")
OUT_TEAM_SCORING = os.path.join(config.DATA_DIR, "era_team_scoring.csv")
OUT_SIG_TESTS = os.path.join(config.DATA_DIR, "era_significance_tests.csv")
OUT_YEARLY = os.path.join(config.DATA_DIR, "era_yearly_trends.csv")
OUT_SUMMARY_JSON = os.path.join(config.DATA_DIR, "era_summary.json")

ERA_BOUNDS = [
    ("pre-1965", 1897, 1964),
    ("1965-1990", 1965, 1990),
    ("1991-2010", 1991, 2010),
    ("2011-present", 2011, 2026),
]
ERA_ORDER = [name for (name, _, _) in ERA_BOUNDS]

PLAYER_METRICS = [
    "kicks",
    "marks",
    "handballs",
    "disposals",
    "goals",
    "behinds",
    "hit_outs",
    "tackles",
    "rebound_50s",
    "inside_50s",
    "clearances",
    "clangers",
    "free_kicks_for",
    "free_kicks_against",
    "brownlow_votes",
    "contested_possessions",
    "uncontested_possessions",
    "contested_marks",
    "marks_inside_50",
    "one_percenters",
    "bounces",
    "goal_assist",
    "percentage_of_game_played",
]


def assign_era(year: int) -> str:
    for name, lo, hi in ERA_BOUNDS:
        if lo <= year <= hi:
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# 1. Player-game level analysis
# ---------------------------------------------------------------------------
def load_player_games() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(PLAYER_DATA_DIR, "*_performance_details.csv")))
    print(f"[1/4] Loading {len(files)} player performance files...")

    use_cols = ["team", "year"] + PLAYER_METRICS
    frames = []
    bad = 0
    for i, path in enumerate(files):
        try:
            df = pd.read_csv(path, usecols=lambda c: c in use_cols, low_memory=False)
            if df.empty or "year" not in df.columns:
                continue
            # Coerce. Missing stats become NaN, NOT zero, so they don't deflate means.
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df.dropna(subset=["year"])
            df["year"] = df["year"].astype(int)
            for m in PLAYER_METRICS:
                if m in df.columns:
                    df[m] = pd.to_numeric(df[m], errors="coerce")
                else:
                    df[m] = np.nan
            frames.append(df)
        except Exception:
            bad += 1
        if (i + 1) % 2000 == 0:
            print(f"   ...{i + 1}/{len(files)} files")
    print(f"   loaded ok, {bad} files unreadable")

    games = pd.concat(frames, ignore_index=True)
    games["era"] = games["year"].map(assign_era)
    print(f"   total player-game rows: {len(games):,}")
    print(f"   year range: {games['year'].min()} - {games['year'].max()}")
    print(f"   per era: {games['era'].value_counts().to_dict()}")
    return games


def summarise_per_era(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for era in ERA_ORDER:
        sub = games[games["era"] == era]
        n_rows = len(sub)
        n_players_approx = n_rows  # we don't have player_id here, but row count is informative
        for m in PLAYER_METRICS:
            vals = sub[m].dropna()
            n_obs = len(vals)
            if n_obs == 0:
                rows.append({
                    "era": era,
                    "metric": m,
                    "n_player_games": n_rows,
                    "n_with_metric": 0,
                    "mean_per_game": np.nan,
                    "std_per_game": np.nan,
                    "median_per_game": np.nan,
                    "mean_per_100pct_played": np.nan,
                })
                continue

            # Per-game raw mean (over rows where the stat is recorded)
            mean_pg = float(vals.mean())
            std_pg = float(vals.std())
            med_pg = float(vals.median())

            # Per-100%-time-played: scale each row by 100/pct_played, then mean.
            # Only for rows where pct_played is recorded and > 5 (avoid blow-ups
            # from bad data where pct=1).
            if m == "percentage_of_game_played":
                mean_norm = mean_pg
            else:
                pct = sub.loc[vals.index, "percentage_of_game_played"]
                mask = pct.notna() & (pct >= 25)
                if mask.sum() > 0:
                    scaled = vals[mask] * (100.0 / pct[mask])
                    mean_norm = float(scaled.mean())
                else:
                    mean_norm = np.nan

            rows.append({
                "era": era,
                "metric": m,
                "n_player_games": n_rows,
                "n_with_metric": n_obs,
                "mean_per_game": mean_pg,
                "std_per_game": std_pg,
                "median_per_game": med_pg,
                "mean_per_100pct_played": mean_norm,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_ERA_STATS, index=False)
    print(f"   wrote {OUT_ERA_STATS}")
    return df


def yearly_trends(games: pd.DataFrame) -> pd.DataFrame:
    grp = games.groupby("year")
    out = grp[PLAYER_METRICS].mean().reset_index()
    out["n_player_games"] = grp.size().values
    out.to_csv(OUT_YEARLY, index=False)
    print(f"   wrote {OUT_YEARLY}")
    return out


def adjacent_era_tests(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(len(ERA_ORDER) - 1):
        a_name = ERA_ORDER[i]
        b_name = ERA_ORDER[i + 1]
        a = games[games["era"] == a_name]
        b = games[games["era"] == b_name]
        for m in PLAYER_METRICS:
            av = a[m].dropna().values
            bv = b[m].dropna().values
            if len(av) < 30 or len(bv) < 30:
                rows.append({
                    "era_a": a_name,
                    "era_b": b_name,
                    "metric": m,
                    "n_a": len(av),
                    "n_b": len(bv),
                    "mean_a": float(av.mean()) if len(av) else np.nan,
                    "mean_b": float(bv.mean()) if len(bv) else np.nan,
                    "delta": np.nan,
                    "cohens_d": np.nan,
                    "welch_t": np.nan,
                    "p_value": np.nan,
                    "note": "insufficient data",
                })
                continue
            ma, mb = float(av.mean()), float(bv.mean())
            sa, sb = float(av.std(ddof=1)), float(bv.std(ddof=1))
            pooled = np.sqrt((sa ** 2 + sb ** 2) / 2.0) if (sa + sb) > 0 else np.nan
            d = (mb - ma) / pooled if pooled and pooled > 0 else np.nan
            t, p = stats.ttest_ind(av, bv, equal_var=False)
            rows.append({
                "era_a": a_name,
                "era_b": b_name,
                "metric": m,
                "n_a": len(av),
                "n_b": len(bv),
                "mean_a": ma,
                "mean_b": mb,
                "delta": mb - ma,
                "cohens_d": float(d) if d == d else np.nan,
                "welch_t": float(t),
                "p_value": float(p),
                "note": "",
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_SIG_TESTS, index=False)
    print(f"   wrote {OUT_SIG_TESTS}")
    return df


# ---------------------------------------------------------------------------
# 2. Match-level (team scoring / pace) analysis
# ---------------------------------------------------------------------------
def load_matches() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(MATCHES_DIR, "matches_*.csv")))
    print(f"[2/4] Loading {len(files)} match year files...")
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path)
            frames.append(df)
        except Exception as e:
            print(f"   skipped {path}: {e}")
    matches = pd.concat(frames, ignore_index=True)
    print(f"   loaded {len(matches):,} matches")
    return matches


def summarise_match_scoring(matches: pd.DataFrame) -> pd.DataFrame:
    """Two team-game records per match. Compute per-team-game scoring stats per era."""
    df = matches.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df["era"] = df["year"].map(assign_era)

    a = pd.DataFrame({
        "year": df["year"],
        "era": df["era"],
        "team": df["team_1_team_name"],
        "goals": pd.to_numeric(df["team_1_final_goals"], errors="coerce"),
        "behinds": pd.to_numeric(df["team_1_final_behinds"], errors="coerce"),
    })
    b = pd.DataFrame({
        "year": df["year"],
        "era": df["era"],
        "team": df["team_2_team_name"],
        "goals": pd.to_numeric(df["team_2_final_goals"], errors="coerce"),
        "behinds": pd.to_numeric(df["team_2_final_behinds"], errors="coerce"),
    })
    tg = pd.concat([a, b], ignore_index=True).dropna(subset=["goals", "behinds"])
    tg["points"] = tg["goals"] * 6 + tg["behinds"]
    tg["scoring_shots"] = tg["goals"] + tg["behinds"]
    tg["goal_accuracy"] = tg["goals"] / tg["scoring_shots"].replace(0, np.nan)

    by_era = (
        tg.groupby("era")
          .agg(
              n_team_games=("points", "count"),
              mean_goals=("goals", "mean"),
              mean_behinds=("behinds", "mean"),
              mean_points=("points", "mean"),
              mean_scoring_shots=("scoring_shots", "mean"),
              mean_goal_accuracy=("goal_accuracy", "mean"),
              std_points=("points", "std"),
          )
          .reindex(ERA_ORDER)
          .reset_index()
    )
    # Combined (both teams) score per match — match total
    df["match_total_goals"] = (
        pd.to_numeric(df["team_1_final_goals"], errors="coerce")
        + pd.to_numeric(df["team_2_final_goals"], errors="coerce")
    )
    df["match_total_behinds"] = (
        pd.to_numeric(df["team_1_final_behinds"], errors="coerce")
        + pd.to_numeric(df["team_2_final_behinds"], errors="coerce")
    )
    df["match_total_points"] = df["match_total_goals"] * 6 + df["match_total_behinds"]
    match_totals = (
        df.groupby("era")
          .agg(
              mean_match_total_goals=("match_total_goals", "mean"),
              mean_match_total_points=("match_total_points", "mean"),
              n_matches=("match_total_points", "count"),
          )
          .reindex(ERA_ORDER)
          .reset_index()
    )
    out = by_era.merge(match_totals, on="era", how="left")
    out.to_csv(OUT_TEAM_SCORING, index=False)
    print(f"   wrote {OUT_TEAM_SCORING}")
    return out


# ---------------------------------------------------------------------------
# 3. Print human-readable summary + JSON dump for downstream use
# ---------------------------------------------------------------------------
def build_summary(era_df: pd.DataFrame, scoring_df: pd.DataFrame, sig_df: pd.DataFrame) -> dict:
    summary = {"eras": ERA_ORDER, "era_stats": {}, "team_scoring": {}, "highlights": {}}

    # Reshape era_df: era -> metric -> {mean_per_game, ...}
    for era in ERA_ORDER:
        sub = era_df[era_df["era"] == era]
        summary["era_stats"][era] = {}
        for _, r in sub.iterrows():
            summary["era_stats"][era][r["metric"]] = {
                "n_player_games": int(r["n_player_games"]),
                "n_with_metric": int(r["n_with_metric"]),
                "mean_per_game": None if pd.isna(r["mean_per_game"]) else round(float(r["mean_per_game"]), 3),
                "median_per_game": None if pd.isna(r["median_per_game"]) else round(float(r["median_per_game"]), 3),
                "mean_per_100pct_played": None if pd.isna(r["mean_per_100pct_played"]) else round(float(r["mean_per_100pct_played"]), 3),
            }

    for _, r in scoring_df.iterrows():
        summary["team_scoring"][r["era"]] = {
            "n_team_games": int(r["n_team_games"]) if not pd.isna(r["n_team_games"]) else 0,
            "n_matches": int(r["n_matches"]) if not pd.isna(r["n_matches"]) else 0,
            "mean_goals": None if pd.isna(r["mean_goals"]) else round(float(r["mean_goals"]), 2),
            "mean_behinds": None if pd.isna(r["mean_behinds"]) else round(float(r["mean_behinds"]), 2),
            "mean_points": None if pd.isna(r["mean_points"]) else round(float(r["mean_points"]), 2),
            "mean_scoring_shots": None if pd.isna(r["mean_scoring_shots"]) else round(float(r["mean_scoring_shots"]), 2),
            "mean_goal_accuracy": None if pd.isna(r["mean_goal_accuracy"]) else round(float(r["mean_goal_accuracy"]), 4),
            "mean_match_total_points": None if pd.isna(r["mean_match_total_points"]) else round(float(r["mean_match_total_points"]), 2),
        }

    # A few highlights — biggest era-to-era jumps with non-trivial effect size
    big = sig_df.dropna(subset=["cohens_d"]).copy()
    big["abs_d"] = big["cohens_d"].abs()
    top = big.sort_values("abs_d", ascending=False).head(15)
    summary["highlights"]["largest_era_shifts"] = top[
        ["era_a", "era_b", "metric", "mean_a", "mean_b", "delta", "cohens_d", "p_value"]
    ].round(4).to_dict(orient="records")

    with open(OUT_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   wrote {OUT_SUMMARY_JSON}")
    return summary


def print_human_summary(summary: dict):
    print()
    print("=" * 78)
    print("ERA SUMMARY")
    print("=" * 78)
    pretty_metrics = [
        "kicks", "handballs", "disposals", "marks", "tackles",
        "clearances", "contested_possessions", "inside_50s",
        "goals", "behinds",
    ]
    header = f"{'metric':<26}" + "".join(f"{e:>14}" for e in ERA_ORDER)
    print(header)
    print("-" * len(header))
    for m in pretty_metrics:
        row = f"{m + ' (per game)':<26}"
        for e in ERA_ORDER:
            v = summary["era_stats"][e].get(m, {}).get("mean_per_game")
            n = summary["era_stats"][e].get(m, {}).get("n_with_metric", 0)
            if v is None or n < 100:
                cell = f"{'n/a':>14}"
            else:
                cell = f"{v:>14.2f}"
            row += cell
        print(row)
    print()
    print(f"{'team scoring (per team-game)':<30}")
    print(f"{'metric':<26}" + "".join(f"{e:>14}" for e in ERA_ORDER))
    print("-" * len(header))
    for key, label in [
        ("mean_goals", "goals"),
        ("mean_behinds", "behinds"),
        ("mean_points", "points"),
        ("mean_scoring_shots", "scoring shots"),
        ("mean_goal_accuracy", "goal %"),
        ("mean_match_total_points", "match total pts"),
    ]:
        row = f"{label:<26}"
        for e in ERA_ORDER:
            v = summary["team_scoring"].get(e, {}).get(key)
            if v is None:
                row += f"{'n/a':>14}"
            else:
                if key == "mean_goal_accuracy":
                    row += f"{v * 100:>13.1f}%"
                else:
                    row += f"{v:>14.2f}"
        print(row)

    print()
    print("LARGEST ERA-TO-ERA SHIFTS (Cohen's d, abs descending; p-values are rough indicators)")
    print("-" * 78)
    for h in summary["highlights"]["largest_era_shifts"]:
        print(
            f"  {h['era_a']} -> {h['era_b']}  {h['metric']:<24} "
            f"{h['mean_a']:>7.2f} -> {h['mean_b']:>7.2f}  "
            f"d={h['cohens_d']:>+6.2f}  p={h['p_value']:.2e}"
        )
    print("=" * 78)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    games = load_player_games()
    print("[1/4] summarising per-era player metrics...")
    era_df = summarise_per_era(games)
    print("[1/4] yearly trends...")
    yearly_trends(games)
    print("[1/4] adjacent era significance tests...")
    sig_df = adjacent_era_tests(games)

    matches = load_matches()
    print("[2/4] summarising match-level scoring...")
    scoring_df = summarise_match_scoring(matches)

    print("[3/4] building summary...")
    summary = build_summary(era_df, scoring_df, sig_df)
    print("[4/4] printing report...")
    print_human_summary(summary)
