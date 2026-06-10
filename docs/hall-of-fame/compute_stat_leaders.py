#!/usr/bin/env python3
"""Compute all-time AFL statistical leaders from the player_data corpus.

For every player file in data/player_data, sum career totals across every
performance row, then rank the top 20 in each category. Outputs:

    docs/hall-of-fame/_stat_leaders.json   # full structured output
    docs/hall-of-fame/_stat_leaders.md     # human-readable tables
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

REPO = Path(config.REPO_ROOT)
PLAYER_DIR = REPO / "data" / "player_data"
OUT_DIR = REPO / "docs" / "hall-of-fame"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# stat columns we care about (must exist in performance files)
STAT_COLS = [
    "kicks", "marks", "handballs", "disposals", "goals", "behinds",
    "hit_outs", "tackles", "rebound_50s", "inside_50s", "clearances",
    "clangers", "free_kicks_for", "free_kicks_against", "brownlow_votes",
    "contested_possessions", "uncontested_possessions", "contested_marks",
    "marks_inside_50", "one_percenters", "bounces", "goal_assist",
]


def player_id_from_perf(path: str) -> str:
    name = os.path.basename(path).replace("_performance_details.csv", "")
    return name


def pretty_name(player_id: str, personal_csv: Optional[Path]) -> str:
    """Return 'First Last' if personal CSV is available, else build from id."""
    if personal_csv and personal_csv.exists():
        try:
            p = pd.read_csv(personal_csv)
            if not p.empty and "first_name" in p.columns and "last_name" in p.columns:
                fn = str(p["first_name"].iloc[0]).strip()
                ln = str(p["last_name"].iloc[0]).strip()
                if fn and ln and fn.lower() != "nan":
                    return f"{fn} {ln}"
        except Exception:
            pass
    # fallback: parse id
    parts = player_id.split("_")
    if len(parts) >= 2:
        return f"{parts[1].capitalize()} {parts[0].capitalize()}"
    return player_id


def load_career(path: str):
    """Load one player's performance file and return a career-summary dict."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    if df.empty:
        return None

    # only keep rows where the player actually played a recognised game
    if "year" not in df.columns or "team" not in df.columns:
        return None

    # career span
    years = pd.to_numeric(df["year"], errors="coerce").dropna()
    if years.empty:
        return None
    year_min, year_max = int(years.min()), int(years.max())

    # games played: the per-row `games_played` column carries the running
    # AFL games tally (e.g. "432" or "432↑" for the season-debut bump). Use
    # its max as the authoritative count - row count alone undercounts
    # because drawn Grand Finals are collapsed into a single row and a
    # handful of finals appearances are not separately rowed in the
    # FanFooty-derived per-player files. Fall back to row count if the
    # column is missing or unparseable.
    games = int(df["team"].notna().sum())
    if "games_played" in df.columns:
        gp_clean = (
            df["games_played"]
            .astype(str)
            .str.replace("↑", "", regex=False)
            .str.replace("↓", "", regex=False)
        )
        gp_num = pd.to_numeric(gp_clean, errors="coerce")
        if gp_num.notna().any():
            # Counter when it leads (catches games with no stat-detail row), but
            # never below the row count — a trailing run of NaN games_played
            # values must not undercount a player who clearly played those rows.
            games = max(games, int(gp_num.max()))

    teams = (
        df.dropna(subset=["team"])  # type: ignore[arg-type]
        .loc[:, "team"]
        .astype(str)
        .unique()
        .tolist()
    )
    teams_str = " - ".join(sorted({t.strip() for t in teams if t.strip()}))

    out = {
        "games": games,
        "year_min": year_min,
        "year_max": year_max,
        "teams": teams_str,
    }

    for col in STAT_COLS:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            out[col + "_total"] = float(s.sum(skipna=True))
            # for single-season records we group by year
            season_max = s.groupby(df["year"]).sum(min_count=1).max()
            out[col + "_season_max"] = (
                float(season_max) if pd.notna(season_max) else 0.0
            )
            # which season produced the max
            try:
                grp = s.groupby(df["year"]).sum(min_count=1)
                if grp.notna().any():
                    out[col + "_season_max_year"] = int(grp.idxmax())
                else:
                    out[col + "_season_max_year"] = None
            except Exception:
                out[col + "_season_max_year"] = None
        else:
            out[col + "_total"] = 0.0
            out[col + "_season_max"] = 0.0
            out[col + "_season_max_year"] = None
    return out


def main() -> None:
    perf_files = sorted(glob.glob(str(PLAYER_DIR / "*_performance_details.csv")))
    print(f"perf files: {len(perf_files)}")

    rows = []
    for i, path in enumerate(perf_files):
        if i % 1000 == 0:
            print(f"  {i}/{len(perf_files)}")
        pid = player_id_from_perf(path)
        personal = PLAYER_DIR / f"{pid}_personal_details.csv"
        career = load_career(path)
        if career is None:
            continue
        if career["games"] < 1:
            continue
        career["player_id"] = pid
        career["name"] = pretty_name(pid, personal)
        rows.append(career)

    df = pd.DataFrame(rows)
    print(f"players with at least one game: {len(df)}")

    # de-duplicate by (name, year_min, year_max, teams) just in case
    # (no exact duplicates expected because of unique birth-date-based ids)

    # --- categories --------------------------------------------------------
    categories = {
        "career_goals": ("goals_total", "Career goals"),
        "career_disposals": ("disposals_total", "Career disposals"),
        "career_marks": ("marks_total", "Career marks"),
        "career_tackles": ("tackles_total", "Career tackles"),
        "career_contested_possessions": (
            "contested_possessions_total",
            "Career contested possessions",
        ),
        "career_clearances": ("clearances_total", "Career clearances"),
        "career_inside_50s": ("inside_50s_total", "Career inside 50s"),
        "career_hit_outs": ("hit_outs_total", "Career hit-outs"),
        "career_brownlow_votes": (
            "brownlow_votes_total",
            "Career Brownlow votes",
        ),
        "career_games": ("games", "Career games played"),
        "career_goal_assists": ("goal_assist_total", "Career goal assists"),
        "career_kicks": ("kicks_total", "Career kicks"),
        "career_handballs": ("handballs_total", "Career handballs"),
    }

    output: dict = {"meta": {"player_count": int(len(df))}, "categories": {}}

    for key, (col, label) in categories.items():
        if col not in df.columns:
            print(f"  skipping {key}: column {col} missing")
            continue
        # only consider players where the stat is positive — avoids returning
        # players with all-zero stat lines for stats not recorded in their era
        sub = df[df[col] > 0].copy()
        sub = sub.sort_values(col, ascending=False).head(20).reset_index(drop=True)
        # tie-aware ranks: identical totals share the same rank, displayed
        # with a trailing "=" (e.g. "1=" / "1="). This matters first for
        # career_games where Harvey and Pendlebury sit equal at 432; it also
        # future-proofs every other category in case of identical totals.
        totals_list = [float(v) for v in sub[col].tolist()]
        rec = []
        for r, row in sub.iterrows():
            tot_val = float(row[col])
            # min-rank among rows with the same total (1-indexed)
            base_rank = next(
                i for i, v in enumerate(totals_list, start=1) if v == tot_val
            )
            tie_count = sum(1 for v in totals_list if v == tot_val)
            rank_label = f"{base_rank}=" if tie_count > 1 else str(base_rank)
            rec.append(
                {
                    "rank": base_rank,
                    "rank_label": rank_label,
                    "tied": tie_count > 1,
                    "name": row["name"],
                    "teams": row["teams"],
                    "year_min": int(row["year_min"]),
                    "year_max": int(row["year_max"]),
                    "games": int(row["games"]),
                    "total": float(row[col]),
                    "per_game": float(row[col]) / max(int(row["games"]), 1),
                }
            )
        output["categories"][key] = {"label": label, "leaders": rec}

    # --- single-season records -------------------------------------------
    single_season = {}
    for stat in [
        "goals", "disposals", "marks", "tackles", "kicks", "handballs",
        "clearances", "inside_50s", "contested_possessions", "hit_outs",
        "brownlow_votes",
    ]:
        col = stat + "_season_max"
        ycol = stat + "_season_max_year"
        if col not in df.columns:
            continue
        sub = df[df[col] > 0].copy()
        sub = sub.sort_values(col, ascending=False).head(10).reset_index(drop=True)
        rec = []
        for r, row in sub.iterrows():
            rec.append(
                {
                    "rank": int(r) + 1,
                    "name": row["name"],
                    "teams": row["teams"],
                    "season": int(row[ycol]) if pd.notna(row[ycol]) else None,
                    "total": float(row[col]),
                }
            )
        single_season[stat] = rec
    output["single_season"] = single_season

    out_json = OUT_DIR / "_stat_leaders.json"
    out_json.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_json}")

    # --- markdown summary -------------------------------------------------
    lines = ["# All-time leaders (auto-generated)\n"]
    for key, payload in output["categories"].items():
        lines.append(f"\n## {payload['label']}\n")
        lines.append("| # | Player | Club(s) | Span | Games | Total | Per game |")
        lines.append("|--:|--------|---------|------|------:|------:|---------:|")
        for r in payload["leaders"]:
            tot = r["total"]
            tot_str = f"{tot:,.0f}" if tot >= 100 else f"{tot:,.1f}"
            rank_disp = r.get("rank_label", str(r["rank"]))
            lines.append(
                f"| {rank_disp} | {r['name']} | {r['teams']} | "
                f"{r['year_min']}-{r['year_max']} | {r['games']} | "
                f"{tot_str} | {r['per_game']:.2f} |"
            )
    lines.append("\n## Single-season records\n")
    for stat, rec in single_season.items():
        lines.append(f"\n### {stat.replace('_', ' ').title()}\n")
        lines.append("| # | Player | Club(s) | Season | Total |")
        lines.append("|--:|--------|---------|-------:|------:|")
        for r in rec[:10]:
            tot = r["total"]
            tot_str = f"{tot:,.0f}" if tot >= 100 else f"{tot:,.1f}"
            season_str = str(r["season"]) if r["season"] else "—"
            lines.append(
                f"| {r['rank']} | {r['name']} | {r['teams']} | {season_str} | {tot_str} |"
            )
    out_md = OUT_DIR / "_stat_leaders.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
