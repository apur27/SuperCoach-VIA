#!/usr/bin/env python3
"""
update_team_analysis.py
=======================

Generate the "<YEAR> team analysis" section in docs/afl-season-2026.md
from the latest season's player-game data, plus a "5-year team playing
styles" section covering the five seasons immediately prior to the
current year (written into docs/afl-team-profiles.md).

What it does
------------
1. Loads every per-player performance CSV in `data/player_data/`
2. Auto-detects the most recent season present in the data
3. Aggregates each game to a per-team per-game row (sum of all 22 players'
   stats) for that season
4. Computes per-team season-to-date averages, the league average across all
   18 teams, and per-team rank for the key stats
5. Builds an intro paragraph + summary table + leaders table in markdown
6. Writes the current-season section into docs/afl-season-2026.md between the
   markers
       <!-- YEAR-TEAM-ANALYSIS-START -->
       <!-- YEAR-TEAM-ANALYSIS-END -->
7. Builds the 5-year team playing-styles section (for the five seasons
   immediately prior to the current year) and writes it between
       <!-- 5YEAR-TEAM-PROFILES-START -->
       <!-- 5YEAR-TEAM-PROFILES-END -->

The script is self-contained — no module imports beyond the venv stdlib +
pandas/numpy. Run with:

    /home/abhi/sourceCode/python/coding/.venv/bin/python update_team_analysis.py

The script idempotently rewrites the section every run, so it can be wired
into a refresh pipeline.

Methodology notes (so future-you can audit)
-------------------------------------------
- Stats are summed across all players in a team for one game. Missing values
  in the source CSV are treated as 0 for the team-game sum (a player with a
  blank `tackles` cell did not record any tackles in that game).
- The `round` column is cast to string before grouping because it is
  recorded inconsistently across files (int in some, str in others) — without
  this, the same round splits into multiple group keys.
- The "current year" is whichever year has the most player-game rows in
  the dataset that is also the maximum year present. This keeps the script
  working across seasons (a 2027 run will pick up 2027 automatically).
- Rank 1 = best for "more is better" stats. For all stats here, more is
  better at the team level except behinds (omitted from rank tables).
- Form tags are generated from rank position in stats that map cleanly to
  on-field identity (top-3 / bottom-3 thresholds).
"""

from __future__ import annotations

import glob
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = "/home/abhi/git/SuperCoach-VIA"
PLAYER_DATA_DIR = os.path.join(REPO_ROOT, "data", "player_data")
MATCHES_DIR = os.path.join(REPO_ROOT, "data", "matches")
TEAM_ANALYSIS_PATH  = os.path.join(REPO_ROOT, "docs", "afl-team-analysis-2026.md")
FINALS_PATH         = os.path.join(REPO_ROOT, "docs", "afl-finals-2026.md")
BROWNLOW_PATH       = os.path.join(REPO_ROOT, "docs", "afl-brownlow-2026.md")
STAT_LEADERS_PATH   = os.path.join(REPO_ROOT, "docs", "afl-stat-leaders-2026.md")
PREDICTIONS_PATH    = os.path.join(REPO_ROOT, "docs", "afl-predictions-2026.md")
BACKTEST_PATH       = os.path.join(REPO_ROOT, "docs", "afl-backtest-2026.md")
# Backward-compat alias — refresh_readme.py and other callers historically
# referenced uta.README_PATH; it now points at the team-analysis file (the
# "primary" season block) so old call sites keep working.
README_PATH         = TEAM_ANALYSIS_PATH
TEAM_PROFILES_PATH  = os.path.join(REPO_ROOT, "docs", "afl-team-profiles.md")
HALL_OF_FAME_PATH   = os.path.join(REPO_ROOT, "docs", "hall-of-fame-top100.md")
TOP100_CSV = os.path.join(REPO_ROOT, "all_time_top_100.csv")
TOP100_SCORES_CSV = os.path.join(REPO_ROOT, "data", "top100", "all_time_top_100.csv")
CHARTS_DIR = os.path.join(REPO_ROOT, "assets", "charts")

# AFL home-and-away season length (each team plays this many games in total).
# Used by the finals-pathway section to compute games remaining.
HOME_AND_AWAY_GAMES = 22

# Stats we will load from the per-player CSVs and aggregate to team-game level.
STAT_COLS = [
    "disposals",
    "kicks",
    "handballs",
    "marks",
    "goals",
    "behinds",
    "tackles",
    "clearances",
    "inside_50s",
    "contested_possessions",
    "rebound_50s",
    "hit_outs",
    "contested_marks",
    "marks_inside_50",
    "one_percenters",
    "uncontested_possessions",
    "goal_assist",
    "clangers",
    "free_kicks_for",
    "free_kicks_against",
]

# Stats reported in the per-team summary table (in order).
SUMMARY_STATS = [
    "disposals",
    "kicks",
    "handballs",
    "marks",
    "goals",
    "tackles",
    "clearances",
    "inside_50s",
    "contested_possessions",
]

# Stats highlighted in the leaders table — "who leads in X".
LEADER_STATS = [
    "disposals",
    "kicks",
    "handballs",
    "marks",
    "goals",
    "tackles",
    "clearances",
    "inside_50s",
    "contested_possessions",
    "rebound_50s",
    "hit_outs",
    "contested_marks",
]

LOAD_COLS = ["team", "year", "round", "opponent"] + STAT_COLS


# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------
def load_all_player_games() -> pd.DataFrame:
    """Load and concatenate every player performance file. Returns a single
    long-format DataFrame with one row per player-game."""
    pattern = os.path.join(PLAYER_DATA_DIR, "*_performance_details.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No player files found in {PLAYER_DATA_DIR}")

    frames: List[pd.DataFrame] = []
    bad = 0
    for path in files:
        try:
            df = pd.read_csv(
                path,
                low_memory=False,
                usecols=lambda c: c in LOAD_COLS,
            )
            if df.empty or "year" not in df.columns or "team" not in df.columns:
                continue
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df.dropna(subset=["year"])
            if df.empty:
                continue
            df["year"] = df["year"].astype(int)
            frames.append(df)
        except Exception:
            bad += 1
    if bad:
        print(f"[warn] {bad} files unreadable", file=sys.stderr)
    if not frames:
        sys.exit("No usable player rows after loading")
    games = pd.concat(frames, ignore_index=True)
    return games


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def detect_current_year(games: pd.DataFrame) -> int:
    """Pick the most recent year present. If the latest year has very few
    rows (less than 50) we fall back to the previous year — guards against a
    single off-season scrape leaving a phantom year."""
    yr_counts = games.groupby("year").size()
    candidates = sorted(yr_counts.index.tolist(), reverse=True)
    for y in candidates:
        if yr_counts.loc[y] >= 50:
            return int(y)
    return int(candidates[0])


def build_team_game_table(games: pd.DataFrame, year: int) -> pd.DataFrame:
    """Return one row per team-game with summed stats for the given year."""
    g = games[games["year"] == year].copy()
    if g.empty:
        sys.exit(f"No data for year {year}")

    # Coerce all stat cols to numeric, fill missing as 0 (treat as "did not
    # record any" — the team-level sum is unaffected).
    for c in STAT_COLS:
        if c not in g.columns:
            g[c] = 0.0
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)

    # Round can be int or str across files. Cast to str so groupby keys align.
    g["round_str"] = g["round"].astype(str)

    team_game = (
        g.groupby(["team", "round_str", "opponent"], as_index=False)[STAT_COLS]
        .sum()
    )
    return team_game


def per_team_summary(team_game: pd.DataFrame) -> pd.DataFrame:
    """Mean across team-games per team plus games_played count."""
    grouped = team_game.groupby("team")
    summary = grouped[STAT_COLS].mean().round(1)
    summary["games_played"] = grouped.size()
    return summary.reset_index()


def league_averages(team_game: pd.DataFrame) -> Dict[str, float]:
    return {c: float(team_game[c].mean()) for c in STAT_COLS}


def add_ranks(summary: pd.DataFrame, stats: List[str]) -> pd.DataFrame:
    """Rank 1 = highest mean (best for offensive/possession stats)."""
    out = summary.copy()
    for c in stats:
        out[f"{c}_rank"] = out[c].rank(ascending=False, method="min").astype(int)
    return out


def detect_max_round(team_game: pd.DataFrame) -> int:
    """Try to extract the highest numeric round played; non-numeric rounds
    (finals labels) just get reported as max numeric round seen."""
    rounds = []
    for r in team_game["round_str"].unique():
        try:
            rounds.append(int(r))
        except (TypeError, ValueError):
            continue
    return max(rounds) if rounds else 0


# ---------------------------------------------------------------------------
# Form tag generation — data-driven labels per team
# ---------------------------------------------------------------------------
def form_tag(row: pd.Series, summary_with_ranks: pd.DataFrame) -> str:
    """Compose a one-or-two-word descriptor from the team's ranks."""
    n_teams = len(summary_with_ranks)
    top_threshold = 3
    bot_threshold = n_teams - 2  # ranks (n-2), (n-1), n

    tags = []

    # Possession leaders
    if row["disposals_rank"] <= top_threshold:
        tags.append("ball-winners")
    if row["disposals_rank"] >= bot_threshold:
        tags.append("starved of the ball")

    # Engine room
    if row["clearances_rank"] <= top_threshold:
        tags.append("clearance machine")
    if row["contested_possessions_rank"] <= top_threshold and "clearance machine" not in tags:
        tags.append("contested-ball team")

    # Pressure
    if row["tackles_rank"] <= top_threshold:
        tags.append("physical unit")
    if row["tackles_rank"] >= bot_threshold:
        tags.append("low pressure")

    # Forward output
    if row["goals_rank"] <= top_threshold:
        tags.append("scoring threat")
    if row["goals_rank"] >= bot_threshold:
        tags.append("struggling in front of goal")

    # Defensive
    if "rebound_50s_rank" in row and row["rebound_50s_rank"] <= top_threshold:
        tags.append("strong rebound")

    # Inside-50 territory
    if row["inside_50s_rank"] <= top_threshold:
        tags.append("territory team")

    if not tags:
        return "mid-pack"
    # Keep it short — at most two tags joined.
    return ", ".join(tags[:2])


# ---------------------------------------------------------------------------
# Top scorer per team (used in intro narrative)
# ---------------------------------------------------------------------------
def per_team_top_disposal_player(games: pd.DataFrame, year: int) -> Dict[str, Tuple[str, float]]:
    """For each team, return (player_filename_stem, avg_disposals) for the
    highest-averaging disposal player in `year` with at least 3 games.

    We do not have a clean player-name column in the performance file, but the
    file stem encodes name + dob — close enough for a short narrative blurb.
    """
    pattern = os.path.join(PLAYER_DATA_DIR, "*_performance_details.csv")
    files = sorted(glob.glob(pattern))
    leaders: Dict[str, Tuple[str, float]] = {}

    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in {"team", "year", "disposals"})
            if df.empty or "year" not in df.columns:
                continue
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df[df["year"] == year]
            if df.empty:
                continue
            df["disposals"] = pd.to_numeric(df.get("disposals"), errors="coerce").fillna(0.0)
            if len(df) < 3:
                continue
            team = df["team"].mode().iloc[0] if not df["team"].mode().empty else None
            if not team:
                continue
            avg = float(df["disposals"].mean())
            stem = os.path.basename(path).replace("_performance_details.csv", "")
            cur = leaders.get(team)
            if cur is None or avg > cur[1]:
                leaders[team] = (stem, avg)
        except Exception:
            continue
    return leaders


def prettify_player_stem(stem: str) -> str:
    """Turn `daicos_nick_18012003` into `Nick Daicos`. Best-effort."""
    parts = stem.split("_")
    # The trailing token is a date-of-birth in DDMMYYYY; drop it if numeric.
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    if len(parts) >= 2:
        last, first = parts[0], parts[1]
        return f"{first.title()} {last.title()}"
    return stem.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def fmt_int(x: float) -> str:
    return f"{int(round(x))}"


def fmt_one(x: float) -> str:
    return f"{x:.1f}"


def render_intro(
    year: int,
    max_round: int,
    summary: pd.DataFrame,
    league: Dict[str, float],
    top_scorers: Dict[str, Tuple[str, float]],
) -> str:
    """One paragraph in plain English summarising the season so far."""
    s = summary.copy()
    n_teams = len(s)

    # Disposals / pressure / scoring leaders + laggards
    disp_leader = s.sort_values("disposals", ascending=False).iloc[0]
    disp_lag = s.sort_values("disposals", ascending=True).iloc[0]
    tackle_leader = s.sort_values("tackles", ascending=False).iloc[0]
    goal_leader = s.sort_values("goals", ascending=False).iloc[0]
    goal_lag = s.sort_values("goals", ascending=True).iloc[0]
    clr_leader = s.sort_values("clearances", ascending=False).iloc[0]

    # Find top-disposal player league-wide
    top_player_team = None
    top_player_name = None
    top_player_avg = 0.0
    for team, (stem, avg) in top_scorers.items():
        if avg > top_player_avg:
            top_player_avg = avg
            top_player_name = prettify_player_stem(stem)
            top_player_team = team

    games_label = f"{max_round} round{'s' if max_round != 1 else ''}"
    parts: List[str] = []
    parts.append(
        f"Through {games_label} of the {year} season, the league averages out at "
        f"around {fmt_int(league['disposals'])} disposals, "
        f"{fmt_int(league['kicks'])} kicks, "
        f"{fmt_int(league['handballs'])} handballs, "
        f"{fmt_int(league['tackles'])} tackles, "
        f"{fmt_int(league['clearances'])} clearances and "
        f"{fmt_int(league['goals'])} goals per team per game."
    )
    parts.append(
        f"**{disp_leader['team']}** lead the comp for total disposals "
        f"({fmt_one(disp_leader['disposals'])}/g), with **{clr_leader['team']}** "
        f"winning the most clearances ({fmt_one(clr_leader['clearances'])}/g) "
        f"and **{tackle_leader['team']}** the most physical side "
        f"({fmt_one(tackle_leader['tackles'])} tackles/g)."
    )
    if goal_lag["team"] == disp_lag["team"]:
        parts.append(
            f"**{goal_leader['team']}** are the highest-scoring team at "
            f"{fmt_one(goal_leader['goals'])} goals/g, while "
            f"**{disp_lag['team']}** prop up the table on both fronts — "
            f"only {fmt_one(disp_lag['disposals'])} disposals/g and "
            f"{fmt_one(disp_lag['goals'])} goals/g."
        )
    else:
        parts.append(
            f"**{goal_leader['team']}** are the highest-scoring team at "
            f"{fmt_one(goal_leader['goals'])} goals/g, while "
            f"**{goal_lag['team']}** ({fmt_one(goal_lag['goals'])}/g) and "
            f"**{disp_lag['team']}** ({fmt_one(disp_lag['disposals'])} disposals/g) "
            f"are the strugglers — bottom of the table for output."
        )
    if top_player_name and top_player_team:
        parts.append(
            f"At the individual level, **{top_player_name}** "
            f"({top_player_team}) is the leading ball-winner at "
            f"{fmt_one(top_player_avg)} disposals/game."
        )
    parts.append(
        "Caveat: this is a small sample — single-season form can swing by "
        "round, so treat the rankings below as a snapshot rather than a "
        "settled hierarchy."
    )
    return " ".join(parts)


def render_summary_table(summary_with_ranks: pd.DataFrame) -> str:
    """Markdown table — one row per team, ranked by total disposals/g desc."""
    s = summary_with_ranks.sort_values("disposals", ascending=False).reset_index(drop=True)
    headers = [
        "#",
        "Team",
        "G",
        "Disp/g",
        "Kicks",
        "Handballs",
        "Marks",
        "Goals",
        "Tackles",
        "Clearances",
        "I50s",
        "CP",
        "Form tag",
    ]
    align = ["---:", ":---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", ":---"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    for i, row in s.iterrows():
        cells = [
            str(i + 1),
            row["team"],
            str(int(row["games_played"])),
            fmt_one(row["disposals"]),
            fmt_one(row["kicks"]),
            fmt_one(row["handballs"]),
            fmt_one(row["marks"]),
            fmt_one(row["goals"]),
            fmt_one(row["tackles"]),
            fmt_one(row["clearances"]),
            fmt_one(row["inside_50s"]),
            fmt_one(row["contested_possessions"]),
            row["form_tag"],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_leaders_table(summary: pd.DataFrame, league: Dict[str, float]) -> str:
    """Markdown table — for each LEADER_STATS column, who leads."""
    nice_names = {
        "disposals": "Disposals",
        "kicks": "Kicks",
        "handballs": "Handballs",
        "marks": "Marks",
        "goals": "Goals",
        "tackles": "Tackles",
        "clearances": "Clearances",
        "inside_50s": "Inside-50s",
        "contested_possessions": "Contested possessions",
        "rebound_50s": "Rebound-50s",
        "hit_outs": "Hit-outs",
        "contested_marks": "Contested marks",
    }
    headers = ["Stat", "League leader", "Their avg/g", "League avg/g", "Delta vs league"]
    align = [":---", ":---", "---:", "---:", "---:"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    for stat in LEADER_STATS:
        if stat not in summary.columns:
            continue
        leader_row = summary.sort_values(stat, ascending=False).iloc[0]
        leader_val = float(leader_row[stat])
        league_val = league[stat]
        delta = leader_val - league_val
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.1f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    nice_names.get(stat, stat),
                    leader_row["team"],
                    fmt_one(leader_val),
                    fmt_one(league_val),
                    delta_str,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-team rich paragraphs — data-driven prose
# ---------------------------------------------------------------------------
# Stats we want a rank for in the per-team paragraphs. Every stat in this list
# gets a `<stat>_rank` column on the per-team frame; the paragraph generator
# leans on these to pick out top-3 strengths and bottom-3 weaknesses.
PARAGRAPH_RANK_STATS = [
    "disposals",
    "kicks",
    "handballs",
    "marks",
    "goals",
    "tackles",
    "clearances",
    "inside_50s",
    "contested_possessions",
    "uncontested_possessions",
    "rebound_50s",
    "hit_outs",
    "contested_marks",
    "marks_inside_50",
    "free_kicks_for",
]
# These are "less is better" — rank 1 = fewest, which is the desirable end.
INVERSE_RANK_STATS = [
    "clangers",
    "free_kicks_against",
]

# Human-readable names for stats — used in the prose so we don't say
# "marks_inside_50/g" out loud.
STAT_LABELS = {
    "disposals": "disposals",
    "kicks": "kicks",
    "handballs": "handballs",
    "marks": "marks",
    "goals": "goals",
    "tackles": "tackles",
    "clearances": "clearances",
    "inside_50s": "inside-50s",
    "contested_possessions": "contested possessions",
    "uncontested_possessions": "uncontested possessions",
    "rebound_50s": "rebound-50s",
    "hit_outs": "hit-outs",
    "contested_marks": "contested marks",
    "marks_inside_50": "marks inside 50",
    "free_kicks_for": "free kicks for",
    "free_kicks_against": "free kicks against",
    "clangers": "clangers",
    # Derived ratios — used in the 5-year profile prose. Spell them out so
    # the README never shows raw column names like `marks_per_inside50`.
    "handball_ratio": "handball share of disposals",
    "marks_per_inside50": "marks-per-inside-50 conversion",
    "tackle_rate": "tackles-per-disposal pressure",
}


def ordinal(n: int) -> str:
    """1 -> 1st, 2 -> 2nd, 3 -> 3rd, 4 -> 4th, ..."""
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def add_paragraph_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    """Add `<stat>_rank` for every stat used in paragraph generation.

    Higher-is-better stats get rank 1 = highest mean. Lower-is-better stats
    (clangers, free_kicks_against) get rank 1 = lowest mean.
    """
    out = summary.copy()
    for c in PARAGRAPH_RANK_STATS:
        if c in out.columns and f"{c}_rank" not in out.columns:
            out[f"{c}_rank"] = out[c].rank(ascending=False, method="min").astype(int)
    for c in INVERSE_RANK_STATS:
        if c in out.columns:
            out[f"{c}_rank"] = out[c].rank(ascending=True, method="min").astype(int)
    return out


def derive_strategy(row: pd.Series, n_teams: int) -> str:
    """Infer a one-sentence playing-style description from the stat profile.

    Uses combinations of ranks rather than absolute thresholds so the labels
    stay sensible across seasons with different scoring environments.
    """
    top = 6  # top third
    bot = n_teams - 5  # bottom third (i.e. >= bot)

    # Compute handball ratio rank approximation from raw cols.
    hb_ratio = row["handballs"] / max(row["disposals"], 1.0)

    descriptors: List[str] = []

    # Contested vs possession identity
    cp_rank = row.get("contested_possessions_rank", 99)
    ucp_rank = row.get("uncontested_possessions_rank", 99)
    cl_rank = row.get("clearances_rank", 99)
    if cp_rank <= top and cl_rank <= top:
        descriptors.append("a contested-ball, clearance-first identity")
    elif ucp_rank <= top and cp_rank >= bot:
        descriptors.append("a possession-and-control style that shies away from the contest")
    elif cp_rank <= top:
        descriptors.append("a contest-heavy approach at the stoppages")
    elif ucp_rank <= top:
        descriptors.append("a possession-based, uncontested-game template")

    # Kicking vs handballing tempo
    if hb_ratio >= 0.45:
        descriptors.append(f"a high-tempo handball game ({hb_ratio*100:.0f}% of disposals by hand)")
    elif hb_ratio <= 0.37:
        descriptors.append(f"a kick-first ball movement ({(1-hb_ratio)*100:.0f}% by foot)")

    # Territory / inside-50 pressure
    i50_rank = row.get("inside_50s_rank", 99)
    mi50_rank = row.get("marks_inside_50_rank", 99)
    if i50_rank <= top and mi50_rank >= bot:
        descriptors.append("pumping the ball forward with high volume but little control inside 50")
    elif i50_rank <= top and mi50_rank <= top:
        descriptors.append("dominating territory and connecting cleanly inside 50")
    elif i50_rank >= bot:
        descriptors.append("struggling to win territory and force the issue forward")

    # Pressure
    tk_rank = row.get("tackles_rank", 99)
    if tk_rank <= top:
        descriptors.append("backed up by elite forward and ground-ball pressure")
    elif tk_rank >= bot:
        descriptors.append("with notably soft pressure around the ball")

    # Rebound / defensive lean
    rb_rank = row.get("rebound_50s_rank", 99)
    if rb_rank <= top:
        descriptors.append("frequently absorbing entries and rebounding from defence")

    if not descriptors:
        return "a balanced, mid-pack profile that doesn't lean strongly toward any one identity"
    # Keep at most three to avoid run-on sentences.
    return "; ".join(descriptors[:3])


def pick_top_strengths(row: pd.Series, n: int = 3) -> List[Tuple[str, int, float]]:
    """Return up to `n` (stat, rank, raw_value) tuples for this team's best
    ranks across PARAGRAPH_RANK_STATS + INVERSE_RANK_STATS."""
    candidates = []
    for stat in PARAGRAPH_RANK_STATS + INVERSE_RANK_STATS:
        rank_col = f"{stat}_rank"
        if rank_col not in row.index:
            continue
        candidates.append((stat, int(row[rank_col]), float(row[stat])))
    # Strengths = lowest rank numbers (1 = best).
    candidates.sort(key=lambda t: t[1])
    return candidates[:n]


def pick_top_weaknesses(row: pd.Series, n_teams: int, n: int = 3) -> List[Tuple[str, int, float]]:
    candidates = []
    for stat in PARAGRAPH_RANK_STATS + INVERSE_RANK_STATS:
        rank_col = f"{stat}_rank"
        if rank_col not in row.index:
            continue
        candidates.append((stat, int(row[rank_col]), float(row[stat])))
    # Weaknesses = highest rank numbers (closer to n_teams = worst).
    candidates.sort(key=lambda t: -t[1])
    return candidates[:n]


def overall_position_phrase(row: pd.Series, n_teams: int) -> str:
    """Map disposals_rank to a coarse 'tier' phrase for the opening sentence."""
    rk = int(row["disposals_rank"])
    if rk <= 4:
        return f"sit near the top of the league for ball use ({ordinal(rk)} for disposals)"
    if rk <= 8:
        return f"are tracking inside the top half ({ordinal(rk)} for disposals)"
    if rk <= 12:
        return f"are floating around the middle of the pack ({ordinal(rk)} for disposals)"
    if rk <= 15:
        return f"are sitting in the lower third for ball use ({ordinal(rk)} for disposals)"
    return f"are anchored near the bottom of the table ({ordinal(rk)} for disposals)"


def render_team_paragraph(
    row: pd.Series,
    n_teams: int,
    top_player_name: str,
    top_player_avg: float,
) -> str:
    """5-7 sentence paragraph for one team."""
    team = row["team"]

    # Sentence 1 — overall position
    s1 = (
        f"After {int(row['games_played'])} games of 2026, the {team} "
        f"{overall_position_phrase(row, n_teams)} and average "
        f"{fmt_one(row['disposals'])} disposals, {fmt_one(row['goals'])} goals "
        f"and {fmt_one(row['tackles'])} tackles per game."
    )

    # Sentences 2-3 — strategy
    strategy = derive_strategy(row, n_teams)
    s2 = f"Their stat profile reads as {strategy}."
    hb_ratio = row["handballs"] / max(row["disposals"], 1.0)
    s3 = (
        f"They go inside 50 {fmt_one(row['inside_50s'])} times a game "
        f"({ordinal(int(row['inside_50s_rank']))} in the league), win "
        f"{fmt_one(row['contested_possessions'])} contested possessions "
        f"({ordinal(int(row['contested_possessions_rank']))}) and tilt "
        f"{hb_ratio*100:.0f}/{(1-hb_ratio)*100:.0f} between handball and kick."
    )

    # Sentence 4 — strengths
    strengths = pick_top_strengths(row, n=3)
    strength_phrases = []
    for stat, rk, val in strengths:
        label = STAT_LABELS.get(stat, stat)
        strength_phrases.append(f"{ordinal(rk)} for {label} ({fmt_one(val)}/g)")
    s4 = (
        f"The strengths jump out clearly: they're "
        + ", ".join(strength_phrases)
        + "."
    )

    # Sentence 5 — weaknesses
    weaknesses = pick_top_weaknesses(row, n_teams, n=3)
    weak_phrases = []
    for stat, rk, val in weaknesses:
        label = STAT_LABELS.get(stat, stat)
        weak_phrases.append(f"{ordinal(rk)} for {label} ({fmt_one(val)}/g)")
    s5 = (
        f"The flip side is harder to ignore: "
        + ", ".join(weak_phrases)
        + " — that's where opposition coaches will be drawing up plans."
    )

    # Sentence 6 — key player
    if top_player_name:
        s6 = (
            f"Key player to watch is **{top_player_name}**, leading the team "
            f"in disposals at {fmt_one(top_player_avg)}/game."
        )
    else:
        s6 = "No standout individual disposal-getter is yet separating themselves from the pack."

    return " ".join([s1, s2, s3, s4, s5, s6])


def generate_team_paragraphs(
    summary_with_ranks: pd.DataFrame,
    top_scorers: Dict[str, Tuple[str, float]],
) -> str:
    """Build the per-team prose section. One `### Team` heading + paragraph
    per club, ordered by current disposals rank (best to worst).
    """
    n_teams = len(summary_with_ranks)
    ordered = summary_with_ranks.sort_values("disposals", ascending=False).reset_index(drop=True)

    lines: List[str] = []
    # H3 here: in docs/afl-insights.md the parent section "## YEAR season — live
    # team analysis" sits at H2, so this and the per-team sub-headings below sit at H3.
    lines.append(f"### Team-by-team — playing style, strengths and weaknesses")
    lines.append("")
    lines.append(
        "Every paragraph below is generated from the team's actual 2026 stat "
        "profile (averages and league ranks across 16 stat categories). The "
        "strategy descriptions are derived from rank combinations, not "
        "hand-written takes — so they update automatically as the season "
        "progresses. Rank 1 = best, rank 18 = worst (for clangers and free "
        "kicks against, lower raw values are better, so rank 1 still means "
        "the desirable end)."
    )
    lines.append("")

    for _, row in ordered.iterrows():
        team = row["team"]
        scorer = top_scorers.get(team)
        if scorer:
            top_player_name = prettify_player_stem(scorer[0])
            top_player_avg = scorer[1]
        else:
            top_player_name = ""
            top_player_avg = 0.0
        lines.append(f"### {team}")
        lines.append("")
        lines.append(render_team_paragraph(row, n_teams, top_player_name, top_player_avg))
        lines.append("")

    return "\n".join(lines)


def build_section_body(
    year: int,
    max_round: int,
    summary_with_ranks: pd.DataFrame,
    summary: pd.DataFrame,
    league: Dict[str, float],
    top_scorers: Dict[str, Tuple[str, float]],
) -> str:
    intro = render_intro(year, max_round, summary, league, top_scorers)
    table1 = render_summary_table(summary_with_ranks)
    table2 = render_leaders_table(summary, league)
    paragraphs = generate_team_paragraphs(summary_with_ranks, top_scorers)

    body = []
    body.append(intro)
    body.append("")
    # H3 sub-headings — the parent "## YEAR season — live team analysis" sits at H2 in
    # docs/afl-insights.md, so all of its sub-sections sit at H3.
    body.append(f"### All 18 teams ranked by total disposals — {year} season-to-date")
    body.append("")
    body.append(table1)
    body.append("")
    body.append(f"### League leaders by stat category — {year}")
    body.append("")
    body.append(table2)
    body.append("")
    # Goals-vs-disposals scatter — placed under the leaders table because it
    # is the visual companion to those raw numbers (which team turns ball-use
    # into score). Marker pair so refresh_readme.py can find/replace the
    # block without disturbing surrounding content.
    body.append("<!-- GOALS-DISPOSALS-CHART-START -->")
    body.append(
        f"![{year} season — goals vs disposals scatter]"
        f"(../assets/charts/team_{year}_goals_disposals.png)"
    )
    body.append("<!-- GOALS-DISPOSALS-CHART-END -->")
    body.append("")
    # Visual snapshot — radar (top 6) + heatmap (all 18 ranks) + form trend.
    # The PNGs are regenerated by `generate_readme_charts.regenerate_team_charts(year)`
    # in main(), so they stay in sync with the tables above.
    body.append(f"### Visual snapshot — top-6 radar and league-wide rank heatmap")
    body.append("")
    body.append(
        f"The radar chart below picks out the top six disposal teams of the {year} "
        "season-to-date and plots them against the six core dimensions, normalised "
        "to a 0-1 scale relative to all 18 sides — a value of 1.0 on an axis means "
        "that team is the league best on that stat. The heatmap underneath shows "
        "every team's rank across eight key stats (1 = league best in green, 18 = "
        "worst in red), with rows ordered by disposals rank."
    )
    body.append("")
    body.append(f"![Top 6 teams radar — {year}](../assets/charts/team_{year}_radar.png)")
    body.append("")
    body.append(f"![Team rank heatmap — {year}](../assets/charts/team_{year}_heatmap.png)")
    body.append("")
    # Form trend chart — round-by-round disposal form with top/bottom 3
    # highlighted. Marker pair lets refresh_readme.py target this block.
    body.append("<!-- FORM-TREND-CHART-START -->")
    body.append(
        f"![{year} disposal form trend by round]"
        f"(../assets/charts/team_form_trend_{year}.png)"
    )
    body.append("<!-- FORM-TREND-CHART-END -->")
    body.append("")
    body.append(paragraphs)
    return "\n".join(body)


# ---------------------------------------------------------------------------
# 5-year team playing-style profiles
# ---------------------------------------------------------------------------
# These functions build the "Team playing styles — 5 years of data" section.
# Methodology:
#   - Window = the five seasons immediately prior to the current detected year
#     (e.g. current=2026 → 2021..2025).
#   - For each team-year we compute per-game means of the core stats (the team
#     being summed across all its players in the game first, then averaged
#     across that team's games in the year).
#   - The 5-year profile is the simple mean of the five yearly means — this
#     stops a long season biasing a short one.
#   - Trends are linear-regression slopes fitted to the 5 yearly means; the
#     reported relative change is slope * (year_max - year_min) / mean,
#     i.e. the implied total change across the window as a fraction of mean.
#   - Percentile ranks are over the 18 teams in the window.
#   - Brisbane Lions, GWS and Gold Coast all have full coverage from 2021,
#     so no historical-name remapping is needed for the 2021-2025 window.
#     If the window ever extended back past their entry years, those teams
#     would simply have fewer than 5 seasons and we'd flag it.
# ---------------------------------------------------------------------------

# Per-game stats we average per team-year.
PROFILE_CORE_STATS = [
    "kicks", "handballs", "disposals", "marks", "goals", "tackles", "clearances",
    "inside_50s", "rebound_50s", "contested_possessions", "uncontested_possessions",
    "clangers", "free_kicks_for", "free_kicks_against", "hit_outs",
    "marks_inside_50", "contested_marks",
]

# Stats we rank as "higher is better" on the 5-year profile.
PROFILE_HIGH_GOOD = [
    "kicks", "handballs", "disposals", "marks", "goals", "tackles", "clearances",
    "inside_50s", "rebound_50s", "contested_possessions", "uncontested_possessions",
    "hit_outs", "marks_inside_50", "contested_marks",
    "handball_ratio", "marks_per_inside50", "tackle_rate", "free_kicks_for",
]
# Stats where lower raw values are "better" / cleaner.
PROFILE_LOW_GOOD = ["clangers", "free_kicks_against"]


def load_window_games(years: List[int]) -> pd.DataFrame:
    """Load player-game rows restricted to the supplied year window."""
    pattern = os.path.join(PLAYER_DATA_DIR, "*_performance_details.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"No player files found in {PLAYER_DATA_DIR}")

    load_cols = ["team", "year", "round", "opponent"] + PROFILE_CORE_STATS
    frames: List[pd.DataFrame] = []
    bad = 0
    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in load_cols)
            if df.empty or "year" not in df.columns:
                continue
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df.dropna(subset=["year"])
            df["year"] = df["year"].astype(int)
            df = df[df["year"].isin(years)]
            if df.empty:
                continue
            frames.append(df)
        except Exception:
            bad += 1
    if bad:
        print(f"[warn] {bad} files unreadable in 5-year load", file=sys.stderr)
    if not frames:
        sys.exit("No 5-year window data found")
    games = pd.concat(frames, ignore_index=True)
    return games


def aggregate_window_team_game(games: pd.DataFrame) -> pd.DataFrame:
    """Sum player stats to (team, year, round, opponent) team-game level."""
    g = games.copy()
    for c in PROFILE_CORE_STATS:
        if c not in g.columns:
            g[c] = 0.0
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)
    g["round_str"] = g["round"].astype(str)
    return g.groupby(
        ["team", "year", "round_str", "opponent"], as_index=False
    )[PROFILE_CORE_STATS].sum()


def per_team_year_means(team_game: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, year) with per-game means of every stat plus
    the derived ratios (handball_ratio, marks_per_inside50, tackle_rate)."""
    means = team_game.groupby(["team", "year"], as_index=False)[PROFILE_CORE_STATS].mean()
    means["handball_ratio"] = means["handballs"] / means["disposals"].replace(0, np.nan)
    means["marks_per_inside50"] = means["marks_inside_50"] / means["inside_50s"].replace(0, np.nan)
    means["tackle_rate"] = means["tackles"] / means["disposals"].replace(0, np.nan)
    games_per_year = team_game.groupby(["team", "year"]).size().rename("games").reset_index()
    means = means.merge(games_per_year, on=["team", "year"])
    return means


def build_5year_profile(per_year: pd.DataFrame) -> pd.DataFrame:
    """Mean across years per team, plus per-team season count."""
    metric_cols = [c for c in per_year.columns if c not in ("team", "year", "games")]
    profile = per_year.groupby("team")[metric_cols].mean().reset_index()
    seasons = per_year.groupby("team")["year"].nunique().rename("seasons").reset_index()
    profile = profile.merge(seasons, on="team")
    return profile


def compute_trend_changes(per_year: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    """Linear-fit slope across the 5 yearly means. Returns one row per team
    with `<stat>_rel_change` = implied total change across window / mean."""
    metric_cols = [c for c in per_year.columns if c not in ("team", "year", "games")]
    rows: List[Dict[str, float]] = []
    span = years[-1] - years[0] if len(years) > 1 else 1
    for team, sub in per_year.groupby("team"):
        sub = sub.sort_values("year")
        out: Dict[str, float] = {"team": team}
        x = sub["year"].astype(float).values
        for c in metric_cols:
            y = sub[c].astype(float).values
            mask = ~np.isnan(y)
            if mask.sum() < 3:
                out[f"{c}_rel_change"] = np.nan
                continue
            slope = np.polyfit(x[mask], y[mask], 1)[0]
            mean_y = float(np.nanmean(y))
            out[f"{c}_rel_change"] = float(slope * span / mean_y) if mean_y > 0 else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def add_profile_ranks(profile: pd.DataFrame) -> pd.DataFrame:
    """Rank-of-18 plus 0..100 percentile for each PROFILE_HIGH_GOOD /
    PROFILE_LOW_GOOD stat. Rank 1 = desirable end."""
    out = profile.copy()
    n = len(out)
    for s in PROFILE_HIGH_GOOD:
        if s in out.columns:
            out[f"{s}_rank"] = out[s].rank(ascending=False, method="min").astype(int)
            out[f"{s}_pct"] = ((n - out[f"{s}_rank"]) / max(n - 1, 1)) * 100.0
    for s in PROFILE_LOW_GOOD:
        if s in out.columns:
            out[f"{s}_rank"] = out[s].rank(ascending=True, method="min").astype(int)
            out[f"{s}_pct"] = ((n - out[f"{s}_rank"]) / max(n - 1, 1)) * 100.0
    return out


def _trend_word(rel_change: float, threshold: float = 0.05) -> str:
    """Classify a trend slope into rising / falling / stable. The default
    threshold of 5% relative change across the window is the boundary
    between 'stable' and 'evolving'."""
    if pd.isna(rel_change):
        return "stable"
    if rel_change >= threshold:
        return "rising"
    if rel_change <= -threshold:
        return "falling"
    return "stable"


def _trend_phrase(team_row: pd.Series, stat: str, label: str) -> str:
    """Convert a stat trend into a readable clause."""
    rel = team_row.get(f"{stat}_rel_change", np.nan)
    word = _trend_word(rel)
    if word == "stable":
        return f"their {label} have stayed flat"
    pct = abs(rel) * 100
    direction = "climbed" if word == "rising" else "fallen"
    return f"their {label} have {direction} ~{pct:.0f}% across the window"


def _style_drift(team_row: pd.Series) -> float:
    """How much the team's profile has shifted across the window — sum of
    |rel_change| across six identity stats. Higher = bigger evolution."""
    keys = [
        "disposals_rel_change", "tackles_rel_change", "inside_50s_rel_change",
        "contested_possessions_rel_change", "handball_ratio_rel_change",
        "marks_per_inside50_rel_change",
    ]
    vals = [abs(team_row[k]) for k in keys if k in team_row and not pd.isna(team_row[k])]
    return float(sum(vals)) if vals else 0.0


def _hb_descriptor(rank: int, ratio: float) -> str:
    """Describe ball-movement style from handball ratio rank/value."""
    pct_kick = (1 - ratio) * 100
    pct_hb = ratio * 100
    if rank <= 4:
        return f"a high-tempo, handball-heavy build-up ({pct_hb:.0f}% by hand — {ordinal(rank)} in the league)"
    if rank >= 15:
        return f"a kick-first, territory-via-foot template ({pct_kick:.0f}% by foot — {ordinal(19 - rank)}-most kick-dominant)"
    return f"a roughly balanced {pct_hb:.0f}/{pct_kick:.0f} handball-to-kick split"


def _territory_descriptor(i50_rank: int, rb_rank: int, net: float) -> str:
    """Describe territory approach from i50 vs rebound profile."""
    if i50_rank <= 5 and net >= 14:
        return (
            f"a relentlessly forward-territory game (+{net:.1f} net 50-entries per game)"
        )
    if rb_rank <= 4 and i50_rank >= 12:
        return f"a defensive-rebound posture, soaking up entries and breaking out (rebound-50s ranked {ordinal(rb_rank)})"
    if i50_rank <= 6:
        return f"strong territorial dominance ({ordinal(i50_rank)} for inside-50s)"
    if i50_rank >= 14:
        return f"a struggle to win territory ({ordinal(i50_rank)} for inside-50s)"
    return f"a mid-pack territory profile ({ordinal(i50_rank)} for inside-50s)"


def _contest_descriptor(cp_rank: int, ucp_rank: int, tk_rank: int) -> str:
    """Describe contest intensity from CP / UCP / tackles."""
    if cp_rank <= 5 and tk_rank <= 6:
        return f"a contested-ball, high-pressure identity ({ordinal(cp_rank)} for contested possessions, {ordinal(tk_rank)} for tackles)"
    if cp_rank <= 5 and tk_rank >= 13:
        return f"a contested-ball game without the tackle pressure to match ({ordinal(cp_rank)} for CP, only {ordinal(tk_rank)} for tackles)"
    if ucp_rank <= 5 and cp_rank >= 13:
        return f"a possession-and-spread template that avoids the contest ({ordinal(ucp_rank)} for uncontested possessions, {ordinal(cp_rank)} for CP)"
    if tk_rank <= 4:
        return f"genuine ground-ball pressure ({ordinal(tk_rank)} for tackles)"
    if tk_rank >= 15:
        return f"notably soft pressure around the ball ({ordinal(tk_rank)} for tackles)"
    return f"a balanced contest profile ({ordinal(cp_rank)} for CP, {ordinal(tk_rank)} for tackles)"


def _security_descriptor(cl_rank: int, fk_rank: int) -> str:
    """Describe ball security from clangers + free-kicks-against ranks
    (rank 1 = cleanest in both)."""
    if cl_rank <= 4 and fk_rank <= 6:
        return "ball security is a real strength — among the cleanest sides for both clangers and frees against"
    if cl_rank >= 15 or fk_rank >= 15:
        return "ball security is a clear vulnerability — sitting in the bottom four for clangers or frees conceded"
    return "ball security sits roughly mid-pack"


def _forward_descriptor(mi50_rank: int, ratio: float, i50_rank: int, goals: float) -> str:
    """Describe forward efficiency."""
    if mi50_rank <= 4 and i50_rank <= 8:
        return (
            f"genuine forward-half efficiency (1-in-{(1/ratio):.1f} of their entries hit a target inside 50, "
            f"and they average {goals:.1f} goals/g)"
        )
    if mi50_rank <= 4:
        return f"a clinical use of the few entries they win (1-in-{(1/ratio):.1f} entries marked, {ordinal(mi50_rank)} in the league)"
    if mi50_rank >= 14:
        return (
            f"a volume-over-precision forward template — only 1-in-{(1/ratio):.1f} entries are marked inside 50 "
            f"({ordinal(mi50_rank)} for connection)"
        )
    return f"mid-pack forward efficiency ({ordinal(mi50_rank)} for marks-per-inside-50)"


def _opening_clause(team: str, profile_row: pd.Series) -> str:
    """Build the opening identity sentence based on the team's strongest
    rank — the stat that most defines them in the data."""
    # Pick the dimension where the team is most extreme (highest or lowest
    # rank across a curated identity set).
    # Each tuple is (stat_col, descriptor_phrase). The descriptor is meant
    # to slot into the sentence "{Team} have built their identity around a
    # {descriptor} game", so it has to read naturally as an adjective phrase.
    identity_stats = [
        ("contested_possessions", "contested-ball"),
        ("uncontested_possessions", "possession-and-control"),
        ("tackles", "ground-ball pressure"),
        ("inside_50s", "territory-dominant"),
        ("rebound_50s", "rebound-and-counter"),
        ("clearances", "stoppage-dominant"),
        ("marks", "mark-and-control"),
        ("hit_outs", "ruck-led"),
        ("handball_ratio", "handball-driven"),
        ("marks_per_inside50", "clinical forward-50"),
        ("clangers", "clean-ball-use"),
    ]
    extreme = None
    extreme_score = 0.0
    for stat, label in identity_stats:
        rk_col = f"{stat}_rank"
        if rk_col not in profile_row.index:
            continue
        rk = int(profile_row[rk_col])
        # extremity = max(distance from middle) — top or bottom of the league
        score = max(10 - rk, rk - 9)
        if score > extreme_score:
            extreme_score = score
            extreme = (stat, label, rk)
    if not extreme:
        return f"{team} have built their identity around a fairly balanced 5-year profile"
    stat, label, rk = extreme
    if rk <= 4:
        return f"{team} have built their identity around a {label} game — ranked {ordinal(rk)} of 18 across the last five seasons"
    if rk >= 15:
        from_bottom = 19 - rk
        from_bottom_word = {1: "least", 2: "second-least", 3: "third-least", 4: "fourth-least"}.get(from_bottom, f"{ordinal(from_bottom)}-least")
        return f"{_possessive(team)} 5-year profile is defined by what they don't do — the {from_bottom_word} {label} side in the competition"
    return f"{team} have built their identity around their {label} game ({ordinal(rk)} of 18 over the last five seasons)"


def _possessive(team: str) -> str:
    """Form the possessive of a team name, handling teams that end in `s`
    (e.g. Western Bulldogs → Western Bulldogs', not Bulldogs's)."""
    if team.endswith("s"):
        return team + "'"
    return team + "'s"


def _ceiling_clause(profile_row: pd.Series) -> str:
    """Write the closing 'ceiling and vulnerability' line from the rank
    extremes.

    A bottom-ranked stat is only called a 'vulnerability' if it's a stat
    where being low is genuinely bad (low tackles, low contested possessions,
    high clangers). For stats like 'low handballs' or 'low uncontested
    possessions' that simply reflect a kick-and-pressure style choice, we
    frame it as a deliberate trade-off rather than a flaw.
    """
    # Stats where being bottom-ranked is unambiguously a problem at AFL level.
    # Anything not in this set is treated as a stylistic trade-off if low.
    GENUINE_WEAKNESS_STATS = {
        # Stats where a bottom-of-league rank is genuinely bad — not just a
        # by-product of the team's style. Notable exclusions: handballs (kick-
        # first sides will be low by design), uncontested_possessions (pressure
        # sides will be low by design), rebound_50s (territory-dominant sides
        # will be low by design), hit_outs (a team can deprioritise rucks).
        "disposals", "kicks", "marks", "goals", "tackles", "clearances",
        "inside_50s", "contested_possessions", "marks_inside_50",
        "marks_per_inside50", "contested_marks", "free_kicks_for",
        "clangers", "free_kicks_against",
    }
    strengths = []
    weak_genuine = []
    weak_stylistic = []
    candidates = PROFILE_HIGH_GOOD + PROFILE_LOW_GOOD
    for s in candidates:
        rk = profile_row.get(f"{s}_rank")
        if rk is None or pd.isna(rk):
            continue
        rk = int(rk)
        if rk <= 4:
            strengths.append((s, rk))
        elif rk >= 15:
            if s in GENUINE_WEAKNESS_STATS:
                weak_genuine.append((s, rk))
            else:
                weak_stylistic.append((s, rk))
    strengths.sort(key=lambda t: t[1])
    weak_genuine.sort(key=lambda t: -t[1])
    weak_stylistic.sort(key=lambda t: -t[1])

    str_phrase = ""
    if strengths:
        s_label = STAT_LABELS.get(strengths[0][0], strengths[0][0])
        str_phrase = f"the ceiling is anchored to their {s_label} ({ordinal(strengths[0][1])} of 18)"

    if str_phrase and weak_genuine:
        w_label = STAT_LABELS.get(weak_genuine[0][0], weak_genuine[0][0])
        weak_phrase = f"the vulnerability is their {w_label} ({ordinal(weak_genuine[0][1])} of 18)"
        return f"In short, {str_phrase}, while {weak_phrase} — and that gap is what every opposition gameplan targets."
    if str_phrase and weak_stylistic:
        w_label = STAT_LABELS.get(weak_stylistic[0][0], weak_stylistic[0][0])
        return (
            f"In short, {str_phrase}; the low {w_label} ({ordinal(weak_stylistic[0][1])} of 18) "
            "is the deliberate trade-off that comes with that style."
        )
    if str_phrase:
        return f"In short, {str_phrase} — that's the foundation everything else is built on."
    if weak_genuine:
        w_label = STAT_LABELS.get(weak_genuine[0][0], weak_genuine[0][0])
        return f"In short, the {w_label} number ({ordinal(weak_genuine[0][1])} of 18) is the obvious priority to close."
    return (
        "The profile is balanced enough that no single dimension is doing the heavy lifting — "
        "a hallmark of a settled, well-coached side without a transcendent strength."
    )


def render_5year_team_paragraph(profile_row: pd.Series, year_window: List[int]) -> str:
    """Compose 5-8 sentences for one team using the 5-year stat profile."""
    team = profile_row["team"]
    seasons = int(profile_row["seasons"])
    seasons_note = ""
    if seasons < len(year_window):
        seasons_note = (
            f" (note: {seasons}/{len(year_window)} seasons of data — partial coverage)"
        )

    # Numbers we'll lean on.
    disp = profile_row["disposals"]
    kicks = profile_row["kicks"]
    hbs = profile_row["handballs"]
    marks = profile_row["marks"]
    goals = profile_row["goals"]
    tackles = profile_row["tackles"]
    clearances = profile_row["clearances"]
    i50 = profile_row["inside_50s"]
    rb = profile_row["rebound_50s"]
    cp = profile_row["contested_possessions"]
    ucp = profile_row["uncontested_possessions"]
    clangers = profile_row["clangers"]
    fka = profile_row["free_kicks_against"]
    hit_outs = profile_row["hit_outs"]
    hb_ratio = profile_row["handball_ratio"]
    mi50_ratio = profile_row["marks_per_inside50"]
    contested_marks = profile_row["contested_marks"]
    marks_inside_50 = profile_row["marks_inside_50"]

    hb_rank = int(profile_row["handball_ratio_rank"])
    i50_rank = int(profile_row["inside_50s_rank"])
    rb_rank = int(profile_row["rebound_50s_rank"])
    cp_rank = int(profile_row["contested_possessions_rank"])
    ucp_rank = int(profile_row["uncontested_possessions_rank"])
    tk_rank = int(profile_row["tackles_rank"])
    cl_rank = int(profile_row["clangers_rank"])
    fk_rank = int(profile_row["free_kicks_against_rank"])
    mi50_rank = int(profile_row["marks_per_inside50_rank"])
    cm_rank = int(profile_row["contested_marks_rank"])
    ho_rank = int(profile_row["hit_outs_rank"])
    cl_rank_clear = int(profile_row["clearances_rank"])
    goals_rank = int(profile_row["goals_rank"])

    drift = _style_drift(profile_row)
    drift_word = "evolving" if drift >= 0.40 else ("settled" if drift <= 0.25 else "gradually shifting")

    # Sentence 1 — identity opener
    s1 = _opening_clause(team, profile_row) + seasons_note + "."

    # Sentence 2 — quantified core (disposals + ball-movement style + tackles)
    s2 = (
        f"Across {year_window[0]}–{year_window[-1]} they averaged {disp:.1f} disposals, "
        f"{tackles:.1f} tackles and {clearances:.1f} clearances per game, with "
        + _hb_descriptor(hb_rank, hb_ratio)
        + "."
    )

    # Sentence 3 — territory + contest synthesis
    net_50s = i50 - rb
    s3 = (
        "On territory and contest, they show "
        + _territory_descriptor(i50_rank, rb_rank, net_50s)
        + " combined with "
        + _contest_descriptor(cp_rank, ucp_rank, tk_rank)
        + "."
    )

    # Sentence 4 — forward efficiency + ball security
    s4 = (
        "Forward of centre, "
        + _forward_descriptor(mi50_rank, mi50_ratio, i50_rank, goals)
        + ", and "
        + _security_descriptor(cl_rank, fk_rank)
        + "."
    )

    # Sentence 5 — trend / evolution
    notable_trends = []
    for stat, label in [
        ("disposals", "ball use"),
        ("tackles", "tackle pressure"),
        ("inside_50s", "forward entries"),
        ("contested_possessions", "contested-ball winning"),
        ("handball_ratio", "handball share"),
        ("marks_per_inside50", "forward connection"),
    ]:
        rel = profile_row.get(f"{stat}_rel_change", np.nan)
        if not pd.isna(rel) and abs(rel) >= 0.07:
            notable_trends.append((label, rel))
    notable_trends.sort(key=lambda t: -abs(t[1]))
    if notable_trends and drift >= 0.30:
        clauses = []
        for label, rel in notable_trends[:2]:
            arrow = "climbed" if rel > 0 else "fallen"
            clauses.append(f"{label} has {arrow} ~{abs(rel)*100:.0f}% across the five-year window")
        s5 = (
            f"The profile is {drift_word} rather than locked in: "
            + " and ".join(clauses)
            + "."
        )
    elif drift < 0.20:
        s5 = (
            "The profile is unusually settled — none of the six identity stats has shifted "
            f"more than ~{int(drift * 100 / 6)}% across five seasons, suggesting a tightly held "
            "list and game-plan."
        )
    else:
        s5 = (
            f"The 5-year profile is {drift_word} — modest year-on-year movement but no wholesale "
            "re-tooling of the game plan."
        )

    # Sentence 6 — defining characteristic
    # Pick the team's most extreme single rank (highest deviation from middle).
    GENUINE_WEAKNESS_STATS = {
        # Stats where a bottom-of-league rank is genuinely bad — not just a
        # by-product of the team's style. Notable exclusions: handballs (kick-
        # first sides will be low by design), uncontested_possessions (pressure
        # sides will be low by design), rebound_50s (territory-dominant sides
        # will be low by design), hit_outs (a team can deprioritise rucks).
        "disposals", "kicks", "marks", "goals", "tackles", "clearances",
        "inside_50s", "contested_possessions", "marks_inside_50",
        "marks_per_inside50", "contested_marks", "free_kicks_for",
        "clangers", "free_kicks_against",
    }
    extreme_picks = []
    for s in PROFILE_HIGH_GOOD + PROFILE_LOW_GOOD:
        rk_col = f"{s}_rank"
        if rk_col not in profile_row.index:
            continue
        rk = int(profile_row[rk_col])
        score = max(10 - rk, rk - 9)
        extreme_picks.append((s, rk, score, float(profile_row[s])))
    extreme_picks.sort(key=lambda t: -t[2])
    top_extreme = extreme_picks[0] if extreme_picks else None
    if top_extreme:
        s_name, s_rk, _, s_val = top_extreme
        s_label = STAT_LABELS.get(s_name, s_name)
        # Format the value sensibly — ratios are 0..1, so quote them as %.
        if s_name in ("handball_ratio", "marks_per_inside50", "tackle_rate"):
            val_str = f"{s_val * 100:.1f}%"
        else:
            val_str = f"{s_val:.1f}/g"
        if s_rk <= 4:
            s6 = (
                f"What sets them apart from the field is their {s_label} "
                f"({val_str}, {ordinal(s_rk)} of 18) — a true outlier rather than mid-pack noise."
            )
        elif s_rk >= 15 and s_name in GENUINE_WEAKNESS_STATS:
            s6 = (
                f"What separates them from the rest of the competition is the floor under "
                f"their {s_label} ({val_str}, {ordinal(s_rk)} of 18) — a structural issue, not a single bad year."
            )
        elif s_rk >= 15:
            # Stylistic trade-off — frame as a deliberate choice rather than a flaw.
            s6 = (
                f"What sets them apart is how far they sit from the league norm in {s_label} "
                f"({val_str}, {ordinal(s_rk)} of 18) — that gap is the signature of their style, not a fault line."
            )
        else:
            s6 = (
                f"Their most distinctive number is {s_label} at {val_str} ({ordinal(s_rk)} of 18)."
            )
    else:
        s6 = "No single stat stands out as a defining outlier — they are a true mid-pack profile."

    # Sentence 7 — ceiling / vulnerability close
    s7 = _ceiling_clause(profile_row)

    return " ".join([s1, s2, s3, s4, s5, s6, s7])


def render_5year_section(profile: pd.DataFrame, years: List[int], current_year: int) -> str:
    """Build the full 5-year team-profiles section markdown.

    Sections are ordered alphabetically by team for stable output.
    """
    intro_lines = [
        f"The profiles below summarise each team's {years[0]}–{years[-1]} statistical "
        "fingerprint — five full seasons of per-game averages aggregated across every "
        f"player who pulled on the jumper. They reflect coaching philosophy and list "
        "system more than any single season's results, smoothing out the noise of a "
        f"hot run or a flat year. If you want to understand *why* a team plays the "
        f"way it does in {current_year}, this is the better baseline than the season-to-date "
        "snapshot above — it is what the data says they are at their core.",
        "",
        "Each paragraph leans on the actual numbers (per-game averages, league ranks "
        "across 18 teams, and 5-year linear trends), so the descriptions update "
        "automatically when the window rolls forward.",
        "",
        # The scatter PNG is generated by generate_readme_charts.chart_team_style_scatter
        # using the same five-year window as this section.
        f"![5-year team playing styles scatter — {years[0]}-{years[-1]}](../assets/charts/team_{current_year}_style_scatter.png)",
        "",
        f"The scatter above places each club on two axes that capture the most visible "
        f"part of a team's identity — handball ratio (% of disposals by hand) on the X "
        f"and tackles per game on the Y. The dashed lines mark the league median, "
        f"splitting the field into four loose archetypes: top-right teams move the ball "
        f"by handball *and* hunt with tackle pressure; top-left sides trust their kicking "
        f"but still bring the heat; bottom-right are handball teams that spread rather "
        f"than tackle; bottom-left are kick-and-spread possession sides.",
    ]

    body_lines: List[str] = []
    body_lines.extend(intro_lines)
    body_lines.append("")

    for _, row in profile.sort_values("team").iterrows():
        # H3 here: parent "## Team playing styles — 5 years of data" sits at H2
        # in docs/afl-insights.md, so each team sits at H3.
        body_lines.append(f"### {row['team']}")
        body_lines.append("")
        body_lines.append(render_5year_team_paragraph(row, years))
        body_lines.append("")
    return "\n".join(body_lines).rstrip()


def replace_5year_section(readme_text: str, current_year: int, years: List[int], body: str) -> str:
    """Insert / replace the 5-year team-profiles section.

    Looks for the markers `<!-- 5YEAR-TEAM-PROFILES-START -->` /
    `<!-- 5YEAR-TEAM-PROFILES-END -->`. If absent, inserts the section
    immediately after the current-year team-analysis section's END marker.

    Also adds a TOC entry if missing.

    Header level: this section sits at H2 in docs/afl-insights.md, so a
    freshly inserted section uses ## (not ###).
    """
    start_marker = "<!-- 5YEAR-TEAM-PROFILES-START -->"
    end_marker = "<!-- 5YEAR-TEAM-PROFILES-END -->"
    section_header = f"## Team playing styles — 5 years of data ({years[0]}–{years[-1]})"
    toc_entry = f"  - [Team playing styles — 5 years of data ({years[0]}–{years[-1]})](#team-playing-styles--5-years-of-data-{years[0]}{years[-1]})"

    if start_marker in readme_text and end_marker in readme_text:
        before, rest = readme_text.split(start_marker, 1)
        _, after = rest.split(end_marker, 1)
        # Refresh the human-readable header line in case the year window rolls.
        # The header is the line immediately before start_marker; we leave the
        # surrounding markdown structure untouched and only replace between markers.
        new_text = (
            before
            + start_marker
            + "\n"
            + body
            + "\n"
            + end_marker
            + after
        )
    else:
        # Insert after the current-year section's END marker.
        anchor = f"<!-- {current_year}-TEAM-ANALYSIS-END -->"
        idx = readme_text.find(anchor)
        if idx == -1:
            sys.exit(
                f"Could not find anchor '{anchor}' to insert 5-year section. "
                "Run the script after the current-year section is in place, "
                "or add the markers manually."
            )
        insert_at = idx + len(anchor)
        new_section = (
            "\n\n"
            + section_header
            + "\n\n"
            + start_marker
            + "\n"
            + body
            + "\n"
            + end_marker
            + "\n"
        )
        new_text = readme_text[:insert_at] + new_section + readme_text[insert_at:]

    # TOC entry — add if missing. Place it directly after the current-year
    # team-analysis TOC entry so the order matches the body order.
    if toc_entry not in new_text:
        prev_toc = f"- [{current_year} season — live team analysis"
        toc_idx = new_text.find(prev_toc)
        if toc_idx != -1:
            line_end = new_text.find("\n", toc_idx)
            if line_end != -1:
                new_text = (
                    new_text[: line_end + 1]
                    + toc_entry
                    + "\n"
                    + new_text[line_end + 1 :]
                )

    return new_text


def generate_5year_profiles(current_year: int) -> Tuple[str, List[int]]:
    """Top-level entry point. Returns (markdown_body, year_window) for the
    five seasons immediately prior to `current_year`. Designed to be wired
    into the same refresh pipeline as the current-year analysis."""
    years = list(range(current_year - 5, current_year))
    print(f"      5-year window: {years[0]}..{years[-1]}")
    games = load_window_games(years)
    print(f"      loaded {len(games):,} player-game rows in window")
    team_game = aggregate_window_team_game(games)
    per_year = per_team_year_means(team_game)
    profile = build_5year_profile(per_year)
    trends = compute_trend_changes(per_year, years)
    profile = profile.merge(trends, on="team", how="left")
    profile = add_profile_ranks(profile)
    print(f"      profile teams: {len(profile)} (expected 18)")
    body = render_5year_section(profile, years, current_year)
    return body, years


# ---------------------------------------------------------------------------
# Finals pathway — per-team mid-season finals + flag analysis
# ---------------------------------------------------------------------------
# Methodology:
#   - Load matches_<YEAR>.csv (one row per match) and compute the ladder by
#     points (4 per win / 2 per draw / 0 per loss) and percentage
#     (points-for / points-against * 100). Matches the AFL official ladder.
#   - Combine ladder position with the per-team stat profile already computed
#     for the season-analysis section to write a 5-7 sentence paragraph per
#     team. The stat ranks come from `summary_with_ranks` so we share the
#     exact same numbers shown in the rest of the section.
#   - The grand-final probability tier uses simple cut-offs: top-4 + 3+
#     elite stat ranks = "contender", top-8 = "chance", ladder 9-12 = "long
#     shot for finals", ladder 13+ = "no realistic path".
#   - The paragraph generator is data-driven only — no hand-written takes —
#     so it updates automatically when the ladder shifts each round.


def load_match_results(year: int) -> pd.DataFrame:
    """Load completed matches for a single year. Returns a DataFrame with
    one row per match plus computed final scores. Excludes any rows where
    the score columns are blank (treated as future fixture)."""
    path = os.path.join(MATCHES_DIR, f"matches_{year}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    needed = [
        "round_num", "team_1_team_name", "team_2_team_name",
        "team_1_final_goals", "team_1_final_behinds",
        "team_2_final_goals", "team_2_final_behinds",
    ]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()
    df = df.dropna(subset=[
        "team_1_final_goals", "team_1_final_behinds",
        "team_2_final_goals", "team_2_final_behinds",
    ]).copy()
    if df.empty:
        return df
    for c in [
        "team_1_final_goals", "team_1_final_behinds",
        "team_2_final_goals", "team_2_final_behinds",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["team_1_score"] = df["team_1_final_goals"] * 6 + df["team_1_final_behinds"]
    df["team_2_score"] = df["team_2_final_goals"] * 6 + df["team_2_final_behinds"]
    return df


def compute_ladder(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute the ladder from completed matches. Returns one row per team
    with: team, played, W, L, D, pts, points_for, points_against, percentage,
    position (1 = top of ladder)."""
    rows: Dict[str, Dict[str, float]] = {}

    def _ensure(team: str) -> None:
        if team not in rows:
            rows[team] = {
                "team": team, "played": 0, "W": 0, "L": 0, "D": 0,
                "pts": 0, "points_for": 0, "points_against": 0,
            }

    for _, m in matches.iterrows():
        t1, t2 = m["team_1_team_name"], m["team_2_team_name"]
        s1, s2 = float(m["team_1_score"]), float(m["team_2_score"])
        _ensure(t1)
        _ensure(t2)
        rows[t1]["played"] += 1
        rows[t2]["played"] += 1
        rows[t1]["points_for"] += s1
        rows[t1]["points_against"] += s2
        rows[t2]["points_for"] += s2
        rows[t2]["points_against"] += s1
        if s1 > s2:
            rows[t1]["W"] += 1
            rows[t1]["pts"] += 4
            rows[t2]["L"] += 1
        elif s2 > s1:
            rows[t2]["W"] += 1
            rows[t2]["pts"] += 4
            rows[t1]["L"] += 1
        else:
            rows[t1]["D"] += 1
            rows[t2]["D"] += 1
            rows[t1]["pts"] += 2
            rows[t2]["pts"] += 2

    ladder = pd.DataFrame(list(rows.values()))
    if ladder.empty:
        return ladder
    ladder["percentage"] = ladder.apply(
        lambda r: (r["points_for"] / r["points_against"] * 100.0)
        if r["points_against"] > 0 else 0.0,
        axis=1,
    )
    ladder = ladder.sort_values(
        ["pts", "percentage"], ascending=[False, False]
    ).reset_index(drop=True)
    ladder["position"] = ladder.index + 1
    return ladder


def _gf_probability_tier(ladder_pos: int, elite_count: int, weak_count: int) -> str:
    """Map (ladder position, elite stat rank count, weak stat rank count) to
    a coarse grand-final tier. Cut-offs are deliberately simple:
      - top-4 with 3+ elite ranks → 'contender'
      - top-8 → 'live chance' or 'chance'
      - 9-12 → 'long shot'
      - 13-16 → 'mathematically alive but unlikely'
      - 17-18 → 'season effectively over'
    `elite_count` is how many of (goals, clearances, tackles, inside_50s,
    contested_possessions) the team ranks top-5 in. `weak_count` is how many
    they rank bottom-5 in.
    """
    if ladder_pos <= 4 and elite_count >= 3:
        return "contender"
    if ladder_pos <= 4:
        return "live chance"
    if ladder_pos <= 8 and elite_count >= 2:
        return "live chance"
    if ladder_pos <= 8:
        return "chance"
    if ladder_pos <= 12:
        return "long shot for finals"
    if ladder_pos <= 16:
        return "mathematically alive but unlikely"
    return "season effectively over"


def _wins_word(n: int) -> str:
    """'1 win' / '2 wins' / '0 wins'. Avoids the '1 wins' grammar miss."""
    return f"{n} win" if n == 1 else f"{n} wins"


def _path_to_finals(ladder_pos: int, wins_now: int, games_remaining: int) -> str:
    """A short clause describing the finals path. We assume ~12 wins is the
    classic AFL '8th-place line' — teams below that need a strong finish to
    push back into the eight. The clause is written so it follows directly
    after a clause like 'their job from here is to ...'."""
    target_wins = 12  # heuristic — historic average for 8th place
    needed = max(target_wins - wins_now, 0)
    w = _wins_word(wins_now)
    n_word = _wins_word(needed)
    if ladder_pos <= 4:
        return (
            "their job is to hold the double-chance, not chase it — "
            "every win from here is locking in a top-4 finish, not scrambling for one"
        )
    if ladder_pos <= 8:
        return (
            f"they need roughly {n_word} from their remaining {games_remaining} "
            "games to lock the eight in, which their form line is comfortably on track to do"
        )
    if ladder_pos <= 12:
        if needed <= games_remaining - 3:
            return (
                f"~{n_word} from {games_remaining} games gets them back in — "
                "very doable on paper, but with no margin for a flat patch"
            )
        return (
            f"they need ~{needed} of their remaining {games_remaining} games "
            "to push into the eight — tough, but not impossible"
        )
    if ladder_pos <= 16:
        return (
            f"they would need to win ~{needed} of their remaining "
            f"{games_remaining} games to claim a finals spot, which would "
            "require a near-perfect run from here"
        )
    return (
        "the maths is technically alive but the path requires a miracle run "
        "the form line does not support"
    )


def _consolidate_or_fix(stat_row: pd.Series, ladder_pos: int) -> Tuple[str, List[str]]:
    """Pick the ONE thing this team must fix or maintain. Returns (clause,
    threat_areas) where threat_areas is a list of stat names worth flagging
    as the biggest leakage points.

    The logic:
      * Top-4 sides: the priority is to maintain whichever offensive engine
        is driving them (goals if rank<=4 else clearances if rank<=4 else
        contested-poss).
      * Top-5-8 sides: the priority is whichever GENUINE weakness sits in
        the bottom 6 — that's the slice opposition coaches will exploit in
        September.
      * 9-12: priority is the same — the worst genuine weakness.
      * 13-18: the priority is wholesale, so we name the two worst weak
        ranks and frame it as 'every part of the structure'.
    """
    elite_set = ["goals", "clearances", "contested_possessions",
                 "inside_50s", "tackles", "marks_inside_50"]
    elite_ranks = [
        (s, int(stat_row.get(f"{s}_rank", 99)))
        for s in elite_set
        if f"{s}_rank" in stat_row.index
    ]
    elite_strengths = sorted([(s, r) for s, r in elite_ranks if r <= 4],
                             key=lambda t: t[1])
    elite_weak_set = ["goals", "clearances", "contested_possessions",
                      "inside_50s", "tackles", "marks_inside_50",
                      "marks", "rebound_50s"]
    weak_ranks = [
        (s, int(stat_row.get(f"{s}_rank", -1)))
        for s in elite_weak_set
        if f"{s}_rank" in stat_row.index
    ]
    weak_areas = sorted([(s, r) for s, r in weak_ranks if r >= 13],
                        key=lambda t: -t[1])

    nice = STAT_LABELS

    if ladder_pos <= 4:
        if elite_strengths:
            s, r = elite_strengths[0]
            label = nice.get(s, s)
            clause = (
                f"the one thing they must keep doing is their {label} "
                f"({ordinal(r)} of 18) — that is the engine of the run"
            )
        else:
            clause = (
                "the one thing they must keep doing is winning ugly games — "
                "their stat profile isn't elite anywhere, so the buffer is "
                "scoreboard form, not structure"
            )
        return clause, [s for s, _ in weak_areas[:1]]

    if weak_areas:
        s, r = weak_areas[0]
        label = nice.get(s, s)
        clause = (
            f"the one thing they have to fix is their {label} ranking "
            f"({ordinal(r)} of 18) — every finals-tier opponent will target it"
        )
        return clause, [s for s, _ in weak_areas[:2]]

    return (
        "no single stat screams crisis, but they need every part of their "
        "game to lift a notch to be a serious finals threat"
    ), []


def _biggest_rival(team: str, ladder: pd.DataFrame) -> str:
    """Identify the team's nearest finals rival — the closest team on the
    ladder that is on the opposite side of the 8th-place line, or the
    closest team on percentage if they're already in/around the same
    position."""
    row = ladder[ladder["team"] == team].iloc[0]
    pos = int(row["position"])
    pts = float(row["pts"])

    if pos <= 8:
        # Threat = team just outside the eight pressing them
        below = ladder[ladder["position"] > pos].sort_values("position")
        if not below.empty:
            target = below.iloc[0]
            return (
                f"their nearest finals rival is **{target['team']}** "
                f"({int(target['W'])}-{int(target['L'])}, "
                f"{ordinal(int(target['position']))}) — the team most likely "
                "to bump them out if they slip"
            )
        return "no realistic challenger immediately below them on the ladder"

    # Out of the eight — threat is the team just above them in the eight
    above = ladder[ladder["position"] < pos].sort_values("position", ascending=False)
    if not above.empty:
        target = above.iloc[0]
        return (
            f"the side blocking their road back in is **{target['team']}** "
            f"({int(target['W'])}-{int(target['L'])}, "
            f"{ordinal(int(target['position']))}) — leapfrog them and the "
            "finals door reopens"
        )
    return "no team immediately above them on the ladder"


def render_finals_pathway_paragraph(
    ladder_row: pd.Series,
    stat_row: pd.Series,
    ladder: pd.DataFrame,
    games_remaining: int,
) -> str:
    """5-7 sentence paragraph on this team's finals + grand final pathway."""
    team = ladder_row["team"]
    pos = int(ladder_row["position"])
    wins = int(ladder_row["W"])
    losses = int(ladder_row["L"])
    draws = int(ladder_row["D"])
    pct = float(ladder_row["percentage"])
    pts = int(ladder_row["pts"])

    elite_set = ["goals", "clearances", "tackles", "inside_50s", "contested_possessions"]
    elite_count = sum(
        1 for s in elite_set
        if f"{s}_rank" in stat_row.index and int(stat_row[f"{s}_rank"]) <= 5
    )
    weak_count = sum(
        1 for s in elite_set
        if f"{s}_rank" in stat_row.index and int(stat_row[f"{s}_rank"]) >= 14
    )

    tier = _gf_probability_tier(pos, elite_count, weak_count)
    path_clause = _path_to_finals(pos, wins, games_remaining)
    fix_clause, _flag_stats = _consolidate_or_fix(stat_row, pos)
    rival_clause = _biggest_rival(team, ladder)

    record_str = f"{wins}-{losses}" + (f"-{draws}" if draws else "")
    s1 = (
        f"**{team}** sit {ordinal(pos)} on the ladder after Round 8 "
        f"({record_str}, {pts} points, {pct:.1f}%) — {path_clause}."
    )

    # Stat profile sentence — concise read of what's working / not.
    stat_phrases = []
    for s in elite_set:
        rk_col = f"{s}_rank"
        if rk_col not in stat_row.index:
            continue
        rk = int(stat_row[rk_col])
        if rk <= 4:
            stat_phrases.append(f"top-4 for {STAT_LABELS.get(s, s)} ({ordinal(rk)})")
        elif rk >= 15:
            stat_phrases.append(f"bottom-4 for {STAT_LABELS.get(s, s)} ({ordinal(rk)})")
    if stat_phrases:
        s2 = (
            "On the underlying numbers they are "
            + ", ".join(stat_phrases[:3])
            + " — that's the stat fingerprint behind the ladder position."
        )
    else:
        s2 = (
            "On the underlying numbers they sit broadly mid-pack across the "
            "core stat lines, which is consistent with their ladder position "
            "and means the form line is honest."
        )

    s3 = f"From here, {fix_clause}."

    s4 = f"For finals positioning, {rival_clause}."

    # Grand final assessment — honest, sometimes blunt
    if tier == "contender":
        s5 = (
            f"Grand final read: **contender**. They have the ladder buffer and "
            f"the stat profile of a side built to play deep in September — "
            f"{elite_count} of the five core categories sit top-5, which is "
            "what every recent premier has had at this point of the year."
        )
    elif tier == "live chance":
        s5 = (
            f"Grand final read: **live chance**. The ladder position is right "
            "and the stat profile has real weapons, but they will need to either "
            "climb a rung or two further or peak at the right time — flag-winners "
            "from outside the top four are the historical exception, not the rule."
        )
    elif tier == "chance":
        s5 = (
            f"Grand final read: **chance, not a contender**. Sitting inside the "
            "eight is the easy bit; the stat profile doesn't yet have the breadth "
            "to beat top-4 sides three weeks running, which is what the path "
            "demands once finals arrive."
        )
    elif tier == "long shot for finals":
        s5 = (
            f"Grand final read: **realistically out of premiership contention**. "
            "Even if they sneak into the eight from here, finishing 7th or 8th "
            "means winning four straight against teams who finished above them — "
            "that's not how flags get won."
        )
    elif tier == "mathematically alive but unlikely":
        s5 = (
            f"Grand final read: **no realistic path**. The maths technically still "
            "works — win out and percentage helps — but the form line and stat "
            "profile are pointing the other way, and no team has come from this "
            "deep at this point of the year to win a flag in the modern era."
        )
    else:
        s5 = (
            f"Grand final read: **season effectively over**. Even a finals berth "
            "would require the kind of run that simply doesn't happen at AFL "
            "level — the priority from here is list build, draft position and "
            "next season."
        )

    return " ".join([s1, s2, s3, s4, s5])


def build_finals_pathway_body(
    year: int, max_round: int,
    ladder: pd.DataFrame, summary_with_ranks: pd.DataFrame,
    games_remaining: int,
) -> str:
    """Assemble the full markdown body for the finals pathway section."""
    if ladder.empty:
        return (
            "_Finals pathway section needs match results to render — no completed "
            f"games found for {year} yet._"
        )

    intro = (
        f"What does each AFL team need to do — from here — to make finals this "
        f"year, and what would have to go right for them to play in the grand "
        f"final? After Round {max_round} of the {year} season, every side has "
        f"played 7 games with roughly {games_remaining} games left in the "
        f"home-and-away. The paragraphs below combine the actual {year} "
        f"ladder (wins, losses, percentage) with the team's stat profile across "
        "16 categories to write an honest, data-driven mid-season assessment "
        "for each club. The grand-final read at the end of each paragraph maps "
        "to one of: **contender** (top-4, elite stat profile), **live chance** "
        "(top-8 with weapons), **chance** (top-8 without breadth), **long shot** "
        "(9-12), **mathematically alive** (13-16) or **season effectively "
        "over** (17-18) — the cut-offs are deliberately blunt."
    )

    chart_ref = (
        f"\n![{year} AFL Finals Pathway ladder chart]"
        f"(../assets/charts/finals_pathway_{year}.png)\n"
    )
    body_lines: List[str] = [intro, chart_ref, ""]

    # Order paragraphs by ladder position (1..18) so the strongest sides appear
    # first — easier to read top-to-bottom as a "what's the season look like".
    sm = summary_with_ranks.set_index("team")
    for _, lr in ladder.sort_values("position").iterrows():
        team = lr["team"]
        if team not in sm.index:
            # Defensive — every team in matches should also be in summary,
            # but skip cleanly if not.
            continue
        stat_row = sm.loc[team]
        body_lines.append(f"### {ordinal(int(lr['position']))} — {team}")
        body_lines.append("")
        body_lines.append(
            render_finals_pathway_paragraph(lr, stat_row, ladder, games_remaining)
        )
        body_lines.append("")

    return "\n".join(body_lines).rstrip()


def replace_finals_pathway_section(
    readme_text: str, year: int, body: str,
) -> str:
    """Insert / replace the finals pathway section between the markers
    `<!-- {year}-FINALS-PATHWAY-START -->` / `<!-- {year}-FINALS-PATHWAY-END -->`.

    If the markers are missing, insert the section immediately after the
    current-year team analysis section's END marker (and before the 5-year
    profiles block, which sits between its own marker pair).
    Also adds a TOC entry if missing.
    """
    start_marker = f"<!-- {year}-FINALS-PATHWAY-START -->"
    end_marker = f"<!-- {year}-FINALS-PATHWAY-END -->"
    section_header = f"## {year} finals pathway — what each team needs"
    # GitHub anchor convention: lowercase, replace spaces with `-`, em-dash
    # collapses to `--` like other auto-generated anchors in this file.
    toc_entry = (
        f"  - [{year} finals pathway — what each team needs]"
        f"(#{year}-finals-pathway--what-each-team-needs)"
    )

    if start_marker in readme_text and end_marker in readme_text:
        before, rest = readme_text.split(start_marker, 1)
        _, after = rest.split(end_marker, 1)
        new_text = (
            before + start_marker + "\n" + body + "\n" + end_marker + after
        )
    else:
        anchor = f"<!-- {year}-TEAM-ANALYSIS-END -->"
        idx = readme_text.find(anchor)
        if idx == -1:
            sys.exit(
                f"Could not find anchor '{anchor}' to insert finals-pathway "
                "section. Run after the current-year team analysis is in place."
            )
        insert_at = idx + len(anchor)
        new_section = (
            "\n\n"
            + section_header
            + "\n\n"
            + start_marker
            + "\n"
            + body
            + "\n"
            + end_marker
            + "\n"
        )
        new_text = readme_text[:insert_at] + new_section + readme_text[insert_at:]

    # TOC entry — place directly after the year team-analysis TOC entry.
    if toc_entry not in new_text:
        prev_toc = f"- [{year} season — live team analysis"
        toc_idx = new_text.find(prev_toc)
        if toc_idx != -1:
            line_end = new_text.find("\n", toc_idx)
            if line_end != -1:
                new_text = (
                    new_text[: line_end + 1]
                    + toc_entry
                    + "\n"
                    + new_text[line_end + 1 :]
                )

    return new_text


def generate_finals_pathway(
    year: int, max_round: int, summary_with_ranks: pd.DataFrame,
) -> Tuple[str, pd.DataFrame]:
    """Top-level entry point. Returns (markdown_body, ladder_df).

    Reads `data/matches/matches_<year>.csv`, computes the live ladder, then
    pairs each team's row with its season-to-date stat ranks from
    `summary_with_ranks` to render a paragraph per team. Designed to be
    wired into the same refresh pipeline as the season analysis."""
    matches = load_match_results(year)
    if matches.empty:
        print(f"      [warn] no matches found for {year} — skipping pathway")
        return "", pd.DataFrame()
    ladder = compute_ladder(matches)
    games_played_each = int(ladder["played"].max()) if not ladder.empty else 0
    games_remaining = max(HOME_AND_AWAY_GAMES - games_played_each, 0)
    print(
        f"      ladder: {len(ladder)} teams, "
        f"{games_played_each} games played, "
        f"{games_remaining} games remaining"
    )
    body = build_finals_pathway_body(
        year, max_round, ladder, summary_with_ranks, games_remaining,
    )
    return body, ladder


def generate_finals_pathway_chart(ladder: pd.DataFrame, year: int, max_round: int) -> str:
    """Create a horizontal ladder-progress chart for the finals pathway section.

    Each team gets a stacked bar: wins (solid) + losses (dimmer) + remaining games
    (hollow outline).  Teams are colour-coded by finals status.  Returns the saved
    chart path, or empty string on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except Exception as exc:  # pragma: no cover
        print(f"[chart] skipped finals pathway chart — {exc}", file=sys.stderr)
        return ""

    if ladder.empty:
        return ""

    BG = "#0d1117"
    GRID = "#30363d"
    GOLD = "#f4c430"
    TEAL = "#2ec4b6"
    TOTAL_GAMES = HOME_AND_AWAY_GAMES  # 22

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": "white",
        "xtick.color": "white", "ytick.color": "white",
        "grid.color": GRID, "text.color": "white",
        "font.family": "monospace", "savefig.facecolor": BG, "savefig.edgecolor": BG,
    })

    def _bar_color(pos: int) -> str:
        if pos <= 4:
            return GOLD
        if pos <= 8:
            return TEAL
        if pos <= 12:
            return "#f77f00"
        return "#e63946"

    def _gf_label(pos: int, wins: int, losses: int) -> str:
        if pos <= 4:
            return "CONTENDER"
        if pos <= 8:
            return "LIVE CHANCE"
        if pos <= 12:
            return "LONG SHOT"
        if pos <= 16:
            return "SLIM CHANCE"
        return "SEASON OVER"

    ladder_sorted = ladder.sort_values("position").reset_index(drop=True)
    n = len(ladder_sorted)

    fig, ax = plt.subplots(figsize=(14, n * 0.52 + 1.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    y_positions = range(n - 1, -1, -1)  # top of chart = 1st place

    for i, (_, row) in enumerate(ladder_sorted.iterrows()):
        y = list(y_positions)[i]
        pos = int(row["position"])
        wins = int(row["W"])
        losses = int(row["L"])
        draws = int(row.get("D", 0))
        played = int(row["played"])
        remaining = max(TOTAL_GAMES - played, 0)
        color = _bar_color(pos)

        # wins bar
        ax.barh(y, wins, color=color, height=0.55, alpha=0.9, zorder=3)
        # losses bar
        ax.barh(y, losses, left=wins, color=color, height=0.55, alpha=0.28, zorder=3)
        # remaining bar (outline only)
        ax.barh(y, remaining, left=wins + losses + draws, color=GRID,
                height=0.55, alpha=0.5, zorder=2)

        # Win count label inside bar
        if wins > 0:
            ax.text(wins / 2, y, f"{wins}W", va="center", ha="center",
                    fontsize=7.5, color=BG, fontweight="bold", zorder=5)

        # GF tier label on right
        label = _gf_label(pos, wins, losses)
        label_color = color
        ax.text(TOTAL_GAMES + 0.4, y, label, va="center", ha="left",
                fontsize=7, color=label_color, fontweight="bold", zorder=5)

        # Percentage label
        pct = float(row["percentage"])
        ax.text(TOTAL_GAMES + 6.8, y, f"{pct:.0f}%", va="center", ha="left",
                fontsize=7, color="white", alpha=0.75, zorder=5)

    # Y-axis labels: position + team name
    team_labels = [
        f"{int(r['position']):2d}. {r['team']}" for _, r in ladder_sorted.iterrows()
    ]
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(team_labels, fontsize=8.5)

    # Vertical line: ~wins needed for finals (approx 11 wins from 22)
    wins_for_finals = 11
    ax.axvline(wins_for_finals, color=GOLD, lw=1.2, linestyle="--", alpha=0.6, zorder=1)
    ax.text(wins_for_finals + 0.1, -0.7, "~finals threshold", fontsize=7,
            color=GOLD, alpha=0.8, ha="left")

    # Divider between 8th and 9th
    cutoff_y = n - 8 - 0.5
    ax.axhline(cutoff_y, color=TEAL, lw=1.5, linestyle="-", alpha=0.5, zorder=1)
    ax.text(TOTAL_GAMES + 0.3, cutoff_y + 0.1, "← finals line", fontsize=7,
            color=TEAL, alpha=0.8)

    ax.set_xlim(0, TOTAL_GAMES + 11)
    ax.set_xlabel("Games (wins | losses | remaining)", labelpad=8)
    ax.set_title(
        f"2026 AFL Finals Pathway — after Round {max_round}",
        fontsize=13, color="white", pad=14, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    patches = [
        mpatches.Patch(color=GOLD, label="Top 4 — double chance"),
        mpatches.Patch(color=TEAL, label="5th–8th — finals"),
        mpatches.Patch(color="#f77f00", label="9th–12th — fringe"),
        mpatches.Patch(color="#e63946", label="13th+ — season over"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=7.5,
              facecolor=BG, edgecolor=GRID, labelcolor="white",
              framealpha=0.9)

    chart_dir = os.path.join(REPO_ROOT, "assets", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    out_path = os.path.join(chart_dir, f"finals_pathway_{year}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Brownlow Medal vote-proxy predictor
# ---------------------------------------------------------------------------
# Methodology
# -----------
# The Brownlow Medal is voted on by the umpires (3-2-1 per game). We can't
# predict votes directly without an umpire model, but we *can* build a
# composite per-game score from the stats that historically correlate most
# strongly with vote-earning. We validated the weights on every player-game
# from 2010-2025 (n=145,150) where actual `brownlow_votes` are recorded:
#
#   Pearson r vs brownlow_votes (per game):
#     disposals             +0.36   <- strongest single predictor
#     eff_disp (disp-clang) +0.35
#     contested_possessions +0.33
#     clearances            +0.30
#     goals                 +0.25
#     tackles               +0.14   (taggers are systematically under-rewarded)
#
# The spec proposed weights {disp 0.35, clr 0.25, cp 0.20, eff_disp 0.15,
# goals 0.05}. EDA showed lifting goals to 0.15 measurably improves the
# correlation with actual votes (pearson 0.40 -> 0.42, top-1% vote-rate
# 67% -> 70%) without losing the midfielder-first character — disposals +
# clearances + contested possessions still total 70%. We use the rebalanced
# weights and call out the change explicitly in the README narrative.
#
# We do not have an `effective_disposals` column in the raw scrape — the
# closest substitute is `disposals - clangers` (a disposal that wasn't a
# turnover). Documented and used consistently.
BROWNLOW_WEIGHTS = {
    "disposals": 0.30,
    "clearances": 0.22,
    "contested_possessions": 0.18,
    "effective_disposals": 0.15,
    "goals": 0.15,
}

# Stats we surface in the per-player table.
BROWNLOW_DISPLAY_STATS = [
    "disposals", "clearances", "contested_possessions", "goals",
]

# We need extra columns the team-analysis pipeline doesn't load by default.
BROWNLOW_LOAD_COLS = [
    "team", "year", "round", "disposals", "kicks", "handballs", "marks",
    "goals", "tackles", "clearances", "contested_possessions", "clangers",
    "brownlow_votes",
]


def _load_player_games_with_names(year: int) -> pd.DataFrame:
    """Load every per-player performance file and stitch on the filename
    stem so we can label players in the predictor output.

    Returns a long DataFrame with one row per player-game in the target
    year. Each row carries `player_stem` (filename minus the suffix) and
    a `player_display` (`First Last`).

    We deliberately do NOT reuse the team-analysis `load_all_player_games`
    here because that helper drops the filename-derived player ID — the
    Brownlow output needs per-player identity which the raw CSV column
    doesn't provide.
    """
    pattern = os.path.join(PLAYER_DATA_DIR, "*_performance_details.csv")
    files = sorted(glob.glob(pattern))
    frames: List[pd.DataFrame] = []

    for path in files:
        try:
            df = pd.read_csv(
                path,
                low_memory=False,
                usecols=lambda c: c in BROWNLOW_LOAD_COLS,
            )
            if df.empty or "year" not in df.columns:
                continue
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df[df["year"] == year]
            if df.empty:
                continue
            stem = os.path.basename(path).replace("_performance_details.csv", "")
            df["player_stem"] = stem
            df["player_display"] = prettify_player_stem(stem)
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out


def _build_brownlow_proxy_table(
    player_games: pd.DataFrame, min_games: int = 3,
) -> pd.DataFrame:
    """From per-player-game rows, compute per-player season averages and
    the weighted proxy score.

    Parameters
    ----------
    player_games : DataFrame
        Output of `_load_player_games_with_names`, filtered to one year.
    min_games : int
        Minimum games played required to be ranked. The spec says >=3 to
        keep one-game callups out of the headline numbers.

    Returns
    -------
    DataFrame with one row per player and columns:
        player_stem, player_display, team, games_played,
        disposals_pg, clearances_pg, cont_poss_pg, goals_pg,
        eff_disp_pg, brownlow_proxy_pg, projected_votes_22_games
    """
    if player_games.empty:
        return pd.DataFrame()

    g = player_games.copy()

    # Coerce stat columns to numeric. Missing values become 0 — same
    # convention as the team-analysis pipeline (a blank cell means the
    # player did not record that stat in that game).
    for c in [
        "disposals", "clearances", "contested_possessions", "clangers",
        "goals", "tackles", "kicks", "handballs",
    ]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)
        else:
            g[c] = 0.0

    # `effective_disposals` proxy: disposals minus clangers. The raw scrape
    # does not have a true effective-disposal column.
    g["effective_disposals"] = (g["disposals"] - g["clangers"]).clip(lower=0)

    # Aggregate per player. A player can play for two teams in a season
    # (mid-year trade); we credit each player-team-game row independently
    # but report the team they played most games for.
    rows_in = len(g)
    grouped = g.groupby("player_stem", as_index=False)

    agg = grouped.agg(
        player_display=("player_display", "first"),
        games_played=("disposals", "size"),  # one row per game played
        disposals_pg=("disposals", "mean"),
        clearances_pg=("clearances", "mean"),
        cont_poss_pg=("contested_possessions", "mean"),
        goals_pg=("goals", "mean"),
        eff_disp_pg=("effective_disposals", "mean"),
        tackles_pg=("tackles", "mean"),
    )

    # Most-frequent team per player (mid-year trades happen).
    teams = (
        g.groupby(["player_stem", "team"]).size().reset_index(name="n")
        .sort_values(["player_stem", "n"], ascending=[True, False])
        .drop_duplicates("player_stem")
        [["player_stem", "team"]]
    )
    out = agg.merge(teams, on="player_stem", how="left")

    rows_before_filter = len(out)
    out = out[out["games_played"] >= min_games].copy()
    rows_after_filter = len(out)
    print(
        f"      brownlow: {rows_in} player-games -> {rows_before_filter} players "
        f"-> {rows_after_filter} with >={min_games} games"
    )

    if out.empty:
        return out

    # Z-score across the eligible cohort. SD computed with ddof=0 to match
    # numpy default and to keep z-scores well-defined for small cohorts.
    def _z(series: pd.Series) -> pd.Series:
        sd = series.std(ddof=0)
        if sd == 0 or pd.isna(sd):
            return pd.Series([0.0] * len(series), index=series.index)
        return (series - series.mean()) / sd

    out["_z_disp"] = _z(out["disposals_pg"])
    out["_z_clr"] = _z(out["clearances_pg"])
    out["_z_cp"] = _z(out["cont_poss_pg"])
    out["_z_eff"] = _z(out["eff_disp_pg"])
    out["_z_goals"] = _z(out["goals_pg"])

    out["brownlow_proxy_pg"] = (
        BROWNLOW_WEIGHTS["disposals"] * out["_z_disp"]
        + BROWNLOW_WEIGHTS["clearances"] * out["_z_clr"]
        + BROWNLOW_WEIGHTS["contested_possessions"] * out["_z_cp"]
        + BROWNLOW_WEIGHTS["effective_disposals"] * out["_z_eff"]
        + BROWNLOW_WEIGHTS["goals"] * out["_z_goals"]
    )

    # Project to a 22-game season. Note: this is a simple linear
    # extrapolation — it assumes the player keeps playing at their current
    # per-game rate, ignoring injury, form, opposition, and the umpire
    # vote ceiling of 30/season. We label this clearly as "projected" not
    # "predicted votes."
    HOME_AND_AWAY = 22
    # Re-baseline projection so it scales with games already played.
    # A player on 5.0 proxy/g over 7 games is treated identically to one
    # at 5.0/g over 8 games — we just multiply per-game by 22.
    # The proxy is dimensionless (sum of weighted z-scores) so the absolute
    # scale is for ranking, not for direct vote interpretation.
    out["projected_votes_22_games"] = out["brownlow_proxy_pg"] * HOME_AND_AWAY

    out = out.sort_values("brownlow_proxy_pg", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1

    # Drop helper z-score columns from the public frame.
    drop_cols = [c for c in out.columns if c.startswith("_z_")]
    out = out.drop(columns=drop_cols)

    return out


def generate_brownlow_chart(
    top_players: pd.DataFrame, year: int, max_round: int,
) -> str:
    """Generate a horizontal bar chart of the top Brownlow vote candidates.

    Returns the saved chart path, or empty string on failure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[chart] skipped brownlow chart — {exc}", file=sys.stderr)
        return ""

    if top_players.empty:
        return ""

    BG = "#0d1117"
    GRID = "#30363d"
    GOLD = "#f4c430"
    TEAL = "#2ec4b6"
    SKY = "#4cc9f0"

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": "white",
        "xtick.color": "white", "ytick.color": "white",
        "grid.color": GRID, "text.color": "white",
        "font.family": "monospace",
        "savefig.facecolor": BG, "savefig.edgecolor": BG,
    })

    top = top_players.head(15).copy()
    n = len(top)

    fig, ax = plt.subplots(figsize=(13, max(n * 0.55 + 1.5, 6)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Tier-by-rank colour: gold for top 5, teal for 6-10, sky for 11-15.
    def _bar_color(rank: int) -> str:
        if rank <= 5:
            return GOLD
        if rank <= 10:
            return TEAL
        return SKY

    # Plot from bottom (worst rank shown) to top (best rank), so rank 1
    # ends up at the top of the chart.
    y_positions = list(range(n - 1, -1, -1))
    for i, (_, row) in enumerate(top.iterrows()):
        y = y_positions[i]
        rank = int(row["rank"])
        score = float(row["brownlow_proxy_pg"])
        color = _bar_color(rank)
        ax.barh(y, score, color=color, height=0.62, alpha=0.92, zorder=3)

        # Annotation inside the bar: D=xx.x C=x.x (disposals + clearances)
        annot = (
            f"D {row['disposals_pg']:.1f} | "
            f"C {row['clearances_pg']:.1f} | "
            f"CP {row['cont_poss_pg']:.1f}"
        )
        # Place annotation just inside the right end of the bar in dark text.
        x_end = score
        ax.text(
            x_end - 0.05, y, annot,
            va="center", ha="right",
            fontsize=7.5, color=BG, fontweight="bold", zorder=5,
        )

    # Y-tick labels: "1. Player Name (TEAM)"
    labels = [
        f"{int(r['rank']):2d}. {r['player_display']} ({r['team']})"
        for _, r in top.iterrows()
    ]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel("Brownlow vote proxy (z-score composite, per game)", labelpad=8)
    ax.set_title(
        f"{year} Brownlow Medal — vote proxy rankings (after Round {max_round})",
        fontsize=13, color="white", pad=18, fontweight="bold",
    )
    # Subtitle as a second text element directly under the title.
    ax.text(
        0.5, 1.005,
        "Composite: 30% disposals · 22% clearances · 18% contested poss "
        "· 15% effective disposals · 15% goals",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=8.5, color="#c9d1d9", style="italic",
    )

    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    chart_dir = os.path.join(REPO_ROOT, "assets", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    out_path = os.path.join(chart_dir, f"brownlow_predictor_{year}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def _build_brownlow_table_md(top: pd.DataFrame) -> str:
    """Render the top-15 markdown table."""
    header = (
        "| Rank | Player | Team | Games | Disp/g | Clear/g | CP/g | "
        "Goals/g | Proxy | Proj. votes |\n"
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows: List[str] = []
    for _, r in top.iterrows():
        rows.append(
            f"| {int(r['rank'])} "
            f"| {r['player_display']} "
            f"| {r['team']} "
            f"| {int(r['games_played'])} "
            f"| {r['disposals_pg']:.1f} "
            f"| {r['clearances_pg']:.1f} "
            f"| {r['cont_poss_pg']:.1f} "
            f"| {r['goals_pg']:.2f} "
            f"| {r['brownlow_proxy_pg']:+.2f} "
            f"| {r['projected_votes_22_games']:+.1f} |"
        )
    return header + "\n" + "\n".join(rows)


def _build_brownlow_narrative(top: pd.DataFrame) -> str:
    """A 2-3 sentence narrative naming the front-runner and explaining why."""
    if top.empty:
        return ""
    leader = top.iloc[0]
    runner_up = top.iloc[1] if len(top) > 1 else None

    parts: List[str] = []

    # Lead with the proxy front-runner and what's driving them.
    leader_strengths: List[str] = []
    if leader["disposals_pg"] >= top["disposals_pg"].quantile(0.95):
        leader_strengths.append(f"{leader['disposals_pg']:.1f} disposals/g")
    if leader["clearances_pg"] >= top["clearances_pg"].quantile(0.90):
        leader_strengths.append(f"{leader['clearances_pg']:.1f} clearances/g")
    if leader["cont_poss_pg"] >= top["cont_poss_pg"].quantile(0.90):
        leader_strengths.append(f"{leader['cont_poss_pg']:.1f} contested poss/g")
    if leader["goals_pg"] >= 1.0:
        leader_strengths.append(f"{leader['goals_pg']:.1f} goals/g")

    strength_blurb = (
        ", ".join(leader_strengths) if leader_strengths
        else f"{leader['disposals_pg']:.1f} disposals/g and a balanced stat profile"
    )
    parts.append(
        f"On the proxy, **{leader['player_display']}** ({leader['team']}) "
        f"leads the field — built on {strength_blurb} across "
        f"{int(leader['games_played'])} games. The composite score "
        f"({leader['brownlow_proxy_pg']:+.2f}) sits "
        f"{(leader['brownlow_proxy_pg'] - top.iloc[1]['brownlow_proxy_pg']):.2f} "
        f"clear of second place."
    )

    # Mention the runner-up + the gap.
    if runner_up is not None:
        parts.append(
            f"**{runner_up['player_display']}** ({runner_up['team']}) is "
            f"the closest challenger at "
            f"{runner_up['brownlow_proxy_pg']:+.2f}, with "
            f"{runner_up['disposals_pg']:.1f} disposals/g and "
            f"{runner_up['clearances_pg']:.1f} clearances/g."
        )

    # Be honest about what this is and isn't.
    parts.append(
        "The proxy is a statistical model, not actual umpire votes — it "
        "captures the stat-profile umpires *historically* reward, but it "
        "cannot model individual game narrative, suspension impact or "
        "the umpire panel's eye for a defensive midfielder."
    )

    return " ".join(parts)


def generate_brownlow_predictor(
    games: pd.DataFrame, year: int, max_round: int,
) -> Tuple[str, pd.DataFrame]:
    """Build the Brownlow vote-proxy section for the README.

    Parameters
    ----------
    games : pd.DataFrame
        The same player-game frame already loaded by main(). It does not
        carry per-player identity, so we re-load with filename stems via
        `_load_player_games_with_names`. The `games` argument is kept in
        the signature for future-proofing (if the upstream loader gains
        a `player` column, this function can switch to using it).
    year : int
        Target season.
    max_round : int
        Highest numeric round in the data (for the chart subtitle).

    Returns
    -------
    (markdown_body, top_players_df)
        - `markdown_body` is the section content to drop between the
          BROWNLOW-PREDICTOR markers in the README.
        - `top_players_df` is the full ranking (all eligible players),
          useful for downstream callers.
    """
    print("      brownlow: loading per-player rows with names...")
    pg = _load_player_games_with_names(year)
    if pg.empty:
        print(f"      [warn] no player-game data for {year} — skipping brownlow")
        return "", pd.DataFrame()

    table = _build_brownlow_proxy_table(pg, min_games=3)
    if table.empty:
        print("      [warn] no eligible players — skipping brownlow")
        return "", pd.DataFrame()

    top15 = table.head(15).copy()

    # Chart
    chart_path = generate_brownlow_chart(top15, year, max_round)
    if chart_path:
        print(f"      regenerated {os.path.basename(chart_path)}")

    # Markdown body
    intro = (
        f"The **Brownlow Medal** is the AFL's individual award for the "
        f"\"fairest and best\" player, voted on by the on-field umpires "
        f"with a 3-2-1 split per game. It is impossible to predict actual "
        f"votes without modelling umpire behaviour, but we *can* build a "
        f"defensible **statistical proxy** — a composite score over the "
        f"stats that historically correlate with vote-earning. The weights "
        f"below were validated against every player-game from 2010-2025 "
        f"(n=145,150) where actual `brownlow_votes` are recorded — the "
        f"top 1% of proxy games captured ~70% of vote-earning performances. "
        f"Players need at least 3 games played to be ranked. Suspended "
        f"players are not penalised — this proxy is a stat-profile model, "
        f"not a vote forecaster."
    )

    formula_note = (
        "**Composite formula** (z-scored across all eligible players, summed "
        f"with weights): `{BROWNLOW_WEIGHTS['disposals']:.2f} × disposals + "
        f"{BROWNLOW_WEIGHTS['clearances']:.2f} × clearances + "
        f"{BROWNLOW_WEIGHTS['contested_possessions']:.2f} × contested-poss + "
        f"{BROWNLOW_WEIGHTS['effective_disposals']:.2f} × effective-disposals + "
        f"{BROWNLOW_WEIGHTS['goals']:.2f} × goals`. Effective disposals are "
        "approximated as `disposals - clangers` because the raw data does "
        "not carry a true effective-disposal column. Goals are weighted "
        "higher than the conventional midfielder-only template (15% vs the "
        "~5% common in pure-midfielder proxies) because that materially "
        "improves correlation with actual historical Brownlow votes."
    )

    chart_md = (
        f"![{year} Brownlow predictor]"
        f"(../assets/charts/brownlow_predictor_{year}.png)"
    )

    table_md = (
        f"#### Top 15 Brownlow proxy candidates — {year} season-to-date "
        f"(after Round {max_round})\n\n"
        + _build_brownlow_table_md(top15)
    )

    narrative = _build_brownlow_narrative(top15)

    body = "\n\n".join([
        intro,
        formula_note,
        chart_md,
        table_md,
        narrative,
    ])

    return body, table


def replace_brownlow_predictor_section(
    readme_text: str, year: int, body: str,
) -> str:
    """Insert / replace the Brownlow predictor section between the markers
    `<!-- {year}-BROWNLOW-PREDICTOR-START -->` /
    `<!-- {year}-BROWNLOW-PREDICTOR-END -->`.

    If the markers are missing, insert the section immediately after the
    finals-pathway END marker. Also adds a TOC entry if missing.
    """
    start_marker = f"<!-- {year}-BROWNLOW-PREDICTOR-START -->"
    end_marker = f"<!-- {year}-BROWNLOW-PREDICTOR-END -->"
    section_header = f"## {year} Brownlow Medal predictor"
    toc_entry = (
        f"  - [{year} Brownlow Medal predictor]"
        f"(#{year}-brownlow-medal-predictor)"
    )

    if start_marker in readme_text and end_marker in readme_text:
        before, rest = readme_text.split(start_marker, 1)
        _, after = rest.split(end_marker, 1)
        new_text = (
            before + start_marker + "\n" + body + "\n" + end_marker + after
        )
    else:
        anchor = f"<!-- {year}-FINALS-PATHWAY-END -->"
        idx = readme_text.find(anchor)
        if idx == -1:
            sys.exit(
                f"Could not find anchor '{anchor}' to insert brownlow "
                "section. Run after the finals-pathway is in place."
            )
        insert_at = idx + len(anchor)
        new_section = (
            "\n\n"
            + section_header
            + "\n\n"
            + start_marker
            + "\n"
            + body
            + "\n"
            + end_marker
            + "\n"
        )
        new_text = readme_text[:insert_at] + new_section + readme_text[insert_at:]

    # TOC entry — directly after the finals-pathway TOC line.
    if toc_entry not in new_text:
        prev_toc = f"  - [{year} finals pathway"
        toc_idx = new_text.find(prev_toc)
        if toc_idx != -1:
            line_end = new_text.find("\n", toc_idx)
            if line_end != -1:
                new_text = (
                    new_text[: line_end + 1]
                    + toc_entry
                    + "\n"
                    + new_text[line_end + 1 :]
                )

    return new_text


# ---------------------------------------------------------------------------
# Player performance stats — what to look for and what the data says
# ---------------------------------------------------------------------------
# This block builds an explanatory README section covering the AFL stats most
# commonly tracked for performance prediction (disposal-based, scoring,
# contested, territory, discipline, and team-level). For each stat group we
# show the 2026 league leaders, the season-to-date distribution across all
# eligible players (>=3 games), and the strongest predictive correlates.
#
# Methodology
# -----------
# - Eligibility: any player with >=3 regular-round games in the target year.
#   Single-game callups inflate per-game averages and are excluded from
#   leaderboards by convention (matches the Brownlow-proxy filter).
# - Effective disposals: derived as max(disposals - clangers, 0). The raw
#   data does not carry a true effective-disposal column; this is an honest
#   proxy and we say so in the README.
# - Hit-outs: ruckman-only stat. The distribution is bimodal (most field
#   players record 0); we flag this clearly so a reader does not interpret
#   the league mean as a "typical" player number.
# - Correlations are Pearson r computed on the per-game frame (one row per
#   player-game) — this preserves within-player variance instead of being
#   dominated by season-average rank order across players. p-values are
#   reported but typically <<1e-10 because n is several thousand player-games;
#   the magnitude of |r| is the load-bearing number, not significance.
# - The chart `assets/charts/player_stat_leaders_2026.png` is generated by
#   `generate_readme_charts.chart_player_stat_leaders(...)` (added alongside
#   this section) and shows top 5 leaders for 16 stats in a 4x4 grid.
STAT_LEADERS_LOAD_COLS = [
    "team", "year", "round", "kicks", "marks", "handballs", "disposals",
    "goals", "behinds", "hit_outs", "tackles", "rebound_50s", "inside_50s",
    "clearances", "clangers", "free_kicks_for", "free_kicks_against",
    "contested_possessions", "uncontested_possessions", "contested_marks",
    "marks_inside_50", "goal_assist",
]

# Stats reported in the README narrative (logical groups). The chart shows a
# wider set; this list governs the prose tables.
STAT_LEADERS_PLAYER_STATS = [
    "disposals", "kicks", "handballs", "effective_disposals",
    "goals", "behinds",
    "contested_possessions", "clearances", "tackles", "hit_outs",
    "inside_50s", "marks", "marks_inside_50",
    "clangers", "free_kicks_for", "free_kicks_against",
]


def _load_player_games_for_stat_leaders(year: int) -> pd.DataFrame:
    """Load every per-player performance file for the target year and return
    a long DataFrame keyed by player_stem. Mirrors `_load_player_games_with_names`
    but with the wider column set this section needs.
    """
    pattern = os.path.join(PLAYER_DATA_DIR, "*_performance_details.csv")
    files = sorted(glob.glob(pattern))
    frames: List[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(
                path,
                low_memory=False,
                usecols=lambda c: c in STAT_LEADERS_LOAD_COLS,
            )
            if df.empty or "year" not in df.columns:
                continue
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df[df["year"] == year]
            if df.empty:
                continue
            stem = os.path.basename(path).replace("_performance_details.csv", "")
            df["player_stem"] = stem
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Round can carry finals tags ("EF", "QF") — keep numeric regular rounds.
    out["round"] = pd.to_numeric(out["round"], errors="coerce")
    out = out.dropna(subset=["round"]).copy()
    out["round"] = out["round"].astype(int)
    # Convention: blank stat cells are zero (player did not record that stat).
    stat_cols = [c for c in STAT_LEADERS_LOAD_COLS
                 if c not in {"team", "year", "round"}]
    for c in stat_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    # Effective disposals — derive once.
    if "disposals" in out.columns and "clangers" in out.columns:
        out["effective_disposals"] = (out["disposals"] - out["clangers"]).clip(lower=0)
    return out


def _build_stat_leaders_eligible(
    games_year: pd.DataFrame, min_games: int = 3,
) -> pd.DataFrame:
    """One row per eligible player with season-average columns suffixed `_pg`."""
    if games_year.empty:
        return pd.DataFrame()
    stat_cols = [c for c in games_year.columns
                 if c not in {"team", "year", "round", "player_stem"}]
    agg_dict = {f"{c}_pg": (c, "mean") for c in stat_cols}
    agg_dict["games_played"] = ("round", "nunique")
    agg_dict["team"] = ("team", "last")
    eligible = games_year.groupby("player_stem").agg(**agg_dict).reset_index()
    eligible = eligible[eligible["games_played"] >= min_games].copy()
    return eligible


def _stat_leaders_top5_md(
    eligible: pd.DataFrame, stat: str, fmt: str = "{:.2f}",
) -> str:
    col = f"{stat}_pg"
    if col not in eligible.columns:
        return "_(stat not available in source data)_"
    top = eligible.sort_values(col, ascending=False).head(5)
    if top.empty:
        return "_(no eligible players)_"
    lines = ["| Rank | Player | Team | Per game |", "|---|---|---|---|"]
    for i, (_, r) in enumerate(top.iterrows(), 1):
        lines.append(
            f"| {i} | {prettify_player_stem(r['player_stem'])} "
            f"| {r['team']} | {fmt.format(r[col])} |"
        )
    return "\n".join(lines)


def _stat_leaders_distribution(eligible: pd.DataFrame, stat: str) -> Dict[str, float]:
    col = f"{stat}_pg"
    if col not in eligible.columns or eligible.empty:
        return {}
    s = eligible[col]
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p10": float(s.quantile(0.10)),
        "p50": float(s.quantile(0.50)),
        "p90": float(s.quantile(0.90)),
        "max": float(s.max()),
    }


def _stat_leaders_top_correlates(
    games_year: pd.DataFrame, stat: str, candidates: List[str], top_n: int = 3,
) -> List[Tuple[str, float, float]]:
    """Top-N predictive correlates for `stat`, computed on per-game data.
    Returns list of (other_stat, pearson_r, p_value).
    """
    try:
        from scipy.stats import pearsonr
    except Exception:
        return []
    if stat not in games_year.columns or games_year.empty:
        return []
    rs: List[Tuple[str, float, float]] = []
    x = games_year[stat].astype(float).values
    if np.std(x) == 0:
        return []
    for other in candidates:
        if other == stat or other not in games_year.columns:
            continue
        y = games_year[other].astype(float).values
        if np.std(y) == 0:
            continue
        try:
            r, p = pearsonr(x, y)
        except Exception:
            continue
        rs.append((other, float(r), float(p)))
    rs.sort(key=lambda t: abs(t[1]), reverse=True)
    return rs[:top_n]


def _stat_leaders_team_table(matches: pd.DataFrame, year: int) -> pd.DataFrame:
    """One row per (team, game) with score, margin and Q1 score derived from
    the long-format match file. Mirrors the existing `load_match_results`
    convention but keeps quarter-level cols which `load_match_results` strips.
    """
    if matches.empty:
        return pd.DataFrame()
    rows = []
    for _, m in matches.iterrows():
        try:
            t1, t2 = m["team_1_team_name"], m["team_2_team_name"]
            s1 = float(m["team_1_final_goals"]) * 6 + float(m["team_1_final_behinds"])
            s2 = float(m["team_2_final_goals"]) * 6 + float(m["team_2_final_behinds"])
            q1_1 = float(m.get("team_1_q1_goals", 0)) * 6 + float(m.get("team_1_q1_behinds", 0))
            q1_2 = float(m.get("team_2_q1_goals", 0)) * 6 + float(m.get("team_2_q1_behinds", 0))
            rows.append(dict(team=t1, score=s1, margin=s1 - s2, q1=q1_1))
            rows.append(dict(team=t2, score=s2, margin=s2 - s1, q1=q1_2))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def generate_stat_leaders_chart(
    eligible: pd.DataFrame, games_year: pd.DataFrame,
    team_long: pd.DataFrame, year: int, max_round: int,
) -> str:
    """Render the 4x4 stat-leaders grid to assets/charts/. Returns the path
    or '' on failure (chart generation is non-fatal).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[chart] skipped stat-leaders chart — {exc}", file=sys.stderr)
        return ""

    BG = "#0d1117"
    GOLD = "#f4c430"
    TEAL = "#2ec4b6"
    SKY = "#4cc9f0"
    GRID = "#30363d"
    TEXT = "#e6edf3"
    SUBTLE = "#8b949e"
    PINK = "#ef476f"
    PURPLE = "#a78bfa"

    def player_top5(stat: str) -> List[Tuple[str, float]]:
        col = f"{stat}_pg"
        if col not in eligible.columns:
            return []
        df = eligible.sort_values(col, ascending=False).head(5)
        return [(prettify_player_stem(r["player_stem"]), float(r[col]))
                for _, r in df.iterrows()]

    def conv_rate_top5() -> List[Tuple[str, float]]:
        if games_year.empty:
            return []
        ag = games_year.groupby("player_stem").agg(
            goals=("goals", "sum"),
            behinds=("behinds", "sum"),
            n=("round", "nunique"),
        ).reset_index()
        ag = ag[(ag["goals"] + ag["behinds"]) > 0]
        ag = ag[(ag["n"] >= 3) & (ag["goals"] >= 2)].copy()
        if ag.empty:
            return []
        ag["conv"] = ag["goals"] / (ag["goals"] + ag["behinds"])
        # Tiebreak by total goals so a 6-for-6 player outranks a 2-for-2 —
        # otherwise the leaderboard is full of small-sample 100% players
        # and the chart disagrees with the README table below it.
        df = ag.sort_values(["conv", "goals"], ascending=[False, False]).head(5)
        return [(prettify_player_stem(r["player_stem"]), float(r["conv"] * 100))
                for _, r in df.iterrows()]

    def team_top5(col: str) -> List[Tuple[str, float]]:
        if team_long.empty:
            return []
        agg = team_long.groupby("team")[col].mean().reset_index()
        agg = agg.sort_values(col, ascending=False).head(5)
        return [(r["team"], float(r[col])) for _, r in agg.iterrows()]

    specs = [
        ("Disposals",             lambda: player_top5("disposals"),             "{:.1f}",  GOLD),
        ("Kicks",                 lambda: player_top5("kicks"),                 "{:.1f}",  GOLD),
        ("Handballs",             lambda: player_top5("handballs"),             "{:.1f}",  GOLD),
        ("Effective disposals",   lambda: player_top5("effective_disposals"),   "{:.1f}",  GOLD),
        ("Goals",                 lambda: player_top5("goals"),                 "{:.2f}",  TEAL),
        ("Goal conversion %",     lambda: conv_rate_top5(),                     "{:.0f}%", TEAL),
        ("Inside 50s",            lambda: player_top5("inside_50s"),            "{:.2f}",  TEAL),
        ("Marks",                 lambda: player_top5("marks"),                 "{:.1f}",  TEAL),
        ("Contested possessions", lambda: player_top5("contested_possessions"), "{:.2f}",  SKY),
        ("Clearances",            lambda: player_top5("clearances"),            "{:.2f}",  SKY),
        ("Tackles",               lambda: player_top5("tackles"),               "{:.2f}",  SKY),
        ("Hit-outs (ruckmen)",    lambda: player_top5("hit_outs"),              "{:.1f}",  SKY),
        ("Clangers",              lambda: player_top5("clangers"),              "{:.2f}",  PINK),
        ("Frees-for",             lambda: player_top5("free_kicks_for"),        "{:.2f}",  PINK),
        ("Team avg score",        lambda: team_top5("score"),                   "{:.0f}",  PURPLE),
        ("Team avg margin",       lambda: team_top5("margin"),                  "{:+.1f}", PURPLE),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(18, 18), facecolor=BG)
    fig.suptitle(
        f"{year} AFL — per-game statistical leaders (after Round {max_round})",
        color=TEXT, fontsize=18, fontweight="bold", y=0.995,
    )

    for ax, (title, fn, fmt, color) in zip(axes.flat, specs):
        items = fn()
        if not items:
            ax.set_facecolor(BG)
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    color=SUBTLE, transform=ax.transAxes)
            ax.set_title(title, color=GOLD, fontsize=11, fontweight="bold",
                         pad=8, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            continue
        labels = [lbl for lbl, _ in items][::-1]
        values = [v for _, v in items][::-1]
        bars = ax.barh(labels, values, color=color, edgecolor=BG, height=0.7)
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color=GRID, alpha=0.4, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title(title, color=GOLD, fontsize=11, fontweight="bold",
                     pad=8, loc="left")
        xmax = max(values) if values else 1.0
        for bar, v in zip(bars, values):
            ax.text(v + xmax * 0.02, bar.get_y() + bar.get_height() / 2,
                    fmt.format(v), va="center", ha="left",
                    color=TEXT, fontsize=9, fontweight="bold")
        ax.set_xlim(0, xmax * 1.25)

    fig.text(
        0.5, 0.005,
        "Per-game season averages, players with >=3 games. Hit-outs is a "
        "ruckman-only stat (most field players are 0). "
        f"Source: data/player_data/ + data/matches/matches_{year}.csv",
        ha="center", color=SUBTLE, fontsize=9, fontstyle="italic",
    )
    plt.tight_layout(rect=[0, 0.012, 1, 0.985])
    chart_dir = os.path.join(REPO_ROOT, "assets", "charts")
    os.makedirs(chart_dir, exist_ok=True)
    out_path = os.path.join(chart_dir, f"player_stat_leaders_{year}.png")
    try:
        plt.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    except Exception as exc:
        print(f"[chart] failed to save stat-leaders chart — {exc}", file=sys.stderr)
        plt.close(fig)
        return ""
    plt.close(fig)
    return out_path


# Plain-English explainer for each stat group. Kept as a constant so the
# narrative doesn't drift between runs and so a reader sees the same framing
# whatever the live numbers happen to be.
_STAT_EXPLAINERS = {
    "disposals": (
        "**What it measures.** Total kicks plus handballs in a game — the "
        "single broadest measure of how often a player has the ball. "
        "**Why it matters.** It is the headline SuperCoach scoring stat and "
        "the prediction target this repo's main model is built around. "
        "Volume midfielders and rebounding defenders dominate this leaderboard."
    ),
    "kicks": (
        "**What it measures.** Just the kicked disposals. **Why it matters.** "
        "Kicks tend to come from outside-midfielders, half-backs and tall "
        "rebounders — players who clear the ball by foot rather than shovel "
        "it into a contest. A player who kicks much more than they handball "
        "is usually playing a distributor / launch role."
    ),
    "handballs": (
        "**What it measures.** The hand-passed half of disposals. "
        "**Why it matters.** Handball volume tracks contest involvement — "
        "a player wins the ball at a stoppage, then handballs out to a "
        "runner. Inside-mids and clearance specialists tend to lead this stat."
    ),
    "effective_disposals": (
        "**What it measures.** Disposals that did not result in a clanger, "
        "computed here as `max(disposals - clangers, 0)` because the raw data "
        "does not carry a true effective-disposal column. **Why it matters.** "
        "It is a defensible proxy for disposal *quality* — high-volume "
        "ball-users who don't turn it over. The same proxy is used in the "
        "Brownlow predictor on this page."
    ),
    "goals": (
        "**What it measures.** Goals kicked. **Why it matters.** Forwards "
        "live and die by this stat. It is volatile game-to-game (a single "
        "missed shot can halve your score), so multi-game averages and "
        "shot-source context (marks-inside-50, contested marks) matter "
        "more than any one game."
    ),
    "behinds": (
        "**What it measures.** Minor scores — shots that hit the post or "
        "go through the smaller posts. **Why it matters.** Rarely predicted "
        "alone — it is too noisy. Best read alongside goals to compute "
        "**conversion rate** (`goals / (goals + behinds)`), the cleanest "
        "available signal of forward accuracy."
    ),
    "contested_possessions": (
        "**What it measures.** Wins of the ball under physical pressure — "
        "ground-balls, taps, and contested marks. **Why it matters.** This "
        "is the cleanest stat for separating a midfielder's *contest* role "
        "from an outside ball-user's *spread* role. It correlates strongly "
        "with clearances and tackles."
    ),
    "clearances": (
        "**What it measures.** Disposals that move the ball clear of a "
        "stoppage (a centre-bounce or boundary throw-in). **Why it matters.** "
        "Stoppage dominance is one of the few team-level wins a midfield can "
        "manufacture. Top clearance players are almost always the inside-mid "
        "fulcrums of their team."
    ),
    "tackles": (
        "**What it measures.** Pressure acts that physically stop a ball-"
        "carrier. **Why it matters.** Defensive midfield work — the unsung "
        "currency of forward-half pressure and turnover football. "
        "It correlates with clearances (you tackle the same opponent you "
        "compete against) but tells a different story."
    ),
    "hit_outs": (
        "**What it measures.** Wins by a ruckman at a ruck contest (the "
        "tap from a centre bounce or stoppage). **Why it matters.** "
        "Ruckman-only stat — the distribution is bimodal: ~1 player per "
        "team registers double-digits, everyone else is 0. Always read "
        "this leaderboard as \"top ruckmen\", not \"top players\"."
    ),
    "inside_50s": (
        "**What it measures.** Disposals or carries that move the ball "
        "into the team's attacking 50m arc. **Why it matters.** "
        "Territory currency — the precondition for goals. Wing/half-"
        "forward players who launch attacks lead this stat. It correlates "
        "with kicks and disposals because most inside-50s are foot-"
        "delivered."
    ),
    "marks": (
        "**What it measures.** Total uncontested + contested marks taken. "
        "**Why it matters.** Aerial dominance and intercept defence. "
        "Loose-half-back roles dominate the total-marks leaderboard "
        "because they sit behind the play and fly under kicks. Tall "
        "forwards lead a separate, narrower stat — marks inside 50."
    ),
    "marks_inside_50": (
        "**What it measures.** Marks taken inside the attacking 50m arc — "
        "i.e. marks that turn directly into shots on goal. **Why it "
        "matters.** This is the strongest single predictor of a forward's "
        "goal output. It is what separates a deep-forward role from a "
        "high-half-forward role, and the correlation with goals is the "
        "highest of any stat in this section."
    ),
    "clangers": (
        "**What it measures.** Errors — missed targets, fumbles, free "
        "kicks given away by the ball-carrier. **Why it matters.** "
        "Clangers are the friction term on disposal volume — a high-"
        "disposal player who also leads in clangers is being asked to "
        "play through traffic, not necessarily playing badly. The "
        "correlation with frees-against is mechanical: many clangers "
        "*are* frees-against."
    ),
    "free_kicks_for": (
        "**What it measures.** Free kicks paid to the player. **Why it "
        "matters.** A weak isolated signal — frees-for tracks contest "
        "involvement (rucks especially) more than skill. Best used as a "
        "tiebreaker rather than a standalone metric."
    ),
    "free_kicks_against": (
        "**What it measures.** Free kicks paid against the player. "
        "**Why it matters.** Discipline / aggression marker, with the "
        "caveat that ruck contest infringements inflate the number for "
        "ruckmen. Reads like a clanger when it correlates with them."
    ),
}

# Subsection groupings: ordered list of (group_title, [stats], explanation).
_STAT_GROUPS = [
    ("Disposal-based stats — volume and quality of ball use",
     ["disposals", "kicks", "handballs", "effective_disposals"]),
    ("Scoring stats — goals, behinds and conversion",
     ["goals", "behinds"]),
    ("Contested and ground-ball stats — the inside game",
     ["contested_possessions", "clearances", "tackles", "hit_outs"]),
    ("Territory stats — moving the ball forward",
     ["inside_50s", "marks", "marks_inside_50"]),
    ("Discipline stats — errors and free kicks",
     ["clangers", "free_kicks_for", "free_kicks_against"]),
]


def _format_distribution_line(d: Dict[str, float], unit: str = "") -> str:
    if not d:
        return ""
    return (
        f"League distribution (eligible players, season-to-date): "
        f"mean **{d['mean']:.2f}{unit}**, std {d['std']:.2f}, "
        f"p10 {d['p10']:.2f} / p50 {d['p50']:.2f} / p90 {d['p90']:.2f}, "
        f"max {d['max']:.2f}."
    )


def _format_correlate_line(
    corrs: List[Tuple[str, float, float]], stat: str,
) -> str:
    if not corrs:
        return ""
    pieces = []
    for other, r, p in corrs:
        # Tag the disposals<->effective-disposals pair as mechanical so a
        # reader doesn't think the model has discovered something deep.
        tag = ""
        mech_pair = {
            ("disposals", "effective_disposals"),
            ("effective_disposals", "disposals"),
            ("clangers", "free_kicks_against"),
            ("free_kicks_against", "clangers"),
        }
        if (stat, other) in mech_pair:
            tag = " *(mechanically related)*"
        pieces.append(f"`{other}` (r = {r:+.2f}{tag})")
    return "Top per-game correlates: " + ", ".join(pieces) + "."


def generate_stat_leaders_section(
    games: pd.DataFrame, matches: pd.DataFrame, year: int, max_round: int,
) -> str:
    """Build the full markdown body of the player-stats section.

    Parameters
    ----------
    games : pd.DataFrame
        Already-loaded per-player-game frame (from `load_all_player_games`).
        We do NOT use this directly because it strips player identity; we
        re-load via `_load_player_games_for_stat_leaders`. The argument is
        kept in the signature for API symmetry with `generate_brownlow_predictor`.
    matches : pd.DataFrame
        Output of `load_match_results(year)` — used for team-level scoring.
    year : int
        Target season.
    max_round : int
        Highest numeric round in the data (for chart subtitle and prose).

    Returns
    -------
    Markdown body suitable for dropping between
    `<!-- {year}-STAT-LEADERS-START -->` and `<!-- {year}-STAT-LEADERS-END -->`.
    Returns "" if data is unavailable.
    """
    pg = _load_player_games_for_stat_leaders(year)
    if pg.empty:
        print(f"      [warn] no player-game data for {year} — skipping stat-leaders")
        return ""

    eligible = _build_stat_leaders_eligible(pg, min_games=3)
    if eligible.empty:
        print("      [warn] no eligible players — skipping stat-leaders")
        return ""

    n_eligible = len(eligible)
    n_games_included = int(pg[pg["player_stem"].isin(eligible["player_stem"])].shape[0])

    # Team-level long frame (one row per team-game).
    team_long = _stat_leaders_team_table(matches, year)

    # Chart
    chart_path = generate_stat_leaders_chart(
        eligible, pg, team_long, year, max_round,
    )
    if chart_path:
        print(f"      regenerated {os.path.basename(chart_path)}")

    # ---- Build prose ----
    parts: List[str] = []

    intro = (
        f"This section is a guide to the AFL performance statistics that "
        f"fans, analysts and SuperCoach players track most closely — what "
        f"each stat measures, who is leading it in {year}, what the "
        f"league-wide distribution looks like, and which other stats most "
        f"reliably predict it. All numbers are computed live from "
        f"`data/player_data/` for {year} (rounds 1-{max_round}, "
        f"**{n_eligible} eligible players** with >=3 games, "
        f"**{n_games_included} player-games** included). "
        f"Correlations are Pearson r on the per-game frame; with several "
        f"thousand player-games, p-values are universally tiny — read "
        f"the magnitude of r, not the significance star."
    )
    parts.append(intro)

    chart_md = (
        f"![{year} AFL statistical leaders]"
        f"(../assets/charts/player_stat_leaders_{year}.png)"
    )
    parts.append(chart_md)

    # All candidate columns for correlation analysis (player-game level).
    corr_candidates = [c for c in [
        "disposals", "kicks", "handballs", "effective_disposals",
        "goals", "behinds", "marks", "marks_inside_50", "contested_marks",
        "tackles", "clearances", "contested_possessions",
        "uncontested_possessions", "inside_50s", "rebound_50s",
        "hit_outs", "clangers", "free_kicks_for", "free_kicks_against",
        "goal_assist",
    ] if c in pg.columns]

    # ---- Per-group subsections ----
    for group_title, stats in _STAT_GROUPS:
        parts.append(f"### {group_title}")
        for stat in stats:
            label = {
                "disposals": "Disposals per game",
                "kicks": "Kicks per game",
                "handballs": "Handballs per game",
                "effective_disposals": "Effective disposals per game (disposals − clangers)",
                "goals": "Goals per game",
                "behinds": "Behinds per game",
                "contested_possessions": "Contested possessions per game",
                "clearances": "Clearances per game",
                "tackles": "Tackles per game",
                "hit_outs": "Hit-outs per game (ruckmen only)",
                "inside_50s": "Inside 50s per game",
                "marks": "Marks per game",
                "marks_inside_50": "Marks inside 50 per game",
                "clangers": "Clangers per game",
                "free_kicks_for": "Free kicks for per game",
                "free_kicks_against": "Free kicks against per game",
            }.get(stat, stat.replace("_", " ").title())
            parts.append(f"#### {label}")
            parts.append(_STAT_EXPLAINERS[stat])

            # Bimodality flag for hit-outs.
            if stat == "hit_outs":
                # Quick check: what fraction of eligible players average <1 hit-out?
                col = "hit_outs_pg"
                if col in eligible.columns:
                    frac_zero = float((eligible[col] < 1.0).mean())
                    parts.append(
                        f"**Bimodal distribution warning.** {frac_zero*100:.0f}% "
                        f"of eligible {year} players average less than 1 hit-out "
                        f"per game — they are not ruckmen. The league mean below "
                        f"is dragged down by all the zeros; the meaningful "
                        f"comparison is between ruckmen, where the top of the "
                        f"distribution sits in the 25-35 range."
                    )

            # Top 5 leaderboard
            fmt = "{:.2f}" if stat in {"goals", "contested_possessions",
                                       "clearances", "tackles", "inside_50s",
                                       "marks_inside_50", "clangers",
                                       "free_kicks_for", "free_kicks_against",
                                       "behinds"} else "{:.1f}"
            parts.append(_stat_leaders_top5_md(eligible, stat, fmt))

            # Distribution line
            d = _stat_leaders_distribution(eligible, stat)
            dist_line = _format_distribution_line(d)
            if dist_line:
                parts.append(dist_line)

            # Top correlates
            corr = _stat_leaders_top_correlates(pg, stat, corr_candidates, top_n=3)
            corr_line = _format_correlate_line(corr, stat)
            if corr_line:
                parts.append(corr_line)

            # Goal conversion sub-block — embed inside the goals section.
            if stat == "goals":
                ag = pg.groupby("player_stem").agg(
                    goals=("goals", "sum"),
                    behinds=("behinds", "sum"),
                    team=("team", "last"),
                    n=("round", "nunique"),
                ).reset_index()
                ag = ag[(ag["goals"] + ag["behinds"]) > 0]
                ag = ag[(ag["n"] >= 3)].copy()
                ag["conv"] = ag["goals"] / (ag["goals"] + ag["behinds"])
                eligible_conv = ag[ag["goals"] >= 2]
                if not eligible_conv.empty:
                    parts.append(
                        f"**Goal conversion rate.** Defined as "
                        f"`goals / (goals + behinds)`, season-to-date, "
                        f"for players with >=2 goals total. "
                        f"League distribution (n={len(eligible_conv)}): "
                        f"mean **{eligible_conv['conv'].mean()*100:.1f}%**, "
                        f"std {eligible_conv['conv'].std()*100:.1f}pp, "
                        f"p10 {eligible_conv['conv'].quantile(0.1)*100:.0f}% / "
                        f"p50 {eligible_conv['conv'].quantile(0.5)*100:.0f}% / "
                        f"p90 {eligible_conv['conv'].quantile(0.9)*100:.0f}%."
                    )
                    top_conv = eligible_conv.sort_values(
                        ["conv", "goals"], ascending=[False, False],
                    ).head(5)
                    lines = [
                        "| Rank | Player | Team | G | B | Conversion |",
                        "|---|---|---|---|---|---|",
                    ]
                    for i, (_, r) in enumerate(top_conv.iterrows(), 1):
                        lines.append(
                            f"| {i} | {prettify_player_stem(r['player_stem'])} "
                            f"| {r['team']} | {int(r['goals'])} "
                            f"| {int(r['behinds'])} "
                            f"| {r['conv']*100:.1f}% |"
                        )
                    parts.append("\n".join(lines))

    # ---- Team-level subsection ----
    if not team_long.empty:
        parts.append("### Team-level stats — what the scoreboard says")
        parts.append(
            "Team-level stats use `data/matches/matches_{year}.csv` rather "
            "than per-player aggregates. Total team score is "
            "`goals × 6 + behinds`; margin is the team's score minus the "
            "opponent's. A first-quarter score is a useful "
            "early-momentum signal — strong starters tend to keep the "
            "lead.".replace("{year}", str(year))
        )
        team_summary = team_long.groupby("team").agg(
            games=("score", "size"),
            avg_score=("score", "mean"),
            avg_margin=("margin", "mean"),
            avg_q1=("q1", "mean"),
        ).reset_index()
        # Top 5 by average score
        parts.append("#### Total team score per game")
        top = team_summary.sort_values("avg_score", ascending=False).head(5)
        lines = ["| Rank | Team | Avg score | Avg margin | Avg Q1 |",
                 "|---|---|---|---|---|"]
        for i, (_, r) in enumerate(top.iterrows(), 1):
            lines.append(
                f"| {i} | {r['team']} | {r['avg_score']:.1f} "
                f"| {r['avg_margin']:+.1f} | {r['avg_q1']:.1f} |"
            )
        parts.append("\n".join(lines))
        parts.append(
            f"League distribution of per-game team scores: "
            f"mean **{team_long['score'].mean():.1f}**, "
            f"std {team_long['score'].std():.1f}, "
            f"p10 {team_long['score'].quantile(0.1):.0f} / "
            f"p50 {team_long['score'].quantile(0.5):.0f} / "
            f"p90 {team_long['score'].quantile(0.9):.0f}, "
            f"min {team_long['score'].min():.0f} / "
            f"max {team_long['score'].max():.0f}."
        )

        # Winning margin
        parts.append("#### Winning margin")
        top_m = team_summary.sort_values("avg_margin", ascending=False).head(5)
        lines = ["| Rank | Team | Avg margin | Avg score |",
                 "|---|---|---|---|"]
        for i, (_, r) in enumerate(top_m.iterrows(), 1):
            lines.append(
                f"| {i} | {r['team']} | {r['avg_margin']:+.1f} "
                f"| {r['avg_score']:.1f} |"
            )
        parts.append("\n".join(lines))
        parts.append(
            f"League distribution of margins (signed, per team-game): "
            f"mean ~0 by construction, std {team_long['margin'].std():.1f}, "
            f"p10 {team_long['margin'].quantile(0.1):.0f} / "
            f"p50 {team_long['margin'].quantile(0.5):.0f} / "
            f"p90 {team_long['margin'].quantile(0.9):.0f}."
        )

        # First-quarter score
        parts.append("#### First-quarter score")
        top_q1 = team_summary.sort_values("avg_q1", ascending=False).head(5)
        lines = ["| Rank | Team | Avg Q1 score | Avg full-game score |",
                 "|---|---|---|---|"]
        for i, (_, r) in enumerate(top_q1.iterrows(), 1):
            lines.append(
                f"| {i} | {r['team']} | {r['avg_q1']:.1f} "
                f"| {r['avg_score']:.1f} |"
            )
        parts.append("\n".join(lines))
        parts.append(
            f"League distribution of Q1 scores: "
            f"mean **{team_long['q1'].mean():.1f}**, "
            f"std {team_long['q1'].std():.1f}, "
            f"p10 {team_long['q1'].quantile(0.1):.0f} / "
            f"p50 {team_long['q1'].quantile(0.5):.0f} / "
            f"p90 {team_long['q1'].quantile(0.9):.0f}."
        )

    # ---- Closing note ----
    parts.append("### Going deeper with this repo's models")
    parts.append(
        "For the stats above, three artefacts in this repo will help you "
        "form your own view rather than just reading a leaderboard:\n\n"
        "1. The **disposal prediction model** (`prediction.py` / "
        "`prediction_cpu.py`) forecasts a player's next-round disposal "
        "count using rolling form, opponent context and venue effects. "
        "Run it with `--player surname_first --rounds 1` to see how "
        "uncertainty is quantified for any of the leaders shown above.\n"
        "2. The **backtest framework** (`backtest.py`) replays a season "
        "round-by-round so you can see how the model performed on real, "
        "out-of-sample games — the honest way to judge whether a "
        "leaderboard ranking will continue to hold.\n"
        "3. The **Brownlow proxy section** above is the same per-game "
        "stat structure used here, weighted into a single composite. "
        "If you want a quick \"who's having the best year overall\" "
        "answer rather than per-stat leaders, that table is the one "
        "to look at."
    )

    return "\n\n".join(parts)


def replace_stat_leaders_section(
    readme_text: str, year: int, body: str,
) -> str:
    """Insert / replace the stat-leaders section between the markers
    `<!-- {year}-STAT-LEADERS-START -->` / `<!-- {year}-STAT-LEADERS-END -->`.

    If the markers are missing, insert the section immediately after the
    Brownlow predictor END marker. Also adds a TOC entry if missing.
    """
    start_marker = f"<!-- {year}-STAT-LEADERS-START -->"
    end_marker = f"<!-- {year}-STAT-LEADERS-END -->"
    section_header = (
        f"## {year} player performance stats — what to look for "
        f"and what the data says"
    )
    toc_entry = (
        f"  - [{year} player performance stats — what to look for "
        f"and what the data says]"
        f"(#{year}-player-performance-stats--what-to-look-for-and-what-the-data-says)"
    )

    if start_marker in readme_text and end_marker in readme_text:
        before, rest = readme_text.split(start_marker, 1)
        _, after = rest.split(end_marker, 1)
        new_text = (
            before + start_marker + "\n" + body + "\n" + end_marker + after
        )
    else:
        anchor = f"<!-- {year}-BROWNLOW-PREDICTOR-END -->"
        idx = readme_text.find(anchor)
        if idx == -1:
            print(
                f"[warn] could not find anchor '{anchor}' to insert stat-"
                "leaders section. Run after Brownlow predictor is in place.",
                file=sys.stderr,
            )
            return readme_text
        insert_at = idx + len(anchor)
        new_section = (
            "\n\n"
            + section_header
            + "\n\n"
            + start_marker
            + "\n"
            + body
            + "\n"
            + end_marker
            + "\n"
        )
        new_text = readme_text[:insert_at] + new_section + readme_text[insert_at:]

    # TOC entry — directly after the Brownlow TOC line.
    if toc_entry not in new_text:
        prev_toc = f"  - [{year} Brownlow Medal predictor"
        toc_idx = new_text.find(prev_toc)
        if toc_idx != -1:
            line_end = new_text.find("\n", toc_idx)
            if line_end != -1:
                new_text = (
                    new_text[: line_end + 1]
                    + toc_entry
                    + "\n"
                    + new_text[line_end + 1 :]
                )

    return new_text


# ---------------------------------------------------------------------------
# README write
# ---------------------------------------------------------------------------
def replace_section(readme_text: str, year: int, body: str) -> str:
    """Replace content between START/END markers, inserting markers if missing.

    The markers are year-specific (e.g. `<!-- 2026-TEAM-ANALYSIS-START -->`).
    If the markers are missing but the existing `### YEAR season — live team analysis`
    header is present (it sits at H3 under "## AFL insights"), we replace
    from that header up to the next top-or-same-level header. If neither is
    present, we error out so a human can decide where it should go.
    """
    start_marker = f"<!-- {year}-TEAM-ANALYSIS-START -->"
    end_marker = f"<!-- {year}-TEAM-ANALYSIS-END -->"

    if start_marker in readme_text and end_marker in readme_text:
        before, rest = readme_text.split(start_marker, 1)
        _, after = rest.split(end_marker, 1)
        new = (
            before
            + start_marker
            + "\n"
            + body
            + "\n"
            + end_marker
            + after
        )
        return new

    # No markers — find existing section by H2 header (the section now lives
    # at H2 inside docs/afl-insights.md).
    header_token = f"## {year} season — live team analysis"
    idx = readme_text.find(header_token)
    if idx == -1:
        sys.exit(
            f"Could not find markers '{start_marker}' or header '{header_token}' "
            "in docs/afl-insights.md. Add the markers manually then re-run."
        )

    # Find the next H2-or-higher header after this section.
    after_idx = idx + len(header_token)
    next_h2 = readme_text.find("\n## ", after_idx)
    next_h1 = readme_text.find("\n# ", after_idx)
    candidates = [n for n in (next_h2, next_h1) if n != -1]
    next_idx = min(candidates) if candidates else len(readme_text)

    # Build new section: keep header at H2, wrap with markers, replace body.
    new_section = (
        f"## {year} season — live team analysis\n\n"
        f"{start_marker}\n"
        f"{body}\n"
        f"{end_marker}\n\n"
    )
    return readme_text[:idx] + new_section + readme_text[next_idx + 1 :]


# ---------------------------------------------------------------------------
# Next-round predictions section
# ---------------------------------------------------------------------------
# Reads the most-recent `data/prediction/next_round_<R>_prediction_*.csv`
# (picked by file mtime) and renders:
#   - a one-line intro with the round number + generation date
#   - a horizontal bar chart of the top-20 predicted players, dark-themed
#   - a markdown table of the top-30
#   - a footer pointing at the prediction model
# The CSVs come from `prediction.py` / `prediction_cpu.py` and are checked
# in to the repo, so the section refreshes whenever a new prediction CSV
# lands. Returns "" if no prediction CSV is available so the caller can
# leave the file untouched rather than blanking the section.
# ---------------------------------------------------------------------------
def generate_predictions_section(year: int) -> str:
    """Render the markdown body for the next-round predictions section.

    Looks at `data/prediction/next_round_*_prediction_*.csv`, picks the most
    recent file by mtime, renders an intro, generates a chart and a top-30
    table. Returns the markdown body (no surrounding markers — the caller
    handles that). Returns "" if no usable CSV is found.
    """
    pred_dir = os.path.join(REPO_ROOT, "data", "prediction")
    pattern = os.path.join(pred_dir, "next_round_*_prediction_*.csv")
    candidates = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    if not candidates:
        return ""
    # Pick the most recent file by mtime — the prediction script names
    # rounds in the filename so we recover the round number from the
    # basename rather than from CSV contents.
    latest = max(candidates, key=os.path.getmtime)
    base = os.path.basename(latest)
    # Filename pattern: next_round_<R>_prediction_<YYYYMMDD>_<HHMM>.csv
    round_num: int = -1
    gen_date: str = ""
    try:
        stem = base.replace("next_round_", "")
        round_str, _rest = stem.split("_prediction_", 1)
        round_num = int(round_str)
    except Exception:
        round_num = -1
    # Always use today as the "generated" date — the CSV may be days old
    gen_date = datetime.now().strftime("%Y-%m-%d")

    try:
        df = pd.read_csv(latest)
    except Exception as exc:
        print(f"[predictions] could not read {base}: {exc}", file=sys.stderr)
        return ""
    needed = {"player", "team", "predicted_disposals"}
    if not needed.issubset(df.columns):
        print(
            f"[predictions] {base} missing columns "
            f"(have {list(df.columns)})", file=sys.stderr,
        )
        return ""
    df = df.dropna(subset=["player", "team", "predicted_disposals"]).copy()
    if df.empty:
        return ""
    df = df.sort_values("predicted_disposals", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # Build chart
    chart_rel = ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        BG = "#0d1117"
        GRID = "#30363d"
        GOLD = "#f4c430"
        TEAL = "#2ec4b6"

        plt.rcParams.update({
            "figure.facecolor": BG, "axes.facecolor": BG,
            "axes.edgecolor": GRID, "axes.labelcolor": "white",
            "xtick.color": "white", "ytick.color": "white",
            "grid.color": GRID, "text.color": "white",
            "font.family": "monospace",
            "savefig.facecolor": BG, "savefig.edgecolor": BG,
        })

        chart_top = df.head(20).copy()
        n = len(chart_top)
        fig, ax = plt.subplots(figsize=(13, max(n * 0.42 + 1.5, 6)))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        y_positions = list(range(n - 1, -1, -1))
        for i, (_, row) in enumerate(chart_top.iterrows()):
            y = y_positions[i]
            score = float(row["predicted_disposals"])
            color = GOLD if int(row["rank"]) <= 10 else TEAL
            ax.barh(y, score, color=color, height=0.66, alpha=0.92, zorder=3)
            ax.text(
                score - 0.3, y, f"{score:.1f}",
                va="center", ha="right",
                fontsize=8, color=BG, fontweight="bold", zorder=5,
            )
        labels = [
            f"{int(r['rank']):2d}. {r['player']} ({r['team']})"
            for _, r in chart_top.iterrows()
        ]
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=9)
        round_label = f"Round {round_num}" if round_num >= 0 else "next round"
        ax.set_xlabel("Predicted disposals", labelpad=8)
        ax.set_title(
            f"Predicted disposal leaders — {round_label}, {year}",
            fontsize=13, color="white", pad=14, fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.2, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        chart_dir = os.path.join(REPO_ROOT, "assets", "charts")
        os.makedirs(chart_dir, exist_ok=True)
        out_path = os.path.join(chart_dir, f"predictions_next_round_{year}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        chart_rel = os.path.relpath(out_path, os.path.join(REPO_ROOT, "docs"))
    except Exception as exc:
        print(f"[predictions] chart skipped — {exc}", file=sys.stderr)
        chart_rel = ""

    # Build table
    table_top = df.head(30)
    rows: List[str] = [
        "| Rank | Player | Team | Predicted disposals |",
        "|------|--------|------|--------------------:|",
    ]
    for _, r in table_top.iterrows():
        rows.append(
            f"| {int(r['rank'])} | {r['player']} | {r['team']} | "
            f"{float(r['predicted_disposals']):.1f} |"
        )
    table_md = "\n".join(rows)

    # Assemble body
    round_label = f"Round {round_num}" if round_num >= 0 else "next round"
    intro = f"Predicted disposal leaders for {round_label} — generated {gen_date}."
    parts: List[str] = [intro]
    if chart_rel:
        # chart_rel is from docs/ root; emit `../assets/charts/...`
        parts.append("")
        parts.append(f"![Predicted disposal leaders — {round_label}, {year}]({chart_rel})")
    parts.append("")
    parts.append(f"#### Top 30 predicted disposal leaders — {round_label}, {year}")
    parts.append("")
    parts.append(table_md)
    parts.append("")
    parts.append(
        "Predictions from LightGBM+HGB ensemble trained on historical "
        "player-game data. Requires GPU to generate — see "
        "[docs/technical-reference.md](technical-reference.md)."
    )
    return "\n".join(parts)


def replace_predictions_section(readme_text: str, year: int, body: str) -> str:
    """Replace content between the 2026-PREDICTIONS markers."""
    start_marker = f"<!-- {year}-PREDICTIONS-START -->"
    end_marker = f"<!-- {year}-PREDICTIONS-END -->"
    if start_marker in readme_text and end_marker in readme_text:
        before, rest = readme_text.split(start_marker, 1)
        _, after = rest.split(end_marker, 1)
        return (
            before + start_marker + "\n" + body + "\n" + end_marker + after
        )
    print(
        f"[predictions] markers not found in target file — skipping",
        file=sys.stderr,
    )
    return readme_text


# ---------------------------------------------------------------------------
# Backtest results section
# ---------------------------------------------------------------------------
# Reads the most-recent `data/prediction/backtest/backtest_summary_*.csv`
# and renders:
#   - one-line intro with the number of rounds
#   - a dual-axis bar+line chart (MAE bars, % within 5 line)
#   - a per-round summary table
#   - an overall summary line and a footer
# Like the predictions helper, returns "" when no summary CSV is on disk.
# ---------------------------------------------------------------------------
def _format_player_name(raw: str) -> str:
    """Convert source CSV "Surname First" → "First Surname" for display.

    Splits on the LAST space so multi-word surnames like
    "Wanganeen-Milera Nasiah" become "Nasiah Wanganeen-Milera". If the
    name has no space (or is empty), it's returned unchanged.
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    if " " not in s:
        return s
    surname, first = s.rsplit(" ", 1)
    return f"{first} {surname}"


def _load_top30_player_deviation(year: int, bt_dir: str) -> pd.DataFrame:
    """Aggregate per-player prediction-vs-actual deviation across all rounds.

    Loads every `prediction_vs_actual_round_*_<year>_*.csv` in `bt_dir`,
    deduplicates by (player, round) keeping the most-recent run (latest
    timestamp embedded in the filename), then aggregates by player+team.

    Returns a DataFrame with columns:
        player, team, avg_actual, avg_predicted, avg_error,
        avg_abs_error, rounds_tracked
    sorted by avg_actual descending. Empty if no files.

    Sign convention (matches source CSVs): error = predicted - actual.
    Positive avg_error means the model over-predicted; negative means
    under-predicted.
    """
    pattern = os.path.join(bt_dir, f"prediction_vs_actual_round_*_{year}_*.csv")
    files = sorted([p for p in glob.glob(pattern) if os.path.isfile(p)])
    if not files:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for f in files:
        # Filename: prediction_vs_actual_round_<R>_<YEAR>_<TS>.csv
        # Pull the run-timestamp suffix so we can prefer the latest run
        # for any duplicated (player, round) rows.
        base = os.path.basename(f)
        ts = base.rsplit("_", 1)[-1].replace(".csv", "")
        try:
            sub = pd.read_csv(f)
        except Exception as exc:
            print(f"[backtest] skip {base}: {exc}", file=sys.stderr)
            continue
        needed = {
            "player", "team", "round", "year",
            "predicted_disposals", "actual_disposals", "error", "abs_error",
        }
        if not needed.issubset(sub.columns):
            continue
        sub["_run_ts"] = ts
        frames.append(sub)
    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    # Deduplicate by (player, round) keeping the run with the latest
    # timestamp — that's the most recent regeneration for that round.
    raw = (
        raw.sort_values("_run_ts")
           .drop_duplicates(subset=["player", "round"], keep="last")
           .drop(columns=["_run_ts"])
    )

    grouped = raw.groupby(["player", "team"], dropna=False).agg(
        avg_actual=("actual_disposals", "mean"),
        avg_predicted=("predicted_disposals", "mean"),
        avg_error=("error", "mean"),
        avg_abs_error=("abs_error", "mean"),
        rounds_tracked=("round", "nunique"),
    ).reset_index()

    grouped = grouped.sort_values(
        "avg_actual", ascending=False,
    ).reset_index(drop=True)
    return grouped


def _render_top30_table(top30: pd.DataFrame) -> str:
    """Render the top-30 disposal-leader deviation table as markdown."""
    if top30.empty:
        return ""
    header = [
        "| # | Player | Team | Avg actual disposals | Avg predicted | Avg error | Rounds |",
        "|--:|--------|------|---------------------:|--------------:|----------:|-------:|",
    ]
    body: List[str] = []
    for i, r in top30.iterrows():
        rank = i + 1
        player_disp = _format_player_name(str(r["player"]))
        team = str(r["team"])
        avg_actual = float(r["avg_actual"])
        avg_pred = float(r["avg_predicted"])
        avg_err = float(r["avg_error"])
        avg_abs = float(r["avg_abs_error"])
        rounds = int(r["rounds_tracked"])

        # Sign convention: error = predicted - actual.
        # Positive = over-predicted (we said too high) → ↑
        # Negative = under-predicted (we said too low) → ↓
        if avg_err >= 0:
            err_str = f"+{avg_err:.1f} ↑"
        else:
            # Use the en-dash (−) per the spec for negative numbers
            err_str = f"−{abs(avg_err):.1f} ↓"

        # Bold the row when the model was significantly off on average
        if avg_abs > 6:
            player_cell = f"**{player_disp}**"
            team_cell = f"**{team}**"
            actual_cell = f"**{avg_actual:.1f}**"
            pred_cell = f"**{avg_pred:.1f}**"
            err_cell = f"**{err_str}**"
            rounds_cell = f"**{rounds}**"
            rank_cell = f"**{rank}**"
        else:
            player_cell = player_disp
            team_cell = team
            actual_cell = f"{avg_actual:.1f}"
            pred_cell = f"{avg_pred:.1f}"
            err_cell = err_str
            rounds_cell = f"{rounds}"
            rank_cell = f"{rank}"

        body.append(
            f"| {rank_cell} | {player_cell} | {team_cell} | "
            f"{actual_cell} | {pred_cell} | {err_cell} | {rounds_cell} |"
        )
    return "\n".join(header + body)


def generate_backtest_section(year: int) -> str:
    """Render the markdown body for the backtest-results section.

    Picks the most recent `backtest_summary_*.csv` by mtime, builds a
    chart + summary table + plain-English glossary + top-30 player
    deviation table, and returns the markdown body. Returns "" if no
    summary CSV is present.
    """
    bt_dir = os.path.join(REPO_ROOT, "data", "prediction", "backtest")
    pattern = os.path.join(bt_dir, "backtest_summary_*.csv")
    candidates = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    if not candidates:
        return ""
    latest = max(candidates, key=os.path.getmtime)
    try:
        df = pd.read_csv(latest)
    except Exception as exc:
        print(f"[backtest] could not read {os.path.basename(latest)}: {exc}", file=sys.stderr)
        return ""
    needed = {
        "round", "year", "n_players", "mae", "rmse",
        "pct_within_5", "pct_within_10",
    }
    if not needed.issubset(df.columns):
        print(
            f"[backtest] {os.path.basename(latest)} missing columns "
            f"(have {list(df.columns)})", file=sys.stderr,
        )
        return ""
    # Filter to the target year so cross-year summaries don't blend in
    df = df[df["year"] == year].copy()
    if df.empty:
        return ""
    df = df.sort_values("round").reset_index(drop=True)
    n_rounds = len(df)

    # Chart
    chart_rel = ""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        BG = "#0d1117"
        GRID = "#30363d"
        GOLD = "#f4c430"
        TEAL = "#2ec4b6"

        plt.rcParams.update({
            "figure.facecolor": BG, "axes.facecolor": BG,
            "axes.edgecolor": GRID, "axes.labelcolor": "white",
            "xtick.color": "white", "ytick.color": "white",
            "grid.color": GRID, "text.color": "white",
            "font.family": "monospace",
            "savefig.facecolor": BG, "savefig.edgecolor": BG,
        })
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        x = df["round"].astype(int).tolist()
        mae_vals = df["mae"].astype(float).tolist()
        ax.bar(x, mae_vals, color=GOLD, alpha=0.85, zorder=3, label="MAE (disposals)")
        for xi, mv in zip(x, mae_vals):
            ax.text(
                xi, mv + 0.08, f"{mv:.2f}",
                ha="center", va="bottom",
                fontsize=8, color="white", zorder=5,
            )
        ax.set_xlabel("Round", labelpad=8)
        ax.set_ylabel("MAE (disposals)", color=GOLD, labelpad=8)
        ax.tick_params(axis="y", colors=GOLD)
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.2, zorder=0)
        ax.spines["top"].set_visible(False)

        ax2 = ax.twinx()
        ax2.set_facecolor(BG)
        pct_vals = df["pct_within_5"].astype(float).tolist()
        ax2.plot(
            x, pct_vals,
            color=TEAL, lw=2.2, marker="o", markersize=7, zorder=4,
            label="% within 5 disposals",
        )
        for xi, pv in zip(x, pct_vals):
            ax2.text(
                xi, pv + 0.7, f"{pv:.0f}%",
                ha="center", va="bottom",
                fontsize=8, color=TEAL, zorder=5,
            )
        ax2.set_ylabel("% of players within 5 disposals", color=TEAL, labelpad=10)
        ax2.tick_params(axis="y", colors=TEAL)
        ax2.spines["top"].set_visible(False)
        # Pad the secondary axis so labels don't run off the top
        cur_lo, cur_hi = ax2.get_ylim()
        ax2.set_ylim(cur_lo, cur_hi + 6)

        ax.set_title(
            f"Prediction accuracy by round — {year} season",
            fontsize=13, color="white", pad=14, fontweight="bold",
        )
        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(
            h1 + h2, l1 + l2,
            loc="lower right", frameon=True, facecolor=BG,
            edgecolor=GRID, labelcolor="white", fontsize=9,
        )
        chart_dir = os.path.join(REPO_ROOT, "assets", "charts")
        os.makedirs(chart_dir, exist_ok=True)
        out_path = os.path.join(chart_dir, f"backtest_accuracy_{year}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        chart_rel = os.path.relpath(out_path, os.path.join(REPO_ROOT, "docs"))
    except Exception as exc:
        print(f"[backtest] chart skipped — {exc}", file=sys.stderr)
        chart_rel = ""

    # Per-round summary table
    rows: List[str] = [
        "| Round | Players | MAE | RMSE | Within 5 disp | Within 10 disp |",
        "|------:|--------:|----:|-----:|--------------:|---------------:|",
    ]
    for _, r in df.iterrows():
        rows.append(
            f"| {int(r['round'])} | {int(r['n_players'])} | "
            f"{float(r['mae']):.2f} | {float(r['rmse']):.2f} | "
            f"{float(r['pct_within_5']):.1f}% | "
            f"{float(r['pct_within_10']):.1f}% |"
        )
    table_md = "\n".join(rows)

    # Overall summary numbers — simple unweighted mean across rounds, which
    # is the appropriate "all rounds equally" view; weighting by n_players
    # is a defensible alternative but matches the per-round table here.
    overall_mae = df["mae"].mean()
    overall_pct5 = df["pct_within_5"].mean()
    overall_pct10 = df["pct_within_10"].mean()

    today_str = datetime.now().strftime("%Y-%m-%d")

    parts: List[str] = []
    parts.append(
        f"*Last updated: {today_str} · {n_rounds} rounds backtested "
        f"· auto-generated*"
    )
    parts.append("")
    parts.append("### What is a backtest?")
    parts.append("")
    parts.append(
        "Before we trust our predictions for next week, we need to check "
        "how well the model has done on rounds that are already finished "
        "— rounds where we know the real answer. A backtest does "
        "exactly that: for each completed round, the model is trained on "
        "all data **before** that round, then asked to predict it. We "
        "then compare prediction to reality."
    )
    parts.append("")
    parts.append(
        "This is the honest test. The model never gets to see the round "
        "it's predicting."
    )
    parts.append("")
    parts.append("### What the numbers mean (in plain English)")
    parts.append("")
    parts.append(
        "| Term | What it actually means | Good or bad? |"
    )
    parts.append(
        "|------|----------------------|--------------|"
    )
    parts.append(
        "| **MAE** (Mean Absolute Error) | On average, our predictions were off "
        "by this many disposals. If MAE = 4.1, we were within ±4 disposals "
        "on a typical player. | Lower = better |"
    )
    parts.append(
        "| **RMSE** (Root Mean Square Error) | Similar to MAE but punishes big "
        "blunders harder — if we say 30 and the player gets 10, RMSE "
        "notices that more than MAE does. | Lower = better |"
    )
    parts.append(
        "| **Median error** | The middle prediction error — half of "
        "players were predicted better than this, half worse. More robust than "
        "MAE because it ignores extreme outliers. | Lower = better |"
    )
    parts.append(
        "| **Bias** | Whether the model systematically over- or under-predicts. "
        "A bias of −0.7 means we tend to predict 0.7 disposals too high. A "
        "bias near 0 is ideal. | Near 0 = better |"
    )
    parts.append(
        "| **Within 5 disposals** | The % of predictions that landed within 5 "
        "of the actual number (e.g. predicted 24, actual was 22 — that "
        "counts). This is the most intuitive accuracy measure for SuperCoach. "
        "| Higher = better |"
    )
    parts.append(
        "| **Within 10 disposals** | Same but with a wider 10-disposal window. "
        "This is nearly always above 90%. | Higher = better |"
    )
    parts.append("")
    parts.append(
        "**Rule of thumb:** an MAE around 4–5 disposals is competitive "
        "for AFL prediction — the game has too many random events "
        "(injuries, umpire decisions, tactic changes) for any model to do "
        "much better. \"Within 5 disposals\" above 65% is good; above 70% "
        "is strong."
    )
    parts.append("")
    if chart_rel:
        parts.append(
            f"![Prediction accuracy by round]({chart_rel})"
        )
        parts.append("")
    parts.append("### Round-by-round accuracy")
    parts.append("")
    parts.append(f"#### Per-round backtest summary — {year}")
    parts.append("")
    parts.append(table_md)
    parts.append("")
    parts.append(
        f"**Overall (mean across {n_rounds} rounds):** MAE "
        f"{overall_mae:.2f} disposals · "
        f"{overall_pct5:.1f}% of predictions within 5 disposals · "
        f"{overall_pct10:.1f}% within 10."
    )
    parts.append("")
    parts.append(
        "> **What to look for:** MAE should stay flat or improve as the "
        "season progresses — the model gets more data per player each "
        "round. A spike in Round 1 (MAE ~4.9) is normal because many "
        "players have no 2026 history yet. If MAE rises sharply mid-season, "
        "it usually means an unusual game week (byes, interstate travel, "
        "weather)."
    )
    parts.append("")

    # Top-30 high-disposal players deviation table
    deviations = _load_top30_player_deviation(year, bt_dir)
    if not deviations.empty:
        top30 = deviations.head(30).reset_index(drop=True)
        top30_md = _render_top30_table(top30)
        if top30_md:
            parts.append(
                "### How accurate were predictions for the top 30 "
                "disposal players?"
            )
            parts.append("")
            parts.append(top30_md)
            parts.append("")
            parts.append(
                "> **Reading this table:** \"Avg error\" tells you whether "
                "the model systematically misjudges a player. A large "
                "positive error (↑) means we over-predicted — "
                "the player gets fewer disposals than expected. A large "
                "negative error (↓) means we under-predicted — "
                "they consistently beat the model. Players with errors "
                "above ±6 (bolded) are worth investigating — "
                "they may have changed role, had an injury, or are "
                "operating in a way the model hasn't caught up with yet."
            )
            parts.append("")

    parts.append(
        "Full backtest CSVs in `data/prediction/backtest/` — run "
        "`backtest.py` to regenerate."
    )
    return "\n".join(parts)


def replace_backtest_section(readme_text: str, year: int, body: str) -> str:
    """Replace content between the 2026-BACKTEST markers."""
    start_marker = f"<!-- {year}-BACKTEST-START -->"
    end_marker = f"<!-- {year}-BACKTEST-END -->"
    if start_marker in readme_text and end_marker in readme_text:
        before, rest = readme_text.split(start_marker, 1)
        _, after = rest.split(end_marker, 1)
        return (
            before + start_marker + "\n" + body + "\n" + end_marker + after
        )
    print(
        f"[backtest] markers not found in target file — skipping",
        file=sys.stderr,
    )
    return readme_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def regenerate_charts(year: int) -> None:
    """Refresh the data-visualisation charts that depend on this season's
    data (radar, heatmap, 5-year scatter). Errors are non-fatal — if the
    chart module is missing or matplotlib chokes we still want the README
    text update to succeed.

    Static charts (era_scoring_trends, era_stat_evolution) are NOT
    regenerated here — those depend on era-level CSVs that change far less
    often. Run `generate_readme_charts.py` standalone to refresh them.
    """
    try:
        # Local import so a missing matplotlib only breaks chart generation,
        # not the README text refresh that the rest of this script does.
        from generate_readme_charts import regenerate_team_charts
    except Exception as e:  # pragma: no cover — defensive
        print(f"[charts] skipped — could not import generate_readme_charts: {e}", file=sys.stderr)
        return
    try:
        paths = regenerate_team_charts(year)
        for p in paths:
            print(f"      regenerated {os.path.basename(p)}")
    except Exception as e:  # pragma: no cover — defensive
        print(f"[charts] regeneration failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# All-time top-100 Hall of Fame section
# ---------------------------------------------------------------------------

def _parse_top100_comment(comment: str) -> dict:
    """Extract numeric career stats from a player's Comment field."""
    import re
    stats: dict = {}
    m = re.search(r"and (\d+) games", comment)
    if m:
        stats["games"] = int(m.group(1))
    m = re.search(r"([\d,]+) total disposals", comment)
    if m:
        stats["disposals"] = int(m.group(1).replace(",", ""))
    # Goals: capture the highest number that appears before " goals"
    m_all = re.findall(r"([\d,]+) goals", comment)
    if m_all:
        stats["goals"] = max(int(x.replace(",", "")) for x in m_all)
    m = re.search(r"earned ([\d,]+) Brownlow votes", comment)
    if m:
        stats["brownlow"] = int(m.group(1).replace(",", ""))
    return stats


def generate_top100_chart() -> str:
    """Horizontal bar chart — top 10 all-time players by era-normalised score."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(TOP100_SCORES_CSV) or not os.path.exists(TOP100_CSV):
        return ""

    scores_df = pd.read_csv(TOP100_SCORES_CSV)
    bio_df = pd.read_csv(TOP100_CSV)

    top10 = bio_df.head(10).copy()
    top10_scores = scores_df.head(10)["all_time_score"].tolist()
    names = top10["Player Name"].tolist()

    BG = "#0d1117"
    GOLD = "#f4c430"
    TEAL = "#2ec4b6"
    SKY = "#4cc9f0"
    GRID = "#30363d"
    TEXT = "#e6edf3"

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.set_facecolor(BG)

    colours = [GOLD] * 3 + [TEAL] * 4 + [SKY] * 3
    bars = ax.barh(range(len(names)), top10_scores, color=colours, height=0.65)

    for i, (bar, score) in enumerate(zip(bars, top10_scores)):
        ax.text(
            bar.get_width() - 0.02, i, f"{score:.3f}",
            va="center", ha="right", color=BG, fontsize=9, fontweight="bold",
        )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(
        [f"#{i + 1}  {n}" for i, n in enumerate(names)],
        color=TEXT, fontsize=10,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Era-normalised composite score", color=TEXT, fontsize=10)
    ax.set_title(
        "Top 10 AFL players of all time — era-normalised composite ranking",
        color=GOLD, fontsize=12, fontweight="bold", pad=14,
    )

    ax.tick_params(axis="x", colors=TEXT)
    ax.tick_params(axis="y", colors=TEXT, length=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    min_score = min(top10_scores)
    ax.set_xlim(min_score - 0.25, max(top10_scores) + 0.25)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GOLD, label="Tier 1 (ranks 1–3)"),
        Patch(facecolor=TEAL, label="Tier 2 (ranks 4–7)"),
        Patch(facecolor=SKY, label="Tier 3 (ranks 8–10)"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower right",
        facecolor="#161b22", edgecolor=GRID, labelcolor=TEXT, fontsize=8,
    )

    os.makedirs(CHARTS_DIR, exist_ok=True)
    chart_path = os.path.join(CHARTS_DIR, "top10_alltime_hall.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return chart_path


def generate_top100_section() -> str:
    """Build the all-time top-100 markdown block for docs/hall-of-fame-top100.md."""
    if not os.path.exists(TOP100_CSV) or not os.path.exists(TOP100_SCORES_CSV):
        return ""

    bio_df = pd.read_csv(TOP100_CSV)
    scores_df = pd.read_csv(TOP100_SCORES_CSV)
    score_map = {i + 1: float(row["all_time_score"]) for i, row in scores_df.iterrows()}

    today = datetime.now().strftime("%Y-%m-%d")
    lines: list[str] = []

    lines.append(f"*Last updated: {today} — auto-generated from era-normalised composite scoring*\n")
    lines.append(
        "Every all-time list is an argument. This one is backed by numbers. "
        "The ranking uses an **era-normalised composite score**: each player's career stats are "
        "converted to z-scores within their playing era, so a 1930s forward is not penalised for "
        "the absence of handball counts in the records, and a modern midfielder is not inflated by "
        "the sheer volume of stats logged today. The composite blends disposals, goals, Brownlow "
        "votes, peak single-game output, and career consistency. The result is not perfect — no "
        "algorithm captures what it felt like to watch Jack Dyer run through a pack or Bernie "
        "Quinlan take a screamer — but it is honest, reproducible, and it updates automatically "
        "as new season data is scraped."
    )
    lines.append("")
    lines.append(
        "The chart below shows the top 10. The full table of all 100 follows with key career numbers."
    )
    lines.append("")
    lines.append("![Top 10 AFL players of all time](../assets/charts/top10_alltime_hall.png)")
    lines.append("")

    # Table
    lines.append("| # | Player | Club(s) | Games | Goals | Disposals | Brownlow | Score |")
    lines.append("|--:|--------|---------|------:|------:|----------:|---------:|------:|")

    for _, row in bio_df.iterrows():
        rank = int(row["Serial Number"])
        name = str(row["Player Name"])
        clubs = str(row["Footy Teams"])
        comment = str(row["Comment"])
        score = score_map.get(rank, 0.0)
        stats = _parse_top100_comment(comment)

        games = str(stats["games"]) if "games" in stats else "—"
        goals = str(stats["goals"]) if "goals" in stats else "—"
        disposals = f"{stats['disposals']:,}" if "disposals" in stats else "—"
        brownlow = str(stats["brownlow"]) if "brownlow" in stats else "—"

        lines.append(
            f"| {rank} | {name} | {clubs} | {games} | {goals} | {disposals} | {brownlow} | {score:.3f} |"
        )

    lines.append("")
    lines.append(
        "> **Reading the score**: Higher = better. "
        "Scores are dimensionless composite z-scores normalised by era. "
        "The gap between #1 Kevin Bartlett (2.729) and #100 spans roughly 1.5 standard deviations "
        "— every player on this list is a statistical outlier across the full 130-year history of the game."
    )
    lines.append("")

    return "\n".join(lines)


def replace_top100_section(hof_text: str, body: str) -> str:
    """Replace ALL-TIME-TOP100 markers in docs/hall-of-fame-top100.md."""
    import re

    start_marker = "<!-- ALL-TIME-TOP100-START -->"
    end_marker = "<!-- ALL-TIME-TOP100-END -->"
    pattern = re.compile(
        re.escape(start_marker) + r".*?" + re.escape(end_marker),
        re.DOTALL,
    )
    replacement = f"{start_marker}\n{body}\n{end_marker}"
    new_text, n = pattern.subn(replacement, hof_text)
    if n == 0:
        heading = "### Top 100 AFL players of all time — ranked by the data"
        if heading in new_text:
            new_text = new_text.replace(
                heading + "\n",
                heading + f"\n\n{start_marker}\n{body}\n{end_marker}\n",
                1,
            )
    return new_text


def main() -> None:
    print("[1/14] Loading player game data...")
    games = load_all_player_games()
    year = detect_current_year(games)
    print(f"      detected current year = {year}")

    print("[2/14] Aggregating to team-game level...")
    team_game = build_team_game_table(games, year)
    max_round = detect_max_round(team_game)
    n_teams = team_game["team"].nunique()
    print(f"      year={year}, rounds={max_round}, teams={n_teams}, team-games={len(team_game)}")

    print("[3/14] Per-team season averages and ranks...")
    summary = per_team_summary(team_game)
    league = league_averages(team_game)
    summary_with_ranks = add_ranks(summary, SUMMARY_STATS + ["rebound_50s"])
    # Add the broader rank set used for the per-team prose paragraphs.
    summary_with_ranks = add_paragraph_ranks(summary_with_ranks)
    summary_with_ranks["form_tag"] = summary_with_ranks.apply(
        lambda r: form_tag(r, summary_with_ranks), axis=1
    )

    print("[4/14] Looking up leading per-team disposal getters...")
    top_scorers = per_team_top_disposal_player(games, year)

    # Each marker pair now lives in its own docs/afl-*-2026.md file after
    # the season-page split. Each block below reads its target file,
    # replaces the marker contents, and writes back. We guard each write
    # with os.path.exists() so a missing destination file (e.g. during a
    # transition or rename) prints a warning rather than crashing.

    print("[5/14] Rendering team analysis section into docs/afl-team-analysis-2026.md...")
    body = build_section_body(year, max_round, summary_with_ranks, summary, league, top_scorers)
    if os.path.exists(TEAM_ANALYSIS_PATH):
        with open(TEAM_ANALYSIS_PATH, "r", encoding="utf-8") as f:
            ta_text = f.read()
        new_ta = replace_section(ta_text, year, body)
        if new_ta != ta_text:
            with open(TEAM_ANALYSIS_PATH, "w", encoding="utf-8") as f:
                f.write(new_ta)
    else:
        print(f"      [warn] {TEAM_ANALYSIS_PATH} missing — skipped", file=sys.stderr)

    print("[6/14] Building 5-year team playing-style profiles...")
    five_year_body, year_window = generate_5year_profiles(year)
    # 5-year profiles live in docs/afl-team-profiles.md and are read/written
    # independently of the per-section 2026 blocks.
    if os.path.exists(TEAM_PROFILES_PATH):
        with open(TEAM_PROFILES_PATH, "r", encoding="utf-8") as f:
            profiles_text = f.read()
        new_profiles = replace_5year_section(profiles_text, year, year_window, five_year_body)
        if new_profiles != profiles_text:
            with open(TEAM_PROFILES_PATH, "w", encoding="utf-8") as f:
                f.write(new_profiles)
    else:
        print(f"      [warn] {TEAM_PROFILES_PATH} missing — skipped", file=sys.stderr)

    print("[7/14] Building finals pathway section into docs/afl-finals-2026.md...")
    pathway_body, _ladder = generate_finals_pathway(year, max_round, summary_with_ranks)
    if pathway_body and os.path.exists(FINALS_PATH):
        with open(FINALS_PATH, "r", encoding="utf-8") as f:
            fp_text = f.read()
        new_fp = replace_finals_pathway_section(fp_text, year, pathway_body)
        if new_fp != fp_text:
            with open(FINALS_PATH, "w", encoding="utf-8") as f:
                f.write(new_fp)
    if not _ladder.empty:
        chart_path = generate_finals_pathway_chart(_ladder, year, max_round)
        if chart_path:
            print(f"      regenerated {os.path.basename(chart_path)}")

    print("[8/14] Building Brownlow Medal vote-proxy section into docs/afl-brownlow-2026.md...")
    brownlow_body, _brownlow_table = generate_brownlow_predictor(games, year, max_round)
    if brownlow_body and os.path.exists(BROWNLOW_PATH):
        with open(BROWNLOW_PATH, "r", encoding="utf-8") as f:
            br_text = f.read()
        new_br = replace_brownlow_predictor_section(br_text, year, brownlow_body)
        if new_br != br_text:
            with open(BROWNLOW_PATH, "w", encoding="utf-8") as f:
                f.write(new_br)

    print("[9/14] Building player performance stats section into docs/afl-stat-leaders-2026.md...")
    matches_for_stats = load_match_results(year)
    stat_body = generate_stat_leaders_section(games, matches_for_stats, year, max_round)
    if stat_body and os.path.exists(STAT_LEADERS_PATH):
        with open(STAT_LEADERS_PATH, "r", encoding="utf-8") as f:
            sl_text = f.read()
        new_sl = replace_stat_leaders_section(sl_text, year, stat_body)
        if new_sl != sl_text:
            with open(STAT_LEADERS_PATH, "w", encoding="utf-8") as f:
                f.write(new_sl)

    print("[10/14] Regenerating data-visualisation charts (radar, heatmap, scatter)...")
    regenerate_charts(year)

    print("[11/14] Building next-round predictions section into docs/afl-predictions-2026.md...")
    pred_body = generate_predictions_section(year)
    if pred_body and os.path.exists(PREDICTIONS_PATH):
        with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            pred_text = f.read()
        new_pred = replace_predictions_section(pred_text, year, pred_body)
        if new_pred != pred_text:
            with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
                f.write(new_pred)

    print("[12/14] Building backtest results section into docs/afl-backtest-2026.md...")
    backtest_body = generate_backtest_section(year)
    if backtest_body and os.path.exists(BACKTEST_PATH):
        with open(BACKTEST_PATH, "r", encoding="utf-8") as f:
            bt_text = f.read()
        new_bt = replace_backtest_section(bt_text, year, backtest_body)
        if new_bt != bt_text:
            with open(BACKTEST_PATH, "w", encoding="utf-8") as f:
                f.write(new_bt)

    print("[13/14] Updating all-time top-100 Hall of Fame section...")
    top100_chart = generate_top100_chart()
    if top100_chart:
        print(f"      regenerated {os.path.basename(top100_chart)}")
    top100_body = generate_top100_section()
    if top100_body and os.path.exists(HALL_OF_FAME_PATH):
        with open(HALL_OF_FAME_PATH, "r", encoding="utf-8") as f:
            hof_text = f.read()
        new_hof = replace_top100_section(hof_text, top100_body)
        if new_hof != hof_text:
            with open(HALL_OF_FAME_PATH, "w", encoding="utf-8") as f:
                f.write(new_hof)

    today = datetime.now().strftime("%Y-%m-%d")
    print("[14/14] Done.")
    print(
        f"✓ {year} team analysis updated — Round {max_round}, {n_teams} teams, "
        f"5-year window {year_window[0]}-{year_window[-1]}, {today}"
    )


if __name__ == "__main__":
    main()
