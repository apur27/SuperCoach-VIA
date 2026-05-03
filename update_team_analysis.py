#!/usr/bin/env python3
"""
update_team_analysis.py
=======================

Generate the "<YEAR> team analysis" section in README.md from the latest
season's player-game data, plus a "5-year team playing styles" section
covering the five seasons immediately prior to the current year.

What it does
------------
1. Loads every per-player performance CSV in `data/player_data/`
2. Auto-detects the most recent season present in the data
3. Aggregates each game to a per-team per-game row (sum of all 22 players'
   stats) for that season
4. Computes per-team season-to-date averages, the league average across all
   18 teams, and per-team rank for the key stats
5. Builds an intro paragraph + summary table + leaders table in markdown
6. Writes the current-season section into README.md between the markers
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
README_PATH = os.path.join(REPO_ROOT, "README.md")

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
    body.append(f"### All 18 teams ranked by total disposals — {year} season-to-date")
    body.append("")
    body.append(table1)
    body.append("")
    body.append(f"### League leaders by stat category — {year}")
    body.append("")
    body.append(table2)
    body.append("")
    # Visual snapshot — radar (top 6) + heatmap (all 18 ranks). The PNGs are
    # regenerated by `generate_readme_charts.regenerate_team_charts(year)` in
    # main(), so they stay in sync with the tables above.
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
    body.append(f"![Top 6 teams radar — {year}](assets/charts/team_{year}_radar.png)")
    body.append("")
    body.append(f"![Team rank heatmap — {year}](assets/charts/team_{year}_heatmap.png)")
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
        f"![5-year team playing styles scatter — {years[0]}-{years[-1]}](assets/charts/team_{current_year}_style_scatter.png)",
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
    """
    start_marker = "<!-- 5YEAR-TEAM-PROFILES-START -->"
    end_marker = "<!-- 5YEAR-TEAM-PROFILES-END -->"
    section_header = f"## Team playing styles — 5 years of data ({years[0]}–{years[-1]})"
    toc_entry = f"- [Team playing styles — 5 years of data ({years[0]}–{years[-1]})](#team-playing-styles--5-years-of-data-{years[0]}{years[-1]})"

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
        prev_toc = f"- [{current_year} team analysis"
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
# README write
# ---------------------------------------------------------------------------
def replace_section(readme_text: str, year: int, body: str) -> str:
    """Replace content between START/END markers, inserting markers if missing.

    The markers are year-specific (e.g. `<!-- 2026-TEAM-ANALYSIS-START -->`).
    If the markers are missing but the existing `## YEAR team analysis` header
    is present, we replace from that header up to the next top-level `##`
    header. If neither is present, we error out so a human can decide where it
    should go.
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

    # No markers — find existing section by header.
    header_token = f"## {year} team analysis"
    idx = readme_text.find(header_token)
    if idx == -1:
        sys.exit(
            f"Could not find markers '{start_marker}' or header '{header_token}' "
            "in README.md. Add the markers manually then re-run."
        )

    # Find next "## " header after this section.
    next_idx = readme_text.find("\n## ", idx + len(header_token))
    if next_idx == -1:
        next_idx = len(readme_text)

    # Build new section: keep header, wrap with markers, replace body.
    new_section = (
        f"## {year} team analysis — what the data says\n\n"
        f"{start_marker}\n"
        f"{body}\n"
        f"{end_marker}\n\n"
    )
    return readme_text[:idx] + new_section + readme_text[next_idx + 1 :]


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


def main() -> None:
    print("[1/6] Loading player game data...")
    games = load_all_player_games()
    year = detect_current_year(games)
    print(f"      detected current year = {year}")

    print("[2/6] Aggregating to team-game level...")
    team_game = build_team_game_table(games, year)
    max_round = detect_max_round(team_game)
    n_teams = team_game["team"].nunique()
    print(f"      year={year}, rounds={max_round}, teams={n_teams}, team-games={len(team_game)}")

    print("[3/6] Per-team season averages and ranks...")
    summary = per_team_summary(team_game)
    league = league_averages(team_game)
    summary_with_ranks = add_ranks(summary, SUMMARY_STATS + ["rebound_50s"])
    # Add the broader rank set used for the per-team prose paragraphs.
    summary_with_ranks = add_paragraph_ranks(summary_with_ranks)
    summary_with_ranks["form_tag"] = summary_with_ranks.apply(
        lambda r: form_tag(r, summary_with_ranks), axis=1
    )

    print("[4/6] Looking up leading per-team disposal getters...")
    top_scorers = per_team_top_disposal_player(games, year)

    print("[5/6] Rendering current-season markdown and updating README.md...")
    body = build_section_body(year, max_round, summary_with_ranks, summary, league, top_scorers)

    with open(README_PATH, "r", encoding="utf-8") as f:
        readme_text = f.read()
    new_readme = replace_section(readme_text, year, body)

    print("[6/6] Building 5-year team playing-style profiles...")
    five_year_body, year_window = generate_5year_profiles(year)
    new_readme = replace_5year_section(new_readme, year, year_window, five_year_body)

    if new_readme != readme_text:
        with open(README_PATH, "w", encoding="utf-8") as f:
            f.write(new_readme)

    print("[7/7] Regenerating data-visualisation charts (radar, heatmap, scatter)...")
    regenerate_charts(year)

    today = datetime.now().strftime("%Y-%m-%d")
    print(
        f"✓ {year} team analysis updated — Round {max_round}, {n_teams} teams, "
        f"5-year window {year_window[0]}-{year_window[-1]}, {today}"
    )


if __name__ == "__main__":
    main()
