#!/usr/bin/env python3
"""
update_team_analysis.py
=======================

Generate the "<YEAR> team analysis" section in README.md from the latest
season's player-game data.

What it does
------------
1. Loads every per-player performance CSV in `data/player_data/`
2. Auto-detects the most recent season present in the data
3. Aggregates each game to a per-team per-game row (sum of all 22 players'
   stats) for that season
4. Computes per-team season-to-date averages, the league average across all
   18 teams, and per-team rank for the key stats
5. Builds an intro paragraph + summary table + leaders table in markdown
6. Writes the section into README.md between the markers
       <!-- YEAR-TEAM-ANALYSIS-START -->
       <!-- YEAR-TEAM-ANALYSIS-END -->

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
    return "\n".join(body)


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
def main() -> None:
    print("[1/5] Loading player game data...")
    games = load_all_player_games()
    year = detect_current_year(games)
    print(f"      detected current year = {year}")

    print("[2/5] Aggregating to team-game level...")
    team_game = build_team_game_table(games, year)
    max_round = detect_max_round(team_game)
    n_teams = team_game["team"].nunique()
    print(f"      year={year}, rounds={max_round}, teams={n_teams}, team-games={len(team_game)}")

    print("[3/5] Per-team season averages and ranks...")
    summary = per_team_summary(team_game)
    league = league_averages(team_game)
    summary_with_ranks = add_ranks(summary, SUMMARY_STATS + ["rebound_50s"])
    summary_with_ranks["form_tag"] = summary_with_ranks.apply(
        lambda r: form_tag(r, summary_with_ranks), axis=1
    )

    print("[4/5] Looking up leading per-team disposal getters...")
    top_scorers = per_team_top_disposal_player(games, year)

    print("[5/5] Rendering markdown and updating README.md...")
    body = build_section_body(year, max_round, summary_with_ranks, summary, league, top_scorers)

    with open(README_PATH, "r", encoding="utf-8") as f:
        readme_text = f.read()
    new_readme = replace_section(readme_text, year, body)
    if new_readme != readme_text:
        with open(README_PATH, "w", encoding="utf-8") as f:
            f.write(new_readme)

    today = datetime.now().strftime("%Y-%m-%d")
    print(f"✓ {year} team analysis updated — Round {max_round}, {n_teams} teams, {today}")


if __name__ == "__main__":
    main()
