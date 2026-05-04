#!/usr/bin/env python3
"""
generate_readme_charts.py
=========================

Generate the five data-visualisation charts that are embedded in
docs/afl-insights.md (and a couple in docs/hall-of-fame-top100.md).

Charts produced (all PNG, dark theme, 150 DPI, saved to assets/charts/):

  1. era_scoring_trends.png   — line chart of avg team score per game by year
                                 (1965-current). Annotates the 1980s peak and
                                 the modern decline.
  2. era_stat_evolution.png   — grouped bar chart of 4 per-player stats
                                 (kicks, handballs, tackles, clearances)
                                 across the 4 historic eras.
  3. team_2026_radar.png      — radar/spider chart comparing the top-6 teams
                                 of the latest season across 6 dimensions,
                                 normalised 0-1 vs the rest of the league.
  4. team_2026_heatmap.png    — rank heatmap of all 18 teams x 8 stats for the
                                 latest season (1=best, 18=worst, green→red).
  5. team_2026_style_scatter  — handball-ratio (X) vs tackles-per-game (Y)
                                 scatter for the prior 5-year window
                                 (e.g. 2021-2025), with quadrant labels.

Functions are split into:

  * `chart_era_scoring_trends`   } static-data charts (era_*.csv +
  * `chart_era_stat_evolution`   } match files); refreshed only when the
                                   underlying era CSVs are regenerated.

  * `chart_team_radar`           } current-season + 5-year-window charts;
  * `chart_team_heatmap`         } regenerated each time
  * `chart_team_style_scatter`   } update_team_analysis.py runs.

`update_team_analysis.py` calls `regenerate_team_charts()` after refreshing
docs/afl-insights.md so the team-specific charts always match the latest data.

Run standalone:

    /home/abhi/sourceCode/python/coding/.venv/bin/python generate_readme_charts.py

Methodology notes (so future-you can audit):

  * Avg team score per year is computed from `data/matches/matches_*.csv` —
    points = goals*6 + behinds, averaged across both teams in every match of
    the year. Not from `era_yearly_trends.csv` (that file does not actually
    contain a `avg_team_score` column despite the name suggesting it might).
  * Era buckets follow the project's existing convention: pre-1965,
    1965-1990, 1991-2010, 2011-present. Stats not recorded in an era are
    rendered as 0 with an "n/a" annotation in the bar chart.
  * Top-6 in the radar chart is by total disposals/game in the current year;
    same ordering used by `update_team_analysis.py` summary table.
  * Heatmap colour scale: 1=best maps to green, 18=worst maps to red, via a
    reversed RdYlGn colormap.
  * 5-year scatter uses the same window logic as `update_team_analysis.py`'s
    5-year profile section: the five seasons immediately prior to the latest
    detected season.
  * All charts use a consistent palette anchored on the banner colours
    (gold #f4c430, teal #2ec4b6, navy/blue #4cc9f0). Background #0d1117
    matches GitHub dark mode.
"""
from __future__ import annotations

import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.path import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REPO_ROOT = "/home/abhi/git/SuperCoach-VIA"
ASSETS_DIR = os.path.join(REPO_ROOT, "assets", "charts")
DATA_DIR = os.path.join(REPO_ROOT, "data")
PLAYER_DATA_DIR = os.path.join(DATA_DIR, "player_data")
MATCHES_DIR = os.path.join(DATA_DIR, "matches")

# Banner-aligned palette. Background hex is GitHub's dark canvas.
BG = "#0d1117"
PANEL_BG = "#161b22"
GOLD = "#f4c430"
TEAL = "#2ec4b6"
SKY = "#4cc9f0"
SOFT_GREY = "#8b949e"
GRID = "#30363d"

# 18-team palette for radar / heatmap rows. Picks readable colours on dark.
TEAM_PALETTE = [
    "#f4c430", "#2ec4b6", "#4cc9f0", "#f72585", "#7209b7", "#b5179e",
    "#80ed99", "#ff9f1c", "#ffb4a2", "#9b5de5", "#43aa8b", "#ffd166",
    "#ef476f", "#06d6a0", "#118ab2", "#f08080", "#c77dff", "#90e0ef",
]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _apply_dark_style() -> None:
    """Apply the project's standard dark style. Called by every chart fn so
    callers can mix-and-match without remembering setup order."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": GRID,
        "grid.alpha": 0.4,
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
    })


# ---------------------------------------------------------------------------
# Chart 1 — era scoring trends
# ---------------------------------------------------------------------------
def _avg_team_score_per_year(start: int = 1965, end: int = 2025) -> pd.DataFrame:
    """Compute mean per-team final score per year from match files.

    Returns a DataFrame with columns [year, avg_team_score, n_matches].
    Years with no match file (or no parseable rows) are skipped.
    """
    rows: List[Dict[str, float]] = []
    for y in range(start, end + 1):
        f = os.path.join(MATCHES_DIR, f"matches_{y}.csv")
        if not os.path.exists(f):
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        # Team_1 / team_2 final goals and behinds — coerce numeric.
        for col in ("team_1_final_goals", "team_1_final_behinds",
                    "team_2_final_goals", "team_2_final_behinds"):
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["team_1_final_goals", "team_1_final_behinds",
                                "team_2_final_goals", "team_2_final_behinds"])
        if df.empty:
            continue
        s1 = df["team_1_final_goals"] * 6 + df["team_1_final_behinds"]
        s2 = df["team_2_final_goals"] * 6 + df["team_2_final_behinds"]
        avg = float(pd.concat([s1, s2]).mean())
        rows.append({"year": y, "avg_team_score": avg, "n_matches": int(len(df))})
    return pd.DataFrame(rows)


def chart_era_scoring_trends() -> str:
    """Generate era_scoring_trends.png. Returns full path written."""
    _apply_dark_style()
    df = _avg_team_score_per_year(1965, 2025)
    if df.empty:
        sys.exit("no match data found for scoring trend chart")

    out = os.path.join(ASSETS_DIR, "era_scoring_trends.png")
    fig, ax = plt.subplots(figsize=(14, 6.5))

    ax.plot(df["year"], df["avg_team_score"],
            color=GOLD, lw=2.4, marker="o", markersize=3.5,
            markerfacecolor=GOLD, markeredgecolor=BG, alpha=0.95)
    # 5-year rolling smooth as a secondary translucent line — smooths noise.
    smooth = df["avg_team_score"].rolling(5, center=True, min_periods=2).mean()
    ax.plot(df["year"], smooth, color=GOLD, lw=4.5, alpha=0.18)

    ax.set_title("AFL team scoring — 1965 to 2025",
                 color="white", pad=14)
    ax.set_xlabel("Year", color="white")
    ax.set_ylabel("Average score per team per game", color="white")
    ax.grid(True, ls=":", lw=0.8, alpha=0.35)
    ax.tick_params(colors="white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(GRID)

    # Find the 1980s peak and the modern level for annotation.
    peak_window = df[(df["year"] >= 1980) & (df["year"] <= 1995)]
    if not peak_window.empty:
        peak_row = peak_window.loc[peak_window["avg_team_score"].idxmax()]
        ax.annotate(
            f"1980s/90s peak\n~{peak_row['avg_team_score']:.0f} pts/g ({int(peak_row['year'])})",
            xy=(peak_row["year"], peak_row["avg_team_score"]),
            xytext=(peak_row["year"] - 2, peak_row["avg_team_score"] + 12),
            color="white", fontsize=16, ha="center",
            arrowprops=dict(arrowstyle="->", color=SOFT_GREY, lw=1.0, alpha=0.8),
            bbox=dict(boxstyle="round,pad=0.4", fc=PANEL_BG, ec=GOLD, lw=1.0, alpha=0.95),
        )

    modern_window = df[df["year"] >= 2018]
    if not modern_window.empty:
        modern_row = modern_window.iloc[-1]
        ax.annotate(
            f"Modern decline\n~{modern_row['avg_team_score']:.0f} pts/g ({int(modern_row['year'])})",
            xy=(modern_row["year"], modern_row["avg_team_score"]),
            xytext=(modern_row["year"] - 14, modern_row["avg_team_score"] - 18),
            color="white", fontsize=16, ha="center",
            arrowprops=dict(arrowstyle="->", color=SOFT_GREY, lw=1.0, alpha=0.8),
            bbox=dict(boxstyle="round,pad=0.4", fc=PANEL_BG, ec=TEAL, lw=1.0, alpha=0.95),
        )

    fig.text(0.99, 0.02,
             "Source: data/matches/  |  points = goals*6 + behinds, averaged across both teams per match",
             ha="right", va="bottom", color=SOFT_GREY, fontsize=14, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Chart 2 — era stat evolution
# ---------------------------------------------------------------------------
ERAS = ["pre-1965", "1965-1990", "1991-2010", "2011-present"]
ERA_STATS_OF_INTEREST = [
    ("kicks", "Kicks"),
    ("handballs", "Handballs"),
    ("tackles", "Tackles"),
    ("clearances", "Clearances"),
]
ERA_STAT_COLORS = {
    "kicks": GOLD,
    "handballs": TEAL,
    "tackles": SKY,
    "clearances": "#f72585",
}


def _load_era_stats() -> pd.DataFrame:
    f = os.path.join(DATA_DIR, "era_stats.csv")
    if not os.path.exists(f):
        sys.exit(f"missing {f} — run era_based_statistical_analysis.py first")
    df = pd.read_csv(f)
    # Pivot: era x metric -> mean_per_game
    pv = df.pivot_table(index="era", columns="metric", values="mean_per_game", aggfunc="first")
    return pv.reindex(ERAS)


def chart_era_stat_evolution() -> str:
    _apply_dark_style()
    pv = _load_era_stats()
    out = os.path.join(ASSETS_DIR, "era_stat_evolution.png")

    fig, ax = plt.subplots(figsize=(14, 6.5))
    n_groups = len(ERAS)
    n_stats = len(ERA_STATS_OF_INTEREST)
    bar_w = 0.8 / n_stats
    x = np.arange(n_groups)

    for i, (col, label) in enumerate(ERA_STATS_OF_INTEREST):
        if col not in pv.columns:
            vals = np.zeros(n_groups)
        else:
            vals = pv[col].fillna(0.0).values
        offset = (i - (n_stats - 1) / 2) * bar_w
        bars = ax.bar(x + offset, vals, bar_w,
                      label=label, color=ERA_STAT_COLORS[col],
                      edgecolor=BG, linewidth=1.2, alpha=0.92)
        # Label each bar — show "n/a" when data not recorded that era.
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.25,
                        f"{v:.1f}", ha="center", va="bottom",
                        color="white", fontsize=15, alpha=0.9)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.4,
                        "n/a", ha="center", va="bottom",
                        color=SOFT_GREY, fontsize=14, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(ERAS, color="white")
    ax.set_ylabel("Per-player average per game", color="white")
    ax.set_title("How player workload has changed — kicks, handballs, tackles, clearances by era",
                 color="white", pad=12)
    ax.grid(True, axis="y", ls=":", lw=0.8, alpha=0.30)
    ax.set_axisbelow(True)
    ax.tick_params(colors="white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(GRID)

    leg = ax.legend(loc="upper left", frameon=True, facecolor=PANEL_BG,
                    edgecolor=GRID, labelcolor="white", framealpha=0.85)
    for text in leg.get_texts():
        text.set_color("white")

    fig.text(0.99, 0.02,
             "Source: data/era_stats.csv  |  'n/a' = stat not tracked in that era",
             ha="right", va="bottom", color=SOFT_GREY, fontsize=14, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Team data loading (shared by charts 3, 4, 5)
# ---------------------------------------------------------------------------
TEAM_LOAD_COLS = [
    "team", "year", "round", "opponent",
    "kicks", "handballs", "disposals", "marks", "goals", "tackles",
    "clearances", "inside_50s", "rebound_50s", "contested_possessions",
    "uncontested_possessions", "hit_outs", "marks_inside_50",
    "contested_marks",
]


def _load_team_year_player_games(years: List[int]) -> pd.DataFrame:
    """Load all player-game rows where year ∈ `years`. Returns concatenated
    long-format dataframe; missing stat cells are NOT filled here — caller
    decides."""
    pattern = os.path.join(PLAYER_DATA_DIR, "*_performance_details.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no player files in {PLAYER_DATA_DIR}")
    frames: List[pd.DataFrame] = []
    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False,
                             usecols=lambda c: c in TEAM_LOAD_COLS)
            if df.empty or "year" not in df.columns or "team" not in df.columns:
                continue
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df.dropna(subset=["year"])
            df["year"] = df["year"].astype(int)
            df = df[df["year"].isin(years)]
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        sys.exit(f"no player rows for years {years}")
    return pd.concat(frames, ignore_index=True)


def _aggregate_team_game(games: pd.DataFrame, stats: List[str]) -> pd.DataFrame:
    """Sum player rows to one row per (team, year, round, opponent)."""
    g = games.copy()
    for c in stats:
        if c not in g.columns:
            g[c] = 0.0
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0.0)
    g["round_str"] = g["round"].astype(str)
    return g.groupby(["team", "year", "round_str", "opponent"], as_index=False)[stats].sum()


def _detect_current_year() -> int:
    """Pick the most recent year in player data with at least 50 rows."""
    pattern = os.path.join(PLAYER_DATA_DIR, "*_performance_details.csv")
    files = sorted(glob.glob(pattern))
    counts: Dict[int, int] = {}
    for path in files[:2000]:
        try:
            df = pd.read_csv(path, low_memory=False, usecols=lambda c: c == "year")
            if df.empty:
                continue
            df["year"] = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
            for y, n in df["year"].value_counts().items():
                counts[int(y)] = counts.get(int(y), 0) + int(n)
        except Exception:
            continue
    candidates = sorted(counts.keys(), reverse=True)
    for y in candidates:
        if counts[y] >= 50:
            return y
    return candidates[0] if candidates else 2025


# ---------------------------------------------------------------------------
# Chart 3 — current-year top-6 radar
# ---------------------------------------------------------------------------
RADAR_DIMENSIONS = [
    ("disposals", "Disposals"),
    ("goals", "Goals"),
    ("tackles", "Tackles"),
    ("clearances", "Clearances"),
    ("inside_50s", "Inside 50s"),
    ("rebound_50s", "Rebound 50s"),
]


def chart_team_radar(year: int = None) -> str:
    """Top-6 teams in `year` across 6 radar dimensions, normalised to the
    league min/max so the smallest team in each axis sits at 0 and the
    largest at 1.

    Top-6 = teams with highest disposals/game, matching the afl-insights.md
    summary table ordering.
    """
    _apply_dark_style()
    if year is None:
        year = _detect_current_year()

    stats = [s for s, _ in RADAR_DIMENSIONS]
    games = _load_team_year_player_games([year])
    team_game = _aggregate_team_game(games, stats)
    team_means = team_game.groupby("team", as_index=False)[stats].mean()

    # Normalise per-stat to 0..1 across all 18 teams.
    normed = team_means.copy()
    for s in stats:
        col_min = team_means[s].min()
        col_max = team_means[s].max()
        rng = col_max - col_min
        normed[s] = (team_means[s] - col_min) / rng if rng > 0 else 0.0

    # Top-6 by raw disposals.
    top6 = team_means.sort_values("disposals", ascending=False).head(6)["team"].tolist()

    # Radar axes setup.
    n_dims = len(RADAR_DIMENSIONS)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    # 13x10 canvas with manual axis placement so the title sits clear of the
    # top "Disposals" label and the legend has room for full team names.
    fig = plt.figure(figsize=(13, 10), facecolor=BG)
    # Polar axes positioned to leave room for: title (top), legend (right),
    # and source caption (bottom). [left, bottom, width, height] in 0..1.
    ax = fig.add_axes([0.04, 0.08, 0.55, 0.78], polar=True)
    ax.set_facecolor(BG)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Dim grid + labels — pad labels out so they sit clear of the polygon.
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([lbl for _, lbl in RADAR_DIMENSIONS], color="white", fontsize=18)
    ax.tick_params(axis="x", pad=18)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color=SOFT_GREY, fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.spines["polar"].set_color(GRID)
    ax.grid(color=GRID, alpha=0.5, lw=0.8)

    out = os.path.join(ASSETS_DIR, f"team_{year}_radar.png")

    for i, team in enumerate(top6):
        row = normed[normed["team"] == team].iloc[0]
        vals = [float(row[s]) for s in stats]
        vals += vals[:1]
        col = TEAM_PALETTE[i % len(TEAM_PALETTE)]
        ax.plot(angles, vals, color=col, lw=2.6, label=team, alpha=0.95)
        ax.fill(angles, vals, color=col, alpha=0.10)

    # Title + subtitle drawn at the very top of the figure, in clear space
    # above the polar axes (which top out at y ≈ 0.86).
    fig.text(0.5, 0.955,
             f"Top 6 teams of {year} — radar across 6 stat dimensions",
             color="white", fontsize=23, fontweight="bold", ha="center")
    fig.text(0.5, 0.918,
             "Normalised 0-1 vs all 18 teams (1.0 = league best on that axis)",
             color=SOFT_GREY, fontsize=18, ha="center")

    leg = ax.legend(loc="center left", bbox_to_anchor=(1.18, 0.5),
                    frameon=True, facecolor=PANEL_BG, edgecolor=GRID,
                    labelcolor="white", framealpha=0.9, fontsize=18,
                    title="Team (top 6 by disposals)", title_fontsize=20)
    leg.get_title().set_color("white")
    for text in leg.get_texts():
        text.set_color("white")

    fig.text(0.99, 0.02,
             f"Source: data/player_data/ year={year}  |  top-6 by disposals/g",
             ha="right", va="bottom", color=SOFT_GREY, fontsize=15, alpha=0.7)
    fig.savefig(out, dpi=180, facecolor=BG, edgecolor=BG)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Chart 4 — current-year team x stat rank heatmap
# ---------------------------------------------------------------------------
HEATMAP_STATS = [
    ("disposals", "Disposals"),
    ("goals", "Goals"),
    ("tackles", "Tackles"),
    ("clearances", "Clearances"),
    ("inside_50s", "Inside 50s"),
    ("marks", "Marks"),
    ("contested_possessions", "Contested poss."),
    ("rebound_50s", "Rebound 50s"),
]


def chart_team_heatmap(year: int = None) -> str:
    """Heatmap of all 18 teams x 8 stats showing rank (1=best, 18=worst).
    Colour scale is reversed RdYlGn so 1 reads as green, 18 as red."""
    _apply_dark_style()
    if year is None:
        year = _detect_current_year()

    stats = [s for s, _ in HEATMAP_STATS]
    games = _load_team_year_player_games([year])
    team_game = _aggregate_team_game(games, stats)
    team_means = team_game.groupby("team", as_index=False)[stats].mean()

    # Rank: 1 = highest mean (= best for these stats — all are higher-better).
    ranks = team_means.copy()
    for s in stats:
        ranks[s] = team_means[s].rank(ascending=False, method="min").astype(int)

    # Sort teams by disposals rank (1 at top).
    ranks = ranks.sort_values("disposals").reset_index(drop=True)
    teams = ranks["team"].tolist()
    matrix = ranks[stats].values  # rows=teams, cols=stats

    out = os.path.join(ASSETS_DIR, f"team_{year}_heatmap.png")
    fig, ax = plt.subplots(figsize=(14, 7))

    cmap = plt.get_cmap("RdYlGn_r")
    # Invert so rank 1 -> low colour value (green), rank 18 -> high (red).
    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                   vmin=1, vmax=len(teams))

    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels([lbl for _, lbl in HEATMAP_STATS],
                       rotation=20, ha="right", color="white", fontsize=18)
    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(teams, color="white", fontsize=16)

    # Rank text in each cell. White on dark cells, black on light.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            rk = int(matrix[i, j])
            # Mid-range ranks get black text, extremes get white for contrast.
            txt_col = "black" if 5 < rk < 14 else "white"
            ax.text(j, i, str(rk), ha="center", va="center",
                    color=txt_col, fontsize=16, fontweight="bold")

    ax.set_title(f"{year} season-to-date — team rank by stat (1 = league best, 18 = worst)",
                 color="white", pad=14, fontsize=20, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color(GRID)

    # Compact colourbar — maps green (1) to red (18).
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, ticks=[1, 9, 18])
    cbar.ax.set_yticklabels(["1 (best)", "9", "18 (worst)"], color="white")
    cbar.outline.set_color(GRID)
    cbar.ax.tick_params(colors="white")

    fig.text(0.99, 0.01,
             f"Source: data/player_data/ year={year}  |  rows sorted by disposals/g rank",
             ha="right", va="bottom", color=SOFT_GREY, fontsize=14, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Chart 5 — 5-year team style scatter (handball ratio vs tackles)
# ---------------------------------------------------------------------------
def chart_team_style_scatter(current_year: int = None) -> str:
    """Scatter: x = handball ratio (%), y = tackles per game.
    Window = 5 seasons immediately prior to `current_year`.
    Quadrant dividers split at the league median on each axis.
    """
    _apply_dark_style()
    if current_year is None:
        current_year = _detect_current_year()
    years = list(range(current_year - 5, current_year))

    stats = ["handballs", "disposals", "tackles"]
    games = _load_team_year_player_games(years)
    team_game = _aggregate_team_game(games, stats)
    team_year = team_game.groupby(["team", "year"], as_index=False)[stats].mean()

    # 5-year mean per team (mean of yearly means — equal weight to each year,
    # matches the methodology used in update_team_analysis.py).
    profile = team_year.groupby("team", as_index=False)[stats].mean()
    profile["hb_ratio_pct"] = (profile["handballs"] / profile["disposals"].replace(0, np.nan)) * 100
    profile = profile.dropna(subset=["hb_ratio_pct"])

    out = os.path.join(ASSETS_DIR, f"team_{current_year}_style_scatter.png")
    fig, ax = plt.subplots(figsize=(10, 8))

    x = profile["hb_ratio_pct"].values
    y = profile["tackles"].values
    teams = profile["team"].tolist()

    # Quadrant dividers at league medians.
    x_med = float(np.median(x))
    y_med = float(np.median(y))
    ax.axvline(x_med, color=SOFT_GREY, lw=1.0, ls="--", alpha=0.5)
    ax.axhline(y_med, color=SOFT_GREY, lw=1.0, ls="--", alpha=0.5)

    # Quadrant labels — corners of the plot, partly transparent.
    pad_x = (x.max() - x.min()) * 0.04
    pad_y = (y.max() - y.min()) * 0.04
    quadrants = [
        # (x, y, ha, va, label)
        (x.max() - pad_x, y.max() - pad_y, "right", "top",  "Handball + Physical"),
        (x.min() + pad_x, y.max() - pad_y, "left",  "top",  "Kick + Physical"),
        (x.max() - pad_x, y.min() + pad_y, "right", "bottom", "Handball + Spread"),
        (x.min() + pad_x, y.min() + pad_y, "left",  "bottom", "Kick + Spread"),
    ]
    for qx, qy, ha, va, label in quadrants:
        ax.text(qx, qy, label, ha=ha, va=va, color=SOFT_GREY,
                fontsize=16, fontstyle="italic", alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_BG, ec=GRID, lw=0.8, alpha=0.6))

    ax.scatter(x, y, s=160, c=GOLD, edgecolors="white", linewidths=1.0,
               alpha=0.9, zorder=3)

    # Team labels — slight offset to the right of each dot.
    label_offset_x = (x.max() - x.min()) * 0.012
    for xi, yi, name in zip(x, y, teams):
        ax.annotate(name, (xi + label_offset_x, yi),
                    color="white", fontsize=15, ha="left", va="center",
                    alpha=0.95)

    ax.set_xlabel("Handball ratio (% of disposals by hand)", color="white", fontsize=18)
    ax.set_ylabel("Tackles per game", color="white", fontsize=18)
    ax.set_title(
        f"Team playing styles, {years[0]}-{years[-1]} 5-year average\n"
        "Handball-vs-kick build-up x ground-ball pressure",
        color="white", pad=14, fontsize=20, fontweight="bold")
    ax.grid(True, ls=":", lw=0.8, alpha=0.30)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(GRID)

    fig.text(0.99, 0.02,
             f"Source: data/player_data/ years={years[0]}-{years[-1]}  |  dashed lines = league median",
             ha="right", va="bottom", color=SOFT_GREY, fontsize=14, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Chart A — All-time top 10 horizontal bar
# ---------------------------------------------------------------------------
TOP_LEVEL_TOP100 = os.path.join(REPO_ROOT, "all_time_top_100.csv")
TOP100_INTERNAL = os.path.join(DATA_DIR, "top100", "all_time_top_100.csv")

POSITION_COLORS = {
    "key_forward": GOLD,
    "forward_mid": TEAL,
    "midfielder": SKY,
    "backline": "#9b5de5",
}


def _career_position_for_player(player_stem: str) -> str:
    """Look up career goals/game for a player file stem to assign a position
    bucket. Mirrors `_career_position_group` in top_players_comprehensive.py.

    Returns one of {'key_forward', 'forward_mid', 'midfielder', 'backline'}.
    Falls back to 'midfielder' if the file is unreadable.
    """
    path = os.path.join(PLAYER_DATA_DIR, f"{player_stem}_performance_details.csv")
    if not os.path.exists(path):
        return "midfielder"
    try:
        df = pd.read_csv(path, low_memory=False, usecols=lambda c: c in {"goals"})
        if df.empty or "goals" not in df.columns:
            return "midfielder"
        goals = pd.to_numeric(df["goals"], errors="coerce").fillna(0.0)
        n_games = len(goals)
        if n_games == 0:
            return "midfielder"
        gpg = float(goals.sum()) / float(n_games)
    except Exception:
        return "midfielder"
    if gpg >= 3.0:
        return "key_forward"
    if gpg >= 0.80:
        return "forward_mid"
    if gpg >= 0.30:
        return "midfielder"
    return "backline"


def _player_name_to_stem(name: str) -> Optional[str]:
    """Resolve `Wayne Carey` -> `carey_wayne_<dob>` by globbing the player_data
    directory. If multiple matches, prefer the one with the most rows.
    """
    parts = name.strip().split()
    if len(parts) < 2:
        return None
    last = parts[-1].lower()
    first = " ".join(parts[:-1]).lower().replace(" ", "_")
    # Pattern: <last>_<first>_<digits>_performance_details.csv
    pattern = os.path.join(PLAYER_DATA_DIR, f"{last}_{first}_*_performance_details.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        # Try first/last swap (some files may invert).
        pattern2 = os.path.join(PLAYER_DATA_DIR, f"{first}_{last}_*_performance_details.csv")
        matches = sorted(glob.glob(pattern2))
    if not matches:
        return None
    if len(matches) == 1:
        return os.path.basename(matches[0]).replace("_performance_details.csv", "")
    # Multiple: pick longest file (most rows, likely the canonical career).
    best = max(matches, key=lambda p: os.path.getsize(p))
    return os.path.basename(best).replace("_performance_details.csv", "")


def _load_internal_top100() -> pd.DataFrame:
    """Internal CSV has player_stem + score; preferred for the bar chart since
    the score is already canonical."""
    if os.path.exists(TOP100_INTERNAL):
        df = pd.read_csv(TOP100_INTERNAL)
        df = df.rename(columns={"player": "stem"})
        df["rank"] = np.arange(1, len(df) + 1)
        return df
    sys.exit(f"missing {TOP100_INTERNAL} — run top_players_comprehensive.py first")


def _load_top_level_top100() -> pd.DataFrame:
    """Top-level formatted CSV has display name + serial. Used to map stem ->
    pretty name when both files are available; falls back to stem-prettify."""
    if not os.path.exists(TOP_LEVEL_TOP100):
        return pd.DataFrame()
    try:
        df = pd.read_csv(TOP_LEVEL_TOP100)
        # Columns are typically "Serial Number","Player Name","Footy Teams","Comment"
        if "Serial Number" in df.columns and "Player Name" in df.columns:
            df = df.rename(columns={"Serial Number": "rank", "Player Name": "name"})
            df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
            return df[["rank", "name"]]
    except Exception:
        pass
    return pd.DataFrame()


def _stem_to_pretty(stem: str) -> str:
    parts = stem.split("_")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    if len(parts) >= 2:
        last, first = parts[0], parts[1]
        return f"{first.title()} {last.title()}"
    return stem.replace("_", " ").title()


def chart_top10_alltime() -> str:
    """Horizontal bar chart of the all-time top 10. Bars coloured by career
    position group, annotated with the score, ordered with #1 at the bottom
    so the longest bar is most prominent."""
    _apply_dark_style()
    internal = _load_internal_top100()
    top10 = internal.head(10).copy()

    # Pretty names from the top-level CSV when available; fallback to stem.
    pretty = _load_top_level_top100()
    name_by_rank: Dict[int, str] = {}
    if not pretty.empty:
        for _, r in pretty.iterrows():
            try:
                name_by_rank[int(r["rank"])] = str(r["name"])
            except (TypeError, ValueError):
                continue

    top10["name"] = [
        name_by_rank.get(int(r), _stem_to_pretty(s))
        for r, s in zip(top10["rank"], top10["stem"])
    ]
    top10["pos"] = [_career_position_for_player(s) for s in top10["stem"]]
    top10["color"] = [POSITION_COLORS.get(p, SKY) for p in top10["pos"]]

    # Reverse order: rank 10 at top, rank 1 at bottom.
    plot_df = top10.sort_values("rank", ascending=False).reset_index(drop=True)

    out = os.path.join(ASSETS_DIR, "top10_alltime.png")
    fig, ax = plt.subplots(figsize=(13, 8))
    y_pos = np.arange(len(plot_df))
    bars = ax.barh(y_pos, plot_df["all_time_score"].values,
                   color=plot_df["color"].tolist(),
                   edgecolor=BG, linewidth=1.0, alpha=0.95)

    for i, (bar, val, name, rk) in enumerate(zip(
            bars, plot_df["all_time_score"].values, plot_df["name"].values,
            plot_df["rank"].values)):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f" {val:.2f}", va="center", ha="left",
                color="white", fontsize=16, fontweight="bold")

    labels = [f"#{int(r)}  {n}" for r, n in zip(plot_df["rank"], plot_df["name"])]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="white", fontsize=18)
    ax.set_xlabel("All-time score (composite of best-8 seasons + bonuses)",
                  color="white", fontsize=18)
    ax.set_title("All-Time Top 10 AFL/VFL Players",
                 color="white", fontsize=21, pad=14, fontweight="bold")
    ax.grid(True, axis="x", ls=":", lw=0.8, alpha=0.30)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors="white")

    # Legend explaining the position colours actually present in the top 10.
    present = [p for p in ("key_forward", "forward_mid", "midfielder", "backline")
               if p in set(plot_df["pos"])]
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=POSITION_COLORS[p],
                      label={"key_forward": "Key forward (≥3 g/g)",
                             "forward_mid": "Forward-mid (0.8–2.99 g/g)",
                             "midfielder": "Midfielder (0.3–0.79 g/g)",
                             "backline": "Defender / ruck (<0.3 g/g)"}[p])
        for p in present
    ]
    # Place legend outside the plotting area to the right so it never
    # collides with score labels.
    leg = ax.legend(handles=legend_handles, loc="upper left",
                    bbox_to_anchor=(1.005, 1.0),
                    frameon=True, facecolor=PANEL_BG, edgecolor=GRID,
                    framealpha=0.95, fontsize=16, borderpad=0.8,
                    labelspacing=1.0)
    for text in leg.get_texts():
        text.set_color("white")

    # Modest headroom on the x-axis for the inline score labels.
    ax.set_xlim(0, float(plot_df["all_time_score"].max()) * 1.10)

    fig.text(0.99, 0.015,
             "Source: data/top100/all_time_top_100.csv  |  position bucket from career goals/game",
             ha="right", va="bottom", color=SOFT_GREY, fontsize=14, alpha=0.7)
    fig.tight_layout(rect=[0, 0.05, 0.82, 1])
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Chart B — Avg team score by decade
# ---------------------------------------------------------------------------
def chart_scoring_by_decade() -> str:
    """Grouped bar chart of avg team score per decade, 1900s–2020s.
    Colour ramp from dark blue (oldest decade) to bright gold (latest), with
    the peak decade annotated."""
    _apply_dark_style()
    df = _avg_team_score_per_year(1900, 2025)
    if df.empty:
        sys.exit("no match data found for decade chart")

    df["decade"] = (df["year"] // 10) * 10
    decade_means = (
        df.groupby("decade")
        .apply(lambda g: float((g["avg_team_score"] * g["n_matches"]).sum() / g["n_matches"].sum()))
        .reset_index(name="avg_team_score")
    )
    decade_means["decade_label"] = decade_means["decade"].astype(int).astype(str) + "s"
    # Order decades chronologically (already sorted via groupby).
    decade_means = decade_means.sort_values("decade").reset_index(drop=True)

    # Colour ramp: dark blue -> bright gold across the decades.
    n = len(decade_means)
    cmap = plt.get_cmap("YlOrBr")
    # Skip the very pale start of the ramp by sampling 0.30 .. 1.0.
    colors = [cmap(0.30 + 0.70 * (i / max(1, n - 1))) for i in range(n)]

    out = os.path.join(ASSETS_DIR, "scoring_by_decade.png")
    fig, ax = plt.subplots(figsize=(14, 6.5))
    bars = ax.bar(np.arange(n), decade_means["avg_team_score"].values,
                  color=colors, edgecolor=BG, linewidth=1.2, alpha=0.95)

    for bar, v in zip(bars, decade_means["avg_team_score"].values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2,
                f"{v:.1f}", ha="center", va="bottom",
                color="white", fontsize=15, fontweight="bold")

    peak_idx = int(np.argmax(decade_means["avg_team_score"].values))
    peak_lbl = decade_means.iloc[peak_idx]["decade_label"]
    peak_val = decade_means.iloc[peak_idx]["avg_team_score"]
    ax.annotate(
        f"Peak: {peak_lbl} ({peak_val:.1f} pts/g)",
        xy=(peak_idx, peak_val),
        xytext=(peak_idx, peak_val + 14),
        ha="center", color="white", fontsize=18, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.4),
        bbox=dict(boxstyle="round,pad=0.4", fc=PANEL_BG, ec=GOLD, lw=1.0, alpha=0.95),
    )

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(decade_means["decade_label"], color="white")
    ax.set_ylabel("Average team score per game", color="white")
    ax.set_title("Average Team Score by Decade (1900s–2020s)",
                 color="white", pad=14, fontsize=20, fontweight="bold")
    ax.grid(True, axis="y", ls=":", lw=0.8, alpha=0.30)
    ax.set_axisbelow(True)
    ax.tick_params(colors="white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(GRID)
    # Headroom for the annotation arrow.
    ymax = float(decade_means["avg_team_score"].max())
    ax.set_ylim(0, ymax * 1.20)

    fig.text(0.99, 0.015,
             "Source: data/matches/  |  match-weighted decade mean of (goals*6 + behinds) per team",
             ha="right", va="bottom", color=SOFT_GREY, fontsize=14, alpha=0.7)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Chart C — Goals vs disposals scatter (current season)
# ---------------------------------------------------------------------------
def chart_team_goals_disposals(year: int = None) -> str:
    """Scatter: x = avg disposals/g, y = avg goals/g for all 18 teams in `year`.
    Dot size = avg tackles/g. Quadrants split at league medians with labels."""
    _apply_dark_style()
    if year is None:
        year = _detect_current_year()

    stats = ["disposals", "goals", "tackles"]
    games = _load_team_year_player_games([year])
    team_game = _aggregate_team_game(games, stats)
    team_means = team_game.groupby("team", as_index=False)[stats].mean()
    if team_means.empty:
        sys.exit(f"no team data for {year}")

    out = os.path.join(ASSETS_DIR, f"team_{year}_goals_disposals.png")
    fig, ax = plt.subplots(figsize=(13, 8.5))

    x = team_means["disposals"].values
    y = team_means["goals"].values
    s = team_means["tackles"].values
    teams = team_means["team"].tolist()

    x_med = float(np.median(x))
    y_med = float(np.median(y))

    # Pad axis ranges so quadrant labels and the size legend don't collide
    # with team dots near the corners.
    rx = x.max() - x.min()
    ry = y.max() - y.min()
    ax.set_xlim(x.min() - rx * 0.10, x.max() + rx * 0.12)
    ax.set_ylim(y.min() - ry * 0.18, y.max() + ry * 0.18)

    ax.axvline(x_med, color=SOFT_GREY, lw=1.0, ls="--", alpha=0.5)
    ax.axhline(y_med, color=SOFT_GREY, lw=1.0, ls="--", alpha=0.5)

    # Quadrant labels — placed near the four corners using axes coords (so
    # they sit clear of the actual data range).
    quadrants = [
        (0.99, 0.97, "right", "top",    "High disposal + clinical"),
        (0.01, 0.97, "left",  "top",    "Low disposal + clinical"),
        (0.99, 0.03, "right", "bottom", "High disposal + struggling"),
        (0.01, 0.03, "left",  "bottom", "Low disposal + struggling"),
    ]
    for qx, qy, ha, va, label in quadrants:
        ax.text(qx, qy, label, ha=ha, va=va, color=SOFT_GREY,
                fontsize=16, fontstyle="italic", alpha=0.85,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_BG, ec=GRID, lw=0.8, alpha=0.7))

    # Dot size: scale tackles to a readable circle area.
    s_min, s_max = float(np.min(s)), float(np.max(s))
    s_rng = max(s_max - s_min, 1e-6)
    sizes = 80 + ((s - s_min) / s_rng) * 520

    ax.scatter(x, y, s=sizes, c=GOLD, edgecolors="white", linewidths=1.0,
               alpha=0.85, zorder=3)

    label_offset_x = rx * 0.012
    for xi, yi, name in zip(x, y, teams):
        ax.annotate(name, (xi + label_offset_x, yi),
                    color="white", fontsize=15, ha="left", va="center", alpha=0.95)

    ax.set_xlabel("Average disposals per game", color="white", fontsize=18)
    ax.set_ylabel("Average goals per game", color="white", fontsize=18)
    ax.set_title(f"{year} Season: Efficiency vs Ball Use",
                 color="white", pad=14, fontsize=20, fontweight="bold")
    ax.grid(True, ls=":", lw=0.8, alpha=0.30)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(GRID)

    # Legend explaining dot size — placed outside the data area, pinned to
    # the figure right edge so it never overlaps team dots.
    handles = [
        plt.scatter([], [], s=80,  c=GOLD, edgecolors="white", linewidths=1.0,
                    alpha=0.85, label=f"Low tackles/g (~{s_min:.0f})"),
        plt.scatter([], [], s=300, c=GOLD, edgecolors="white", linewidths=1.0,
                    alpha=0.85, label="Mid tackles/g"),
        plt.scatter([], [], s=600, c=GOLD, edgecolors="white", linewidths=1.0,
                    alpha=0.85, label=f"High tackles/g (~{s_max:.0f})"),
    ]
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(1.005, 1.0), frameon=True,
                    facecolor=PANEL_BG, edgecolor=GRID, framealpha=0.95,
                    fontsize=15, title="Dot size = tackles/g", title_fontsize=19,
                    borderpad=1.0, labelspacing=1.4, handletextpad=1.0)
    leg.get_title().set_color("white")
    for text in leg.get_texts():
        text.set_color("white")

    fig.text(0.99, 0.015,
             f"Source: data/player_data/ year={year}  |  dashed lines = league median",
             ha="right", va="bottom", color=SOFT_GREY, fontsize=14, alpha=0.7)
    fig.tight_layout(rect=[0, 0.04, 0.86, 1])
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Chart D — Tackles + clearances dual-axis line
# ---------------------------------------------------------------------------
def chart_era_tackles_clearances() -> str:
    """Dual-axis line chart: tackles per player per game (1987-) on the
    left axis, clearances per player per game (1998-) on the right.
    Built off `data/era_yearly_trends.csv`."""
    _apply_dark_style()
    f = os.path.join(DATA_DIR, "era_yearly_trends.csv")
    if not os.path.exists(f):
        sys.exit(f"missing {f} — run era_based_statistical_analysis.py first")
    df = pd.read_csv(f)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 1987) & (df["year"] <= 2025)].sort_values("year")

    tackles_df = df.dropna(subset=["tackles"])[["year", "tackles"]]
    clearances_df = df.dropna(subset=["clearances"])[["year", "clearances"]]

    out = os.path.join(ASSETS_DIR, "era_tackles_clearances.png")
    fig, ax1 = plt.subplots(figsize=(14, 6.5))

    ax1.plot(tackles_df["year"], tackles_df["tackles"],
             color=GOLD, lw=2.6, marker="o", markersize=4,
             markeredgecolor=BG, label="Tackles / player / game")
    ax1.set_xlabel("Year", color="white")
    ax1.set_ylabel("Tackles per player per game", color=GOLD)
    ax1.tick_params(axis="y", colors=GOLD)
    ax1.tick_params(axis="x", colors="white")
    ax1.grid(True, ls=":", lw=0.8, alpha=0.30)
    ax1.set_axisbelow(True)
    for spine in ("top",):
        ax1.spines[spine].set_visible(False)
    ax1.spines["left"].set_color(GOLD)
    ax1.spines["bottom"].set_color(GRID)

    ax2 = ax1.twinx()
    ax2.plot(clearances_df["year"], clearances_df["clearances"],
             color=TEAL, lw=2.6, marker="s", markersize=4,
             markeredgecolor=BG, label="Clearances / player / game")
    ax2.set_ylabel("Clearances per player per game", color=TEAL)
    ax2.tick_params(axis="y", colors=TEAL)
    for spine in ("top",):
        ax2.spines[spine].set_visible(False)
    ax2.spines["right"].set_color(TEAL)

    ax1.set_title(
        "The Intensification of AFL — Tackles & Clearances Over Time",
        color="white", pad=14, fontsize=20, fontweight="bold")

    # Combined legend (both axes).
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(h1 + h2, l1 + l2, loc="upper left",
                     frameon=True, facecolor=PANEL_BG, edgecolor=GRID,
                     framealpha=0.9, fontsize=16)
    for text in leg.get_texts():
        text.set_color("white")

    fig.text(0.99, 0.02,
             "Source: data/era_yearly_trends.csv  |  tackles tracked from 1987, clearances from 1998",
             ha="right", va="bottom", color=SOFT_GREY, fontsize=14, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Chart E — Top 100 position breakdown (donut + bar)
# ---------------------------------------------------------------------------
def chart_top100_position_breakdown() -> str:
    """Two-panel: (left) donut of top 100 by position group, (right) bar of
    avg `all_time_score` by position group."""
    _apply_dark_style()
    internal = _load_internal_top100()
    internal["pos"] = [_career_position_for_player(s) for s in internal["stem"]]

    order = ["key_forward", "forward_mid", "midfielder", "backline"]
    label_map = {
        "key_forward": "Key forwards",
        "forward_mid": "Forward-mids",
        "midfielder": "Midfielders",
        "backline": "Defenders / rucks",
    }

    counts = internal["pos"].value_counts().reindex(order).fillna(0).astype(int)
    avgs = internal.groupby("pos")["all_time_score"].mean().reindex(order).fillna(0.0)

    # Drop empty categories so the donut isn't littered with 0-slices.
    keep = counts > 0
    plot_order = [p for p, k in zip(order, keep) if k]
    plot_counts = counts[keep].values
    plot_avgs = avgs[keep].values
    plot_labels = [label_map[p] for p in plot_order]
    plot_colors = [POSITION_COLORS[p] for p in plot_order]

    out = os.path.join(ASSETS_DIR, "top100_position_breakdown.png")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [1.05, 1.0]})

    # Donut.
    ax_d = axes[0]
    wedges, _texts, _autotexts = ax_d.pie(
        plot_counts,
        labels=None,
        colors=plot_colors,
        startangle=90,
        wedgeprops=dict(width=0.40, edgecolor=BG, linewidth=2),
        autopct=lambda v: f"{v:.0f}%" if v >= 4 else "",
        pctdistance=0.80,
        textprops=dict(color="white", fontsize=18, fontweight="bold"),
    )
    # Centre text — total count.
    ax_d.text(0, 0, f"{int(plot_counts.sum())}\nplayers",
              ha="center", va="center", color="white", fontsize=21,
              fontweight="bold")
    ax_d.set_title("Top 100 by position group", color="white", pad=12,
                   fontsize=19, fontweight="bold")

    # Legend with raw counts beside each label.
    legend_labels = [f"{lbl} (n={c})" for lbl, c in zip(plot_labels, plot_counts)]
    leg = ax_d.legend(wedges, legend_labels, loc="center left",
                      bbox_to_anchor=(-0.15, -0.05),
                      frameon=True, facecolor=PANEL_BG, edgecolor=GRID,
                      framealpha=0.9, fontsize=16)
    for text in leg.get_texts():
        text.set_color("white")

    # Bar (avg score by position).
    ax_b = axes[1]
    y_pos = np.arange(len(plot_labels))
    bars = ax_b.barh(y_pos, plot_avgs, color=plot_colors,
                     edgecolor=BG, linewidth=1.2, alpha=0.95)
    for bar, v in zip(bars, plot_avgs):
        ax_b.text(v + 0.015, bar.get_y() + bar.get_height() / 2,
                  f"{v:.2f}", va="center", ha="left",
                  color="white", fontsize=16, fontweight="bold")
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(plot_labels, color="white", fontsize=18)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Average all-time score", color="white", fontsize=18)
    ax_b.set_title("Average score by position group", color="white", pad=12,
                   fontsize=19, fontweight="bold")
    ax_b.grid(True, axis="x", ls=":", lw=0.8, alpha=0.30)
    ax_b.set_axisbelow(True)
    for spine in ("top", "right"):
        ax_b.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax_b.spines[spine].set_color(GRID)
    ax_b.tick_params(colors="white")

    fig.suptitle("Top 100 — Position Breakdown",
                 color="white", fontsize=21, fontweight="bold", y=0.995)
    fig.text(0.99, 0.02,
             "Source: data/top100/all_time_top_100.csv  |  position bucket from career goals/game",
             ha="right", va="bottom", color=SOFT_GREY, fontsize=14, alpha=0.7)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Chart F — Round-by-round form trend (current season)
# ---------------------------------------------------------------------------
def chart_team_form_trend(year: int = None) -> str:
    """Line chart of each team's rolling-3 disposals/g across rounds 1..N for
    the current season. Top-3 teams (by season disposal avg) highlighted in
    gold/teal/sky, bottom-3 in red shades, the rest greyed out."""
    _apply_dark_style()
    if year is None:
        year = _detect_current_year()

    stats = ["disposals"]
    games = _load_team_year_player_games([year])
    team_game = _aggregate_team_game(games, stats)
    if team_game.empty:
        sys.exit(f"no team data for {year}")
    # Round numeric where possible; drop finals labels (non-numeric).
    team_game["round_num"] = pd.to_numeric(team_game["round_str"], errors="coerce")
    team_game = team_game.dropna(subset=["round_num"]).copy()
    team_game["round_num"] = team_game["round_num"].astype(int)

    # Per (team, round) mean disposals (each cell is already a team-game sum,
    # so the mean across opponents in that round just collapses double-up
    # rounds where a team had a bye-week-replacement).
    by_round = (
        team_game.groupby(["team", "round_num"], as_index=False)["disposals"].mean()
        .sort_values(["team", "round_num"])
    )

    # Season averages — used to pick top-3 and bottom-3.
    season_avg = (
        by_round.groupby("team", as_index=False)["disposals"].mean()
        .sort_values("disposals", ascending=False)
        .reset_index(drop=True)
    )
    n_teams = len(season_avg)
    top3 = season_avg.head(3)["team"].tolist()
    bottom3 = season_avg.tail(3)["team"].tolist()
    mid = [t for t in season_avg["team"] if t not in top3 + bottom3]

    top_colors = [GOLD, TEAL, SKY]
    bottom_colors = ["#ef476f", "#f08080", "#ffb4a2"]

    max_round = int(by_round["round_num"].max())
    out = os.path.join(ASSETS_DIR, f"team_form_trend_{year}.png")
    fig, ax = plt.subplots(figsize=(13, 7.5))

    # Mid teams — light grey, thin lines.
    for t in mid:
        d = by_round[by_round["team"] == t]
        if d.empty:
            continue
        # Rolling-3 with min_periods=1 so early rounds still plot.
        smooth = d["disposals"].rolling(3, min_periods=1).mean().values
        ax.plot(d["round_num"], smooth, color="#444c56", lw=1.2, alpha=0.55)

    # Top 3.
    for color, t in zip(top_colors, top3):
        d = by_round[by_round["team"] == t]
        if d.empty:
            continue
        smooth = d["disposals"].rolling(3, min_periods=1).mean().values
        ax.plot(d["round_num"], smooth, color=color, lw=2.6,
                marker="o", markersize=5, markeredgecolor=BG,
                label=f"{t}", zorder=5)
        # Label at end of line.
        ax.annotate(t, (d["round_num"].iloc[-1], smooth[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=16, fontweight="bold",
                    va="center", ha="left")

    # Bottom 3.
    for color, t in zip(bottom_colors, bottom3):
        d = by_round[by_round["team"] == t]
        if d.empty:
            continue
        smooth = d["disposals"].rolling(3, min_periods=1).mean().values
        ax.plot(d["round_num"], smooth, color=color, lw=2.4,
                marker="o", markersize=5, markeredgecolor=BG,
                linestyle="--", label=f"{t}", zorder=4)
        ax.annotate(t, (d["round_num"].iloc[-1], smooth[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=color, fontsize=16, fontweight="bold",
                    va="center", ha="left")

    ax.set_xlabel("Round", color="white", fontsize=18)
    ax.set_ylabel("Disposals per game (3-round rolling avg)", color="white", fontsize=18)
    ax.set_title(f"{year} Disposal Trends — Rounds 1–{max_round}",
                 color="white", pad=14, fontsize=20, fontweight="bold")
    ax.set_xticks(range(1, max_round + 1))
    ax.grid(True, ls=":", lw=0.8, alpha=0.30)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors="white")
    # Give the right side a bit of headroom for the team-name labels.
    ax.set_xlim(0.7, max_round + 1.6)

    leg = ax.legend(loc="lower left", frameon=True, facecolor=PANEL_BG,
                    edgecolor=GRID, framealpha=0.9, fontsize=15,
                    title="Highlighted: top 3 (solid) + bottom 3 (dashed)",
                    title_fontsize=18)
    leg.get_title().set_color("white")
    for text in leg.get_texts():
        text.set_color("white")

    fig.text(0.99, 0.02,
             f"Source: data/player_data/ year={year}  |  rolling-3 mean of round-by-round disposals/g",
             ha="right", va="bottom", color=SOFT_GREY, fontsize=14, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def regenerate_team_charts(year: int = None) -> List[str]:
    """Regenerate the season-dependent team charts.

    Called by update_team_analysis.py after refreshing docs/afl-insights.md.
    Returns a list of paths actually written; charts that fail are logged
    and skipped rather than aborting the rest.
    """
    _ensure_dir(ASSETS_DIR)
    if year is None:
        year = _detect_current_year()
    targets = [
        ("team_radar",            lambda: chart_team_radar(year)),
        ("team_heatmap",          lambda: chart_team_heatmap(year)),
        ("team_style_scatter",    lambda: chart_team_style_scatter(year)),
        ("team_goals_disposals",  lambda: chart_team_goals_disposals(year)),
        ("team_form_trend",       lambda: chart_team_form_trend(year)),
    ]
    out: List[str] = []
    for name, fn in targets:
        try:
            out.append(fn())
        except SystemExit as e:
            print(f"[charts] skipped {name}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[charts] {name} failed: {e}", file=sys.stderr)
    return out


def regenerate_static_charts() -> List[str]:
    """Regenerate the charts that depend only on era stats / match files."""
    _ensure_dir(ASSETS_DIR)
    targets = [
        ("era_scoring_trends",    chart_era_scoring_trends),
        ("era_stat_evolution",    chart_era_stat_evolution),
        ("scoring_by_decade",     chart_scoring_by_decade),
        ("era_tackles_clearances", chart_era_tackles_clearances),
    ]
    out: List[str] = []
    for name, fn in targets:
        try:
            out.append(fn())
        except SystemExit as e:
            print(f"[charts] skipped {name}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[charts] {name} failed: {e}", file=sys.stderr)
    return out


def regenerate_top100_charts() -> List[str]:
    """Regenerate the charts that depend on the all-time top 100 ranking."""
    _ensure_dir(ASSETS_DIR)
    targets = [
        ("top10_alltime",             chart_top10_alltime),
        ("top100_position_breakdown", chart_top100_position_breakdown),
    ]
    out: List[str] = []
    for name, fn in targets:
        try:
            out.append(fn())
        except SystemExit as e:
            print(f"[charts] skipped {name}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[charts] {name} failed: {e}", file=sys.stderr)
    return out


def main() -> None:
    _ensure_dir(ASSETS_DIR)
    print(f"[1/11] era scoring trends      -> {chart_era_scoring_trends()}")
    print(f"[2/11] era stat evolution      -> {chart_era_stat_evolution()}")
    print(f"[3/11] scoring by decade       -> {chart_scoring_by_decade()}")
    print(f"[4/11] era tackles+clearances  -> {chart_era_tackles_clearances()}")
    year = _detect_current_year()
    print(f"       detected current year = {year}")
    print(f"[5/11] {year} top-6 radar        -> {chart_team_radar(year)}")
    print(f"[6/11] {year} rank heatmap       -> {chart_team_heatmap(year)}")
    print(f"[7/11] 5-year style scatter    -> {chart_team_style_scatter(year)}")
    print(f"[8/11] {year} goals vs disposals -> {chart_team_goals_disposals(year)}")
    print(f"[9/11] {year} form trend          -> {chart_team_form_trend(year)}")
    print(f"[10/11] all-time top 10        -> {chart_top10_alltime()}")
    print(f"[11/11] top 100 positions      -> {chart_top100_position_breakdown()}")


if __name__ == "__main__":
    main()
