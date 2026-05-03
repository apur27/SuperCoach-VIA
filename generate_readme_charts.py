#!/usr/bin/env python3
"""
generate_readme_charts.py
=========================

Generate the five data-visualisation charts that are embedded in README.md.

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

`update_team_analysis.py` calls `regenerate_team_charts()` after refreshing the
README so the team-specific charts always match the latest data.

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
import sys
from typing import Dict, List, Tuple

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
    fig, ax = plt.subplots(figsize=(12, 5))

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
            color="white", fontsize=10, ha="center",
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
            color="white", fontsize=10, ha="center",
            arrowprops=dict(arrowstyle="->", color=SOFT_GREY, lw=1.0, alpha=0.8),
            bbox=dict(boxstyle="round,pad=0.4", fc=PANEL_BG, ec=TEAL, lw=1.0, alpha=0.95),
        )

    fig.text(0.99, 0.02,
             "Source: data/matches/  |  points = goals*6 + behinds, averaged across both teams per match",
             ha="right", va="bottom", color=SOFT_GREY, fontsize=8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
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

    fig, ax = plt.subplots(figsize=(12, 5))
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
                        color="white", fontsize=8.5, alpha=0.9)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.4,
                        "n/a", ha="center", va="bottom",
                        color=SOFT_GREY, fontsize=8, style="italic")

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
             ha="right", va="bottom", color=SOFT_GREY, fontsize=8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
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

    Top-6 = teams with highest disposals/game, matching the README summary
    table ordering.
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
    ax.set_xticklabels([lbl for _, lbl in RADAR_DIMENSIONS], color="white", fontsize=12)
    ax.tick_params(axis="x", pad=18)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color=SOFT_GREY, fontsize=8)
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
             color="white", fontsize=17, fontweight="bold", ha="center")
    fig.text(0.5, 0.918,
             "Normalised 0-1 vs all 18 teams (1.0 = league best on that axis)",
             color=SOFT_GREY, fontsize=11, ha="center")

    leg = ax.legend(loc="center left", bbox_to_anchor=(1.18, 0.5),
                    frameon=True, facecolor=PANEL_BG, edgecolor=GRID,
                    labelcolor="white", framealpha=0.9, fontsize=12,
                    title="Team (top 6 by disposals)", title_fontsize=11)
    leg.get_title().set_color("white")
    for text in leg.get_texts():
        text.set_color("white")

    fig.text(0.99, 0.02,
             f"Source: data/player_data/ year={year}  |  top-6 by disposals/g",
             ha="right", va="bottom", color=SOFT_GREY, fontsize=9, alpha=0.7)
    fig.savefig(out, dpi=150, facecolor=BG, edgecolor=BG)
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
                       rotation=20, ha="right", color="white", fontsize=11)
    ax.set_yticks(range(len(teams)))
    ax.set_yticklabels(teams, color="white", fontsize=10)

    # Rank text in each cell. White on dark cells, black on light.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            rk = int(matrix[i, j])
            # Mid-range ranks get black text, extremes get white for contrast.
            txt_col = "black" if 5 < rk < 14 else "white"
            ax.text(j, i, str(rk), ha="center", va="center",
                    color=txt_col, fontsize=10, fontweight="bold")

    ax.set_title(f"{year} season-to-date — team rank by stat (1 = league best, 18 = worst)",
                 color="white", pad=14, fontsize=14, fontweight="bold")
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
             ha="right", va="bottom", color=SOFT_GREY, fontsize=8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
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
                fontsize=10, fontstyle="italic", alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_BG, ec=GRID, lw=0.8, alpha=0.6))

    ax.scatter(x, y, s=160, c=GOLD, edgecolors="white", linewidths=1.0,
               alpha=0.9, zorder=3)

    # Team labels — slight offset to the right of each dot.
    label_offset_x = (x.max() - x.min()) * 0.012
    for xi, yi, name in zip(x, y, teams):
        ax.annotate(name, (xi + label_offset_x, yi),
                    color="white", fontsize=9, ha="left", va="center",
                    alpha=0.95)

    ax.set_xlabel("Handball ratio (% of disposals by hand)", color="white", fontsize=12)
    ax.set_ylabel("Tackles per game", color="white", fontsize=12)
    ax.set_title(
        f"Team playing styles, {years[0]}-{years[-1]} 5-year average\n"
        "Handball-vs-kick build-up x ground-ball pressure",
        color="white", pad=14, fontsize=14, fontweight="bold")
    ax.grid(True, ls=":", lw=0.8, alpha=0.30)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(GRID)

    fig.text(0.99, 0.02,
             f"Source: data/player_data/ years={years[0]}-{years[-1]}  |  dashed lines = league median",
             ha="right", va="bottom", color=SOFT_GREY, fontsize=8, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def regenerate_team_charts(year: int = None) -> List[str]:
    """Regenerate the three charts that depend on the latest season's data.
    Called by update_team_analysis.py after refreshing the README."""
    _ensure_dir(ASSETS_DIR)
    if year is None:
        year = _detect_current_year()
    paths = [
        chart_team_radar(year),
        chart_team_heatmap(year),
        chart_team_style_scatter(year),
    ]
    return paths


def regenerate_static_charts() -> List[str]:
    """Regenerate the two charts that depend only on era stats / match files."""
    _ensure_dir(ASSETS_DIR)
    return [
        chart_era_scoring_trends(),
        chart_era_stat_evolution(),
    ]


def main() -> None:
    _ensure_dir(ASSETS_DIR)
    print(f"[1/5] era scoring trends     -> {chart_era_scoring_trends()}")
    print(f"[2/5] era stat evolution     -> {chart_era_stat_evolution()}")
    year = _detect_current_year()
    print(f"      detected current year = {year}")
    print(f"[3/5] {year} top-6 radar       -> {chart_team_radar(year)}")
    print(f"[4/5] {year} rank heatmap      -> {chart_team_heatmap(year)}")
    print(f"[5/5] 5-year style scatter   -> {chart_team_style_scatter(year)}")


if __name__ == "__main__":
    main()
