"""Generate visualisations for the Coaches Strategy Corner briefs.

Charts produced (all PNG, dark theme, 180 DPI):
  1. adelaide_mcg_form.png            — Adelaide MCG win-rate by 5-year era
  2. richmond_vs_adelaide_h2h.png     — Wins by era block, stacked bar
  3. team_stat_comparison_2026.png    — Radar comparing Richmond vs Adelaide
  4. richmond_quarterly_differential_2026.png — Q1-Q4 differential bar chart
  5. key_player_disposal_comparison.png — Top 5 disposal players each side
  6. h2h_recent_results.png           — Last 10 H2H meetings, lollipop chart

Style follows the project's standard dark-mode palette (background #0d1117,
gold #f4c430, teal #2ec4b6). Min font size 14, DPI 180.

Run:  /home/abhi/sourceCode/python/coding/.venv/bin/python \
       /home/abhi/git/SuperCoach-VIA/docs/coaches-strategy-corner/generate_strategy_charts.py
"""
from __future__ import annotations

import glob
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config

REPO_ROOT = config.REPO_ROOT
DATA_DIR = os.path.join(REPO_ROOT, "data")
PLAYER_DIR = os.path.join(DATA_DIR, "player_data")
MATCH_DIR = os.path.join(DATA_DIR, "matches")
OUT_DIR = os.path.join(REPO_ROOT, "assets", "charts", "strategy")
os.makedirs(OUT_DIR, exist_ok=True)

# Palette (project standard)
BG = "#0d1117"
PANEL_BG = "#161b22"
GOLD = "#f4c430"
TEAL = "#2ec4b6"
RED = "#ef476f"
GREY = "#8b949e"
GRID = "#30363d"
DPI = 180


def _apply_style() -> None:
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
        "font.size": 14,
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
    })


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _load_long_matches(start: int = 1991, end: int = 2026) -> pd.DataFrame:
    rows: List[dict] = []
    for y in range(start, end + 1):
        f = os.path.join(MATCH_DIR, f"matches_{y}.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        for _, r in df.iterrows():
            for side in ("1", "2"):
                other = "2" if side == "1" else "1"
                t = r.get(f"team_{side}_team_name")
                o = r.get(f"team_{other}_team_name")
                pts = r.get(f"team_{side}_final_goals", 0) * 6 + r.get(f"team_{side}_final_behinds", 0)
                opp = r.get(f"team_{other}_final_goals", 0) * 6 + r.get(f"team_{other}_final_behinds", 0)
                rows.append({
                    "year": y, "round": r.get("round_num"), "venue": r.get("venue"),
                    "team": t, "opp": o, "pts": pts, "opp_pts": opp,
                    "margin": pts - opp,
                    "win": 1 if pts > opp else (0 if pts < opp else 0.5),
                })
    return pd.DataFrame(rows)


def _load_2026_team_stats() -> pd.DataFrame:
    """Aggregate per-player 2026 rows to per-team-game totals, then average."""
    files = sorted(glob.glob(os.path.join(PLAYER_DIR, "*_performance_details.csv")))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if df.empty or "year" not in df.columns:
            continue
        sub = df[df["year"] == 2026].copy()
        if sub.empty:
            continue
        sub["player_id"] = os.path.basename(f).replace("_performance_details.csv", "")
        frames.append(sub)
    full = pd.concat(frames, ignore_index=True)
    num_cols = ["disposals", "goals", "marks", "tackles", "hit_outs", "rebound_50s",
                "inside_50s", "clearances", "contested_possessions",
                "contested_marks", "marks_inside_50", "one_percenters", "clangers"]
    for c in num_cols:
        full[c] = pd.to_numeric(full[c], errors="coerce").fillna(0)
    full["round"] = full["round"].astype(str)
    team_game = full.groupby(["team", "year", "round"], as_index=False)[num_cols].sum()
    return team_game.groupby("team", as_index=False)[num_cols].mean()


# ---------------------------------------------------------------------------
# Chart 1 — Adelaide MCG form by 5-year era
# ---------------------------------------------------------------------------
def chart_adelaide_mcg_form() -> None:
    _apply_style()
    matches = _load_long_matches(1991, 2026)
    ade_mcg = matches[
        (matches["team"] == "Adelaide")
        & matches["venue"].astype(str).str.contains("M.C.G.", regex=False, na=False)
    ].copy()

    def era5(y: int) -> str:
        base = ((y - 1991) // 5) * 5 + 1991
        return f"{base}-{base + 4}"

    ade_mcg["era"] = ade_mcg["year"].apply(era5)
    # Drop the open 2026-2030 era (one game) for clarity
    ade_mcg = ade_mcg[ade_mcg["era"] != "2026-2030"]
    grp = ade_mcg.groupby("era").agg(games=("win", "count"),
                                     wins=("win", lambda s: (s == 1).sum()))
    grp["win_pct"] = grp["wins"] / grp["games"] * 100
    grp = grp.reset_index().sort_values("era")

    fig, ax = plt.subplots(figsize=(13, 6.5))
    bar_colors = [GOLD if v >= 50 else (TEAL if v >= 40 else RED) for v in grp["win_pct"]]
    bars = ax.bar(grp["era"], grp["win_pct"], color=bar_colors, edgecolor=BG, linewidth=2.5)

    ax.axhline(50, color=GREY, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.text(len(grp) - 0.45, 51, "50% line", color=GREY, fontsize=12, va="bottom", ha="right")

    for bar, games, wins in zip(bars, grp["games"], grp["wins"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2,
                f"{h:.0f}%", ha="center", va="bottom", color="white",
                fontsize=14, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 2,
                f"{wins}/{games}", ha="center", va="bottom",
                color=BG, fontsize=12, fontweight="bold")

    ax.set_ylim(0, 60)
    ax.set_ylabel("Win %", color="white")
    ax.set_title("Adelaide at the M.C.G. — win rate by 5-year era (1991–2025)",
                 color="white", pad=14)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.xticks(rotation=20, ha="right")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "adelaide_mcg_form.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Chart 2 — Richmond vs Adelaide H2H by era block (stacked horizontal bars)
# ---------------------------------------------------------------------------
def chart_h2h_by_era() -> None:
    _apply_style()
    matches = _load_long_matches(1991, 2025)
    h2h = matches[(matches["team"] == "Richmond") & (matches["opp"] == "Adelaide")].copy()

    bins = [
        ("1991–1996", 1991, 1996),
        ("1997–2010", 1997, 2010),
        ("2011–2020", 2011, 2020),
        ("2021–2025", 2021, 2025),
    ]
    rows = []
    for label, lo, hi in bins:
        sub = h2h[(h2h["year"] >= lo) & (h2h["year"] <= hi)]
        rows.append({
            "label": label,
            "rich_w": int((sub["win"] == 1).sum()),
            "ade_w": int((sub["win"] == 0).sum()),
            "total": int(len(sub)),
        })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["rich_w"], color=GOLD, edgecolor=BG, linewidth=2,
            label="Richmond wins")
    ax.barh(y_pos, df["ade_w"], left=df["rich_w"], color=TEAL,
            edgecolor=BG, linewidth=2, label="Adelaide wins")

    for i, row in df.iterrows():
        if row["rich_w"] > 0:
            ax.text(row["rich_w"] / 2, i, str(row["rich_w"]),
                    ha="center", va="center", color=BG, fontweight="bold", fontsize=15)
        if row["ade_w"] > 0:
            ax.text(row["rich_w"] + row["ade_w"] / 2, i, str(row["ade_w"]),
                    ha="center", va="center", color=BG, fontweight="bold", fontsize=15)
        ax.text(row["total"] + 0.3, i,
                f"{row['total']} games  ·  {row['rich_w']/(row['total'] or 1)*100:.0f}% R",
                va="center", color=GREY, fontsize=12)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Games won", color="white")
    ax.set_title("Richmond vs Adelaide head-to-head — by era block",
                 color="white", pad=14)
    ax.set_xlim(0, max(df["total"]) + 6)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=13)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "richmond_vs_adelaide_h2h.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Chart 3 — Stat profile comparison radar
# ---------------------------------------------------------------------------
def chart_stat_comparison() -> None:
    _apply_style()
    team_avg = _load_2026_team_stats()
    metrics = [
        ("disposals", "Disposals"),
        ("clearances", "Clearances"),
        ("inside_50s", "Inside 50s"),
        ("rebound_50s", "Rebound 50s"),
        ("tackles", "Tackles"),
        ("hit_outs", "Hit-outs"),
        ("goals", "Goals"),
        ("contested_possessions", "Contested poss"),
    ]
    league_max = team_avg[[m[0] for m in metrics]].max()

    rich = team_avg[team_avg["team"] == "Richmond"].iloc[0]
    ade = team_avg[team_avg["team"] == "Adelaide"].iloc[0]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    rich_vals = [rich[m[0]] / league_max[m[0]] for m in metrics]
    ade_vals = [ade[m[0]] / league_max[m[0]] for m in metrics]
    rich_vals += rich_vals[:1]
    ade_vals += ade_vals[:1]
    raw_rich = [rich[m[0]] for m in metrics]
    raw_ade = [ade[m[0]] for m in metrics]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(angles, rich_vals, color=GOLD, linewidth=3, label="Richmond")
    ax.fill(angles, rich_vals, color=GOLD, alpha=0.20)
    ax.plot(angles, ade_vals, color=TEAL, linewidth=3, label="Adelaide")
    ax.fill(angles, ade_vals, color=TEAL, alpha=0.20)

    # Compose two-line tick labels showing metric + raw R/A values
    tick_labels = [
        f"{m[1]}\nR {rv:.1f}  |  A {av:.1f}"
        for m, rv, av in zip(metrics, raw_rich, raw_ade)
    ]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tick_labels, fontsize=13, color="white")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "league-leading"],
                       color=GREY, fontsize=11)
    ax.tick_params(axis="x", pad=32)
    ax.grid(color=GRID, alpha=0.5)
    ax.spines["polar"].set_color(GRID)

    ax.set_title("Richmond vs Adelaide — 2026 stat profile (% of league-leading team)",
                 color="white", pad=48, fontsize=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.10),
              frameon=False, fontsize=14)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "team_stat_comparison_2026.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Chart 4 — Richmond quarterly differential
# ---------------------------------------------------------------------------
def chart_richmond_quarters() -> None:
    _apply_style()
    m26 = pd.read_csv(os.path.join(MATCH_DIR, "matches_2026.csv"))
    rows = []
    for _, r in m26.iterrows():
        for side in ("1", "2"):
            other = "2" if side == "1" else "1"
            t = r.get(f"team_{side}_team_name")
            if t != "Richmond":
                continue
            q1 = r.get(f"team_{side}_q1_goals", 0) * 6 + r.get(f"team_{side}_q1_behinds", 0)
            q2c = r.get(f"team_{side}_q2_goals", 0) * 6 + r.get(f"team_{side}_q2_behinds", 0)
            q3c = r.get(f"team_{side}_q3_goals", 0) * 6 + r.get(f"team_{side}_q3_behinds", 0)
            q4c = r.get(f"team_{side}_final_goals", 0) * 6 + r.get(f"team_{side}_final_behinds", 0)
            o_q1 = r.get(f"team_{other}_q1_goals", 0) * 6 + r.get(f"team_{other}_q1_behinds", 0)
            o_q2c = r.get(f"team_{other}_q2_goals", 0) * 6 + r.get(f"team_{other}_q2_behinds", 0)
            o_q3c = r.get(f"team_{other}_q3_goals", 0) * 6 + r.get(f"team_{other}_q3_behinds", 0)
            o_q4c = r.get(f"team_{other}_final_goals", 0) * 6 + r.get(f"team_{other}_final_behinds", 0)
            rows.append({
                "Q1": q1 - o_q1,
                "Q2": (q2c - q1) - (o_q2c - o_q1),
                "Q3": (q3c - q2c) - (o_q3c - o_q2c),
                "Q4": (q4c - q3c) - (o_q4c - o_q3c),
            })
    df = pd.DataFrame(rows)
    means = df.mean()

    fig, ax = plt.subplots(figsize=(11, 6.5))
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    vals = [means[q] for q in quarters]
    colors = [GOLD if v >= 0 else RED for v in vals]
    bars = ax.bar(quarters, vals, color=colors, edgecolor=BG, linewidth=2.5, width=0.65)

    ax.axhline(0, color=GREY, linewidth=1.2, alpha=0.6)
    for bar, v in zip(bars, vals):
        offset = 0.6 if v >= 0 else -0.6
        va = "bottom" if v >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                f"{v:+.1f}", ha="center", va=va, color="white",
                fontsize=15, fontweight="bold")

    ax.set_ylabel("Average score differential (points)", color="white")
    ax.set_title("Richmond 2026 — quarter-by-quarter score differential",
                 color="white", pad=14)
    sub = f"Across {len(df)} games. Negative = Richmond outscored. Q1 has been the persistent leak."
    ax.text(0.5, -0.18, sub, transform=ax.transAxes, ha="center",
            color=GREY, fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    lo = min(vals) - 4
    hi = max(vals) + 3
    ax.set_ylim(lo, hi if hi > 3 else 3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "richmond_quarterly_differential_2026.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Chart 5 — Top 5 disposal players each side
# ---------------------------------------------------------------------------
def chart_player_disposals() -> None:
    _apply_style()
    files = sorted(glob.glob(os.path.join(PLAYER_DIR, "*_performance_details.csv")))
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if df.empty or "year" not in df.columns:
            continue
        sub = df[df["year"] == 2026].copy()
        if sub.empty:
            continue
        team = sub["team"].iloc[0] if "team" in sub.columns else None
        if team not in ("Richmond", "Adelaide"):
            continue
        # only games actually played (percentage_of_game_played > 0)
        if "percentage_of_game_played" in sub.columns:
            played = sub[pd.to_numeric(sub["percentage_of_game_played"], errors="coerce").fillna(0) > 0]
        else:
            played = sub.dropna(subset=["disposals"])
        if played.empty:
            continue
        disp = pd.to_numeric(played["disposals"], errors="coerce").fillna(0)
        n_games = len(played)
        if n_games < 4:
            continue
        pid = os.path.basename(f).replace("_performance_details.csv", "")
        # Pretty name from id: surname_first  → "First Surname"
        bits = pid.split("_")
        if len(bits) >= 2:
            name = f"{bits[1].title()} {bits[0].title()}"
        else:
            name = pid.replace("_", " ").title()
        rows.append({"team": team, "name": name, "disp_avg": disp.mean(), "games": n_games})

    df = pd.DataFrame(rows)
    rich = df[df["team"] == "Richmond"].sort_values("disp_avg", ascending=False).head(5)
    ade = df[df["team"] == "Adelaide"].sort_values("disp_avg", ascending=False).head(5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)
    for ax, sub_df, color, label in [
        (axes[0], rich.iloc[::-1], GOLD, "Richmond"),
        (axes[1], ade.iloc[::-1], TEAL, "Adelaide"),
    ]:
        bars = ax.barh(sub_df["name"], sub_df["disp_avg"],
                       color=color, edgecolor=BG, linewidth=2)
        for bar, v, g in zip(bars, sub_df["disp_avg"], sub_df["games"]):
            ax.text(v + 0.4, bar.get_y() + bar.get_height() / 2,
                    f"{v:.1f}  ({g}g)", va="center", color="white", fontsize=12)
        ax.set_title(label, color=color, fontsize=16, pad=10)
        ax.set_xlabel("Disposals / game", color="white")
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(rich["disp_avg"].max(), ade["disp_avg"].max()) + 4)

    fig.suptitle("Top 5 disposal-getters in 2026 — Richmond vs Adelaide",
                 color="white", fontsize=17, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "key_player_disposal_comparison.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Chart 6 — Last 10 H2H meetings, lollipop
# ---------------------------------------------------------------------------
def chart_h2h_recent() -> None:
    _apply_style()
    matches = _load_long_matches(1991, 2025)
    h2h = matches[(matches["team"] == "Richmond") & (matches["opp"] == "Adelaide")].copy()
    h2h["round_num"] = pd.to_numeric(h2h["round"], errors="coerce")
    h2h = h2h.sort_values(["year", "round_num"]).tail(10)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(h2h))
    margins = h2h["margin"].values
    colors = [GOLD if m > 0 else RED for m in margins]

    ax.vlines(x, 0, margins, colors=colors, lw=3, alpha=0.85)
    ax.scatter(x, margins, s=180, c=colors, edgecolors=BG, zorder=4, linewidth=2)
    ax.axhline(0, color=GREY, linewidth=1.2, alpha=0.6)

    for xi, m, yr, rnd in zip(x, margins, h2h["year"], h2h["round"]):
        offset = 5 if m >= 0 else -5
        va = "bottom" if m >= 0 else "top"
        ax.text(xi, m + offset, f"{m:+.0f}", ha="center", va=va,
                color="white", fontweight="bold", fontsize=12)

    def _round_label(r):
        s = str(r)
        if s.replace(".0", "").isdigit():
            return f"R{int(float(s))}"
        return s
    labels = [f"{int(y)}\n{_round_label(r)}" for y, r in zip(h2h["year"], h2h["round"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=12)
    ax.set_ylabel("Margin (Richmond perspective)", color="white")
    ax.set_title("Last 10 Richmond vs Adelaide meetings — margin over time",
                 color="white", pad=14)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    sub = ("Gold = Richmond win   ·   Red = Adelaide win   ·   "
           f"Period: {int(h2h['year'].iloc[0])}–{int(h2h['year'].iloc[-1])}")
    ax.text(0.5, -0.22, sub, transform=ax.transAxes, ha="center",
            color=GREY, fontsize=12)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "h2h_recent_results.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  wrote {path}")


def main() -> None:
    print(f"Output dir: {OUT_DIR}")
    chart_adelaide_mcg_form()
    chart_h2h_by_era()
    chart_stat_comparison()
    chart_richmond_quarters()
    chart_player_disposals()
    chart_h2h_recent()
    print("Done.")


if __name__ == "__main__":
    main()
