#!/usr/bin/env python3
"""Generate dark-theme charts for docs/hall-of-fame-stat-leaders.md.

Reads docs/hall-of-fame/_stat_leaders.json (produced by
compute_stat_leaders.py) and writes:

    assets/charts/hall/alltime_top20_goals.png
    assets/charts/hall/alltime_top20_games.png
    assets/charts/hall/alltime_top20_disposals.png
    assets/charts/hall/alltime_top20_tackles.png
    assets/charts/hall/alltime_stat_categories_leaders.png

Style matches the rest of the project: bg #0d1117, gold #f4c430,
teal #2ec4b6, DPI 180, min font ~14.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/home/abhi/git/SuperCoach-VIA")
LEADERS_JSON = REPO / "docs" / "hall-of-fame" / "_stat_leaders.json"
OUT_DIR = REPO / "assets" / "charts" / "hall"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = "#0d1117"
GOLD = "#f4c430"
TEAL = "#2ec4b6"
SKY = "#4cc9f0"
PINK = "#f72585"
GREEN = "#80ed99"
GRID = "#30363d"
TEXT = "#ffffff"


def _apply_dark_style() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "grid.color": GRID,
            "grid.alpha": 0.35,
            "font.size": 14,
            "axes.titleweight": "bold",
            "axes.titlesize": 18,
            "savefig.facecolor": BG,
            "savefig.edgecolor": BG,
            "savefig.dpi": 180,
        }
    )


def _decade_color(year_min: int) -> str:
    """Colour bars by debut decade — gives a quick era cue."""
    if year_min < 1950:
        return PINK
    if year_min < 1970:
        return SKY
    if year_min < 1990:
        return TEAL
    if year_min < 2010:
        return GOLD
    return GREEN


def _bar_chart(
    leaders: List[Dict],
    title: str,
    xlabel: str,
    out_name: str,
    value_fmt: str = "{:,.0f}",
) -> Path:
    _apply_dark_style()

    n = len(leaders)
    names = [r["name"] for r in leaders]
    totals = [r["total"] for r in leaders]
    spans = [f"{r['year_min']}-{r['year_max']}" for r in leaders]
    colors = [_decade_color(r["year_min"]) for r in leaders]

    # reverse so #1 is at the top of the chart
    names_r = list(reversed(names))
    totals_r = list(reversed(totals))
    spans_r = list(reversed(spans))
    colors_r = list(reversed(colors))

    fig, ax = plt.subplots(figsize=(13, 9.5))
    y_pos = np.arange(n)
    bars = ax.barh(y_pos, totals_r, color=colors_r, edgecolor=BG, linewidth=0.6)

    # Y-tick labels: name + era
    ytick_labels = [f"{nm}  ({sp})" for nm, sp in zip(names_r, spans_r)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ytick_labels, fontsize=12)

    # Annotate bar end with total
    max_total = max(totals_r)
    for bar, val in zip(bars, totals_r):
        ax.text(
            bar.get_width() + max_total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            value_fmt.format(val),
            va="center",
            ha="left",
            color=TEXT,
            fontsize=12,
            fontweight="bold",
        )

    # Era legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=PINK, label="Pre-1950"),
        plt.Rectangle((0, 0), 1, 1, color=SKY, label="1950-1969"),
        plt.Rectangle((0, 0), 1, 1, color=TEAL, label="1970-1989"),
        plt.Rectangle((0, 0), 1, 1, color=GOLD, label="1990-2009"),
        plt.Rectangle((0, 0), 1, 1, color=GREEN, label="2010-now"),
    ]
    ax.legend(
        handles=legend_elements,
        title="Debut era",
        loc="lower right",
        fontsize=11,
        title_fontsize=12,
        facecolor=BG,
        edgecolor=GRID,
        framealpha=0.9,
    )

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_title(title, color=GOLD, pad=14)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max_total * 1.13)
    ax.invert_yaxis()  # already reversed but keep #1 visible at top
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = OUT_DIR / out_name
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def chart_wall_of_records(data: dict) -> Path:
    """Summary chart: one row per category, showing the #1 holder & total."""
    _apply_dark_style()

    # Order chosen for narrative impact: glamour stats first, support stats last.
    order = [
        ("career_goals", "Goals"),
        ("career_disposals", "Disposals"),
        ("career_marks", "Marks"),
        ("career_tackles", "Tackles"),
        ("career_hit_outs", "Hit-outs"),
        ("career_clearances", "Clearances"),
        ("career_inside_50s", "Inside 50s"),
        ("career_contested_possessions", "Contested poss."),
        ("career_brownlow_votes", "Brownlow votes"),
        ("career_goal_assists", "Goal assists"),
        ("career_games", "Games"),
    ]

    rows = []
    for key, label in order:
        if key not in data["categories"]:
            continue
        leaders = data["categories"][key]["leaders"]
        if not leaders:
            continue
        top = leaders[0]
        rows.append((label, top["name"], top["total"], top["year_min"]))

    n = len(rows)
    fig, ax = plt.subplots(figsize=(14, 0.65 * n + 2.4))
    y_pos = np.arange(n)[::-1]  # top-to-bottom in declared order

    labels = [r[0] for r in rows]
    holders = [r[1] for r in rows]
    totals = [r[2] for r in rows]
    decades = [r[3] for r in rows]
    colors = [_decade_color(d) for d in decades]

    # Bars are normalised to 1 — this chart is about who, not how big.
    bars = ax.barh(
        y_pos,
        [1] * n,
        color=colors,
        edgecolor=BG,
        linewidth=0.7,
        alpha=0.92,
    )

    # Category label on the left side of the bar
    for yi, lab in zip(y_pos, labels):
        ax.text(
            -0.01,
            yi,
            lab,
            va="center",
            ha="right",
            color=TEXT,
            fontsize=14,
            fontweight="bold",
        )

    # Holder + total inside the bar
    for yi, holder, total in zip(y_pos, holders, totals):
        if total >= 1000:
            num_str = f"{total:,.0f}"
        else:
            num_str = f"{total:,.0f}"
        ax.text(
            0.02,
            yi,
            holder,
            va="center",
            ha="left",
            color=BG,
            fontsize=14,
            fontweight="bold",
        )
        ax.text(
            0.98,
            yi,
            num_str,
            va="center",
            ha="right",
            color=BG,
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.set_title(
        "Wall of records — #1 in every major category",
        color=GOLD,
        pad=14,
    )

    plt.tight_layout()
    out_path = OUT_DIR / "alltime_stat_categories_leaders.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main() -> None:
    if not LEADERS_JSON.exists():
        raise SystemExit(f"missing {LEADERS_JSON} — run compute_stat_leaders.py first")
    data = json.loads(LEADERS_JSON.read_text())

    cats = data["categories"]

    _bar_chart(
        cats["career_goals"]["leaders"],
        title="All-time top 20 goal scorers",
        xlabel="Career goals",
        out_name="alltime_top20_goals.png",
    )
    _bar_chart(
        cats["career_games"]["leaders"],
        title="All-time top 20 most games played",
        xlabel="Career games",
        out_name="alltime_top20_games.png",
    )
    _bar_chart(
        cats["career_disposals"]["leaders"],
        title="All-time top 20 disposal-getters",
        xlabel="Career disposals",
        out_name="alltime_top20_disposals.png",
    )
    _bar_chart(
        cats["career_tackles"]["leaders"],
        title="All-time top 20 tacklers (since 1987)",
        xlabel="Career tackles",
        out_name="alltime_top20_tackles.png",
    )

    chart_wall_of_records(data)


if __name__ == "__main__":
    main()
