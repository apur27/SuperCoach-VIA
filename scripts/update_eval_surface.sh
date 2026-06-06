#!/usr/bin/env bash
# =============================================================================
# update_eval_surface.sh — refresh the README "Eval results — current" section
# and docs/banner.svg from the latest backtest figures.
#
# Source of truth: data/prediction/backtest/backtest_summary_*.csv (per-round)
#                  data/prediction/backtest/backtest_by_team_*.csv  (per-round per-team)
# Both are merged across ALL runs and deduped by (year, round[, team]) keeping
# the newest entry — the SAME merge logic update_team_analysis.py uses to build
# docs/afl-backtest-2026.md. This script does NOT author numbers; it re-derives
# already-verified figures and re-renders the presentation surface.
#
# Touches ONLY:
#   README.md      — the "## Eval results — current" section (table + 2 prose figures)
#   docs/banner.svg — header pills, Band 1 player count, Band 2 numbers + round label
#
# Does NOT touch: the README news block, docs/news/, any other section.
# Idempotent: safe to run repeatedly; output depends only on the CSVs on disk.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/home/abhi/sourceCode/python/coding/.venv/bin/python
cd "$REPO_ROOT"

PLAYER_FILE_COUNT=$(ls data/player_data/*performance_details.csv 2>/dev/null | wc -l | tr -d ' ')

"$PYTHON" - "$PLAYER_FILE_COUNT" <<'PYEOF'
import sys, os, glob, re
import pandas as pd

REPO = os.getcwd()
PLAYER_FILE_COUNT = int(sys.argv[1])
YEAR = 2026

bt = os.path.join(REPO, "data", "prediction", "backtest")

# ---- merge all per-round summary CSVs (newest entry per round wins) ----
need = {"round", "year", "n_players", "mae", "rmse",
        "pct_within_5", "pct_within_10", "bias"}
frames = []
for p in sorted(glob.glob(os.path.join(bt, "backtest_summary_*.csv")),
                key=os.path.getmtime):
    try:
        c = pd.read_csv(p)
    except Exception:
        continue
    if need.issubset(c.columns):
        frames.append(c)
if not frames:
    sys.exit("update_eval_surface: no usable backtest_summary CSVs found")
df = pd.concat(frames, ignore_index=True)
df = df.drop_duplicates(subset=["year", "round"], keep="last")
df = df[df["year"] == YEAR].sort_values("round").reset_index(drop=True)
if df.empty:
    sys.exit(f"update_eval_surface: no {YEAR} rounds in backtest summaries")

r_lo, r_hi = int(df["round"].min()), int(df["round"].max())
window = f"R{r_lo}-R{r_hi}"
window_svg = f"R{r_lo}&#8211;R{r_hi}"  # en-dash entity, matches banner style
N = int(df["n_players"].sum())

def wmean(col):
    return float((df[col] * df["n_players"]).sum() / df["n_players"].sum())

mae_w = wmean("mae")
w5_w = wmean("pct_within_5")
w10_w = wmean("pct_within_10")
bias_w = wmean("bias")

# named extreme rounds (recomputed each run so labels self-correct)
hardest = df.loc[df["mae"].idxmax()]      # highest MAE
best_mae = df.loc[df["mae"].idxmin()]     # lowest MAE
best_w5 = df.loc[df["pct_within_5"].idxmax()]

def rrow(label, r):
    return (f"| {label} | **[data]** {int(r['n_players'])} | "
            f"**[data]** {float(r['mae']):.2f} | "
            f"**[data]** {float(r['pct_within_5']):.1f}% | "
            f"**[data]** {float(r['pct_within_10']):.1f}% | — |")

# de-dup named rows: if best-MAE and best-w5 are the same round, keep one label
named = []
named.append(("Round {} (hardest)".format(int(hardest['round'])), hardest))
seen = {int(hardest['round'])}
if int(best_mae['round']) not in seen:
    named.append(("Round {} (best MAE)".format(int(best_mae['round'])), best_mae))
    seen.add(int(best_mae['round']))
if int(best_w5['round']) not in seen:
    named.append(("Round {} (best within-5)".format(int(best_w5['round'])), best_w5))
    seen.add(int(best_w5['round']))

table_lines = [
    "| Window | Player-rounds | MAE | Within 5 | Within 10 | Bias |",
    "|---|--:|--:|--:|--:|--:|",
    (f"| **{window} player-weighted** | **[data]** {N:,} | "
     f"**[data]** {mae_w:.3f} | **[data]** {w5_w:.1f}% | "
     f"**[data]** {w10_w:.1f}% | **[data]** {bias_w:.3f} |"),
]
for label, r in named:
    table_lines.append(rrow(label, r))
table_md = "\n".join(table_lines)

# ---- season team bias from merged per-round per-team CSVs ----
tframes = []
for p in sorted(glob.glob(os.path.join(bt, "backtest_by_team_*.csv")),
                key=os.path.getmtime):
    try:
        c = pd.read_csv(p)
    except Exception:
        continue
    if {"year", "round", "team", "n", "bias"}.issubset(c.columns):
        tframes.append(c)
if not tframes:
    sys.exit("update_eval_surface: no usable backtest_by_team CSVs found")
tdf = pd.concat(tframes, ignore_index=True)
tdf = tdf.drop_duplicates(subset=["year", "round", "team"], keep="last")
tdf = tdf[tdf["year"] == YEAR]
g = (tdf.groupby("team")
        .apply(lambda x: (x["bias"] * x["n"]).sum() / x["n"].sum(),
               include_groups=False)
        .sort_values())
team_under, bias_under = g.index[0], float(g.iloc[0])     # most under-predicted (min)
team_over, bias_over = g.index[-1], float(g.iloc[-1])     # most over-predicted (max)
mean_abs_bias = float(g.abs().mean())

# =====================================================================
# 1) README.md — replace the Eval results table + 2 prose figures
# =====================================================================
readme_path = os.path.join(REPO, "README.md")
with open(readme_path, encoding="utf-8") as f:
    md = f.read()

# Intro line round window: "Rounds 1–13"
md = re.sub(
    r"(Walk-forward backtest, 2026 season, Rounds )\d+–\d+",
    rf"\g<1>{r_lo}–{r_hi}",
    md,
    count=1,
)

# Replace the table: anchor from the header row through the last data row.
table_re = re.compile(
    r"\| Window \| Player-rounds \| MAE \| Within 5 \| Within 10 \| Bias \|.*?"
    r"(?=\n\n\*\*Plain English:\*\*)",
    re.DOTALL,
)
if not table_re.search(md):
    sys.exit("update_eval_surface: README eval table anchor not found")
md = table_re.sub(table_md.replace("\\", "\\\\"), md, count=1)

# ---- "The numbers" summary table (lines ~35-42) ----
md = re.sub(
    r"(\| Backtest window \| \*\*\[data\]\*\* )R\d+–R\d+, 2026",
    rf"\g<1>R{r_lo}–R{r_hi}, 2026", md, count=1)
md = re.sub(
    r"(\| Player-round predictions scored \| \*\*\[data\]\*\* )[\d,]+",
    rf"\g<1>{N:,}", md, count=1)
md = re.sub(
    r"(\| Mean absolute error \(disposals\) \| \*\*\[data\]\*\* )[\d.]+",
    rf"\g<1>{mae_w:.3f}", md, count=1)
md = re.sub(
    r"(\| Within 5 disposals \| \*\*\[data\]\*\* )[\d.]+%",
    rf"\g<1>{w5_w:.1f}%", md, count=1)
md = re.sub(
    r"(\| Within 10 disposals \| \*\*\[data\]\*\* )[\d.]+%",
    rf"\g<1>{w10_w:.1f}%", md, count=1)
md = re.sub(
    r"(\| Aggregate bias \| \*\*\[data\]\*\* )[-+]?[\d.]+",
    rf"\g<1>{bias_w:.3f}", md, count=1)
# Plain English sentence in "The numbers" section
md = re.sub(
    r"(measured honestly across )[\d,]+( predictions\.)",
    rf"\g<1>{N:,}\g<2>", md, count=1)

# ---- Prediction model prose (within 5 / within 10 inline) ----
md = re.sub(
    r"(within 5 disposals \*\*\[data\]\*\* )[\d.]+%( of the time and within 10 "
    r"\*\*\[data\]\*\* )[\d.]+%( of the time)",
    rf"\g<1>{w5_w:.1f}%\g<2>{w10_w:.1f}%\g<3>", md, count=1)

# ---- ML inference table row ----
md = re.sub(
    r"(Walk-forward backtest: \*\*\[data\]\*\* MAE )[\d.]+ across [\d,]+ "
    r"player-rounds \(R\d+–R\d+, \d+\)\.",
    rf"\g<1>{mae_w:.3f} across {N:,} player-rounds (R{r_lo}–R{r_hi}, {YEAR}).",
    md, count=1)

# Team-bias sentence figures in the Technical paragraph.
md = re.sub(
    r"Team-level signed bias spans \*\*\[data\]\*\* [-+]?\d+\.\d+ \([^)]+\) "
    r"to \*\*\[data\]\*\* [-+]?\d+\.\d+ \([^)]+\), with mean absolute team bias "
    r"\*\*\[data\]\*\* \d+\.\d+ disposals\.",
    (f"Team-level signed bias spans **[data]** {bias_under:.2f} "
     f"({team_under}, most under-predicted) to **[data]** {bias_over:+.2f} "
     f"({team_over}, most over-predicted), with mean absolute team bias "
     f"**[data]** {mean_abs_bias:.2f} disposals."),
    md,
    count=1,
)

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(md)

# =====================================================================
# 2) docs/banner.svg — pills, Band 1 player count, Band 2 numbers/label
# =====================================================================
svg_path = os.path.join(REPO, "docs", "banner.svg")
with open(svg_path, encoding="utf-8") as f:
    svg = f.read()

mae_s = f"{mae_w:.3f}"
w5_s = f"{w5_w:.1f}%"
w10_s = f"{w10_w:.1f}%"

# aria-label summary line
svg = re.sub(
    r"2026 season R\d+&#8211;R\d+: MAE \d+\.\d+, [\d.]+% within 5 disposals",
    f"2026 season {window_svg}: MAE {mae_s}, {w5_s} within 5 disposals",
    svg, count=1,
)

# Pill 1 — round label
svg = re.sub(
    r'(<text x="181"[^>]*>)R\d+&#8211;R\d+ &#183; 2026(</text>)',
    rf"\g<1>{window_svg} &#183; 2026\g<2>",
    svg, count=1,
)
# Pill 2 — MAE
svg = re.sub(
    r'(<text x="365"[^>]*>)MAE \d+\.\d+(</text>)',
    rf"\g<1>MAE {mae_s}\g<2>",
    svg, count=1,
)
# Pill 3 — within 5
svg = re.sub(
    r'(<text x="563"[^>]*>)[\d.]+% within 5(</text>)',
    rf"\g<1>{w5_s} within 5\g<2>",
    svg, count=1,
)

# Band 1 — player file count
svg = re.sub(
    r"130 seasons &#183; [\d,]+ player files",
    f"130 seasons &#183; {PLAYER_FILE_COUNT:,} player files",
    svg, count=1,
)

# Band 2 — section label "(R1–R13 · 4,806 player-rounds)"
svg = re.sub(
    r"(PREDICTION ACCURACY &#8212; 2026 SEASON \()R\d+&#8211;R\d+ &#183; "
    r"[\d,]+ player-rounds\)",
    rf"\g<1>{window_svg} &#183; {N:,} player-rounds)",
    svg, count=1,
)

# Band 2 — three big numbers (anchored by their x + font-size 54)
svg = re.sub(
    r'(<text x="300" y="665"[^>]*font-size="54"[^>]*>)\d+\.\d+(</text>)',
    rf"\g<1>{mae_s}\g<2>", svg, count=1)
svg = re.sub(
    r'(<text x="600" y="665"[^>]*font-size="54"[^>]*>)[\d.]+%(</text>)',
    rf"\g<1>{w5_s}\g<2>", svg, count=1)
svg = re.sub(
    r'(<text x="900" y="665"[^>]*font-size="54"[^>]*>)[\d.]+%(</text>)',
    rf"\g<1>{w10_s}\g<2>", svg, count=1)

with open(svg_path, "w", encoding="utf-8") as f:
    f.write(svg)

print(f"update_eval_surface: {window} | player-rounds {N:,} | "
      f"MAE {mae_s} | within5 {w5_s} | within10 {w10_s} | bias {bias_w:.3f}")
print(f"  team bias: {team_under} {bias_under:+.2f} .. {team_over} "
      f"{bias_over:+.2f} | mean-abs {mean_abs_bias:.2f}")
print(f"  player files: {PLAYER_FILE_COUNT:,}")
PYEOF
