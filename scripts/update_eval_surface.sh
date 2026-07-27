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
        c["_mtime"] = os.path.getmtime(p)   # vintage, for supersede-by-file below
        tframes.append(c)
if not tframes:
    sys.exit("update_eval_surface: no usable backtest_by_team CSVs found")
tdf = pd.concat(tframes, ignore_index=True)
# Supersede by FILE, not by (year, round, team). Deduping per-team keeps rows from
# an older vintage whenever the newer run covers FEWER teams — R18 2026 was
# re-scored from an archived forward CSV that predated the full fixture (14 teams,
# not 18), so four teams survived from the previous file and injected 92 phantom
# player-rounds. One carried bias -3.26 over n=23 and single-handedly moved the
# published "most under-predicted" figure from -0.583 to -0.733. The result was
# not stale but incoherent: headline from one vintage, four teams from another.
newest = tdf.groupby(["year", "round"])["_mtime"].transform("max")
tdf = tdf[tdf["_mtime"] == newest]
tdf = tdf.drop_duplicates(subset=["year", "round", "team"], keep="last")
tdf = tdf[tdf["year"] == YEAR]
g = (tdf.groupby("team")
        .apply(lambda x: (x["bias"] * x["n"]).sum() / x["n"].sum(),
               include_groups=False)
        .sort_values())
team_under, bias_under = g.index[0], float(g.iloc[0])     # most under-predicted (min)
team_over, bias_over = g.index[-1], float(g.iloc[-1])     # most over-predicted (max)
mean_abs_bias = float(g.abs().mean())

# Reconciliation: the per-team rows describe the same scored population as the
# per-round summary, so their n must agree exactly. A mismatch means two vintages
# have been crossed and the team sentence would disagree with the headline it sits
# next to. Fail closed rather than publish a hybrid.
team_n = int(tdf["n"].sum())
summary_n = int(df["n_players"].sum())
if team_n != summary_n:
    sys.exit(
        f"update_eval_surface: team-n reconciliation FAILED — per-team rows sum to "
        f"{team_n:,} but the round summaries sum to {summary_n:,} (delta "
        f"{team_n - summary_n:+,}). Two backtest vintages have been crossed; the "
        f"team-level sentence would contradict the headline figures. Refusing to write."
    )

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
# The authored row uses U+2212 MINUS SIGN, not ASCII hyphen. An ASCII-only class
# matched nothing, re.sub returned the input unchanged, and the cell silently froze
# at its R1-R13 value (-0.093) while the same metric updated elsewhere in the file.
# Accept every minus form; we always write ASCII, as this script does everywhere.
md = re.sub(
    r"(\| Aggregate bias \| \*\*\[data\]\*\* )[-+−]?[\d.]+",
    rf"\g<1>{bias_w:.3f}", md, count=1)
# ---- "Full per-round table (all N rounds)" ----
md = re.sub(
    r"(Full per-round table \(all )\d+( rounds\))",
    rf"\g<1>{len(df)}\g<2>", md, count=1)
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

# ---- Shields.io badge: "data-2026%20season%20round%20N" ----
md = re.sub(
    r"(data-2026%20season%20round%20)\d+(-green)",
    rf"\g<1>{r_hi}\g<2>", md, count=1)

# ---- Player file count — 3 README locations ----
# 1. "The numbers" table row
md = re.sub(
    r"(\| Player performance files \| \*\*\[data\]\*\* )[\d,]+",
    rf"\g<1>{PLAYER_FILE_COUNT:,}", md, count=1)
# 2. "The data" narrative paragraph
md = re.sub(
    r"(\*\*\[data\]\*\* )[\d,]+( individual player files)",
    rf"\g<1>{PLAYER_FILE_COUNT:,}\g<2>", md, count=1)
# 3. "Under the hood" table row
md = re.sub(
    r"(\*\*\[data\]\*\* )[\d,]+( player performance files \(one row)",
    rf"\g<1>{PLAYER_FILE_COUNT:,}\g<2>", md, count=1)

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

# Round-range dash: the file has historically carried BOTH the `&#8211;` entity
# (pills) and a literal en-dash (aria-label). A pattern hard-coded to one form
# silently no-ops on the other — that is how the aria-label froze at R1-R13 /
# MAE 4.020 while the pills tracked reality, reading stale accuracy figures to
# screen-reader users. Match every form; we always WRITE the entity form.
DASH = r"(?:&#8211;|&#x2013;|–|-)"

# aria-label summary line
svg = re.sub(
    rf"2026 season R\d+{DASH}R\d+: MAE \d+\.\d+, [\d.]+% within 5 disposals",
    f"2026 season {window_svg}: MAE {mae_s}, {w5_s} within 5 disposals",
    svg, count=1,
)

# aria-label player-file count. The visible Band 1 text says "130 seasons ·
# N player files" while the aria-label says "130 years of AFL data, N player
# files" — different wording, so the Band 1 pattern below never touched it and
# the announced count froze at 13,329 while the band tracked reality.
svg = re.sub(
    r"(130 years of AFL data, )[\d,]+( player files)",
    rf"\g<1>{PLAYER_FILE_COUNT:,}\g<2>",
    svg, count=1,
)

# Pill 1 — round label
svg = re.sub(
    rf'(<text x="181"[^>]*>)R\d+{DASH}R\d+ &#183; 2026(</text>)',
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

# =====================================================================
# 3) docs/afl-backtest-2026.md — the three blocks that had no regenerator
#
# This page published five inconsistent figure sets because nothing rewrote
# them: a cumulative summary frozen at R1-R13 whose values could no longer be
# reproduced from any vintage on disk, a team-bias table frozen at R1-R13, and
# a notable-misses table frozen at R1-R11 — all sitting beneath an auto-generated
# per-round table that tracked R1-R20.
#
# The per-round table and its closing "mean across N rounds" line are NOT touched
# here. That line is an unweighted mean across rounds and is correct as labelled;
# the blocks below are player-weighted. They are different statistics, and making
# them agree would be the defect, not the fix.
# =====================================================================
bt_doc = os.path.join(REPO, "docs", "afl-backtest-2026.md")
if os.path.exists(bt_doc):
    with open(bt_doc, encoding="utf-8") as f:
        doc = f.read()

    def _swap(text, name, body):
        pat = re.compile(rf"(<!-- {name}-START -->\n).*?(<!-- {name}-END -->)", re.DOTALL)
        if not pat.search(text):
            sys.exit(f"update_eval_surface: {name} markers not found in {bt_doc}")
        return pat.sub(lambda m: m.group(1) + body + m.group(2), text, count=1)

    # ---- Cumulative summary (player-weighted over the pooled rows) ----
    # RMSE is pooled in SQUARED space. Running it through wmean() like the other
    # metrics gives a plausible-looking wrong answer (5.0843 vs a correct 5.0937)
    # because a root-mean-square does not average linearly.
    rmse_pooled = float(((df["rmse"] ** 2 * df["n_players"]).sum() / N) ** 0.5)
    cum_rows = [
        "| Metric | Value | What it means |",
        "|---|---|---|",
        f"| Rounds backtested | {len(df)} (R{r_lo}–R{r_hi}) | Walk-forward — each round predicted using only data from rounds before it |",
        f"| Player predictions scored | **{N:,}** | Total prediction-vs-actual pairs across the {len(df)} rounds |",
        f"| **MAE (overall)** | **{mae_w:.3f} disposals** | Average absolute miss across every player-round |",
        f"| **RMSE (overall)** | **{rmse_pooled:.3f} disposals** | Penalises large misses more heavily; pooled in squared space, not averaged |",
        f"| **Bias (overall)** | **{bias_w:.3f} disposals** | Signed mean error — negative means the model predicts too low |",
        f"| Cumulative MAE (mean of round MAE) | {df['mae'].mean():.2f} | Equally weights each round, unlike the player-weighted figure above |",
        f"| Median round MAE | {df['mae'].median():.2f} | Half the rounds beat this number, half fell short |",
    ]
    doc = _swap(doc, "CUMULATIVE", "\n".join(cum_rows) + "\n")

    # ---- Team-level bias (supersede-by-file already applied to tdf) ----
    tstat = (tdf.groupby("team")
                .apply(lambda x: pd.Series({
                    "n": int(x["n"].sum()),
                    "bias": (x["bias"] * x["n"]).sum() / x["n"].sum(),
                }), include_groups=False)
                .sort_values("bias"))
    team_rows = ["| Team | Predictions (n) | Bias | Direction |",
                 "|------|----------------:|-----:|-----------|"]
    for team, row in tstat.iterrows():
        direction = "under-predict" if row["bias"] < 0 else "over-predict"
        team_rows.append(f"| {team} | {int(row['n'])} | {row['bias']:+.2f} | {direction} |")
    doc = _swap(doc, "TEAMBIAS", "\n".join(team_rows) + "\n")

    # ---- Round-by-round notable misses, from the per-player detail CSVs ----
    # Same vintage discipline as everything else: newest FILE per (year, round).
    dets = []
    for p in sorted(glob.glob(os.path.join(bt, "prediction_vs_actual_round_*.csv")),
                    key=os.path.getmtime):
        m = re.search(r"round_(\d+)_(\d{4})_(\d{8}_\d{6})\.csv$", os.path.basename(p))
        if not m or int(m.group(2)) != YEAR:
            continue
        try:
            c = pd.read_csv(p)
        except Exception:
            continue
        if not {"player", "round", "predicted_disposals", "actual_disposals"}.issubset(c.columns):
            continue
        c["_mtime"] = os.path.getmtime(p)
        c["round"] = int(m.group(1))
        dets.append(c)

    if dets:
        d = pd.concat(dets, ignore_index=True)
        d = d[d["_mtime"] == d.groupby("round")["_mtime"].transform("max")]
        d = d.dropna(subset=["predicted_disposals", "actual_disposals"])
        d["err"] = d["predicted_disposals"] - d["actual_disposals"]

        def _natural(name):
            # CSVs store "Surname Firstname"; readers expect "Firstname Surname",
            # which is how the hand-written table this replaced always read. The
            # first name is the LAST token, so multi-token surnames ("Ah Chee",
            # "Wanganeen-Milera") stay intact — splitting on the first space would
            # turn "Ah Chee Callum" into "Chee Callum Ah".
            parts = str(name).split()
            return f"{parts[-1]} {' '.join(parts[:-1])}" if len(parts) > 1 else str(name)

        def _fmt(sub):
            return "; ".join(
                f"{_natural(r.player)} ({int(r.predicted_disposals)}→"
                f"{int(r.actual_disposals)}, {int(r.err):+d})" for r in sub.itertuples()
            )

        miss_rows = ["| Round | Top under-predictions (model too low) | Top over-predictions (model too high) |",
                     "|------:|----------------------------------------|----------------------------------------|"]
        for rnd in sorted(d["round"].unique()):
            sub = d[d["round"] == rnd]
            miss_rows.append(
                f"| {rnd} | {_fmt(sub.nsmallest(5, 'err'))} | {_fmt(sub.nlargest(5, 'err'))} |"
            )
        doc = _swap(doc, "MISSES", "\n".join(miss_rows) + "\n")

        # Reconciliation, third leg: for every round we actually hold detail for,
        # the pooled per-player rows must equal that round's n_players. Scoped to
        # those rounds on purpose — a round whose detail CSV is simply absent is a
        # coverage gap, not a crossed vintage, and must not block the whole surface
        # (README and the banner are already written by this point, so a blanket
        # abort here would leave a partial update).
        covered = sorted(d["round"].unique())
        expect = int(df[df["round"].isin(covered)]["n_players"].sum())
        if len(d) != expect:
            sys.exit(
                f"update_eval_surface: detail reconciliation FAILED — pooled per-player "
                f"rows {len(d):,} != summary n_players {expect:,} over rounds "
                f"{covered[0]}-{covered[-1]} (delta {len(d) - expect:+,}). "
                f"A backtest vintage has been crossed; refusing to write."
            )

    with open(bt_doc, "w", encoding="utf-8") as f:
        f.write(doc)

print(f"update_eval_surface: {window} | player-rounds {N:,} | "
      f"MAE {mae_s} | within5 {w5_s} | within10 {w10_s} | bias {bias_w:.3f}")
print(f"  team bias: {team_under} {bias_under:+.2f} .. {team_over} "
      f"{bias_over:+.2f} | mean-abs {mean_abs_bias:.2f}")
print(f"  player files: {PLAYER_FILE_COUNT:,}")
PYEOF
