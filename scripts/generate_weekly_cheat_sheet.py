#!/usr/bin/env python3
"""Generate the weekly cheat-sheet markdown from the latest prediction CSV.

Reads the most recent ``data/prediction/next_round_<N>_prediction_<timestamp>.csv``,
sorts by predicted disposals, and writes a markdown file to
``docs/weekly/round-<N>-<year>.md`` (and overwrites
``docs/weekly/round-current-<year>.md`` so the README link is stable).

Usage from repo root::

    python scripts/generate_weekly_cheat_sheet.py
    python scripts/generate_weekly_cheat_sheet.py --csv path/to/file.csv
    python scripts/generate_weekly_cheat_sheet.py --year 2026

The script is intentionally pure-pandas - no model retraining, no scraping.
It is safe to run as a docs refresh step.

Reproducibility: deterministic. No randomness. Output depends only on the
input CSV file.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTION_DIR = REPO_ROOT / "data" / "prediction"
WEEKLY_DIR = REPO_ROOT / "docs" / "weekly"

PRED_FILE_RE = re.compile(
    r"next_round_(?P<round>\d+)_prediction_(?P<ts>\d{8}_\d{4})\.csv$"
)


def find_latest_prediction(prediction_dir: Path) -> Path:
    """Return the path to the most recent next_round_*_prediction_*.csv."""
    candidates = sorted(
        glob.glob(str(prediction_dir / "next_round_*_prediction_*.csv")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No prediction CSVs found under {prediction_dir}. "
            "Run prediction.py first."
        )
    return Path(candidates[0])


def parse_round_from_filename(path: Path) -> int | None:
    m = PRED_FILE_RE.search(path.name)
    return int(m.group("round")) if m else None


def render_markdown(
    df: pd.DataFrame, round_num: int | None, year: int, source_path: Path
) -> str:
    """Render the cheat-sheet markdown from a sorted prediction dataframe."""
    df = df.sort_values("predicted_disposals", ascending=False).reset_index(drop=True)

    n_rows = len(df)
    pred_min = df["predicted_disposals"].min()
    pred_max = df["predicted_disposals"].max()
    pred_mean = df["predicted_disposals"].mean()

    round_label = f"Round {round_num}" if round_num is not None else "Next round"
    rel_csv = os.path.relpath(source_path, REPO_ROOT)

    lines: list[str] = []
    lines.append(f"# {round_label} cheat sheet - {year} (experimental v0)")
    lines.append("")
    lines.append(
        "> [← Back to fan landing page](../start-here-no-code.md) | "
        "[← Back to main README](../../README.md)"
    )
    lines.append("")
    lines.append(
        "> **Status: experimental v0.** Predictions are model output, not "
        "certainties - typical error is ±4 disposals. Always cross-check "
        "team lists before lockout."
    )
    lines.append("")
    lines.append(
        f"*Source: `{rel_csv}` ({n_rows} player rows, mean predicted "
        f"disposals {pred_mean:.2f}, range {pred_min:.1f}-{pred_max:.1f}).*"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## Top 30 predicted disposal leaders - {round_label}, {year}")
    lines.append("")
    lines.append("| Rank | Player | Team | Predicted disposals |")
    lines.append("|------|--------|------|--------------------:|")
    for i, row in df.head(30).iterrows():
        lines.append(
            f"| {i + 1} | {row['player']} | {row['team']} | "
            f"{row['predicted_disposals']:.1f} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Top 3 per club")
    lines.append("")
    lines.append(
        "The model's three highest-predicted players from each club. Useful "
        "for trade-target sanity checks."
    )
    lines.append("")
    for team in sorted(df["team"].unique()):
        sub = df[df["team"] == team].head(3)
        lines.append(f"### {team}")
        for rank, (_, r) in enumerate(sub.iterrows(), start=1):
            lines.append(
                f"{rank}. {r['player']} - {r['predicted_disposals']:.1f}"
            )
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Reading this cheat sheet")
    lines.append("")
    lines.append(
        "- **Predictions are disposals only**, not SuperCoach fantasy points."
    )
    lines.append(
        "- **Typical error: ±4 disposals** (see "
        "[backtest results](../afl-backtest-2026.md))."
    )
    lines.append(
        "- **The model is slow on role changes and tag jobs.**"
    )
    lines.append("- **Late outs are not handled.** Always check team lists.")
    lines.append("")
    lines.append(
        "For the full honest version, see "
        "[How to use this for SuperCoach](../how-to-use-this-for-supercoach.md)."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## How this page is generated")
    lines.append("")
    lines.append(
        "Run `python scripts/generate_weekly_cheat_sheet.py` from the repo "
        "root. The script reads the most recent prediction CSV and writes "
        "this markdown."
    )
    lines.append("")
    lines.append(
        f"*Last generated: {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}*"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to a specific prediction CSV (default: most recent).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=_dt.datetime.now().year,
        help="Year to use in the cheat sheet title (default: current).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=WEEKLY_DIR,
        help="Where to write the markdown (default: docs/weekly/).",
    )
    args = parser.parse_args(argv)

    csv_path: Path = args.csv if args.csv else find_latest_prediction(PREDICTION_DIR)
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    expected = {"player", "team", "predicted_disposals"}
    missing = expected - set(df.columns)
    if missing:
        print(
            f"ERROR: prediction CSV is missing columns: {sorted(missing)}",
            file=sys.stderr,
        )
        return 2

    round_num = parse_round_from_filename(csv_path)
    md = render_markdown(df, round_num, args.year, csv_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if round_num is not None:
        per_round = args.out_dir / f"round-{round_num:02d}-{args.year}.md"
        per_round.write_text(md, encoding="utf-8")
        print(f"Wrote {per_round}")

    current = args.out_dir / f"round-current-{args.year}.md"
    current.write_text(md, encoding="utf-8")
    print(f"Wrote {current}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
