"""Generate per-player PNG prediction cards from the latest prediction CSV.

This is a scaffold. Running it as `python scripts/generate_player_cards.py`
produces one PNG per top-N player (defaults to top 30) in `assets/cards/`,
each card showing:

    - Player name
    - Team
    - Predicted disposals (the headline number)
    - Trend indicator vs the player's last 5-round actual mean (up / flat / down)

Trend logic:
    - If we cannot find recent actuals for a player (new player, name mismatch
      with the per-player files), the trend is reported as "n/a".
    - Otherwise: pred - last5_mean > +1.5 disposals -> "up";
                 pred - last5_mean < -1.5 disposals -> "down";
                 else "flat".

The card layout intentionally avoids fantasy-points language - we predict
disposals only. See docs/how-to-use-this-for-supercoach.md.

Usage:

    python scripts/generate_player_cards.py
    python scripts/generate_player_cards.py --top 50
    python scripts/generate_player_cards.py --csv data/prediction/next_round_9_prediction_20260430_2322.csv
    python scripts/generate_player_cards.py --out-dir assets/cards/round-09

The script is intentionally conservative: it never overwrites a card with the
same filename in the same run unless --overwrite is set, and it logs a summary
of what was written / skipped at the end.

This is a scaffold - it will not generate cards on import. The runnable entry
point is the `if __name__ == "__main__":` block at the bottom.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# matplotlib is imported lazily inside main() so that --help works without it
# installed.

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRED_DIR = REPO_ROOT / "data" / "prediction"
DEFAULT_PLAYER_DIR = REPO_ROOT / "data" / "player_data"
DEFAULT_OUT_DIR = REPO_ROOT / "assets" / "cards"


@dataclass
class CardData:
    player: str
    team: str
    predicted: float
    last5_mean: Optional[float]
    trend: str  # "up", "flat", "down", "n/a"

    @property
    def trend_glyph(self) -> str:
        return {"up": "^", "flat": "-", "down": "v", "n/a": "?"}[self.trend]

    @property
    def trend_color(self) -> str:
        return {
            "up": "#1b9e77",
            "flat": "#7570b3",
            "down": "#d95f02",
            "n/a": "#888888",
        }[self.trend]


# --- I/O helpers -----------------------------------------------------------


def find_latest_prediction_csv(pred_dir: Path) -> Path:
    """Return the most recently modified next_round_*_prediction_*.csv."""
    candidates = sorted(
        pred_dir.glob("next_round_*_prediction_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No prediction CSV found in {pred_dir}. "
            "Run prediction.py first."
        )
    return candidates[0]


def load_predictions(csv_path: Path):
    import pandas as pd  # local import to keep --help cheap

    df = pd.read_csv(csv_path)
    expected = {"player", "team", "predicted_disposals"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Prediction CSV {csv_path} is missing columns: {sorted(missing)}. "
            f"Got: {list(df.columns)}"
        )
    df = df.sort_values("predicted_disposals", ascending=False).reset_index(drop=True)
    return df


def find_player_file(player_dir: Path, player_name: str) -> Optional[Path]:
    """Best-effort: match `Surname Firstname` -> `surname_firstname_DDMMYYYY_*.csv`.

    Player CSVs in this repo are named:
        <surname>_<firstname>_<DDMMYYYY>_performance_details.csv

    The prediction CSV uses `Surname Firstname`. We translate to lowercase
    snake_case and glob. If the prediction has an apostrophe or hyphen, we
    strip them for the match.
    """
    raw = player_name.lower().replace("'", "").replace("-", "_")
    parts = raw.split()
    if len(parts) < 2:
        return None
    surname = parts[0]
    firstname = "_".join(parts[1:])
    pattern = f"{surname}_{firstname}_*_performance_details.csv"
    matches = sorted(player_dir.glob(pattern))
    if not matches:
        return None
    # If multiple (unlikely), prefer the largest file (typically more games).
    matches.sort(key=lambda p: p.stat().st_size, reverse=True)
    return matches[0]


def last5_mean_disposals(player_file: Path) -> Optional[float]:
    """Mean disposals over the player's most recent 5 games. None if unavailable."""
    import pandas as pd

    try:
        df = pd.read_csv(player_file)
    except Exception:
        return None
    if "disposals" not in df.columns:
        return None
    # Sort by year then round if present; otherwise by file order (assumed chronological).
    sort_cols = [c for c in ("year", "round") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    tail = df.tail(5)
    if tail.empty:
        return None
    return float(tail["disposals"].astype(float).mean())


def classify_trend(pred: float, last5: Optional[float], threshold: float = 1.5) -> str:
    if last5 is None:
        return "n/a"
    diff = pred - last5
    if diff > threshold:
        return "up"
    if diff < -threshold:
        return "down"
    return "flat"


# --- Card rendering --------------------------------------------------------


def render_card(card: CardData, out_path: Path) -> None:
    """Render a single PNG card. Pure matplotlib, no external assets."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(4.0, 5.0), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background card
    bg = FancyBboxPatch(
        (0.04, 0.04), 0.92, 0.92,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=2.0,
        edgecolor="#222222",
        facecolor="#fafafa",
    )
    ax.add_patch(bg)

    # Header bar
    header = FancyBboxPatch(
        (0.04, 0.78), 0.92, 0.18,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=0,
        facecolor="#222222",
    )
    ax.add_patch(header)

    # Player name
    ax.text(0.5, 0.90, card.player, ha="center", va="center",
            fontsize=14, color="white", fontweight="bold")
    ax.text(0.5, 0.835, card.team, ha="center", va="center",
            fontsize=10, color="#cccccc")

    # Predicted disposals - the headline
    ax.text(0.5, 0.55, f"{card.predicted:.1f}", ha="center", va="center",
            fontsize=44, color="#222222", fontweight="bold")
    ax.text(0.5, 0.42, "predicted disposals", ha="center", va="center",
            fontsize=10, color="#555555")

    # Trend
    if card.last5_mean is not None:
        last5_str = f"last 5 avg: {card.last5_mean:.1f}"
    else:
        last5_str = "last 5 avg: n/a"
    ax.text(0.5, 0.30, last5_str, ha="center", va="center",
            fontsize=9, color="#666666")
    ax.text(0.5, 0.20, f"trend  {card.trend_glyph}  {card.trend}",
            ha="center", va="center",
            fontsize=12, color=card.trend_color, fontweight="bold")

    # Footer disclaimer - consistency with how-to doc
    ax.text(0.5, 0.10, "MAE around 4 disposals - tilt, not certainty",
            ha="center", va="center",
            fontsize=7, color="#888888", style="italic")
    ax.text(0.5, 0.065, "SuperCoach VIA", ha="center", va="center",
            fontsize=7, color="#aaaaaa")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --- Filename helpers ------------------------------------------------------


def safe_filename(player_name: str) -> str:
    return (
        player_name.lower()
        .replace(" ", "_")
        .replace("'", "")
        .replace("-", "_")
        .replace("/", "_")
    )


# --- Main ------------------------------------------------------------------


def build_cards(rows: Iterable[dict], player_dir: Path) -> Iterable[CardData]:
    for row in rows:
        player = str(row["player"])
        pred = float(row["predicted_disposals"])
        team = str(row["team"])

        player_file = find_player_file(player_dir, player)
        last5 = last5_mean_disposals(player_file) if player_file else None
        trend = classify_trend(pred, last5)

        yield CardData(
            player=player,
            team=team,
            predicted=pred,
            last5_mean=last5,
            trend=trend,
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate per-player PNG prediction cards.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to the prediction CSV. Defaults to the latest file in "
             "data/prediction/.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="How many top players to render cards for (default: 30).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to write cards into (default: assets/cards/).",
    )
    parser.add_argument(
        "--player-dir",
        type=Path,
        default=DEFAULT_PLAYER_DIR,
        help="Directory of per-player CSVs for trend lookup "
             "(default: data/player_data/).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing card files (default: skip).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing PNGs.",
    )
    args = parser.parse_args(argv)

    csv_path: Path = (
        args.csv if args.csv else find_latest_prediction_csv(DEFAULT_PRED_DIR)
    )
    print(f"Reading predictions from: {csv_path}")
    df = load_predictions(csv_path)
    head = df.head(args.top)
    print(f"Generating cards for top {len(head)} of {len(df)} players")

    written = 0
    skipped = 0
    failed = 0
    for rank, row in enumerate(head.to_dict(orient="records"), start=1):
        try:
            card = next(iter(build_cards([row], args.player_dir)))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  ! could not build card for {row.get('player')!r}: {exc}",
                  file=sys.stderr)
            failed += 1
            continue
        out_name = f"{rank:02d}_{safe_filename(card.player)}.png"
        out_path = args.out_dir / out_name
        if args.dry_run:
            print(f"  [dry-run] would write {out_path}  "
                  f"({card.predicted:.1f} disp, trend={card.trend})")
            continue
        if out_path.exists() and not args.overwrite:
            print(f"  - skip (exists): {out_path}")
            skipped += 1
            continue
        try:
            render_card(card, out_path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  ! render failed for {card.player}: {exc}",
                  file=sys.stderr)
            failed += 1
            continue
        print(f"  + wrote {out_path}  "
              f"({card.predicted:.1f} disp, trend={card.trend})")
        written += 1

    print()
    print(f"Done. wrote={written} skipped={skipped} failed={failed} "
          f"(dry_run={args.dry_run})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
