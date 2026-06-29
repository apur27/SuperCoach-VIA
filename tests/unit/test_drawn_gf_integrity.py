"""Data-integrity guard for collapsed drawn Grand Finals.

The 2010 Grand Final (Collingwood v St Kilda) was DRAWN and replayed the
following week. They are two distinct games with different stat lines, so a
player who appeared in both must have two rows. Commit 58f1a4f20 deduped
player files on (year, round, opponent) and wrongly collapsed Steele
Sidebottom's two 2010 GF rows into one, deleting a real game's stats.

This test reproduces that bug and guards against any future (year, round,
opponent) dedup re-collapsing a drawn GF + replay.
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent.parent.parent
SIDEBOTTOM = (
    REPO / "data" / "player_data"
    / "sidebottom_steele_02011991_performance_details.csv"
)


def _gp(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace("↑", "", regex=False).str.replace(
        "↓", "", regex=False
    )
    return pd.to_numeric(cleaned, errors="coerce")


def test_sidebottom_has_both_2010_grand_finals():
    df = pd.read_csv(SIDEBOTTOM, low_memory=False)
    gf = df[(df["year"] == 2010) & (df["round"].astype(str) == "GF")]
    # the drawn GF and the replay are two distinct games
    assert len(gf) == 2, f"expected 2 rows for 2010 GF (draw + replay), got {len(gf)}"
    # one was a Draw, one a Win (Collingwood drew then won the replay)
    assert set(gf["result"].astype(str)) == {"D", "W"}
    # distinct running game counters 35 (draw) and 36 (replay)
    assert set(_gp(gf["games_played"]).astype("Int64").tolist()) == {35, 36}


def test_sidebottom_no_missing_game_rows():
    """Row count must equal the authoritative games_played counter max:
    every counted game has a detailed stat row (no silent stat loss)."""
    df = pd.read_csv(SIDEBOTTOM, low_memory=False)
    counter_max = int(_gp(df["games_played"]).max())
    assert len(df) == counter_max, (
        f"row count {len(df)} != games_played max {counter_max}: a real game's "
        "stat row is missing"
    )
