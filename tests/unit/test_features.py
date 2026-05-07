"""Smoke test: feature data loads and has expected shape."""

import glob
import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PLAYER_DIR = os.path.join(REPO_ROOT, "data", "player_data")


def test_player_data_exists():
    files = glob.glob(os.path.join(PLAYER_DIR, "*performance*.csv"))
    assert len(files) > 1000, f"Expected 1000+ player files, got {len(files)}"


def test_player_file_has_expected_columns():
    files = glob.glob(os.path.join(PLAYER_DIR, "*performance*.csv"))
    df = pd.read_csv(files[0])
    expected = {"year", "disposals", "goals", "kicks", "handballs"}
    missing = expected - set(df.columns)
    assert not missing, f"Missing columns: {missing}"


def test_no_all_negative_disposals():
    """Disposals should be non-negative (NaN allowed — empty cells exist in raw data)."""
    files = glob.glob(os.path.join(PLAYER_DIR, "*performance*.csv"))[:50]
    for f in files:
        df = pd.read_csv(f)
        if "disposals" in df.columns:
            non_null = df["disposals"].dropna()
            assert (non_null >= 0).all(), f"Negative disposals in {f}"
