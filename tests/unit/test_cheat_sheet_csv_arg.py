"""
Verify that generate_weekly_cheat_sheet.py honours the --csv argument and
bypasses the mtime-based find_latest_prediction fallback when a path is given.
"""
import sys
import os
import csv
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.generate_weekly_cheat_sheet import find_latest_prediction


def _make_prediction_csv(directory: Path, round_num: int, timestamp: str) -> Path:
    fname = f"next_round_{round_num}_prediction_{timestamp}.csv"
    p = directory / fname
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["player", "team", "predicted_disposals"])
        writer.writerow(["Test Player", "Test Team", "25.0"])
    return p


def test_find_latest_prediction_returns_highest_round(tmp_path):
    """find_latest_prediction must pick the highest round, not the newest mtime."""
    pred_dir = tmp_path / "prediction"
    pred_dir.mkdir()
    # Write lower round last (so it has a newer mtime — mtime would pick this incorrectly)
    high = _make_prediction_csv(pred_dir, 19, "20260707_1800")
    import time; time.sleep(0.01)
    low = _make_prediction_csv(pred_dir, 9, "20260430_1822")
    result = find_latest_prediction(pred_dir)
    assert result == high, f"Expected R19 CSV, got {result.name}"


def test_find_latest_prediction_raises_when_empty(tmp_path):
    """find_latest_prediction must raise when no CSVs exist."""
    empty_dir = tmp_path / "prediction"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        find_latest_prediction(empty_dir)


def test_find_latest_prediction_tie_broken_by_timestamp(tmp_path):
    """Same round, different timestamps — pick the later timestamp."""
    pred_dir = tmp_path / "prediction"
    pred_dir.mkdir()
    early = _make_prediction_csv(pred_dir, 20, "20260714_0800")
    late = _make_prediction_csv(pred_dir, 20, "20260714_1200")
    result = find_latest_prediction(pred_dir)
    assert result == late, f"Expected later timestamp CSV, got {result.name}"
