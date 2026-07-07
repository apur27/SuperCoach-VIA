"""Regression guard for prediction-file round detection (Surveyor CR-1).

The weekly refresh must select the *newest* prediction CSV to label the
cheat sheet and the afl-insights recap. A prior bug selected the "latest"
file with a lexicographic `ls | sort | tail -1`, which ranks
`next_round_9_...` above `next_round_18_...` because the string "9" sorts
after "1". That published weekly recaps under the wrong round label for weeks.

These tests pin the correct behaviour of the shared helper in
generate_weekly_cheat_sheet.py (mtime-based selection + numeric round
extraction) so the lexicographic bug cannot be reintroduced. weekly_refresh.sh
now shells out to this same helper, so this is the single source of truth.
"""
import importlib.util
import os
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MODULE_PATH = REPO_ROOT / "scripts" / "generate_weekly_cheat_sheet.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gwcs_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gwcs = _load_module()


def _touch(path: pathlib.Path, mtime: float) -> None:
    path.write_text("player,predicted_disposals\nA,20\n")
    os.utime(path, (mtime, mtime))


def test_selects_newest_round_not_lexicographic_max(tmp_path):
    """Round 18 (newest mtime) must beat round 9, even though '9' > '1' lexically."""
    # Round 9 written first (older), round 18 written later (newer) — exactly the
    # trap that broke the old `ls | sort | tail -1` logic.
    _touch(tmp_path / "next_round_9_prediction_20260511_2346.csv", mtime=1_000_000)
    _touch(tmp_path / "next_round_18_prediction_20260707_1200.csv", mtime=2_000_000)

    latest = gwcs.find_latest_prediction(tmp_path)
    assert gwcs.parse_round_from_filename(latest) == 18


def test_parse_round_is_numeric(tmp_path):
    p = tmp_path / "next_round_7_prediction_20260430_1724.csv"
    _touch(p, mtime=1_000_000)
    assert gwcs.parse_round_from_filename(p) == 7


def test_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        gwcs.find_latest_prediction(tmp_path)
