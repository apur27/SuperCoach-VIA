"""_load_top30_player_deviation must pick the newest run by FULL timestamp.

Backtest artifacts are named `..._<YEAR>_<YYYYMMDD>_<HHMMSS>.csv`. The dedup key
took `base.rsplit("_", 1)[-1]`, which is the HHMMSS field alone — the DATE was
discarded. Sorting on time-of-day means a re-run that starts earlier in the clock
day than the run it supersedes silently loses, and the STALE vintage is published.

Today all ten multi-vintage 2026 rounds happen to resolve correctly, purely
because the authoritative runs also had the later wall-clock times. That is luck,
not correctness, and it is the same failure class as the tainted-provenance
incident that started this cycle: a selector that looks right because its inputs
have so far been kind to it.

Hermetic: synthetic CSVs under tmp_path, no real data files touched.
"""
import os
import sys

import pandas as pd
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import update_team_analysis as uta

COLS = ["player", "team", "round", "year",
        "predicted_disposals", "actual_disposals", "error", "abs_error"]


def _vintage(bt_dir, rnd, ts, actual, year=2026, player="Smith John"):
    pd.DataFrame(
        [[player, "Test Team", rnd, year, 20.0, actual, 20.0 - actual,
          abs(20.0 - actual)]],
        columns=COLS,
    ).to_csv(bt_dir / f"prediction_vs_actual_round_{rnd}_{year}_{ts}.csv", index=False)


def test_later_date_wins_even_with_an_earlier_time_of_day(tmp_path):
    """THE BUG: the superseding run started earlier in the clock day.

    05-18 at 19:18 is authoritative; 05-25 at 18:21 supersedes it by date but
    loses on HHMMSS. A date-blind key publishes the stale actual (10.0).
    """
    _vintage(tmp_path, 1, "20260518_191837", actual=10.0)   # older date, later time
    _vintage(tmp_path, 1, "20260525_182141", actual=99.0)   # newer date, earlier time

    df = uta._load_top30_player_deviation(2026, str(tmp_path))

    assert not df.empty
    assert df.iloc[0]["avg_actual"] == 99.0, (
        "selected the stale vintage — the dedup key is comparing time-of-day, "
        "not the full YYYYMMDD_HHMMSS timestamp"
    )


def test_later_time_wins_within_the_same_date(tmp_path):
    """Control: the fix must not break same-day ordering."""
    _vintage(tmp_path, 2, "20260518_101500", actual=10.0)
    _vintage(tmp_path, 2, "20260518_191837", actual=42.0)

    df = uta._load_top30_player_deviation(2026, str(tmp_path))
    assert df.iloc[0]["avg_actual"] == 42.0


def test_current_repo_vintages_still_resolve_the_same(tmp_path):
    """The real files resolve correctly today by luck; the fix must be a no-op
    for them, so this change cannot move a published figure."""
    _vintage(tmp_path, 1, "20260430_142823", actual=1.0)
    _vintage(tmp_path, 1, "20260511_191837", actual=2.0)    # authoritative
    df = uta._load_top30_player_deviation(2026, str(tmp_path))
    assert df.iloc[0]["avg_actual"] == 2.0


def test_malformed_timestamp_does_not_crash_the_loader(tmp_path):
    """A file that does not carry a parseable timestamp must be skipped, not
    allowed to throw and take the whole backtest section down."""
    _vintage(tmp_path, 3, "20260518_191837", actual=7.0)
    pd.DataFrame(
        [["Smith John", "Test Team", 3, 2026, 20.0, 5.0, 15.0, 15.0]], columns=COLS
    ).to_csv(tmp_path / "prediction_vs_actual_round_3_2026_NOTATIMESTAMP.csv",
             index=False)

    df = uta._load_top30_player_deviation(2026, str(tmp_path))
    assert not df.empty
    assert df.iloc[0]["avg_actual"] == 7.0
