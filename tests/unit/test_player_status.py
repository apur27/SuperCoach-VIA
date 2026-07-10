"""Unit tests for scripts/generate_player_status.py.

The generator scans data/player_data/*_performance_details.csv and emits
data/player_status.csv with status=active iff a player's max(year) equals the
current season, where the current season is DERIVED from the data (the global
max year across all players), never hardcoded.
"""
import importlib.util
import os
import warnings

import pandas as pd
import pytest

_SPEC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "generate_player_status.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_player_status", _SPEC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_player(dirpath, slug, years, dates=None, games=None):
    """Write a minimal synthetic performance CSV for one player."""
    n = len(years)
    dates = dates or [f"{y}-06-01" for y in years]
    games = games or list(range(1, n + 1))
    df = pd.DataFrame({"team": ["X"] * n, "year": years, "date": dates,
                       "games_played": games, "goals": [0] * n})
    df.to_csv(os.path.join(dirpath, f"{slug}_performance_details.csv"), index=False)


@pytest.fixture
def player_dir(tmp_path):
    d = tmp_path / "player_data"
    d.mkdir()
    # current-season player (max year 2024) and a retired one (max year 2019)
    _write_player(str(d), "dangerfield_patrick_05041990", [2022, 2023, 2024])
    _write_player(str(d), "martin_dustin_26061991", [2017, 2018, 2019])
    return str(d)


def test_active_player_detected(player_dir):
    mod = _load_module()
    df = mod.build_status_table(player_dir)
    row = df[df["player_slug"] == "dangerfield_patrick_05041990"].iloc[0]
    assert row["status"] == "active"
    assert row["last_year"] == 2024


def test_retired_player_detected(player_dir):
    mod = _load_module()
    df = mod.build_status_table(player_dir)
    row = df[df["player_slug"] == "martin_dustin_26061991"].iloc[0]
    assert row["status"] == "retired"
    assert row["last_year"] == 2019


def test_current_season_derived_from_data(tmp_path):
    """Current season is the global max year, not a hardcoded constant."""
    d = tmp_path / "player_data"
    d.mkdir()
    # No player reaches 2026 here; the max year in the data is 2021.
    _write_player(str(d), "aaa_one_01011990", [2019, 2020, 2021])
    _write_player(str(d), "bbb_two_01011991", [2018, 2019])
    mod = _load_module()
    df = mod.build_status_table(str(d))
    # The 2021 player must be active (2021 is the derived current season),
    # the 2019 player retired — proving the season came from data, not 2026.
    assert df[df["player_slug"] == "aaa_one_01011990"].iloc[0]["status"] == "active"
    assert df[df["player_slug"] == "bbb_two_01011991"].iloc[0]["status"] == "retired"


def test_output_schema(player_dir):
    mod = _load_module()
    df = mod.build_status_table(player_dir)
    expected = ["player_slug", "first_name", "last_name", "dob_key",
                "last_year", "last_game_date", "career_games", "status"]
    assert list(df.columns) == expected


def test_idempotent(player_dir):
    mod = _load_module()
    a = mod.build_status_table(player_dir)
    b = mod.build_status_table(player_dir)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_age_over_45_active_warns(tmp_path):
    """An active player whose age would exceed 45 triggers a review warning."""
    d = tmp_path / "player_data"
    d.mkdir()
    # born 1970, active in 2024 -> age 54 -> implausible, should warn
    _write_player(str(d), "old_timer_01011970", [2023, 2024])
    _write_player(str(d), "young_gun_01012000", [2023, 2024])
    mod = _load_module()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod.build_status_table(str(d))
    msgs = [str(w.message) for w in caught]
    assert any("old_timer_01011970" in m for m in msgs)
    assert not any("young_gun_01012000" in m for m in msgs)
