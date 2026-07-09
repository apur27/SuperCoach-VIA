"""F10: assert all_time_top_100.csv and data/top100/all_time_top_100.csv are in sync.

Both files are written by top_players_comprehensive.py in the same run:
  1. compile_all_time_top_100() -> data/top100/all_time_top_100.csv  (player, all_time_score)
  2. format_top_100()           -> all_time_top_100.csv              (Serial Number, Player Name, ...)

These tests assert both files exist, have 100 rows, and agree on player ordering.
"""
import pandas as pd
import pytest


DATA_PATH = "data/top100/all_time_top_100.csv"
FORMATTED_PATH = "all_time_top_100.csv"


def _pid_to_name(player_id: str) -> str:
    """Convert 'bartlett_kevin_06031947' -> 'Kevin Bartlett'."""
    parts = player_id.split("_")
    # parts[0]=lastname, parts[1]=firstname, parts[2]=DOB
    # Handle hyphenated names encoded as double-underscore (if any)
    last = parts[0].replace("-", " ").title()
    first = parts[1].replace("-", " ").title()
    return f"{first} {last}"


def test_both_files_exist():
    import os
    assert os.path.exists(DATA_PATH), f"{DATA_PATH} missing"
    assert os.path.exists(FORMATTED_PATH), f"{FORMATTED_PATH} missing"


def test_both_files_have_100_rows():
    data = pd.read_csv(DATA_PATH)
    formatted = pd.read_csv(FORMATTED_PATH)
    assert len(data) == 100, f"{DATA_PATH} has {len(data)} rows, expected 100"
    assert len(formatted) == 100, f"{FORMATTED_PATH} has {len(formatted)} rows, expected 100"


def test_data_file_schema():
    data = pd.read_csv(DATA_PATH)
    assert "player" in data.columns, "data/top100 missing 'player' column"
    assert "all_time_score" in data.columns, "data/top100 missing 'all_time_score' column"
    assert data["all_time_score"].dtype in ("float64", "float32"), "all_time_score should be numeric"


def test_formatted_file_schema():
    formatted = pd.read_csv(FORMATTED_PATH)
    for col in ("Serial Number", "Player Name", "Footy Teams", "Comment"):
        assert col in formatted.columns, f"all_time_top_100.csv missing column '{col}'"


def test_player_ordering_matches():
    """The player at rank N in the data file must match the player at rank N in the formatted file."""
    data = pd.read_csv(DATA_PATH)
    formatted = pd.read_csv(FORMATTED_PATH)
    mismatches = []
    for i, (pid, formatted_name) in enumerate(zip(data["player"], formatted["Player Name"]), start=1):
        derived_name = _pid_to_name(pid)
        # Compare case-insensitively; ignore middle names in formatted file
        derived_parts = set(derived_name.lower().split())
        formatted_parts = set(formatted_name.lower().split())
        # Both first and last name must appear in the formatted name
        if not derived_parts.issubset(formatted_parts | {"jr", "sr", "ii", "iii"}):
            mismatches.append(f"  rank {i}: data={pid!r} -> derived={derived_name!r}, formatted={formatted_name!r}")
    assert not mismatches, "Player ordering mismatch between data and formatted files:\n" + "\n".join(mismatches[:5])


def test_serial_numbers_are_sequential():
    formatted = pd.read_csv(FORMATTED_PATH)
    serials = list(formatted["Serial Number"])
    assert serials == list(range(1, 101)), f"Serial numbers are not 1-100 sequential"
