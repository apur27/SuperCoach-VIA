"""TDD tests for scripts/fix_synthetic_dates.py.

Context: 176 player performance CSVs carry a corrupt date `2025-03-01` on a
`round=1 / year=2025` row -- a scraper artifact. The real match dates live in
data/matches/matches_2025.csv. A synthetic date poisons the
`days_since_last_game` feature (supercoach/prediction.py) for the bad row AND
the player's next game.

Two classes of corruption exist in the real data:

1. SIMPLE (137 files): the `round=1` label is correct. The three real Opening
   Round 2025 games (Sydney-Hawthorn, GWS-Collingwood, Gold Coast-Essendon)
   exist in matches with `round_num == 1`. A direct team+opponent+round lookup
   fixes these.

2. COMPOUND (39 files): Geelong-vs-Brisbane. There is NO `round_num == 1`
   Geelong-Brisbane game in matches -- the `round=1` label is ITSELF corrupt.
   The row sits chronologically between the player's round-3 and round-5 games,
   so the true game is Round 4 (2025-03-29). The fix must resolve this via the
   chronological neighbour window, not the (non-existent) round-1 match.

No network, no real files: pure DataFrame fixtures.
"""

import os
import sys

import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.fix_synthetic_dates import fix_synthetic_dates


def _matches(rows):
    cols = ["round_num", "year", "date",
            "team_1_team_name", "team_2_team_name"]
    return pd.DataFrame(rows, columns=cols)


def _player(rows):
    # minimal player schema slice the fixer touches
    cols = ["team", "year", "games_played", "opponent", "round", "date"]
    return pd.DataFrame(rows, columns=cols)


def test_simple_round1_fix_strips_time_component():
    """round=1 label correct -> direct match, time component stripped."""
    player = _player([
        ["Sydney", 2025, 100, "Hawthorn", 1, "2025-03-01"],
    ])
    matches = _matches([
        [1, 2025, "2025-08-27 19:20", "Sydney", "Hawthorn"],
    ])
    fixed, actions = fix_synthetic_dates(player, matches)
    assert fixed.loc[0, "date"] == "2025-08-27"
    assert len(actions) == 1
    assert actions[0]["status"] == "fixed"


def test_compound_geelong_brisbane_resolves_to_round4():
    """round=1 label is itself corrupt; neighbour window (R3<bad<R5) -> R4.

    Geelong-Brisbane never played a round_num==1 game, so a naive round-1
    lookup must NOT fire. The bad row sits between the player's R3 and R5
    games, so the only Geelong-Brisbane match in that window (Round 4,
    2025-03-29) is the answer.
    """
    player = _player([
        ["Brisbane Lions", 2025, 141, "West Coast", 3, "2025-03-15"],
        ["Brisbane Lions", 2025, 142, "Geelong", 1, "2025-03-01"],  # bad
        ["Brisbane Lions", 2025, 143, "Richmond", 5, "2025-03-29"],
    ])
    matches = _matches([
        [4, 2025, "2025-03-29 18:35", "Brisbane Lions", "Geelong"],
        [16, 2025, "2025-06-20 19:40", "Geelong", "Brisbane Lions"],
        ["Qualifying Final", 2025, "2025-09-05 19:40", "Geelong", "Brisbane Lions"],
        ["Grand Final", 2025, "2025-09-27 14:30", "Geelong", "Brisbane Lions"],
    ])
    fixed, actions = fix_synthetic_dates(player, matches)
    bad = fixed[fixed["opponent"] == "Geelong"].iloc[0]
    assert bad["date"] == "2025-03-29"
    assert actions[0]["status"] == "fixed"
    assert str(actions[0]["resolved_round"]) == "4"


def test_no_match_is_skipped_not_guessed():
    """No candidate pairing in matches -> skip and log, leave date untouched."""
    player = _player([
        ["Adelaide", 2025, 50, "Fremantle", 1, "2025-03-01"],
    ])
    matches = _matches([
        [1, 2025, "2025-03-07 19:40", "Sydney", "Hawthorn"],
    ])
    fixed, actions = fix_synthetic_dates(player, matches)
    assert fixed.loc[0, "date"] == "2025-03-01"  # unchanged
    assert actions[0]["status"] == "skipped"


def test_legit_rows_are_untouched():
    """Only the artifact row is targeted; a normal row is left alone."""
    player = _player([
        ["Sydney", 2025, 100, "Hawthorn", 1, "2025-03-01"],
        ["Sydney", 2025, 101, "Essendon", 5, "2025-04-01"],
        ["Sydney", 2024, 99, "Carlton", 1, "2024-03-20"],  # different year
    ])
    matches = _matches([
        [1, 2025, "2025-03-07 19:40", "Sydney", "Hawthorn"],
    ])
    fixed, _ = fix_synthetic_dates(player, matches)
    assert fixed.loc[1, "date"] == "2025-04-01"
    assert fixed.loc[2, "date"] == "2024-03-20"


def test_round_label_may_be_string_or_int():
    """round column may be '1' (str) or 1 (int) -- both must be handled."""
    player = _player([
        ["Sydney", 2025, 100, "Hawthorn", "1", "2025-03-01"],
    ])
    matches = _matches([
        [1, 2025, "2025-03-07 19:40", "Sydney", "Hawthorn"],
    ])
    fixed, actions = fix_synthetic_dates(player, matches)
    assert fixed.loc[0, "date"] == "2025-03-07"
    assert actions[0]["status"] == "fixed"
