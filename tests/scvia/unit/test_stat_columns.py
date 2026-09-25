"""Positional stat blocks for player pages: exact round trip to StatValue records."""

from __future__ import annotations

import pytest

from supercoach_via.domain.metrics import StatAggregate
from supercoach_via.publish.view_models import StatColumns, StatValue, expand_stats, to_stat_columns

NAMES = ["disposals", "goals", "tackles", "brownlow_votes"]


def _values(scope: int) -> list[StatValue]:
    aggs = [
        StatAggregate("disposals", 412.0, 18, 18, scope),
        StatAggregate("goals", 7.0, 3, 18, scope),  # 1/3 coverage-style repeating means
        StatAggregate("tackles", None, 0, 0, scope),  # not recorded in this era: unknown, not zero
        StatAggregate("brownlow_votes", 0.0, 18, 18, scope),  # a real zero
    ]
    return [a.to_stat_value() for a in aggs]


@pytest.mark.parametrize("scope", [18, 20, 0])
def test_round_trip_is_exact(scope: int) -> None:
    vals = _values(scope)
    cols = to_stat_columns(NAMES, vals, scope)
    assert cols == StatColumns(total=[412.0, 7.0, None, 0.0], observed_games=[18, 3, 0, 18], eligible_games=[18, 18, 0, 18])
    assert expand_stats(NAMES, cols, scope) == vals


def test_misaligned_or_inconsistent_values_are_refused() -> None:
    vals = _values(18)
    with pytest.raises(ValueError, match="order"):
        to_stat_columns(list(reversed(NAMES)), vals, 18)
    bad = [vals[0].model_copy(update={"mean": 99.0}), *vals[1:]]
    with pytest.raises(ValueError, match="derived"):
        to_stat_columns(NAMES, bad, 18)
    with pytest.raises(ValueError, match="coverage"):
        to_stat_columns(NAMES, vals, 25)  # values were computed for an 18-game scope
    with pytest.raises(ValueError, match="length"):
        StatColumns(total=[1.0], observed_games=[1, 2], eligible_games=[1])


def test_game_log_columns_round_trip_and_alignment() -> None:
    from datetime import date

    from supercoach_via.publish.view_models import PlayerGame, PlayerGameColumns, game_rows, to_game_columns

    rows = [
        PlayerGame(match_id="m:1", match_date=date(2026, 3, 7), date_quality="fixture_verified", stage_label="1",
                   club_id="a", opponent_club_id="b", opponent_name="B", result="W", career_game_counter=1,
                   stats=[10.0, None]),
        PlayerGame(match_id="m:2", match_date=None, date_quality="unknown", stage_label="Qualifying Final",
                   club_id="a", opponent_club_id=None, opponent_name=None, result=None, career_game_counter=None,
                   stats=[0.0, 2.0]),
    ]  # fmt: skip
    cols = to_game_columns(rows)
    assert cols.match_id == ["m:1", "m:2"] and cols.stats == [[10.0, None], [0.0, 2.0]]
    assert game_rows(cols) == rows
    assert game_rows(to_game_columns([])) == []
    with pytest.raises(ValueError, match="length"):
        PlayerGameColumns.model_validate({**cols.model_dump(), "result": ["W"]})
