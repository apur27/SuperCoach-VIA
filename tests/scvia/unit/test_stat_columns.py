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


def test_shared_match_facts_restore_date_stage_and_opponent() -> None:
    from datetime import date

    from supercoach_via.publish.view_models import (
        MatchSummary,
        PlayerGame,
        PlayerSeasonGames,
        TeamScore,
        apply_match_facts,
        game_rows,
        share_match_facts,
        to_game_columns,
    )
    from supercoach_via.publish.web_data import canonical_json_bytes

    rows = [
        PlayerGame(match_id="m:1", match_date=date(2026, 3, 7), date_quality="source", stage_label="Round 1",
                   club_id="a", opponent_club_id="b", opponent_name="Demo Ridge", result="W", career_game_counter=1,
                   stats=[10.0]),
        PlayerGame(match_id="m:2", match_date=date(2026, 3, 14), date_quality="source", stage_label="Round 2",
                   club_id="a", opponent_club_id="c", opponent_name="Demo Coast", result="L", career_game_counter=2,
                   stats=[4.0]),
    ]  # fmt: skip
    full = PlayerSeasonGames(player_id="p", season=2026, stat_columns=["disposals"], games=to_game_columns(rows))
    slim = share_match_facts(full, "matches/2026/index.json")
    assert slim.match_facts == "matches/2026/index.json"
    assert slim.games.match_date == [] and slim.games.opponent_name == []
    assert slim.games.stats == [[10.0], [4.0]] and slim.games.date_quality == ["source", "source"]
    assert len(canonical_json_bytes(slim.model_dump(mode="json"))) < len(canonical_json_bytes(full.model_dump(mode="json")))

    def summary(mid: str, day: date, stage: str, home: str, away: str, away_name: str) -> MatchSummary:
        return MatchSummary(
            match_id=mid, season=2026, stage_id="r01", stage_label=stage, stage_type="regular", round_number=1,
            stage_order=1, replay_occurrence=1, local_start=None, match_date=day, date_precision="day",
            status="complete", venue=None,
            home=TeamScore(club_id=home, name="Demo Harbour", goals=10, behinds=8, score=68),
            away=TeamScore(club_id=away, name=away_name, goals=8, behinds=6, score=54),
            winner_club_id=home,
        )

    facts = {
        "m:1": summary("m:1", date(2026, 3, 7), "Round 1", "a", "b", "Demo Ridge"),
        "m:2": summary("m:2", date(2026, 3, 14), "Round 2", "a", "c", "Demo Coast"),
    }
    restored = game_rows(apply_match_facts(slim.games, facts))
    assert [(r.match_date, r.stage_label, r.opponent_name, r.result, r.stats) for r in restored] == [
        (date(2026, 3, 7), "Round 1", "Demo Ridge", "W", [10.0]),
        (date(2026, 3, 14), "Round 2", "Demo Coast", "L", [4.0]),
    ]


def test_box_score_columns_round_trip_and_alignment() -> None:
    from supercoach_via.publish.view_models import BoxScoreColumns, BoxScoreRow, box_rows, to_box_columns

    rows = [BoxScoreRow(player_id="p:1", name="A", stats=[1.0, None]), BoxScoreRow(player_id="p:2", name="B", stats=[0.0, 3.0])]
    cols = to_box_columns(rows)
    assert cols == BoxScoreColumns(player_id=["p:1", "p:2"], name=["A", "B"], stats=[[1.0, None], [0.0, 3.0]])
    assert box_rows(cols) == rows and box_rows(to_box_columns([])) == []
    with pytest.raises(ValueError, match="length"):
        BoxScoreColumns(player_id=["p:1"], name=[], stats=[[1.0]])
