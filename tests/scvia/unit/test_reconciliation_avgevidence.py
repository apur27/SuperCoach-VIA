"""Evidence from the source's own printed season figures for whole-team blank columns (DESIGN section 15 note).

AFL Tables prints a zero as a blank (a team total too: no team total of "0" occurs in the frozen capture), and its
notes state that games where a statistic is missing do not count in a player's average. So a player's printed
season average tells whether a team-match whose column is entirely blank was counted (a recorded zero) or not
(not recorded) - but only when exactly one count of such games reproduces the printed average.
"""

from __future__ import annotations

from decimal import Decimal

from supercoach_via.reconciliation import avgevidence as AV

K1 = ("m1", "Alpha")
K2 = ("m2", "Alpha")


def _g(key: tuple[str, str], state: str, value: int | None = None) -> AV.GameState:
    return AV.GameState(key, state, None if value is None else Decimal(value))


def test_the_counted_games_solution_must_be_unique() -> None:
    # total 10 over 4 or 5 games: 2.50 vs 2.00 - unique
    assert AV.counted_solution(Decimal(10), "2.50", known=4, ambiguous=1) == 0
    assert AV.counted_solution(Decimal(10), "2.00", known=4, ambiguous=1) == 1
    # 1/7 = 0.1429 prints 0.14; 1/8 = 0.125 lies outside 0.14's rounding interval
    assert AV.counted_solution(Decimal(1), "0.14", known=7, ambiguous=1) == 0
    # a total of 0 prints no average: no evidence
    assert AV.counted_solution(Decimal(0), "", known=3, ambiguous=1) is None
    # several counts reproduce the printed value: no evidence (1/99, 1/100 and 1/101 all print 0.01)
    assert AV.counted_solution(Decimal(1), "0.01", known=99, ambiguous=2) is None


def test_a_player_whose_average_counts_the_blank_game_proves_a_recorded_zero_for_the_team_match() -> None:
    games = [_g(K1, "ambiguous"), _g(("m3", "Alpha"), "value", 4), _g(("m4", "Alpha"), "value", 2)]
    assert AV.player_votes(games, "2.00") == {K1: "recorded"}  # 6 / 3
    assert AV.player_votes(games, "3.00") == {K1: "unrecorded"}  # 6 / 2


def test_partial_solutions_and_blank_averages_give_no_evidence() -> None:
    games = [_g(K1, "ambiguous"), _g(K2, "ambiguous"), _g(("m3", "Alpha"), "value", 6)]
    assert AV.player_votes(games, "3.00") == {}  # 6/2: one of two counted - which one is unknown
    assert AV.player_votes(games, "") == {}
    assert AV.player_votes([_g(("m3", "Alpha"), "value", 6)], "6.00") == {}  # nothing ambiguous


def test_excluded_games_never_enter_the_denominator_and_dntf_does() -> None:
    games = [_g(K1, "ambiguous"), _g(("m3", "Alpha"), "value", 6), _g(("m4", "Alpha"), "excluded"),
             _g(("m5", "Alpha"), "dntf")]  # fmt: skip
    assert AV.player_votes(games, "3.00") == {K1: "unrecorded"}  # 6 / (value + dntf)
    assert AV.player_votes(games, "2.00") == {K1: "recorded"}


def test_team_evidence_is_unanimous_or_a_conflict() -> None:
    ev = AV.team_evidence([(K1, "hitouts", "recorded"), (K1, "hitouts", "recorded"), (K2, "hitouts", "unrecorded"),
                           (K2, "hitouts", "recorded")])  # fmt: skip
    assert ev == {(K1, "hitouts"): "recorded", (K2, "hitouts"): "conflict"}


def test_brownlow_blanks_are_zero_only_when_the_printed_season_total_equals_the_printed_game_votes() -> None:
    assert AV.brownlow_season_complete([Decimal(3), Decimal(1)], "4") is True
    assert AV.brownlow_season_complete([Decimal(3)], "5") is False  # votes are missing from the game rows
    assert AV.brownlow_season_complete([], "") is True  # no votes at all, printed blank
    assert AV.brownlow_season_complete([Decimal(2)], None) is False  # no printed season total: no evidence
