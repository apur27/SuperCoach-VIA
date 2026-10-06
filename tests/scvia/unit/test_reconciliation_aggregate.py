"""Season / stint / career aggregates and the printed-average model (T09, T32, T33; DESIGN section 8)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from supercoach_via.reconciliation import aggregate as A
from supercoach_via.reconciliation import cells as C

D = Decimal
S = C.St


def cell(state: C.St, v: int | None = None, rule: str = "r") -> C.Cell:
    return C.Cell(state, None if v is None else D(v), rule)


def acc(*cells: C.Cell, double: bool = True) -> A.StatAcc:
    a = A.StatAcc()
    for c in cells:
        a.add(c, double_sourced=double)
    return a


def test_stat_accumulator_sums_recorded_values_and_counts_every_state() -> None:
    a = acc(cell(S.VALUE, 7), cell(S.ZERO, 0), cell(S.NOT_RECORDED), cell(S.NA), cell(S.DNTF), cell(S.UNRESOLVED))
    assert a.total == D(7) and (a.n_value, a.n_zero, a.n_unrec, a.n_na, a.n_dntf, a.n_bad) == (1, 1, 1, 1, 1, 1)
    assert a.games == 6 and a.recorded == 2 and not a.single_sourced


def test_contribution_kinds() -> None:
    assert A.contribution(acc(cell(S.VALUE, 3), cell(S.ZERO, 0)), None).kind == "per_game"
    assert A.contribution(acc(cell(S.NOT_RECORDED), cell(S.NOT_RECORDED)), None).kind == "unavailable"
    assert A.contribution(acc(cell(S.VALUE, 3), cell(S.UNRESOLVED)), None).kind == "unresolved"
    assert A.contribution(acc(cell(S.VALUE, 3), cell(S.NOT_RECORDED)), None).kind == "per_game"  # recorded part counts
    assert A.contribution(acc(cell(S.NA), cell(S.NA)), None).kind == "not_applicable"
    assert (
        A.contribution(acc(cell(S.DNTF)), None).kind == "unavailable"
    )  # a sub who never took the field: no measurement


def test_summary_only_season_value_is_a_reference_not_a_conflict() -> None:  # S-01
    c = A.contribution(acc(cell(S.NOT_RECORDED), cell(S.NOT_RECORDED)), D(13), summary_only=True)
    assert (c.kind, c.value) == ("summary_only", D(13))


def test_composed_reference_sums_per_game_and_summary_only_seasons_and_counts_unrecorded() -> None:  # N-04 / C-1
    parts = [
        A.contribution(acc(cell(S.VALUE, 5), cell(S.VALUE, 2)), None),  # per game: 7
        A.contribution(acc(cell(S.NOT_RECORDED)), D(13), summary_only=True),  # season value: 13
        A.contribution(acc(cell(S.NOT_RECORDED)), None),  # contributes nothing
    ]
    c = A.compose(parts)
    assert (c.kind, c.value, c.recorded_seasons, c.unrecorded_seasons) == ("composed", D(20), 2, 1)


def test_all_unrecorded_contributions_make_the_aggregate_source_unavailable() -> None:  # N-02
    c = A.compose([A.contribution(acc(cell(S.NOT_RECORDED)), None), A.contribution(acc(cell(S.NOT_RECORDED)), None)])
    assert c.kind == "unavailable"


def test_one_unresolved_contribution_leaves_the_whole_aggregate_unresolved() -> None:
    c = A.compose(
        [A.contribution(acc(cell(S.VALUE, 1)), None), A.contribution(acc(cell(S.VALUE, 1), cell(S.UNRESOLVED)), None)]
    )
    assert c.kind == "unresolved"


def test_local_aggregate_outcomes_equal_mismatch_missing_summary_unavailable() -> None:  # T33
    comp = A.compose([A.contribution(acc(cell(S.VALUE, 5), cell(S.VALUE, 2)), None)])
    assert A.judge(comp, A.LocalAcc(total=D(7), n_values=2, n_null=0)).outcome == "equal"
    assert A.judge(comp, A.LocalAcc(total=D(6), n_values=2, n_null=0)).outcome == "mismatch"
    summ = A.compose([A.contribution(acc(cell(S.NOT_RECORDED)), D(13), summary_only=True)])
    j = A.judge(summ, A.LocalAcc(total=D(0), n_values=0, n_null=1))
    assert j.outcome == "local_missing_summary" and "13" in j.detail
    # precedence: counted once (both facts in the detail) even though the totals also differ
    assert A.judge(summ, A.LocalAcc(total=D(2), n_values=1, n_null=0)).outcome == "local_missing_summary"
    unav = A.compose([A.contribution(acc(cell(S.NOT_RECORDED)), None)])
    assert A.judge(unav, A.LocalAcc(total=D(0), n_values=0, n_null=3)).outcome == "source_unavailable"
    assert (
        A.judge(A.compose([A.contribution(acc(cell(S.NA)), None)]), A.LocalAcc(D(0), 0, 0)).outcome == "not_applicable"
    )
    assert (
        A.judge(A.compose([A.contribution(acc(cell(S.UNRESOLVED)), None)]), A.LocalAcc(D(0), 0, 0)).outcome
        == "unresolved"
    )


def test_missing_game_changes_the_full_scope_aggregate_not_just_the_common_games() -> None:  # T05 / design step 2
    comp = A.compose([A.contribution(acc(cell(S.VALUE, 5), cell(S.VALUE, 2), cell(S.VALUE, 4)), None)])
    # local lacks the third game: the full-scope sum is 7, the source composed 11
    assert A.judge(comp, A.LocalAcc(total=D(7), n_values=2, n_null=0)).outcome == "mismatch"


@pytest.mark.parametrize(
    ("printed", "composed", "expect"),
    [
        ("13", D(13), "equal"),
        ("", D(0), "equal"),  # a printed blank means zero or unavailable
        ("", None, "equal"),
        ("14", D(13), "mismatch"),
        ("", D(4), "mismatch"),
        ("x", D(4), "malformed"),
    ],
)
def test_printed_total_is_a_derived_figure_checked_against_the_composed_reference(
    printed: str, composed: D | None, expect: str
) -> None:
    kind = "unavailable" if composed is None else "composed"
    comp = A.Composed(kind=kind, value=composed or D(0), recorded_seasons=1, unrecorded_seasons=0, single_sourced=False)
    assert A.check_printed(printed, comp) == expect


def test_derived_mismatch_with_single_sourced_cells_is_unresolved_not_inconsistent() -> None:  # S-07
    comp = A.Composed(kind="composed", value=D(13), recorded_seasons=1, unrecorded_seasons=0, single_sourced=True)
    assert A.check_printed("14", comp) == "unresolved"


@pytest.mark.parametrize(
    ("total", "den", "printed", "ok"),
    [
        (D(28125), D(1000), "28.13", True),  # exact tie 28.125 printed upward
        (D(28125), D(1000), "28.12", True),  # ...and a tie may equally be printed downward (real pages do both)
        (D(5), D(8), "0.63", True),  # 0.625
        (D(5), D(8), "0.62", True),
        (D(5), D(8), "0.64", False),  # outside the rounding interval: always a miss
        (D(137), D(40), "3.42", True),  # 3.425 printed 3.42 on a real profile
        (D(12560), D(1000), "12.56", True),
        (D(7), D(3), "2.33", True),
        (D(7), D(3), "2.34", False),
    ],
)
def test_display_rounding_accepts_both_neighbours_of_an_exact_tie_and_nothing_further(
    total: D, den: D, printed: str, ok: bool
) -> None:  # T32
    assert A.average_matches(total, den, printed) is ok


def test_season_average_uses_recorded_games_and_brownlow_uses_home_and_away_games() -> None:  # T32
    a = acc(
        *[cell(S.VALUE, 4)] * 15, cell(S.NOT_RECORDED), cell(S.DNTF)
    )  # 18 games, one unrecorded (real 1975-style gap)
    assert A.season_denominator("kicks", a, home_and_away=17) == 16
    assert A.season_denominator("brownlow_votes", a, home_and_away=17) == 17


def test_career_average_denominator_excludes_unrecorded_games_but_includes_dntf() -> None:  # T32
    a = acc(cell(S.VALUE, 4), cell(S.ZERO, 0), cell(S.DNTF), cell(S.NOT_RECORDED), cell(S.NA))
    assert A.career_denominator("kicks", a, home_and_away_in_award_seasons=0) == 3  # value + zero + dntf


def test_brownlow_career_denominator_counts_every_home_and_away_game_of_award_seasons() -> None:  # T32 / N-03
    a = acc(cell(S.NOT_RECORDED), cell(S.NOT_RECORDED))  # pre-1984 per-game cells are not recorded
    assert A.career_denominator("brownlow_votes", a, home_and_away_in_award_seasons=44) == 44


def test_a_model_miss_lists_every_candidate_denominator_and_is_never_a_local_verdict() -> None:  # T32 / S-03
    a = acc(cell(S.VALUE, 10), cell(S.VALUE, 10), cell(S.NOT_RECORDED))
    res = A.check_career_average("kicks", printed="9.00", acc=a, games=3, home_and_away_in_award_seasons=3)
    assert res.outcome == "miss"
    cands = dict(res.candidates)
    assert cands["games"] is False and cands["recorded_games"] is False and "model" in cands
    assert (
        A.check_career_average("kicks", printed="10.00", acc=a, games=3, home_and_away_in_award_seasons=3).outcome
        == "ok"
    )


def test_games_average_divides_by_distinct_seasons_and_win_percentage_counts_draws_as_half() -> None:  # T32
    assert A.games_average_ok(442, 21, "21.05")
    assert not A.games_average_ok(442, 21, "21.04")
    assert A.win_pct_ok(266, 6, 442, "60.86%")
    assert not A.win_pct_ok(266, 6, 442, "60.85%")


def test_two_club_season_counts_once_in_the_career_and_season_levels() -> None:  # T09
    stint1 = A.contribution(acc(cell(S.VALUE, 10), cell(S.VALUE, 12)), None)
    stint2 = A.contribution(acc(cell(S.VALUE, 9)), None)
    season = A.compose([stint1, stint2])
    career = A.compose([season_as_contribution(season), A.contribution(acc(cell(S.VALUE, 4)), None)])
    assert season.value == D(31) and career.value == D(35)  # not 31 + 31 + 4


def season_as_contribution(c: A.Composed) -> A.Contribution:
    return A.Contribution(kind="per_game", value=c.value, single_sourced=c.single_sourced)
