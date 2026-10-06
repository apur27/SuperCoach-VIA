"""Source cell semantics (T10, T11, T13, T14, T27, T34; DESIGN section 8)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from supercoach_via.reconciliation import cells as C
from supercoach_via.reconciliation.schema import STAT_FIELDS

IDX = {f: i for i, f in enumerate(STAT_FIELDS)}
ALL = frozenset(STAT_FIELDS)


def row(**kw: str | None) -> tuple[str | None, ...]:
    """23 raw cells, blank by default."""
    out: list[str | None] = [""] * len(STAT_FIELDS)
    for k, v in kw.items():
        out[IDX[k]] = v
    return tuple(out)


def totals(**kw: str | None) -> tuple[str | None, ...]:
    return row(**kw)


def ctx(**kw: object) -> C.AppearanceCtx:
    base: dict[str, object] = {
        "season": 2026,
        "is_final": False,
        "match_usable": True,
        "player_in_lineup": True,
        "column_present": ALL,
        "team_totals": totals(kicks="10", disposals="20", goals="2", behinds="1"),
        "team_goals": 2,
        "team_behinds": 1,
        "available": ALL,
        "exception_fields": frozenset(),
        "bad_fields": frozenset(),
        "team_pct_recorded": True,
    }
    base.update(kw)
    return C.AppearanceCtx(**base)  # type: ignore[arg-type]


def states(prof: tuple[str | None, ...], match: tuple[str | None, ...] | None, c: C.AppearanceCtx) -> dict[str, C.Cell]:
    return dict(zip(STAT_FIELDS, C.classify_appearance(prof, match, c), strict=True))


def test_printed_value_agreeing_on_both_pages_is_recorded_value() -> None:
    cs = states(
        row(kicks="7", disposals="12", time_on_ground_pct="66"),
        row(kicks="7", disposals="12", time_on_ground_pct="66"),
        ctx(),
    )
    assert (cs["kicks"].state, cs["kicks"].value) == (C.St.VALUE, Decimal(7))
    assert cs["time_on_ground_pct"].value == Decimal(66)


def test_blank_cell_with_nonblank_team_total_is_recorded_zero_not_unknown() -> None:  # T10
    cs = states(row(kicks="7"), row(kicks="7"), ctx())
    assert (cs["goals"].state, cs["goals"].value, cs["goals"].rule) == (C.St.ZERO, Decimal(0), "R-TOTAL-NONBLANK")


def test_blank_team_total_in_an_unrecorded_era_is_not_recorded_not_zero() -> None:  # T10
    c = ctx(
        season=1975,
        team_totals=totals(goals="19"),
        available=frozenset({"kicks", "goals"}),
        team_goals=19,
        team_pct_recorded=False,
    )
    cs = states(row(goals="1"), row(goals="1"), c)
    assert cs["tackles"].state is C.St.NOT_RECORDED and cs["tackles"].rule == "R-NOTES-MATRIX"
    assert cs["goals"].state is C.St.VALUE


def test_pre_1965_blank_totals_are_not_recorded_by_match_structure_only() -> None:  # T10
    c = ctx(
        season=1950,
        team_totals=totals(goals="9"),
        available=None,
        team_goals=9,
        team_behinds=None,
        team_pct_recorded=False,
    )
    cs = states(row(goals="3"), row(goals="3"), c)
    assert cs["kicks"].state is C.St.NOT_RECORDED and cs["kicks"].rule == "R-PRE1965-STRUCTURE"
    assert cs["behinds"].state is C.St.NOT_RECORDED


def test_documented_missing_column_in_a_matrix_year_uses_the_notes_exception() -> None:  # T10
    c = ctx(season=1975, team_totals=totals(goals="9"), available=frozenset({"kicks", "hitouts", "goals"}),
            exception_fields=frozenset({"hitouts"}), team_pct_recorded=False)  # fmt: skip
    cs = states(row(), row(), c)
    assert cs["hitouts"].state is C.St.NOT_RECORDED and cs["hitouts"].rule == "R-NOTES-EXCEPTION"
    assert (
        cs["kicks"].state is C.St.UNRESOLVED
    )  # recorded that year, no exception, blank team total: all-zero column unproven


def test_all_zero_team_column_stays_unresolved_until_a_fixture_proves_its_encoding() -> None:  # T10
    c = ctx(team_totals=totals(kicks="10"))  # tackles total blank in a fully recorded era
    cs = states(row(kicks="10"), row(kicks="10"), c)
    assert cs["tackles"].state is C.St.UNRESOLVED and cs["tackles"].rule == "R-ALL-ZERO-UNPROVEN"


def test_blank_goals_and_behinds_are_zero_when_the_header_score_proves_zero() -> None:  # T10
    c = ctx(team_totals=totals(kicks="10"), team_goals=0, team_behinds=0)
    cs = states(row(kicks="10"), row(kicks="10"), c)
    assert cs["goals"].state is C.St.ZERO and cs["goals"].rule == "R-HEADER-ZERO"
    assert cs["behinds"].state is C.St.ZERO


def test_positive_teammate_values_never_turn_a_blank_into_zero_by_themselves() -> None:  # T10
    c = ctx(team_totals=totals(kicks="10", tackles=""), available=ALL)
    cs = states(row(kicks="10"), row(kicks="10"), c)
    assert cs["tackles"].state is C.St.UNRESOLVED


def test_time_on_ground_blank_is_never_zero_filled() -> None:  # T10
    cs = states(row(kicks="5"), row(kicks="5"), ctx())
    assert cs["time_on_ground_pct"].state is C.St.UNRESOLVED
    old = states(
        row(kicks="5"),
        row(kicks="5"),
        ctx(season=1990, available=frozenset(STAT_FIELDS) - {"time_on_ground_pct"}, team_pct_recorded=False),
    )
    assert old["time_on_ground_pct"].state is C.St.NOT_RECORDED


def test_finals_brownlow_votes_are_not_applicable() -> None:  # T11
    cs = states(row(kicks="5"), row(kicks="5"), ctx(is_final=True))
    assert cs["brownlow_votes"].state is C.St.NA and cs["brownlow_votes"].rule == "R-BR-FINALS"


def test_column_absent_from_the_match_table_is_not_recorded() -> None:  # T11
    c = ctx(column_present=ALL - {"bounces"})
    cs = states(
        row(kicks="5"),
        row(kicks="5") if False else tuple(None if i == IDX["bounces"] else v for i, v in enumerate(row(kicks="5"))),
        c,
    )
    assert cs["bounces"].state is C.St.NOT_RECORDED and cs["bounces"].rule == "R-COLUMN-ABSENT"


def test_credited_unused_substitute_has_not_applicable_dntf_cells() -> None:  # T34
    c = ctx(team_pct_recorded=True)
    cs = states(row(), row(), c)
    assert {x.state for x in cs.values()} == {C.St.DNTF}
    assert C.is_dntf(row(), row(), c)


def test_dntf_requires_all_four_conditions() -> None:  # T34/S-02
    assert not C.is_dntf(row(), row(), ctx(team_pct_recorded=False))  # team %P not recorded
    assert not C.is_dntf(row(), row(), ctx(player_in_lineup=False))  # not listed on the match page
    assert not C.is_dntf(row(), None, ctx())  # no match page row to confirm
    assert not C.is_dntf(row(kicks="1"), row(kicks="1"), ctx())  # not all blank


def test_pre_2003_all_blank_player_is_not_dntf_even_with_goals_only_recorded() -> None:  # T34
    c = ctx(
        season=1975, team_pct_recorded=False, available=frozenset({"kicks", "goals"}), team_totals=totals(goals="9")
    )
    cs = states(row(), row(), c)
    assert cs["goals"].state is C.St.ZERO and not C.is_dntf(row(), row(), c)


def test_profile_and_match_cells_that_disagree_are_a_source_conflict() -> None:  # T14
    cs = states(row(kicks="7"), row(kicks="8"), ctx())
    assert cs["kicks"].state is C.St.CONFLICT and "7" in cs["kicks"].detail and "8" in cs["kicks"].detail


def test_cell_printed_on_one_page_and_blank_on_the_other_is_a_source_conflict() -> None:  # T14 / C-2
    cs = states(row(brownlow_votes="3"), row(), ctx())
    assert cs["brownlow_votes"].state is C.St.CONFLICT
    cs2 = states(row(), row(brownlow_votes="3"), ctx())
    assert cs2["brownlow_votes"].state is C.St.CONFLICT


def test_a_versioned_rule_can_exempt_a_one_sided_cell_only_with_an_id() -> None:  # C-2
    rule = C.OneSidedRule(
        rule_id="R-TEST", field="brownlow_votes", first_season=1931, last_season=1934, printed_on="profile"
    )
    c = ctx(season=1933, one_sided_rules=(rule,), team_totals=totals(brownlow_votes="6"))
    cs = states(row(brownlow_votes="3"), row(), c)
    assert cs["brownlow_votes"].state is C.St.VALUE and cs["brownlow_votes"].rule == "R-TEST"


def test_single_sourced_cell_when_no_match_page_is_used_and_blank_is_unresolved() -> None:
    c = ctx(match_usable=False, player_in_lineup=False, team_totals=None)
    cs = states(row(kicks="5"), None, c)
    assert cs["kicks"].state is C.St.VALUE and cs["kicks"].rule == "R-PROFILE-ONLY"
    assert cs["goals"].state is C.St.UNRESOLVED and cs["goals"].rule == "R-NO-MATCH-PAGE"


@pytest.mark.parametrize("raw", ["x", "1,000", "1e3", "-1", "NaN", "Infinity", "True", "١٢", "1 2", "+5", "0x10", "07"])
def test_integer_statistic_parser_is_strict(raw: str) -> None:  # T13
    cs = states(row(kicks=raw), row(kicks=raw), ctx())
    assert cs["kicks"].state is C.St.MALFORMED


@pytest.mark.parametrize(("raw", "value"), [("0", 0), ("12", 12), ("100", 100)])
def test_valid_integers_parse(raw: str, value: int) -> None:  # T13
    cs = states(row(kicks=raw), row(kicks=raw), ctx())
    assert cs["kicks"].value == Decimal(value) and cs["kicks"].state is C.St.VALUE


@pytest.mark.parametrize(("raw", "value"), [("66", "66"), ("66.5", "66.5"), ("66%", "66"), ("100", "100"), ("0", "0")])
def test_time_on_ground_is_an_exact_decimal(raw: str, value: str) -> None:  # T13
    cs = states(row(time_on_ground_pct=raw), row(time_on_ground_pct=raw), ctx())
    assert cs["time_on_ground_pct"].value == Decimal(value) and cs["time_on_ground_pct"].state is C.St.VALUE


@pytest.mark.parametrize("raw", ["66,5", "6 6", "1e2", "inf", "-3", "66.", ".5", "%"])
def test_time_on_ground_rejects_comma_and_float_noise(raw: str) -> None:  # T13
    cs = states(row(time_on_ground_pct=raw), row(time_on_ground_pct=raw), ctx())
    assert cs["time_on_ground_pct"].state is C.St.MALFORMED


def test_duplicate_label_field_is_malformed_even_when_the_values_look_fine() -> None:  # T12/S-10
    cs = states(row(kicks="7"), row(kicks="7"), ctx(bad_fields=frozenset({"kicks"})))
    assert cs["kicks"].state is C.St.MALFORMED and cs["kicks"].rule == "R-LABEL-DUPLICATE"


def test_malformed_beats_conflict_and_every_cell_gets_exactly_one_state() -> None:
    cs = C.classify_appearance(row(kicks="7", goals="x"), row(kicks="8", goals="x"), ctx())
    assert len(cs) == 23 and all(isinstance(c.state, C.St) for c in cs)


def test_summary_only_brownlow_rule_requires_all_three_conditions() -> None:  # S-01
    games = [row(), row(), row()]
    team_totals_blank = [totals(), totals(), totals()]
    assert C.summary_only(season_value="13", game_cells=games, team_totals=team_totals_blank, field="brownlow_votes")
    assert not C.summary_only(season_value="", game_cells=games, team_totals=team_totals_blank, field="brownlow_votes")
    assert not C.summary_only(season_value="13", game_cells=[row(brownlow_votes="1"), row(), row()],
                              team_totals=team_totals_blank, field="brownlow_votes")  # fmt: skip
    assert not C.summary_only(season_value="13", game_cells=games,
                              team_totals=[totals(), totals(brownlow_votes="6"), totals()], field="brownlow_votes")  # fmt: skip


def test_local_value_comparison_outcomes() -> None:
    A = C.Cell
    S = C.St
    assert C.compare_local(A(S.VALUE, Decimal(7), "r"), 7).outcome == "equal"
    assert C.compare_local(A(S.VALUE, Decimal(7), "r"), 8).outcome == "mismatch"
    assert C.compare_local(A(S.VALUE, Decimal(7), "r"), None).outcome == "mismatch"
    assert C.compare_local(A(S.ZERO, Decimal(0), "r"), 0).outcome == "equal"
    assert C.compare_local(A(S.ZERO, Decimal(0), "r"), 3).outcome == "mismatch"
    assert C.compare_local(A(S.ZERO, Decimal(0), "r"), None).outcome == "mismatch_local_null"
    assert C.compare_local(A(S.NOT_RECORDED, None, "r"), None).outcome == "source_unavailable"
    assert C.compare_local(A(S.NOT_RECORDED, None, "r"), 0).outcome == "unsupported_local_numeric"
    assert C.compare_local(A(S.NA, None, "r"), None).outcome == "not_applicable"
    assert C.compare_local(A(S.NA, None, "r"), 4).outcome == "mismatch"
    assert C.compare_local(A(S.DNTF, None, "r"), None).outcome == "not_applicable_dntf"
    assert C.compare_local(A(S.DNTF, None, "r"), 0).outcome == "not_applicable_dntf"
    assert C.compare_local(A(S.DNTF, None, "r"), 2).outcome == "mismatch"
    assert C.compare_local(A(S.UNRESOLVED, None, "r"), 5).outcome == "unresolved"
    assert C.compare_local(A(S.MALFORMED, None, "r"), 5).outcome == "unresolved"
    assert C.compare_local(A(S.CONFLICT, None, "r"), 5).outcome == "unresolved"


def test_percent_on_ground_compares_exact_decimals_not_floats() -> None:  # T13
    A = C.Cell
    S = C.St
    assert C.compare_local(A(S.VALUE, Decimal("66.3"), "r"), 66.3).outcome == "equal"
    assert C.compare_local(A(S.VALUE, Decimal("66"), "r"), 66.0).outcome == "equal"
    assert C.compare_local(A(S.VALUE, Decimal("66"), "r"), 66.0000001).outcome == "mismatch"  # no tolerance
    assert C.compare_local(A(S.VALUE, Decimal("66"), "r"), 67.0).outcome == "mismatch"


def test_bool_nan_and_infinity_are_never_valid_local_numbers() -> None:  # T13
    A = C.Cell
    S = C.St
    for bad in (True, False, float("nan"), float("inf")):
        assert C.compare_local(A(S.VALUE, Decimal(1), "r"), bad).outcome == "malformed_local"  # type: ignore[arg-type]


def test_without_a_usable_notes_page_post_1965_blanks_with_blank_totals_are_unresolved() -> None:
    c = ctx(season=1990, notes_usable=False, team_totals=totals(kicks="10"), available=None)
    cs = states(row(kicks="10"), row(kicks="10"), c)
    assert cs["tackles"].state is C.St.UNRESOLVED and cs["tackles"].rule == "R-NO-NOTES"
    old = ctx(season=1950, notes_usable=False, team_totals=totals(goals="9"), available=None, team_goals=9)
    assert states(row(goals="3"), row(goals="3"), old)["kicks"].state is C.St.NOT_RECORDED  # pre-1965 needs no notes


def br(mine: str, theirs: str | None, **kw: object) -> C.Cell:
    """The Brownlow state of a blank player cell in a season whose pages record votes (award total 6)."""
    base: dict[str, object] = {
        "season": 2010,
        "team_totals": totals(kicks="10", brownlow_votes=mine),
        "opp_totals": None if theirs is None else totals(brownlow_votes=theirs),
        "br_season_recorded": True,
        "br_award_total": 6,
    }
    base.update(kw)
    return states(row(kicks="10"), row(kicks="10"), ctx(**base))["brownlow_votes"]


def test_a_blank_brownlow_cell_is_a_proven_zero_only_when_the_two_team_totals_sum_to_the_award_total() -> None:  # A8
    for mine, theirs in (("6", ""), ("", "6"), ("2", "4"), ("3", "3")):
        c = br(mine, theirs)
        assert (c.state, c.value, c.rule) == (C.St.ZERO, Decimal(0), "R-BR-AWARD-SUM"), (mine, theirs)


def test_a_non_blank_team_total_alone_no_longer_proves_a_brownlow_zero() -> None:  # A8 / B2: 1931-34 partial matches
    for mine, theirs in (("3", ""), ("3", "1"), ("", ""), ("5", "5")):
        c = br(mine, theirs)
        assert (c.state, c.rule) == (C.St.UNRESOLVED, "R-BR-AWARD-SUM-MISMATCH"), (mine, theirs)
    assert "sum to" in br("3", "").detail


def test_the_award_total_is_twelve_in_1976_77_and_a_blank_beside_six_is_not_a_zero_then() -> None:  # A8
    assert br("6", "6", season=1976, br_award_total=12).state is C.St.ZERO
    assert br("6", "", season=1976, br_award_total=12).state is C.St.UNRESOLVED


def test_brownlow_without_an_award_total_or_opposition_totals_is_unresolved_not_zero() -> None:  # A8
    assert br("6", "", br_award_total=None).rule == "R-BR-NO-AWARD-TOTAL"
    assert br("6", None).rule == "R-NO-OPP-TOTALS"
    assert br("x", "").rule == "R-BR-TOTAL-FORMAT"


def test_seasons_with_no_medal_and_finals_are_not_applicable_for_brownlow_votes() -> None:  # A8
    assert br("", "", br_no_award=True).state is C.St.NA and br("", "", br_no_award=True).rule == "R-BR-NO-AWARD"
    assert br("6", "", is_final=True).rule == "R-BR-FINALS"


def test_a_season_whose_pages_record_no_votes_keeps_the_structural_state_for_brownlow() -> None:  # A8: 1935-83
    c = br("", "", season=1950, br_season_recorded=False, br_award_total=6, available=None)
    assert (c.state, c.rule) == (C.St.NOT_RECORDED, "R-PRE1965-STRUCTURE")


def test_a_notes_exception_takes_precedence_over_a_non_blank_team_total() -> None:  # A8 / B2: 1975 R11
    c = ctx(
        season=1975,
        team_totals=totals(goals="19", behinds="12"),
        team_behinds=12,
        exception_fields=frozenset({"behinds"}),
        available=frozenset({"kicks", "goals", "behinds"}),
    )
    cs = states(row(goals="3"), row(goals="3"), c)
    assert (cs["behinds"].state, cs["behinds"].rule) == (C.St.NOT_RECORDED, "R-NOTES-EXCEPTION")
    # the printed values stay values: the exception removes only the blank's claim to be a zero
    printed = states(row(goals="3", behinds="2"), row(goals="3", behinds="2"), c)
    assert (printed["behinds"].state, printed["behinds"].value) == (C.St.VALUE, Decimal(2))
    # without the exception the same non-blank total proves a zero
    plain = states(
        row(goals="3"),
        row(goals="3"),
        ctx(
            season=1975,
            team_totals=totals(goals="19", behinds="12"),
            available=frozenset({"kicks", "goals", "behinds"}),
        ),
    )
    assert plain["behinds"].state is C.St.ZERO and plain["behinds"].rule == "R-TOTAL-NONBLANK"
