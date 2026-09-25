"""Stage parsing, chronological stage order, replays and match IDs (D09-D12)."""

from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given
from hypothesis import strategies as st

from supercoach_via.domain import season
from supercoach_via.domain.schemas import DateQuality, StageType, is_safe_id


class TestParseStage:
    @pytest.mark.parametrize(
        ("token", "stage_id", "stage_type", "round_number"),
        [
            ("1", "r01", StageType.REGULAR, 1),
            ("23", "r23", StageType.REGULAR, 23),
            ("Qualifying Final", "qf", StageType.FINAL, None),
            ("QF", "qf", StageType.FINAL, None),
            ("Elimination Final", "ef", StageType.FINAL, None),
            ("EF", "ef", StageType.FINAL, None),
            ("Semi Final", "sf", StageType.FINAL, None),
            ("SF", "sf", StageType.FINAL, None),
            ("Preliminary Final", "pf", StageType.FINAL, None),
            ("PF", "pf", StageType.FINAL, None),
            ("Grand Final", "gf", StageType.FINAL, None),
            ("GF", "gf", StageType.FINAL, None),
            ("Wildcard Final", "wf", StageType.FINAL, None),
            ("WF", "wf", StageType.FINAL, None),
            ("Opening Round", "r00", StageType.REGULAR, 0),
            ("0", "r00", StageType.REGULAR, 0),
        ],
    )
    def test_known_tokens(self, token: str, stage_id: str, stage_type: StageType, round_number: int | None) -> None:
        s = season.parse_stage(token)
        assert s.stage_id == stage_id
        assert s.stage_type is stage_type
        assert s.round_number == round_number
        assert s.recognized
        assert s.label == token

    def test_wildcard_is_not_a_normal_round(self) -> None:
        # regression: the legacy parser dropped WF and fabricated a date for it
        s = season.parse_stage("WF")
        assert s.round_number is None and s.stage_type is StageType.FINAL

    @pytest.mark.parametrize("token", ["Challenge Final", "Round X", "", "  ", "-3"])
    def test_unrecognized_is_preserved_and_flagged(self, token: str) -> None:
        s = season.parse_stage(token)
        assert not s.recognized
        assert s.stage_type is StageType.OTHER
        assert s.round_number is None
        assert s.label == token
        assert is_safe_id(s.stage_id) and s.stage_id.startswith("x")

    def test_unrecognized_ids_differ_by_token(self) -> None:
        assert season.parse_stage("Foo").stage_id != season.parse_stage("Bar").stage_id

    def test_player_and_match_tokens_agree(self) -> None:
        for short, long_ in [
            ("QF", "Qualifying Final"),
            ("GF", "Grand Final"),
            ("WF", "Wildcard Final"),
        ]:
            assert season.parse_stage(short).stage_id == season.parse_stage(long_).stage_id


class TestStageOrder:
    def test_opening_round_and_postponed_round_ordered_by_chronology(self) -> None:
        # D10/D11: round 1 has a postponed match in August; order uses the FIRST date.
        dates = {
            "r00": [date(2026, 3, 5)],
            "r01": [date(2026, 3, 12), date(2026, 8, 27)],
            "r02": [date(2026, 3, 19)],
            "qf": [date(2026, 9, 4)],
            "ef": [date(2026, 9, 5)],
            "wf": [date(2026, 8, 28)],
            "sf": [date(2026, 9, 11)],
            "pf": [date(2026, 9, 18)],
            "gf": [date(2026, 9, 26)],
        }
        order = season.stage_orders(dates)
        ranked = sorted(order, key=order.__getitem__)
        assert ranked == ["r00", "r01", "r02", "wf", "qf", "ef", "sf", "pf", "gf"]
        assert sorted(order.values()) == list(range(1, 10))

    def test_chronology_beats_round_number(self) -> None:
        # a later-numbered round played earlier (rescheduled) comes first
        order = season.stage_orders({"r05": [date(2020, 6, 1)], "r02": [date(2020, 6, 10)]})
        assert order["r05"] < order["r02"]

    def test_undated_stages_fall_back_to_nominal_rank_after_dated(self) -> None:
        order = season.stage_orders({"gf": [None], "r01": [date(1900, 5, 1)], "r02": [None]})
        assert order == {"r01": 1, "r02": 2, "gf": 3}


class TestReplaysAndMatchIds:
    def test_drawn_final_and_replay_get_distinct_occurrences(self) -> None:
        # D09: 1977 Grand Final draw and replay a week later are two rows
        rows = [
            ("gf", "collingwood", "north_melbourne", date(1977, 10, 1), 2),
            ("gf", "north_melbourne", "collingwood", date(1977, 9, 24), 1),
        ]
        occ = season.replay_occurrences([(s, a, b, d) for s, a, b, d, _ in rows], [r for *_, r in rows])
        assert occ == [1, 0]

    def test_same_day_matches_are_distinct(self) -> None:
        # D12: two matches on the same day in the same round are different identities
        a = season.make_match_id(2026, "r01", "carlton", "sydney", 0)
        b = season.make_match_id(2026, "r01", "geelong", "gold_coast", 0)
        assert a != b

    def test_match_id_independent_of_date_and_home_away_order(self) -> None:
        # D12: a corrected fixture date must not change the match id
        a = season.make_match_id(2025, "r01", "gold_coast", "essendon", 0)
        b = season.make_match_id(2025, "r01", "essendon", "gold_coast", 0)
        assert a == b == "m:2025:r01:essendon:gold_coast:0"
        assert is_safe_id(a)

    def test_replay_changes_match_id(self) -> None:
        assert season.make_match_id(2010, "gf", "a", "b", 0) != season.make_match_id(2010, "gf", "a", "b", 1)

    @given(
        st.lists(
            st.dates(min_value=date(1897, 1, 1), max_value=date(2030, 12, 31)),
            min_size=1,
            max_size=6,
        )
    )
    def test_occurrences_are_a_permutation(self, dates: list[date]) -> None:
        keys = [("sf", "x", "y", d) for d in dates]
        occ = season.replay_occurrences(keys, list(range(len(keys))))
        assert sorted(occ) == list(range(len(keys)))


class TestLegacyDates:
    def test_march_weeks_reconstruction_detected(self) -> None:
        assert season.is_legacy_synthetic_date(date(2006, 5, 3), 2006)  # 03-01 + 9 weeks
        assert season.is_legacy_synthetic_date(date(2024, 3, 1), 2024)  # finals placeholder
        assert season.is_legacy_synthetic_date(date(2022, 8, 9), 2022)  # 03-01 + 23 weeks
        assert not season.is_legacy_synthetic_date(date(2006, 6, 3), 2006)
        assert not season.is_legacy_synthetic_date(date(2006, 2, 22), 2006)

    def test_date_quality_rules(self) -> None:
        match = date(2006, 6, 3)
        assert season.row_date_quality(date(2006, 6, 3), match, 2006) is DateQuality.FIXTURE_VERIFIED
        assert season.row_date_quality(date(2006, 5, 3), match, 2006) is DateQuality.INFERRED
        assert season.row_date_quality(date(2006, 6, 4), match, 2006) is DateQuality.SOURCE
        assert season.row_date_quality(date(2006, 6, 4), None, 2006) is DateQuality.SOURCE
        assert season.row_date_quality(None, match, 2006) is DateQuality.UNKNOWN
        # a synthetic-looking date that equals the real fixture is verified, not inferred
        assert season.row_date_quality(date(2006, 3, 29), date(2006, 3, 29), 2006) is DateQuality.FIXTURE_VERIFIED
