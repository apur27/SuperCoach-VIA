"""Coverage-aware stat aggregates (PLAN 4.3, AUDIT C10, pending-decision 3; test A03)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from supercoach_via.domain import metrics
from supercoach_via.publish.view_models import StatValue

REPO = Path(__file__).resolve().parents[3]

opt_counts = st.lists(st.one_of(st.none(), st.integers(min_value=0, max_value=60)), max_size=40)


class TestAggregate:
    def test_blank_counts_stay_null_and_do_not_enter_the_denominator(self) -> None:
        agg = metrics.aggregate("disposals", [20, None, 30, None], career_games=4)
        assert agg.total == 50
        assert agg.observed_games == 2
        assert agg.mean == 25.0  # observed denominator, not 12.5
        assert agg.coverage == 0.5
        assert agg.n_of_m() == "2 of 4"

    def test_sum_with_no_observed_games_is_null_not_zero(self) -> None:
        agg = metrics.aggregate("tackles", [None, None, None], career_games=3)
        assert agg.total is None
        assert agg.mean is None
        assert agg.observed_games == 0
        assert agg.coverage == 0.0

    def test_observed_zero_is_a_real_zero(self) -> None:
        agg = metrics.aggregate("goals", [0, 0], career_games=2)
        assert agg.total == 0
        assert agg.mean == 0.0

    def test_empty_scope_has_unknown_coverage(self) -> None:
        agg = metrics.aggregate("goals", [], career_games=0)
        assert agg.total is None and agg.mean is None and agg.coverage is None

    def test_career_games_defaults_to_row_count_and_counter_can_exceed_rows(self) -> None:
        assert metrics.aggregate("goals", [1, 2]).career_games == 2
        agg = metrics.aggregate("goals", [1, 2], career_games=metrics.canonical_games(2, 5))
        assert agg.career_games == 5
        assert agg.coverage == pytest.approx(0.4)

    def test_eligible_games_counts_rows_inside_the_recording_era(self) -> None:
        agg = metrics.aggregate(
            "tackles", [None, None, 3, 4], seasons=[1985, 1986, 1987, 1988], recorded_from=1987
        )
        assert agg.eligible_games == 2
        assert agg.observed_games == 2
        assert agg.career_games == 4
        assert agg.eligible_coverage == 1.0

    def test_seasons_length_must_match(self) -> None:
        with pytest.raises(ValueError):
            metrics.aggregate("x", [1, 2], seasons=[2000], recorded_from=1990)

    def test_nan_is_treated_as_missing(self) -> None:
        agg = metrics.aggregate("x", [float("nan"), 2.0])
        assert agg.observed_games == 1 and agg.total == 2.0

    def test_to_stat_value_public_model(self) -> None:
        sv = metrics.aggregate("disposals", [10, None], career_games=2).to_stat_value()
        assert isinstance(sv, StatValue)
        assert sv.total == 10 and sv.mean == 10 and sv.observed_games == 1
        assert sv.coverage == 0.5
        assert sv.eligible_games == 2

    @given(opt_counts)
    def test_property_null_aggregate_rules(self, values: list[int | None]) -> None:
        agg = metrics.aggregate("s", values)
        observed = [v for v in values if v is not None]
        assert agg.observed_games == len(observed)
        assert agg.career_games == len(values)
        if not observed:
            assert agg.total is None and agg.mean is None
        else:
            assert agg.total == sum(observed)
            assert agg.mean is not None
            assert math.isclose(agg.mean, sum(observed) / len(observed))
        if values:
            assert agg.coverage is not None and 0.0 <= agg.coverage <= 1.0

    @given(opt_counts, st.integers(min_value=0, max_value=10))
    def test_property_counter_never_reduces_denominator(self, values: list[int | None], extra: int) -> None:
        cg = metrics.canonical_games(len(values), len(values) + extra)
        agg = metrics.aggregate("s", values, career_games=cg)
        assert agg.career_games >= len(values)
        if agg.coverage is not None:
            assert agg.coverage <= 1.0

    def test_career_games_below_rows_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            metrics.aggregate("s", [1, 2, 3], career_games=2)


class TestLegacyZeroFill:
    def test_named_legacy_imputation_differs_from_coverage_rule(self) -> None:
        values = [None, None]
        assert metrics.legacy_zero_fill_sum(values) == 0.0
        assert metrics.aggregate("s", values).total is None


class TestGamesCounter:
    @pytest.mark.parametrize(
        ("token", "expected"),
        [("12", 12), ("12↑", 12), ("↓7", 7), ("7↓", 7), ("", None), (None, None), ("abc", None), ("3.0", 3)],
    )
    def test_parse_counter_token(self, token: str | None, expected: int | None) -> None:
        assert metrics.parse_counter_token(token) == expected

    def test_canonical_games_is_max_of_rows_and_counter(self) -> None:
        assert metrics.canonical_games(432, 435) == 435
        assert metrics.canonical_games(173, 167) == 173
        assert metrics.canonical_games(10, None) == 10


class TestCoverageEras:
    def test_loads_repo_config_under_canonical_names(self) -> None:
        eras = metrics.CoverageEras.load(REPO / "config" / "stat_coverage_eras.yaml")
        assert eras.recorded_from("tackles") == 1987
        assert eras.recorded_from("hitouts") == 1966  # legacy hit_outs renamed
        assert eras.recorded_from("goal_assists") == 2003
        assert eras.recorded_from("frees_for") == 1965
        assert eras.recorded_from("goals") == 1897
        assert eras.recorded_from("unknown_stat") is None

    def test_is_recorded(self) -> None:
        eras = metrics.CoverageEras({"tackles": 1987})
        assert eras.is_recorded("tackles", 1987)
        assert not eras.is_recorded("tackles", 1986)
        assert eras.is_recorded("goals", 1900)  # unknown era -> no boundary asserted


class TestSql:
    def test_sql_aggregate_columns_null_semantics(self) -> None:
        import duckdb

        con = duckdb.connect()
        con.execute(
            "CREATE TABLE g AS SELECT * FROM (VALUES ('a',1985,NULL),('a',1987,3),('b',1985,NULL)) t(p, season, tackles)"
        )
        cols = metrics.sql_aggregate_columns(["tackles"], metrics.CoverageEras({"tackles": 1987}))
        rows = con.execute(f"SELECT p, {cols} FROM g GROUP BY p ORDER BY p").fetchall()
        # (player, total, observed, eligible)
        assert rows == [("a", 3, 1, 1), ("b", None, 0, 0)]
