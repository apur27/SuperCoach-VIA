"""Unit tests for age and experience feature engineering.

Covers the two S7 features:
  * compute_age_years          -> player_age_at_match
  * compute_career_games_to_date -> career_games_to_date

The load-bearing property under test is the temporal-cutoff / no-leakage
invariant for career_games_to_date: a row's count must depend ONLY on games
strictly before it, so inserting or removing a future game must never change
an earlier row's value.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.feature_engineering import (
    compute_age_years,
    compute_career_games_to_date,
    DAYS_PER_YEAR,
)


# --------------------------------------------------------------------------- #
# compute_age_years
# --------------------------------------------------------------------------- #
def test_age_exact_known_value_scalar_dob():
    """Age is (match_date - dob) / 365.25 years."""
    born = pd.Timestamp("1994-12-08")
    matches = pd.Series([pd.Timestamp("2020-03-29"), pd.Timestamp("2024-03-29")])
    age = compute_age_years(matches, born)
    expected0 = (matches.iloc[0] - born).days / DAYS_PER_YEAR
    expected1 = (matches.iloc[1] - born).days / DAYS_PER_YEAR
    assert age.iloc[0] == pytest.approx(expected0)
    assert age.iloc[1] == pytest.approx(expected1)
    # Sanity: a 2020 game for a 1994-born player is ~25.3 years.
    assert 25.0 < age.iloc[0] < 25.6


def test_age_monotonic_increasing_over_a_career():
    """Later matches for the same player yield strictly larger ages."""
    born = pd.Timestamp("1990-01-01")
    matches = pd.Series(pd.to_datetime(["2010-04-01", "2015-04-01", "2020-04-01"]))
    age = compute_age_years(matches, born)
    assert age.is_monotonic_increasing
    assert age.iloc[-1] > age.iloc[0]


def test_age_series_dob_aligns_by_index():
    """A per-row Series DOB aligns by index (different players, different DOBs)."""
    idx = [10, 11, 12]
    matches = pd.Series(pd.to_datetime(["2020-06-01", "2020-06-01", "2020-06-01"]), index=idx)
    born = pd.Series(
        pd.to_datetime(["2000-06-01", "1990-06-01", "1980-06-01"]), index=idx
    )
    age = compute_age_years(matches, born)
    assert age.loc[10] == pytest.approx(20 * 365 / DAYS_PER_YEAR, abs=0.05)
    assert age.loc[11] == pytest.approx(30 * 365 / DAYS_PER_YEAR, abs=0.05)
    assert age.loc[12] == pytest.approx(40 * 365 / DAYS_PER_YEAR, abs=0.05)


def test_age_missing_dob_is_nan():
    """A missing DOB propagates as NaN rather than crashing or fabricating an age."""
    matches = pd.Series(pd.to_datetime(["2020-03-29"]))
    age = compute_age_years(matches, pd.NaT)
    assert age.isna().all()


def test_age_missing_match_date_is_nan():
    matches = pd.Series([pd.NaT, pd.Timestamp("2020-03-29")])
    age = compute_age_years(matches, pd.Timestamp("1994-12-08"))
    assert pd.isna(age.iloc[0])
    assert not pd.isna(age.iloc[1])


# --------------------------------------------------------------------------- #
# compute_career_games_to_date  (leak-proof strict cutoff)
# --------------------------------------------------------------------------- #
def _frame(rows):
    """Build a small performance frame from (player, year, round_number, date) tuples."""
    df = pd.DataFrame(rows, columns=["player", "year", "round_number", "date"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_career_games_first_game_is_zero():
    """A player's debut game has zero prior games."""
    df = _frame([("A", 2020, 1, "2020-03-20")])
    counts = compute_career_games_to_date(df)
    assert counts.iloc[0] == 0


def test_career_games_counts_only_strictly_prior():
    """The count is the number of games STRICTLY before each row, in order."""
    df = _frame(
        [
            ("A", 2020, 1, "2020-03-20"),
            ("A", 2020, 2, "2020-03-27"),
            ("A", 2020, 3, "2020-04-03"),
        ]
    )
    counts = compute_career_games_to_date(df)
    assert list(counts) == [0, 1, 2]


def test_career_games_no_future_leakage_row_order_independent():
    """The frame arriving out of chronological order must not change counts.

    We build the same 3-game history shuffled and confirm each game still sees
    only its strictly-prior games.
    """
    df = _frame(
        [
            ("A", 2020, 3, "2020-04-03"),  # last game listed first
            ("A", 2020, 1, "2020-03-20"),  # debut listed last-ish
            ("A", 2020, 2, "2020-03-27"),
        ]
    )
    counts = compute_career_games_to_date(df)
    # Align back by round to check the temporal-order semantics regardless of
    # the input row order.
    by_round = dict(zip(df["round_number"], counts))
    assert by_round[1] == 0
    assert by_round[2] == 1
    assert by_round[3] == 2


def test_career_games_appending_future_game_does_not_change_earlier_counts():
    """LEAKAGE GUARD: adding a later game must leave earlier rows' counts intact.

    This is the defining property of a strict temporal cutoff — an earlier
    row cannot 'see' a game that happens after it.
    """
    base = _frame(
        [
            ("A", 2020, 1, "2020-03-20"),
            ("A", 2020, 2, "2020-03-27"),
        ]
    )
    counts_base = compute_career_games_to_date(base)

    with_future = _frame(
        [
            ("A", 2020, 1, "2020-03-20"),
            ("A", 2020, 2, "2020-03-27"),
            ("A", 2020, 3, "2020-04-03"),  # a future game appended
        ]
    )
    counts_future = compute_career_games_to_date(with_future)

    # The first two rows' counts are identical whether or not the future game exists.
    assert list(counts_base) == list(counts_future.iloc[:2])
    assert list(counts_base) == [0, 1]


def test_career_games_independent_per_player():
    """Counts reset per player — one player's history never bleeds into another's."""
    df = _frame(
        [
            ("A", 2020, 1, "2020-03-20"),
            ("B", 2020, 1, "2020-03-21"),
            ("A", 2020, 2, "2020-03-27"),
            ("B", 2020, 2, "2020-03-28"),
            ("B", 2020, 3, "2020-04-04"),
        ]
    )
    counts = compute_career_games_to_date(df)
    df = df.assign(cg=counts.values)
    a = df[df["player"] == "A"].sort_values("round_number")["cg"].tolist()
    b = df[df["player"] == "B"].sort_values("round_number")["cg"].tolist()
    assert a == [0, 1]
    assert b == [0, 1, 2]


def test_career_games_spans_multiple_seasons():
    """Prior-season games count toward the running career total (across-season)."""
    df = _frame(
        [
            ("A", 2019, 20, "2019-08-01"),
            ("A", 2019, 21, "2019-08-08"),
            ("A", 2020, 1, "2020-03-20"),
        ]
    )
    counts = compute_career_games_to_date(df)
    by_key = {(r.year, r.round_number): c for r, c in zip(df.itertuples(), counts)}
    assert by_key[(2019, 20)] == 0
    assert by_key[(2019, 21)] == 1
    assert by_key[(2020, 1)] == 2  # two prior-season games precede the 2020 debut


def test_career_games_index_preserved():
    """Returned Series aligns to the input frame's (possibly non-default) index."""
    df = _frame(
        [
            ("A", 2020, 2, "2020-03-27"),
            ("A", 2020, 1, "2020-03-20"),
        ]
    )
    df.index = [99, 7]
    counts = compute_career_games_to_date(df)
    assert set(counts.index) == {99, 7}
    # Row with index 7 is the debut (round 1) -> 0 prior games.
    assert counts.loc[7] == 0
    assert counts.loc[99] == 1
