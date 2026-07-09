"""Age and experience feature engineering for the disposal predictor.

Pure, leak-proof feature functions kept separate from the model class so they
can be unit-tested in isolation and reused by both the live predictor
(``supercoach/prediction.py``) and the leak-proof backtest (``backtest.py``).

Two features (task S7):

* ``player_age_at_match``  -- age in years at the time of each game, computed
  from the player's date of birth and the match date.
* ``career_games_to_date`` -- number of games the player has played STRICTLY
  BEFORE the current game, in temporal order.

Temporal-cutoff invariant
--------------------------
Both features use only information available at or before each match, so they
compose safely with ``LeakProofPredictor``'s hard temporal cutoff:

* ``compute_age_years`` is deterministic from DOB and match date. It contains
  no future information by construction — a player's age at round N does not
  depend on any later round.
* ``compute_career_games_to_date`` is a 0-indexed ``groupby.cumcount`` over a
  ``(year, round_number, date)``-sorted frame. Cumcount is 0 for the first
  game and increments by one per subsequent game, so a row's value counts ONLY
  the rows that precede it in temporal order — the current game and every
  future game are excluded. When the caller has already applied a temporal
  cutoff (as the backtest does, dropping all rows at/after the scored round
  before feature engineering), future rows are simply absent from the frame
  and therefore cannot be counted. The leakage guard is verified by
  ``tests/unit/test_age_experience_features.py::
  test_career_games_appending_future_game_does_not_change_earlier_counts``.

DOB source
----------
The pipeline already parses each player's DOB from the filename token
(``surname_first_DDMMYYYY_performance_details.csv``) via
``extract_dob_and_name``. That token was verified equal to the
``personal_details.born_date`` field on a 40-file sample (0 mismatches), so it
is used directly as the join key rather than re-reading personal_details.
"""
from __future__ import annotations

import pandas as pd

# Average Gregorian year length; accounts for leap years so ages are not
# systematically inflated over long careers.
DAYS_PER_YEAR = 365.25


def compute_age_years(match_dates: pd.Series, born_date) -> pd.Series:
    """Age in years at each match date.

    Parameters
    ----------
    match_dates : pd.Series
        Match dates (anything parseable by ``pd.to_datetime``).
    born_date : scalar date or pd.Series
        A single DOB broadcast across all rows, or a per-row Series aligned to
        ``match_dates`` by index (e.g. when the frame mixes multiple players).

    Returns
    -------
    pd.Series
        Float ages in years, indexed like ``match_dates``. NaN where either the
        match date or the DOB is missing.
    """
    md = pd.to_datetime(match_dates, errors="coerce")
    bd = pd.to_datetime(born_date, errors="coerce")
    # Series - Series aligns by index; Series - scalar broadcasts.
    delta = md - bd
    return (delta.dt.days / DAYS_PER_YEAR).astype("float64")


def compute_career_games_to_date(
    df: pd.DataFrame,
    group_col: str = "player",
    sort_cols=("year", "round_number", "date"),
) -> pd.Series:
    """Number of the player's games STRICTLY BEFORE each row, in temporal order.

    Leak-proof by construction: a 0-indexed cumulative count over a temporally
    sorted frame, so each row counts only its strictly-prior games (see module
    docstring for the invariant). The current game and any future games are
    never counted.

    Parameters
    ----------
    df : pd.DataFrame
        Per-game rows. Must contain ``group_col`` and whichever of
        ``sort_cols`` are present (missing sort columns are skipped).
    group_col : str
        Column identifying a player. Counts reset per group.
    sort_cols : sequence of str
        Temporal ordering keys, most-significant first.

    Returns
    -------
    pd.Series
        Int64 counts aligned to ``df.index``.
    """
    use_cols = [c for c in sort_cols if c in df.columns]
    # mergesort is stable, so ties (e.g. duplicate dates) keep the frame's
    # incoming relative order deterministically.
    order = df.sort_values(list(use_cols), kind="mergesort").index
    counts = df.loc[order].groupby(group_col, observed=False).cumcount()
    return counts.reindex(df.index).astype("int64")
