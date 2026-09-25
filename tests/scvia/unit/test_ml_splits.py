"""Chronological split contracts (M03 split properties)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from supercoach_via.ml import splits as S


def _rows(n_days: int = 60, per_day: int = 3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    days = [date(2020, 1, 1) + timedelta(days=int(d)) for d in sorted(rng.integers(0, 400, n_days))]
    dates, matches = [], []
    for i, d in enumerate(days):
        for j in range(per_day):
            dates.append(d)
            matches.append(f"m{i}_{j % 2}")
    return np.array(dates, dtype=object), np.array(matches, dtype=object)


def test_plan_blocks_are_ordered_and_disjoint() -> None:
    dates, matches = _rows()
    plan = S.plan_splits(
        dates, matches, train_cutoff=date(2020, 10, 1), calibration_end=date(2020, 12, 1),
        holdout_end=None, n_folds=3,
    )
    blocks = S.assign_blocks(dates, plan)
    assert set(blocks) <= {"train", "calibration", "holdout"}
    assert max(d for d, b in zip(dates, blocks, strict=True) if b == "train") < date(2020, 10, 1)
    cal = [d for d, b in zip(dates, blocks, strict=True) if b == "calibration"]
    hold = [d for d, b in zip(dates, blocks, strict=True) if b == "holdout"]
    assert min(cal) >= date(2020, 10, 1) and max(cal) < min(hold)
    for fold in plan.folds:
        tr, va = fold.masks(dates)
        assert tr.any() and va.any()
        assert max(dates[tr]) < min(dates[va])
        assert not set(matches[tr]) & set(matches[va])
        # folds live entirely inside the training block: never touch calibration/holdout
        assert all(b == "train" for b in blocks[tr | va])


def test_no_later_game_in_train_of_earlier_validation_target() -> None:
    dates, matches = _rows(per_day=4)
    plan = S.plan_splits(dates, matches, train_cutoff=date(2021, 3, 1),
                         calibration_end=date(2021, 3, 1), holdout_end=None, n_folds=4)
    for fold in plan.folds:
        tr, va = fold.masks(dates)
        assert np.all(dates[tr].max() < dates[va])


def test_match_spanning_two_dates_is_rejected() -> None:
    dates = np.array([date(2020, 1, 1), date(2020, 1, 2)], dtype=object)
    matches = np.array(["m1", "m1"], dtype=object)
    with pytest.raises(S.SplitError):
        S.plan_splits(dates, matches, train_cutoff=date(2020, 2, 1),
                      calibration_end=date(2020, 2, 1), holdout_end=None, n_folds=1)


def test_season_folds_validate_whole_seasons() -> None:
    dates = np.array([date(y, m, 1) for y in (2019, 2020, 2021, 2022) for m in (4, 6, 8)], dtype=object)
    folds = S.season_folds(dates, (2021, 2022))
    assert [f.valid_start for f in folds] == [date(2021, 4, 1), date(2022, 4, 1)]
    tr, va = folds[0].masks(dates)
    assert tr.sum() == 6 and va.sum() == 3


def test_group_kfold_player_is_only_a_diagnostic_with_disjoint_players() -> None:
    players = np.array([f"p{i % 7}" for i in range(70)], dtype=object)
    splits = S.player_group_diagnostic(players, n_splits=3)
    assert len(splits) == 3
    for tr, va in splits:
        assert not set(players[tr]) & set(players[va])


def test_plan_fingerprint_is_stable_and_sensitive() -> None:
    dates, matches = _rows()
    kw = dict(train_cutoff=date(2020, 10, 1), calibration_end=date(2020, 12, 1), holdout_end=None)
    a = S.plan_splits(dates, matches, n_folds=3, **kw)  # type: ignore[arg-type]
    b = S.plan_splits(dates, matches, n_folds=3, **kw)  # type: ignore[arg-type]
    c = S.plan_splits(dates, matches, n_folds=2, **kw)  # type: ignore[arg-type]
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()


@settings(max_examples=40, deadline=None)
@given(n_days=st.integers(20, 120), per_day=st.integers(1, 5), n_folds=st.integers(1, 5),
       seed=st.integers(0, 10_000))
def test_property_expanding_folds(n_days: int, per_day: int, n_folds: int, seed: int) -> None:
    dates, matches = _rows(n_days, per_day, seed)
    cutoff = sorted(set(dates))[-1] + timedelta(days=1)
    folds = S.expanding_date_folds(dates, matches, n_folds=n_folds, end=cutoff)
    prev_valid_end = None
    for f in folds:
        tr, va = f.masks(dates)
        if not va.any():
            continue
        assert dates[tr].max() < dates[va].min()
        assert not set(matches[tr]) & set(matches[va])
        if prev_valid_end is not None:
            assert f.valid_start >= prev_valid_end
        prev_valid_end = f.valid_end
