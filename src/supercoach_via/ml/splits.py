"""Grouped chronological fold definitions (PLAN 7.3).

Rows are grouped by calendar date blocks, and every match has exactly one date, so no
match is ever split across a train/validation boundary. Expanding folds: fold ``i``
trains on every date before its validation block. The layout of one training run is::

    [ ---- train (inner expanding folds live here) ---- ) [ calibration ) [ holdout ]
                                                   train_cutoff   calibration_end

The holdout block is never used for tuning, feature, calibration or interval
selection; the calibration block is later than all training rows and earlier than the
holdout. ``player_group_diagnostic`` (GroupKFold by player) is an unseen-player
diagnostic only, never the primary future-game evaluation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import numpy as np


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: date | None  # inclusive; None = from the beginning
    valid_start: date  # train uses dates < valid_start
    valid_end: date  # exclusive

    def masks(self, dates: Sequence[date] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = np.asarray(dates, dtype=object)
        tr = np.array([x < self.valid_start and (self.train_start is None or x >= self.train_start)
                       for x in d], dtype=bool)
        va = np.array([self.valid_start <= x < self.valid_end for x in d], dtype=bool)
        return tr, va


@dataclass(frozen=True)
class SplitPlan:
    folds: tuple[Fold, ...]
    train_cutoff: date  # training rows: date < train_cutoff
    calibration_end: date  # calibration rows: train_cutoff <= date < calibration_end
    holdout_end: date | None  # holdout rows: calibration_end <= date (< holdout_end)

    def fingerprint(self) -> str:
        payload: dict[str, Any] = asdict(self)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _check_match_dates(dates: np.ndarray, matches: np.ndarray) -> None:
    seen: dict[Any, date] = {}
    for d, m in zip(dates, matches, strict=True):
        if seen.setdefault(m, d) != d:
            raise SplitError(f"match {m!r} spans more than one date")


def expanding_date_folds(
    dates: Sequence[date] | np.ndarray,
    matches: Sequence[str] | np.ndarray,
    *,
    n_folds: int,
    end: date,
    min_train_fraction: float = 0.4,
) -> tuple[Fold, ...]:
    """Split rows with date < ``end`` into an initial train block + ``n_folds`` validation
    blocks of roughly equal row counts, cut only between whole dates."""
    d = np.asarray(dates, dtype=object)
    m = np.asarray(matches, dtype=object)
    _check_match_dates(d, m)
    keep = np.array([x < end for x in d], dtype=bool)
    if not keep.any() or n_folds < 1:
        return ()
    uniq, counts = np.unique(d[keep], return_counts=True)
    if len(uniq) < n_folds + 1:
        raise SplitError("not enough distinct dates for the requested folds")
    cum = np.cumsum(counts) / counts.sum()
    first = min_train_fraction
    step = (1.0 - first) / n_folds
    bounds: list[int] = []  # index into uniq where each validation block starts
    for i in range(n_folds):
        target = first + i * step
        idx = int(np.searchsorted(cum, target, "right"))
        idx = max(idx, (bounds[-1] + 1) if bounds else 1)
        idx = min(idx, len(uniq) - (n_folds - i))
        bounds.append(idx)
    folds = []
    for i, b in enumerate(bounds):
        vend = uniq[bounds[i + 1]] if i + 1 < len(bounds) else end
        folds.append(Fold(index=i, train_start=None, valid_start=uniq[b], valid_end=vend))
    return tuple(folds)


def season_folds(dates: Sequence[date] | np.ndarray, validation_seasons: Sequence[int]) -> tuple[Fold, ...]:
    """One expanding fold per validation season (calendar-year seasons, AFL)."""
    d = np.asarray(dates, dtype=object)
    folds = []
    for i, s in enumerate(sorted(validation_seasons)):
        in_season = [x for x in d if x.year == s]
        if not in_season:
            raise SplitError(f"no rows in validation season {s}")
        later = [x for x in d if x.year > s]
        vend = min(later) if later else date(s + 1, 1, 1)
        folds.append(Fold(index=i, train_start=None, valid_start=min(in_season), valid_end=vend))
    return tuple(folds)


def plan_splits(
    dates: Sequence[date] | np.ndarray,
    matches: Sequence[str] | np.ndarray,
    *,
    train_cutoff: date,
    calibration_end: date,
    holdout_end: date | None,
    n_folds: int,
    folds: tuple[Fold, ...] | None = None,
) -> SplitPlan:
    if calibration_end < train_cutoff:
        raise SplitError("calibration_end precedes train_cutoff")
    if holdout_end is not None and holdout_end <= calibration_end:
        raise SplitError("holdout_end must follow calibration_end")
    d = np.asarray(dates, dtype=object)
    m = np.asarray(matches, dtype=object)
    _check_match_dates(d, m)
    if folds is None:
        folds = expanding_date_folds(d, m, n_folds=n_folds, end=train_cutoff)
    for f in folds:
        if f.valid_end > train_cutoff:
            raise SplitError("a fold's validation block extends past the train cutoff")
    return SplitPlan(folds=folds, train_cutoff=train_cutoff, calibration_end=calibration_end,
                     holdout_end=holdout_end)


def assign_blocks(dates: Sequence[date] | np.ndarray, plan: SplitPlan) -> np.ndarray:
    """Label each row train | calibration | holdout | unused (after holdout_end)."""
    out = []
    for x in np.asarray(dates, dtype=object):
        if x < plan.train_cutoff:
            out.append("train")
        elif x < plan.calibration_end:
            out.append("calibration")
        elif plan.holdout_end is None or x < plan.holdout_end:
            out.append("holdout")
        else:
            out.append("unused")
    return np.array(out, dtype=object)


def player_group_diagnostic(
    players: Sequence[str] | np.ndarray, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold(player) index pairs — an unseen-player diagnostic only."""
    from sklearn.model_selection import GroupKFold

    p = np.asarray(players, dtype=object)
    gkf = GroupKFold(n_splits=n_splits)
    return [(tr, va) for tr, va in gkf.split(np.zeros(len(p)), groups=p)]
