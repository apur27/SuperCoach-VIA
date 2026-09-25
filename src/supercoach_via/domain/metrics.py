"""Coverage-aware count/sum/mean definitions (PLAN 4.3; AUDIT C10; pending-decision 3).

Rules implemented here and nowhere else:

* A blank count stays null. It is never zero-filled by these helpers.
* A stat sum with no observed games is null (unknown), not zero.
* A mean divides by the games where the stat was *observed* (dropna over recorded games —
  the resolved recorded-games denominator of ``docs/pending-decisions.md`` item 3).
  Because that rate can rest on a small part of a career, every aggregate also reports
  ``observed_games`` of ``career_games`` ("N of M") and the coverage fraction.
* ``eligible_games`` = games inside the stat's recording era (``recorded_from`` in
  ``config/stat_coverage_eras.yaml``); it tells a reader whether missing values are
  "not recorded in that era" or "missing inside the era".
* ``career_games`` = ``max(observed row count, max source career counter)``: the source
  counter can lead the rows (missing drawn-final/finals rows) and trail them (arrows,
  trailing blanks). Both inputs are exposed separately by callers.

The legacy ranking methodology (``legacy_v1``) zero-fills missing stats. That is kept as a
named, documented imputation (:func:`legacy_zero_fill_sum`) used only by that
methodology — never by general statistics.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from supercoach_via.domain.schemas import LEGACY_PLAYER_COLUMN_MAP

if TYPE_CHECKING:  # pragma: no cover
    from supercoach_via.publish.view_models import StatValue

Number = int | float

_ARROWS = str.maketrans("", "", "↑↓")
_LEADING_DIGITS = re.compile(r"^\s*(\d+)")


def _is_missing(value: Number | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def parse_counter_token(token: str | int | float | None) -> int | None:
    """Parse an AFLTables career game counter token (``'12↑'``, ``'7↓'``) to an int."""
    if token is None:
        return None
    if isinstance(token, int | float):
        return None if isinstance(token, float) and math.isnan(token) else int(token)
    match = _LEADING_DIGITS.match(token.translate(_ARROWS))
    return int(match.group(1)) if match else None


def canonical_games(row_count: int, counter_max: int | None) -> int:
    """Canonical games metric: ``max(observed rows, max career counter)``."""
    return max(row_count, counter_max or 0)


@dataclass(frozen=True)
class StatAggregate:
    """A disclosed aggregate: total/mean with their denominators."""

    stat: str
    total: float | None
    observed_games: int
    eligible_games: int
    career_games: int

    @property
    def mean(self) -> float | None:
        if self.total is None or self.observed_games == 0:
            return None
        return self.total / self.observed_games

    @property
    def coverage(self) -> float | None:
        """Observed games / career (scope) games; ``None`` when the scope is empty."""
        if self.career_games <= 0:
            return None
        return min(1.0, self.observed_games / self.career_games)

    @property
    def eligible_coverage(self) -> float | None:
        """Observed games / games inside the recording era."""
        if self.eligible_games <= 0:
            return None
        return min(1.0, self.observed_games / self.eligible_games)

    def n_of_m(self) -> str:
        return f"{self.observed_games} of {self.career_games}"

    def to_stat_value(self) -> StatValue:
        from supercoach_via.publish.view_models import StatValue

        return StatValue(
            stat=self.stat,
            total=self.total,
            mean=self.mean,
            observed_games=self.observed_games,
            eligible_games=self.eligible_games,
            coverage=self.coverage,
        )


def aggregate(
    stat: str,
    values: Sequence[Number | None],
    *,
    seasons: Sequence[int] | None = None,
    recorded_from: int | None = None,
    career_games: int | None = None,
) -> StatAggregate:
    """Aggregate one stat over one scope (a career, a season, a team-game...).

    ``values`` holds one entry per game row in scope (``None``/NaN = not recorded).
    ``career_games`` defaults to the row count and may not be smaller than it.
    """
    n_rows = len(values)
    if seasons is not None and len(seasons) != n_rows:
        raise ValueError("seasons must align with values")
    if career_games is None:
        career_games = n_rows
    if career_games < n_rows:
        raise ValueError(f"career_games {career_games} < observed rows {n_rows}")
    observed = [float(v) for v in values if v is not None and not _is_missing(v)]
    total: float | None = sum(observed) if observed else None
    eligible = (
        n_rows
        if seasons is None or recorded_from is None
        else sum(1 for s in seasons if s >= recorded_from)
    )
    return StatAggregate(
        stat=stat,
        total=total,
        observed_games=len(observed),
        eligible_games=eligible,
        career_games=career_games,
    )


def legacy_zero_fill_sum(values: Iterable[Number | None]) -> float:
    """``legacy_v1`` imputation: a blank stat cell counts as zero.

    Reproduces ``pd.to_numeric(..).fillna(0).sum()`` in ``top_players_comprehensive.py``.
    Only the historical ranking methodology may use this.
    """
    total = 0.0
    for v in values:
        if v is not None and not _is_missing(v):
            total += float(v)
    return total


@dataclass(frozen=True)
class CoverageEras:
    """``recorded_from`` season per canonical stat name."""

    recorded: Mapping[str, int]

    @classmethod
    def load(cls, path: Path) -> CoverageEras:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        out: dict[str, int] = {}
        for name, spec in (raw.get("stats") or {}).items():
            canonical = LEGACY_PLAYER_COLUMN_MAP.get(name, name)
            out[canonical] = int(spec["recorded_from"])
        return cls(out)

    def recorded_from(self, stat: str) -> int | None:
        return self.recorded.get(LEGACY_PLAYER_COLUMN_MAP.get(stat, stat))

    def is_recorded(self, stat: str, season: int) -> bool:
        """True unless a boundary is configured and ``season`` precedes it."""
        start = self.recorded_from(stat)
        return start is None or season >= start


def _ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe column name {name!r}")
    return f'"{name}"'


def sql_aggregate_columns(
    stats: Iterable[str], eras: CoverageEras, *, season_column: str = "season"
) -> str:
    """DuckDB select-list computing, per stat: ``<s>_total, <s>_observed, <s>_eligible``.

    ``SUM`` over only-null input is NULL (a sum with no observed games is unknown);
    ``COUNT(col)`` counts observed games; eligibility uses the recording era.
    """
    parts: list[str] = []
    season = _ident(season_column)
    for stat in stats:
        col = _ident(stat)
        start = eras.recorded_from(stat)
        eligible = (
            "COUNT(*)"
            if start is None
            else f"COUNT(*) FILTER (WHERE {season} >= {int(start)})"
        )
        parts.append(
            f"SUM({col}) AS {_ident(stat + '_total')}, "
            f"COUNT({col}) AS {_ident(stat + '_observed')}, "
            f"{eligible} AS {_ident(stat + '_eligible')}"
        )
    return ", ".join(parts)
