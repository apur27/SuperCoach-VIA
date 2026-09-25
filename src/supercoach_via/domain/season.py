"""Fixture stage, chronology, replay and date-quality rules.

Pure functions only. A source stage token (match ``round_num`` or player ``round``) is
kept verbatim as ``label`` and resolved to explicit stage fields. Unknown tokens are
preserved and flagged (``recognized=False``, ``StageType.OTHER``) -- never silently
assigned a normal round. Stage order comes from observed chronology, not round numbers.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from supercoach_via.domain.schemas import DateQuality, StageType, is_safe_id

#: canonical final code -> (accepted tokens, nominal rank within the finals series)
_FINALS: dict[str, tuple[frozenset[str], int]] = {
    "wf": (frozenset({"wf", "wildcard final", "wild card final", "wildcard round"}), 1),
    "qf": (frozenset({"qf", "qualifying final"}), 2),
    "ef": (frozenset({"ef", "elimination final"}), 2),
    "sf": (frozenset({"sf", "semi final", "semi-final"}), 3),
    "pf": (frozenset({"pf", "preliminary final"}), 4),
    "gf": (frozenset({"gf", "grand final"}), 5),
}
_TOKEN_TO_FINAL = {tok: code for code, (toks, _) in _FINALS.items() for tok in toks}
_OPENING = frozenset({"opening round", "or", "0"})
_NUMERIC = re.compile(r"^\d{1,2}$")

#: Nominal ranks used only to break ties / order undated stages.
_REGULAR_BASE = 0
_FINALS_BASE = 1000
_OTHER_RANK = 10_000


@dataclass(frozen=True)
class Stage:
    label: str  # source token exactly as read
    stage_id: str  # safe token: r00..r99, wf/qf/ef/sf/pf/gf, or x<hash> when unrecognized
    stage_type: StageType
    round_number: int | None
    recognized: bool

    @property
    def nominal_rank(self) -> int:
        if self.stage_type is StageType.REGULAR and self.round_number is not None:
            return _REGULAR_BASE + self.round_number
        if self.stage_type is StageType.FINAL:
            return _FINALS_BASE + _FINALS[self.stage_id][1]
        return _OTHER_RANK


def parse_stage(token: str) -> Stage:
    """Resolve a legacy stage token. Unknown tokens are preserved and flagged."""
    norm = " ".join(token.strip().lower().split())
    if norm in _OPENING:
        return Stage(token, "r00", StageType.REGULAR, 0, True)
    if _NUMERIC.match(norm):
        n = int(norm)
        return Stage(token, f"r{n:02d}", StageType.REGULAR, n, True)
    code = _TOKEN_TO_FINAL.get(norm)
    if code is not None:
        return Stage(token, code, StageType.FINAL, None, True)
    digest = hashlib.sha256(token.encode()).hexdigest()[:12]
    return Stage(token, f"x{digest}", StageType.OTHER, None, False)


_STAGE_CACHE: dict[str, Stage] = {}


def parse_stage_cached(token: str) -> Stage:
    stage = _STAGE_CACHE.get(token)
    if stage is None:
        stage = _STAGE_CACHE[token] = parse_stage(token)
    return stage


def _rank_for_id(stage_id: str) -> int:
    if stage_id.startswith("r") and stage_id[1:].isdigit():
        return _REGULAR_BASE + int(stage_id[1:])
    if stage_id in _FINALS:
        return _FINALS_BASE + _FINALS[stage_id][1]
    return _OTHER_RANK


def stage_orders(stage_dates: Mapping[str, Iterable[date | None]]) -> dict[str, int]:
    """1-based within-season stage order from chronology (earliest match date per stage).

    Dated stages are ordered by first date, ties broken by nominal rank then id.
    Undated stages follow all dated stages in nominal rank order.
    """
    keyed = []
    for stage_id, dates in stage_dates.items():
        known = [d for d in dates if d is not None]
        first = min(known) if known else None
        keyed.append((first is None, first or date.min, _rank_for_id(stage_id), stage_id))
    keyed.sort()
    return {k[3]: i for i, k in enumerate(keyed, start=1)}


def replay_occurrences(keys: Sequence[tuple[str, str, str, date | None]], tiebreak: Sequence[int]) -> list[int]:
    """Occurrence index per (stage_id, unordered club pair), ordered by date then tiebreak.

    ``keys`` are (stage_id, club_a, club_b, date); ``tiebreak`` is a stable secondary key
    (e.g. source row). 0 = first meeting, 1 = first replay/repeat, ...
    """
    groups: dict[tuple[str, str, str], list[int]] = {}
    for i, (stage_id, a, b, _d) in enumerate(keys):
        lo, hi = sorted((a, b))
        groups.setdefault((stage_id, lo, hi), []).append(i)
    out = [0] * len(keys)
    for members in groups.values():
        members.sort(key=lambda i: (keys[i][3] is None, keys[i][3] or date.min, tiebreak[i]))
        for occ, i in enumerate(members):
            out[i] = occ
    return out


def make_match_id(season: int, stage_id: str, club_a: str, club_b: str, occurrence: int) -> str:
    """Persistent surrogate: season + stage + unordered participants + occurrence (no date)."""
    lo, hi = sorted((club_a, club_b))
    mid = f"m:{season}:{stage_id}:{lo}:{hi}:{occurrence}"
    if not is_safe_id(mid):
        raise ValueError(f"unsafe match id {mid!r}")
    return mid


def is_legacy_synthetic_date(value: date, season: int) -> bool:
    """True for the legacy 'March 1 + whole weeks' reconstruction (incl. YYYY-03-01)."""
    delta = (value - date(season, 3, 1)).days
    return delta >= 0 and delta % 7 == 0


def row_date_quality(row_date: date | None, match_date: date | None, season: int) -> DateQuality:
    if row_date is None:
        return DateQuality.UNKNOWN
    if match_date is not None and row_date == match_date:
        return DateQuality.FIXTURE_VERIFIED
    if is_legacy_synthetic_date(row_date, season):
        return DateQuality.INFERRED
    return DateQuality.SOURCE
