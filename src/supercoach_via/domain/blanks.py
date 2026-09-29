"""What a blank AFL Tables statistic cell means (one rule for the legacy and source imports).

AFL Tables prints a zero count as a blank cell. Treating every blank as "not recorded"
turned real zeros into missing data and inflated observed-denominator means. Treating
every blank as zero would invent data for unreported eras, matches and rows. The rule
here converts a blank to ``0`` only where the source proves the column was reported for
that match and the player took the field. Evidence and annotated cases:
``tests/scvia/fixtures/zero_semantics/annotations.json``.

For each blank (null) cell of statistic ``s`` in match ``M``:

* ``time_on_ground_pct`` stays null: a player who took the field always has a positive value.
* Brownlow votes in a final stay null: votes are never awarded in finals (not applicable).
* Before ``recorded_from[s]`` the blank stays null, even when an isolated value exists.
* If no row of ``M`` has a value for ``s``, the column was not reported for ``M``: null.
* A row is a game in the player's log. If the match reports a statistic everyone who takes
  the field records (kicks, marks, handballs, disposals, time on ground) and the row has none
  of them and no other value, it is not evidence of a zero game: its blanks stay null. In
  goals-only eras a non-scorer's row is blank by construction and counts as played.
* Otherwise the blank is ``0``.

Recorded values are never changed. The rule is idempotent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from supercoach_via.domain.schemas import PLAYER_STAT_COLUMNS

if TYPE_CHECKING:  # pragma: no cover
    import pyarrow as pa

RULE_VERSION = "blanks-v1"
NEVER_ZERO = frozenset({"time_on_ground_pct"})
FINALS_NOT_APPLICABLE = frozenset({"brownlow_votes"})
#: statistics whose presence shows a row took the field (Brownlow is awarded, not played)
_PLAYED_EVIDENCE = tuple(s for s in PLAYER_STAT_COLUMNS if s != "brownlow_votes")
#: statistics every player who takes the field records when the match reports them
_UNIVERSAL = ("kicks", "marks", "handballs", "disposals", "time_on_ground_pct")


def resolve_blanks(
    games: pa.Table, stage_types: Mapping[str, str], recorded_from: Mapping[str, int]
) -> tuple[pa.Table, dict[str, int]]:
    """``games`` (player_games columns incl. match_id and season) with source blanks resolved.

    ``stage_types`` maps match_id -> ``regular`` | ``final`` | ``other``. Returns the new table
    and the number of cells converted to zero per statistic.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    n = games.num_rows
    counts = {s: 0 for s in PLAYER_STAT_COLUMNS}
    if n == 0:
        return games, counts
    match_ids = games.column("match_id").to_pylist()
    seasons = np.asarray(games.column("season").to_pylist(), dtype=np.int64)
    # group rows by match
    codes: dict[str, int] = {}
    group = np.fromiter((codes.setdefault(m, len(codes)) for m in match_ids), dtype=np.int64, count=n)
    final_match = np.array([stage_types.get(m) == "final" for m in codes], dtype=bool)
    in_final = final_match[group]
    played = np.zeros(n, dtype=bool)
    valid: dict[str, np.ndarray] = {}
    for s in PLAYER_STAT_COLUMNS:
        if s in games.column_names:
            valid[s] = pc.is_valid(games.column(s)).to_numpy(zero_copy_only=False)
    for s in _PLAYED_EVIDENCE:
        if s in valid:
            played |= valid[s]
    universal = np.zeros(len(codes), dtype=bool)
    for s in _UNIVERSAL:
        if s in valid:
            np.logical_or.at(universal, group, valid[s])
    # in a match without universal statistics every listed row is a game played
    played |= ~universal[group]
    out = games
    for s, has_value in valid.items():
        if s in NEVER_ZERO:
            continue
        reported = np.zeros(len(codes), dtype=bool)
        np.logical_or.at(reported, group, has_value)
        convert = ~has_value & played & reported[group]
        start = recorded_from.get(s)
        if start is not None:
            convert &= seasons >= int(start)
        if s in FINALS_NOT_APPLICABLE:
            convert &= ~in_final
        k = int(convert.sum())
        if not k:
            continue
        counts[s] = k
        col = out.column(s)
        zero = pa.scalar(0, type=col.type)
        idx = out.column_names.index(s)
        out = out.set_column(idx, out.schema.field(idx), pc.if_else(pa.array(convert), zero, col))
    return out, counts
