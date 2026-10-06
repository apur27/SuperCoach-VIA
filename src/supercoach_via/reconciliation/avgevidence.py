"""Evidence from the source's own printed season figures for whole-team blank columns.

AFL Tables prints a zero as a blank, team totals included (no team total of "0" occurs in the frozen capture), so
a team-match whose column is blank for every player is either a recorded zero or a statistic the source did not
record for that match. The notes page states that games where a statistic is missing do not count when the
player's average is calculated. A player's printed season average therefore says which: the average divides the
season total by the counted games (``aggregate.season_denominator``: printed values, recorded zeros and credited
did-not-take-the-field games). When exactly one count of the ambiguous games reproduces the printed average, and
that count is all of them or none, the player's own printed figure proves the team-match was counted or not.

Brownlow votes are an award, not a team statistic, so a partial match record proves nothing about another
player. A player's blank Brownlow cell is a zero only when his printed season total equals the votes printed in his
game rows: no vote of his can then be missing from a blank game.

Pure functions; no I/O.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from supercoach_via.reconciliation.aggregate import average_matches

Key = tuple[str, str]  # (match url, team as printed on the match page)


@dataclass(frozen=True)
class GameState:
    key: Key
    #: value | zero | dntf (counted) ; ambiguous (whole-team blank) ; excluded (not recorded, not applicable)
    state: str
    value: Decimal | None = None


def counted_solution(total: Decimal, printed: str | None, *, known: int, ambiguous: int) -> int | None:
    """The unique number of ambiguous games ``x`` (0..ambiguous) whose inclusion reproduces ``printed``."""
    if not printed or not printed.strip() or total == 0:
        return None
    fits = [x for x in range(ambiguous + 1) if average_matches(total, known + x, printed.strip())]
    return fits[0] if len(fits) == 1 else None


def player_votes(games: list[GameState], printed_avg: str | None) -> dict[Key, str]:
    """``recorded`` / ``unrecorded`` for each ambiguous team-match of one player-club-season, or nothing."""
    amb = [g.key for g in games if g.state == "ambiguous"]
    if not amb:
        return {}
    total = sum((g.value for g in games if g.state == "value" and g.value is not None), Decimal(0))
    known = sum(1 for g in games if g.state in ("value", "zero", "dntf"))
    x = counted_solution(total, printed_avg, known=known, ambiguous=len(amb))
    if x == len(amb):
        return dict.fromkeys(amb, "recorded")
    if x == 0:
        return dict.fromkeys(amb, "unrecorded")
    return {}


def team_evidence(votes: Iterable[tuple[Key, str, str]]) -> dict[tuple[Key, str], str]:
    """Unanimous votes per (team-match, statistic); disagreeing players make it a ``conflict``."""
    seen: dict[tuple[Key, str], set[str]] = defaultdict(set)
    for key, field_name, vote in votes:
        seen[(key, field_name)].add(vote)
    return {k: (next(iter(v)) if len(v) == 1 else "conflict") for k, v in seen.items()}


def brownlow_season_complete(game_votes: list[Decimal], printed_total: str | None) -> bool:
    """The printed season total equals the votes printed in the player's game rows (blank total = 0 votes)."""
    if printed_total is None:
        return False
    text = printed_total.strip()
    try:
        total = Decimal(text) if text else Decimal(0)
    except InvalidOperation:
        return False
    return sum(game_votes, Decimal(0)) == total
