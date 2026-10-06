"""Whole legacy CSV rows rebuilt from captured AFL Tables pages, in each file's own column order.

A player-game row is the profile's game row exactly as printed (a blank stays blank: the legacy layer stores what
the source prints) plus the match page's date; a match row is the match page's stage, venue, local start,
attendance and cumulative quarter scores. Nothing here is inferred: a value the page does not print is blank.
"""

from __future__ import annotations

from supercoach_via.reconciliation.local import LEGACY_COLUMNS
from supercoach_via.reconciliation.schema import STAT_FIELDS
from supercoach_via.reconciliation.source import MatchFacts, ProfileGame

PERF_HEADER = (
    "team", "year", "games_played", "opponent", "round", "result", "jersey_num",
    *LEGACY_COLUMNS, "date",
)  # fmt: skip
PERSONAL_HEADER = ("first_name", "last_name", "born_date", "debut_date", "height", "weight")
_QUARTERS = ("q1", "q2", "q3", "final")


def player_row(header: list[str] | tuple[str, ...], g: ProfileGame, match_date: str | None) -> list[str]:
    stat = {col: (g.cells[STAT_FIELDS.index(f)] or "") for col, f in LEGACY_COLUMNS.items()}
    fixed = {
        "team": g.club,
        "year": str(g.season),
        "games_played": "" if g.counter is None else str(g.counter),
        "opponent": g.opponent,
        "round": g.rd_token,
        "result": g.result,
        "jersey_num": "".join(ch for ch in g.jersey_token if ch.isdigit()),
        "date": match_date or "",
    }
    return [fixed[c] if c in fixed else stat.get(c, "") for c in header]


def match_row(header: list[str] | tuple[str, ...], m: MatchFacts) -> list[str]:
    rec: dict[str, str] = {
        "round_num": m.stage_text or "",
        "venue": m.venue or "",
        "date": m.local_start or (m.match_date or ""),
        "year": (m.match_date or "")[:4],
        "attendance": "" if m.attendance is None else str(m.attendance),
    }
    for n, team in enumerate(m.teams[:2], 1):
        rec[f"team_{n}_team_name"] = team.name
        # q1-q3 as printed; "final" is the result, i.e. the last score line (after extra time when played)
        lines = (*team.quarters[:3], team.quarters[-1]) if len(team.quarters) >= 4 else team.quarters
        for q, (goals, behinds) in zip(_QUARTERS, lines, strict=False):
            rec[f"team_{n}_{q}_goals"] = str(goals)
            rec[f"team_{n}_{q}_behinds"] = str(behinds)
    return [rec.get(c, "") for c in header]
