"""Deterministic, labelled live "reads" (PLAN 10).

Every read names the field family it came from (``[scoreboard]`` or
``[reliable: <fields>]``) and uses only the scoreboard and the reviewed reliable
per-player fields. Unreliable per-player goals/behinds/clangers and unnamed columns
are never used. Output depends only on the parsed feed (ties break by name), so the
same payload always yields the same reads. No free text from the feed is echoed.
"""

from __future__ import annotations

from supercoach_via.ingest.fanfooty import FeedParse, FeedPlayer


def _phase_note(feed: FeedParse) -> str:
    p = feed.phase
    if p.final:
        return "final"
    if p.is_break:
        return f"{p.label} break"
    return f"{p.label} in progress; partial quarter"


def _leader(players: list[FeedPlayer], fields: tuple[str, ...]) -> tuple[FeedPlayer, int] | None:
    best: tuple[int, str, FeedPlayer] | None = None
    for pl in players:
        vals = [pl.reliable.get(f) for f in fields]
        if any(not isinstance(v, int) for v in vals):
            continue
        total = sum(int(v) for v in vals if isinstance(v, int))
        key = (-total, pl.name, pl)
        if best is None or key[:2] < best[:2]:
            best = key
    return (best[2], -best[0]) if best else None


def reads_for(feed: FeedParse) -> list[str]:
    h = feed.header
    out: list[str] = []
    if h is None or h.home_score is None or h.away_score is None:
        return ["[scoreboard] unavailable"]
    margin = h.home_score - h.away_score
    note = _phase_note(feed)
    if margin == 0:
        out.append(f"[scoreboard] Scores level at {h.home_score} ({note})")
    else:
        lead, trail = (h.home_name, h.away_name) if margin > 0 else (h.away_name, h.home_name)
        verb = "won by" if feed.phase.final else "lead by"
        out.append(f"[scoreboard] {lead} {verb} {abs(margin)} points over {trail} ({note})")
    for fields, label in (
        (("kicks", "handballs"), "Most disposals"),
        (("af",), "Top AFL Fantasy score"),
        (("sc",), "Top SuperCoach score"),
        (("tackles",), "Most tackles"),
    ):
        top = _leader(feed.players, fields)
        if top is not None:
            pl, value = top
            out.append(f"[reliable: {'+'.join(fields)}] {label}: {pl.name} ({pl.team_code}) {value}")
    return out
