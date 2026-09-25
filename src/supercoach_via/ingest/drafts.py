"""Pure draft adapters: national/rookie drafts (Wikipedia) and DraftGuru year pages.

Tables are located by section heading and header names, rowspan/colspan are expanded
so later picks of a round stay aligned, and blank cells stay unknown (never zero).
A page without the expected section/table is ``FAIL``, not an empty success.
Player identity resolution belongs to ``domain.ids``: rows carry ``player_id=None``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup, Tag

from supercoach_via.domain.schemas import CheckOutcome, Provenance
from supercoach_via.ingest.http import fit_table_row

ADAPTER_VERSION = "1"
_FOOTNOTE_RE = re.compile(r"\[[^\]]*\]")
_EMPTY = {"", "-", "–", "—", "n/a", "na"}


def _check_season(season: int) -> None:
    if not 1900 <= season <= 2100:
        raise ValueError(f"season out of range: {season}")


def wikipedia_url(season: int) -> str:
    _check_season(season)
    return f"https://en.wikipedia.org/wiki/{season}_AFL_draft"


def draftguru_url(season: int) -> str:
    _check_season(season)
    return f"https://www.draftguru.com.au/years/{season}"


def clean_name(raw: str) -> str:
    return " ".join(_FOOTNOTE_RE.sub("", raw.replace("​", "")).split()).strip()


def _first_int(text: str) -> int | None:
    t = " ".join(text.split())
    if t.lower() in _EMPTY:
        return None
    m = re.match(r"\d+", t.replace(",", ""))
    return int(m.group()) if m else None


@dataclass(frozen=True)
class DraftEvent:
    season: int
    event_type: str
    draft_round: int | None
    pick: int | None
    club: str | None
    player_name: str
    recruited_from: str | None
    grade: str | None = None
    games: int | None = None
    source: str = ""

    def to_row(self, *, source_url: str, source_sha256: str | None) -> dict[str, Any]:
        row: dict[str, Any] = {
            "draft_event_id": f"draft:{self.source}:{self.season}:{self.event_type}:"
            f"{self.draft_round if self.draft_round is not None else 'x'}:{self.pick}",
            "season": self.season,
            "event_type": self.event_type,
            "draft_round": self.draft_round,
            "pick": self.pick,
            "club_id": None,
            "club_source_name": self.club,
            "player_name": self.player_name,
            "player_id": None,
            "recruited_from": self.recruited_from,
            "grade": self.grade,
            "source_family": self.source,
            "provenance": Provenance.SOURCE_FETCH.value,
            "source_path": source_url,
            "source_sha256": source_sha256,
            "source_row": None,
        }
        return fit_table_row("draft_events", row)


@dataclass
class DraftParse:
    season: int
    outcome: CheckOutcome
    events: list[DraftEvent] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def expand_rows(table: Tag) -> list[list[str]]:
    """Dense grid of cell texts honouring rowspan and colspan."""
    grid: list[list[str]] = []
    pending: dict[int, tuple[str, int]] = {}
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        row: list[str] = []
        col = ci = 0
        while ci < len(cells) or any(rem > 0 for _t, rem in pending.values()):
            if col in pending and pending[col][1] > 0:
                text, rem = pending[col]
                row.append(text)
                pending[col] = (text, rem - 1)
                col += 1
                continue
            if ci >= len(cells):
                break
            cell = cells[ci]
            ci += 1
            text = cell.get_text(" ", strip=True)
            span = _first_int(str(cell.get("colspan", "1"))) or 1
            rspan = _first_int(str(cell.get("rowspan", "1"))) or 1
            for _ in range(span):
                row.append(text)
                if rspan > 1:
                    pending[col] = (text, rspan - 1)
                col += 1
        grid.append(row)
    return grid


def _idx(headers: list[str], *names: str) -> int | None:
    low = [h.lower().strip() for h in headers]
    for n in names:
        if n in low:
            return low.index(n)
    return None


def _classify(heading: str) -> str | None:
    t = heading.lower()
    if "national draft" in t:
        return "national"
    if "rookie" in t:
        return "rookie_mid_season" if ("mid-season" in t or "mid season" in t) else "rookie_end_season"
    return None


def parse_wikipedia_draft(content: bytes | str, *, season: int) -> DraftParse:
    out = DraftParse(season=season, outcome=CheckOutcome.FAIL)
    soup = BeautifulSoup(content.decode("utf-8", "replace") if isinstance(content, bytes) else content, "html.parser")
    found_national = False
    for table in soup.find_all("table"):
        grid = expand_rows(table)
        if not grid:
            continue
        headers = grid[0]
        pick_i, player_i = _idx(headers, "pick", "#"), _idx(headers, "player")
        if pick_i is None or player_i is None:
            continue
        heading = table.find_previous(["h2", "h3", "h4"])
        kind = _classify(heading.get_text(" ", strip=True)) if heading else None
        if kind is None:
            continue
        found_national = found_national or kind == "national"
        round_i = _idx(headers, "round", "rd.", "rd")
        club_i = _idx(headers, "club", "recruited to", "drafted to", "recruited by")
        from_i = _idx(headers, "recruited from")
        for row in grid[1:]:
            if len(row) <= max(pick_i, player_i):
                continue
            pick = _first_int(row[pick_i])
            name = clean_name(row[player_i])
            if pick is None or not name:
                continue  # separators / "Pass" rows
            out.events.append(
                DraftEvent(
                    season=season,
                    event_type=kind,
                    draft_round=_first_int(row[round_i]) if round_i is not None and round_i < len(row) else None,
                    pick=pick,
                    club=clean_name(row[club_i]) or None if club_i is not None and club_i < len(row) else None,
                    player_name=name,
                    recruited_from=clean_name(row[from_i]) or None
                    if from_i is not None and from_i < len(row)
                    else None,
                    source="wikipedia",
                )
            )
    if not found_national:
        out.issues.append("no national draft section/table found")
    if not out.events:
        out.issues.append("no draft rows parsed")
    out.outcome = CheckOutcome.FAIL if out.issues else CheckOutcome.PASS
    return out


def parse_draftguru(content: bytes | str, *, season: int) -> DraftParse:
    out = DraftParse(season=season, outcome=CheckOutcome.FAIL)
    soup = BeautifulSoup(content.decode("utf-8", "replace") if isinstance(content, bytes) else content, "html.parser")
    for table in soup.find_all("table"):
        grid = expand_rows(table)
        if not grid:
            continue
        headers = grid[0]
        draft_i, player_i = _idx(headers, "draft"), _idx(headers, "player")
        hash_i = next((i for i, h in enumerate(headers) if h.strip().startswith("#")), None)
        if draft_i is None or player_i is None or hash_i is None:
            continue
        club_i = _idx(headers, "club")
        orig_i, grade_i, games_i = _idx(headers, "original club"), _idx(headers, "grade"), _idx(headers, "games")
        for row in grid[1:]:
            if len(row) != len(headers):
                continue
            name = clean_name(row[player_i])
            if not name:
                continue
            grade = row[grade_i].strip() if grade_i is not None else ""
            out.events.append(
                DraftEvent(
                    season=season,
                    event_type=_slug(row[draft_i]) or "unknown",
                    draft_round=None,
                    pick=_first_int(row[hash_i]),
                    club=clean_name(row[club_i]) or None if club_i is not None else None,
                    player_name=name,
                    recruited_from=clean_name(row[orig_i]) or None if orig_i is not None else None,
                    grade=None if grade.lower() in _EMPTY else grade,
                    games=_first_int(row[games_i]) if games_i is not None else None,
                    source="draftguru",
                )
            )
        break
    if not out.events:
        out.issues.append("no DraftGuru year table with Draft/#/Player headers")
    out.outcome = CheckOutcome.FAIL if out.issues else CheckOutcome.PASS
    return out


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
