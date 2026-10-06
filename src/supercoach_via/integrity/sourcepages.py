"""Independent reader for archived AFL Tables pages (integrity checker only).

This module deliberately shares no parsing code with ``ingest.afltables``. It reads the
archived bytes with the standard-library HTML tokenizer, keeps every cell's text, links
and ``colspan``, and maps statistic columns through its own label table. The integrity
checker compares what this reader extracts with the canonical rows the production adapter
produced, so a mapping mistake in one of them surfaces as a disagreement instead of being
confirmed by the same code.

Nothing here performs I/O. Callers pass bytes they have already verified by SHA-256.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from html.parser import HTMLParser

#: AFL Tables column label -> canonical ``player_games`` column. Written independently of
#: ``ingest.afltables.STAT_CODE_MAP``; the unit tests assert both agree on real pages.
SOURCE_STAT_LABELS: dict[str, str] = {
    "KI": "kicks",
    "MK": "marks",
    "HB": "handballs",
    "DI": "disposals",
    "GL": "goals",
    "BH": "behinds",
    "HO": "hitouts",
    "TK": "tackles",
    "RB": "rebound_50s",
    "IF": "inside_50s",
    "CL": "clearances",
    "CG": "clangers",
    "FF": "frees_for",
    "FA": "frees_against",
    "BR": "brownlow_votes",
    "CP": "contested_possessions",
    "UP": "uncontested_possessions",
    "CM": "contested_marks",
    "MI": "marks_inside_50",
    "1%": "one_percenters",
    "BO": "bounces",
    "GA": "goal_assists",
    "%P": "time_on_ground_pct",
}
FINAL_NAMES = (
    "Wildcard Final",
    "Qualifying Final",
    "Elimination Final",
    "Semi Final",
    "Preliminary Final",
    "Grand Final",
)
_SECTIONS = {"thead": "head", "tbody": "body", "tfoot": "foot"}
_WS = re.compile(r"\s+")
_GBP = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_GB = re.compile(r"^(\d+)\.(\d+)$")
_MONTHS = {
    m: i for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)
}
_DATE = re.compile(r"(\d{1,2})-([A-Z][a-z]{2})-(\d{4})(?:\s+(\d{1,2}):(\d{2})\s*([AP]M))?")


@dataclass
class Cell:
    tag: str
    text: str
    links: list[str]
    colspan: int
    bold: bool
    #: appended last with a default so every existing construction is unchanged; the
    #: reconciliation reader treats a rowspan inside a data table as malformed (S-10)
    rowspan: int = 1


@dataclass
class Table:
    classes: tuple[str, ...]
    rows: list[list[Cell]]
    #: section of each row: "head", "body", "foot" or "" when the table has no sections
    sections: list[str]


class _TableReader(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[Table] = []
        self._stack: list[Table] = []
        self._section: list[str] = []
        self._cell: Cell | None = None
        self._cell_parts: list[str] = []
        self._bold = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: (v or "") for k, v in attrs}
        if tag == "table":
            self._close_cell()
            t = Table(classes=tuple(a.get("class", "").split()), rows=[], sections=[])
            self.tables.append(t)
            self._stack.append(t)
            self._section.append("")
        elif not self._stack:
            return
        elif tag in _SECTIONS:
            self._section[-1] = _SECTIONS[tag]
        elif tag == "tr":
            self._close_cell()
            self._stack[-1].rows.append([])
            self._stack[-1].sections.append(self._section[-1])
        elif tag in ("td", "th"):
            self._close_cell()
            if not self._stack[-1].rows:
                self._stack[-1].rows.append([])
                self._stack[-1].sections.append(self._section[-1])
            try:
                span = max(1, int(a.get("colspan") or 1))
            except ValueError:
                span = 1
            try:
                rspan = max(1, int(a.get("rowspan") or 1))
            except ValueError:
                rspan = 1
            self._cell = Cell(tag=tag, text="", links=[], colspan=span, bold=False, rowspan=rspan)
            self._cell_parts = []
        elif tag == "a" and self._cell is not None and a.get("href"):
            self._cell.links.append(a["href"])
        elif tag == "b" and self._cell is not None:
            self._bold += 1
            self._cell.bold = True
        elif tag == "br" and self._cell is not None:
            self._cell_parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if not self._stack:
            return
        if tag in ("td", "th", "tr"):
            self._close_cell()
        elif tag in _SECTIONS:
            self._section[-1] = ""
        elif tag == "table":
            self._close_cell()
            self._stack.pop()
            self._section.pop()

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell_parts.append(data)

    def _close_cell(self) -> None:
        if self._cell is None or not self._stack:
            self._cell = None
            return
        self._cell.text = _WS.sub(" ", "".join(self._cell_parts).replace("\xa0", " ")).strip()
        self._stack[-1].rows[-1].append(self._cell)
        self._cell = None
        self._cell_parts = []


def read_tables(content: bytes) -> list[Table]:
    """Every ``<table>`` in document order; nested tables are separate entries."""
    reader = _TableReader()
    reader.feed(content.decode("utf-8", errors="replace"))
    reader.close()
    return reader.tables


def expand(row: list[Cell]) -> list[Cell]:
    """Repeat each cell ``colspan`` times so cells align with header positions."""
    out: list[Cell] = []
    for c in row:
        out.extend([c] * c.colspan)
    return out


def _local_start(text: str) -> tuple[date | None, str | None]:
    m = _DATE.search(text)
    if not m or m.group(2) not in _MONTHS:
        return None, None
    day = date(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))
    if m.group(4) is None:
        return day, None
    hour = int(m.group(4)) % 12 + (12 if m.group(6) == "PM" else 0)
    return day, datetime(day.year, day.month, day.day, hour, int(m.group(5))).strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# Match page
# ---------------------------------------------------------------------------


@dataclass
class SourceTeamLine:
    name: str
    link: str | None
    #: cumulative (goals, behinds) at q1, q2, q3 and final, as printed
    quarters: list[tuple[int, int]]
    #: points printed beside each quarter
    points: list[int]


@dataclass
class SourcePlayer:
    team: str
    link: str | None
    name: str
    jersey_token: str
    #: canonical stat -> raw cell text ("" = blank)
    cells: dict[str, str]


@dataclass
class SourceMatch:
    stage_text: str | None = None
    venue: str | None = None
    match_date: date | None = None
    local_start: str | None = None
    attendance: int | None = None
    teams: list[SourceTeamLine] = field(default_factory=list)
    players: list[SourcePlayer] = field(default_factory=list)
    #: team -> canonical stat -> totals-row text
    totals: dict[str, dict[str, str]] = field(default_factory=dict)
    #: team -> rushed behinds, when the footer states them
    rushed: dict[str, int | None] = field(default_factory=dict)
    #: team -> labels present in that table's header (source order)
    columns: dict[str, list[str]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def read_match_page(content: bytes) -> SourceMatch:
    out = SourceMatch()
    tables = read_tables(content)
    header = next((t for t in tables if t.rows and any("Round:" in c.text for c in t.rows[0])), None)
    if header is None:
        out.problems.append("no match header table")
        return out
    # the header row also holds navigation-arrow cells; read only the cell that states the match
    head_text = next(c.text for c in header.rows[0] if "Round:" in c.text)
    m = re.search(r"Round:\s*(.+?)\s+Venue:\s*(.+?)\s+Date:\s*(.+?)(?:\s+Attendance:|$)", head_text)
    if not m:
        out.problems.append("match header not recognised")
        return out
    out.stage_text = m.group(1).strip()
    out.venue = m.group(2).strip()
    out.match_date, out.local_start = _local_start(m.group(3))
    att = re.search(r"Attendance:\s*([\d,]+)", head_text)
    out.attendance = int(att.group(1).replace(",", "")) if att else None
    for row in header.rows[1:]:
        if len(row) < 5 or not any("/teams/" in h for h in row[0].links):
            continue
        quarters: list[tuple[int, int]] = []
        points: list[int] = []
        for c in row[1:5]:
            q = _GBP.match(c.text.replace(" ", ""))
            if q is None:
                break
            quarters.append((int(q.group(1)), int(q.group(2))))
            points.append(int(q.group(3)))
        if len(quarters) != 4:
            out.problems.append(f"team row {row[0].text!r}: quarter cells not recognised")
            continue
        out.teams.append(SourceTeamLine(row[0].text, row[0].links[0], quarters, points))
    if len(out.teams) != 2:
        out.problems.append(f"expected two team score lines, found {len(out.teams)}")
    for t in tables:
        if "sortable" not in t.classes or not t.rows or "Match Statistics" not in t.rows[0][0].text:
            continue
        team = t.rows[0][0].text.split(" Match Statistics", 1)[0].strip()
        hdr_i = next((i for i, r in enumerate(t.rows) if r and r[0].tag == "th" and r[0].text == "#"), None)
        if hdr_i is None:
            out.problems.append(f"{team}: statistics header missing")
            continue
        labels = [c.text for c in expand(t.rows[hdr_i])]
        out.columns[team] = labels
        if labels[:2] != ["#", "Player"]:
            out.problems.append(f"{team}: statistics header starts {labels[:2]!r}")
            continue
        unknown = [lab for lab in labels[2:] if lab not in SOURCE_STAT_LABELS]
        if unknown:
            out.problems.append(f"{team}: unknown statistic labels {unknown!r}")
        for r, section in zip(t.rows[hdr_i + 1 :], t.sections[hdr_i + 1 :], strict=True):
            cells = expand(r)
            if section == "foot" or (cells and cells[0].text in ("Rushed", "Totals", "Opposition")):
                label = cells[0].text if cells else ""
                if label == "Rushed" and len(cells) == len(labels):
                    bh = labels.index("BH") if "BH" in labels else None
                    txt = cells[bh].text if bh is not None else ""
                    out.rushed[team] = int(txt) if txt.isdigit() else None
                elif label == "Totals" and len(cells) == len(labels):
                    out.totals[team] = {
                        SOURCE_STAT_LABELS[lab]: cells[i].text
                        for i, lab in enumerate(labels)
                        if lab in SOURCE_STAT_LABELS
                    }
                continue
            if len(cells) != len(labels):
                out.problems.append(f"{team}: row with {len(cells)} cells, header has {len(labels)}")
                continue
            out.players.append(
                SourcePlayer(
                    team=team,
                    link=cells[1].links[0] if cells[1].links else None,
                    name=cells[1].text,
                    jersey_token=cells[0].text,
                    cells={
                        SOURCE_STAT_LABELS[lab]: cells[i].text
                        for i, lab in enumerate(labels)
                        if lab in SOURCE_STAT_LABELS
                    },
                )
            )
    if not out.columns:
        out.problems.append("no team statistics tables")
    return out


# ---------------------------------------------------------------------------
# Season page
# ---------------------------------------------------------------------------


@dataclass
class SourceFixture:
    stage_text: str
    home: str
    away: str
    home_link: str | None
    away_link: str | None
    match_date: date | None
    local_start: str | None
    venue: str | None
    attendance: int | None
    #: None when the fixture has no score yet
    home_quarters: list[tuple[int, int]] | None
    away_quarters: list[tuple[int, int]] | None
    home_points: int | None
    away_points: int | None
    game_link: str | None


@dataclass
class SourceSeason:
    fixtures: list[SourceFixture] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _heading(table: Table) -> str | None:
    if len(table.rows) != 1 or not table.rows[0]:
        return None
    first = table.rows[0][0]
    if not first.bold:
        return None
    m = re.match(r"^Round (\d+)\b", first.text)
    if m:
        return f"Round {int(m.group(1))}"
    for name in FINAL_NAMES:
        if first.text.startswith(name):
            return name
    return None


def _quarters(text: str) -> list[tuple[int, int]] | None:
    toks = text.split()
    if len(toks) != 4:
        return None
    out = []
    for tok in toks:
        m = _GB.match(tok)
        if m is None:
            return None
        out.append((int(m.group(1)), int(m.group(2))))
    return out


def read_season_page(content: bytes) -> SourceSeason:
    out = SourceSeason()
    stage: str | None = None
    for t in read_tables(content):
        h = _heading(t)
        if h is not None:
            stage = h
            continue
        rows = [r for r in t.rows if r]
        if len(rows) != 2 or not all(len(r) >= 3 and any("/teams/" in x for x in r[0].links) for r in rows):
            continue
        if stage is None:
            out.problems.append("match table before any stage heading")
            continue
        (home, away) = rows
        hq, aq = _quarters(home[1].text), _quarters(away[1].text)
        hp = int(home[2].text) if home[2].text.isdigit() else None
        ap = int(away[2].text) if away[2].text.isdigit() else None
        info = home[3].text if len(home) > 3 else ""
        day, start = _local_start(info)
        att = re.search(r"Att:\s*([\d,]+)", info)
        ven = re.search(r"Venue:\s*(.+)$", info)
        game = next((x for r in rows for c in r for x in c.links if "/stats/games/" in x), None)
        out.fixtures.append(
            SourceFixture(
                stage_text=stage,
                home=home[0].text,
                away=away[0].text,
                home_link=home[0].links[0],
                away_link=away[0].links[0],
                match_date=day,
                local_start=start,
                venue=ven.group(1).strip() if ven else None,
                attendance=int(att.group(1).replace(",", "")) if att else None,
                home_quarters=hq,
                away_quarters=aq,
                home_points=hp,
                away_points=ap,
                game_link=game,
            )
        )
    if not out.fixtures:
        out.problems.append("no fixtures recognised")
    return out


def stage_label_for(stage_text: str) -> str:
    """Canonical ``stage_label`` for a source stage heading (``Round 4`` -> ``4``)."""
    m = re.fullmatch(r"(?:Round\s+)?(\d+)", stage_text.strip())
    return str(int(m.group(1))) if m else stage_text.strip()


def cell_value(stat: str, text: str) -> int | float | str | None:
    """Raw cell -> value. Blank is None. Unparseable text is returned as the string itself
    so the comparison reports it rather than silently treating it as missing."""
    t = text.strip()
    if not t:
        return None
    try:
        return float(t) if stat == "time_on_ground_pct" else int(t)
    except ValueError:
        return t


def jersey(token: str) -> int | None:
    digits = "".join(ch for ch in token if ch.isdigit())
    return int(digits) if digits else None


def person_name(source_name: str) -> str:
    """``"Amiss, Jye"`` -> ``"Jye Amiss"``; other shapes are returned unchanged."""
    if "," in source_name:
        last, first = (p.strip() for p in source_name.split(",", 1))
        return f"{first} {last}".strip()
    return source_name.strip()


# ---------------------------------------------------------------------------
# Player pages (``/afl/stats/players/<L>/<Name>.html``): one game table per club-season
# ---------------------------------------------------------------------------

#: player-page round token for a final -> canonical ``stage_label``
FINAL_TOKENS = {
    "WF": "Wildcard Final",
    "QF": "Qualifying Final",
    "EF": "Elimination Final",
    "SF": "Semi Final",
    "PF": "Preliminary Final",
    "GF": "Grand Final",
}
_SEASON_HEAD = re.compile(r"^(.+?) - (\d{4})$")


@dataclass
class SourcePlayerGame:
    team: str
    season: int
    counter: int | None
    counter_token: str
    opponent: str
    round_token: str
    result: str
    jersey: str
    #: canonical stat -> raw cell text ("" = blank)
    cells: dict[str, str]


@dataclass
class SourcePlayerPage:
    name: str | None = None
    games: list[SourcePlayerGame] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def read_player_page(content: bytes) -> SourcePlayerPage:
    """Every game row of a player page, as raw text (no value interpretation)."""
    out = SourcePlayerPage()
    m = re.search(rb"<h1>(.*?)</h1>", content, re.S)
    out.name = _WS.sub(" ", m.group(1).decode("utf-8", "replace")).strip() if m else None
    for t in read_tables(content):
        head = [r for r, s in zip(t.rows, t.sections, strict=True) if s == "head"]
        if not head or not head[0] or head[0][0].colspan != 28:
            continue
        hm = _SEASON_HEAD.match(head[0][0].text)
        if hm is None:
            out.problems.append(f"season heading {head[0][0].text[:40]!r}")
            continue
        team, season = hm.group(1), int(hm.group(2))
        labels = [c.text for c in expand(head[-1])] if len(head) > 1 else []
        if labels[:5] != ["Gm", "Opponent", "Rd", "R", "#"] or any(x not in SOURCE_STAT_LABELS for x in labels[5:]):
            out.problems.append(f"{team} {season}: column headings {labels[:8]}")
            continue
        for row, sec in zip(t.rows, t.sections, strict=True):
            if sec != "body" or not row:
                continue
            cells = expand(row)
            if len(cells) != len(labels):
                out.problems.append(f"{team} {season}: row with {len(cells)} cells for {len(labels)} columns")
                continue
            digits = "".join(ch for ch in cells[0].text if ch.isdigit())
            out.games.append(
                SourcePlayerGame(
                    team=team,
                    season=season,
                    counter=int(digits) if digits else None,
                    counter_token=cells[0].text,
                    opponent=cells[1].text,
                    round_token=cells[2].text,
                    result=cells[3].text,
                    jersey=cells[4].text,
                    cells={SOURCE_STAT_LABELS[lab]: c.text for lab, c in zip(labels[5:], cells[5:], strict=True)},
                )
            )
    if not out.games and not out.problems:
        out.problems.append("no game tables recognised")
    return out


def stage_label_for_round(token: str) -> str:
    """Player-page ``Rd`` token -> canonical ``stage_label`` (``5`` -> ``5``, ``GF`` -> ``Grand Final``)."""
    t = token.strip()
    return FINAL_TOKENS.get(t, str(int(t)) if t.isdigit() else t)
