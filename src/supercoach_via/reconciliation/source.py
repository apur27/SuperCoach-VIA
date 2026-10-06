"""Independent readers for captured AFL Tables pages (comparison side; DESIGN sections 3, 8).

Built on the existing independent tokenizer in ``integrity.sourcepages`` and nothing from the
production importer. Every cell keeps its original text; header labels are validated, and a
duplicated label, unknown numeric label, rowspan inside a data table or short/long row is
recorded as a problem (never silently resolved by position or by keeping the last value, S-10).

Facts are strict frozen models so they can be cached as JSON and replayed byte-for-byte.
Cell tuples are aligned with ``schema.STAT_FIELDS`` (23 entries); ``None`` marks a statistic
whose column the page does not have, ``""`` a printed blank.
"""

from __future__ import annotations

import html
import re
from datetime import date

from supercoach_via.integrity.sourcepages import Cell, Table, expand, read_tables
from supercoach_via.reconciliation import discover as D
from supercoach_via.reconciliation.schema import LABEL_TO_FIELD, STAT_FIELDS, STAT_LABELS, SUMMARY_LABELS, Model
from supercoach_via.reconciliation.urls import normalise_link

_N = len(STAT_FIELDS)
_MONTHS = {
    m: i for i, m in enumerate(("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), 1)
}
_BORN = re.compile(rb"<b>Born:</b>\s*(\d{1,2})-([A-Z][a-z]{2})-(\d{4})", re.I)
_BORN_ANY = re.compile(rb"<b>Born:</b>\s*([^<(]*)", re.I)
_SEASON_HEAD = re.compile(r"^(.+) - (\d{4})$")
_GM_WDL = re.compile(r"^(\d+) \((\d+)-(\d+)-(\d+)\)$")
_CAREER_GAMES = re.compile(r"^(\d+) \((\d+)-(\d+)-(\d+)(?: [0-9.]+%)?\)$")
_WDL = re.compile(r"^(\d+)-(\d+)-(\d+)$")
_DIGITS = re.compile(r"^\d+")
FIXED_GAME_LABELS = ["Gm", "Opponent", "Rd", "R", "#"]
SUMMARY_HEAD = ["Year", "Team", "#", "GM", "W-D-L", *SUMMARY_LABELS]


class SourceCell(Model):
    """Not used for storage: documents the cell contract (raw text, ``None`` = column absent)."""


def _row_cells(labels: list[str], cells: list[Cell]) -> tuple[str | None, ...]:
    """Raw texts in canonical order for the labels present; absent columns are ``None``."""
    by_label = {lab: c.text for lab, c in zip(labels, cells, strict=True) if lab in LABEL_TO_FIELD}
    return tuple(by_label.get(lab) for lab, _ in STAT_LABELS)


def _dups(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for lab in labels:
        if lab in seen and lab not in out:
            out.append(lab)
        seen.add(lab)
    return out


def _has_rowspan(table: Table) -> bool:
    return any(c.rowspan > 1 for row in table.rows for c in row)


def counter_of(token: str) -> int | None:
    m = _DIGITS.match(token.strip())
    return int(m.group(0)) if m else None


def parse_date_text(text: str) -> date | None:
    m = re.search(r"(\d{1,2})-([A-Z][a-z]{2})-(\d{4})", text)
    if not m or m.group(2) not in _MONTHS:
        return None
    try:
        return date(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Player profile
# ---------------------------------------------------------------------------


class ProfileGame(Model):
    club: str
    season: int
    table: int
    row: int
    counter_token: str
    counter: int | None
    opponent: str
    rd_token: str
    match_url: str | None
    result: str
    jersey_token: str
    cells: tuple[str | None, ...]
    #: the row (or its whole table) could not be read reliably
    malformed: bool = False


class GameFoot(Model):
    gm: int | None
    wdl: tuple[int, int, int] | None
    text: str
    cells: tuple[str | None, ...]


class ProfileClubSeason(Model):
    club: str
    season: int
    table: int
    labels: tuple[str, ...]
    #: canonical fields whose column is duplicated or otherwise unusable in this table
    bad_fields: tuple[str, ...]
    malformed: bool
    games: int
    foot: GameFoot | None


class SummaryRow(Model):
    label: str
    year: int | None
    club: str
    jersey_token: str
    gm_text: str
    wdl_text: str
    cells: tuple[str | None, ...]


class ProfileFacts(Model):
    url: str
    h1: str | None
    title: str | None
    born_raw: str | None
    born: str | None
    games: tuple[ProfileGame, ...]
    club_seasons: tuple[ProfileClubSeason, ...]
    season_totals: tuple[SummaryRow, ...]
    season_averages: tuple[SummaryRow, ...]
    totals_foot: SummaryRow | None
    averages_foot: SummaryRow | None
    problems: tuple[str, ...]
    #: the career counter, read in printed order, is not 1, 2, 3, ... (gaps, duplicates, reordering, unreadable)
    counter_issues: tuple[str, ...] = ()


def counter_sequence_issues(games: tuple[ProfileGame, ...] | list[ProfileGame], *, limit: int = 20) -> tuple[str, ...]:
    """Independent check of the profile's career counter (DESIGN section 7): the rows, in printed order, must carry
    1, 2, 3, ... Each deviation is one ``kind:detail`` string; at most ``limit`` are listed, then a ``more:n`` entry."""
    issues: list[str] = []
    prev = 0
    for g in games:
        if g.malformed:
            continue
        c = g.counter
        where = f"{g.club} {g.season} row {g.row}"
        if c is None:
            issues.append(f"unreadable:{g.counter_token!r} at {where}")
            continue
        if c == prev:
            issues.append(f"duplicate:{c} at {where}")
        elif c < prev:
            issues.append(f"out_of_order:{c} after {prev} at {where}")
            prev = c
        elif c > prev + 1:
            missing = f"{prev + 1}" if c == prev + 2 else f"{prev + 1}-{c - 1}"
            issues.append(f"gap:{missing} missing before {c} at {where}")
            prev = c
        else:
            prev = c
    if len(issues) > limit:
        return (*issues[:limit], f"more:{len(issues) - limit}")
    return tuple(issues)


def _summary_rows(
    table: Table, labels: list[str], kind: str, problems: list[str]
) -> tuple[list[SummaryRow], list[SummaryRow]]:
    body: list[SummaryRow] = []
    foot: list[SummaryRow] = []
    for row, section in zip(table.rows, table.sections, strict=True):
        if section not in ("body", "foot") or not row:
            continue
        cells = expand(row)
        if len(cells) != len(labels):
            problems.append(f"SUMMARY_ROW_SHAPE: {kind} row has {len(cells)} cells for {len(labels)} columns")
            continue
        by = {lab: c.text for lab, c in zip(labels[5:], cells[5:], strict=True)}
        stat_cells = tuple(by.get(lab) for lab, _ in STAT_LABELS)
        if section == "foot":
            foot.append(
                SummaryRow(
                    label=cells[0].text,
                    year=None,
                    club="",
                    jersey_token="",
                    gm_text=cells[3].text,
                    wdl_text=cells[4].text,
                    cells=stat_cells,
                )
            )
        else:
            year = counter_of(cells[0].text)
            body.append(
                SummaryRow(
                    label=kind,
                    year=year,
                    club=cells[1].text,
                    jersey_token=cells[2].text,
                    gm_text=cells[3].text,
                    wdl_text=cells[4].text,
                    cells=stat_cells,
                )
            )
    return body, foot


def read_profile(content: bytes, url: str) -> ProfileFacts:
    problems: list[str] = []
    tables = read_tables(content)
    h1 = D.page_h1(content)
    title = D.page_title(content)
    born_raw: str | None = None
    born_iso: str | None = None
    m = _BORN.search(content)
    if m and m.group(2).decode() in _MONTHS:
        day, mon, year = m.group(1).decode(), m.group(2).decode(), m.group(3).decode()
        try:
            born_iso = date(int(year), _MONTHS[mon], int(day)).isoformat()
        except ValueError:
            problems.append("BORN_INVALID: born date is not a calendar date")
        born_raw = f"{day}-{mon}-{year}"
    else:
        m2 = _BORN_ANY.search(content)
        if m2:
            born_raw = html.unescape(m2.group(1).decode("utf-8", "replace")).strip()
            problems.append(f"BORN_UNPARSED: {born_raw[:30]!r}")
    games: list[ProfileGame] = []
    seasons: list[ProfileClubSeason] = []
    totals: list[SummaryRow] = []
    averages: list[SummaryRow] = []
    totals_foot: SummaryRow | None = None
    averages_foot: SummaryRow | None = None
    summary_seen = 0
    for ti, t in enumerate(tables):
        if not t.rows or not t.rows[0]:
            continue
        head_rows = [r for r, s in zip(t.rows, t.sections, strict=True) if s == "head" and r]
        first = head_rows[0] if head_rows else t.rows[0]
        if first[0].text == "Year" and len(first) == len(SUMMARY_HEAD):
            labels = [c.text for c in first]
            if labels != SUMMARY_HEAD:
                problems.append(f"SUMMARY_LABELS: table {ti} header {labels[:8]}")
                continue
            summary_seen += 1
            if summary_seen > 2:
                continue
            kind = "totals" if summary_seen == 1 else "averages"
            body, foot = _summary_rows(t, labels, kind, problems)
            if kind == "totals":
                totals = body
                totals_foot = next((f for f in foot if f.label == "Totals"), None)
                averages_foot = next((f for f in foot if f.label == "Averages"), None)
            else:
                averages = body
            continue
        if len(head_rows) == 2 and len(first) == 1:
            hm = _SEASON_HEAD.match(first[0].text)
            if hm is None:
                continue
            club, season = hm.group(1), int(hm.group(2))
            labels = [c.text for c in expand(head_rows[1])]
            if labels[:5] != FIXED_GAME_LABELS:
                problems.append(f"GAME_LABELS: {club} {season} header starts {labels[:5]}")
                continue
            stat_labels = labels[5:]
            where = f"{club} {season}"
            bad: list[str] = []
            malformed_table = False
            for lab in stat_labels:
                if lab not in LABEL_TO_FIELD:
                    problems.append(f"UNKNOWN_LABEL: {where}: {lab!r} is not a supported statistic column")
            for lab in _dups(stat_labels):
                problems.append(f"DUPLICATE_LABEL: {where}: {lab!r} appears more than once")
                if lab in LABEL_TO_FIELD:
                    bad.append(LABEL_TO_FIELD[lab])
            if first[0].colspan != len(labels):
                problems.append(f"COLSPAN: {where}: heading spans {first[0].colspan} for {len(labels)} columns")
            if _has_rowspan(t):
                problems.append(f"ROWSPAN: {where}: rowspan inside a data table")
                malformed_table = True
            row_no = 0
            game_foot: GameFoot | None = None
            for row, sec in zip(t.rows, t.sections, strict=True):
                if sec == "foot" and row:
                    cells = expand(row)
                    if len(cells) == len(labels) and cells[1].text == "Totals":
                        gm_text = cells[2].text
                        mm = _GM_WDL.match(gm_text)
                        game_foot = GameFoot(
                            gm=int(mm.group(1)) if mm else None,
                            wdl=(int(mm.group(2)), int(mm.group(3)), int(mm.group(4))) if mm else None,
                            text=gm_text,
                            cells=_row_cells(stat_labels, cells[5:]),
                        )
                        if mm is None:
                            problems.append(f"FOOTER_FORMAT: {where}: games cell {gm_text!r}")
                    else:
                        problems.append(f"FOOTER_SHAPE: {where}: footer row not recognised")
                    continue
                if sec != "body" or not row:
                    continue
                row_no += 1
                cells = expand(row)
                if len(cells) != len(labels):
                    problems.append(f"ROW_SHAPE: {where} row {row_no} has {len(cells)} cells for {len(labels)} columns")
                    games.append(
                        ProfileGame(
                            club=club,
                            season=season,
                            table=ti,
                            row=row_no,
                            counter_token="",
                            counter=None,
                            opponent="",
                            rd_token="",
                            match_url=None,
                            result="",
                            jersey_token="",
                            cells=(None,) * _N,
                            malformed=True,
                        )
                    )
                    continue
                rd_cell = cells[2]
                links = rd_cell.links
                if len(links) != 1:
                    problems.append(f"MATCH_LINK: {where} row {row_no} has {len(links)} match links")
                match_url = normalise_link(links[0], url) if len(links) == 1 else None
                games.append(
                    ProfileGame(
                        club=club,
                        season=season,
                        table=ti,
                        row=row_no,
                        counter_token=cells[0].text,
                        counter=counter_of(cells[0].text),
                        opponent=cells[1].text,
                        rd_token=rd_cell.text,
                        match_url=match_url,
                        result=cells[3].text,
                        jersey_token=cells[4].text,
                        cells=_row_cells(stat_labels, cells[5:]),
                        malformed=malformed_table,
                    )
                )
            seasons.append(
                ProfileClubSeason(
                    club=club,
                    season=season,
                    table=ti,
                    labels=tuple(stat_labels),
                    bad_fields=tuple(sorted(set(bad))),
                    malformed=malformed_table,
                    games=row_no,
                    foot=game_foot,
                )
            )
    if not games and not problems:
        problems.append("NO_GAME_TABLES: no per-game table recognised")
    return ProfileFacts(
        url=url,
        h1=h1,
        title=title,
        born_raw=born_raw,
        born=born_iso,
        games=tuple(games),
        club_seasons=tuple(seasons),
        season_totals=tuple(totals),
        season_averages=tuple(averages),
        totals_foot=totals_foot,
        averages_foot=averages_foot,
        problems=tuple(problems),
        counter_issues=counter_sequence_issues(games),
    )


# ---------------------------------------------------------------------------
# Match page
# ---------------------------------------------------------------------------


class MatchPlayer(Model):
    team: str
    link: str | None
    name: str
    jersey_token: str
    row: int
    cells: tuple[str | None, ...]


class PlayerDetail(Model):
    """One row of a ``<Team> Player Details`` table: the career games the page prints up to and including this match."""

    team: str
    link: str | None
    name: str
    jersey_token: str
    row: int
    #: leading integer of ``Career Games (W-D-L W%)``; ``None`` when blank or unreadable
    career_games: int | None
    career_text: str


class MatchTeam(Model):
    name: str
    link: str | None
    #: cumulative (goals, behinds) at q1, q2, q3, full time and, when played, extra time: the last is the result
    quarters: tuple[tuple[int, int], ...]
    points: tuple[int, ...]


class MatchFacts(Model):
    url: str
    title: str | None
    stage_text: str | None
    venue: str | None
    match_date: str | None
    local_start: str | None
    attendance: int | None
    teams: tuple[MatchTeam, ...]
    players: tuple[MatchPlayer, ...]
    totals: dict[str, tuple[str | None, ...]]
    rushed: dict[str, int | None]
    columns: dict[str, tuple[str, ...]]
    #: team -> canonical fields whose column is duplicated; team table malformed
    bad_fields: dict[str, tuple[str, ...]]
    bad_tables: tuple[str, ...]
    problems: tuple[str, ...]
    player_details: tuple[PlayerDetail, ...] = ()


_GBP = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def read_match(content: bytes, url: str) -> MatchFacts:
    from supercoach_via.integrity.sourcepages import _local_start  # the shared date-time grammar

    problems: list[str] = []
    tables = read_tables(content)
    title = D.page_title(content)
    header = next((t for t in tables if t.rows and any("Round:" in c.text for c in t.rows[0])), None)
    stage = venue = None
    mdate: str | None = None
    start: str | None = None
    attendance: int | None = None
    teams: list[MatchTeam] = []
    if header is None:
        problems.append("HEADER_MISSING: no match header table")
    else:
        head_text = next(c.text for c in header.rows[0] if "Round:" in c.text)
        m = re.search(r"Round:\s*(.+?)\s+Venue:\s*(.+?)\s+Date:\s*(.+?)(?:\s+Attendance:|$)", head_text)
        if not m:
            problems.append("HEADER_FORMAT: match header not recognised")
        else:
            stage, venue = m.group(1).strip(), m.group(2).strip()
            d, start = _local_start(m.group(3))
            mdate = d.isoformat() if d else None
            if d is None:
                problems.append(f"DATE_FORMAT: {m.group(3)[:40]!r}")
        att = re.search(r"Attendance:\s*([\d,]+)", head_text)
        attendance = int(att.group(1).replace(",", "")) if att else None
        for row in header.rows[1:]:
            if len(row) < 5 or not any("/teams/" in h for h in row[0].links):
                continue
            qs: list[tuple[int, int]] = []
            pts: list[int] = []
            for c in row[1:]:  # four quarters, plus an extra-time cell when the match went to extra time
                q = _GBP.match(c.text.replace(" ", ""))
                if q is None:
                    break
                qs.append((int(q.group(1)), int(q.group(2))))
                pts.append(int(q.group(3)))
            if len(qs) < 4:
                problems.append(f"SCORE_FORMAT: team row {row[0].text!r}")
                continue
            teams.append(
                MatchTeam(
                    name=row[0].text, link=normalise_link(row[0].links[0], url), quarters=tuple(qs), points=tuple(pts)
                )
            )
        if len(teams) != 2:
            problems.append(f"TEAM_LINES: expected two team score lines, found {len(teams)}")
    players: list[MatchPlayer] = []
    totals: dict[str, tuple[str | None, ...]] = {}
    rushed: dict[str, int | None] = {}
    columns: dict[str, tuple[str, ...]] = {}
    bad_fields: dict[str, tuple[str, ...]] = {}
    bad_tables: list[str] = []
    for t in tables:
        if "sortable" not in t.classes or not t.rows or not t.rows[0] or "Match Statistics" not in t.rows[0][0].text:
            continue
        team = t.rows[0][0].text.split(" Match Statistics", 1)[0].strip()
        hdr_i = next((i for i, r in enumerate(t.rows) if r and r[0].tag == "th" and r[0].text == "#"), None)
        if hdr_i is None:
            problems.append(f"HEADER_COLUMNS: {team}: statistics header missing")
            bad_tables.append(team)
            continue
        labels = [c.text for c in expand(t.rows[hdr_i])]
        columns[team] = tuple(labels)
        if labels[:2] != ["#", "Player"]:
            problems.append(f"HEADER_COLUMNS: {team}: starts {labels[:2]!r}")
            bad_tables.append(team)
            continue
        stat_labels = labels[2:]
        for lab in stat_labels:
            if lab not in LABEL_TO_FIELD and lab != "SU":
                problems.append(f"UNKNOWN_LABEL: {team}: {lab!r} is not a supported statistic column")
        dup = [lab for lab in _dups(stat_labels) if lab in LABEL_TO_FIELD]
        for lab in dup:
            problems.append(f"DUPLICATE_LABEL: {team}: {lab!r} appears more than once")
        if dup:
            bad_fields[team] = tuple(sorted(LABEL_TO_FIELD[x] for x in dup))
        if _has_rowspan(t):
            problems.append(f"ROWSPAN: {team}: rowspan inside a data table")
            bad_tables.append(team)
        row_no = 0
        for r, section in zip(t.rows[hdr_i + 1 :], t.sections[hdr_i + 1 :], strict=True):
            cells = expand(r)
            label = cells[0].text if cells else ""
            if section == "foot" or label in ("Rushed", "Totals", "Opposition"):
                if label == "Rushed" and len(cells) == len(labels):
                    bh = labels.index("BH") if "BH" in labels else None
                    txt = cells[bh].text if bh is not None else ""
                    rushed[team] = int(txt) if txt.isdigit() else None
                elif label == "Totals" and len(cells) == len(labels):
                    totals[team] = _row_cells(labels[2:], cells[2:])
                continue
            row_no += 1
            if len(cells) != len(labels):
                problems.append(f"ROW_SHAPE: {team} row {row_no} has {len(cells)} cells for {len(labels)} columns")
                bad_tables.append(team)
                continue
            players.append(
                MatchPlayer(
                    team=team,
                    link=normalise_link(cells[1].links[0], url) if cells[1].links else None,
                    name=cells[1].text,
                    jersey_token=cells[0].text,
                    row=row_no,
                    cells=_row_cells(labels[2:], cells[2:]),
                )
            )
    if not columns:
        problems.append("NO_TEAM_TABLES: no team statistics tables")
    details = _read_player_details(tables, url, problems)
    return MatchFacts(
        url=url,
        title=title,
        stage_text=stage,
        venue=venue,
        match_date=mdate,
        local_start=start,
        attendance=attendance,
        teams=tuple(teams),
        players=tuple(players),
        totals=totals,
        rushed=rushed,
        columns=columns,
        bad_fields=bad_fields,
        bad_tables=tuple(sorted(set(bad_tables))),
        problems=tuple(problems),
        player_details=tuple(details),
    )


def _read_player_details(tables: list[Table], url: str, problems: list[str]) -> list[PlayerDetail]:
    """Every ``<Team> Player Details`` table: the career games the match page prints for each person (DESIGN S-07)."""
    out: list[PlayerDetail] = []
    seen = 0
    for t in tables:
        head = t.rows[0] if t.rows else None
        if "sortable" not in t.classes or not head or not head[0].text.endswith(" Player Details"):
            continue
        seen += 1
        team = t.rows[0][0].text[: -len(" Player Details")].strip()
        hdr_i = next((i for i, r in enumerate(t.rows) if r and r[0].tag == "th" and r[0].text == "#"), None)
        if hdr_i is None:
            problems.append(f"PLAYER_DETAILS_COLUMNS: {team}: header row missing")
            continue
        labels = [c.text for c in expand(t.rows[hdr_i])]
        col = next((i for i, lab in enumerate(labels) if lab.startswith("Career Games")), None)
        if labels[:2] != ["#", "Player"] or col is None:
            problems.append(f"PLAYER_DETAILS_COLUMNS: {team}: no 'Career Games' column in {labels[:4]!r}")
            continue
        row_no = 0
        for r in t.rows[hdr_i + 1 :]:
            cells = expand(r)
            if not cells or len(cells) <= col:
                continue
            row_no += 1
            text = cells[col].text.strip()
            m = _CAREER_GAMES.match(text)
            games = int(m.group(1)) if m else None
            link = normalise_link(cells[1].links[0], url) if cells[1].links else None
            if text and m is None:
                problems.append(f"PLAYER_DETAILS_FORMAT: {team} row {row_no}: games cell {text!r}")
            out.append(
                PlayerDetail(
                    team=team,
                    link=link,
                    name=cells[1].text,
                    jersey_token=cells[0].text,
                    row=row_no,
                    career_games=games,
                    career_text=text,
                )
            )
    if seen == 0:
        problems.append("PLAYER_DETAILS_MISSING: no Player Details table on the match page")
    return out


# ---------------------------------------------------------------------------
# Notes page: which statistics exist for which seasons, and the documented exceptions
# ---------------------------------------------------------------------------

NOTES_CATEGORIES = {
    "Missing all but goals": "all_but_goals",
    "Missing hitouts": "hitouts",
    "Missing behinds": "behinds",
    "Missing free kicks for": "frees_for",
    "Missing free kicks against": "frees_against",
}


class NotesException(Model):
    category: str
    season: int
    round_lo: int
    round_hi: int
    #: "all" | "teams" | "matchups"
    scope: str
    teams: tuple[str, ...]
    matchups: tuple[tuple[str, str], ...]
    raw: str


class NotesFacts(Model):
    #: season -> canonical fields marked available in the notes table (``disposals`` follows kicks+handballs)
    availability: dict[int, tuple[str, ...]]
    exceptions: tuple[NotesException, ...]
    problems: tuple[str, ...]


def read_notes(content: bytes, notes_labels: dict[str, str] | None = None) -> NotesFacts:
    aliases = {"I5": "IF", "OP": "1%", **(notes_labels or {})}
    problems: list[str] = []
    availability: dict[int, tuple[str, ...]] = {}
    exceptions: list[NotesException] = []
    tables = read_tables(content)
    matrix = next((t for t in tables if t.rows and t.rows[0] and t.rows[0][0].text == "Year"), None)
    if matrix is None:
        problems.append("NOTES_MATRIX: availability table not found")
    else:
        labels = [aliases.get(c.text, c.text) for c in matrix.rows[0]]
        for lab in labels[1:]:
            if lab not in LABEL_TO_FIELD and lab not in ("SU", "DI"):
                problems.append(f"NOTES_LABEL: {lab!r} is not a known statistic column")
        for row in matrix.rows[1:]:
            if len(row) != len(labels) or not row[0].text.strip().isdigit():
                problems.append("NOTES_ROW: availability row not recognised")
                continue
            have = {
                LABEL_TO_FIELD[lab]
                for lab, c in zip(labels[1:], row[1:], strict=True)
                if c.text == "X" and lab in LABEL_TO_FIELD
            }
            if {"kicks", "handballs"} <= have:
                have.add("disposals")
            availability[int(row[0].text)] = tuple(f for f in STAT_FIELDS if f in have)
    exc_table = next((t for t in tables if t.rows and t.rows[0] and t.rows[0][0].text.startswith("Missing")), None)
    if exc_table is None:
        problems.append("NOTES_EXCEPTIONS: exceptions table not found")
    else:
        category: str | None = None
        for row in exc_table.rows:
            if len(row) == 1 or (row and row[0].tag == "th"):
                head = row[0].text.strip()
                category = NOTES_CATEGORIES.get(head)
                if category is None:
                    problems.append(f"NOTES_CATEGORY: {head!r} is not a known exception category")
                continue
            if len(row) != 2 or category is None:
                problems.append("NOTES_ROW: exception row not recognised")
                continue
            m = re.match(r"^\s*(\d{4})[,\s]+R0*(\d+)(?:\s*-\s*R?0*(\d+))?\s*$", row[0].text)
            if not m:
                problems.append(f"NOTES_ROUND: {row[0].text!r}")
                continue
            lo = int(m.group(2))
            hi = int(m.group(3)) if m.group(3) else lo
            right = row[1].text.strip()
            teams: tuple[str, ...] = ()
            matchups: tuple[tuple[str, str], ...] = ()
            if right.lower() == "all games":
                scope = "all"
            elif " v " in right:
                scope = "matchups"
                matchups = tuple(
                    (a.strip(), b.strip())
                    for a, b in (part.split(" v ", 1) for part in right.split(",") if " v " in part)
                )
            else:
                scope = "teams"
                teams = tuple(x.strip() for x in right.split(",") if x.strip())
            exceptions.append(
                NotesException(
                    category=category,
                    season=int(m.group(1)),
                    round_lo=lo,
                    round_hi=hi,
                    scope=scope,
                    teams=teams,
                    matchups=matchups,
                    raw=f"{row[0].text.strip()} | {right}",
                )
            )
    return NotesFacts(availability=availability, exceptions=tuple(exceptions), problems=tuple(problems))
