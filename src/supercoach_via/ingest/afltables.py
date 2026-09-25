"""Pure AFLTables adapters: season fixtures, match details and player pages.

No I/O happens here: callers fetch bytes through :class:`ingest.http.HttpClient` and
pass them in. Every parser returns an explicit ``CheckOutcome``; parser drift, empty
but expected pages and unmapped stages are ``FAIL`` with issues, never zero-row success.

Stage rules (AUDIT C03/C08, DATA_REFRESH): the season page's stage headings are the
authority. Player-page round tokens (``1``..``n``, ``WF``, ``QF``...) are resolved to a
match and its date *only* through the parsed season table (team pair + stage code);
an unresolvable row keeps an unknown date. There is no round-to-weeks approximation.

Integration notes: ``match_id`` for a source-keyed match is ``afltables:<game_id>``;
scheduled fixtures without a source key get ``afltables-fixture:<season>:<stage_id>:
<home>-<away>:<occurrence>``. Club IDs are left as source-name slugs
(``src:<team-page-slug>``) for ``domain.ids`` to bind; see the ingest report.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup, Tag

from supercoach_via.domain.schemas import (
    PLAYER_STAT_COLUMNS,
    CheckOutcome,
    DatePrecision,
    DateQuality,
    MatchStatus,
    Provenance,
    StageType,
)
from supercoach_via.domain.season import make_match_id, stage_orders
from supercoach_via.ingest.http import fit_table_row

ClubResolver = Callable[[str, int], str | None]

ADAPTER_VERSION = "1"
ORIGIN = "https://afltables.com"
SEASON_PATH_RE = re.compile(r"^/afl/seas/(?P<season>(18|19|20)\d{2})\.html$")
GAME_PATH_RE = re.compile(r"^/afl/stats/games/(?P<season>(18|19|20)\d{2})/(?P<gid>[0-9]{6,20})\.html$")
PLAYER_PATH_RE = re.compile(r"^/afl/stats/players/(?P<tok>[A-Z]/[A-Za-z0-9_\-]{1,80})\.html$")
_GAME_ID_RE = re.compile(r"^[0-9]{6,20}$")
_PLAYER_TOKEN_RE = re.compile(r"^[A-Z]/[A-Za-z0-9_\-]{1,80}$")

#: AFLTables stat column code -> canonical player_games column.
STAT_CODE_MAP: dict[str, str] = {
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
assert set(STAT_CODE_MAP.values()) == set(PLAYER_STAT_COLUMNS)

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------


def season_url(season: int) -> str:
    if not 1897 <= season <= 2100:
        raise ValueError(f"season out of range: {season}")
    return f"{ORIGIN}/afl/seas/{season}.html"


def match_url(season: int, game_id: str) -> str:
    if not _GAME_ID_RE.fullmatch(game_id):
        raise ValueError(f"invalid AFLTables game id {game_id!r}")
    season_url(season)
    return f"{ORIGIN}/afl/stats/games/{season}/{game_id}.html"


def player_url(token: str) -> str:
    """``token`` is ``<Initial>/<Page_Name>`` as discovered from a source href."""
    if not _PLAYER_TOKEN_RE.fullmatch(token):
        raise ValueError(f"invalid AFLTables player token {token!r}")
    return f"{ORIGIN}/afl/stats/players/{token}.html"


def resolve_href(href: str, page_url: str, grammar: re.Pattern[str]) -> str | None:
    """Resolve a source-controlled href against the page and accept it only on-grammar."""
    if not href or any(c in href for c in ("\\", " ", "\n", "\t")):
        return None
    absolute = urljoin(page_url, href.strip())
    parts = urlsplit(absolute)
    if parts.scheme != "https" or parts.netloc != "afltables.com" or parts.query or parts.fragment:
        return None
    if not grammar.fullmatch(parts.path):
        return None
    return f"{ORIGIN}{parts.path}"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageInfo:
    label: str  # canonical source stage label (legacy matches round_num style)
    stage_type: StageType
    round_number: int | None
    stage_order: int
    stage_id: str
    code: str  # player-page round token


_FINALS: dict[str, tuple[str, str, int]] = {
    # heading -> (code, stage_id, order)
    "wildcard final": ("WF", "wf", 1010),
    "qualifying final": ("QF", "qf", 1020),
    "elimination final": ("EF", "ef", 1020),
    "semi final": ("SF", "sf", 1030),
    "preliminary final": ("PF", "pf", 1040),
    "grand final": ("GF", "gf", 1050),
}
FINAL_CODES: dict[str, str] = {code: head.title() for head, (code, _s, _o) in _FINALS.items()}
_ROUND_RE = re.compile(r"^round:?\s*(\d{1,2})(?!\d)")


def resolve_stage(heading: str) -> StageInfo | None:
    """Map a season-page stage heading to a stage; ``None`` for anything unmapped."""
    text = " ".join(heading.replace("\xa0", " ").split()).strip()
    low = text.lower()
    m = _ROUND_RE.match(low)
    if m:
        n = int(m.group(1))
        return StageInfo(str(n), StageType.REGULAR, n, n * 10, f"r{n:02d}", str(n))
    if low.startswith("opening round"):
        return StageInfo("Opening Round", StageType.REGULAR, 0, 0, "r00", "OR")
    for head, (code, sid, order) in _FINALS.items():
        if low == head or low.startswith(head + " "):
            return StageInfo(head.title(), StageType.FINAL, None, order, sid, code)
    return None


def stage_for_token(token: str) -> tuple[str | None, str]:
    """Player-page round token -> (stage label, code). Numeric tokens are regular rounds."""
    tok = token.strip().upper()
    if tok.isdigit():
        return str(int(tok)), str(int(tok))
    if tok == "OR":
        return "Opening Round", "OR"
    return FINAL_CODES.get(tok), tok


# ---------------------------------------------------------------------------
# Shared cell parsing
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(
    r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+(\d{1,2})-([A-Za-z]{3})-(\d{4})"
    r"(?:\s+(\d{1,2}):(\d{2})\s*(AM|PM))?"
)


def _parse_date(text: str) -> tuple[str | None, date | None, DatePrecision]:
    m = _DATE_RE.search(text)
    if not m:
        return None, None, DatePrecision.UNKNOWN
    day, mon, year, hh, mm, ampm = m.groups()
    try:
        d = datetime.strptime(f"{day}-{mon}-{year}", "%d-%b-%Y").date()
    except ValueError:
        return None, None, DatePrecision.UNKNOWN
    if hh is None:
        return d.isoformat(), d, DatePrecision.DAY
    hour = int(hh) % 12 + (12 if ampm == "PM" else 0)
    return f"{d.isoformat()} {hour:02d}:{int(mm):02d}", d, DatePrecision.MINUTE


def _text(node: Tag | None) -> str:
    return " ".join(node.get_text(" ", strip=True).replace("\xa0", " ").split()) if node else ""


def _int_or_none(text: str) -> int | None:
    t = text.replace(",", "").strip()
    return int(t) if t.isdigit() else None


def _gb(token: str) -> tuple[int, int] | None:
    m = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?", token.strip())
    if not m:
        return None
    g, b = int(m.group(1)), int(m.group(2))
    if m.group(3) is not None and int(m.group(3)) != g * 6 + b:
        return None
    return g, b


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "x"


def _soup(content: bytes | str) -> BeautifulSoup:
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    return BeautifulSoup(content, "html.parser")


# ---------------------------------------------------------------------------
# Season page
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureMatch:
    season: int
    match_id: str
    source_game_id: str | None
    stage: StageInfo
    replay_occurrence: int
    home_name: str
    away_name: str
    home_club_id: str | None
    away_club_id: str | None
    home_team_slug: str
    away_team_slug: str
    venue: str | None
    local_start: str | None
    match_date: date | None
    date_precision: DatePrecision
    status: MatchStatus
    attendance: int | None
    home_quarters: tuple[tuple[int, int], ...] | None
    away_quarters: tuple[tuple[int, int], ...] | None
    home_score: int | None
    away_score: int | None
    detail_url: str | None

    @property
    def team_pair(self) -> frozenset[str]:
        return frozenset((self.home_name, self.away_name))

    def to_row(self, *, source_ref: str, source_sha256: str | None) -> dict[str, Any]:
        """Canonical ``matches`` row (club ids are unbound source slugs; see module doc)."""
        row: dict[str, Any] = {
            "match_id": self.match_id,
            "season": self.season,
            "stage_label": self.stage.label,
            "stage_type": self.stage.stage_type.value,
            "round_number": self.stage.round_number,
            "stage_order": self.stage.stage_order,
            "stage_id": self.stage.stage_id,
            "replay_occurrence": self.replay_occurrence,
            "home_club_id": self.home_club_id or f"src:{self.home_team_slug}",
            "away_club_id": self.away_club_id or f"src:{self.away_team_slug}",
            "home_source_name": self.home_name,
            "away_source_name": self.away_name,
            "venue_id": None,
            "venue_source_name": self.venue,
            "local_start": self.local_start,
            "match_date": self.match_date,
            "date_precision": self.date_precision.value,
            "status": self.status.value,
            "attendance": self.attendance,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "provenance": Provenance.SOURCE_FETCH.value,
            "source_path": self.detail_url or source_ref,
            "source_sha256": source_sha256,
            "source_row": None,
        }
        for side, qs in (("home", self.home_quarters), ("away", self.away_quarters)):
            for i, q in enumerate(("q1", "q2", "q3", "final")):
                g_b = None if qs is None else (qs[i] if q != "final" else qs[-1])
                row[f"{side}_{q}_goals"] = None if g_b is None else g_b[0]
                row[f"{side}_{q}_behinds"] = None if g_b is None else g_b[1]
        return fit_table_row("matches", row)


@dataclass
class SeasonFixture:
    season: int
    outcome: CheckOutcome
    matches: list[FixtureMatch] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    byes: int = 0
    headings: int = 0

    def completed(self) -> list[FixtureMatch]:
        return [m for m in self.matches if m.status is MatchStatus.COMPLETE]


def _team_row(tr: Tag) -> tuple[list[Tag], str, str] | None:
    cells = tr.find_all("td", recursive=False)
    if not cells:
        return None
    link = cells[0].find("a", href=True)
    href = str(link["href"]) if link else ""
    m = re.search(r"teams/([a-z0-9_]+?)(?:_idx)?\.html$", href)
    if not link or not m:
        return None
    return cells, _text(link), m.group(1)


def parse_season_page(content: bytes | str, *, season: int, club_resolver: ClubResolver | None = None) -> SeasonFixture:
    """Parse a season page. With ``club_resolver`` (e.g. ``ClubRegistry.resolve``), match IDs
    use the shared ``domain.season.make_match_id`` surrogate so refreshed rows upsert onto
    imported legacy rows; an unresolvable club is then a blocking issue."""
    page_url = season_url(season)
    fx = SeasonFixture(season=season, outcome=CheckOutcome.FAIL)
    if not content:
        fx.issues.append("empty payload")
        return fx
    soup = _soup(content)
    h1 = _text(soup.find("h1"))
    if not re.match(rf"^{season}\b", h1):
        fx.issues.append(f"page title {h1[:60]!r} is not the {season} season page")
        return fx

    stage: StageInfo | None = None
    unmapped: str | None = None
    occurrences: dict[tuple[str, frozenset[str]], int] = {}
    for table in soup.find_all("table"):
        if table.find("table") is not None or "sortable" in (table.get("class") or []):
            continue
        rows = table.find_all("tr", recursive=False) or table.find_all("tr")
        if str(table.get("border", "")) == "2":
            head = _text(rows[0].find("td")) if rows else ""
            if head.lower() == "finals":
                continue
            fx.headings += 1
            stage = resolve_stage(head)
            unmapped = None if stage else head
            if stage is None:
                fx.issues.append(f"unmapped stage heading {head[:60]!r}")
            continue
        team_rows = [r for r in (_team_row(tr) for tr in rows) if r is not None]
        if not team_rows:
            continue
        if len(team_rows) == 1 and len(team_rows[0][0]) == 2 and _text(team_rows[0][0][1]) == "Bye":
            fx.byes += 1
            continue
        if len(team_rows) != 2 or any(len(c) < 4 for c, _n, _s in team_rows):
            fx.issues.append(f"unexpected match block with {len(team_rows)} team rows")
            continue
        if stage is None:
            fx.issues.append(f"match block outside a mapped stage ({unmapped!r})")
            continue
        match = _parse_match_block(team_rows, stage, season, page_url, occurrences, fx.issues, club_resolver)
        if match is not None:
            fx.matches.append(match)

    if fx.headings == 0 or (not fx.matches and not fx.byes):
        fx.issues.append("no stage headings or match blocks found (parser drift or empty page)")
    orders = stage_orders(
        {
            sid: [m.match_date for m in fx.matches if m.stage.stage_id == sid]
            for sid in {m.stage.stage_id for m in fx.matches}
        }
    )
    fx.matches = [replace(m, stage=replace(m.stage, stage_order=orders[m.stage.stage_id])) for m in fx.matches]
    fx.outcome = CheckOutcome.FAIL if fx.issues else CheckOutcome.PASS
    return fx


def _parse_match_block(
    team_rows: list[tuple[list[Tag], str, str]],
    stage: StageInfo,
    season: int,
    page_url: str,
    occurrences: dict[tuple[str, frozenset[str]], int],
    issues: list[str],
    club_resolver: ClubResolver | None,
) -> FixtureMatch | None:
    (hc, home, hslug), (ac, away, aslug) = team_rows
    info = _text(hc[3])
    local_start, mdate, precision = _parse_date(info)
    if mdate is not None and mdate.year != season:
        issues.append(f"{home} v {away}: date {mdate} outside season {season}")
    venue_m = re.search(r"Venue:\s*(.+)$", info)
    venue = venue_m.group(1).strip() if venue_m else None
    att_m = re.search(r"Att:\s*([\d,]+)", info)
    attendance = _int_or_none(att_m.group(1)) if att_m else None

    quarters: list[tuple[tuple[int, int], ...] | None] = []
    totals: list[int | None] = []
    for cells in (hc, ac):
        tokens = _text(cells[1]).split()
        parsed = [_gb(t) for t in tokens]
        quarters.append(tuple(p for p in parsed if p) if tokens and all(parsed) else None)
        totals.append(_int_or_none(_text(cells[2])))
        if tokens and not all(parsed):
            issues.append(f"{home} v {away}: unparseable quarter scores {tokens!r}")

    link = ac[3].find("a", href=re.compile(r"stats/games/"))
    detail_url = None
    game_id = None
    if link is not None:
        detail_url = resolve_href(str(link["href"]), page_url, GAME_PATH_RE)
        if detail_url is None:
            issues.append(f"{home} v {away}: off-grammar match link rejected")
        else:
            gm = GAME_PATH_RE.fullmatch(urlsplit(detail_url).path)
            game_id = gm.group("gid") if gm else None

    hq, aq = quarters
    hs, as_ = totals
    scored = hq is not None and aq is not None and hs is not None and as_ is not None
    blank = hq is None and aq is None and hs is None and as_ is None
    if scored:
        assert hq is not None and aq is not None
        for qs, total, name in ((hq, hs, home), (aq, as_, away)):
            if len(qs) < 4:
                issues.append(f"{name}: fewer than four quarter scores")
            elif qs[-1][0] * 6 + qs[-1][1] != total:
                issues.append(f"{home} v {away}: score arithmetic mismatch for {name}")
        status = MatchStatus.COMPLETE if detail_url else MatchStatus.UNKNOWN
        if not detail_url:
            issues.append(f"{home} v {away}: scored match without a match-stats link")
    elif blank:
        status = MatchStatus.SCHEDULED
    else:
        status = MatchStatus.UNKNOWN
        issues.append(f"{home} v {away}: partial score cells")

    key = (stage.stage_id, frozenset((home, away)))
    occ = occurrences.get(key, 0)
    occurrences[key] = occ + 1
    home_club = club_resolver(home, season) if club_resolver else None
    away_club = club_resolver(away, season) if club_resolver else None
    if club_resolver is not None and (home_club is None or away_club is None):
        issues.append(f"{home} v {away}: unresolved club identity")
    if home_club and away_club:
        match_id = make_match_id(season, stage.stage_id, home_club, away_club, occ)
    elif game_id:
        match_id = f"afltables:{game_id}"
    else:
        match_id = f"afltables-fixture:{season}:{stage.stage_id}:{hslug}-{aslug}:{occ}"
    return FixtureMatch(
        season=season,
        match_id=match_id,
        source_game_id=game_id,
        stage=stage,
        replay_occurrence=occ,
        home_name=home,
        away_name=away,
        home_club_id=home_club,
        away_club_id=away_club,
        home_team_slug=hslug,
        away_team_slug=aslug,
        venue=venue,
        local_start=local_start,
        match_date=mdate,
        date_precision=precision,
        status=status,
        attendance=attendance,
        home_quarters=hq if scored else None,
        away_quarters=aq if scored else None,
        home_score=hs if scored else None,
        away_score=as_ if scored else None,
        detail_url=detail_url,
    )


# ---------------------------------------------------------------------------
# Match detail
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchPlayerRow:
    team: str
    opponent: str
    player_url: str | None
    source_name: str
    jersey_token: str
    stats: dict[str, int | float | None]


@dataclass
class MatchDetail:
    season: int
    game_id: str
    outcome: CheckOutcome
    stage_label: str | None = None
    venue: str | None = None
    local_start: str | None = None
    attendance: int | None = None
    home_name: str | None = None
    away_name: str | None = None
    home_score: int | None = None
    away_score: int | None = None
    players: list[MatchPlayerRow] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def _stat_value(column: str, text: str) -> int | float | None:
    t = text.replace("\xa0", "").strip()
    if not t:
        return None
    try:
        return float(t) if column == "time_on_ground_pct" else int(t)
    except ValueError:
        return None


def _check_disposals(stats: dict[str, int | float | None]) -> bool:
    k, h, d = stats.get("kicks"), stats.get("handballs"), stats.get("disposals")
    return k is None or h is None or d is None or k + h == d


def parse_match_detail(content: bytes | str, *, season: int, game_id: str) -> MatchDetail:
    url = match_url(season, game_id)
    out = MatchDetail(season=season, game_id=game_id, outcome=CheckOutcome.FAIL)
    if not content:
        out.issues.append("empty payload")
        return out
    soup = _soup(content)
    tables = soup.find_all("table")
    if not tables:
        out.issues.append("no tables (parser drift)")
        return out
    header = tables[0]
    head_text = _text(header)
    m = re.search(r"Round:\s*(.+?)\s+Venue:\s*(.+?)\s+Date:\s*(.+?)(?:\s+Attendance:\s*([\d,]+))?(?:\s|$)", head_text)
    if not m:
        out.issues.append("match header not recognised (parser drift)")
        return out
    round_text = m.group(1).strip()
    stage = resolve_stage(f"Round {round_text}") if round_text.isdigit() else resolve_stage(round_text)
    out.stage_label = stage.label if stage else round_text
    if stage is None:
        out.issues.append(f"unmapped stage {round_text!r}")
    out.venue = m.group(2).strip()
    out.local_start, _d, _p = _parse_date(head_text)
    att_m = re.search(r"Attendance:\s*([\d,]+)", head_text)
    out.attendance = _int_or_none(att_m.group(1)) if att_m else None
    teams: list[tuple[str, int]] = []
    for tr in header.find_all("tr"):
        link = tr.find("a", href=re.compile(r"teams/"))
        cells = tr.find_all("td")
        if link is None or len(cells) < 5:
            continue
        final = _text(cells[-1])
        mm = re.fullmatch(r"(\d+)\.(\d+)\.\s*(\d+)", final)
        if not mm or int(mm.group(1)) * 6 + int(mm.group(2)) != int(mm.group(3)):
            out.issues.append(f"unrecognised final score {final!r}")
            continue
        teams.append((_text(link), int(mm.group(3))))
    if len(teams) != 2:
        out.issues.append("expected two team score rows")
        return out
    (out.home_name, out.home_score), (out.away_name, out.away_score) = teams

    stat_tables = [
        t for t in tables if "sortable" in (t.get("class") or []) and "Match Statistics" in _text(t.find("th"))
    ]
    if len(stat_tables) != 2:
        out.issues.append(f"expected 2 team statistics tables, found {len(stat_tables)}")
        return out
    for table, team, opp in (
        (stat_tables[0], out.home_name, out.away_name),
        (stat_tables[1], out.away_name, out.home_name),
    ):
        thead = table.find("thead")
        header_rows = thead.find_all("tr") if thead is not None else []
        cols = [_text(th) for th in header_rows[-1].find_all("th")] if header_rows else []
        if cols[:2] != ["#", "Player"] or any(c not in STAT_CODE_MAP for c in cols[2:]):
            out.issues.append(f"{team}: statistics header drift {cols[:6]!r}")
            continue
        body = table.find("tbody")
        rows = body.find_all("tr") if body else []
        if not rows:
            out.issues.append(f"{team}: no player rows")
        for tr in rows:
            cells = tr.find_all("td")
            if len(cells) != len(cols):
                out.issues.append(f"{team}: row with {len(cells)} cells, expected {len(cols)}")
                continue
            link = cells[1].find("a", href=True)
            purl = resolve_href(str(link["href"]), url, PLAYER_PATH_RE) if link else None
            if purl is None:
                out.issues.append(f"{team}: player without an on-grammar source link ({_text(cells[1])!r})")
            stats: dict[str, int | float | None] = {c: None for c in PLAYER_STAT_COLUMNS}
            for code, cell in zip(cols[2:], cells[2:], strict=True):
                stats[STAT_CODE_MAP[code]] = _stat_value(STAT_CODE_MAP[code], _text(cell))
            if not _check_disposals(stats):
                out.issues.append(f"{team}: disposals != kicks + handballs for {_text(cells[1])!r}")
            out.players.append(
                MatchPlayerRow(
                    team=team,
                    opponent=opp,
                    player_url=purl,
                    source_name=_text(cells[1]),
                    jersey_token=_text(cells[0]),
                    stats=stats,
                )
            )
    out.outcome = CheckOutcome.FAIL if out.issues else CheckOutcome.PASS
    return out


# ---------------------------------------------------------------------------
# Player page
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerPageGame:
    team: str
    season: int
    opponent: str
    round_token: str
    result: str | None
    jersey_token: str
    career_game_counter: int | None
    career_game_counter_token: str
    stats: dict[str, int | float | None]


@dataclass
class PlayerPage:
    page_url: str
    outcome: CheckOutcome
    name: str | None = None
    birth_date: date | None = None
    games: list[PlayerPageGame] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def parse_player_page(content: bytes | str, *, page_url: str) -> PlayerPage:
    out = PlayerPage(page_url=page_url, outcome=CheckOutcome.FAIL)
    if not PLAYER_PATH_RE.fullmatch(urlsplit(page_url).path):
        out.issues.append("page_url is not an AFLTables player URL")
        return out
    if not content:
        out.issues.append("empty payload")
        return out
    soup = _soup(content)
    out.name = _text(soup.find("h1")) or None
    born = soup.find("b", string=re.compile(r"^\s*Born:\s*$"))
    if born is not None and born.next_sibling is not None:
        m = re.search(r"(\d{1,2}-[A-Za-z]{3}-\d{4})", str(born.next_sibling))
        if m:
            try:
                out.birth_date = datetime.strptime(m.group(1), "%d-%b-%Y").date()
            except ValueError:
                out.birth_date = None
    tables = 0
    for table in soup.find_all("table"):
        th = table.find("th", attrs={"colspan": "28"})
        if th is None:
            continue
        hm = re.fullmatch(r"(.+?) - (\d{4})", _text(th))
        if not hm:
            out.issues.append(f"season header drift {_text(th)[:40]!r}")
            continue
        tables += 1
        team, year = hm.group(1), int(hm.group(2))
        thead = table.find("thead")
        head_rows = thead.find_all("tr") if thead is not None else []
        cols = [_text(x) for x in head_rows[-1].find_all("th")] if len(head_rows) > 1 else []
        if cols[:5] != ["Gm", "Opponent", "Rd", "R", "#"] or any(c not in STAT_CODE_MAP for c in cols[5:]):
            out.issues.append(f"{team} {year}: column header drift")
            continue
        body = table.find("tbody")
        for tr in body.find_all("tr") if body else []:
            cells = [_text(td) for td in tr.find_all("td")]
            if len(cells) != len(cols):
                out.issues.append(f"{team} {year}: row with {len(cells)} cells")
                continue
            token = cells[0]
            digits = re.sub(r"[^0-9]", "", token)
            stats: dict[str, int | float | None] = {c: None for c in PLAYER_STAT_COLUMNS}
            for code, text in zip(cols[5:], cells[5:], strict=True):
                stats[STAT_CODE_MAP[code]] = _stat_value(STAT_CODE_MAP[code], text)
            out.games.append(
                PlayerPageGame(
                    team=team,
                    season=year,
                    opponent=cells[1],
                    round_token=cells[2],
                    result=cells[3] or None,
                    jersey_token=cells[4],
                    career_game_counter=int(digits) if digits else None,
                    career_game_counter_token=token,
                    stats=stats,
                )
            )
    if tables == 0:
        out.issues.append("no per-season game tables (parser drift or empty page)")
    out.outcome = CheckOutcome.FAIL if out.issues else CheckOutcome.PASS
    return out


@dataclass(frozen=True)
class ResolvedGame:
    game: PlayerPageGame
    match_id: str | None
    stage_label: str | None
    match_date: date | None
    date_quality: DateQuality


def resolve_player_games(games: list[PlayerPageGame], fixture: SeasonFixture) -> list[ResolvedGame]:
    """Resolve player-page rows to fixture matches by (stage code, team pair, occurrence)."""
    seen: dict[tuple[str, frozenset[str]], int] = {}
    out: list[ResolvedGame] = []
    for g in games:
        label, code = stage_for_token(g.round_token)
        if g.season != fixture.season or label is None:
            out.append(ResolvedGame(g, None, label, None, DateQuality.UNKNOWN))
            continue
        pair = frozenset((g.team, g.opponent))
        key = (code, pair)
        idx = seen.get(key, 0)
        seen[key] = idx + 1
        cands = [m for m in fixture.matches if m.stage.code == code and m.team_pair == pair]
        if idx < len(cands) and cands[idx].match_date is not None:
            m = cands[idx]
            out.append(ResolvedGame(g, m.match_id, m.stage.label, m.match_date, DateQuality.FIXTURE_VERIFIED))
        else:
            out.append(ResolvedGame(g, None, label, None, DateQuality.UNKNOWN))
    return out
