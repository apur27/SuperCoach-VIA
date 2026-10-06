"""Link-level extraction and page-identity checks used by the capture coordinator.

The capture plan pins this module's hash, so it is kept small and stable: it finds *which*
pages to fetch (census letters, profiles, seasons, matches) and whether a fetched body is
the page that was asked for. Cell-level parsing for the comparison lives in ``source.py``.

Identity is decided from the HTTP status (by the caller) plus the page's title/H1 and the
presence of its defining structure, never from body-size or wording heuristics. A
challenge, login or "page has been sent off" body with HTTP 200 is an unusable page.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import date

from supercoach_via.integrity import sourcepages as sp
from supercoach_via.reconciliation.urls import SITE, normalise_link

_HREF = re.compile(rb"""<a\s[^>]*?href\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.I | re.S)
_TITLE = re.compile(rb"<title>(.*?)</title>", re.I | re.S)
_H1 = re.compile(rb"<h1[^>]*>(.*?)</h1>", re.I | re.S)
_TAGS = re.compile(rb"<[^>]+>")
_PROFILE_PATH = re.compile(r"^https://afltables\.com/afl/stats/players/[A-Z]/[^/]+\.html$")
_GAME_PATH = re.compile(r"^https://afltables\.com/afl/stats/games/\d{4}/\d+\.html$")
_LETTER_PATH = re.compile(r"^https://afltables\.com/afl/stats/players([A-Z])_idx\.html$")
_CHALLENGE = re.compile(
    rb"just a moment|cf-chl|attention required|captcha|access denied|verify you are human"
    rb"|enable javascript and cookies",
    re.I,
)
_LOGIN = re.compile(rb"type\s*=\s*[\"']?password|please log in|sign in to continue", re.I)


def _text(raw: bytes) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAGS.sub(b"", raw).decode("utf-8", "replace"))).strip()


def page_title(content: bytes) -> str | None:
    m = _TITLE.search(content)
    return _text(m.group(1)) if m else None


def page_h1(content: bytes) -> str | None:
    m = _H1.search(content)
    return _text(m.group(1)) if m else None


def anchors(content: bytes, page_url: str) -> list[tuple[str, str]]:
    """``(absolute URL, anchor text)`` for every link that points at a fetchable resource."""
    out: list[tuple[str, str]] = []
    for m in _HREF.finditer(content):
        raw = (m.group(1) or m.group(2) or m.group(3) or b"").decode("utf-8", "replace")
        url = normalise_link(html.unescape(raw), page_url)
        if url is None:
            continue
        end = content.find(b"</a>", m.end())
        label = _text(content[m.end() : end].split(b">", 1)[-1]) if end != -1 else ""
        out.append((url, label))
    return out


# ---------------------------------------------------------------------------
# Index, notes and census
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatsIndex:
    seasons: list[int]
    all_players_url: str | None
    notes_url: str | None
    problems: list[str]


def read_stats_index(content: bytes, page_url: str) -> StatsIndex:
    seasons: set[int] = set()
    all_players: str | None = None
    notes: str | None = None
    for url, label in anchors(content, page_url):
        m = re.fullmatch(r"https://afltables\.com/afl/stats/(\d{4})s?\.html", url)
        if m:
            seasons.add(int(m.group(1)))
        elif _LETTER_PATH.match(url) and label == "All Players":
            all_players = url
        elif url == SITE + "/afl/stats/notes.html":
            notes = url
    problems = []
    if not seasons:
        problems.append("no season links found")
    elif sorted(seasons) != list(range(min(seasons), max(seasons) + 1)):
        problems.append("season links are not contiguous")
    if all_players is None:
        problems.append("no All Players link")
    return StatsIndex(sorted(seasons), all_players, notes, problems)


@dataclass(frozen=True)
class ProfileLink:
    display_name: str
    url: str


@dataclass(frozen=True)
class CensusPage:
    letter: str | None
    #: (letter, URL or None for the current page) in navigation order
    nav: list[tuple[str, str | None]]
    profiles: list[ProfileLink]
    problems: list[str]


def read_census_page(content: bytes, page_url: str) -> CensusPage:
    problems: list[str] = []
    h1 = page_h1(content) or ""
    m = re.fullmatch(r"All Players - ([A-Z])", h1)
    letter = m.group(1) if m else None
    expected = _LETTER_PATH.match(page_url)
    if letter is None:
        problems.append(f"census heading not recognised: {h1[:40]!r}")
    elif expected and expected.group(1) != letter:
        problems.append(f"declared letter {letter} but requested letter {expected.group(1)}")
    nav: list[tuple[str, str | None]] = []
    links = {url: label for url, label in anchors(content, page_url) if _LETTER_PATH.match(url)}
    for code in range(ord("A"), ord("Z") + 1):
        ch = chr(code)
        url = f"{SITE}/afl/stats/players{ch}_idx.html"
        if url in links and links[url] == ch:
            nav.append((ch, url))
        elif ch == letter:
            nav.append((ch, None))
        else:
            problems.append(f"navigation link for {ch} missing")
            nav.append((ch, None))
    seen: dict[str, ProfileLink] = {}
    for url, label in anchors(content, page_url):
        if _PROFILE_PATH.match(url) and url not in seen:
            seen[url] = ProfileLink(label, url)
    return CensusPage(letter, nav, list(seen.values()), problems)


# ---------------------------------------------------------------------------
# Season, match and profile pages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeasonFixtureLink:
    stage_text: str
    home: str
    away: str
    match_date: date | None
    game_url: str | None
    has_score: bool


@dataclass(frozen=True)
class SeasonInfo:
    year: int | None
    fixtures: list[SeasonFixtureLink]
    problems: list[str]


def read_season_page(content: bytes, page_url: str) -> SeasonInfo:
    h1 = page_h1(content) or ""
    m = re.match(r"(\d{4}) Season Scores", h1)
    problems: list[str] = []
    year = int(m.group(1)) if m else None
    if year is None:
        problems.append(f"season heading not recognised: {h1[:40]!r}")
    page = sp.read_season_page(content)
    problems.extend(page.problems)
    fixtures = []
    for fx in page.fixtures:
        game = normalise_link(fx.game_link, page_url) if fx.game_link else None
        scored = fx.home_points is not None and fx.away_points is not None
        if scored and game is None:
            problems.append(f"scored fixture {fx.home} v {fx.away} has no game link")
        fixtures.append(SeasonFixtureLink(fx.stage_text, fx.home, fx.away, fx.match_date, game, scored))
    page_games = {u for u, _ in anchors(content, page_url) if _GAME_PATH.match(u)}
    listed = {f.game_url for f in fixtures if f.game_url}
    if page_games != listed:
        problems.append(f"game links on the page ({len(page_games)}) differ from fixtures listed ({len(listed)})")
    return SeasonInfo(year, fixtures, problems)


def match_profile_links(content: bytes, page_url: str) -> list[str]:
    return sorted({u for u, _ in anchors(content, page_url) if _PROFILE_PATH.match(u)})


def profile_game_links(content: bytes, page_url: str) -> list[str]:
    return sorted({u for u, _ in anchors(content, page_url) if _GAME_PATH.match(u)})


# ---------------------------------------------------------------------------
# Page identity
# ---------------------------------------------------------------------------


def page_problem(kind: str, content: bytes, url: str) -> str | None:
    """Why a HTTP-200 body is not the page that was requested, or ``None`` when it is."""
    if not content.strip():
        return "empty body"
    if _CHALLENGE.search(content[:8192]):
        return "challenge page"
    if _LOGIN.search(content[:8192]):
        return "login page"
    title = page_title(content) or ""
    h1 = page_h1(content) or ""
    if title == "Broked!":
        return "source error page ('this page has been sent off')"
    if kind == "stats_index":
        return None if "Player, Coach and Umpire Stats" in title else f"unexpected title {title[:60]!r}"
    if kind == "notes":
        return None if "Notes on Player Stats" in title else f"unexpected title {title[:60]!r}"
    if kind == "letter":
        m = _LETTER_PATH.match(url)
        want = f"All Players - {m.group(1)}" if m else None
        return None if want and h1 == want else f"unexpected heading {h1[:60]!r}"
    if kind == "season":
        m2 = re.search(r"/seas/(\d{4})\.html$", url)
        return None if m2 and h1.startswith(f"{m2.group(1)} Season Scores") else f"unexpected heading {h1[:60]!r}"
    if kind == "match":
        if not title.endswith("Match Stats"):
            return f"unexpected title {title[:60]!r}"
        return None if b"Round:" in content else "match header missing"
    if kind == "profile":
        if not h1 or not title.endswith("Stats - Statistics"):
            return f"unexpected title/heading {title[:60]!r}/{h1[:30]!r}"
        return None
    return f"unknown page kind {kind!r}"
