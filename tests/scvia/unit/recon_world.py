"""A tiny consistent AFL Tables 'world' for the reconciliation tests.

One specification renders (a) the source pages the capture would store (stats index, census
letters, season pages, match pages, profiles) in the real site's markup, and (b) the matching
local rows. Tests perturb either side to create the defect under test. Markup mirrors the
captured pages in ``tests/scvia/fixtures/reconciliation``.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field, replace
from datetime import date

from supercoach_via.reconciliation.schema import STAT_LABELS

SITE = "https://afltables.com"
LABELS = dict(STAT_LABELS)  # source label -> field
FIELD_LABEL = {f: lab for lab, f in STAT_LABELS}
SUMMARY_FIELDS = [f for lab, f in STAT_LABELS if lab != "%P"]
STAGE_TOKEN = {
    "Qualifying Final": "QF",
    "Elimination Final": "EF",
    "Semi Final": "SF",
    "Preliminary Final": "PF",
    "Grand Final": "GF",
    "Wildcard Final": "WF",
}
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@dataclass(frozen=True)
class P:
    slug: str  # URL stem, e.g. "Ann_Able" or "Gary_Ablett0"
    first: str
    last: str
    born: str | None = "01-Jul-1990"  # source format d-Mon-yyyy; None = no Born line

    @property
    def url(self) -> str:
        return f"{SITE}/afl/stats/players/{self.slug[0].upper()}/{self.slug}.html"

    @property
    def display(self) -> str:
        return f"{self.first} {self.last}"


@dataclass(frozen=True)
class A:
    pid: str
    club: str
    jersey: str = "5"
    cells: dict[str, int | float | None] = field(default_factory=dict)
    counter_token: str | None = None


@dataclass(frozen=True)
class M:
    gid: str  # digits only, e.g. "041520260305"
    year: int
    stage: str  # "1".."25" or "Qualifying Final"/"Grand Final"...
    home: str
    away: str
    when: date
    time: str = "2:10 PM"
    venue: str = "Demo Oval"
    hq: tuple[tuple[int, int], ...] = ((2, 3), (4, 5), (6, 7), (8, 9))  # cumulative (g, b)
    aq: tuple[tuple[int, int], ...] = ((1, 2), (3, 3), (5, 4), (7, 6))
    apps: tuple[A, ...] = ()

    @property
    def url(self) -> str:
        return f"{SITE}/afl/stats/games/{self.year}/{self.gid}.html"

    def score(self, side: str) -> int:
        g, b = (self.hq if side == "home" else self.aq)[-1]
        return 6 * g + b

    @property
    def is_final(self) -> bool:
        return not self.stage.isdigit()

    @property
    def rd_token(self) -> str:
        return STAGE_TOKEN.get(self.stage, self.stage)


@dataclass(frozen=True)
class World:
    players: dict[str, P]
    matches: tuple[M, ...]
    seasons: tuple[int, ...] = (2026,)
    #: (player id, year, field) -> season value printed in the summary table although per-game cells are blank
    summary_overrides: dict[tuple[str, int, str], int] = field(default_factory=dict)

    def with_matches(self, matches: tuple[M, ...]) -> World:
        return replace(self, matches=matches)

    def career_to_date(self) -> dict[tuple[str, str], int]:
        """(match gid, player id) -> the career game counter the profile prints for that game."""
        out: dict[tuple[str, str], int] = {}
        counts: dict[str, int] = {}
        for m in sorted(self.matches, key=lambda x: (x.when, x.gid)):
            for a in m.apps:
                counts[a.pid] = counts.get(a.pid, 0) + 1
                tok = a.counter_token
                out[(m.gid, a.pid)] = int("".join(c for c in tok if c.isdigit())) if tok else counts[a.pid]
        return out


def _fmt_date(d: date, time: str, *, comma: bool = True) -> str:
    return f"{d.strftime('%a')}{',' if comma else ''} {d.day}-{MONTHS[d.month - 1]}-{d.year} {time}"


def _cell(v: int | float | str | None) -> str:
    # the real site prints a zero count as a blank cell (the team total then proves it was recorded)
    if v is None or v == "" or (isinstance(v, int | float) and not isinstance(v, bool) and v == 0):
        return "<td>&nbsp;</td>"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"<td align=center>{v}</td>"


def _page(title: str, body: str, h1: str | None = None) -> bytes:
    head = f"<html><head><title>{html.escape(title)}</title></head><body><center>"
    h = f"<H1>{html.escape(h1)}</H1>" if h1 else ""
    return (head + h + body + "</center></body></html>").encode()


# ---------------------------------------------------------------------------
# Index, notes, census
# ---------------------------------------------------------------------------


def stats_index(seasons: list[int]) -> bytes:
    links = "".join(f'[<a href="{y}{"s" if y >= 1965 else ""}.html">{y}</a>]' for y in seasons)
    body = (
        f'[<a href="../afl_index.html">AFL Main</a>]<br>{links}<br>[<a href="notes.html">Notes on player stats]</a>'
        '[<a href="playersA_idx.html">All Players</a>]'
    )
    return _page("AFL Tables - Player, Coach and Umpire Stats", body, "Player, Coach and Umpire Statistics")


def notes_page() -> bytes:
    return _page("AFL Tables - Notes on Player Stats", "<p>Notes</p>", "Notes on player stats")


def census_page(letter: str, players: list[P], *, omit_nav: str | None = None) -> bytes:
    nav = "".join(
        f"[{c}]" if c == letter else (f"[{c}]" if c == omit_nav else f'[<a href="players{c}_idx.html">{c}</a>]')
        for c in (chr(x) for x in range(ord("A"), ord("Z") + 1))
    )
    rows = "".join(
        f'<td><a href="players/{p.slug[0].upper()}/{p.slug}.html">{p.last}, {p.first}</a></td>' for p in players
    )
    body = f'[<a href="stats_idx.html">Stats Main</a>]<br>{nav}<br><table><tr>{rows}</tr></table>'
    return _page(f"AFL Tables - All Players - {letter}", body, f"All Players - {letter}")


# ---------------------------------------------------------------------------
# Season and match pages
# ---------------------------------------------------------------------------


def season_page(year: int, matches: list[M]) -> bytes:
    out = [f'[<a href="../seas/{year - 1}.html">{year - 1}</a>]']
    stage = None
    for m in sorted(matches, key=lambda x: (x.when, x.gid)):
        if m.stage != stage:
            stage = m.stage
            head = f"Round {stage}" if stage.isdigit() else stage
            out.append(f"<table border=2><tr><td align=center width=60%><b>{head}</td></tr></table>")
        rows = []
        for side in ("home", "away"):
            qs = m.hq if side == "home" else m.aq
            q = " ".join(f"{g}.{b}" for g, b in qs)
            team = m.home if side == "home" else m.away
            info = (
                f"{_fmt_date(m.when, m.time, comma=False)} <b>Att: </b>1,000 <b>Venue:</b> "
                f'<a href="../venues/v.html">{m.venue}</a>'
                if side == "home"
                else f'<b>{m.home}</b> won by <b>1 pts </b>[<a href="../stats/games/{m.year}/{m.gid}.html">Match stats</a>]'
            )
            rows.append(
                f'<tr><td width=16%><a href="../teams/{team.lower()}_idx.html">{team}</a></td>'
                f"<td><tt>{q}</tt></td><td> {m.score(side)}</td><td>{info}</td></tr>"
            )
        out.append("<table border=1>" + "".join(rows) + "</table>")
    return _page(f"AFL Tables -  {year} Season Scores", "".join(out), f" {year} Season Scores and Results")


def season_page_with_unplayed(year: int, matches: list[M], unplayed_club: tuple[str, str]) -> bytes:
    base = season_page(year, matches).decode().replace("</center>", "")
    extra = (
        "<table border=2><tr><td align=center><b>Round 99</td></tr></table><table border=1>"
        f'<tr><td><a href="../teams/{unplayed_club[0].lower()}_idx.html">{unplayed_club[0]}</a></td><td><tt></tt></td>'
        '<td></td><td>Sat 31-Dec-2033 2:10 PM <b>Att: </b>0 <b>Venue:</b> <a href="../venues/v.html">X</a></td></tr>'
        f'<tr><td><a href="../teams/{unplayed_club[1].lower()}_idx.html">{unplayed_club[1]}</a></td><td><tt></tt></td>'
        "<td></td><td></td></tr></table>"
    )
    return (base + extra + "</center></body></html>").encode()


def _details_table(m: M, team: str, players: dict[str, P], career: dict[str, int]) -> str:
    """The real page's ``<Team> Player Details`` table: coach row, then one row per lineup player."""
    rows = [
        '<tr><td align=center>C</td><td><a href="../../coaches/Some_Coach.html">Coach, Some</a></td>'
        "<td align=center>50y 1d</td><td align=center>100 (50-0-50 50.00%)</td><td>&nbsp;</td>"
        "<td align=center>&larr;</td><td>&nbsp;</td></tr>"
    ]
    for a in (x for x in m.apps if x.club == team):
        p = players[a.pid]
        n = career.get(a.pid, 1)
        rows.append(
            f'<tr><td align=center>{a.jersey}</td><td><a href="../../players/{p.slug[0].upper()}/{p.slug}.html">'
            f"{p.last}, {p.first}</a></td><td align=center>25y 1d</td>"
            f"<td align=center>{n} ({n}-0-0 100.00%)</td><td>&nbsp;</td><td align=center>&larr;</td><td>&nbsp;</td></tr>"
        )
    return (
        f'<table class="sortable"><thead><tr><th colspan=7>{team} Player Details</th></tr><tr><th>#</th>'
        "<th>Player</th><th>Age</th><th>Career Games (W-D-L W%)</th><th>Career Goals (Ave.)</th>"
        f"<th>{team} Games (W-D-L W%)</th><th>{team} Goals (Ave.)</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def match_page(
    m: M, players: dict[str, P], *, rushed: bool = False, career: dict[str, int] | None = None, details: bool = True
) -> bytes:
    head = [
        f'<table border=2><tr><td rowspan=6><a href="../{m.year}/0.html">&larr;</a></td><td colspan=5 align=center>'
        f'<b>Round: </b>{m.stage} <b>Venue: </b><a href="../../../venues/v.html">{m.venue}</a> '
        f"<b>Date: </b>{_fmt_date(m.when, m.time)} <b>Attendance:</b> 1000</td>"
        f'<td rowspan=6><a href="../{m.year}/9.html">&rarr;</a></td></tr>'
    ]
    for _side, team, qs in (("home", m.home, m.hq), ("away", m.away, m.aq)):
        cells = "".join(f"<td align=center>{g}.{b}.<b>{6 * g + b}</b></td>" for g, b in qs)
        head.append(f'<tr><td><a href="../../../teams/{team.lower()}_idx.html">{team}</a></td>{cells}</tr>')
    head.append("</table>")
    body = list(head)
    for team in (m.home, m.away):
        apps = [a for a in m.apps if a.club == team]
        hdr = "".join(f"<th>{lab}</th>" for lab, _ in STAT_LABELS)
        body.append(
            f'<table class="sortable"><thead><tr><th colspan=25>{team} Match Statistics '
            f'[<a href="../../{m.year}.html#1">Season</a>]</th></tr><tr><th>#</th><th>Player</th>{hdr}</tr></thead><tbody>'
        )
        for a in apps:
            p = players[a.pid]
            body.append(
                f'<tr><td align=center>{a.jersey}</td><td><a href="../../players/{p.slug[0].upper()}/{p.slug}.html">'
                f"{p.last}, {p.first}</a></td>" + "".join(_cell(a.cells.get(f)) for _, f in STAT_LABELS) + "</tr>"
            )
        body.append("</tbody><tfoot>")
        if rushed:
            body.append("<tr><td colspan=7>Rushed</td><td align=center>3</td><td colspan=17>&nbsp;</td></tr>")
        totals = []
        for _lab, f in STAT_LABELS:
            vals = [a.cells.get(f) for a in apps if a.cells.get(f) is not None]
            totals.append(
                "<td>&nbsp;</td>"
                if f == "time_on_ground_pct" or not vals
                else f"<td align=center><b>{sum(vals):g}</td>"
            )
        body.append("<tr><td colspan=2><b>Totals</td>" + "".join(totals) + "</tr></tfoot></table>")
    if details:
        counts = career if career is not None else {}
        for team in (m.home, m.away):
            body.append(_details_table(m, team, players, counts))
    title = f"AFL Tables - {m.home} v {m.away} - {_fmt_date(m.when, m.time)} - Match Stats"
    return _page(title, "".join(body))


# ---------------------------------------------------------------------------
# Player profile
# ---------------------------------------------------------------------------


def _result(m: M, club: str) -> str:
    own, other = (m.score("home"), m.score("away")) if club == m.home else (m.score("away"), m.score("home"))
    return "W" if own > other else "L" if own < other else "D"


def profile_page(
    w: World,
    pid: str,
    *,
    header_extra: str = "",
    rowspan_in_game_table: bool = False,
    duplicate_label: bool = False,
    extra_label: str | None = None,
    counters: dict[str, str] | None = None,
) -> bytes:
    p = w.players[pid]
    games = []
    for m in sorted(w.matches, key=lambda x: (x.when, x.gid)):
        for a in m.apps:
            if a.pid == pid:
                games.append((m, a))
    seasons: dict[tuple[str, int], list[tuple[M, A, str]]] = {}
    for counter, (m, a) in enumerate(games, 1):
        token = (counters or {}).get(m.gid) or a.counter_token or str(counter)
        seasons.setdefault((a.club, m.year), []).append((m, a, token))
    born = f"<b>Born:</b>{p.born} (<b>Debut:</b>20y 1d <b>Last:</b>30y 1d)<br><br>" if p.born else ""
    out = [f"<h1>{p.display}</h1>{born}{header_extra}"]
    # season summary tables (totals, then averages)
    for kind in ("totals", "averages"):
        hdr = "".join(f"<th>{lab}</th>" for lab, _ in STAT_LABELS if lab != "%P")
        out.append(
            f"<table class=sortable><thead><tr><th>Year</th><th>Team</th><th>#</th><th>GM</th><th>W-D-L</th>{hdr}</tr></thead><tbody>"
        )
        tot_gm = 0
        all_wdl = [0, 0, 0]
        tot = dict.fromkeys(SUMMARY_FIELDS, 0.0)
        seen = dict.fromkeys(SUMMARY_FIELDS, False)
        for (club, year), rows in seasons.items():
            wdl = [0, 0, 0]
            sums = dict.fromkeys(SUMMARY_FIELDS, 0.0)
            have = dict.fromkeys(SUMMARY_FIELDS, False)
            for m, a, _t in rows:
                wdl["WDL".index(_result(m, club))] += 1
                for f in SUMMARY_FIELDS:
                    v = a.cells.get(f)
                    if v is not None:
                        sums[f] += v
                        have[f] = True
            for f in SUMMARY_FIELDS:
                ov = w.summary_overrides.get((pid, year, f))
                if ov is not None:
                    sums[f] += ov
                    have[f] = True
            gm = len(rows)
            tot_gm += gm
            all_wdl = [a + b for a, b in zip(all_wdl, wdl, strict=True)]
            for f in SUMMARY_FIELDS:
                tot[f] += sums[f]
                seen[f] = seen[f] or have[f]
            if kind == "totals":
                cells = "".join(_cell(f"{sums[f]:g}" if have[f] else None) for f in SUMMARY_FIELDS)
                out.append(
                    f'<tr><td><a href="../../{year}.html#4">{year}</a></td><td><a href="../../../teams/{club.lower()}_idx.html">{club}</a></td>'
                    f'<td>{rows[0][1].jersey}</td><td><a href="#{year}0">{gm}</a></td><td>{wdl[0]}-{wdl[1]}-{wdl[2]}</td>{cells}</tr>'
                )
            else:
                ha = sum(1 for m, _a, _t in rows if not m.is_final)
                cells = "".join(
                    _cell(
                        f"{sums[f] / (ha if f == 'brownlow_votes' else gm):.2f}"
                        if have[f] and (ha or f != "brownlow_votes")
                        else None
                    )
                    for f in SUMMARY_FIELDS
                )
                out.append(f"<tr><td>{year}</td><td>{club}</td><td></td><td>{gm}</td><td></td>{cells}</tr>")
        if kind == "totals":
            cells = "".join(_cell(f"{tot[f]:g}" if seen[f] else None) for f in SUMMARY_FIELDS)
            ha_all = sum(1 for rows_ in seasons.values() for m, _a, _t in rows_ if not m.is_final)
            avg = "".join(
                _cell(f"{tot[f] / (ha_all if f == 'brownlow_votes' else tot_gm):.2f}" if seen[f] else None)
                for f in SUMMARY_FIELDS
            )
            win_pct = f"{100 * (all_wdl[0] + 0.5 * all_wdl[1]) / tot_gm:.2f}%" if tot_gm else ""
            out.append(
                f"</tbody><tfoot><tr><td colspan=3><b>Totals</td><th>{tot_gm}</th><th>{all_wdl[0]}-{all_wdl[1]}-{all_wdl[2]}</th>{cells}</tr>"
                f"<tr><td colspan=3><b>Averages</td><th>{tot_gm / len(seasons):.2f}</th><th>{win_pct}</th>{avg}</tr></tfoot></table>"
            )
        else:
            out.append("</tbody></table>")
    # per club-season game tables
    labels = [lab for lab, _ in STAT_LABELS]
    if extra_label:
        labels = [*labels, extra_label]
    if duplicate_label:
        labels = [*labels[:-1], labels[0]]
    for (club, year), rows in seasons.items():
        hdr = "".join(f"<th>{lab}</th>" for lab in labels)
        out.append(
            f'<table class="sortable"><thead><tr><th colspan={5 + len(labels)}>{club} - {year}</th></tr>'
            f"<tr><th>Gm</th><th>Opponent</th><th>Rd</th><th>R</th><th>#</th>{hdr}</tr></thead><tbody>"
        )
        wdl = [0, 0, 0]
        for i, (m, a, token) in enumerate(rows):
            opp = m.away if club == m.home else m.home
            res = _result(m, club)
            wdl["WDL".index(res)] += 1
            span = " rowspan=2" if rowspan_in_game_table and i == 0 else ""
            cells = "".join(_cell(a.cells.get(LABELS.get(lab, "")) if lab in LABELS else "1") for lab in labels)
            out.append(
                f"<tr><td{span} align=center>{token}</td><td nowrap>{opp}</td>"
                f'<td align=center><a href="../../games/{m.year}/{m.gid}.html">{m.rd_token}</a></td>'
                f"<td align=center>{res}</td><td align=center>{a.jersey}</td>{cells}</tr>"
            )
        foot = ""
        for lab in labels:
            f = LABELS.get(lab)
            vals = [a.cells.get(f) for _m, a, _t in rows if f and a.cells.get(f) is not None]
            foot += _cell(f"{sum(vals):g}" if vals and f != "time_on_ground_pct" else None)
        out.append(
            f"</tbody><tfoot><tr><td>&nbsp;</td><td><b>Totals</td><th colspan=3>{len(rows)} ({wdl[0]}-{wdl[1]}-{wdl[2]})</th>"
            f"{foot}</tr></tfoot></table>"
        )
    return _page(f"AFL Tables - {p.display} - Stats - Statistics", "".join(out))
