"""Synthetic AFLTables page builders shared by adapter and refresh tests.

Markup mirrors the captured 2026 season page (tests/scvia/fixtures/raw/afltables/) and the
legacy parsers' documented match-stats/player-page layouts. No tests live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STAT_CODES = [
    "KI",
    "MK",
    "HB",
    "DI",
    "GL",
    "BH",
    "HO",
    "TK",
    "RB",
    "IF",
    "CL",
    "CG",
    "FF",
    "FA",
    "BR",
    "CP",
    "UP",
    "CM",
    "MI",
    "1%",
    "BO",
    "GA",
    "%P",
]
TEAM_SLUGS = {
    "Sydney": "swans",
    "Carlton": "carlton",
    "Geelong": "geelong",
    "Hawthorn": "hawthorn",
    "Collingwood": "collingwood",
    "Western Bulldogs": "bullldogs",
    "Melbourne": "melbourne",
    "Fremantle": "fremantle",
    "Adelaide": "adelaide",
    "Brisbane Lions": "brisbanel",
}


@dataclass
class M:
    home: str
    away: str
    date: str  # e.g. "Thu 05-Mar-2026 7:30 PM"
    venue: str = "M.C.G."
    home_q: list[tuple[int, int]] | None = None  # cumulative quarter (goals, behinds)
    away_q: list[tuple[int, int]] | None = None
    game_id: str | None = None
    att: str = "40,372"


@dataclass
class Stage:
    heading: str
    matches: list[M] = field(default_factory=list)
    byes: list[str] = field(default_factory=list)


def _team_cell(name: str) -> str:
    return f'<td width=16%><a href="../teams/{TEAM_SLUGS.get(name, name.lower())}_idx.html">{name}</a></td>'


def _q(qs: list[tuple[int, int]] | None) -> tuple[str, str]:
    if qs is None:
        return "<tt></tt>", ""
    tt = " ".join(f"&nbsp;{g}.{b}" for g, b in qs)
    g, b = qs[-1]
    return f"<tt>{tt} </tt>", f" {g * 6 + b}"


def season_html(season: int, stages: list[Stage]) -> str:
    out = [
        f"<html><title>AFL Tables -  {season} Season Scores</title><body>",
        f"<h1> {season} Season Scores and Results</h1>",
    ]
    finals_marker = False
    for st in stages:
        if not st.heading.startswith("Round") and not finals_marker:
            out.append('<a name="fin"></a><table border=2 width=100%><tr><td align=center><b>Finals</td></tr></table>')
            finals_marker = True
        out.append(
            f"<table border=2 width=100%><tr><td align=center width=60%><b>{st.heading}</td>"
            "<td><b>Rnd Att: </b>1</td></tr></table>"
        )
        if st.heading.startswith("Round"):
            out.append("<table width=100%><tr><td width=85% valign=top>")
        for m in st.matches:
            htt, hs = _q(m.home_q)
            att, as_ = _q(m.away_q)
            if m.home_q is not None and m.game_id:
                result = (
                    f'<b>{m.home}</b> won by 1 pts [<a href="../stats/games/{season}/{m.game_id}.html">Match stats</a>]'
                )
                info = f'{m.date} <b>Att: </b>{m.att} <b>Venue:</b> <a href="../venues/x.html">{m.venue}</a>'
            else:
                result = ""
                info = f'{m.date} <b>Venue:</b> <a href="../venues/x.html">{m.venue}</a>'
            out.append(
                "<table width=100% border=1>"
                f"<tr>{_team_cell(m.home)}<td nowrap width=20% align=center>{htt}</td>"
                f"<td width=5% align=center>{hs}</td><td>{info}</td></tr>"
                f"<tr>{_team_cell(m.away)}<td nowrap width=20% align=center>{att}</td>"
                f"<td width=5% align=center>{as_}</td><td>{result}</td></tr></table>"
            )
        for b in st.byes:
            out.append(f"<table width=100% border=1><tr>{_team_cell(b)}<td colspan=3 width=80%>Bye</td></tr></table>")
        if st.heading.startswith("Round"):
            out.append(
                "</td><td width=15% valign=top><table width=100% border=1>"
                "<tr><td colspan=4>Rd Ladder</td></tr><tr><td>SY</td><td> 1</td><td> 4</td>"
                "<td>191.3</td></tr></table></td></tr></table>"
            )
    out.append(
        '<table class="sortable"><thead><tr><th colspan=15>Ladder</th></tr></thead><tbody>'
        '<tr><td> 1</td><td><a href="../teams/fremantle_idx.html">Fremantle</td><td>23</td></tr>'
        "</tbody></table></body></html>"
    )
    return "".join(out)


@dataclass
class P:
    path: str  # e.g. "E/Errol_Gulden"
    display: str  # "Gulden, Errol"
    jumper: str = "21"
    stats: dict[str, str] = field(default_factory=dict)  # code -> text


def match_html(
    *,
    season: int,
    round_label: str,
    home: str,
    away: str,
    home_q: list[tuple[int, int]],
    away_q: list[tuple[int, int]],
    date: str = "Thu, 05-Mar-2026 7:30 PM",
    venue: str = "S.C.G.",
    attendance: str = "40372",
    home_players: list[P] | None = None,
    away_players: list[P] | None = None,
) -> str:
    def qcells(qs: list[tuple[int, int]]) -> str:
        cells = "".join(f"<td>{g}.{b}</td>" for g, b in qs[:3])
        g, b = qs[3]
        return cells + f"<td>{g}.{b}.{g * 6 + b}</td>"

    head = (
        "<table><tr><td>Match</td>"
        f'<td>Round: {round_label} Venue: <a href="../../venues/x.html">{venue}</a> '
        f"Date: {date} (7:30 PM) Attendance: {attendance}</td>"
        f'<td>q</td></tr><tr><td><a href="../../teams/x.html">{home}</a></td>{qcells(home_q)}</tr>'
        f'<tr><td><a href="../../teams/y.html">{away}</a></td>{qcells(away_q)}</tr></table>'
    )

    def team_table(name: str, players: list[P]) -> str:
        hdr = "".join(f"<th>{c}</th>" for c in ["#", "Player", *STAT_CODES])
        rows = []
        for p in players:
            cells = "".join(f"<td>{p.stats.get(c, '&nbsp;')}</td>" for c in STAT_CODES)
            rows.append(
                f'<tr><td>{p.jumper}</td><td><a href="../../players/{p.path}.html">{p.display}</a></td>{cells}</tr>'
            )
        return (
            '<table class="sortable"><thead>'
            f"<tr><th colspan=25>{name} Match Statistics [Season][Game by Game]</th></tr>"
            f"<tr>{hdr}</tr></thead><tbody>{''.join(rows)}</tbody>"
            "<tfoot><tr><td></td><td>Totals</td></tr></tfoot></table>"
        )

    return (
        f"<html><body>{head}{team_table(home, home_players or [])}{team_table(away, away_players or [])}</body></html>"
    )


@dataclass
class G:
    gm: str
    opponent: str
    rd: str
    result: str = "W"
    jumper: str = "21"
    stats: dict[str, str] = field(default_factory=dict)


def player_html(name: str, born: str, seasons: list[tuple[str, int, list[G]]]) -> str:
    parts = [f"<html><body><h1>{name}</h1><b>Born:</b> {born} (24y 1d)<br>"]
    for team, year, games in seasons:
        hdr = "".join(f"<th>{c}</th>" for c in ["Gm", "Opponent", "Rd", "R", "#", *STAT_CODES])
        rows = "".join(
            f"<tr><td>{g.gm}</td><td>{g.opponent}</td><td>{g.rd}</td><td>{g.result}</td>"
            f"<td>{g.jumper}</td>" + "".join(f"<td>{g.stats.get(c, '&nbsp;')}</td>" for c in STAT_CODES) + "</tr>"
            for g in games
        )
        parts.append(
            f'<table class="sortable"><thead><tr><th colspan="28">{team} - {year}</th></tr>'
            f"<tr>{hdr}</tr></thead><tbody>{rows}</tbody></table>"
        )
    parts.append("</body></html>")
    return "".join(parts)
