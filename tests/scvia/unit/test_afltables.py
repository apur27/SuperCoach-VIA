"""AFLTables pure adapters: season fixtures, match details, player pages, stage mapping."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from supercoach_via.domain.schemas import TABLES, CheckOutcome, DateQuality, MatchStatus, StageType
from supercoach_via.ingest import afltables as at
from tests.scvia.unit.test_afltables_support import G, M, P, Stage, match_html, player_html, season_html

CAPTURED = Path(__file__).resolve().parents[1] / "fixtures" / "raw" / "afltables" / "seas_2026_captured_20260925.html"

Q_SYD = [(0, 3), (2, 6), (14, 9), (20, 12)]
Q_CAR = [(2, 2), (4, 4), (8, 6), (10, 9)]


# ---------------------------------------------------------------------------
# Stage mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("heading", "label", "stype", "rnd", "code", "sid"),
    [
        ("Round 1 * see notes", "1", StageType.REGULAR, 1, "1", "r01"),
        ("Round: 12", "12", StageType.REGULAR, 12, "12", "r12"),
        ("Opening Round", "Opening Round", StageType.REGULAR, 0, "OR", "r00"),
        ("Wildcard Final", "Wildcard Final", StageType.FINAL, None, "WF", "wf"),
        ("Qualifying Final", "Qualifying Final", StageType.FINAL, None, "QF", "qf"),
        ("Grand Final", "Grand Final", StageType.FINAL, None, "GF", "gf"),
    ],
)
def test_resolve_stage(heading: str, label: str, stype: StageType, rnd: int | None, code: str, sid: str) -> None:
    st = at.resolve_stage(heading)
    assert st is not None
    assert (st.label, st.stage_type, st.round_number, st.code, st.stage_id) == (label, stype, rnd, code, sid)


def test_finals_order_after_all_rounds_and_wildcard_first() -> None:
    wf, qf, gf, r25 = (at.resolve_stage(h) for h in ("Wildcard Final", "Qualifying Final", "Grand Final", "Round 25"))
    assert wf and qf and gf and r25
    assert r25.stage_order < wf.stage_order < qf.stage_order < gf.stage_order


def test_unknown_stage_is_not_silently_a_round() -> None:
    assert at.resolve_stage("Section 3 Playoff") is None


# ---------------------------------------------------------------------------
# Season page: captured real payload
# ---------------------------------------------------------------------------


def test_captured_2026_season_page_matches_data_refresh_report() -> None:
    fx = at.parse_season_page(CAPTURED.read_bytes(), season=2026)
    assert fx.outcome is CheckOutcome.PASS, fx.issues
    complete = [m for m in fx.matches if m.status is MatchStatus.COMPLETE]
    assert len(complete) == 217  # DATA_REFRESH.md: 217 completed matches
    assert max(m.match_date for m in complete if m.match_date) == date(2026, 9, 19)
    wf = [m for m in complete if m.stage.code == "WF"]
    assert len(wf) == 2 and all(m.stage.label == "Wildcard Final" for m in wf)
    assert {m.match_date for m in wf} == {date(2026, 8, 28), date(2026, 8, 29)}
    first = complete[0]
    assert (first.home_name, first.away_name, first.home_score, first.away_score) == ("Sydney", "Carlton", 132, 69)
    assert first.local_start == "2026-03-05 19:30" and first.attendance == 40372 and first.venue == "S.C.G."
    assert first.source_game_id == "031620260305"
    assert first.detail_url == "https://afltables.com/afl/stats/games/2026/031620260305.html"
    assert fx.byes > 0
    labels = {m.stage.label for m in complete}
    assert {"Qualifying Final", "Elimination Final", "Semi Final", "Preliminary Final"} <= labels


def test_match_rows_fit_canonical_matches_table() -> None:
    fx = at.parse_season_page(CAPTURED.read_bytes(), season=2026)
    row = fx.matches[0].to_row(source_ref="src:x", source_sha256="0" * 64)
    assert set(row) == set(TABLES["matches"].column_names)
    assert row["status"] == "complete" and row["home_final_goals"] == 20 and row["away_q1_behinds"] == 2


# ---------------------------------------------------------------------------
# Season page: synthetic edge cases
# ---------------------------------------------------------------------------


def test_scheduled_future_fixture_has_no_fake_score() -> None:
    html = season_html(
        2026,
        [
            Stage(
                "Round 1", [M("Sydney", "Carlton", "Thu 05-Mar-2026 7:30 PM", "S.C.G.", Q_SYD, Q_CAR, "031620260305")]
            ),
            Stage("Round 2", [M("Geelong", "Hawthorn", "Sat 14-Mar-2026 1:45 PM", "Kardinia Park")], byes=["Adelaide"]),
        ],
    )
    fx = at.parse_season_page(html, season=2026)
    assert fx.outcome is CheckOutcome.PASS
    sched = fx.matches[1]
    assert sched.status is MatchStatus.SCHEDULED
    assert sched.home_score is None and sched.away_score is None and sched.home_quarters is None
    assert sched.source_game_id is None and sched.detail_url is None
    assert sched.match_id.startswith("afltables-fixture:2026:r02:")
    assert fx.byes == 1


def test_replay_occurrence_for_repeated_final() -> None:
    q = [(1, 1), (2, 2), (3, 3), (10, 10)]
    html = season_html(
        2010,
        [
            Stage(
                "Grand Final", [M("Collingwood", "Geelong", "Sat 25-Sep-2010 2:30 PM", "M.C.G.", q, q, "050720100925")]
            ),
            Stage(
                "Grand Final",
                [
                    M(
                        "Collingwood",
                        "Geelong",
                        "Sat 02-Oct-2010 2:30 PM",
                        "M.C.G.",
                        q,
                        [(1, 1), (2, 2), (3, 3), (9, 9)],
                        "050720101002",
                    )
                ],
            ),
        ],
    )
    fx = at.parse_season_page(html, season=2010)
    assert [m.replay_occurrence for m in fx.matches] == [0, 1]
    assert fx.matches[0].match_id != fx.matches[1].match_id


def test_score_arithmetic_drift_fails() -> None:
    html = season_html(
        2026,
        [Stage("Round 1", [M("Sydney", "Carlton", "Thu 05-Mar-2026 7:30 PM", "S.C.G.", Q_SYD, Q_CAR, "031620260305")])],
    )
    broken = html.replace("> 132<", "> 133<")
    fx = at.parse_season_page(broken, season=2026)
    assert fx.outcome is CheckOutcome.FAIL
    assert any("arithmetic" in i for i in fx.issues)


@pytest.mark.parametrize("payload", ["", "<html><body>Service temporarily unavailable</body></html>"])
def test_empty_or_drifted_page_is_fail_not_zero_rows(payload: str) -> None:
    fx = at.parse_season_page(payload, season=2026)
    assert fx.outcome is CheckOutcome.FAIL and fx.matches == []


def test_wrong_season_page_fails() -> None:
    html = season_html(
        2025,
        [Stage("Round 1", [M("Sydney", "Carlton", "Thu 06-Mar-2025 7:30 PM", "S.C.G.", Q_SYD, Q_CAR, "031620250306")])],
    )
    assert at.parse_season_page(html, season=2026).outcome is CheckOutcome.FAIL


def test_unknown_stage_heading_is_flagged() -> None:
    html = season_html(
        2026,
        [
            Stage(
                "Mystery Cup",
                [M("Sydney", "Carlton", "Thu 05-Mar-2026 7:30 PM", "S.C.G.", Q_SYD, Q_CAR, "031620260305")],
            )
        ],
    )
    fx = at.parse_season_page(html, season=2026)
    assert fx.outcome is CheckOutcome.FAIL
    assert any("unmapped stage" in i for i in fx.issues)


def test_hostile_match_link_is_not_followed() -> None:
    html = season_html(
        2026,
        [Stage("Round 1", [M("Sydney", "Carlton", "Thu 05-Mar-2026 7:30 PM", "S.C.G.", Q_SYD, Q_CAR, "031620260305")])],
    )
    hostile = html.replace("../stats/games/2026/031620260305.html", "https://evil.example/stats/games/2026/1.html")
    fx = at.parse_season_page(hostile, season=2026)
    assert fx.outcome is CheckOutcome.FAIL
    assert all(m.detail_url is None for m in fx.matches)


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------


def test_url_builders_validate_ids() -> None:
    assert at.season_url(2026) == "https://afltables.com/afl/seas/2026.html"
    assert at.match_url(2026, "031620260305").endswith("/afl/stats/games/2026/031620260305.html")
    assert at.player_url("S/Scott_Pendlebury") == "https://afltables.com/afl/stats/players/S/Scott_Pendlebury.html"
    for bad in ("../x", "S/../../etc", "S/a b", "s/lower_initial"):
        with pytest.raises(ValueError):
            at.player_url(bad)
    with pytest.raises(ValueError):
        at.match_url(2026, "12/../3")
    with pytest.raises(ValueError):
        at.season_url(1700)


# ---------------------------------------------------------------------------
# Match detail
# ---------------------------------------------------------------------------


def _detail() -> str:
    return match_html(
        season=2026,
        round_label="1",
        home="Sydney",
        away="Carlton",
        home_q=Q_SYD,
        away_q=Q_CAR,
        home_players=[
            P(
                "E/Errol_Gulden",
                "Gulden, Errol",
                "21",
                {"KI": "20", "HB": "10", "DI": "30", "GL": "2", "BR": "3", "%P": "85"},
            )
        ],
        away_players=[P("P/Patrick_Cripps", "Cripps, Patrick", "9", {"KI": "12", "HB": "14", "DI": "26"})],
    )


def test_parse_match_detail_player_rows_use_source_urls_and_nulls() -> None:
    d = at.parse_match_detail(_detail(), season=2026, game_id="031620260305")
    assert d.outcome is CheckOutcome.PASS, d.issues
    assert d.stage_label == "1" and d.venue == "S.C.G." and d.local_start == "2026-03-05 19:30"
    assert (d.home_name, d.away_name, d.home_score, d.away_score) == ("Sydney", "Carlton", 132, 69)
    gulden = d.players[0]
    assert gulden.player_url == "https://afltables.com/afl/stats/players/E/Errol_Gulden.html"
    assert gulden.team == "Sydney" and gulden.opponent == "Carlton"
    assert gulden.stats["kicks"] == 20 and gulden.stats["brownlow_votes"] == 3
    assert gulden.stats["tackles"] is None  # blank is unknown, not zero
    assert gulden.stats["time_on_ground_pct"] == 85.0
    assert d.players[1].team == "Carlton"


def test_match_detail_without_player_tables_is_fail() -> None:
    html = match_html(season=2026, round_label="1", home="Sydney", away="Carlton", home_q=Q_SYD, away_q=Q_CAR)
    d = at.parse_match_detail(html, season=2026, game_id="031620260305")
    assert d.outcome is CheckOutcome.FAIL


def test_match_detail_disposal_arithmetic_checked() -> None:
    html = _detail().replace("<td>30</td>", "<td>31</td>")
    d = at.parse_match_detail(html, season=2026, game_id="031620260305")
    assert d.outcome is CheckOutcome.FAIL and any("disposals" in i for i in d.issues)


# ---------------------------------------------------------------------------
# Player page + WF resolution from the season table (DATA_REFRESH regression)
# ---------------------------------------------------------------------------


def test_player_page_wf_resolved_from_fixture_never_round_to_weeks() -> None:
    fixture = at.parse_season_page(CAPTURED.read_bytes(), season=2026)
    page = player_html(
        "Marcus Bontempelli",
        "24-Nov-1995",
        [
            (
                "Western Bulldogs",
                2026,
                [
                    G("250", "Adelaide", "25", "L", "4", {"KI": "15", "HB": "10", "DI": "25"}),
                    G("251", "Collingwood", "WF", "W", "4", {"KI": "18", "HB": "9", "DI": "27"}),
                    G("252↓", "Adelaide", "EF", "L", "4", {"KI": "10", "HB": "8", "DI": "18"}),
                ],
            )
        ],
    )
    url = "https://afltables.com/afl/stats/players/M/Marcus_Bontempelli.html"
    pp = at.parse_player_page(page, page_url=url)
    assert pp.outcome is CheckOutcome.PASS, pp.issues
    assert pp.birth_date == date(1995, 11, 24) and pp.name == "Marcus Bontempelli"
    resolved = at.resolve_player_games(pp.games, fixture)
    wf = next(r for r in resolved if r.game.round_token == "WF")
    assert wf.match_date == date(2026, 8, 28) and wf.date_quality is DateQuality.FIXTURE_VERIFIED
    assert wf.stage_label == "Wildcard Final" and wf.match_id == "afltables:040720260828"
    ef = next(r for r in resolved if r.game.round_token == "EF")
    assert ef.match_date == date(2026, 9, 5)
    assert ef.game.career_game_counter == 252 and ef.game.career_game_counter_token == "252↓"


def test_unresolvable_player_round_stays_unknown() -> None:
    fixture = at.parse_season_page(CAPTURED.read_bytes(), season=2026)
    page = player_html("X Y", "01-Jan-2000", [("Sydney", 2026, [G("1", "West Coast", "GF", "W")])])
    pp = at.parse_player_page(page, page_url="https://afltables.com/afl/stats/players/X/X_Y.html")
    (r,) = at.resolve_player_games(pp.games, fixture)
    assert r.match_date is None and r.date_quality is DateQuality.UNKNOWN and r.match_id is None


def test_player_page_missing_birth_date_is_unknown_not_default() -> None:
    page = player_html("X Y", "", [("Sydney", 2026, [G("1", "Carlton", "1")])]).replace("<b>Born:</b>", "")
    pp = at.parse_player_page(page, page_url="https://afltables.com/afl/stats/players/X/X_Y.html")
    assert pp.birth_date is None


def test_player_page_without_game_tables_is_fail() -> None:
    pp = at.parse_player_page("<html><h1>X</h1></html>", page_url="https://afltables.com/afl/stats/players/X/X_Y.html")
    assert pp.outcome is CheckOutcome.FAIL


def test_club_resolver_yields_shared_match_ids_and_chronological_stage_order() -> None:
    from supercoach_via.domain.ids import ClubRegistry

    reg = ClubRegistry.from_csv(Path(__file__).resolve().parents[3] / "config" / "team_aliases.csv")
    fx = at.parse_season_page(CAPTURED.read_bytes(), season=2026, club_resolver=reg.resolve)
    assert fx.outcome is CheckOutcome.PASS, fx.issues
    first = fx.matches[0]
    assert first.match_id == "m:2026:r01:carlton:sydney:0"
    assert first.to_row(source_ref="s", source_sha256=None)["home_club_id"] == "sydney"
    wf = next(m for m in fx.matches if m.stage.code == "WF")
    assert wf.match_id.startswith("m:2026:wf:")
    r25 = next(m for m in fx.matches if m.stage.stage_id == "r25")
    assert r25.stage.stage_order < wf.stage.stage_order


def test_unresolved_club_is_blocking_with_resolver() -> None:
    fx = at.parse_season_page(CAPTURED.read_bytes(), season=2026, club_resolver=lambda n, s: None)
    assert fx.outcome is CheckOutcome.FAIL


GAMES = CAPTURED.parent


@pytest.mark.parametrize(
    ("game_id", "names"),
    [("131820260329", {"Brodie, Will"}), ("091020260406", {"Perez, Flynn", "Dalton, Jack"})],
)
def test_captured_2026_match_pages_parse_and_agree_with_the_season_fixture(game_id: str, names: set[str]) -> None:
    """Real markup captured by the owner-authorised B1 fetch. The synthetic markup missed
    the ``13.12.<b>90</b>`` final-score cell, which renders as "13.12. 90"."""
    page = GAMES / f"game_2026_{game_id}_captured_20260925.html"
    d = at.parse_match_detail(page.read_bytes(), season=2026, game_id=game_id)
    assert d.outcome is CheckOutcome.PASS, d.issues
    fx = at.parse_season_page(CAPTURED.read_bytes(), season=2026)
    (m,) = [x for x in fx.matches if x.source_game_id == game_id]
    got = (d.home_name, d.away_name, d.home_score, d.away_score)
    assert got == (m.home_name, m.away_name, m.home_score, m.away_score)
    assert len(d.players) >= 44 and all(p.player_url for p in d.players)
    assert names <= {p.source_name for p in d.players}
