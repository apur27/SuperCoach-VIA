"""Bounded player-page repair (owner-authorised B1 repair path; PLAN 4.1, 4.4, 5.1, 5.2).

``refresh.repair_player_pages`` fills a named player's missing rows for one season from
that player's AFLTables page, discovered through a match page the base already has:

- request budget is checked before any network use;
- the href is taken from the source match page, never guessed from the name;
- identity is proven: an existing identity must match the page's birth date; a new
  identity must not share a birth date with a same-name identity it must not be;
- rows only fill (match, player) pairs absent from the base; existing rows are compared,
  never overwritten; unresolvable page games block;
- every row carries source provenance (URL, sha256, fetched-at) and no number is typed.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from supercoach_via.domain.schemas import CheckOutcome
from supercoach_via.ingest import afltables as at
from supercoach_via.ingest import refresh as rf
from tests.scvia.unit.test_afltables_support import G, M, P, Stage, match_html, player_html, season_html
from tests.scvia.unit.test_refresh import QA, QB, Site, _resolver, make_context

GID_A, GID_B = "000120260305", "000220260314"
URL_NEW = at.player_url("N/New_Hawk")
URL_OLD = at.player_url("O/Old_Perez")


def _site(*, old_born: str = "25-Aug-2001") -> Site:
    site = Site()
    site.pages[at.season_url(2026)] = season_html(
        2026,
        [
            Stage("Round 1", [M("Hawthorn", "Carlton", "Thu 05-Mar-2026 7:30 PM", "M.C.G.", QA, QB, GID_A)]),
            Stage("Round 2", [M("Geelong", "Hawthorn", "Sat 14-Mar-2026 7:30 PM", "M.C.G.", QB, QA, GID_B)]),
        ],
    ).encode()
    site.pages[at.match_url(2026, GID_A)] = match_html(
        season=2026, round_label="1", home="Hawthorn", away="Carlton", home_q=QA, away_q=QB,
        date="Thu, 05-Mar-2026 7:30 PM",
        home_players=[P("H/Hawk_One", "One, Hawk", "1", {"KI": "5", "HB": "5", "DI": "10"}),
                      P("N/New_Hawk", "Hawk, New", "2", {"KI": "7", "HB": "4", "DI": "11"}),
                      P("O/Old_Perez", "Perez, Old", "3", {"KI": "6", "HB": "3", "DI": "9"})],
        away_players=[P("C/Carlton_One", "One, Carlton", "1", {"KI": "8", "HB": "8", "DI": "16"})],
    ).encode()  # fmt: skip
    site.pages[URL_NEW] = player_html(
        "New Hawk", "01-Feb-2005",
        [("Hawthorn", 2026, [G("1", "Carlton", "1", "W", "2", {"KI": "7", "HB": "4", "DI": "11", "GL": "1"}),
                             G("2", "Geelong", "2", "W", "2", {"KI": "3", "HB": "2", "DI": "5"})])],
    ).encode()  # fmt: skip
    site.pages[URL_OLD] = player_html(
        "Old Perez", old_born,
        [("North Melbourne", 2023, [G("1", "Carlton", "4", "L", "39", {"KI": "1", "HB": "1", "DI": "2"})]),
         ("Hawthorn", 2026, [G("2", "Carlton", "1", "W", "3", {"KI": "6", "HB": "3", "DI": "9"})])],
    ).encode()  # fmt: skip
    return site


def _base(site: Site, *, old_row: dict[str, object] | None = None) -> rf.BaseState:
    fx = at.parse_season_page(site.pages[at.season_url(2026)], season=2026, club_resolver=_resolver)
    matches = {
        m.match_id: rf.BaseMatch(m.match_id, 2026, "complete", m.stage.stage_id, m.home_name, m.away_name,
                                 m.home_score, m.away_score, m.local_start)
        for m in fx.matches
    }  # fmt: skip
    a = next(m.match_id for m in fx.matches if m.source_game_id == GID_A)
    rows = {a: [{"player_id": "legacy:hawk_one", "club_id": "hawthorn", "disposals": 10}]}
    if old_row is not None:
        rows[a].append(old_row)
    return rf.BaseState(snapshot_id="sha256:" + "1" * 64, matches=matches,
                        load_player_rows=lambda season, mid: list(rows.get(mid, [])))  # fmt: skip


def _targets(match_id: str, *, old_dob: date = date(2001, 8, 25)) -> list[rf.PlayerRepairTarget]:
    return [
        rf.PlayerRepairTarget("New Hawk", match_id, "Hawthorn", None, None, (date(1876, 4, 15),)),
        rf.PlayerRepairTarget("Old Perez", match_id, "Hawthorn", "legacy:perez_old_25082001", old_dob),
    ]


def _match_a(base: rf.BaseState) -> str:
    return next(mid for mid, m in base.matches.items() if m.stage_id == "r01")


EXISTING = {"legacy:perez_old_25082001": {"player_id": "legacy:perez_old_25082001", "display_name": "Old Perez",
                                          "source_urls": None}}  # fmt: skip


def _run(site: Site, base: rf.BaseState, tmp_path: Path, **kw: object) -> rf.RepairResult:
    targets = kw.pop("targets", None) or _targets(_match_a(base))
    return rf.repair_player_pages(
        base, 2026, targets, make_context(site, tmp_path), club_resolver=_resolver,  # type: ignore[arg-type]
        existing_players=EXISTING, **kw,  # type: ignore[arg-type]
    )


def test_fills_missing_rows_with_provenance_and_binds_identities(tmp_path: Path) -> None:
    site = _site()
    base = _base(site)
    res = _run(site, base, tmp_path)
    assert res.outcome is CheckOutcome.PASS and res.exit_code == 0
    assert sum(res.request_counts.values()) == 4 and len(site.hits) == 4
    pg = sorted(res.upserts["player_games"], key=lambda r: (r["player_id"], r["stage_id"]))
    assert [(r["player_id"], r["stage_id"], r["disposals"]) for r in pg] == [
        ("legacy:perez_old_25082001", "r01", 9),
        ("src:afltables:N.New_Hawk", "r01", 11),
        ("src:afltables:N.New_Hawk", "r02", 5),
    ]
    for r in pg:
        assert r["provenance"] == "source_fetch" and r["date_quality"] == "fixture_verified"
        assert r["source_path"] in (URL_NEW, URL_OLD) and r["source_sha256"] and r["available_at"] is not None
        assert r["club_id"] == "hawthorn" and r["season"] == 2026
    new = next(r for r in pg if r["stage_id"] == "r02")
    assert new["opponent_club_id"] == "geelong" and new["career_game_counter"] == 2
    players = {p["player_id"]: p for p in res.upserts["players"]}
    assert players["src:afltables:N.New_Hawk"]["birth_date"] == date(2005, 2, 1)
    assert json.loads(players["legacy:perez_old_25082001"]["source_urls"]) == [URL_OLD]
    ev = {e["display_name"]: e for e in res.evidence}
    assert ev["Old Perez"]["page_seasons"] == {"2023": 1, "2026": 1}
    assert ev["Old Perez"]["rows_added"] == 1 and ev["New Hawk"]["rows_added"] == 2
    assert ev["New Hawk"]["player_url"] == URL_NEW and len(ev["New Hawk"]["page_sha256"]) == 64
    assert {o["url"] for o in res.upserts["source_observations"]} >= {URL_NEW, URL_OLD, at.season_url(2026)}


def test_budget_checked_before_any_request(tmp_path: Path) -> None:
    site = _site()
    base = _base(site)
    with pytest.raises(rf.RefreshConfigError, match="budget"):
        _run(site, base, tmp_path, max_requests=3)
    assert not site.hits


def test_existing_identity_with_wrong_birth_date_is_refused(tmp_path: Path) -> None:
    site = _site(old_born="01-Jan-1990")
    base = _base(site)
    res = _run(site, base, tmp_path)
    assert res.outcome is CheckOutcome.FAIL and res.exit_code == 3
    assert all(r["player_id"] != "legacy:perez_old_25082001" for r in res.upserts["player_games"])
    assert {e["display_name"]: e["status"] for e in res.evidence}["Old Perez"] == "identity_mismatch"


def test_new_identity_must_not_be_an_excluded_namesake(tmp_path: Path) -> None:
    site = _site()
    base = _base(site)
    targets = [rf.PlayerRepairTarget("New Hawk", _match_a(base), "Hawthorn", None, None, (date(2005, 2, 1),))]
    res = _run(site, base, tmp_path, targets=targets)
    assert res.outcome is CheckOutcome.FAIL and not res.upserts["player_games"]
    assert res.evidence[0]["status"] == "identity_mismatch"


def test_existing_row_is_compared_not_overwritten(tmp_path: Path) -> None:
    site = _site()
    old = {"player_id": "legacy:perez_old_25082001", "club_id": "hawthorn", "disposals": 99, "kicks": 6}
    base = _base(site, old_row=old)
    res = _run(site, base, tmp_path)
    assert all(r["player_id"] != "legacy:perez_old_25082001" for r in res.upserts["player_games"])
    ev = {e["display_name"]: e for e in res.evidence}["Old Perez"]
    assert ev["rows_added"] == 0 and ev["rows_conflicting"] == 1
    assert res.outcome is CheckOutcome.FAIL  # a disagreement with the base is not silently accepted


def test_unreachable_source_is_unknown_and_writes_no_rows(tmp_path: Path) -> None:
    site = _site()
    site.status[at.season_url(2026)] = 503
    base = _base(site)
    res = _run(site, base, tmp_path)
    assert res.outcome in (CheckOutcome.UNKNOWN, CheckOutcome.FAIL) and res.exit_code == 3
    assert not res.upserts["player_games"] and not res.upserts["players"]
