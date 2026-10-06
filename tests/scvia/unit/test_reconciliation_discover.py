"""Census / discovery link extraction and page identity (DESIGN section 6, T02, T16, T31)."""

from __future__ import annotations

from pathlib import Path

import pytest

from supercoach_via.reconciliation import discover as D

FX = Path(__file__).resolve().parents[1] / "fixtures" / "reconciliation"
SITE = "https://afltables.com"


def fx(name: str) -> bytes:
    return (FX / name).read_bytes()


def test_stats_index_lists_all_130_seasons_and_the_all_players_directory() -> None:
    info = D.read_stats_index(fx("stats_idx.html"), SITE + "/afl/stats/stats_idx.html")
    assert info.seasons == list(range(1897, 2027))
    assert info.all_players_url == SITE + "/afl/stats/playersA_idx.html"
    assert info.notes_url == SITE + "/afl/stats/notes.html"
    assert info.problems == []


def test_census_letter_page_declares_its_letter_navigation_and_profile_links() -> None:
    page = D.read_census_page(fx("census_A.html"), SITE + "/afl/stats/playersA_idx.html")
    assert page.letter == "A"
    assert [x for x, _ in page.nav] == [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    assert page.nav[0][1] is None  # the current letter is plain text, not a link
    assert page.nav[1] == ("B", SITE + "/afl/stats/playersB_idx.html")
    assert page.profiles[0] == D.ProfileLink(
        display_name="Aanensen, Vic", url=SITE + "/afl/stats/players/V/Vic_Aanensen.html"
    )
    assert len(page.profiles) == len({p.url for p in page.profiles}) == 20  # fixture keeps 4 rows of 5 cells


def test_census_page_with_the_wrong_letter_or_missing_navigation_is_a_problem() -> None:
    html = fx("census_A.html").replace(b"All Players - A", b"All Players - B")
    page = D.read_census_page(html, SITE + "/afl/stats/playersA_idx.html")
    assert any("letter" in p for p in page.problems)
    cut = fx("census_A.html").replace(b'<a href="playersQ_idx.html">Q</a>', b"[Q]")
    page2 = D.read_census_page(cut, SITE + "/afl/stats/playersA_idx.html")
    assert any("navigation" in p for p in page2.problems)


def test_season_page_fixtures_carry_game_links_and_dates() -> None:
    info = D.read_season_page(fx("season_1948_head.html"), SITE + "/afl/seas/1948.html")
    assert info.year == 1948
    assert info.fixtures, info.problems
    first = info.fixtures[0]
    assert first.game_url.startswith(SITE + "/afl/stats/games/1948/")
    assert first.match_date is not None and first.match_date.year == 1948
    assert info.problems == []


def test_match_page_lists_lineup_profile_links_only() -> None:
    urls = D.match_profile_links(fx("match_2021_r6.html"), SITE + "/afl/stats/games/2021/162020210424.html")
    assert SITE + "/afl/stats/players/B/Ben_Ainsworth.html" in urls
    assert all("/afl/stats/players/" in u for u in urls)
    assert urls == sorted(set(urls))


def test_profile_game_links_come_from_game_cells_and_resolve_to_absolute_urls() -> None:
    urls = D.profile_game_links(fx("profile_robinson.html"), SITE + "/afl/stats/players/K/Kelly_Robinson.html")
    assert urls
    assert all(u.startswith(SITE + "/afl/stats/games/") for u in urls)
    assert urls == sorted(set(urls))


@pytest.mark.parametrize(
    ("kind", "fixture", "url"),
    [
        ("stats_index", "stats_idx.html", SITE + "/afl/stats/stats_idx.html"),
        ("notes", "notes.html", SITE + "/afl/stats/notes.html"),
        ("letter", "census_A.html", SITE + "/afl/stats/playersA_idx.html"),
        ("season", "season_1948_head.html", SITE + "/afl/seas/1948.html"),
        ("match", "match_2021_r6.html", SITE + "/afl/stats/games/2021/162020210424.html"),
        ("profile", "profile_robinson.html", SITE + "/afl/stats/players/K/Kelly_Robinson.html"),
    ],
)
def test_real_pages_have_their_expected_identity(kind: str, fixture: str, url: str) -> None:
    assert D.page_problem(kind, fx(fixture), url) is None


@pytest.mark.parametrize(
    ("kind", "body", "url"),
    [
        (
            "profile",
            b"<html><head><title>Broked!</title></head><body>This page has been sent off</body></html>",
            SITE + "/afl/stats/players/G/Gary_Ablett.html",
        ),
        (
            "match",
            b"<html><title>Just a moment...</title><body>cf-chl-bypass</body></html>",
            SITE + "/afl/stats/games/2021/162020210424.html",
        ),
        (
            "letter",
            b"<html><title>AFL Tables - All Players - C</title><H1>All Players - C</H1></html>",
            SITE + "/afl/stats/playersB_idx.html",
        ),
        (
            "season",
            b"<html><title>AFL Tables -  2011 Season Scores</title><h1> 2011 Season Scores and Results</h1></html>",
            SITE + "/afl/seas/2010.html",
        ),
        ("notes", b"", SITE + "/afl/stats/notes.html"),
        (
            "profile",
            b"<html><title>Please log in</title><h1>Sign in</h1><input type=password></html>",
            SITE + "/afl/stats/players/K/Kelly_Robinson.html",
        ),
    ],
)
def test_error_challenge_login_and_wrong_pages_are_unusable_even_with_http_200(
    kind: str, body: bytes, url: str
) -> None:
    assert D.page_problem(kind, body, url) is not None


def test_relative_game_link_resolves_against_the_final_url_not_the_requested_one() -> None:
    html = b'<a href="../../games/2006/041920060603.html">1</a>'
    final = SITE + "/afl/stats/players/S/Scott_Pendlebury.html"
    assert D.profile_game_links(html, final) == [SITE + "/afl/stats/games/2006/041920060603.html"]
