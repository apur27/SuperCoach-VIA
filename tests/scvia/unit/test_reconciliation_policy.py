"""T16/T35 support: the reconciliation-only source policy (DESIGN section 6, S-04, S-12, S-13)."""

from __future__ import annotations

from pathlib import Path

import pytest

from supercoach_via.ingest.http import UrlRejectedError, load_policies, validate_url
from supercoach_via.reconciliation.urls import (
    RECONCILIATION_POLICY_NAME,
    load_reconciliation_policies,
    normalise_link,
)

REPO = Path(__file__).resolve().parents[3]
BASE = "https://afltables.com/"


@pytest.fixture(scope="module")
def policies():  # type: ignore[no-untyped-def]
    return load_reconciliation_policies()


def test_policy_pins_the_s04_network_defaults(policies) -> None:  # type: ignore[no-untyped-def]
    p = policies.sources[RECONCILIATION_POLICY_NAME]
    assert p.host == "afltables.com"
    assert p.requests_per_second == 0.5
    assert p.max_concurrent_per_host == 1
    assert p.max_attempts == 1  # the coordinator owns retries


@pytest.mark.parametrize(
    "path",
    [
        "/robots.txt",
        "/afl/stats/stats_idx.html",
        "/afl/stats/playersA_idx.html",
        "/afl/stats/playersZ_idx.html",
        "/afl/stats/notes.html",
        "/afl/seas/1897.html",
        "/afl/seas/2026.html",
        "/afl/stats/games/2010/041520100925.html",
        "/afl/stats/players/S/Scott_Pendlebury.html",
        "/afl/stats/players/G/Gary_Ablett0.html",
    ],
)
def test_observed_paths_are_allowed(policies, path: str) -> None:  # type: ignore[no-untyped-def]
    assert validate_url("https://afltables.com" + path, policies).name == RECONCILIATION_POLICY_NAME


@pytest.mark.parametrize(
    "url",
    [
        "https://afltables.com/afl/stats/1992.html",  # optional per-season list (S-12), not adopted
        "https://afltables.com/afl/stats/1992s.html",
        "https://afltables.com/afl/stats/players/S/../Scott_Pendlebury.html",
        "https://afltables.com/afl/stats/players/s/Scott_Pendlebury.html",
        "https://afltables.com/afl/stats/playersAA_idx.html",
        "http://afltables.com/afl/stats/notes.html",
        "https://www.afltables.com/afl/stats/notes.html",
        "https://afltables.com/afl/stats/notes.html?x=1",
        "https://user:pw@afltables.com/afl/stats/notes.html",
        "https://afltables.com/afl/teams/carlton_idx.html",
        "https://example.com/afl/stats/notes.html",
    ],
)
def test_everything_else_is_rejected(policies, url: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(UrlRejectedError):
        validate_url(url, policies)


def test_production_policy_is_not_loosened() -> None:
    prod = load_policies(REPO / "config" / "source_policies.toml")
    assert prod.sources["afltables"].max_concurrent_per_host == 2  # untouched by this feature
    with pytest.raises(UrlRejectedError):
        validate_url("https://afltables.com/afl/stats/notes.html", prod)


def test_packaged_copy_equals_checkout_copy() -> None:
    a = (REPO / "config" / "reconciliation_source_policy.toml").read_bytes()
    b = (REPO / "src" / "supercoach_via" / "config" / "reconciliation_source_policy.toml").read_bytes()
    assert a == b


@pytest.mark.parametrize(
    ("href", "page", "expected"),
    [
        (
            "players/V/Vic_Aanensen.html",
            BASE + "afl/stats/playersA_idx.html",
            BASE + "afl/stats/players/V/Vic_Aanensen.html",
        ),
        (
            "../../games/2006/041920060603.html",
            BASE + "afl/stats/players/S/Scott_Pendlebury.html",
            BASE + "afl/stats/games/2006/041920060603.html",
        ),
        (
            "../../games/2006/041920060603.html#frag",
            BASE + "afl/stats/players/S/Scott_Pendlebury.html",
            BASE + "afl/stats/games/2006/041920060603.html",
        ),
        ("#20060", BASE + "afl/stats/players/S/Scott_Pendlebury.html", None),
        ("mailto:a@b.c", BASE + "afl/stats/notes.html", None),
        ("https://other.example/x.html", BASE + "afl/stats/notes.html", "https://other.example/x.html"),
    ],
)
def test_links_resolve_against_the_final_url_then_strip_fragment(href: str, page: str, expected: str | None) -> None:
    assert normalise_link(href, page) == expected
