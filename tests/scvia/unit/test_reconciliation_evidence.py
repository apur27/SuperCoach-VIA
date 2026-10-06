"""Brownlow award evidence: its own exact-path policy, polite acquisition, and a fail-closed reader (A9, T16, T35)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from supercoach_via.ingest.http import UrlRejectedError, validate_url
from supercoach_via.reconciliation import evidence as EV
from tests.scvia.unit.recon_site import ERROR_PAGE, FakeClock, FakeSite

FOOTER = (
    "No Medal awarded 1942-1945 due to WWII Voting systems: 1924-1930:One vote per game (equiv. to 3 vote status in "
    "3-2-1 tallies) 1931-present:Six votes per game:3,2,1; (except 1976-77:12 votes per game:3,2,1 from each of two "
    "field umpires) * Ave using adjusted totals"
)


def brownlow_page(
    footer: str = FOOTER, years: list[int] | None = None, title: str = "AFL Tables - Brownlow Medal Winners"
) -> bytes:
    ys = years if years is not None else [y for y in range(1924, 2026) if not 1942 <= y <= 1945]
    rows = "".join(
        f"<tr><td>{y}</td><td>Some Player</td><td>Carlton</td><td>20</td><td>18</td><td>7</td><td>3</td><td>2</td>"
        f"<td>12</td><td>1.11</td></tr>"
        for y in reversed(ys)
    )
    head = "".join(f"<th>{c}</th>" for c in ("Year", "Player", "Team", "Votes", "GM", "3", "2", "1", "GP", "Ave"))
    return (
        f"<html><title>{title}</title><body><h1>Brownlow Medal Winners</h1><table><tr>{head}</tr>{rows}"
        f"<tr><td colspan=10>{footer}</td></tr></table></body></html>"
    ).encode()


# -- the reader --------------------------------------------------------------------------------


def test_the_reader_derives_no_award_seasons_and_votes_per_game_from_the_page_text() -> None:
    f = EV.parse_brownlow(brownlow_page())
    assert f.no_award_seasons == frozenset({1942, 1943, 1944, 1945})
    assert [f.award_total(y) for y in (1927, 1931, 1975, 1976, 1977, 1978, 1984, 2026)] == [1, 6, 6, 12, 12, 6, 6, 6]
    assert f.award_total(1943) is None and f.award_total(1900) is None


def test_the_real_captured_page_parses_to_the_award_seasons_and_votes_per_game() -> None:
    fx = Path(__file__).resolve().parents[1] / "fixtures" / "reconciliation" / "brownlow_idx.html"
    f = EV.parse_brownlow(fx.read_bytes())
    # read by hand from the footer: "No Medal awarded 1942-1945 ... 1924-1930: One vote per game ...
    # 1931-present: Six votes per game ... (except 1976-77: 12 votes per game ...)"
    assert f.no_award_seasons == frozenset({1942, 1943, 1944, 1945})
    assert [(e.first, e.last, e.total) for e in f.eras] == [(1924, 1930, 1), (1931, None, 6)]
    assert [(e.first, e.last, e.total) for e in f.exceptions] == [(1976, 1977, 12)]
    assert min(f.winner_years) == 1924 and not (f.winner_years & f.no_award_seasons)


def test_a_footer_that_disagrees_with_the_winners_table_is_refused() -> None:
    with pytest.raises(EV.EvidenceError, match="disagree"):
        EV.parse_brownlow(brownlow_page(footer=FOOTER.replace("1942-1945", "1942-1944")))


@pytest.mark.parametrize(
    "footer",
    [
        "No Medal awarded 1942-1945 due to WWII",  # no voting systems statement
        FOOTER.replace(
            "1931-present:Six votes per game", "1931-1999:Six votes per game"
        ),  # nothing runs to the present
        FOOTER.replace("Six votes", "Many votes"),  # not a number
    ],
)
def test_a_page_the_reader_cannot_understand_raises_instead_of_guessing(footer: str) -> None:
    with pytest.raises(EV.EvidenceError):
        EV.parse_brownlow(brownlow_page(footer=footer))


def test_a_page_without_the_winners_table_is_refused() -> None:
    with pytest.raises(EV.EvidenceError, match="winners table"):
        EV.parse_brownlow(b"<html><table><tr><td>nothing</td></tr></table></html>")


def test_stored_evidence_must_still_hash_to_the_pinned_digest(tmp_path: Path) -> None:
    body = brownlow_page()
    from supercoach_via.ingest.http import RawArchive

    sha = RawArchive(tmp_path).put(body)
    assert EV.load_evidence(tmp_path, sha) == body
    obj = tmp_path / "objects" / sha[:2] / sha
    obj.chmod(0o644)
    obj.write_bytes(body + b"x")
    with pytest.raises(EV.EvidenceError, match="missing or altered"):
        EV.load_evidence(tmp_path, sha)


# -- the policy ----------------------------------------------------------------------------------


def test_the_evidence_policy_allows_exactly_the_brownlow_index_and_robots() -> None:
    pol = EV.evidence_policies()
    assert validate_url(EV.BROWNLOW_URL, pol).name == EV.EVIDENCE_POLICY_NAME
    assert validate_url(EV.ROBOTS_URL, pol).name == EV.EVIDENCE_POLICY_NAME
    for url in (
        "https://afltables.com/afl/brownlow/brownlow_1999.html",
        "https://afltables.com/afl/brownlow/brownlow_idx.html?x=1",
        "https://afltables.com/afl/stats/notes.html",
        "https://afltables.com/afl/afl_index.html",
        "http://afltables.com/afl/brownlow/brownlow_idx.html",
    ):
        with pytest.raises(UrlRejectedError):
            validate_url(url, pol)
    p = pol.sources[EV.EVIDENCE_POLICY_NAME]
    assert p.requests_per_second == 0.5 and p.max_concurrent_per_host == 1 and p.max_attempts == 1


def test_packaged_evidence_policy_equals_the_checkout_copy() -> None:
    repo = Path(__file__).resolve().parents[3]
    name = EV.EVIDENCE_POLICY_FILE
    assert (repo / "config" / name).read_bytes() == (repo / "src" / "supercoach_via" / "config" / name).read_bytes()


# -- acquisition -----------------------------------------------------------------------------------


def _capture(
    tmp_path: Path, pages: dict[str, bytes] | None = None, **override: object
) -> tuple[EV.EvidenceResult, FakeSite]:
    clock = FakeClock()
    site = FakeSite(clock, {"/afl/brownlow/brownlow_idx.html": brownlow_page(), **(pages or {})})
    site.override.update(override)  # type: ignore[arg-type]
    cap = EV.EvidenceCapture(tmp_path, site.client(EV.evidence_policies(), EV.USER_AGENT), clock=clock)
    return cap.run(), site


def test_capture_fetches_robots_then_the_page_two_seconds_apart_and_stores_a_verifiable_object(tmp_path: Path) -> None:
    res, site = _capture(tmp_path)
    assert res.exit_code == 0 and res.state == "complete" and res.requests == 2
    assert site.paths() == ["/robots.txt", "/afl/brownlow/brownlow_idx.html"]
    t = site.times()
    assert t[1] - t[0] >= EV.MIN_SPACING_S
    assert site.log[0][2]["user-agent"] == EV.USER_AGENT and "if-none-match" not in site.log[1][2]
    body = EV.load_evidence(tmp_path / "evidence", res.sha256 or "")
    assert EV.parse_brownlow(body).no_award_seasons == frozenset({1942, 1943, 1944, 1945})
    receipt = json.loads((tmp_path / "evidence" / "receipt.json").read_text())
    assert receipt["state"] == "complete" and receipt["sha256"] == res.sha256
    obs = [json.loads(x) for x in (tmp_path / "evidence" / "observations.jsonl").read_text().splitlines()]
    assert [o["url"].rsplit("/", 1)[-1] for o in obs] == ["robots.txt", "brownlow_idx.html"]


def test_robots_that_disallows_the_page_stops_before_any_request_for_it(tmp_path: Path) -> None:
    res, site = _capture(tmp_path, {"/robots.txt": b"User-agent: *\nDisallow: /afl/brownlow/\n"})
    assert res.state == "blocked" and res.exit_code == 8 and site.paths() == ["/robots.txt"]


def test_an_unretrievable_robots_file_pauses_with_no_page_request(tmp_path: Path) -> None:
    res, site = _capture(tmp_path, **{"/robots.txt": lambda _r: httpx.Response(503)})
    assert res.state == "paused" and "robots.txt" in (res.reason or "") and site.paths() == ["/robots.txt"]


def test_a_403_stops_rather_than_retrying(tmp_path: Path) -> None:
    res, site = _capture(tmp_path, **{"/afl/brownlow/brownlow_idx.html": lambda _r: httpx.Response(403)})
    assert res.state == "blocked" and site.paths().count("/afl/brownlow/brownlow_idx.html") == 1


def test_a_long_retry_after_pauses_and_a_flaky_page_is_bounded_to_a_handful_of_requests(tmp_path: Path) -> None:
    long_wait = {"/afl/brownlow/brownlow_idx.html": lambda _r: httpx.Response(429, headers={"Retry-After": "3600"})}
    res, _site = _capture(tmp_path / "a", **long_wait)
    assert res.state == "paused" and "Retry-After" in (res.reason or "")
    flaky = {"/afl/brownlow/brownlow_idx.html": lambda _r: httpx.Response(500)}
    res2, site2 = _capture(tmp_path / "b", **flaky)
    assert res2.state == "failed" and len(site2.log) <= EV.MAX_REQUESTS
    gaps = [b - a for a, b in zip(site2.times(), site2.times()[1:], strict=False)]
    assert all(g >= EV.MIN_SPACING_S for g in gaps)


def test_a_200_that_is_not_the_brownlow_index_is_not_accepted(tmp_path: Path) -> None:
    res, _ = _capture(tmp_path, {"/afl/brownlow/brownlow_idx.html": ERROR_PAGE})
    assert res.state == "failed" and "identity" in (res.reason or "")


def test_a_page_the_reader_cannot_understand_is_stored_but_not_complete(tmp_path: Path) -> None:
    res, _ = _capture(tmp_path, {"/afl/brownlow/brownlow_idx.html": brownlow_page(footer="No Medal awarded 1942-1945")})
    assert res.state == "failed" and "not understood" in (res.reason or "")
    assert json.loads((tmp_path / "evidence" / "receipt.json").read_text())["sha256"]


def test_a_second_writer_is_refused(tmp_path: Path) -> None:
    import fcntl

    (tmp_path / "evidence").mkdir()
    with (tmp_path / "evidence" / ".lock").open("wb") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        with pytest.raises(EV.BusyError):
            _capture(tmp_path)


def test_the_capture_refuses_a_policy_that_is_not_its_own(tmp_path: Path) -> None:
    from supercoach_via.reconciliation.urls import load_reconciliation_policies

    clock = FakeClock()
    site = FakeSite(clock, {})
    with pytest.raises(EV.EvidenceError, match="own policy"):
        EV.EvidenceCapture(tmp_path, site.client(load_reconciliation_policies()), clock=clock)
