"""H01-H08: shared HTTP policy (PLAN 5.1). Hermetic: httpx.MockTransport + fake clock."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from supercoach_via.domain.schemas import TABLES, CheckOutcome, SourceMode
from supercoach_via.ingest.http import (
    FetchResult,
    HostRateLimiter,
    HttpClient,
    PolicySet,
    RawArchive,
    UrlRejectedError,
    load_policies,
    manual_fixture_result,
    parse_policies,
    validate_url,
)

REPO = Path(__file__).resolve().parents[3]
SEASON_URL = "https://afltables.com/afl/seas/2026.html"
PUBLIC_IP = ["93.184.215.14"]  # resolver stub; never real DNS in unit tests


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.t += seconds


def _now() -> datetime:
    return datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def public_resolver(_host: str) -> list[str]:
    return list(PUBLIC_IP)


@pytest.fixture
def policies() -> PolicySet:
    return load_policies(REPO / "config" / "source_policies.toml")


def make_client(
    policies: PolicySet,
    handler: Callable[[httpx.Request], httpx.Response],
    tmp_path: Path,
    *,
    resolver: Callable[[str], list[str]] = public_resolver,
    clock: FakeClock | None = None,
) -> tuple[HttpClient, FakeClock]:
    clock = clock or FakeClock()
    client = HttpClient(
        policies,
        user_agent="supercoach-via-test/1.0",
        archive=RawArchive(tmp_path / "raw"),
        transport=httpx.MockTransport(handler),
        resolver=resolver,
        monotonic=clock,
        sleep=clock.sleep,
        now=_now,
        jitter=lambda: 0.5,
    )
    return client, clock


# ---------------------------------------------------------------------------
# Policy file and URL validation (H06, S06)
# ---------------------------------------------------------------------------


def test_policy_file_declares_expected_hosts_and_defaults(policies: PolicySet) -> None:
    hosts = {p.host for p in policies.sources.values()}
    assert hosts == {
        "afltables.com",
        "en.wikipedia.org",
        "www.draftguru.com.au",
        "www.afl.com.au",
        "www.zerohanger.com",
        "www.fanfooty.com.au",
    }
    p = policies.sources["afltables"]
    assert p.requests_per_second == 2.0
    assert p.max_concurrent_per_host == 2
    assert (p.connect_timeout_s, p.read_timeout_s) == (10.0, 30.0)
    assert p.max_attempts == 3
    assert p.max_response_bytes == 10 * 1024 * 1024
    assert policies.live_poll_interval_s == 90


def test_policy_rejects_unknown_keys_and_http_hosts() -> None:
    with pytest.raises(ValueError):
        parse_policies('[defaults]\nbogus = 1\n[sources.x]\nhost="a.com"\npath_patterns=["^/$"]\n')
    with pytest.raises(ValueError):
        parse_policies('[sources.x]\nhost="http://a.com"\npath_patterns=["^/$"]\n')


@pytest.mark.parametrize(
    "url",
    [
        "http://afltables.com/afl/seas/2026.html",  # not https
        "https://user:pw@afltables.com/afl/seas/2026.html",  # credentials
        "https://evil.example/afl/seas/2026.html",  # host not allowed
        "https://afltables.com.evil.example/afl/seas/2026.html",
        "https://afltables.com:8443/afl/seas/2026.html",  # non-default port
        "https://afltables.com/afl/seas/../../etc/passwd",  # traversal
        "https://afltables.com/afl/seas/%2e%2e/2026.html",  # encoded traversal
        "https://afltables.com/afl/stats/players/S/../x.html",
        "https://afltables.com/afl/seas/2026.html?x=1",  # query not in grammar
        "https://afltables.com/etc/passwd",  # outside path grammar
        "https://127.0.0.1/afl/seas/2026.html",
        "",
    ],
)
def test_validate_url_rejects_hostile_urls(policies: PolicySet, url: str) -> None:
    with pytest.raises(UrlRejectedError):
        validate_url(url, policies)


def test_validate_url_accepts_grammar(policies: PolicySet) -> None:
    assert validate_url(SEASON_URL, policies).name == "afltables"
    assert validate_url("https://afltables.com/afl/stats/players/S/Scott_Pendlebury.html", policies).name == "afltables"
    assert validate_url("https://www.fanfooty.com.au/live/9781.txt", policies).name == "fanfooty"


# ---------------------------------------------------------------------------
# Happy path, archive and observation rows
# ---------------------------------------------------------------------------


def test_successful_fetch_archives_payload_and_observation(policies: PolicySet, tmp_path: Path) -> None:
    body = b"<html>season</html>"

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.headers["user-agent"] == "supercoach-via-test/1.0"
        return httpx.Response(200, content=body, headers={"ETag": '"v1"'})

    client, _ = make_client(policies, handler, tmp_path)
    res = client.fetch(SEASON_URL)
    assert res.outcome is CheckOutcome.PASS and res.ok
    assert res.content == body and res.bytes == len(body)
    assert res.source_mode is SourceMode.LIVE and res.freshness == "fresh"
    assert res.attempts == 1 and res.http_status == 200
    assert res.sha256 is not None and client.archive is not None
    assert client.archive.get(res.sha256) == body
    obs = res.observation("afltables.season", "1")
    assert set(obs) == set(TABLES["source_observations"].column_names)
    assert obs["outcome"] == "PASS" and obs["etag"] == '"v1"' and obs["bytes"] == len(body)
    assert client.request_counts == {"afltables.com": 1}


# ---------------------------------------------------------------------------
# H01-H05: failures, retries, backoff, limiter
# ---------------------------------------------------------------------------


def test_404_is_fail_without_retry(policies: PolicySet, tmp_path: Path) -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(404, content=b"nope")

    client, _ = make_client(policies, handler, tmp_path)
    res = client.fetch(SEASON_URL)
    assert res.outcome is CheckOutcome.FAIL and not res.ok
    assert res.content is None and len(calls) == 1 and res.http_status == 404


def test_5xx_retries_with_exponential_backoff_then_unknown(policies: PolicySet, tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client, clock = make_client(policies, handler, tmp_path)
    res = client.fetch(SEASON_URL)
    assert res.outcome is CheckOutcome.UNKNOWN and not res.ok and res.content is None
    assert res.attempts == 3
    backoffs = [s for s in clock.sleeps if s >= 1.0]
    # base 1s * 2**n with jitter factor (0.5 + 0.5) = 1.0 -> 1s, 2s
    assert backoffs == [1.0, 2.0]


def test_429_honours_bounded_retry_after(policies: PolicySet, tmp_path: Path) -> None:
    seq = iter([httpx.Response(429, headers={"Retry-After": "9999"}), httpx.Response(200, content=b"ok")])
    client, clock = make_client(policies, lambda r: next(seq), tmp_path)
    res = client.fetch(SEASON_URL)
    assert res.ok and res.attempts == 2
    assert max(clock.sleeps) == 60.0  # capped at retry_after_max_s


def test_timeout_then_success(policies: PolicySet, tmp_path: Path) -> None:
    state = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ReadTimeout("slow", request=req)
        return httpx.Response(200, content=b"ok")

    client, _ = make_client(policies, handler, tmp_path)
    res = client.fetch(SEASON_URL)
    assert res.ok and res.attempts == 2


def test_connect_errors_exhaust_to_unknown_never_empty_success(policies: PolicySet, tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns blocked", request=req)

    client, _ = make_client(policies, handler, tmp_path)
    res = client.fetch(SEASON_URL)
    assert res.outcome is CheckOutcome.UNKNOWN
    assert res.content is None and res.bytes == 0 and not res.ok
    assert "ConnectError" in (res.error or "")


def test_dns_failure_is_unknown(policies: PolicySet, tmp_path: Path) -> None:
    def resolver(host: str) -> list[str]:
        raise OSError("no dns")

    client, _ = make_client(policies, lambda r: httpx.Response(200), tmp_path, resolver=resolver)
    assert client.fetch(SEASON_URL).outcome is CheckOutcome.UNKNOWN


def test_oversized_body_aborts(policies: PolicySet, tmp_path: Path) -> None:
    big = b"x" * (policies.sources["fanfooty"].max_response_bytes + 1)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=httpx.ByteStream(big))

    client, _ = make_client(policies, handler, tmp_path)
    res = client.fetch("https://www.fanfooty.com.au/live/9781.txt")
    assert res.outcome is CheckOutcome.FAIL and res.content is None
    assert "oversized" in (res.error or "")


def test_declared_content_length_over_cap_rejected(policies: PolicySet, tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x", headers={"Content-Length": str(50 * 1024 * 1024)})

    client, _ = make_client(policies, handler, tmp_path)
    assert client.fetch(SEASON_URL).outcome is CheckOutcome.FAIL


def test_rate_limiter_spaces_requests_per_host() -> None:
    clock = FakeClock()
    lim = HostRateLimiter(monotonic=clock, sleep=clock.sleep)
    for _ in range(5):
        lim.acquire("afltables.com", 2.0)
    # first immediate, then 0.5 s spacing
    assert clock.sleeps == [0.5, 0.5, 0.5, 0.5]
    lim.acquire("www.fanfooty.com.au", 2.0)  # other host: independent bucket
    assert len(clock.sleeps) == 4


def test_rate_limiter_is_global_across_threads() -> None:
    clock = FakeClock()
    lim = HostRateLimiter(monotonic=clock, sleep=clock.sleep)
    threads = [threading.Thread(target=lim.acquire, args=("afltables.com", 2.0)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 8 requests at 2/s need >= 3.5 s of aggregate waiting in total spacing
    assert sum(clock.sleeps) >= 3.5 - 1e-9
    assert len(clock.sleeps) == 7  # every caller after the first had to wait


def test_client_limits_every_attempt(policies: PolicySet, tmp_path: Path) -> None:
    client, clock = make_client(policies, lambda r: httpx.Response(200, content=b"ok"), tmp_path)
    for _ in range(3):
        assert client.fetch(SEASON_URL).ok
    assert clock.sleeps == [0.5, 0.5]


# ---------------------------------------------------------------------------
# H06: redirects, private hosts
# ---------------------------------------------------------------------------


def test_redirect_to_unapproved_host_is_rejected(policies: PolicySet, tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        return httpx.Response(302, headers={"Location": "https://evil.example/x"})

    client, _ = make_client(policies, handler, tmp_path)
    res = client.fetch(SEASON_URL)
    assert res.outcome is CheckOutcome.FAIL and "redirect" in (res.error or "")
    assert seen == [SEASON_URL]  # never followed


def test_allowed_redirect_is_revalidated_and_followed(policies: PolicySet, tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/afl/seas/2026.html":
            return httpx.Response(301, headers={"Location": "/afl/seas/2025.html"})
        return httpx.Response(200, content=b"moved")

    client, _ = make_client(policies, handler, tmp_path)
    res = client.fetch(SEASON_URL)
    assert res.ok and res.final_url == "https://afltables.com/afl/seas/2025.html"


def test_redirect_loop_is_bounded(policies: PolicySet, tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": SEASON_URL})

    client, _ = make_client(policies, handler, tmp_path)
    res = client.fetch(SEASON_URL)
    assert res.outcome is CheckOutcome.FAIL and "redirect" in (res.error or "")
    assert client.request_counts["afltables.com"] == policies.sources["afltables"].max_redirects + 1


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.5",
        "192.168.1.1",
        "169.254.169.254",
        "::1",
        "fe80::1",
        "100.64.0.1",
        "::ffff:127.0.0.1",
        "0.0.0.0",
    ],
)
def test_private_or_local_resolution_is_refused_before_connecting(policies: PolicySet, tmp_path: Path, ip: str) -> None:
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(200)

    client, _ = make_client(policies, handler, tmp_path, resolver=lambda h: [ip])
    res = client.fetch(SEASON_URL)
    assert res.outcome is CheckOutcome.FAIL and calls == []


def test_rejected_url_returns_fail_without_request(policies: PolicySet, tmp_path: Path) -> None:
    calls = []
    client, _ = make_client(policies, lambda r: calls.append(r) or httpx.Response(200), tmp_path)
    res = client.fetch("https://afltables.com/../../etc/passwd")
    assert res.outcome is CheckOutcome.FAIL and calls == []


# ---------------------------------------------------------------------------
# Conditional requests (304) and fixture fallback (H07)
# ---------------------------------------------------------------------------


def test_conditional_304_uses_valid_archived_payload(policies: PolicySet, tmp_path: Path) -> None:
    state = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(200, content=b"v1", headers={"ETag": '"e1"', "Last-Modified": "Mon"})
        assert req.headers.get("if-none-match") == '"e1"'
        assert req.headers.get("if-modified-since") == "Mon"
        return httpx.Response(304)

    client, _ = make_client(policies, handler, tmp_path)
    first = client.fetch(SEASON_URL)
    second = client.fetch(SEASON_URL)
    assert second.ok and second.source_mode is SourceMode.CACHED
    assert second.content == b"v1" and second.sha256 == first.sha256
    assert second.freshness == "revalidated"


def test_304_without_valid_archive_is_fail(policies: PolicySet, tmp_path: Path) -> None:
    state = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            return httpx.Response(200, content=b"v1", headers={"ETag": '"e1"'})
        return httpx.Response(304)

    client, _ = make_client(policies, handler, tmp_path)
    first = client.fetch(SEASON_URL)
    assert first.sha256 and client.archive
    # corrupt the archived object: a 304 must not vouch for it
    client.archive.object_path(first.sha256).write_bytes(b"tampered")
    res = client.fetch(SEASON_URL)
    assert not res.ok
    # and an unsolicited 304 (no validators) is also a failure
    client2, _ = make_client(policies, lambda r: httpx.Response(304), tmp_path / "other")
    assert client2.fetch(SEASON_URL).outcome is CheckOutcome.FAIL


def test_manual_fixture_fallback_is_labelled_never_fresh() -> None:
    res = manual_fixture_result(
        "https://www.zerohanger.com/afl/players/off-contract-2026/",
        b"<html/>",
        source="zerohanger",
        source_date=date(2026, 6, 19),
        now=_now,
    )
    assert isinstance(res, FetchResult)
    assert res.source_mode is SourceMode.MANUAL_FIXTURE
    assert res.outcome is CheckOutcome.UNKNOWN and res.freshness == "unknown"
    assert res.source_date == date(2026, 6, 19)
    assert res.observation("contracts.zerohanger", "1")["source_mode"] == "manual_fixture"
