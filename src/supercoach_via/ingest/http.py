"""One HTTP policy for every source (PLAN 5.1, AUDIT C06/P04/S06).

- HTTPS only, explicit host + path-grammar allowlist from ``config/source_policies.toml``.
- Credentials in URLs, traversal, queries outside the grammar, non-default ports and
  private/loopback/link-local/reserved resolutions are rejected before connecting.
- Redirects are never auto-followed: each hop is revalidated against the policy.
- A thread-safe per-host limiter spaces *every* request (including retries and redirect
  hops) globally; a per-host semaphore bounds concurrency.
- Bounded attempts with exponential backoff + jitter and a capped ``Retry-After``.
- Responses are streamed and aborted past the byte cap.
- ETag/Last-Modified conditional requests; a 304 is accepted only when the archived
  payload it refers to exists and still hashes correctly.
- Failures return a ``FetchResult`` with ``FAIL``/``UNKNOWN`` and no content. There is
  no path from a failure to an empty success, and bundled fixtures are only ever
  returned via :func:`manual_fixture_result`, labelled ``manual_fixture``.

Known limit: the IP check resolves the host before httpx connects, so a DNS answer
that changes between the two lookups (rebinding) is not caught here. The allowlist is
limited to public, well-known hosts, which bounds that exposure.
"""

from __future__ import annotations

import email.utils
import hashlib
import ipaddress
import json
import random
import re
import socket
import threading
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urljoin, urlsplit

import httpx

from supercoach_via.domain.schemas import CheckOutcome, SourceMode
from supercoach_via.storage.snapshots import atomic_write_bytes

Freshness = Literal["fresh", "revalidated", "partial", "unknown"]
Resolver = Callable[[str], list[str]]

RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

_DEFAULT_KEYS: dict[str, type] = {
    "requests_per_second": float,
    "max_concurrent_per_host": int,
    "connect_timeout_s": float,
    "read_timeout_s": float,
    "max_attempts": int,
    "backoff_base_s": float,
    "backoff_max_s": float,
    "retry_after_max_s": float,
    "max_response_bytes": int,
    "max_redirects": int,
}
_BUILTIN_DEFAULTS: dict[str, float | int] = {
    "requests_per_second": 2.0,
    "max_concurrent_per_host": 2,
    "connect_timeout_s": 10.0,
    "read_timeout_s": 30.0,
    "max_attempts": 3,
    "backoff_base_s": 1.0,
    "backoff_max_s": 20.0,
    "retry_after_max_s": 60.0,
    "max_response_bytes": 10 * 1024 * 1024,
    "max_redirects": 3,
}
_HOST_RE = re.compile(r"^[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?)+$")


class UrlRejectedError(ValueError):
    """A URL violates the source policy (scheme, host, credentials, path grammar)."""


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourcePolicy:
    name: str
    host: str
    path_patterns: tuple[re.Pattern[str], ...]
    requests_per_second: float
    max_concurrent_per_host: int
    connect_timeout_s: float
    read_timeout_s: float
    max_attempts: int
    backoff_base_s: float
    backoff_max_s: float
    retry_after_max_s: float
    max_response_bytes: int
    max_redirects: int

    def path_allowed(self, path: str) -> bool:
        return any(p.fullmatch(path) for p in self.path_patterns)


@dataclass(frozen=True)
class PolicySet:
    sources: dict[str, SourcePolicy]
    live_poll_interval_s: float = 90.0

    def for_host(self, host: str) -> list[SourcePolicy]:
        return [p for p in self.sources.values() if p.host == host]


def _coerce(key: str, value: Any, where: str) -> float | int:
    kind = _DEFAULT_KEYS.get(key)
    if kind is None:
        raise ValueError(f"{where}: unknown policy key {key!r}")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{where}: {key} must be numeric")
    out: float | int = float(value) if kind is float else int(value)
    if out <= 0:
        raise ValueError(f"{where}: {key} must be positive")
    return out


def parse_policies(text: str) -> PolicySet:
    """Parse and strictly validate a source-policy TOML document."""
    raw = tomllib.loads(text)
    unknown = set(raw) - {"defaults", "sources", "live"}
    if unknown:
        raise ValueError(f"unknown policy sections: {sorted(unknown)}")
    defaults = dict(_BUILTIN_DEFAULTS)
    for key, value in raw.get("defaults", {}).items():
        defaults[key] = _coerce(key, value, "defaults")
    live = raw.get("live", {})
    if set(live) - {"poll_interval_s"}:
        raise ValueError("unknown [live] keys")
    poll = float(live.get("poll_interval_s", 90.0))
    sources: dict[str, SourcePolicy] = {}
    for name, spec in raw.get("sources", {}).items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", name):
            raise ValueError(f"invalid source name {name!r}")
        spec = dict(spec)
        host = spec.pop("host", None)
        patterns = spec.pop("path_patterns", None)
        if not isinstance(host, str) or not _HOST_RE.fullmatch(host):
            raise ValueError(f"sources.{name}: host must be a bare lowercase hostname")
        if not isinstance(patterns, list) or not patterns:
            raise ValueError(f"sources.{name}: path_patterns required")
        values = dict(defaults)
        for key, value in spec.items():
            values[key] = _coerce(key, value, f"sources.{name}")
        compiled = tuple(re.compile(p) for p in patterns if isinstance(p, str))
        if len(compiled) != len(patterns) or not all(p.pattern.startswith("^/") for p in compiled):
            raise ValueError(f"sources.{name}: path patterns must be anchored strings")
        sources[name] = SourcePolicy(
            name=name,
            host=host,
            path_patterns=compiled,
            requests_per_second=float(values["requests_per_second"]),
            max_concurrent_per_host=int(values["max_concurrent_per_host"]),
            connect_timeout_s=float(values["connect_timeout_s"]),
            read_timeout_s=float(values["read_timeout_s"]),
            max_attempts=int(values["max_attempts"]),
            backoff_base_s=float(values["backoff_base_s"]),
            backoff_max_s=float(values["backoff_max_s"]),
            retry_after_max_s=float(values["retry_after_max_s"]),
            max_response_bytes=int(values["max_response_bytes"]),
            max_redirects=int(values["max_redirects"]),
        )
    if not sources:
        raise ValueError("no sources configured")
    return PolicySet(sources=sources, live_poll_interval_s=poll)


def load_policies(path: Path) -> PolicySet:
    return parse_policies(path.read_text(encoding="utf-8"))


def validate_url(url: str, policies: PolicySet) -> SourcePolicy:
    """Return the policy governing ``url`` or raise :class:`UrlRejectedError`."""
    if not isinstance(url, str) or not url or len(url) > 2048:
        raise UrlRejectedError("empty or oversized URL")
    if any(ch in url for ch in ("\\", "\x00", "\r", "\n", "\t", " ")):
        raise UrlRejectedError("control/whitespace/backslash in URL")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise UrlRejectedError(f"unparseable URL: {exc}") from exc
    if parts.scheme != "https":
        raise UrlRejectedError("only https is allowed")
    if "@" in parts.netloc or parts.username or parts.password:
        raise UrlRejectedError("credentials in URL")
    if port not in (None, 443):
        raise UrlRejectedError("non-default port")
    if parts.query or parts.fragment:
        raise UrlRejectedError("query/fragment not permitted by any path grammar")
    host = (parts.hostname or "").lower()
    candidates = policies.for_host(host)
    if not candidates:
        raise UrlRejectedError(f"host not allowlisted: {host!r}")
    path = parts.path
    decoded = unquote(path)
    if "%" in path or decoded != path:
        raise UrlRejectedError("percent-encoding is not permitted in source paths")
    if any(seg in (".", "..") for seg in path.split("/")):
        raise UrlRejectedError("path traversal")
    for policy in candidates:
        if policy.path_allowed(path):
            return policy
    raise UrlRejectedError(f"path outside source grammar: {path!r}")


def system_resolver(host: str) -> list[str]:  # pragma: no cover - real DNS
    infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]) for info in infos})


def check_public_addresses(addresses: list[str]) -> str | None:
    """Return a reason string when any address is not a public unicast destination."""
    if not addresses:
        return "host resolved to no addresses"
    for text in addresses:
        try:
            ip = ipaddress.ip_address(text.split("%", 1)[0])
        except ValueError:
            return f"unparseable address {text!r}"
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        if (
            not ip.is_global
            or ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return f"non-public destination {ip}"
    return None


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class HostRateLimiter:
    """Thread-safe per-host spacing limiter (a token bucket of depth one).

    Each ``acquire`` reserves the next slot for the host under a lock, then sleeps
    outside the lock until that slot; concurrent callers therefore queue globally.
    """

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._monotonic = monotonic
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next: dict[str, float] = {}

    def acquire(self, host: str, rate_per_second: float) -> float:
        interval = 1.0 / rate_per_second
        with self._lock:
            now = self._monotonic()
            slot = max(now, self._next.get(host, now))
            self._next[host] = slot + interval
        wait = slot - now
        if wait > 0:
            self._sleep(wait)
        return max(wait, 0.0)


# ---------------------------------------------------------------------------
# Raw payload archive
# ---------------------------------------------------------------------------


class RawArchive:
    """Content-addressed raw payload store plus per-URL conditional validators.

    Layout: ``objects/<sha[:2]>/<sha>`` (write-once) and ``validators/<sha(url)>.json``.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def object_path(self, sha256: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("invalid sha256")
        return self.root / "objects" / sha256[:2] / sha256

    def put(self, content: bytes) -> str:
        digest = hashlib.sha256(content).hexdigest()
        path = self.object_path(digest)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            atomic_write_bytes(path, content)
        return digest

    def get(self, sha256: str) -> bytes | None:
        """Return the payload only if it exists and still hashes to ``sha256``."""
        path = self.object_path(sha256)
        if not path.is_file():
            return None
        data = path.read_bytes()
        return data if hashlib.sha256(data).hexdigest() == sha256 else None

    def _validator_path(self, url: str) -> Path:
        return self.root / "validators" / f"{hashlib.sha256(url.encode()).hexdigest()}.json"

    def validators(self, url: str) -> dict[str, str | None] | None:
        path = self._validator_path(url)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("url") != url:
            return None
        return {k: (str(v) if v is not None else None) for k, v in data.items()}

    def set_validators(self, url: str, *, sha256: str, etag: str | None, last_modified: str | None) -> None:
        body = {"url": url, "sha256": sha256, "etag": etag, "last_modified": last_modified}
        atomic_write_bytes(self._validator_path(url), json.dumps(body, sort_keys=True).encode())


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    source: str
    outcome: CheckOutcome
    source_mode: SourceMode
    freshness: Freshness
    fetched_at: datetime
    http_status: int | None = None
    content: bytes | None = None
    sha256: str | None = None
    bytes: int = 0
    attempts: int = 0
    requests: int = 0
    elapsed_s: float = 0.0
    etag: str | None = None
    last_modified: str | None = None
    error: str | None = None
    source_date: date | None = None

    @property
    def ok(self) -> bool:
        """True only for a verified live/revalidated payload (never a fixture fallback)."""
        return (
            self.outcome is CheckOutcome.PASS
            and self.content is not None
            and self.source_mode in (SourceMode.LIVE, SourceMode.CACHED)
        )

    @property
    def source_ref(self) -> str:
        key = f"{self.url}|{self.fetched_at.isoformat()}|{self.sha256 or '-'}|{self.outcome.value}"
        return f"src:{self.source}:{hashlib.sha256(key.encode()).hexdigest()[:24]}"

    def observation(self, adapter: str, adapter_version: str) -> dict[str, Any]:
        """A ``source_observations`` row (see ``domain.schemas.TABLES``)."""
        return {
            "source_ref": self.source_ref,
            "adapter": adapter,
            "adapter_version": adapter_version,
            "url": self.final_url,
            "fetched_at": self.fetched_at,
            "content_sha256": self.sha256,
            "http_status": self.http_status,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "bytes": self.bytes,
            "source_mode": self.source_mode.value,
            "outcome": self.outcome.value,
        }


def manual_fixture_result(
    url: str,
    content: bytes,
    *,
    source: str,
    source_date: date,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> FetchResult:
    """Label a bundled/manual payload honestly: never PASS, never fresh."""
    return FetchResult(
        url=url,
        final_url=url,
        source=source,
        outcome=CheckOutcome.UNKNOWN,
        source_mode=SourceMode.MANUAL_FIXTURE,
        freshness="unknown",
        fetched_at=now(),
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
        bytes=len(content),
        error="manual_fixture fallback; not fetched from source",
        source_date=source_date,
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class _Oversized(Exception):
    pass


@dataclass
class _Attempt:
    status: int | None = None
    content: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)
    final_url: str = ""
    error: str | None = None
    outcome: CheckOutcome | None = None  # terminal decision when set
    retry_after: float | None = None


class HttpClient:
    """Sync pooled client enforcing :class:`PolicySet`. Thread-safe for concurrent fetches."""

    def __init__(
        self,
        policies: PolicySet,
        *,
        user_agent: str,
        archive: RawArchive | None = None,
        transport: httpx.BaseTransport | None = None,
        resolver: Resolver = system_resolver,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if not user_agent.strip():
            raise ValueError("a descriptive user agent is required")
        self.policies = policies
        self.archive = archive
        self._resolver = resolver
        self._monotonic = monotonic
        self._sleep = sleep
        self._now = now
        self._jitter = jitter
        self._limiter = HostRateLimiter(monotonic=monotonic, sleep=sleep)
        self._sems: dict[str, threading.BoundedSemaphore] = {}
        self._lock = threading.Lock()
        self.request_counts: dict[str, int] = {}
        self.bytes_received = 0
        self.observations: list[dict[str, Any]] = []
        self._client = httpx.Client(
            transport=transport,
            headers={"User-Agent": user_agent},
            follow_redirects=False,
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            trust_env=False,
        )

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- internals ----------------------------------------------------------

    def _sem(self, policy: SourcePolicy) -> threading.BoundedSemaphore:
        with self._lock:
            sem = self._sems.get(policy.host)
            if sem is None:
                sem = self._sems[policy.host] = threading.BoundedSemaphore(policy.max_concurrent_per_host)
            return sem

    def _count(self, host: str, nbytes: int = 0) -> None:
        with self._lock:
            self.request_counts[host] = self.request_counts.get(host, 0) + 1
            self.bytes_received += nbytes

    def _backoff(self, policy: SourcePolicy, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(max(retry_after, 0.0), policy.retry_after_max_s)
        base = policy.backoff_base_s * float(2 ** (attempt - 1))
        return float(min(policy.backoff_max_s, base * (0.5 + self._jitter())))

    @staticmethod
    def _parse_retry_after(value: str | None, now: datetime) -> float | None:
        if not value:
            return None
        value = value.strip()
        if value.isdigit():
            return float(value)
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return max((when - now).total_seconds(), 0.0)

    def _one_request(self, url: str, policy: SourcePolicy, conditional: dict[str, str | None] | None) -> _Attempt:
        """Perform one logical request, following revalidated redirects."""
        current, current_policy = url, policy
        for _hop in range(policy.max_redirects + 1):
            host = current_policy.host
            try:
                addresses = self._resolver(host)
            except OSError as exc:
                return _Attempt(error=f"dns: {type(exc).__name__}: {exc}", final_url=current)
            reason = check_public_addresses(addresses)
            if reason is not None:
                return _Attempt(error=f"blocked: {reason}", outcome=CheckOutcome.FAIL, final_url=current)
            headers: dict[str, str] = {}
            if conditional and current == url:
                if conditional.get("etag"):
                    headers["If-None-Match"] = str(conditional["etag"])
                if conditional.get("last_modified"):
                    headers["If-Modified-Since"] = str(conditional["last_modified"])
            timeout = httpx.Timeout(
                current_policy.read_timeout_s,
                connect=current_policy.connect_timeout_s,
            )
            self._limiter.acquire(host, current_policy.requests_per_second)
            sem = self._sem(current_policy)
            with sem:
                self._count(host)
                request = self._client.build_request("GET", current, headers=headers, timeout=timeout)
                response = self._client.send(request, stream=True)
                try:
                    status = response.status_code
                    resp_headers = {k.lower(): v for k, v in response.headers.items()}
                    if status in (301, 302, 303, 307, 308):
                        location = resp_headers.get("location")
                        if not location:
                            return _Attempt(
                                status=status,
                                error="redirect without Location",
                                outcome=CheckOutcome.FAIL,
                                final_url=current,
                            )
                        target = urljoin(current, location)
                        try:
                            current_policy = validate_url(target, self.policies)
                        except UrlRejectedError as exc:
                            return _Attempt(
                                status=status,
                                error=f"redirect rejected: {exc}",
                                outcome=CheckOutcome.FAIL,
                                final_url=current,
                            )
                        current = target
                        continue
                    if status != 200:
                        return _Attempt(status=status, headers=resp_headers, final_url=current)
                    declared = resp_headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > current_policy.max_response_bytes:
                        return _Attempt(
                            status=status,
                            error="oversized: declared content-length",
                            outcome=CheckOutcome.FAIL,
                            final_url=current,
                        )
                    buf = bytearray()
                    try:
                        for chunk in response.iter_bytes():
                            buf.extend(chunk)
                            if len(buf) > current_policy.max_response_bytes:
                                raise _Oversized
                    except _Oversized:
                        return _Attempt(
                            status=status,
                            error="oversized: body exceeded cap",
                            outcome=CheckOutcome.FAIL,
                            final_url=current,
                        )
                    with self._lock:
                        self.bytes_received += len(buf)
                    return _Attempt(status=status, content=bytes(buf), headers=resp_headers, final_url=current)
                finally:
                    response.close()
        return _Attempt(error="redirect limit exceeded", outcome=CheckOutcome.FAIL, final_url=current)

    # -- public -------------------------------------------------------------

    def fetch(self, url: str, *, conditional: bool = True) -> FetchResult:
        started = self._monotonic()
        fetched_at = self._now()

        def result(**kw: Any) -> FetchResult:
            kw.setdefault("final_url", url)
            kw.setdefault("source_mode", SourceMode.LIVE)
            kw.setdefault("freshness", "unknown")
            res = FetchResult(url=url, fetched_at=fetched_at, elapsed_s=self._monotonic() - started, **kw)
            with self._lock:
                self.observations.append(res.observation(f"http.{res.source}", "1"))
            return res

        try:
            policy = validate_url(url, self.policies)
        except UrlRejectedError as exc:
            return result(source="rejected", outcome=CheckOutcome.FAIL, error=f"url rejected: {exc}")

        validators: dict[str, str | None] | None = None
        if conditional and self.archive is not None:
            v = self.archive.validators(url)
            if v and v.get("sha256") and self.archive.get(str(v["sha256"])) is not None:
                validators = v

        last: _Attempt = _Attempt()
        attempts = 0
        for attempt in range(1, policy.max_attempts + 1):
            attempts = attempt
            try:
                last = self._one_request(url, policy, validators)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = _Attempt(error=f"{type(exc).__name__}: {exc}")
            if last.outcome is CheckOutcome.FAIL:
                break
            if last.status == 200 and last.content is not None:
                content = last.content
                sha = self.archive.put(content) if self.archive else hashlib.sha256(content).hexdigest()
                etag = last.headers.get("etag")
                lm = last.headers.get("last-modified")
                if self.archive is not None and (etag or lm):
                    self.archive.set_validators(url, sha256=sha, etag=etag, last_modified=lm)
                return result(
                    source=policy.name,
                    outcome=CheckOutcome.PASS,
                    freshness="fresh",
                    final_url=last.final_url,
                    http_status=200,
                    content=content,
                    sha256=sha,
                    bytes=len(content),
                    attempts=attempts,
                    requests=sum(self.request_counts.values()),
                    etag=etag,
                    last_modified=lm,
                )
            if last.status == 304:
                if validators and self.archive is not None:
                    sha = str(validators["sha256"])
                    payload = self.archive.get(sha)
                    if payload is not None:
                        return result(
                            source=policy.name,
                            outcome=CheckOutcome.PASS,
                            freshness="revalidated",
                            source_mode=SourceMode.CACHED,
                            final_url=last.final_url,
                            http_status=304,
                            content=payload,
                            sha256=sha,
                            bytes=len(payload),
                            attempts=attempts,
                            etag=validators.get("etag"),
                            last_modified=validators.get("last_modified"),
                        )
                last = _Attempt(
                    status=304,
                    error="304 without a valid archived payload",
                    outcome=CheckOutcome.FAIL,
                )
                break
            if last.status is not None and last.status not in RETRYABLE_STATUS:
                last.outcome = CheckOutcome.FAIL
                last.error = last.error or f"http {last.status}"
                break
            if attempt < policy.max_attempts:
                retry_after = self._parse_retry_after(last.headers.get("retry-after"), self._now())
                self._sleep(self._backoff(policy, attempt, retry_after))

        outcome = last.outcome or CheckOutcome.UNKNOWN
        error = last.error or (f"http {last.status}" if last.status else "unavailable")
        return result(
            source=policy.name,
            outcome=outcome,
            final_url=last.final_url or url,
            http_status=last.status,
            attempts=attempts,
            error=error,
        )


def fit_table_row(table: str, row: dict[str, Any]) -> dict[str, Any]:
    """Project ``row`` onto the canonical table's columns (absent columns -> None).

    Raises ``KeyError`` when the adapter emits a column the contract does not define,
    so schema drift between adapters and ``domain.schemas.TABLES`` fails loudly.
    """
    from supercoach_via.domain.schemas import TABLES

    columns = TABLES[table].column_names
    extra = set(row) - set(columns)
    if extra:
        raise KeyError(f"{table}: adapter emitted unknown columns {sorted(extra)}")
    return {c: row.get(c) for c in columns}
