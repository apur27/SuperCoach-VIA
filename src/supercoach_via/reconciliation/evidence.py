"""Brownlow Medal award evidence (DESIGN section 15, A9).

Two jobs, both small:

* ``capture_evidence`` acquires the AFL Tables Brownlow index (one exact path, plus ``robots.txt``) into its
  OWN archive under ``<run>/evidence``: one coordinator, one request in flight, at least two seconds between
  request starts, a descriptive user agent, robots honoured, at most a handful of requests. It shares no
  policy, identity or archive with the frozen audit corpus, so acquiring it can never orphan that corpus.
* ``parse_brownlow`` reads the stored bytes into the facts the cell rules need: the seasons in which no medal
  was awarded and the votes awarded per game in each season. Nothing is typed in or recalled; a page this
  reader cannot understand raises ``EvidenceError`` (never a guess).

The body's SHA-256 is pinned in ``config/reconciliation_rules.toml`` and so enters ``plan_id`` and every
per-season unit digest that depends on it.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import urllib.robotparser
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from supercoach_via.ingest.http import FetchResult, HttpClient, RawArchive
from supercoach_via.integrity.report import canonical_bytes
from supercoach_via.integrity.sourcepages import read_tables
from supercoach_via.reconciliation.capture import BusyError, Clock, SystemClock
from supercoach_via.reconciliation.urls import SITE
from supercoach_via.settings import default_config_dir
from supercoach_via.storage.snapshots import atomic_write_bytes

EVIDENCE_POLICY_FILE = "reconciliation_evidence_policy.toml"
EVIDENCE_POLICY_NAME = "afltables_brownlow_evidence"
BROWNLOW_URL = f"{SITE}/afl/brownlow/brownlow_idx.html"
ROBOTS_URL = f"{SITE}/robots.txt"
EVIDENCE_DIR = "evidence"
USER_AGENT = "supercoach-via-reconciliation/1.0 (read-only audit evidence; 0.5 requests/second)"
MIN_SPACING_S = 2.0
MAX_ATTEMPTS = 3
BACKOFF_S = (30.0, 90.0)
PAUSE_THRESHOLD_S = 900.0
#: requests made by one capture never exceed this (robots + the page, each with bounded retries)
MAX_REQUESTS = 6


class EvidenceError(ValueError):
    """The evidence is missing, altered, or not understood by the reader (exit code 2 when pinned)."""


# ---------------------------------------------------------------------------
# Reading the stored page
# ---------------------------------------------------------------------------

_WORDS = {
    w: i
    for i, w in enumerate(
        ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
    )
}
_NO_MEDAL = re.compile(r"No Medal awarded\s+(\d{4})(?:\s*-\s*(\d{4}))?", re.I)
_ERA = re.compile(r"(\d{4})\s*-\s*(\d{4}|present)\s*:\s*([A-Za-z]+|\d+)\s+votes?\s+per\s+game", re.I)
_EXCEPT = re.compile(r"\(\s*except\s+(\d{4})\s*-\s*(\d{2,4})\s*:\s*([A-Za-z]+|\d+)\s+votes?\s+per\s+game", re.I)


@dataclass(frozen=True)
class VoteEra:
    first: int
    #: ``None`` = "present"
    last: int | None
    total: int

    def covers(self, season: int) -> bool:
        return self.first <= season and (self.last is None or season <= self.last)


@dataclass(frozen=True)
class BrownlowFacts:
    no_award_seasons: frozenset[int]
    #: general eras, then exceptions (an exception overrides the era that contains it)
    eras: tuple[VoteEra, ...]
    exceptions: tuple[VoteEra, ...]
    #: the years the winners table lists (a cross-check of the footer's no-award statement)
    winner_years: frozenset[int]

    def award_total(self, season: int) -> int | None:
        """Votes awarded per game in ``season``; ``None`` when no medal was awarded or the page states no system."""
        if season in self.no_award_seasons:
            return None
        for e in self.exceptions:
            if e.covers(season):
                return e.total
        for e in self.eras:
            if e.covers(season):
                return e.total
        return None


def _number(token: str) -> int:
    if token.isdigit():
        return int(token)
    if token.lower() in _WORDS:
        return _WORDS[token.lower()]
    raise EvidenceError(f"votes per game {token!r} is neither a number nor a number word")


def _year(token: str, base: int) -> int:
    """``77`` after 1976 is 1977; a four digit year is itself."""
    return int(token) if len(token) == 4 else (base // 100) * 100 + int(token)


def parse_brownlow(body: bytes) -> BrownlowFacts:
    """Award seasons and votes per game, parsed from the stored Brownlow index. Fails closed."""
    tables = read_tables(body)
    winners = next(
        (
            t
            for t in tables
            if t.rows and t.rows[0] and t.rows[0][0].text == "Year" and "Votes" in [c.text for c in t.rows[0]]
        ),
        None,
    )
    if winners is None:
        raise EvidenceError("the winners table (Year, Player, Team, Votes ...) was not found")
    years: set[int] = set()
    footer = ""
    for row in winners.rows[1:]:
        if len(row) == 1 or (row and "Voting systems" in row[0].text):
            if "Voting systems" in row[0].text:
                footer = " ".join(row[0].text.split())
            continue
        first = row[0].text.strip()
        if re.fullmatch(r"\d{4}", first):
            years.add(int(first))
    if not footer:
        raise EvidenceError("the footer row stating the voting systems was not found")
    if not years:
        raise EvidenceError("the winners table lists no year")
    no_award: set[int] = set()
    for m in _NO_MEDAL.finditer(footer):
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        if hi < lo:
            raise EvidenceError(f"no-award range {lo}-{hi} is backwards")
        no_award.update(range(lo, hi + 1))
    listed_gaps = {y for y in range(min(years), max(years) + 1) if y not in years}
    if no_award != listed_gaps:
        raise EvidenceError(
            f"the footer's no-award seasons {sorted(no_award)} disagree with the years the winners table omits "
            f"{sorted(listed_gaps)}"
        )
    exceptions = tuple(
        VoteEra(int(m.group(1)), _year(m.group(2), int(m.group(1))), _number(m.group(3)))
        for m in _EXCEPT.finditer(footer)
    )
    eras = tuple(
        VoteEra(int(m.group(1)), None if m.group(2).lower() == "present" else int(m.group(2)), _number(m.group(3)))
        for m in _ERA.finditer(footer)
    )
    if not eras:
        raise EvidenceError("no 'votes per game' voting system was found in the footer")
    if not any(e.last is None for e in eras):
        raise EvidenceError("no voting system runs to the present")
    return BrownlowFacts(frozenset(no_award), eras, exceptions, frozenset(years))


def load_evidence(evidence_dir: Path, sha256: str) -> bytes:
    """The stored body, only if it exists and still hashes to the pinned digest."""
    data = RawArchive(evidence_dir).get(sha256)
    if data is None:
        raise EvidenceError(f"evidence object {sha256[:12]} is missing or altered under {evidence_dir}")
    return data


def evidence_policies(config_dir: Path | None = None) -> Any:
    from supercoach_via.ingest.http import load_policies

    return load_policies((config_dir or default_config_dir()) / EVIDENCE_POLICY_FILE)


# ---------------------------------------------------------------------------
# Acquiring it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceResult:
    exit_code: int  # 0 complete, 8 incomplete (resumable by rerunning)
    state: str  # complete | paused | blocked | failed
    reason: str | None
    sha256: str | None
    requests: int


class EvidenceCapture:
    def __init__(
        self,
        run_dir: Path,
        client: HttpClient,
        *,
        clock: Clock | None = None,
        log: Callable[[str], None] = lambda _m: None,
    ) -> None:
        policy = client.policies.sources.get(EVIDENCE_POLICY_NAME)
        if policy is None or policy.max_attempts != 1 or policy.max_concurrent_per_host != 1:
            raise EvidenceError("evidence capture needs its own policy with max_attempts=1 and concurrency 1")
        if policy.requests_per_second > 1.0 / MIN_SPACING_S:
            raise EvidenceError("the evidence policy allows more than one request per two seconds")
        self.dir = run_dir / EVIDENCE_DIR
        self.client = client
        self.clock: Clock = clock or SystemClock()
        self.log = log
        self.archive = RawArchive(self.dir)
        self._next_slot = 0.0
        self._requests = 0
        self._journal_n: int | None = None
        self._lock: IO[bytes] | None = None

    # -- plumbing ------------------------------------------------------------------------------

    def _iso(self, ts: float) -> str:
        return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _journal(self, entry: dict[str, Any]) -> None:
        path = self.dir / "observations.jsonl"
        if self._journal_n is None:
            self._journal_n = sum(1 for _ in path.open("rb")) if path.exists() else 0
        self._journal_n += 1
        with path.open("ab") as fh:
            fh.write(canonical_bytes({"n": self._journal_n, **entry}))
            fh.flush()
            os.fsync(fh.fileno())

    def _fetch(self, url: str, attempt: int) -> tuple[FetchResult, float]:
        wait = self._next_slot - self.clock.time()
        if wait > 0:
            self.clock.sleep(wait)
        started = self.clock.time()
        before = sum(self.client.request_counts.values())
        res = self.client.fetch(url, conditional=False)
        finished = self.clock.time()
        reqs = max(sum(self.client.request_counts.values()) - before, 1)
        self._requests += reqs
        nxt = (started if reqs == 1 else finished) + MIN_SPACING_S
        if res.retry_after_s is not None:
            nxt = max(nxt, finished + res.retry_after_s)
        self._next_slot = max(self._next_slot, nxt)
        self._journal(
            {
                "url": url,
                "final_url": res.final_url,
                "fetched_at": self._iso(started),
                "http_status": res.http_status,
                "outcome": res.outcome.value,
                "sha256": res.sha256,
                "bytes": res.bytes,
                "last_modified": res.last_modified,
                "etag": res.etag,
                "attempt": attempt,
                "requests": reqs,
                "retry_after_s": res.retry_after_s,
                "error": res.error,
            }
        )
        return res, finished

    def _receipt(self, state: str, reason: str | None, extra: dict[str, Any] | None = None) -> EvidenceResult:
        doc = {
            "kind": "afltables-reconciliation-evidence-receipt",
            "state": state,
            "reason": reason,
            "url": BROWNLOW_URL,
            "policy": EVIDENCE_POLICY_NAME,
            "requests_this_process": self._requests,
            "updated_utc": self._iso(self.clock.time()),
            **(extra or {}),
        }
        atomic_write_bytes(self.dir / "receipt.json", canonical_bytes(doc))
        return EvidenceResult(0 if state == "complete" else 8, state, reason, doc.get("sha256"), self._requests)

    # -- the capture --------------------------------------------------------------------------------

    def run(self) -> EvidenceResult:
        self.dir.mkdir(parents=True, exist_ok=True)
        fh = (self.dir / ".lock").open("wb")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            fh.close()
            raise BusyError(f"another evidence capture holds {self.dir / '.lock'}") from exc
        self._lock = fh
        try:
            return self._run()
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()

    def _run(self) -> EvidenceResult:
        robots, why = self._robots()
        if why is not None:
            return self._receipt("paused", why)
        if robots is not None and not robots.can_fetch(USER_AGENT, BROWNLOW_URL):
            return self._receipt("blocked", "robots.txt disallows the Brownlow index; no request was made for it")
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if self._requests >= MAX_REQUESTS:
                break
            res, _finished = self._fetch(BROWNLOW_URL, attempt)
            if res.ok and res.content is not None and res.http_status == 200:
                return self._accept(res)
            if res.http_status in (401, 403):
                return self._receipt("blocked", f"HTTP {res.http_status}; stopping rather than retrying")
            if res.http_status == 429 or (res.http_status == 503 and res.retry_after_s is not None):
                if res.retry_after_s is not None and res.retry_after_s > PAUSE_THRESHOLD_S:
                    return self._receipt("paused", f"rate limited; Retry-After {int(res.retry_after_s)} s")
                self.clock.sleep(max(res.retry_after_s or 0.0, BACKOFF_S[min(attempt - 1, 1)]))
                continue
            if res.http_status in (404, 410):
                return self._receipt("failed", f"HTTP {res.http_status}: the page is not there")
            if attempt < MAX_ATTEMPTS:
                self.clock.sleep(BACKOFF_S[min(attempt - 1, len(BACKOFF_S) - 1)])
        return self._receipt("failed", f"no usable response after {MAX_ATTEMPTS} attempts")

    def _robots(self) -> tuple[urllib.robotparser.RobotFileParser | None, str | None]:
        """(parser, None) when robots allows normal operation; (None, None) for an absent file;
        (None, reason) to pause."""
        res, _ = self._fetch(ROBOTS_URL, 1)
        if res.http_status in (404, 410):
            return None, None
        if res.ok and res.content is not None and res.http_status == 200:
            rp = urllib.robotparser.RobotFileParser()
            rp.parse(res.content.decode("utf-8", "replace").splitlines())
            return rp, None
        why = f"robots.txt could not be retrieved (HTTP {res.http_status}, {res.error}); nothing else is requested"
        return None, why

    def _accept(self, res: FetchResult) -> EvidenceResult:
        assert res.content is not None
        title_ok = b"Brownlow" in res.content[:4096]
        sha = self.archive.put(res.content)
        if sha != res.sha256 or not title_ok:
            return self._receipt("failed", "the response is not the Brownlow index (identity check failed)")
        try:
            facts = parse_brownlow(res.content)
        except EvidenceError as exc:
            return self._receipt("failed", f"stored but not understood: {exc}", {"sha256": sha, "bytes": res.bytes})
        return self._receipt(
            "complete",
            None,
            {
                "sha256": sha,
                "bytes": res.bytes,
                "last_modified": res.last_modified,
                "final_url": res.final_url,
                "no_award_seasons": sorted(facts.no_award_seasons),
            },
        )
