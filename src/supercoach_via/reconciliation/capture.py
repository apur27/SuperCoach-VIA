"""Resumable, considerate AFL Tables acquisition (DESIGN section 6).

One coordinator, one in-flight request, at least two seconds between request starts (retries,
robots, redirects and revalidation included), persisted in a private SQLite checkpoint so
spacing and a server's ``Retry-After`` survive a restart. The coordinator owns every retry
(at most three attempts per resource); the HTTP client is configured for a single attempt.
No conditional requests are sent and no validators are written. Failures are recorded as
evidence gaps with their reason: nothing here turns a failed fetch into an empty success.

Raw bodies are content-addressed under ``capture/objects/``; every attempt is appended to
``capture/observations.jsonl``; ``capture/manifest.json`` is the frozen evidence manifest.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import shutil
import sqlite3
import time
import urllib.robotparser
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import IO, Any, Protocol

from supercoach_via.domain.schemas import CheckOutcome, SourceMode
from supercoach_via.ingest.http import FetchResult, HttpClient, RawArchive
from supercoach_via.integrity.report import canonical_bytes
from supercoach_via.reconciliation import discover as D
from supercoach_via.reconciliation import schema as S
from supercoach_via.reconciliation.urls import RECONCILIATION_POLICY_NAME, SITE, classify_link
from supercoach_via.storage.snapshots import atomic_write_bytes

MIN_SPACING_S = 2.0
MAX_ATTEMPTS = 3
TRANSIENT_BACKOFF_S = (30.0, 90.0, 270.0)
PAUSE_THRESHOLD_S = 900.0
MIN_FREE_DISK_BYTES = 3 * 1024**3
CONSECUTIVE_EXHAUSTED_PAUSE = 3
USER_AGENT = "supercoach-via-reconciliation/1.0 (read-only audit of a local AFL dataset; 0.5 requests/second)"
PRIO = {"robots": 0, "stats_index": 1, "notes": 2, "letter": 3, "season": 4, "profile": 5, "match": 5}
REVAL_PRIO = 6
TERMINAL_OK = ("done", "absent")
GAP_STATES = ("failed", "missing", "blocked", "rejected")


class CaptureError(RuntimeError):
    """Invalid invocation or incompatible resume (exit code 2)."""


class BusyError(RuntimeError):
    """Another process holds the capture lock (exit code 5)."""


class Clock(Protocol):
    def time(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def time(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class CaptureResult:
    exit_code: int  # 0 complete, 8 incomplete but resumable
    state: str  # complete | incomplete | interrupted | paused | blocked
    reason: str | None
    counts: dict[str, int]
    requests: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS task(
  task_id TEXT PRIMARY KEY, kind TEXT NOT NULL, url TEXT NOT NULL, reval INTEGER NOT NULL DEFAULT 0,
  state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, http_status INTEGER, final_url TEXT,
  sha256 TEXT, bytes INTEGER, reason TEXT, observation INTEGER, prio INTEGER NOT NULL,
  seq INTEGER NOT NULL, next_eligible REAL NOT NULL DEFAULT 0, parser_ok INTEGER NOT NULL DEFAULT 0,
  UNIQUE(url, reval));
CREATE INDEX IF NOT EXISTS task_pending ON task(state, prio, seq);
CREATE INDEX IF NOT EXISTS task_seq ON task(seq);
CREATE TABLE IF NOT EXISTS ref(task_id TEXT NOT NULL, source TEXT NOT NULL, PRIMARY KEY(task_id, source));
CREATE TABLE IF NOT EXISTS census(url TEXT PRIMARY KEY, letter TEXT NOT NULL, display_name TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS lineup(url TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS matchinfo(
  url TEXT PRIMARY KEY, year INTEGER NOT NULL, match_date TEXT, in_scope INTEGER NOT NULL, listed INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS reason(text TEXT PRIMARY KEY);
"""


def task_id(kind: str, url: str, reval: bool = False) -> str:
    return hashlib.sha256(f"{'reval:' if reval else ''}{kind}|{url}".encode()).hexdigest()[:20]


class Capture:
    def __init__(
        self,
        plan: S.Plan,
        run_dir: Path,
        client: HttpClient,
        *,
        clock: Clock | None = None,
        log: Callable[[str], None] = lambda _m: None,
        heartbeat_s: float = 60.0,
        max_requests: int | None = None,
        clear_block: bool = False,
        revalidate: bool = True,
        durable: bool = True,
        seed_from: list[Path] | None = None,
    ) -> None:
        policy = client.policies.sources.get(RECONCILIATION_POLICY_NAME)
        if policy is None or policy.max_attempts != 1 or policy.max_concurrent_per_host != 1:
            raise CaptureError("capture needs the reconciliation policy with max_attempts=1 and concurrency 1")
        if policy.requests_per_second > 1.0 / MIN_SPACING_S:
            raise CaptureError("capture policy allows more than one request per two seconds")
        self.plan = plan
        self.run_dir = run_dir
        self.dir = run_dir / "capture"
        self.client = client
        self.clock: Clock = clock or SystemClock()
        self.log = log
        self.heartbeat_s = heartbeat_s
        self.max_requests = max_requests
        self.clear_block = clear_block
        self.revalidate = revalidate
        self.durable = durable
        self.seed_from = [Path(p) for p in (seed_from or [])]
        self._seeds: dict[str, tuple[RawArchive, str, int, str]] | None = None
        self._journal_n: int | None = None
        self.archive = RawArchive(self.dir)
        self.through = date.fromisoformat(plan.scope.through_date)
        self.stop_requested = False
        self._robots: urllib.robotparser.RobotFileParser | None = None
        self._lock_fh: IO[bytes] | None = None
        self._db: sqlite3.Connection | None = None
        self._requests = 0
        self._reused_count = 0
        self._last_beat = 0.0

    # -- lifecycle -----------------------------------------------------------------

    def request_stop(self) -> None:
        self.stop_requested = True

    @property
    def db(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("capture not open")
        return self._db

    def _open(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        fh = (self.dir / ".lock").open("wb")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            fh.close()
            raise BusyError(f"another capture holds {self.dir / '.lock'}") from exc
        self._lock_fh = fh
        try:
            db = sqlite3.connect(self.dir / "checkpoint.sqlite", isolation_level=None)
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=" + ("FULL" if self.durable else "OFF"))
            db.executescript(_SCHEMA)
        except sqlite3.DatabaseError as exc:
            self._release()
            raise CaptureError(f"checkpoint database is unreadable or corrupt: {exc}") from exc
        self._db = db
        try:
            self._check_code()
            self._check_identity()
        except CaptureError:
            self._close()
            raise

    def _close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
        self._release()

    def _release(self) -> None:
        if self._lock_fh is not None:
            with contextlib.suppress(OSError):
                fcntl.flock(self._lock_fh.fileno(), fcntl.LOCK_UN)
            self._lock_fh.close()
            self._lock_fh = None

    @contextlib.contextmanager
    def _txn(self) -> Iterator[None]:
        """One atomic checkpoint transaction (the connection is in autocommit mode)."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        self.db.execute("COMMIT")

    def _meta(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else default

    def _set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value)
        )

    def _check_code(self) -> None:
        """The files that decide which pages are fetched must be the ones the plan pinned."""
        from supercoach_via.reconciliation.inventory import code_identity

        have = code_identity().capture_files
        if have != self.plan.code.capture_files:
            changed = sorted(k for k in have if have[k] != self.plan.code.capture_files.get(k))
            raise CaptureError(
                f"capture code changed since the plan was written ({', '.join(changed)}); "
                "write a new plan and resume it with --seed-from"
            )

    def _check_identity(self) -> None:
        have = self._meta("capture_identity")
        if have is None:
            self._set_meta("capture_identity", self.plan.capture_identity)
            self._set_meta("plan_id", self.plan.plan_id)
            self._set_meta("started_utc", self._iso(self.clock.time()))
            self._seed()
        elif have != self.plan.capture_identity:
            raise CaptureError(
                "incompatible resume: the checkpoint belongs to capture identity "
                f"{have[:16]} but the plan is {self.plan.capture_identity[:16]}; use a new run directory"
            )
        else:
            self._set_meta("plan_id", self.plan.plan_id)

    @staticmethod
    def _iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # -- queue -----------------------------------------------------------------------

    def _seed(self) -> None:
        with self._txn():
            for url in self.plan.source.seed_urls:
                kind = (
                    "robots"
                    if url.endswith("/robots.txt")
                    else "notes"
                    if url.endswith("notes.html")
                    else "stats_index"
                )
                self._add(kind, url, "seed")

    def _add(self, kind: str, url: str, source: str, *, reval: bool = False) -> bool:
        tid = task_id(kind, url, reval)
        seq = int(self.db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM task").fetchone()[0])
        prio = REVAL_PRIO if reval else PRIO[kind]
        problem = classify_link(url, self.client.policies)
        state, reason = ("pending", None) if problem is None else ("rejected", f"url rejected by policy: {problem}")
        cur = self.db.execute(
            "INSERT OR IGNORE INTO task(task_id,kind,url,reval,state,reason,prio,seq) VALUES(?,?,?,?,?,?,?,?)",
            (tid, kind, url, int(reval), state, reason, prio, seq),
        )
        self.db.execute("INSERT OR IGNORE INTO ref(task_id,source) VALUES(?,?)", (tid, source))
        return cur.rowcount > 0

    def _reason(self, text: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO reason(text) VALUES(?)", (text,))

    def _next(self, now: float) -> tuple[Any, ...] | None:
        # discovery must finish before bulk fetching: a rate-limited season page holds the queue,
        # otherwise profile-linked games would be misread as "not on any season page"
        row = self.db.execute("SELECT MIN(prio) FROM task WHERE state='pending'").fetchone()
        if row[0] is None:
            return None
        prio = int(row[0])
        if not self.db.execute(
            "SELECT 1 FROM task WHERE state='pending' AND prio=? AND next_eligible<=? LIMIT 1", (prio, now)
        ).fetchone():
            return None
        last = self._meta("last_pm", "match")
        base = "SELECT task_id,kind,url,reval,attempts FROM task WHERE state='pending' AND next_eligible<=? AND prio=? "
        found: tuple[Any, ...] | None
        if prio == 5:  # alternate profiles and matches so a partial capture covers both
            found = self.db.execute(
                base + "ORDER BY CASE kind WHEN ? THEN 1 ELSE 0 END, seq LIMIT 1", [now, prio, last]
            ).fetchone()
        else:
            found = self.db.execute(base + "ORDER BY seq LIMIT 1", [now, prio]).fetchone()
        return found

    # -- pacing -------------------------------------------------------------------------

    def _wait_for_slot(self) -> None:
        nxt = float(self._meta("host_next_eligible", "0") or 0)
        wait = nxt - self.clock.time()
        if wait > 0:
            self.clock.sleep(wait)

    def _set_slot(self, started: float, finished: float, requests: int, retry_after: float | None) -> None:
        nxt = started + MIN_SPACING_S
        if requests > 1:  # a redirect hop's start time is not visible here: be conservative
            nxt = finished + MIN_SPACING_S
        if retry_after is not None:
            nxt = max(nxt, finished + retry_after)
        prev = float(self._meta("host_next_eligible", "0") or 0)
        self._set_meta("host_next_eligible", repr(max(prev, nxt)))

    # -- journal ---------------------------------------------------------------------------

    def _journal(self, entry: dict[str, Any]) -> int:
        path = self.dir / "observations.jsonl"
        if self._journal_n is None:
            self._journal_n = 0
            if path.exists():
                with path.open("rb") as fh:
                    self._journal_n = sum(1 for _ in fh)
        self._journal_n += 1
        with path.open("ab") as fh:
            fh.write(canonical_bytes({"n": self._journal_n, **entry}))
            fh.flush()
            if self.durable:
                os.fsync(fh.fileno())
        return self._journal_n

    # -- main loop -------------------------------------------------------------------------

    def run(self) -> CaptureResult:
        self._open()
        try:
            return self._run()
        finally:
            self._close()

    def _run(self) -> CaptureResult:
        if self._meta("blocked") and not self.clear_block:
            return self._finish(
                "blocked", f"previously blocked: {self._meta('blocked')}; rerun with --clear-block after review"
            )
        if self.clear_block:
            self.db.execute("DELETE FROM meta WHERE key='blocked'")
            self.db.execute("UPDATE task SET state='pending', next_eligible=0 WHERE state='blocked'")
        fetches = 0
        consecutive_exhausted = 0
        while True:
            if self.stop_requested:
                return self._finish("interrupted", "stop requested")
            if self.max_requests is not None and fetches >= self.max_requests:
                return self._finish("interrupted", f"stopped after {fetches} fetches (max_requests)")
            now = self.clock.time()
            row = self._next(now)
            if row is None:
                pending = self.db.execute(
                    "SELECT MIN(next_eligible) FROM task WHERE state='pending' AND prio="
                    "(SELECT MIN(prio) FROM task WHERE state='pending')"
                ).fetchone()[0]
                if pending is not None:
                    wait = float(pending) - now
                    if wait > PAUSE_THRESHOLD_S:
                        return self._finish("paused", f"next request not allowed for {int(wait)} s (server deadline)")
                    self.clock.sleep(max(wait, 0.0))
                    continue
                if self._start_revalidation():
                    continue
                return self._finish("complete", None)
            tid, kind, url, reval, attempts = row
            if kind != "robots" and not self._robots_settled():
                return self._finish("paused", "robots.txt could not be retrieved; no other request is made without it")
            if self._robots_blocks(kind, url):
                self.db.execute(
                    "UPDATE task SET state='blocked', reason='robots.txt disallows this path' WHERE task_id=?", (tid,)
                )
                self._set_meta("blocked", "robots.txt disallows an in-scope path")
                return self._finish("blocked", "robots.txt disallows an in-scope path")
            before_reused = self._reused_count
            outcome = self._attempt(tid, kind, url, bool(reval), int(attempts))
            if self._reused_count == before_reused:
                fetches += 1
                self._requests += 1
            if outcome == "blocked":
                return self._finish("blocked", self._meta("blocked"))
            if outcome == "exhausted":
                consecutive_exhausted += 1
                if consecutive_exhausted >= CONSECUTIVE_EXHAUSTED_PAUSE:
                    return self._finish("paused", f"{consecutive_exhausted} consecutive exhausted requests")
            elif outcome in ("ok", "gap"):
                consecutive_exhausted = 0
            if outcome == "pause":
                return self._finish("paused", self._meta("pause_reason"))
            if not self._heartbeat(url):
                return self._finish("paused", "less than 3 GiB of free disk space")

    # -- one attempt -----------------------------------------------------------------------

    def _robots_settled(self) -> bool:
        row = self.db.execute("SELECT state FROM task WHERE kind='robots' AND reval=0").fetchone()
        return row is not None and row[0] in TERMINAL_OK

    def _robots_blocks(self, kind: str, url: str) -> bool:
        if kind == "robots" or self._robots is None:
            return False
        return not self._robots.can_fetch(USER_AGENT, url)

    # -- reuse of verified payloads from earlier captures (never their observations or validators) --------

    def _load_seeds(self) -> dict[str, tuple[RawArchive, str, int, str]]:
        if self._seeds is None:
            seeds: dict[str, tuple[RawArchive, str, int, str]] = {}
            for root in self.seed_from:
                path = root / "manifest.json"
                if not path.is_file():
                    raise CaptureError(f"--seed-from {root}: no manifest.json")
                try:
                    prior = S.Manifest.model_validate_json(path.read_bytes())
                except ValueError as exc:
                    raise CaptureError(f"--seed-from {root}: invalid manifest: {exc}") from exc
                arch = RawArchive(root)
                for r in prior.resources:
                    if r.status == "usable" and r.sha256 and r.http_status == 200:
                        seeds.setdefault(r.url, (arch, r.sha256, r.observation or 0, str(root)))
            self._seeds = seeds
        return self._seeds

    def _reused(self, kind: str, url: str, reval: bool) -> FetchResult | None:
        if reval or kind == "robots" or not self.seed_from:
            return None  # revalidation and robots are always fetched fresh
        hit = self._load_seeds().get(url)
        if hit is None:
            return None
        arch, sha, _obs, _root = hit
        content = arch.get(sha)  # re-hashes the bytes: a missing or corrupt object is simply not reused
        if content is None:
            return None
        return FetchResult(
            url=url,
            final_url=url,
            source="reused",
            outcome=CheckOutcome.PASS,
            source_mode=SourceMode.CACHED,
            freshness="unknown",
            fetched_at=datetime.fromtimestamp(self.clock.time(), UTC),
            http_status=200,
            content=content,
            sha256=sha,
            bytes=len(content),
            attempts=0,
            requests=0,
        )

    def _attempt(self, tid: str, kind: str, url: str, reval: bool, attempts: int) -> str:
        reused = self._reused(kind, url, reval)
        if reused is not None:
            started = finished = self.clock.time()
            res, reqs = reused, 0
            seed = self._load_seeds()[url]
            extra: dict[str, Any] = {"reused_from": seed[3], "original_observation": seed[2]}
            self._reused_count += 1
        else:
            self._wait_for_slot()
            started = self.clock.time()
            before = sum(self.client.request_counts.values())
            res = self.client.fetch(url, conditional=False)
            finished = self.clock.time()
            reqs = sum(self.client.request_counts.values()) - before
            self._set_slot(started, finished, max(reqs, 1), res.retry_after_s)
            extra = {}
        attempt_no = attempts + 1
        obs = self._journal(
            {
                "task_id": tid,
                "kind": kind,
                "url": url,
                "final_url": res.final_url,
                "fetched_at": self._iso(started),
                "http_status": res.http_status,
                "outcome": res.outcome.value,
                "sha256": res.sha256,
                "bytes": res.bytes,
                "attempt": attempt_no,
                "requests": reqs,
                "retry_after_s": res.retry_after_s,
                "error": res.error,
                **extra,
            }
        )
        with self._txn():
            return self._record(tid, kind, url, reval, attempt_no, res, obs, finished)

    def _record(
        self, tid: str, kind: str, url: str, reval: bool, attempt: int, res: FetchResult, obs: int, now: float
    ) -> str:
        status = res.http_status
        if res.ok and res.content is not None and status == 200:
            problem = D.page_problem(kind, res.content, url) if kind != "robots" else None
            if problem is None:
                return self._accept(tid, kind, url, reval, attempt, res, obs)
            if "challenge" in problem or "login" in problem:
                return self._block(tid, attempt, f"{problem} served with HTTP 200")
            if "sent off" in problem:
                self._store(res)
                return self._finalise(tid, "missing", attempt, res, obs, f"HTTP 200 error page: {problem}")
            return self._retry_or_fail(tid, attempt, f"unusable page: {problem}", now, res=res, obs=obs)
        if res.error and res.error.startswith("url rejected"):
            return self._finalise(tid, "rejected", attempt, res, None, res.error)
        if status in (404, 410):
            if kind == "robots":
                return self._finalise(tid, "absent", attempt, res, None, f"HTTP {status}: no robots restrictions")
            return self._finalise(tid, "missing", attempt, res, None, f"HTTP {status}")
        if status in (401, 403):
            return self._block(tid, attempt, f"HTTP {status}")
        if status == 429 or (status == 503 and res.retry_after_s is not None):
            return self._rate_limited(tid, attempt, res, now)
        if status == 304:
            return self._finalise(
                tid, "failed", attempt, res, None, "unsolicited 304 (no conditional request was sent)"
            )
        if res.error and (res.error.startswith(("oversized", "redirect")) or res.error.startswith("blocked:")):
            return self._finalise(tid, "failed", attempt, res, None, res.error)
        return self._retry_or_fail(tid, attempt, res.error or f"HTTP {status}", now, res=res, obs=None)

    def _store(self, res: FetchResult) -> None:
        assert res.content is not None
        sha = self.archive.put(res.content)
        if sha != res.sha256:
            raise CaptureError("archive digest disagrees with the fetch digest")

    def _finalise(
        self, tid: str, state: str, attempt: int, res: FetchResult, obs: int | None, reason: str | None
    ) -> str:
        self.db.execute(
            "UPDATE task SET state=?, attempts=?, http_status=?, final_url=?, sha256=?, bytes=?, reason=?, observation=? WHERE task_id=?",  # noqa: E501
            (
                state,
                attempt,
                res.http_status,
                res.final_url,
                res.sha256 if obs else None,
                res.bytes if obs else None,
                reason,
                obs,
                tid,
            ),
        )
        return "gap" if state != "absent" else "ok"

    def _retry_or_fail(
        self, tid: str, attempt: int, reason: str, now: float, *, res: FetchResult, obs: int | None
    ) -> str:
        if attempt >= MAX_ATTEMPTS:
            self._finalise(tid, "failed", attempt, res, obs, f"exhausted {attempt} attempts: {reason}")
            return "exhausted"
        delay = TRANSIENT_BACKOFF_S[min(attempt - 1, len(TRANSIENT_BACKOFF_S) - 1)]
        self.db.execute(
            "UPDATE task SET attempts=?, reason=?, next_eligible=? WHERE task_id=?", (attempt, reason, now + delay, tid)
        )
        return "retry"

    def _rate_limited(self, tid: str, attempt: int, res: FetchResult, now: float) -> str:
        wait = max(res.retry_after_s if res.retry_after_s is not None else 300.0 * attempt, MIN_SPACING_S)
        if attempt >= MAX_ATTEMPTS:
            self._finalise(
                tid, "failed", attempt, res, None, f"exhausted {attempt} attempts: HTTP {res.http_status} rate limited"
            )
            self._set_meta("pause_reason", f"rate limited; next request not before {self._iso(now + wait)}")
            return "pause"
        self.db.execute(
            "UPDATE task SET attempts=?, reason=?, next_eligible=? WHERE task_id=?",
            (attempt, f"HTTP {res.http_status}: Retry-After {wait:g}s", now + wait, tid),
        )
        self._set_meta("host_next_eligible", repr(max(float(self._meta("host_next_eligible", "0") or 0), now + wait)))
        return "retry"

    def _block(self, tid: str, attempt: int, reason: str) -> str:
        self.db.execute("UPDATE task SET state='blocked', attempts=?, reason=? WHERE task_id=?", (attempt, reason, tid))
        self._set_meta("blocked", reason)
        return "blocked"

    # -- accepting a usable page -----------------------------------------------------------

    def _accept(self, tid: str, kind: str, url: str, reval: bool, attempt: int, res: FetchResult, obs: int) -> str:
        assert res.content is not None
        self._store(res)
        content, final = res.content, res.final_url
        if reval:
            return self._accept_revalidation(tid, kind, url, attempt, res, obs)
        problems: list[str] = []
        if kind == "robots":
            self._robots = urllib.robotparser.RobotFileParser()
            self._robots.parse(content.decode("utf-8", "replace").splitlines())
        elif kind == "stats_index":
            problems = self._discover_index(content, final, tid)
        elif kind == "letter":
            problems = self._discover_letter(content, final, tid, url)
        elif kind == "season":
            problems = self._discover_season(content, final, tid)
        elif kind == "match":
            self._discover_match(content, final, tid)
            self._set_meta("last_pm", "match")
        elif kind == "profile":
            self._discover_profile(content, final, tid)
            self._set_meta("last_pm", "profile")
        reason = "; ".join(problems) if problems else None
        state = "failed" if problems and kind in ("stats_index", "letter") else "done"
        self.db.execute(
            "UPDATE task SET state=?, attempts=?, http_status=?, final_url=?, sha256=?, bytes=?, reason=?, observation=?, parser_ok=? WHERE task_id=?",  # noqa: E501
            (state, attempt, res.http_status, final, res.sha256, res.bytes, reason, obs, int(not problems), tid),
        )
        for p in problems:
            self._reason(f"{kind} {url}: {p}")
        return "ok" if state == "done" else "gap"

    def _accept_revalidation(self, tid: str, kind: str, url: str, attempt: int, res: FetchResult, obs: int) -> str:
        orig = self.db.execute("SELECT sha256 FROM task WHERE url=? AND reval=0", (url,)).fetchone()
        changed = orig is not None and orig[0] != res.sha256
        reason = None
        if changed:
            reason = f"changed during acquisition: {orig[0][:12]} -> {str(res.sha256)[:12]}"
            self._reason(f"revalidation: {kind} {url} {reason}")
        self.db.execute(
            "UPDATE task SET state='done', attempts=?, http_status=?, final_url=?, sha256=?, bytes=?, reason=?, observation=?, parser_ok=1 WHERE task_id=?",  # noqa: E501
            (attempt, res.http_status, res.final_url, res.sha256, res.bytes, reason, obs, tid),
        )
        return "ok"

    def _discover_index(self, content: bytes, final: str, tid: str) -> list[str]:
        info = D.read_stats_index(content, final)
        problems = list(info.problems)
        required = _required_seasons(self.plan, self.through)
        for y in required:
            if y in info.seasons:
                self._add("season", f"{SITE}/afl/seas/{y}.html", f"index:{tid}")
            else:
                problems.append(f"season {y} missing from the statistics index")
        # the directory census defines the full population; a seasons audit takes its players from the lineups
        if info.all_players_url and self.plan.scope.population != "seasons":
            self._add("letter", info.all_players_url, f"index:{tid}")
        return problems

    def _discover_letter(self, content: bytes, final: str, tid: str, url: str) -> list[str]:
        page = D.read_census_page(content, final)
        problems = list(page.problems)
        if self.plan.scope.population == "sample":
            wanted = set(self.plan.scope.sample_profiles)
            profiles = [p for p in page.profiles if p.url in wanted]
        else:
            profiles = page.profiles
        for _letter, link in page.nav:
            if link:
                self._add("letter", link, f"nav:{tid}")
        for p in profiles:
            self.db.execute(
                "INSERT OR IGNORE INTO census(url,letter,display_name) VALUES(?,?,?)",
                (p.url, page.letter or "?", p.display_name),
            )
            self._add("profile", p.url, f"letter:{tid}")
        # a profile link the policy rejects is stored as a rejected task: a coverage gap, never a silent drop
        return problems

    def _discover_season(self, content: bytes, final: str, tid: str) -> list[str]:
        info = D.read_season_page(content, final)
        problems = list(info.problems)
        listed: set[str] = set()
        for fx in info.fixtures:
            if fx.game_url is None:
                continue
            listed.add(fx.game_url)
            year = info.year or 0
            in_scope = fx.match_date is None or fx.match_date <= self.through
            self.db.execute(
                "INSERT OR IGNORE INTO matchinfo(url,year,match_date,in_scope,listed) VALUES(?,?,?,?,1)",
                (fx.game_url, year, fx.match_date.isoformat() if fx.match_date else None, int(in_scope)),
            )
            if in_scope:
                self._add("match", fx.game_url, f"season:{tid}")
        # a game link on the page that no fixture row explains is fetched, never dropped
        for link, _ in D.anchors(content, final):
            if D._GAME_PATH.match(link) and link not in listed:
                y = int(link.split("/games/")[1][:4])
                self.db.execute(
                    "INSERT OR IGNORE INTO matchinfo(url,year,match_date,in_scope,listed) VALUES(?,?,NULL,1,0)",
                    (link, y),
                )
                if y <= self.through.year:
                    self._add("match", link, f"season:{tid}")
        return problems

    def _discover_match(self, content: bytes, final: str, tid: str) -> None:
        for link in D.match_profile_links(content, final):
            self.db.execute("INSERT OR IGNORE INTO lineup(url) VALUES(?)", (link,))
            self._add("profile", link, f"match:{tid}")

    def _discover_profile(self, content: bytes, final: str, tid: str) -> None:
        for link in D.profile_game_links(content, final):
            row = self.db.execute("SELECT in_scope FROM matchinfo WHERE url=?", (link,)).fetchone()
            if row is not None:
                continue  # inventoried by its season page (already a task, or excluded by date)
            year = int(link.split("/games/")[1][:4])
            if self.plan.scope.population == "seasons" and year not in self.plan.scope.seasons:
                continue  # another season of the player's career: outside a seasons audit, never fetched
            if year > self.through.year:
                self.db.execute(
                    "INSERT OR IGNORE INTO matchinfo(url,year,match_date,in_scope,listed) VALUES(?,?,NULL,0,0)",
                    (link, year),
                )
                continue
            self.db.execute(
                "INSERT OR IGNORE INTO matchinfo(url,year,match_date,in_scope,listed) VALUES(?,?,NULL,1,0)",
                (link, year),
            )
            if self._add("match", link, f"profile:{tid}"):
                self._reason(f"profile-linked match not on any season page: {link}")

    # -- revalidation (DESIGN section 5) -------------------------------------------------------

    def _start_revalidation(self) -> bool:
        if not self.revalidate or self._meta("revalidation_started"):
            return False
        self._set_meta("revalidation_started", self._iso(self.clock.time()))
        with self._txn():
            for kind, url in self.db.execute(
                "SELECT kind,url FROM task WHERE reval=0 AND kind IN ('letter','season') AND state='done' ORDER BY seq"
            ).fetchall():
                self._add(kind, url, "revalidation", reval=True)
        return bool(self.db.execute("SELECT 1 FROM task WHERE reval=1 AND state='pending' LIMIT 1").fetchone())

    # -- finishing ------------------------------------------------------------------------------

    def _heartbeat(self, url: str) -> bool:
        """Log and persist progress about once a minute; ``False`` when the disk is nearly full."""
        now = self.clock.time()
        if now - self._last_beat < self.heartbeat_s:
            return True
        self._last_beat = now
        c = self.counts()
        eta_h = c["pending"] * MIN_SPACING_S / 3600
        self.log(
            f"[capture {self._iso(now)}] done={c['done']} pending={c['pending']} gaps={c['gaps']} "
            f"requests={self._requests} min_remaining_h={eta_h:.1f} last={url.rsplit('/', 1)[-1]}"
        )
        free = shutil.disk_usage(self.dir).free
        atomic_write_bytes(
            self.dir / "progress.json",
            canonical_bytes(
                {
                    "updated_utc": self._iso(now),
                    "counts": c,
                    "requests_this_process": self._requests,
                    "min_remaining_hours": round(eta_h, 2),
                    "last": url,
                    "free_disk_bytes": free,
                }
            ),
        )
        return free >= MIN_FREE_DISK_BYTES

    def counts(self) -> dict[str, int]:
        rows = dict(self.db.execute("SELECT state, COUNT(*) FROM task GROUP BY state").fetchall())
        out = {
            k: int(rows.get(k, 0)) for k in ("pending", "done", "absent", "failed", "missing", "blocked", "rejected")
        }
        out["gaps"] = sum(out[k] for k in GAP_STATES)
        return out

    def _finish(self, state: str, reason: str | None) -> CaptureResult:
        manifest = build_manifest(self.db, self.plan, self._meta("started_utc"), self._iso(self.clock.time()))
        atomic_write_bytes(self.dir / "manifest.json", S.canonical_dump(manifest))
        counts = self.counts()
        if state == "complete" and not manifest.capture_complete:
            state, reason = "incomplete", "; ".join(manifest.incomplete_reasons[:5]) or "capture incomplete"
        receipt = {
            "state": state,
            "reason": reason,
            "plan_id": self.plan.plan_id,
            "counts": counts,
            "requests_this_process": self._requests,
            "reused_objects_this_process": self._reused_count,
            "capture_complete": manifest.capture_complete,
            "execution_complete": manifest.execution_complete,
            "updated_utc": self._iso(self.clock.time()),
            "resume": "scvia reconcile-afltables capture --plan <plan.json> --allow-network --resume",
        }
        atomic_write_bytes(self.dir / "receipt.json", canonical_bytes(receipt))
        code = 0 if state == "complete" and manifest.capture_complete else 8
        return CaptureResult(code, state, reason, counts, self._requests)


# ---------------------------------------------------------------------------
# Manifest (offline: built from the checkpoint alone)
# ---------------------------------------------------------------------------


def _required_seasons(plan: S.Plan, through: date) -> list[int]:
    if plan.scope.population == "seasons":
        return [y for y in plan.scope.seasons if y <= through.year]
    return list(range(plan.scope.first_season, through.year + 1))


def build_manifest(db: sqlite3.Connection, plan: S.Plan, started: str | None, finished: str | None) -> S.Manifest:
    through = date.fromisoformat(plan.scope.through_date)
    rows = db.execute(
        "SELECT task_id,kind,url,reval,state,attempts,http_status,final_url,sha256,bytes,reason,observation,parser_ok FROM task ORDER BY kind,url,reval"  # noqa: E501
    ).fetchall()
    refs: dict[str, list[str]] = {}
    for tid, src in db.execute("SELECT task_id,source FROM ref ORDER BY task_id,source"):
        refs.setdefault(tid, []).append(src)
    status_map = {
        "done": "usable",
        "absent": "absent",
        "failed": "failed",
        "missing": "missing",
        "blocked": "blocked",
        "rejected": "rejected",
        "pending": "pending",
    }
    resources: list[S.Resource] = []
    counts: dict[str, dict[str, int]] = {}
    changed: list[str] = []
    reval_unfinished = 0
    for tid, kind, url, reval, state, attempts, http_status, final, sha, nbytes, reason, obs, parser_ok in rows:
        if reval:
            if reason and reason.startswith("changed"):
                changed.append(url)
            if state != "done":
                reval_unfinished += 1  # B6: an interrupted revalidation can never certify completeness
            continue
        status = status_map[state]
        refs_for = refs.get(tid, [])
        resources.append(
            S.Resource(
                task_id=tid,
                kind=kind,
                url=url,
                final_url=final,
                status=status,  # type: ignore[arg-type]
                http_status=http_status,
                sha256=sha,
                bytes=nbytes,
                attempts=attempts,
                observation=obs,
                reason=reason,
                parser_status="supported" if parser_ok else ("unsupported" if state == "done" else "unparsed"),
                discovered_from=refs_for[:3],
                discovered_from_count=len(refs_for),
            )
        )
        counts.setdefault(kind, {})
        counts[kind][status] = counts[kind].get(status, 0) + 1
    letters = [r for r in resources if r.kind == "letter"]
    census_urls = {u for (u,) in db.execute("SELECT url FROM census")}
    lineup = {u for (u,) in db.execute("SELECT url FROM lineup")}
    rejected = sorted(r.url for r in resources if r.status == "rejected")
    letter_failed = sorted(r.url for r in letters if r.status != "usable")
    seasons_usable = sorted(
        int(r.url.rsplit("/", 1)[1][:4]) for r in resources if r.kind == "season" and r.status == "usable"
    )
    required = _required_seasons(plan, through)
    census = plan.scope.population != "seasons"
    in_scope = db.execute("SELECT COUNT(*) FROM matchinfo WHERE in_scope=1").fetchone()[0]
    out_scope = db.execute("SELECT COUNT(*) FROM matchinfo WHERE in_scope=0").fetchone()[0]
    undated = db.execute("SELECT COUNT(*) FROM matchinfo WHERE in_scope=1 AND match_date IS NULL").fetchone()[0]
    reasons = sorted(t for (t,) in db.execute("SELECT text FROM reason"))
    pending = sum(1 for r in resources if r.status == "pending")
    gaps = [r for r in resources if r.status in ("failed", "missing", "blocked", "rejected")]
    not_in_dir = sorted(lineup - census_urls) if plan.scope.population == "all" else []
    incomplete = list(reasons)
    if pending:
        incomplete.append(f"{pending} resources still pending")
    if gaps:
        incomplete.append(f"{len(gaps)} resources failed, missing, blocked or rejected")
    if census and (len(letters) != 26 or letter_failed):
        incomplete.append("the directory census is incomplete (a letter page is missing or failed)")
    if seasons_usable != required:
        incomplete.append(f"season pages usable for {len(seasons_usable)} of {len(required)} required seasons")
    if not_in_dir:
        incomplete.append(f"{len(not_in_dir)} lineup-linked profiles are absent from the directory census")
    if changed:
        incomplete.append(f"{len(changed)} census/season pages changed during acquisition")
    if undated:
        incomplete.append(f"{undated} in-scope matches have no source date (inclusion unclear)")
    if plan.scope.population == "sample":
        incomplete.append("sample population: not a full audit")
    if reval_unfinished:
        incomplete.append(f"end-of-acquisition revalidation is unfinished ({reval_unfinished} tasks not done)")
    if db.execute("SELECT 1 FROM meta WHERE key='revalidation_started'").fetchone() is None:
        incomplete.append("end-of-acquisition revalidation has not run")
    return S.Manifest(
        plan_id=plan.plan_id,
        capture_identity=plan.capture_identity,
        reference_mode="observed_current",
        acquisition_started_utc=started,
        acquisition_finished_utc=finished,
        execution_complete=pending == 0,
        capture_complete=not incomplete,
        incomplete_reasons=incomplete,
        seasons=seasons_usable,
        scope_match_count=int(in_scope),
        out_of_scope_match_count=int(out_scope),
        undated_in_scope_unknown=int(undated),
        census=S.ManifestCensus(
            letters_expected=26 if census else 0,
            letters_usable=len(letters) - len(letter_failed),
            letters_failed=letter_failed,
            profiles_in_directory=len(census_urls),
            profile_urls_rejected=rejected,
            lineup_profiles_not_in_directory=not_in_dir,
            revalidation_changed=sorted(changed),
        ),
        resource_counts=counts,
        resources=resources,
    )
