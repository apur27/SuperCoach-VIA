"""Run IDs, the whole-run writer lock, run manifests and structured event logs."""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from supercoach_via.domain.schemas import RunManifest, RunState, StepRecord, can_transition
from supercoach_via.storage.snapshots import atomic_write_bytes

Clock = Callable[[], datetime]

_SECRET_KEYS = re.compile(r"(token|secret|password|api[_-]?key|authorization|cookie)", re.I)
_URL_CREDENTIALS = re.compile(r"(https?://)[^/@\s]+@")
MAX_FIELD_CHARS = 1000


class LockedError(RuntimeError):
    """Another writer holds the data-root lock (exit code 5)."""


class InvalidTransitionError(RuntimeError):
    pass


def new_run_id(clock: Clock) -> str:
    """UTC timestamp with microseconds plus a random suffix (never minute-only)."""
    now = clock()
    return f"{now.strftime('%Y%m%dT%H%M%S.%f')}Z-{secrets.token_hex(4)}"


def redact(value: Any, key: str = "") -> Any:
    if key and _SECRET_KEYS.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        value = _URL_CREDENTIALS.sub(r"\1[REDACTED]@", value)
        if len(value) > MAX_FIELD_CHARS:
            value = value[:MAX_FIELD_CHARS] + "…[truncated]"
        return value
    if isinstance(value, dict):
        return {k: redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    return value


class WriterLock:
    """Advisory, non-blocking, whole-run lock for one data root."""

    def __init__(self, data_root: Path):
        self.path = data_root / ".writer.lock"
        self._fh: Any = None

    def __enter__(self) -> WriterLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = self.path.open("a+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            fh.close()
            raise LockedError(
                f"another writer holds {self.path}; wait for it to finish or inspect `scvia status`"
            ) from exc
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()}\n")
        fh.flush()
        self._fh = fh
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None


class RunStore:
    """A run directory ``<root>/runs/<run_id>/`` with ``run.json`` and ``events.jsonl``."""

    def __init__(self, directory: Path, manifest: RunManifest, clock: Clock):
        self.directory = directory
        self.manifest = manifest
        self.clock = clock

    @classmethod
    def create(
        cls, data_root: Path, command: str, *, clock: Clock, code_version: str, run_id: str | None = None
    ) -> RunStore:
        rid = run_id or new_run_id(clock)
        directory = data_root / "runs" / rid
        directory.mkdir(parents=True, exist_ok=False)
        now = clock()
        manifest = RunManifest(
            run_id=rid,
            command=command,
            state=RunState.CREATED,
            created_at=now,
            updated_at=now,
            code_version=code_version,
            history=[(RunState.CREATED, now)],
        )
        store = cls(directory, manifest, clock)
        store._save()
        return store

    @classmethod
    def open(cls, data_root: Path, run_id: str, *, clock: Clock) -> RunStore:
        directory = data_root / "runs" / run_id
        manifest = RunManifest.model_validate_json((directory / "run.json").read_bytes())
        return cls(directory, manifest, clock)

    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    def _save(self) -> None:
        atomic_write_bytes(self.directory / "run.json", self.manifest.model_dump_json(indent=1).encode())

    def transition(self, state: RunState, *, error_code: str | None = None, recovery: str | None = None) -> None:
        if not can_transition(self.manifest.state, state):
            raise InvalidTransitionError(f"{self.manifest.state.value} -> {state.value} is not allowed")
        now = self.clock()
        self.manifest.state = state
        self.manifest.updated_at = now
        self.manifest.history.append((state, now))
        if error_code:
            self.manifest.error_code = error_code
        if recovery:
            self.manifest.recovery = recovery
        self._save()
        self.event("state", state=state.value, error_code=error_code)

    def event(self, kind: str, **fields: Any) -> None:
        rec = {"ts": self.clock().isoformat(), "run_id": self.run_id, "event": kind, **redact(fields)}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")

    def record_step(
        self,
        name: str,
        *,
        input_hashes: dict[str, str],
        outputs: dict[str, str],
        state: Literal["pending", "running", "succeeded", "failed", "skipped", "reused"],
        counts: dict[str, int] | None = None,
        error_code: str | None = None,
        message: str | None = None,
    ) -> StepRecord:
        prev = self.manifest.steps.get(name)
        now = self.clock()
        rec = StepRecord(
            name=name,
            state=state,
            input_hashes=input_hashes,
            code_version=self.manifest.code_version,
            outputs=outputs,
            attempts=(prev.attempts if prev else 0) + 1,
            started_at=prev.started_at if prev and prev.started_at else now,
            finished_at=now,
            error_code=error_code,
            message=message,
            counts=counts or {},
        )
        self.manifest.steps[name] = rec
        self.manifest.updated_at = now
        self._save()
        return rec

    def reusable_step(self, name: str, input_hashes: dict[str, str], *, code_version: str) -> StepRecord | None:
        rec = self.manifest.steps.get(name)
        if rec is None or rec.state not in ("succeeded", "reused"):
            return None
        if rec.input_hashes != input_hashes or rec.code_version != code_version:
            return None
        return rec
