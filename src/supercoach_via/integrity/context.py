"""State shared by the check families during one audit."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from supercoach_via.integrity.capture import EvidenceStore, ExternalCapture, ReleaseCapture, SnapshotCapture
from supercoach_via.integrity.policy import IntegrityPolicy
from supercoach_via.integrity.report import Collector, RuleSpec, Status

if TYPE_CHECKING:  # pragma: no cover
    import duckdb


class CheckSkipped(Exception):
    """A check could not run (UNKNOWN) or does not apply (NOT_APPLICABLE)."""

    def __init__(self, status: Status, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass(frozen=True)
class CheckSpec:
    check_id: str
    family: str
    summary: str
    fn: Callable[[AuditContext], list[str] | None]
    #: inputs that must be supplied for the check to be requested at all in a scope
    needs: tuple[str, ...] = ("snapshot",)
    #: required checks that cannot run make the audit incomplete (UNKNOWN)
    required: bool = True


@dataclass
class AuditContext:
    collector: Collector
    policy: IntegrityPolicy
    snapshot: SnapshotCapture | None
    release: ReleaseCapture | None
    evidence: EvidenceStore
    as_of: datetime | None
    current_season: int | None
    models_root: Path | None
    predictions_root: Path | None
    workers: int = 1
    evaluation_dirs: tuple[Path, ...] = ()
    live_root: Path | None = None
    content_root: Path | None = None
    content_manifest: Path | None = None
    examined: dict[str, dict[str, int]] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    check_id: str = ""
    _con: duckdb.DuckDBPyConnection | None = None
    _registered: dict[str, bool] = field(default_factory=dict)
    cache: Any = None
    external: ExternalCapture = field(default_factory=ExternalCapture)

    def add(self, rule_id: str, entity: str, **kw: Any) -> None:
        self.collector.add(rule_id, entity, **kw)

    def count(self, key: str, n: int = 1) -> None:
        bucket = self.examined.setdefault(self.check_id, {})
        bucket[key] = bucket.get(key, 0) + int(n)

    def db(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            import duckdb

            con = duckdb.connect(database=":memory:")
            # one thread: float aggregation order and therefore results never depend on
            # host parallelism; timestamps are rendered in UTC regardless of the host zone
            con.execute("SET threads TO 1")
            con.execute("SET TimeZone = 'UTC'")
            self._con = con
        return self._con

    def need(self, *tables: str) -> None:
        """Register verified snapshot tables in DuckDB or skip the check as UNKNOWN."""
        if self.snapshot is None or self.snapshot.manifest is None:
            raise CheckSkipped(Status.UNKNOWN, "no verifiable snapshot manifest")
        con = self.db()
        missing = []
        for name in tables:
            if name in self._registered:
                if not self._registered[name]:
                    missing.append(name)
                continue
            if name not in self.snapshot.manifest.tables:
                self._registered[name] = False
                missing.append(name)
                continue
            t = self.snapshot.table(name)
            if t is None:
                self._registered[name] = False
                missing.append(name)
                continue
            con.register(name, t)
            self._registered[name] = True
        if missing:
            raise CheckSkipped(Status.UNKNOWN, f"table(s) not verifiable: {', '.join(sorted(missing))}")

    def has(self, name: str) -> bool:
        return (
            self.snapshot is not None and self.snapshot.manifest is not None and name in self.snapshot.manifest.tables
        )

    def rows(self, sql: str, params: list[Any] | None = None) -> list[tuple[Any, ...]]:
        return self.db().execute(sql, params or []).fetchall()

    def records(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        cur = self.db().execute(sql, params or [])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None


def rule(rule_id: str, check_id: str, severity: Any, summary: str, action: str, **kw: Any) -> RuleSpec:
    return RuleSpec(rule_id, check_id, severity, summary, action, **kw)
