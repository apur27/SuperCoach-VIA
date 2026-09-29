"""Integrity report contract: findings, exact counts, bounded samples and canonical bytes.

Severity follows ``domain.schemas.Severity`` and the ``validate_dataset`` convention: a
defect in the current season is escalated to ``blocking`` for rules that allow it, an
accepted exception needs an exact ``(rule_id, entity)`` match and a reason, and a
current-season defect can never be suppressed. The overall outcome is ``FAIL`` when any
open ``blocking`` finding exists, ``UNKNOWN`` when a required check could not run, and
``PASS`` otherwise. ``error`` / ``warning`` / ``info`` findings are reported with exact
counts and permit exit 0, exactly as they permit promotion today.

The canonical report excludes wall-clock times, hosts, absolute paths and cache reuse, so
identical input bytes and options produce identical report bytes.
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from supercoach_via.domain.schemas import Severity

REPORT_SCHEMA = "scvia.integrity-report/1"
EXCEPTION_REJECTED = "policy.exception_rejected_current_season"
EXCEPTION_STALE = "policy.exception_stale"
_SEVERITY_ORDER = {Severity.INFO: 0, Severity.WARNING: 1, Severity.ERROR: 2, Severity.BLOCKING: 3}


class Status(StrEnum):
    PASS = "PASS"  # noqa: S105 - verdict label
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class Outcome(StrEnum):
    PASS = "PASS"  # noqa: S105 - verdict label
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class Kind(StrEnum):
    CONTRADICTION = "contradiction"  # two inputs disagree, or bytes do not match their identity
    ANOMALY = "anomaly"  # a value breaks a documented football/data rule
    MISSING_EVIDENCE = "missing_evidence"  # a check cannot be decided from what was supplied
    POLICY = "policy"  # exception / policy bookkeeping


#: exit codes (``scvia`` convention; 8 and 9 are the checker's additions)
EXIT_OK, EXIT_USAGE, EXIT_VIOLATION, EXIT_INCOMPLETE, EXIT_CHECKER_ERROR = 0, 2, 4, 8, 9


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    check_id: str
    severity: Severity
    summary: str
    action: str
    kind: Kind = Kind.CONTRADICTION
    #: escalate to blocking when the finding's season is the current season
    current_blocks: bool = False


@dataclass(frozen=True)
class AcceptedException:
    rule_id: str
    entity: str
    reason: str


def canonical_bytes(payload: Any) -> bytes:
    """Sorted keys, compact separators, UTF-8, trailing newline. NaN/Infinity raise."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite number in a finding")
        return int(value) if value.is_integer() and abs(value) < 2**53 else value
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


@dataclass(frozen=True)
class Finding:
    rule_id: str
    check_id: str
    kind: Kind
    severity: Severity
    status: str  # open | accepted
    entity: str
    table: str | None
    field: str | None
    season: int | None
    expected: Any
    actual: Any
    evidence: Mapping[str, Any]
    message: str
    action: str
    acceptance: str | None = None

    @staticmethod
    def make(
        spec: RuleSpec,
        entity: str,
        *,
        table: str | None = None,
        field: str | None = None,
        season: int | None = None,
        expected: Any = None,
        actual: Any = None,
        evidence: Mapping[str, Any] | None = None,
        message: str = "",
        severity: Severity | None = None,
        status: str = "open",
        acceptance: str | None = None,
    ) -> Finding:
        return Finding(
            rule_id=spec.rule_id,
            check_id=spec.check_id,
            kind=spec.kind,
            severity=severity or spec.severity,
            status=status,
            entity=entity,
            table=table,
            field=field,
            season=season,
            expected=_jsonable(expected),
            actual=_jsonable(actual),
            evidence=_jsonable(dict(evidence or {})),
            message=message or spec.summary,
            action=spec.action,
            acceptance=acceptance,
        )

    @property
    def issue_id(self) -> str:
        raw = "\x1f".join((self.rule_id, self.entity, self.table or "", self.field or ""))
        return "ic:" + hashlib.sha256(raw.encode()).hexdigest()[:24]

    @property
    def sort_key(self) -> tuple[str, str, str, str]:
        return (self.rule_id, self.entity, self.table or "", self.field or "")

    def as_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "rule_id": self.rule_id,
            "check_id": self.check_id,
            "kind": self.kind.value,
            "severity": self.severity.value,
            "status": self.status,
            "entity": self.entity,
            "table": self.table,
            "field": self.field,
            "season": self.season,
            "expected": self.expected,
            "actual": self.actual,
            "evidence": self.evidence,
            "message": self.message,
            "action": self.action,
            "acceptance": self.acceptance,
        }


@dataclass
class _RuleState:
    total: int = 0
    open: int = 0
    accepted: int = 0
    by_severity: dict[str, int] = field(default_factory=dict)
    open_by_severity: dict[str, int] = field(default_factory=dict)
    unlisted: int = 0
    #: max-heap (negated keys) of the ``sample_limit`` smallest findings
    heap: list[tuple[_Neg, int, Finding]] = field(default_factory=list)


class _Neg:
    """Order-reversing wrapper so ``heapq`` (a min-heap) keeps the smallest keys."""

    __slots__ = ("key",)

    def __init__(self, key: tuple[str, ...]):
        self.key = key

    def __lt__(self, other: _Neg) -> bool:
        return self.key > other.key

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Neg) and self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key)


class Collector:
    """Accumulates findings with exact counts and a deterministic bounded sample per rule.

    The sample holds the ``sample_limit`` findings with the smallest sort keys, so it does
    not depend on insertion order (worker count, query order, directory order). With
    ``keep_all`` every finding is also kept for the optional complete findings stream.
    """

    def __init__(
        self,
        rules: Mapping[str, RuleSpec],
        *,
        sample_limit: int,
        current_season: int | None,
        exceptions: Iterable[AcceptedException] = (),
        keep_all: bool = False,
    ):
        self.rules = dict(rules)
        self.rules.setdefault(
            EXCEPTION_REJECTED,
            RuleSpec(
                EXCEPTION_REJECTED,
                "policy.exceptions",
                Severity.BLOCKING,
                "accepted exception names a current-season defect; current-season defects cannot be suppressed",
                "remove the exception and repair the current-season defect",
                Kind.POLICY,
            ),
        )
        self.rules.setdefault(
            EXCEPTION_STALE,
            RuleSpec(
                EXCEPTION_STALE,
                "policy.exceptions",
                Severity.WARNING,
                "accepted exception matched no finding",
                "delete the stale exception from the policy",
                Kind.POLICY,
            ),
        )
        self.sample_limit = sample_limit
        self.current_season = current_season
        self.keep_all = keep_all
        self._exceptions = {(e.rule_id, e.entity): e for e in exceptions}
        self._used: set[tuple[str, str]] = set()
        self._rejected: set[tuple[str, str]] = set()
        self._state: dict[str, _RuleState] = {}
        self._all: list[Finding] = []
        self._seq = 0

    def add(
        self,
        rule_id: str,
        entity: str,
        *,
        table: str | None = None,
        field: str | None = None,
        season: int | None = None,
        expected: Any = None,
        actual: Any = None,
        evidence: Mapping[str, Any] | None = None,
        message: str = "",
        severity: Severity | None = None,
        accepted_by_producer: str | None = None,
    ) -> None:
        """Record one finding.

        ``severity`` overrides the rule's severity for findings mapped from a producer
        check that already applied its own era escalation (``validate_dataset``);
        ``accepted_by_producer`` carries that check's own exception acceptance.
        """
        spec = self.rules[rule_id]
        current = season is not None and season == self.current_season
        if severity is None:
            severity = spec.severity
            if spec.current_blocks and current and severity is not Severity.INFO:
                severity = Severity.BLOCKING
        status, acceptance = ("accepted", accepted_by_producer) if accepted_by_producer else ("open", None)
        exc = self._exceptions.get((rule_id, entity))
        if exc is not None:
            self._used.add((rule_id, entity))
            if current:
                self._rejected.add((rule_id, entity))
                self._put(
                    Finding.make(
                        self.rules[EXCEPTION_REJECTED],
                        entity,
                        season=season,
                        evidence={"rule_id": rule_id, "reason": exc.reason},
                    )
                )
            else:
                status, acceptance = "accepted", exc.reason
        self._put(
            Finding.make(
                spec,
                entity,
                table=table,
                field=field,
                season=season,
                expected=expected,
                actual=actual,
                evidence=evidence,
                message=message,
                severity=severity,
                status=status,
                acceptance=acceptance,
            )
        )

    def count_unlisted(self, rule_id: str, n: int, severity: Severity) -> None:
        """Rows a producer check found but did not list individually (exact total, no sample)."""
        if n <= 0:
            return
        st = self._state.setdefault(rule_id, _RuleState())
        st.total += n
        st.open += n
        st.unlisted += n
        st.by_severity[severity.value] = st.by_severity.get(severity.value, 0) + n
        st.open_by_severity[severity.value] = st.open_by_severity.get(severity.value, 0) + n

    @property
    def stream_complete(self) -> bool:
        return not any(st.unlisted for st in self._state.values())

    def extend(self, findings: Iterable[Finding]) -> None:
        """Adopt raw findings produced elsewhere (worker processes, the semantic cache).

        Their severity was already decided where they were produced; policy exceptions are
        applied here, once, like any other finding.
        """
        for f in findings:
            self.add(
                f.rule_id,
                f.entity,
                table=f.table,
                field=f.field,
                season=f.season,
                expected=f.expected,
                actual=f.actual,
                evidence=f.evidence,
                message=f.message,
                severity=f.severity,
            )

    def _put(self, f: Finding) -> None:
        st = self._state.setdefault(f.rule_id, _RuleState())
        st.total += 1
        if f.status == "accepted":
            st.accepted += 1
        else:
            st.open += 1
        st.by_severity[f.severity.value] = st.by_severity.get(f.severity.value, 0) + 1
        if f.status != "accepted":
            st.open_by_severity[f.severity.value] = st.open_by_severity.get(f.severity.value, 0) + 1
        self._seq += 1
        item = (_Neg(f.sort_key), self._seq, f)
        if len(st.heap) < self.sample_limit:
            heapq.heappush(st.heap, item)
        elif f.sort_key < st.heap[0][0].key:
            heapq.heapreplace(st.heap, item)
        if self.keep_all:
            self._all.append(f)

    def finish_exceptions(self) -> None:
        for key, exc in sorted(self._exceptions.items()):
            if key not in self._used:
                self._put(
                    Finding.make(
                        self.rules[EXCEPTION_STALE], exc.entity, evidence={"rule_id": exc.rule_id, "reason": exc.reason}
                    )
                )

    def samples(self) -> list[Finding]:
        out = [item[2] for st in self._state.values() for item in st.heap]
        return sorted(out, key=lambda f: f.sort_key)

    def all_findings(self) -> list[Finding]:
        if not self.keep_all:
            raise RuntimeError("collector was created without keep_all")
        return sorted(self._all, key=lambda f: f.sort_key)

    def rule_summary(self) -> dict[str, dict[str, Any]]:
        out = {}
        for rule_id, st in sorted(self._state.items()):
            spec = self.rules[rule_id]
            out[rule_id] = {
                "rule_id": rule_id,
                "check_id": spec.check_id,
                "kind": spec.kind.value,
                "severity": spec.severity.value,
                "total": st.total,
                "open": st.open,
                "accepted": st.accepted,
                "by_severity": dict(sorted(st.by_severity.items())),
                "sampled": len(st.heap),
                "truncated": st.total > len(st.heap),
                "unlisted_by_producer": st.unlisted,
            }
        return out

    def severity_counts(self, *, open_only: bool = True) -> dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for st in self._state.values():
            for sev, n in (st.open_by_severity if open_only else st.by_severity).items():
                counts[sev] += n
        return counts

    def _open_blocking(self, check_id: str) -> bool:
        return any(
            st.open_by_severity.get(Severity.BLOCKING.value, 0)
            for rule_id, st in self._state.items()
            if self.rules[rule_id].check_id == check_id
        )

    def check_status(self, check_id: str, *, unknown: list[str]) -> Status:
        if self._open_blocking(check_id):
            return Status.FAIL
        return Status.UNKNOWN if unknown else Status.PASS

    def exception_summary(self) -> dict[str, list[dict[str, str]]]:
        def rows(keys: Iterable[tuple[str, str]]) -> list[dict[str, str]]:
            return [{"rule_id": r, "entity": e, "reason": self._exceptions[(r, e)].reason} for r, e in sorted(keys)]

        return {
            "accepted": rows(self._used - self._rejected),
            "rejected": rows(self._rejected),
            "stale": rows(set(self._exceptions) - self._used),
        }


def overall(statuses: Iterable[tuple[Status, bool]]) -> Outcome:
    """(status, required) per check -> overall outcome."""
    items = list(statuses)
    if any(s is Status.FAIL for s, _ in items):
        return Outcome.FAIL
    if any(s is Status.UNKNOWN and required for s, required in items):
        return Outcome.UNKNOWN
    return Outcome.PASS


def exit_code(outcome: Outcome) -> int:
    return {Outcome.PASS: EXIT_OK, Outcome.FAIL: EXIT_VIOLATION, Outcome.UNKNOWN: EXIT_INCOMPLETE}[outcome]


def seal_report(body: dict[str, Any]) -> dict[str, Any]:
    """Set ``report_sha256`` to the SHA-256 of the canonical report with that field empty."""
    body = {**body, "report_sha256": ""}
    body["report_sha256"] = hashlib.sha256(canonical_bytes(body)).hexdigest()
    return body


def verify_report_digest(body: Mapping[str, Any]) -> bool:
    claimed = body.get("report_sha256")
    return bool(claimed) and seal_report(dict(body))["report_sha256"] == claimed


def severity_rank(sev: Severity) -> int:
    return _SEVERITY_ORDER[sev]
