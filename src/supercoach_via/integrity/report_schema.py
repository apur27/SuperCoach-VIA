"""Versioned, validated JSON contract of the integrity report (``scvia.integrity-report/1``).

Every report is validated against :class:`IntegrityReport` before it is written;
``schemas/integrity/integrity-report.schema.json`` (outside the browser type generator's
directory scan) is generated from it and checked for drift.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from supercoach_via.integrity.report import REPORT_SCHEMA

StatusLit = Literal["PASS", "FAIL", "UNKNOWN", "NOT_APPLICABLE"]
SeverityLit = Literal["info", "warning", "error", "blocking"]
KindLit = Literal["contradiction", "anomaly", "missing_evidence", "policy"]


class _S(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, populate_by_name=True)


class CheckerIdentity(_S):
    version: str
    code_sha256: str
    rules_sha256: str
    rules: int


class PolicyIdentity(_S):
    version: str
    sha256: str
    files: dict[str, str]


class Inputs(_S):
    snapshot: dict[str, Any] | None
    release: dict[str, Any] | None
    evidence: dict[str, Any]
    models: dict[str, Any] | None
    digest: str


class Scope(_S):
    name: Literal["full", "data"]
    as_of: str | None
    families: list[str]
    restricted: bool
    complete: bool
    current_season: int | None


class Counts(_S):
    checks_requested: int
    checks_performed: int
    checks: dict[str, int]
    findings_open: dict[str, int]
    findings_total: dict[str, int]
    rows_examined: int
    resources_examined: int
    cells_compared: int
    examined: dict[str, int]


class CheckRow(_S):
    check_id: str
    family: str
    summary: str
    required: bool
    status: StatusLit
    reason: str
    examined: dict[str, int]
    findings: dict[str, int]


class RuleRow(_S):
    rule_id: str
    check_id: str
    kind: KindLit
    severity: SeverityLit
    total: int
    open: int
    accepted: int
    by_severity: dict[str, int]
    sampled: int
    truncated: bool
    unlisted_by_producer: int


class FindingRow(_S):
    issue_id: str
    rule_id: str
    check_id: str
    kind: KindLit
    severity: SeverityLit
    status: Literal["open", "accepted"]
    entity: str
    table: str | None
    field: str | None
    season: int | None
    expected: Any
    actual: Any
    evidence: dict[str, Any]
    message: str
    action: str
    acceptance: str | None


class ExceptionRow(_S):
    rule_id: str
    entity: str
    reason: str


class Exceptions(_S):
    accepted: list[ExceptionRow]
    rejected: list[ExceptionRow]
    stale: list[ExceptionRow]


class FindingsStream(_S):
    count: int
    sha256: str
    complete: bool = Field(description="false when a producer check listed only part of its rows")
    unlisted_by_producer: int


class IntegrityReport(_S):
    schema_: Literal["scvia.integrity-report/1"] = Field(alias="schema")
    checker: CheckerIdentity
    policy: PolicyIdentity
    inputs: Inputs
    scope: Scope
    outcome: Literal["PASS", "FAIL", "UNKNOWN"]
    counts: Counts
    checks: list[CheckRow]
    rules: list[RuleRow]
    findings: list[FindingRow]
    exceptions: Exceptions
    coverage: dict[str, Any]
    findings_stream: FindingsStream | None = None
    report_sha256: str


assert REPORT_SCHEMA == "scvia.integrity-report/1"


def schema_json() -> str:
    schema = IntegrityReport.model_json_schema(by_alias=True)
    schema["$id"] = "https://supercoach-via.local/schemas/integrity-report.schema.json"
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return json.dumps(schema, sort_keys=True, indent=2) + "\n"


def validate(body: dict[str, Any]) -> None:
    """Raise ``pydantic.ValidationError`` when a report breaks its contract."""
    IntegrityReport.model_validate(body)
