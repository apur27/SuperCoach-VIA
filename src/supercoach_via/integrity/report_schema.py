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
    comparators: dict[str, Any] | None = Field(
        default=None, description="evaluation, live-capture and content inputs the release was compared with"
    )
    digest: str


class Scope(_S):
    name: Literal["full", "data"]
    as_of: str | None
    families: list[str]
    restricted: bool
    complete: bool
    semantic_complete: bool | None = Field(
        default=None,
        description="every published resource had an executed semantic comparison or is provenance-only by "
        "type (release.coverage PASS); null when no release is audited",
    )
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


class TypeCoverage(_S):
    resources: int
    compared: int = Field(description="files an executed semantic comparison examined")
    provenance_only: int = Field(description="files verified for provenance/bytes only (prose, rendered images)")


class Authority(_S):
    input: str = Field(description="the authoritative input the type is derived from")
    check: str = Field(description="the check that compares it ('provenance' = provenance only)")


class SemanticCoverage(_S):
    by_type: dict[str, TypeCoverage]
    uncompared: dict[str, int] = Field(
        description="type -> files with no executed semantic comparison; non-empty makes the audit UNKNOWN"
    )
    unaudited: list[str] = Field(description="what is deliberately not audited, stated")
    authority: dict[str, Authority]


class Coverage(BaseModel):
    """Per-family counters (open) plus the typed semantic-coverage summary of the release."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    semantic: SemanticCoverage | None = None


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
    coverage: Coverage
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
