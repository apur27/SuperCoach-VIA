"""Strict, versioned contracts for the reconciliation audit (DESIGN sections 5, 6, 9).

Everything persisted by the audit is one of these models serialised canonically (sorted
keys, UTF-8, trailing newline). Models forbid unknown keys and are frozen, so a file with
an extra or renamed field is rejected rather than silently reinterpreted.

The statistic mapping here is written independently of ``ingest.afltables`` and of
``integrity.sourcepages``; a unit test asserts all three agree on the canonical names.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from supercoach_via.reconciliation import SCHEMA_VERSION

# ---------------------------------------------------------------------------
# Statistic contract (DESIGN section 2)
# ---------------------------------------------------------------------------

#: (source label, canonical field) in the order the canonical table stores them. ``GM`` is the
#: appearance count and is derived separately; ``SU`` and the arrows are participation metadata.
STAT_LABELS: tuple[tuple[str, str], ...] = (
    ("KI", "kicks"),
    ("MK", "marks"),
    ("HB", "handballs"),
    ("DI", "disposals"),
    ("GL", "goals"),
    ("BH", "behinds"),
    ("HO", "hitouts"),
    ("TK", "tackles"),
    ("RB", "rebound_50s"),
    ("IF", "inside_50s"),
    ("CL", "clearances"),
    ("CG", "clangers"),
    ("FF", "frees_for"),
    ("FA", "frees_against"),
    ("BR", "brownlow_votes"),
    ("CP", "contested_possessions"),
    ("UP", "uncontested_possessions"),
    ("CM", "contested_marks"),
    ("MI", "marks_inside_50"),
    ("1%", "one_percenters"),
    ("BO", "bounces"),
    ("GA", "goal_assists"),
    ("%P", "time_on_ground_pct"),
)
STAT_FIELDS: tuple[str, ...] = tuple(f for _, f in STAT_LABELS)
LABEL_TO_FIELD: dict[str, str] = dict(STAT_LABELS)
FIELD_TO_LABEL: dict[str, str] = {f: lab for lab, f in STAT_LABELS}
FIELD_INDEX: dict[str, int] = {f: i for i, f in enumerate(STAT_FIELDS)}
#: statistics that are integer counts; ``time_on_ground_pct`` is a printed decimal/percentage
COUNT_FIELDS: tuple[str, ...] = tuple(f for f in STAT_FIELDS if f != "time_on_ground_pct")
#: source labels of the season-summary tables (no ``%P`` column) in table order
SUMMARY_LABELS: tuple[str, ...] = tuple(lab for lab, _ in STAT_LABELS if lab != "%P")

EXCLUSIONS: tuple[str, ...] = (
    "preseason and practice matches",
    "reserves and second-tier competitions",
    "representative football",
    "AFLW and other women's competitions",
    "coaches and umpires",
    "mutable height/weight, fantasy scores, predictions and locally invented proxy measures",
)
LEAGUE = "men's senior VFL/AFL premiership"
FIRST_SEASON = 1897


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


# ---------------------------------------------------------------------------
# Plan (DESIGN section 5)
# ---------------------------------------------------------------------------


class CodeIdentity(Model):
    repo_commit: str | None
    #: package-relative module path -> sha256 of the capture-relevant files (of the code that produced the archive
    #: when the plan is bound to an existing capture with ``--capture-plan``)
    capture_files: dict[str, str]
    #: appended (A11): the capture code current at plan time, recorded when it differs from ``capture_files``
    capture_files_current: dict[str, str] = {}


class ScopeSpec(Model):
    population: Literal["all", "sample", "seasons"]
    league: str
    first_season: int
    through_date: str
    full_population: bool
    exclusions: list[str]
    #: only for ``population == "sample"``: the profile URLs audited
    sample_profiles: list[str] = []
    #: appended: only for ``population == "seasons"``: the seasons audited (e.g. the seasons a weekly scrape changed)
    seasons: list[int] = []


class SnapshotPin(Model):
    snapshot_id: str
    selector: str
    manifest_sha256: str
    pointer_sha256: str | None
    rows: int
    #: table -> partition -> fragment sha256
    fragments: dict[str, dict[str, str]]


class LegacyPin(Model):
    player_files: int
    personal_files: int
    match_files: int
    bytes: int
    membership_sha256: str
    content_sha256: str
    #: ``data/awards/*.csv`` (season-level award values); appended after launch, 0 for older plans
    award_files: int = 0


class InputsSpec(Model):
    snapshot: SnapshotPin
    legacy: LegacyPin | None


class PolicyPins(Model):
    source_policy_sha256: str
    rules_sha256: str
    identity_overrides_sha256: str
    label_map_sha256: str


class SourceSpec(Model):
    host: str
    reference_mode: Literal["observed_current"]
    seed_urls: list[str]


class Operational(Model):
    """Absolute paths and environment: operational metadata, excluded from every identity."""

    run_dir: str
    data_root: str
    legacy_root: str | None
    created_utc: str
    python: str
    platform: str


class Plan(Model):
    kind: Literal["afltables-reconciliation-plan"] = "afltables-reconciliation-plan"
    schema_version: int = SCHEMA_VERSION
    plan_id: str
    #: identity of everything that determines WHICH pages are fetched and how; a capture can be
    #: resumed (and a plan re-issued with different comparison rules) only while this is unchanged
    capture_identity: str
    code: CodeIdentity
    scope: ScopeSpec
    statistics: dict[str, str]
    inputs: InputsSpec
    policies: PolicyPins
    source: SourceSpec
    outputs: dict[str, str]
    operational: Operational


def plan_identity_payload(plan: Plan | dict[str, Any]) -> dict[str, Any]:
    body = plan.model_dump(mode="json") if isinstance(plan, Plan) else dict(plan)
    for key in ("plan_id", "operational"):
        body.pop(key, None)
    return body


def capture_identity_payload(plan: Plan | dict[str, Any]) -> dict[str, Any]:
    body = plan.model_dump(mode="json") if isinstance(plan, Plan) else dict(plan)
    return {
        "schema_version": body["schema_version"],
        "capture_files": body["code"]["capture_files"],
        "scope": body["scope"],
        "source": body["source"],
        "source_policy_sha256": body["policies"]["source_policy_sha256"],
    }


# ---------------------------------------------------------------------------
# Capture manifest (DESIGN section 6)
# ---------------------------------------------------------------------------

ResourceKind = Literal["robots", "stats_index", "notes", "letter", "season", "match", "profile"]
ResourceStatus = Literal["usable", "absent", "missing", "failed", "blocked", "rejected", "pending", "out_of_scope"]


class Resource(Model):
    task_id: str
    kind: ResourceKind
    url: str
    final_url: str | None
    status: ResourceStatus
    http_status: int | None
    sha256: str | None
    bytes: int | None
    attempts: int
    #: line number (1-based) in ``capture/observations.jsonl`` of the observation that produced the payload
    observation: int | None
    reason: str | None
    #: "supported" when ``source.py`` has a reader for this kind
    parser_status: Literal["supported", "unsupported", "unparsed"]
    discovered_from: list[str]
    discovered_from_count: int
    revision: int = 1


class ManifestCensus(Model):
    letters_expected: int
    letters_usable: int
    letters_failed: list[str]
    profiles_in_directory: int
    profile_urls_rejected: list[str]
    #: lineup-linked profiles absent from the directory census (S-06)
    lineup_profiles_not_in_directory: list[str]
    revalidation_changed: list[str]


class Manifest(Model):
    kind: Literal["afltables-reconciliation-capture-manifest"] = "afltables-reconciliation-capture-manifest"
    schema_version: int = SCHEMA_VERSION
    plan_id: str
    capture_identity: str
    reference_mode: Literal["observed_current"]
    acquisition_started_utc: str | None
    acquisition_finished_utc: str | None
    execution_complete: bool
    capture_complete: bool
    incomplete_reasons: list[str]
    seasons: list[int]
    scope_match_count: int
    out_of_scope_match_count: int
    undated_in_scope_unknown: int
    census: ManifestCensus
    resource_counts: dict[str, dict[str, int]]
    resources: list[Resource]


# ---------------------------------------------------------------------------
# Findings and verdicts (DESIGN sections 8-9)
# ---------------------------------------------------------------------------


class Layer(StrEnum):
    SNAPSHOT = "snapshot"
    LEGACY_CSV = "legacy_csv"
    SOURCE = "source"


class Severity(StrEnum):
    FAIL = "fail"  # a confirmed discrepancy in a requested layer
    UNKNOWN = "unknown"  # evidence is missing, conflicting or unresolved
    INFO = "info"  # reported, never changes a layer verdict


class Verdict(StrEnum):
    PASS = "PASS"  # noqa: S105 - a verdict label, not a credential
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


EXIT_CODES = {"PASS": 0, "invalid": 2, "FAIL": 4, "busy": 5, "UNKNOWN": 8, "software": 9}


def canonical_dump(model: BaseModel) -> bytes:
    """Canonical bytes of a model: sorted keys, compact separators, UTF-8, trailing newline."""
    from supercoach_via.integrity.report import canonical_bytes

    return canonical_bytes(model.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Completion and implementation-validation records (DESIGN section 9, 14)
# ---------------------------------------------------------------------------

StepStatus = Literal["PASS", "FAIL", "BASELINE_FAIL", "UNKNOWN", "COMPLETE", "INCOMPLETE", "PENDING", "NOT_RUN"]


class ModelUse(Model):
    requested: str
    resolved: str | None
    resolution_source: str


class LayerOutcome(Model):
    status: StepStatus
    report_sha256: str | None
    note: str


class ReportRef(Model):
    path: str
    report_sha256: str
    findings_sha256: str
    workers: int
    cache: str
    wall_seconds: float | None
    peak_rss_mib: float | None


class Completion(Model):
    """Separate design, implementation, execution and per-layer data statuses (they never collapse into one)."""

    kind: Literal["afltables-reconciliation-completion"] = "afltables-reconciliation-completion"
    schema_version: int = SCHEMA_VERSION
    created_utc: str
    model: ModelUse
    design: LayerOutcome
    implementation: LayerOutcome
    execution: LayerOutcome
    data: dict[str, LayerOutcome]
    overall_data_verdict: Verdict
    inputs: dict[str, str]
    reports: dict[str, ReportRef]
    reproducibility: dict[str, str]
    acceptance: LayerOutcome
    open_items: list[str]


class TestCounts(Model):
    passed: int
    failed: int
    errors: int
    skipped: int
    deselected: int


class ValidationStep(Model):
    """One validation step. Its status is checked against structured counts, never against free text (B5)."""

    name: str
    command: str
    status: StepStatus
    detail: str
    #: a test step's pytest outcome counts (None for a non-test step such as ruff or mypy)
    counts: TestCounts | None = None
    #: every skipped test, grouped by its reason; must account for ``counts.skipped`` exactly
    skip_reasons: dict[str, int] = {}
    #: for BASELINE_FAIL: the same failure at the base commit in the same environment
    baseline_evidence: str | None = None

    @model_validator(mode="after")
    def _status_matches_counts(self) -> ValidationStep:
        c = self.counts
        if c is not None:
            if self.status == "PASS" and (c.failed or c.errors):
                raise ValueError(f"status PASS with {c.failed} failed and {c.errors} errors")
            if sum(self.skip_reasons.values()) != c.skipped:
                raise ValueError(f"skip_reasons account for {sum(self.skip_reasons.values())} of {c.skipped} skips")
        if self.status == "BASELINE_FAIL" and not (self.baseline_evidence or "").strip():
            raise ValueError("a BASELINE_FAIL step needs baseline_evidence: the same failure at the base commit")
        return self


class ImplementationValidation(Model):
    kind: Literal["afltables-reconciliation-implementation-validation"] = (
        "afltables-reconciliation-implementation-validation"
    )
    schema_version: int = SCHEMA_VERSION
    created_utc: str
    branch: str
    base_commit: str
    libraries: dict[str, str]
    steps: list[ValidationStep]
    baseline_failures: list[str]
    new_failures: list[str]


def implementation_status(v: ImplementationValidation) -> Literal["COMPLETE", "INCOMPLETE"]:
    """COMPLETE only when every step passed (a BASELINE_FAIL is a pre-existing, evidenced failure) and no new
    failure is recorded; the completion writer must take this status, never type one in (B5)."""
    ok = all(st.status in ("PASS", "BASELINE_FAIL") for st in v.steps) and not v.new_failures
    return "COMPLETE" if ok else "INCOMPLETE"


class Finding(Model):
    id: str
    category: str
    severity: Literal["fail", "unknown", "info"]
    layer: str
    rule_id: str
    player: dict[str, Any]
    season: int | None
    match: dict[str, Any]
    field: str | None
    expected: Any
    actual: Any
    expected_raw: Any
    actual_raw: Any
    evidence: dict[str, Any]
    local: dict[str, Any]
    detail: str
    suggestion: str


class Report(Model):
    kind: Literal["afltables-reconciliation-report"]
    schema_version: int
    plan_id: str
    capture_identity: str
    capture_manifest_sha256: str
    snapshot_id: str
    legacy_inputs: dict[str, Any] | None
    scope: dict[str, Any]
    policies: dict[str, Any]
    code: dict[str, Any]
    result: dict[str, Any]
    layers: dict[str, Any]
    source: dict[str, Any]
    latest_completed_final: dict[str, Any] | None
    findings: dict[str, Any]
    input_drift: list[str]
    unit_digests: dict[str, str]
    exclusions: list[str]
    limitations: list[str]


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "plan": Plan,
    "capture-manifest": Manifest,
    "finding": Finding,
    "report": Report,
    "completion": Completion,
    "implementation-validation": ImplementationValidation,
}


def export_schemas(directory: Any) -> list[str]:
    """Write one JSON Schema per contract (deterministic bytes); returns the file names."""
    from pathlib import Path

    from supercoach_via.integrity.report import canonical_bytes

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for key, model in sorted(SCHEMA_MODELS.items()):
        doc = model.model_json_schema()
        doc["$id"] = f"https://supercoach-via.invalid/schemas/reconciliation/{key}.schema.json"
        (out / f"{key}.schema.json").write_bytes(canonical_bytes(doc))
        names.append(f"{key}.schema.json")
    return names
