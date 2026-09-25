"""Data, run, release and model contracts.

This module is the shared contract between ingestion, storage, analytics, ML and
publishing. Canonical tables are described declaratively (``TABLES``) and converted
to Arrow schemas lazily so importing this module stays cheap (no pyarrow import).

Column-name changes are breaking; additive nullable columns are allowed and require a
``SCHEMA_VERSION`` bump only when readers must distinguish old snapshots.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:  # pragma: no cover
    import pyarrow as pa

SCHEMA_VERSION = 1

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:_\-.]{0,159}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def is_safe_id(value: str) -> bool:
    """True when ``value`` is an emitted-ID-safe token (no path separators, no ``..``)."""
    return bool(SAFE_ID_RE.match(value)) and ".." not in value


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class StageType(StrEnum):
    REGULAR = "regular"
    FINAL = "final"
    OTHER = "other"


class MatchStatus(StrEnum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class DateQuality(StrEnum):
    FIXTURE_VERIFIED = "fixture_verified"  # resolved against a match/fixture record
    SOURCE = "source"  # stated directly by the source row
    INFERRED = "inferred"  # reconstructed (e.g. round-to-weeks); never time-sensitive
    UNKNOWN = "unknown"


class DatePrecision(StrEnum):
    MINUTE = "minute"
    DAY = "day"
    UNKNOWN = "unknown"


class BirthDateQuality(StrEnum):
    SOURCE = "source"  # personal-details file / source page
    LEGACY_FILENAME = "legacy_filename"  # only the legacy slug DOB token
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"


class IdentityStatus(StrEnum):
    CANONICAL = "canonical"
    ALIAS = "alias"  # verified alias of another canonical identity
    QUARANTINED_DUPLICATE = "quarantined_duplicate"
    AMBIGUOUS = "ambiguous"


class Provenance(StrEnum):
    LEGACY_IMPORT = "legacy_import"
    SOURCE_FETCH = "source_fetch"
    MANUAL_FIXTURE = "manual_fixture"
    DERIVED = "derived"
    DEMO = "demo"


class DatasetStatus(StrEnum):
    LEGACY_UNVERIFIED = "legacy_unverified"
    VERIFIED = "verified"
    PARTIAL = "partial"
    DEMO = "demo"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKING = "blocking"


class Origin(StrEnum):
    PROSPECTIVE = "prospective"
    REPLAY = "replay"
    LEGACY_UNKNOWN = "legacy_unknown"


class SelectionStatus(StrEnum):
    CONFIRMED = "confirmed"  # verified announced lineup, announced before cutoff
    UNCONFIRMED = "unconfirmed"  # candidate roster inferred from prior membership


class ForecastStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    EXPIRED = "expired"


class CheckOutcome(StrEnum):
    PASS = "PASS"  # noqa: S105 - verdict label
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class ReviewVerdict(StrEnum):
    PASS = "PASS"  # noqa: S105 - verdict label
    PASS_WITH_CONCERNS = "PASS_WITH_CONCERNS"  # noqa: S105 - verdict label
    BLOCK = "BLOCK"
    UNKNOWN = "UNKNOWN"


class SourceMode(StrEnum):
    LIVE = "live"
    CACHED = "cached"  # 304 / conditional revalidation of an archived payload
    MANUAL_FIXTURE = "manual_fixture"
    LEGACY = "legacy"


class RunState(StrEnum):
    CREATED = "created"
    PLANNED = "planned"
    FETCHING = "fetching"
    PARSED = "parsed"
    VALIDATED = "validated"
    DATASET_PROMOTED = "dataset_promoted"
    ANALYZED = "analyzed"
    FORECASTED_OR_UNAVAILABLE = "forecasted_or_unavailable"
    RENDERED = "rendered"
    RELEASE_VALIDATED = "release_validated"
    READY_TO_PUBLISH = "ready_to_publish"
    PUBLISHED = "published"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


TERMINAL_RUN_STATES = frozenset({RunState.PUBLISHED, RunState.FAILED, RunState.CANCELLED})

#: Allowed forward transitions (section 5.3). Any state may move to FAILED/CANCELLED;
#: fetch/parse/validate stages may end PARTIAL.
RUN_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset({RunState.PLANNED}),
    RunState.PLANNED: frozenset({RunState.FETCHING, RunState.PARSED}),
    RunState.FETCHING: frozenset({RunState.PARSED, RunState.PARTIAL}),
    RunState.PARSED: frozenset({RunState.VALIDATED, RunState.PARTIAL}),
    RunState.VALIDATED: frozenset({RunState.DATASET_PROMOTED, RunState.PARTIAL}),
    RunState.DATASET_PROMOTED: frozenset({RunState.ANALYZED}),
    RunState.ANALYZED: frozenset({RunState.FORECASTED_OR_UNAVAILABLE}),
    RunState.FORECASTED_OR_UNAVAILABLE: frozenset({RunState.RENDERED}),
    RunState.RENDERED: frozenset({RunState.RELEASE_VALIDATED}),
    RunState.RELEASE_VALIDATED: frozenset({RunState.READY_TO_PUBLISH}),
    RunState.READY_TO_PUBLISH: frozenset({RunState.PUBLISHED}),
    RunState.PUBLISHED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.PARTIAL: frozenset(),
    RunState.CANCELLED: frozenset(),
}


def can_transition(src: RunState, dst: RunState) -> bool:
    if src in TERMINAL_RUN_STATES or src is RunState.PARTIAL:
        return False
    if dst in (RunState.FAILED, RunState.CANCELLED):
        return True
    return dst in RUN_TRANSITIONS[src]


# ---------------------------------------------------------------------------
# Canonical tables
# ---------------------------------------------------------------------------

#: Player-game statistic columns (canonical names), in legacy column order.
PLAYER_STAT_COLUMNS: tuple[str, ...] = (
    "kicks",
    "marks",
    "handballs",
    "disposals",
    "goals",
    "behinds",
    "hitouts",
    "tackles",
    "rebound_50s",
    "inside_50s",
    "clearances",
    "clangers",
    "frees_for",
    "frees_against",
    "brownlow_votes",
    "contested_possessions",
    "uncontested_possessions",
    "contested_marks",
    "marks_inside_50",
    "one_percenters",
    "bounces",
    "goal_assists",
    "time_on_ground_pct",
)

#: Legacy CSV column -> canonical column (section 4.3). Unlisted columns keep their name.
LEGACY_PLAYER_COLUMN_MAP: dict[str, str] = {
    "hit_outs": "hitouts",
    "free_kicks_for": "frees_for",
    "free_kicks_against": "frees_against",
    "goal_assist": "goal_assists",
    "percentage_of_game_played": "time_on_ground_pct",
}

ColumnType = Literal["string", "int32", "int64", "float64", "bool", "date32", "timestamp_utc", "json"]


@dataclass(frozen=True)
class Column:
    name: str
    type: ColumnType
    nullable: bool = True
    doc: str = ""


@dataclass(frozen=True)
class TableSpec:
    name: str
    key: tuple[str, ...]
    columns: tuple[Column, ...]
    partition_by: str | None = None  # e.g. "season" for large fact tables
    doc: str = ""

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def arrow_schema(self) -> pa.Schema:
        import pyarrow as pa  # local: keep module import cheap

        mapping: dict[str, Any] = {
            "string": pa.string(),
            "json": pa.string(),
            "int32": pa.int32(),
            "int64": pa.int64(),
            "float64": pa.float64(),
            "bool": pa.bool_(),
            "date32": pa.date32(),
            "timestamp_utc": pa.timestamp("us", tz="UTC"),
        }
        return pa.schema(
            [pa.field(c.name, mapping[c.type], nullable=c.nullable) for c in self.columns],
            metadata={b"table": self.name.encode(), b"schema_version": str(SCHEMA_VERSION).encode()},
        )


def _c(name: str, type_: ColumnType, nullable: bool = True, doc: str = "") -> Column:
    return Column(name, type_, nullable, doc)


_SOURCE_COLS = (
    _c("provenance", "string", False, "Provenance enum"),
    _c("source_path", "string", True, "Repo-relative legacy path or source URL"),
    _c("source_sha256", "string", True),
    _c("source_row", "int64", True, "1-based data row within the source file"),
)

TABLES: dict[str, TableSpec] = {
    "players": TableSpec(
        "players",
        key=("player_id",),
        columns=(
            _c("player_id", "string", False, "legacy:<slug> for imported identities"),
            _c("legacy_slug", "string", True),
            _c("display_name", "string", False),
            _c("first_name", "string", True),
            _c("last_name", "string", True),
            _c("birth_date", "date32", True),
            _c("birth_date_quality", "string", False, "BirthDateQuality enum"),
            _c("debut_date", "date32", True),
            _c("height_cm", "float64", True),
            _c("weight_kg", "float64", True),
            _c("identity_status", "string", False, "IdentityStatus enum"),
            _c("canonical_player_id", "string", True, "Set when identity_status=alias/duplicate"),
            _c("source_urls", "json", True, "JSON list of verified source URLs"),
            _c("details_source_path", "string", True, "personal-details file (legacy)"),
            _c("details_source_sha256", "string", True),
            *_SOURCE_COLS,
        ),
    ),
    "player_aliases": TableSpec(
        "player_aliases",
        key=("player_id", "alias"),
        columns=(
            _c("player_id", "string", False),
            _c("alias", "string", False),
            _c("alias_kind", "string", False, "source_name | legacy_slug | display_variant"),
            _c("evidence", "string", True),
            _c("source_url", "string", True, "verified source page backing the alias"),
        ),
    ),
    "clubs": TableSpec(
        "clubs",
        key=("club_id",),
        columns=(
            _c("club_id", "string", False),
            _c("name", "string", False),
            _c("lineage_id", "string", False),
            _c("first_season", "int32", True),
            _c("last_season", "int32", True),
            _c("active", "bool", False),
        ),
    ),
    "club_aliases": TableSpec(
        "club_aliases",
        key=("alias", "valid_from_season"),
        columns=(
            _c("alias", "string", False),
            _c("club_id", "string", False),
            _c("valid_from_season", "int32", False),
            _c("valid_to_season", "int32", True),
            _c("note", "string", True),
        ),
    ),
    "venues": TableSpec(
        "venues",
        key=("venue_id",),
        columns=(
            _c("venue_id", "string", False),
            _c("name", "string", False),
            _c("source_names", "json", False),
            _c("timezone", "string", True, "IANA zone when verified; null otherwise"),
        ),
    ),
    "seasons": TableSpec(
        "seasons",
        key=("season",),
        columns=(
            _c("season", "int32", False),
            _c("first_match_date", "date32", True),
            _c("last_match_date", "date32", True),
            _c("matches_complete", "int32", False),
            _c("matches_scheduled", "int32", False),
            _c("fixture_checked_at", "timestamp_utc", True),
            _c("schedule_complete", "bool", True, "null = unknown"),
            _c("source_status", "string", True, "source-declared season status if known"),
        ),
    ),
    "matches": TableSpec(
        "matches",
        key=("match_id",),
        partition_by="season",
        columns=(
            _c("match_id", "string", False),
            _c("season", "int32", False),
            _c("stage_label", "string", False, "source token, e.g. '1', 'QF', 'WF', 'GF'"),
            _c("stage_type", "string", False, "StageType enum"),
            _c("round_number", "int32", True),
            _c("stage_order", "int32", False, "stable within-season order of the stage"),
            _c("stage_id", "string", False, "safe token, e.g. r01, qf, gf"),
            _c("replay_occurrence", "int32", False, "0 = original, 1 = first replay"),
            _c("home_club_id", "string", False, "legacy team_1"),
            _c("away_club_id", "string", False, "legacy team_2"),
            _c("home_source_name", "string", False),
            _c("away_source_name", "string", False),
            _c("venue_id", "string", True),
            _c("venue_source_name", "string", True),
            _c("local_start", "string", True, "local wall time 'YYYY-MM-DD HH:MM' as sourced"),
            _c("match_date", "date32", True, "local calendar date"),
            _c("date_precision", "string", False, "DatePrecision enum"),
            _c("status", "string", False, "MatchStatus enum"),
            _c("attendance", "int64", True),
            *(
                _c(f"{side}_{q}_{k}", "int32", True)
                for side in ("home", "away")
                for q in ("q1", "q2", "q3", "final")
                for k in ("goals", "behinds")
            ),
            _c("home_score", "int32", True),
            _c("away_score", "int32", True),
            *_SOURCE_COLS,
        ),
    ),
    "player_games": TableSpec(
        "player_games",
        key=("match_id", "player_id", "club_id"),
        partition_by="season",
        columns=(
            _c("match_id", "string", False),
            _c("player_id", "string", False),
            _c("club_id", "string", False),
            _c("season", "int32", False),
            _c("opponent_club_id", "string", True),
            _c("stage_label", "string", False),
            _c("stage_id", "string", False),
            _c("club_source_name", "string", False),
            _c("opponent_source_name", "string", True),
            _c("link_method", "string", False, "key | date_tiebreak | row_order"),
            _c("match_date", "date32", True, "row date as sourced; see date_quality"),
            _c("date_quality", "string", False, "DateQuality enum"),
            _c("career_game_counter", "int32", True),
            _c("career_game_counter_token", "string", True, "original token incl. arrows"),
            _c("result", "string", True, "W/L/D as sourced"),
            _c("jersey_number", "int32", True),
            *(_c(s, "float64" if s == "time_on_ground_pct" else "int32", True) for s in PLAYER_STAT_COLUMNS),
            _c("available_at", "timestamp_utc", True, "null = unknown (legacy)"),
            _c("revision_id", "string", False),
            *_SOURCE_COLS,
        ),
    ),
    "lineups": TableSpec(
        "lineups",
        key=("match_id", "club_id", "player_id"),
        partition_by="season",
        columns=(
            _c("match_id", "string", False),
            _c("club_id", "string", False),
            _c("player_id", "string", False),
            _c("season", "int32", False),
            _c("role", "string", False, "named | played | unknown"),
            _c("announced_at", "timestamp_utc", True),
            _c("confidence", "string", False),
            _c("name_token", "string", False, "lineup name token as sourced"),
            _c("resolution", "string", False, "match_participation | club_season | alias"),
            *_SOURCE_COLS,
        ),
    ),
    "draft_events": TableSpec(
        "draft_events",
        key=("draft_event_id",),
        columns=(
            _c("draft_event_id", "string", False),
            _c("season", "int32", False),
            _c("event_type", "string", False, "national | rookie_<type> | ..."),
            _c("draft_round", "int32", True),
            _c("pick", "int32", True),
            _c("club_id", "string", True),
            _c("club_source_name", "string", True),
            _c("player_name", "string", False),
            _c("player_id", "string", True, "null when unresolved"),
            _c("recruited_from", "string", True),
            _c("grade", "string", True),
            _c("source_family", "string", False, "legacy file family the event came from"),
            *_SOURCE_COLS,
        ),
    ),
    "contract_observations": TableSpec(
        "contract_observations",
        key=("observation_id",),
        columns=(
            _c("observation_id", "string", False),
            _c("player_name", "string", False),
            _c("player_id", "string", True),
            _c("club_id", "string", True),
            _c("contract_end", "int32", True),
            _c("fa_category", "string", True),
            _c("observed_at", "date32", True),
            _c("source_type", "string", False, "live | manual_fixture | legacy"),
            _c("source_name", "string", True, "attribution parsed from notes, e.g. AFL.com.au"),
            _c("position", "string", True),
            _c("notes", "string", True),
            _c("confidence", "string", False),
            *_SOURCE_COLS,
        ),
    ),
    "school_observations": TableSpec(
        "school_observations",
        key=("observation_id",),
        columns=(
            _c("observation_id", "string", False),
            _c("draft_year", "int32", True),
            _c("pick", "int32", True),
            _c("player_name", "string", False),
            _c("player_id", "string", True),
            _c("source_wording", "string", True),
            _c("school", "string", True),
            _c("school_type", "string", True),
            _c("classifier_version", "string", False),
            _c("confidence", "string", False),
            *_SOURCE_COLS,
        ),
    ),
    "live_snapshots": TableSpec(
        "live_snapshots",
        key=("source_game_id", "payload_hash"),
        columns=(
            _c("source_game_id", "string", False),
            _c("payload_hash", "string", False),
            _c("match_id", "string", True),
            _c("fetched_at", "timestamp_utc", True),
            _c("status", "string", True),
            _c("quarter", "string", True),
            _c("home_score", "int32", True),
            _c("away_score", "int32", True),
            _c("schema_version", "int32", False),
            _c("anomalies", "json", True),
            _c("payload_kind", "string", False, "state_json | players_csv"),
            _c("home_source_name", "string", True),
            _c("away_source_name", "string", True),
            _c("source_round_label", "string", True),
            *_SOURCE_COLS,
        ),
    ),
    "source_observations": TableSpec(
        "source_observations",
        key=("source_ref",),
        columns=(
            _c("source_ref", "string", False),
            _c("adapter", "string", False),
            _c("adapter_version", "string", False),
            _c("url", "string", True),
            _c("fetched_at", "timestamp_utc", True),
            _c("content_sha256", "string", True),
            _c("http_status", "int32", True),
            _c("etag", "string", True),
            _c("last_modified", "string", True),
            _c("bytes", "int64", True),
            _c("source_mode", "string", False),
            _c("outcome", "string", False, "CheckOutcome enum"),
        ),
    ),
    "quality_issues": TableSpec(
        "quality_issues",
        key=("issue_id",),
        columns=(
            _c("issue_id", "string", False),
            _c("severity", "string", False),
            _c("status", "string", False, "open | accepted | resolved"),
            _c("table_name", "string", True),
            _c("row_key", "string", True),
            _c("source_path", "string", True),
            _c("rule_id", "string", False),
            _c("explanation", "string", False),
            _c("remediation", "string", True),
            _c("acceptance_basis", "string", True),
            _c("season", "int32", True, "season of the affected row when known"),
        ),
    ),
    "quarantine": TableSpec(
        "quarantine",
        key=("quarantine_id",),
        columns=(
            _c("quarantine_id", "string", False),
            _c("table_name", "string", False),
            _c("reason", "string", False),
            _c("candidates", "json", True),
            _c("raw", "json", False, "original row cells exactly as read"),
            _c("season", "int32", True),
            *_SOURCE_COLS,
        ),
    ),
    "source_files": TableSpec(
        "source_files",
        key=("path",),
        doc="Every legacy input file and its import disposition (import report as a table).",
        columns=(
            _c("path", "string", False, "repo-relative path"),
            _c("family", "string", False),
            _c("bytes", "int64", False),
            _c("sha256", "string", False),
            _c("disposition", "string", False, "imported | ignored | quarantined"),
            _c("reason", "string", True),
            _c("rows_read", "int64", True),
            _c("rows_imported", "int64", True),
            _c("rows_quarantined", "int64", True),
        ),
    ),
    "legacy_predictions": TableSpec(
        "legacy_predictions",
        key=("prediction_row_id",),
        doc="Archived legacy forecast files. Labels are claims from filenames; never prospective.",
        columns=(
            _c("prediction_row_id", "string", False),
            _c("artifact_kind", "string", False, "next_round | next_game | prediction_vs_actual"),
            _c("claimed_round_label", "string", True, "round label claimed by the filename"),
            _c("claimed_season", "int32", True, "season claimed by the filename"),
            _c("claimed_timestamp", "string", True, "YYYYMMDD_HHMM from the filename"),
            _c("player_display_name", "string", False, "as written (legacy 'Surname First')"),
            _c("team_source_name", "string", True),
            _c("club_id", "string", True),
            _c("player_id", "string", True, "only when an unambiguous identity mapping exists"),
            _c("predicted_value", "float64", True),
            _c("claimed_actual", "float64", True, "prediction_vs_actual files only"),
            _c("origin", "string", False, "Origin enum; always legacy_unknown"),
            *_SOURCE_COLS,
        ),
    ),
    "legacy_rank_scores": TableSpec(
        "legacy_rank_scores",
        key=("player_slug",),
        doc="data/top100/all_time_top_100.csv numeric ranking export (player,all_time_score).",
        columns=(
            _c("player_slug", "string", False),
            _c("player_id", "string", True),
            _c("rank", "int32", False, "1-based file order"),
            _c("all_time_score", "float64", True),
            *_SOURCE_COLS,
        ),
    ),
    "legacy_rank_yearly": TableSpec(
        "legacy_rank_yearly",
        key=("season", "player_slug"),
        doc="data/top100/yearly/year_*.csv annual ranking exports.",
        columns=(
            _c("season", "int32", False),
            _c("player_slug", "string", False),
            _c("player_id", "string", True),
            _c("rank", "int32", False, "1-based file order"),
            _c("score", "float64", True),
            _c("percentile_rank", "float64", True),
            _c("games_played", "int32", True),
            *_SOURCE_COLS,
        ),
    ),
    "legacy_top100_bios": TableSpec(
        "legacy_top100_bios",
        key=("serial_number",),
        doc="Root all_time_top_100.csv biography/presentation export; not a numeric ranking.",
        columns=(
            _c("serial_number", "int32", False),
            _c("player_name", "string", False),
            _c("footy_teams", "string", True),
            _c("comment", "string", True),
            *_SOURCE_COLS,
        ),
    ),
}


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FragmentRef(_Strict):
    path: str  # relative to the data root's fragment store
    sha256: str
    rows: int
    bytes: int
    partition: str | None = None

    @field_validator("sha256")
    @classmethod
    def _sha(cls, v: str) -> str:
        if not SHA256_RE.match(v):
            raise ValueError("invalid sha256")
        return v

    @field_validator("path")
    @classmethod
    def _path(cls, v: str) -> str:
        if v.startswith("/") or ".." in v.split("/") or "\\" in v:
            raise ValueError("fragment path must be relative and contained")
        return v


class TableEntry(_Strict):
    schema_version: int = SCHEMA_VERSION
    row_count: int
    fragments: tuple[FragmentRef, ...]


class SnapshotManifest(_Strict):
    schema_version: int = SCHEMA_VERSION
    snapshot_id: str  # "sha256:<hex>" over the canonical manifest content
    created_at: datetime
    parent: str | None = None
    status: DatasetStatus
    run_id: str | None = None
    code_version: str
    tables: dict[str, TableEntry]
    source_revisions: dict[str, str] = Field(default_factory=dict)
    quality: dict[str, int] = Field(default_factory=dict)  # severity -> count
    notes: tuple[str, ...] = ()


class CurrentPointer(_Strict):
    schema_version: int = SCHEMA_VERSION
    snapshot_id: str
    manifest_path: str
    promoted_at: datetime
    run_id: str | None = None


class StepRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    state: Literal["pending", "running", "succeeded", "failed", "skipped", "reused"]
    input_hashes: dict[str, str] = Field(default_factory=dict)
    code_version: str = ""
    outputs: dict[str, str] = Field(default_factory=dict)
    attempts: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error_code: str | None = None
    message: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = SCHEMA_VERSION
    run_id: str
    command: str
    state: RunState
    created_at: datetime
    updated_at: datetime
    code_version: str
    history: list[tuple[RunState, datetime]] = Field(default_factory=list)
    steps: dict[str, StepRecord] = Field(default_factory=dict)
    snapshot_id: str | None = None
    release_id: str | None = None
    model_id: str | None = None
    error_code: str | None = None
    recovery: str | None = None


@dataclass
class ValidationReport:
    """Outcome of dataset or release validation."""

    outcome: CheckOutcome
    checks: dict[str, CheckOutcome] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome is CheckOutcome.PASS
