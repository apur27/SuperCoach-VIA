"""Public, release-scoped view models.

One typed object per public resource. The browser JSON, Markdown reports, CSV exports
and chart labels are all rendered from these objects so they cannot disagree.

JSON Schemas are generated from these models (``scvia schemas``) into ``schemas/`` and
TypeScript types are generated from those schemas in ``web/``. Do not hand-maintain a
second contract.

Conventions:
- Every numeric statistic that may be unobserved is ``float | None`` / ``int | None``;
  ``None`` means unknown/not recorded and must never be rendered as zero.
- Aggregates carry ``observed_games`` (denominator) and ``coverage`` (0..1 or None).
- Timestamps are UTC ISO-8601 (``*_at``); local calendar dates are ``date`` strings.
- IDs are safe tokens (see ``domain.schemas.is_safe_id``).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from supercoach_via.domain.schemas import SHA256_RE, is_safe_id

PUBLIC_SCHEMA_VERSION = 1


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


# ---------------------------------------------------------------------------
# Shared fragments
# ---------------------------------------------------------------------------


class ResourceRef(PublicModel):
    path: str
    sha256: str
    bytes: int

    @field_validator("path")
    @classmethod
    def _contained(cls, v: str) -> str:
        if not v or v.startswith("/") or "\\" in v or ".." in v.split("/") or "://" in v or v.startswith("//"):
            raise ValueError("resource path must be relative, same-origin and contained")
        return v

    @field_validator("sha256")
    @classmethod
    def _sha(cls, v: str) -> str:
        if not SHA256_RE.match(v):
            raise ValueError("invalid sha256")
        return v


class Freshness(PublicModel):
    source_checked_at: datetime | None
    latest_completed_match_at: datetime | None = Field(
        description="UTC instant if the local start time and venue zone are known"
    )
    latest_completed_match_date: date | None
    coverage_through: str | None = Field(description="source-derived label, e.g. '2026 Grand Final'")
    generated_at: datetime
    published_at: datetime | None
    validation_state: Literal["PASS", "FAIL", "UNKNOWN"]
    dataset_status: Literal["legacy_unverified", "verified", "partial", "demo"]
    season_active: bool | None = Field(description="null when the source does not declare it")
    stale: bool
    stale_reason: str | None = None


class StatValue(PublicModel):
    """An aggregate that discloses its denominator and coverage."""

    stat: str
    total: float | None
    mean: float | None
    observed_games: int
    eligible_games: int
    coverage: float | None = Field(ge=0, le=1)


class StatColumns(PublicModel):
    """Positional ``StatValue`` data aligned to a parent's ``stat_names`` (compact player pages).

    ``mean`` and ``coverage`` are not stored: they are exactly ``total / observed_games`` and
    ``min(1, observed_games / scope games)`` (``expand_stats``; web ``expandStats``).
    """

    total: list[float | None]
    observed_games: list[int]
    eligible_games: list[int]

    @model_validator(mode="after")
    def _aligned(self) -> StatColumns:
        if not len(self.total) == len(self.observed_games) == len(self.eligible_games):
            raise ValueError("StatColumns arrays differ in length")
        return self


def _derived_mean(total: float | None, observed: int) -> float | None:
    return None if total is None or observed == 0 else total / observed


def _derived_coverage(observed: int, scope_games: int) -> float | None:
    return None if scope_games <= 0 else min(1.0, observed / scope_games)


def to_stat_columns(names: list[str], values: list[StatValue], scope_games: int) -> StatColumns:
    """Pack ``values`` (in ``names`` order); refuses a mean or coverage that is not the derived one."""
    if [v.stat for v in values] != names:
        raise ValueError("stat values are not in stat_names order")
    for v in values:
        if v.mean != _derived_mean(v.total, v.observed_games):
            raise ValueError(f"{v.stat}: mean is not derived from total/observed_games")
        if v.coverage != _derived_coverage(v.observed_games, scope_games):
            raise ValueError(f"{v.stat}: coverage is not derived from observed_games/scope games")
    return StatColumns(
        total=[v.total for v in values],
        observed_games=[v.observed_games for v in values],
        eligible_games=[v.eligible_games for v in values],
    )


def expand_stats(names: list[str], cols: StatColumns, scope_games: int) -> list[StatValue]:
    """Inverse of ``to_stat_columns`` for a scope of ``scope_games`` games (season or career)."""
    if len(names) != len(cols.total):
        raise ValueError("stat_names and StatColumns differ in length")
    return [
        StatValue(
            stat=n,
            total=t,
            mean=_derived_mean(t, o),
            observed_games=o,
            eligible_games=e,
            coverage=_derived_coverage(o, scope_games),
        )
        for n, t, o, e in zip(names, cols.total, cols.observed_games, cols.eligible_games, strict=True)
    ]


class ClubRef(PublicModel):
    club_id: str
    name: str


class Source(PublicModel):
    label: str
    url: str | None = None
    note: str | None = None


# ---------------------------------------------------------------------------
# Release manifest
# ---------------------------------------------------------------------------


class CoverageInfo(PublicModel):
    status: Literal["legacy_unverified", "verified", "partial", "demo"]
    through: str | None


class ForecastInfo(PublicModel):
    status: Literal["available", "unavailable", "expired"]
    reason: str | None
    artifact: str | None = Field(description="resource key of the current prediction set")
    model_id: str | None = None


class ReleaseManifest(PublicModel):
    schema_version: int = PUBLIC_SCHEMA_VERSION
    release_id: str
    snapshot_id: str
    generated_at: datetime
    season: int
    demo: bool
    base_label: str = Field(description="'DEMO' for demo releases, else ''")
    coverage: CoverageInfo
    forecast: ForecastInfo
    resources: dict[str, ResourceRef]

    @field_validator("release_id")
    @classmethod
    def _rid(cls, v: str) -> str:
        if not is_safe_id(v):
            raise ValueError("unsafe release id")
        return v


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------


class TeamScore(PublicModel):
    club_id: str
    name: str
    goals: int | None
    behinds: int | None
    score: int | None


class MatchSummary(PublicModel):
    match_id: str
    season: int
    stage_id: str
    stage_label: str
    stage_type: Literal["regular", "final", "other"]
    round_number: int | None
    stage_order: int
    replay_occurrence: int
    local_start: str | None = Field(description="local wall time as sourced, 'YYYY-MM-DD HH:MM'")
    match_date: date | None
    date_precision: Literal["minute", "day", "unknown"]
    status: Literal["scheduled", "in_progress", "complete", "postponed", "cancelled", "unknown"]
    venue: str | None
    home: TeamScore
    away: TeamScore
    winner_club_id: str | None = Field(description="null for draws, incomplete or unknown")


class MatchIndex(PublicModel):
    season: int
    matches: list[MatchSummary]


class QuarterScore(PublicModel):
    quarter: Literal["q1", "q2", "q3", "final"]
    home_goals: int | None
    home_behinds: int | None
    away_goals: int | None
    away_behinds: int | None


class BoxScoreRow(PublicModel):
    player_id: str
    name: str
    stats: list[float | None] = Field(
        description="positional values aligned to the parent's stat_columns; null = not recorded"
    )


class MatchDetail(PublicModel):
    summary: MatchSummary
    quarters: list[QuarterScore]
    attendance: int | None
    home_players: list[BoxScoreRow]
    away_players: list[BoxScoreRow]
    stat_columns: list[str] = Field(description="stats observed at least once in this file, in canonical order")
    live_snapshots: list[str] = Field(default_factory=list, description="live resource keys")
    sources: list[Source]


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


class PredictionRow(PublicModel):
    prediction_id: str
    prediction_run_id: str
    snapshot_id: str
    model_id: str
    player_id: str
    player_name: str
    club_id: str
    club_name: str
    opponent_club_id: str | None
    opponent_name: str | None
    match_id: str
    season: int
    stage_id: str
    stage_label: str
    scheduled_at: datetime | None
    scheduled_local: str | None
    venue: str | None
    forecast_cutoff: datetime
    origin: Literal["prospective", "replay", "legacy_unknown"]
    generated_at: datetime
    selection_status: Literal["confirmed", "unconfirmed"]
    eligibility_basis: str
    history_games: int
    recent_mean_5: float | None
    predicted_disposals: float
    interval_low: float | None
    interval_high: float | None
    interval_level: float | None
    interval_method: str | None
    warnings: list[str]


class OmissionSummary(PublicModel):
    reason: Literal["unresolved_identity", "insufficient_history", "not_in_fixture", "retired", "other"]
    count: int
    detail: str | None = None


class ModelInfo(PublicModel):
    model_id: str
    kind: Literal["baseline", "model"]
    name: str
    description: str
    trained_cutoff: datetime | None
    promoted: bool
    promotion_note: str


class IntervalInfo(PublicModel):
    available: bool
    level: float | None
    method: str | None
    calibrated: bool
    reason: str | None


class PredictionSet(PublicModel):
    season: int
    stage_id: str
    stage_label: str
    status: Literal["available", "unavailable", "expired"]
    reason: str | None
    units: Literal["disposals"] = "disposals"
    generated_at: datetime
    forecast_cutoff: datetime | None
    target_matches: list[MatchSummary]
    model: ModelInfo | None
    interval: IntervalInfo
    rows: list[PredictionRow]
    omissions: list[OmissionSummary]


class PredictionIndexEntry(PublicModel):
    season: int
    stage_id: str
    stage_label: str
    status: Literal["available", "unavailable", "expired"]
    resource: str
    rows: int


class PredictionIndex(PublicModel):
    current: str | None = Field(description="resource key of the current set, if any")
    status: Literal["available", "unavailable", "expired"]
    reason: str | None
    sets: list[PredictionIndexEntry]


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------


class PlayerIndexEntry(PublicModel):
    id: str
    key: str = Field(description="encoded id used in resource paths")
    name: str
    clubs: list[str]
    first_season: int | None
    last_season: int | None
    games: int
    active: bool
    search: str = Field(description="lower-case diacritic-stripped search terms")


class PlayerIndex(PublicModel):
    count: int
    players: list[PlayerIndexEntry]


class SeasonLine(PublicModel):
    season: int
    clubs: list[str]
    games: int
    stats: list[StatValue]
    games_resource: str = Field(description="resource key of this season's game log")


class PlayerSeason(PublicModel):
    """Compact public ``SeasonLine``: stats are positional (``PlayerDetail.stat_names``)."""

    season: int
    clubs: list[str]
    games: int = Field(description="coverage scope for this season's stats")
    stats: StatColumns
    games_resource: str = Field(description="resource key of this season's game log")


class PlayerDetail(PublicModel):
    id: str
    key: str
    name: str
    first_name: str | None
    last_name: str | None
    birth_date: date | None
    birth_date_quality: Literal["source", "legacy_filename", "conflicting", "unknown"]
    debut_date: date | None
    height_cm: float | None
    weight_kg: float | None
    identity_status: Literal["canonical", "alias", "quarantined_duplicate", "ambiguous"]
    aliases: list[str]
    clubs: list[ClubRef]
    career_games: int
    career_counter_max: int | None = Field(description="source career counter, may exceed rows")
    stat_names: list[str] = Field(description="stat order for career and every season's StatColumns")
    career: StatColumns = Field(description="coverage scope: career_games")
    seasons: list[PlayerSeason]
    forecast: PredictionRow | None
    sources: list[Source]
    coverage_note: str


class PlayerGame(PublicModel):
    match_id: str
    match_date: date | None
    date_quality: Literal["fixture_verified", "source", "inferred", "unknown"]
    stage_label: str
    club_id: str
    opponent_club_id: str | None
    opponent_name: str | None
    result: str | None
    career_game_counter: int | None
    stats: list[float | None] = Field(
        description="positional values aligned to the parent's stat_columns; null = not recorded"
    )


DateQualityLabel = Literal["fixture_verified", "source", "inferred", "unknown"]


class PlayerGameColumns(PublicModel):
    """A season game log as parallel arrays (one entry per game; ``game_rows`` restores rows)."""

    match_id: list[str]
    match_date: list[date | None]
    date_quality: list[DateQualityLabel]
    stage_label: list[str]
    club_id: list[str]
    opponent_club_id: list[str | None]
    opponent_name: list[str | None]
    result: list[str | None]
    career_game_counter: list[int | None]
    stats: list[list[float | None]] = Field(description="per game, positional values aligned to stat_columns")

    @model_validator(mode="after")
    def _aligned(self) -> PlayerGameColumns:
        n = len(self.match_id)
        if any(len(getattr(self, f)) != n for f in _GAME_FIELDS):
            raise ValueError("PlayerGameColumns arrays differ in length")
        return self


_GAME_FIELDS = (
    "match_id", "match_date", "date_quality", "stage_label", "club_id", "opponent_club_id", "opponent_name",
    "result", "career_game_counter", "stats",
)  # fmt: skip


def to_game_columns(rows: list[PlayerGame]) -> PlayerGameColumns:
    return PlayerGameColumns(**{f: [getattr(r, f) for r in rows] for f in _GAME_FIELDS})


def game_rows(cols: PlayerGameColumns) -> list[PlayerGame]:
    return [
        PlayerGame(**dict(zip(_GAME_FIELDS, vals, strict=True)))
        for vals in zip(*(getattr(cols, f) for f in _GAME_FIELDS), strict=True)
    ]


class PlayerSeasonGames(PublicModel):
    player_id: str
    season: int
    stat_columns: list[str] = Field(description="stats observed at least once in this file, in canonical order")
    games: PlayerGameColumns


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------


class LadderRow(PublicModel):
    position: int
    club_id: str
    name: str
    played: int
    won: int
    lost: int
    drawn: int
    points_for: int
    points_against: int
    percentage: float | None = Field(description="null when points_against is zero")
    premiership_points: int


class TeamIndexEntry(PublicModel):
    club_id: str
    name: str
    lineage_id: str
    first_season: int | None
    last_season: int | None
    active: bool
    seasons: list[int]


class TeamIndex(PublicModel):
    teams: list[TeamIndexEntry]


class FormEntry(PublicModel):
    match_id: str
    match_date: date | None
    opponent: str
    result: Literal["W", "L", "D"]
    margin: int


class LeaderRow(PublicModel):
    player_id: str
    name: str
    stat: str
    value: float
    observed_games: int


class Heuristic(PublicModel):
    label: str
    text: str
    method: str


class TeamSeason(PublicModel):
    club: ClubRef
    season: int
    ladder: list[LadderRow]
    ladder_note: str
    position: int | None
    form_window: int
    form: list[FormEntry]
    fixtures: list[MatchSummary]
    team_stats: list[StatValue]
    leaders: list[LeaderRow]
    five_year: list[LadderRow]
    heuristics: list[Heuristic]


# ---------------------------------------------------------------------------
# History / rankings
# ---------------------------------------------------------------------------


class HistoryRow(PublicModel):
    rank: int
    player_id: str | None
    name: str
    clubs: list[str]
    value: float
    value_label: str
    observed_games: int | None
    eligible_games: int | None
    coverage: float | None
    seasons: str | None


class HistoryTable(PublicModel):
    category: str
    title: str
    scope: Literal["career", "single_season", "ranking"]
    era: str
    method: str
    method_version: str
    coverage_note: str
    warning: str | None
    rows: list[HistoryRow]


class HistoryIndexEntry(PublicModel):
    category: str
    title: str
    scope: Literal["career", "single_season", "ranking"]
    eras: list[str]
    resources: dict[str, str]


class HistoryIndex(PublicModel):
    tables: list[HistoryIndexEntry]
    era_summary: list[dict[str, float | int | str | None]]


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


class MetricBlock(PublicModel):
    n: int
    mae: float | None
    rmse: float | None
    bias: float | None
    median_ae: float | None
    within_5: float | None
    within_10: float | None


class CohortMetric(PublicModel):
    dimension: str
    cohort: str
    model: MetricBlock
    baseline: MetricBlock | None
    sufficient: bool


class Populations(PublicModel):
    intended: int
    predicted: int
    joined: int
    played: int
    missing: int
    excluded: int
    exclusion_reasons: dict[str, int]


class AccuracyReport(PublicModel):
    model_id: str
    baseline_id: str | None
    season: int | None
    origin: Literal["prospective", "replay", "legacy_unknown"]
    label: str
    headline: MetricBlock
    baseline_headline: MetricBlock | None
    mean_of_rounds_mae: float | None
    populations: Populations
    cohorts: list[CohortMetric]
    interval: IntervalInfo
    interval_coverage: float | None
    interval_median_width: float | None
    promotion: str
    notes: list[str]
    rows_resource: str | None = Field(description="download key for the scored rows CSV")


class AccuracyIndexEntry(PublicModel):
    model_id: str
    season: int | None
    origin: Literal["prospective", "replay", "legacy_unknown"]
    label: str
    resource: str


class AccuracyIndex(PublicModel):
    champion_model_id: str | None
    baseline_model_id: str | None
    reports: list[AccuracyIndexEntry]
    model_card: str


# ---------------------------------------------------------------------------
# Lists, articles, live, quality, downloads, overview
# ---------------------------------------------------------------------------


class DraftRow(PublicModel):
    season: int
    event_type: str
    draft_round: int | None
    pick: int | None
    club: str | None
    player_name: str
    player_id: str | None
    recruited_from: str | None
    grade: str | None


class ContractRow(PublicModel):
    player_name: str
    player_id: str | None
    club: str | None
    contract_end: int | None
    fa_category: str | None
    observed_at: date | None
    source_type: str
    notes: str | None


class SchoolRow(PublicModel):
    draft_year: int | None
    pick: int | None
    player_name: str
    player_id: str | None
    school: str | None
    school_type: str | None
    confidence: str


class ListsSeason(PublicModel):
    season: int
    drafts: list[DraftRow]
    contracts: list[ContractRow]
    schools: list[SchoolRow]
    source_note: str
    sources: list[Source]


class ListsIndex(PublicModel):
    seasons: list[int]
    resources: dict[str, str]


class ArticleSummary(PublicModel):
    slug: str
    title: str
    category: str
    published: date | None
    as_of: str | None
    scope: Literal["frozen", "live", "archive"]
    excerpt: str
    editorial_state: Literal["published_archive", "generated", "draft"]
    original_path: str
    resource: str


class ArticleIndex(PublicModel):
    articles: list[ArticleSummary]


class Article(PublicModel):
    summary: ArticleSummary
    html: str = Field(description="sanitized at build time; allowlisted tags only")
    sources: list[Source]
    provenance: str


class LivePlayerRow(PublicModel):
    player_id: str
    name: str
    stats: dict[str, float] = Field(description="reliable numeric fields only; an absent key = not reported")


class LiveSnapshot(PublicModel):
    source_game_id: str
    match_id: str | None
    fetched_at: datetime | None
    status: str | None
    quarter: str | None
    home: TeamScore
    away: TeamScore
    reliable_fields: list[str]
    unavailable_fields: list[str]
    players: list[LivePlayerRow]
    timeline: list[dict[str, str | int | None]]
    reads: list[str]
    anomalies: list[str]
    final: bool


class LiveIndexEntry(PublicModel):
    source_game_id: str
    match_id: str | None
    label: str
    last_fetched_at: datetime | None
    final: bool
    resource: str


class LiveIndex(PublicModel):
    delivery: str
    matches: list[LiveIndexEntry]


class QualityIssue(PublicModel):
    rule_id: str
    severity: Literal["info", "warning", "error", "blocking"]
    count: int
    description: str


class QualityReport(PublicModel):
    dataset_status: Literal["legacy_unverified", "verified", "partial", "demo"]
    table_counts: dict[str, int]
    quarantined_rows: int
    issues: list[QualityIssue]
    limitations: list[str]
    sources: list[Source]


class DownloadItem(PublicModel):
    key: str
    label: str
    kind: Literal["csv", "png", "svg", "zip", "json", "md"]
    path: str
    bytes: int
    sha256: str
    as_of: str
    rows: int | None = None


class RetainedRelease(PublicModel):
    release_id: str
    generated_at: datetime
    snapshot_id: str
    path: str


class Downloads(PublicModel):
    release_id: str
    items: list[DownloadItem]
    retained: list[RetainedRelease]


class ModelStatus(PublicModel):
    forecast_status: Literal["available", "unavailable", "expired"]
    reason: str | None
    model: ModelInfo | None


class Overview(PublicModel):
    release_id: str
    snapshot_id: str
    season: int
    demo: bool
    freshness: Freshness
    next_fixture_status: Literal["available", "unavailable", "expired"]
    next_fixture_reason: str | None
    upcoming: list[MatchSummary]
    prediction_highlights: list[PredictionRow]
    form_highlights: list[LeaderRow]
    recent_results: list[MatchSummary]
    latest_articles: list[ArticleSummary]
    leaders: list[LeaderRow]
    model_status: ModelStatus
    warnings: list[str]


#: Resource key -> model, used for schema export and release validation.
PUBLIC_MODELS: dict[str, type[PublicModel]] = {
    "release": ReleaseManifest,
    "overview": Overview,
    "prediction_index": PredictionIndex,
    "prediction_set": PredictionSet,
    "player_index": PlayerIndex,
    "player_detail": PlayerDetail,
    "player_season_games": PlayerSeasonGames,
    "team_index": TeamIndex,
    "team_season": TeamSeason,
    "match_index": MatchIndex,
    "match_detail": MatchDetail,
    "history_index": HistoryIndex,
    "history_table": HistoryTable,
    "accuracy_index": AccuracyIndex,
    "accuracy_report": AccuracyReport,
    "lists_index": ListsIndex,
    "lists_season": ListsSeason,
    "article_index": ArticleIndex,
    "article": Article,
    "live_index": LiveIndex,
    "live_snapshot": LiveSnapshot,
    "quality": QualityReport,
    "downloads": Downloads,
}
