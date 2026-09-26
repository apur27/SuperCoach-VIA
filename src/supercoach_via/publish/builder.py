"""Build one complete public release from one accepted snapshot (PLAN 3, 8.1, 8.2).

``build_release(snapshot, inputs, context)`` renders every browser resource, compatibility
report, CSV/chart download and the fan ZIP from ONE snapshot plus optional, explicitly
passed ML/editorial inputs, stages them with :class:`~supercoach_via.publish.release.ReleaseWriter`
(atomic rename) and then runs :func:`~supercoach_via.publish.release.validate_release`.

Contracts:

* Inputs are explicit. Prediction archives are named directories (hash-verified on load),
  never discovered by glob or mtime; live snapshots are selected from persisted monitor
  state; curated articles only from the public-content manifest.
* The release ID is ``<generated_at>-[demo-]<digest>`` where the digest covers the snapshot,
  every input's content hash, templates/config and the builder version, so identical input
  gives the same ID and byte-identical files. An existing release with that ID is immutable:
  it is re-validated and reused, never rewritten (no duplicate render work).
* A forecast is published only if its artifact verifies, its model manifest is supplied and
  every target is still a scheduled fixture of the snapshot. Otherwise the release says
  ``unavailable``/``expired`` with a reason and the rest of the site still builds.
* Browser JSON, Markdown, CSV rows, chart values and alt text are rendered from the same
  view-model objects. Generated Markdown lives between ``GEN`` markers so curated editorial
  text (report overlays) is never regenerated or duplicated.
* Season-scoped work (match index/details, player-season logs, team pages) runs in ONE
  partition-restricted query context per season: that season's ``player_games`` fragment and
  the matches of that season plus the five before it (five-year ladder), reused for every club.
* Timings and operation counts are returned to the operator on the candidate; they are never
  written into the release, which would break byte identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shutil
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import TypeAdapter

from supercoach_via.analytics import eras as era_analytics
from supercoach_via.analytics import lists as list_analytics
from supercoach_via.analytics import players as player_analytics
from supercoach_via.analytics import rankings as ranking_analytics
from supercoach_via.analytics import teams as team_analytics
from supercoach_via.domain.metrics import CoverageEras
from supercoach_via.domain.schemas import ValidationReport
from supercoach_via.publish import resources
from supercoach_via.publish.bundle import check_member, deterministic_zip, safe_csv_bytes
from supercoach_via.publish.release import ReleaseWriter, check_release_path, list_releases, validate_release
from supercoach_via.publish.reports import replace_marked_section
from supercoach_via.publish.view_models import (
    AccuracyIndex,
    AccuracyIndexEntry,
    AccuracyReport,
    Article,
    ArticleIndex,
    ArticleSummary,
    ClubRef,
    DownloadItem,
    Downloads,
    FormEntry,
    Freshness,
    HistoryIndex,
    HistoryIndexEntry,
    HistoryTable,
    IntervalInfo,
    LadderRow,
    LeaderRow,
    ListsIndex,
    LiveIndex,
    LiveIndexEntry,
    LiveSnapshot,
    MatchSummary,
    MetricBlock,
    ModelInfo,
    ModelStatus,
    Overview,
    PlayerDetail,
    PlayerIndex,
    PlayerSeason,
    Populations,
    PredictionIndex,
    PredictionIndexEntry,
    PredictionRow,
    PredictionSet,
    QualityIssue,
    QualityReport,
    RetainedRelease,
    SeasonLine,
    Source,
    StatColumns,
    StatValue,
    TeamIndex,
    TeamIndexEntry,
    TeamSeason,
)
from supercoach_via.publish.web_data import canonical_json_bytes, sha256_bytes
from supercoach_via.storage.queries import SnapshotQuery
from supercoach_via.storage.snapshots import SnapshotRef, load_snapshot, read_current

if TYPE_CHECKING:  # pragma: no cover
    from supercoach_via.editorial.verify import EditorialOutcome
    from supercoach_via.ml.bundles import BundleManifest, ModelBundle
    from supercoach_via.ml.evaluate import EvaluationArtifact
    from supercoach_via.ml.features import History
    from supercoach_via.ml.predict import PredictionArtifact
    from supercoach_via.settings import RunContext

BUILDER_VERSION = "release-builder-v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEMPLATE_DIR = REPO_ROOT / "templates" / "reports"
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"

REPORTS: dict[str, str] = {
    "season-summary": "Season summary",
    "stat-leaders": "Stat leaders",
    "team-analysis": "Team analysis",
}
LEADER_STATS: tuple[tuple[str, str], ...] = (
    ("disposals", "Disposals"),
    ("goals", "Goals"),
    ("tackles", "Tackles"),
    ("marks", "Marks"),
)
LEADER_N = 10
CAREER_STATS = ("disposals", "goals", "kicks", "handballs", "marks", "tackles", "brownlow_votes")
SEASON_STATS = ("disposals", "goals", "tackles")
PLAYER_CSV_STATS = ("disposals", "kicks", "handballs", "marks", "goals", "tackles")
TEAM_STATS = (
    "disposals",
    "kicks",
    "handballs",
    "marks",
    "goals",
    "tackles",
    "clearances",
    "inside_50s",
    "contested_possessions",
)
FIVE_YEARS = 5
FORM_WINDOW = 5
HISTORY_N = 100
STALE_AFTER = timedelta(days=8)
LIVE_DELIVERY = "snapshot_archive: last accepted live snapshots at build time; not a real-time feed"
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,119}$")
_MD_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
_DT = TypeAdapter(datetime)

Clock = Callable[[], datetime]


class ReleaseBuildError(RuntimeError):
    """The release cannot be built (or an existing release with this ID is not valid)."""


class FanPackError(ReleaseBuildError):
    """A required fan-pack file or dependency is missing; packaging is blocked."""


# ---------------------------------------------------------------------------
# Inputs and result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EditorialDraft:
    """An editorial outcome offered for publication; only ``publishable`` outcomes ship."""

    slug: str
    title: str
    category: str
    outcome: EditorialOutcome


@dataclass(frozen=True)
class ReleaseInputs:
    """Everything optional that a release may include, passed explicitly (never discovered)."""

    prediction_dirs: tuple[Path, ...] = ()
    model_manifests: tuple[BundleManifest, ...] = ()
    evaluations: tuple[EvaluationArtifact, ...] = ()
    editorial: tuple[EditorialDraft, ...] = ()
    content_root: Path | None = None
    content_manifest: Path | None = None
    live_root: Path | None = None
    report_overlays: Mapping[str, str] = field(default_factory=dict)
    clock: Clock | None = None
    public_base: str | None = None
    demo: bool = False
    config_dir: Path | None = None
    template_dir: Path | None = None


@dataclass
class ReleaseCandidate:
    release_id: str
    release_dir: Path
    snapshot_id: str
    validation: ValidationReport
    forecast_status: Literal["available", "unavailable", "expired"]
    forecast_reason: str | None
    reused: bool
    timings: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def public_dir(self) -> Path:
        return self.release_dir / "public"

    @property
    def manifest_path(self) -> Path:
        return self.public_dir / "release.json"

    @property
    def ok(self) -> bool:
        return self.validation.ok


# ---------------------------------------------------------------------------
# Small pure helpers (tested directly)
# ---------------------------------------------------------------------------


def classify_prediction_targets(statuses: Sequence[str | None]) -> Literal["available", "expired"] | None:
    """Status of a forecast from its targets' CURRENT fixture statuses (``None`` = missing/changed).

    All still scheduled -> ``available``; any started/played/postponed -> ``expired``;
    no targets or a target absent from (or re-staged in) the snapshot -> ``None`` (reject).
    """
    if not statuses or any(s is None for s in statuses):
        return None
    return "available" if all(s == "scheduled" for s in statuses) else "expired"


def _iso(dt: datetime) -> str:
    """Exactly the string pydantic writes for this datetime in public JSON."""
    return str(_DT.dump_python(dt, mode="json"))


def stage_text(label: str, stage_type: str) -> str:
    return f"Round {label}" if stage_type == "regular" and label.isdigit() else label


def _num(v: float | int | None) -> str:
    if v is None:
        return "-"
    f = float(v)
    return str(int(f)) if f.is_integer() else f"{f:.1f}"


def _competition_ranks(values: Sequence[float]) -> list[int]:
    """1224 ranking: equal values share the better rank."""
    ranks: list[int] = []
    for i, v in enumerate(values):
        ranks.append(ranks[-1] if i and values[i - 1] == v else i + 1)
    return ranks


def _md(text: object) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def build_fan_pack(
    members: Mapping[str, bytes],
    *,
    required: Iterable[str],
    unavailable: Mapping[str, str],
    meta: Mapping[str, Any],
) -> bytes:
    """Deterministic fan ZIP with README/manifest/checksums; fails on any missing dependency."""
    missing = sorted(r for r in required if r not in members)
    if missing:
        raise FanPackError(f"required fan-pack file(s) missing: {', '.join(missing)}")
    for name, data in sorted(members.items()):
        check_member(name)
        if not name.endswith(".md"):
            continue
        for ref in _MD_LINK.findall(data.decode("utf-8")):
            if ref.startswith(("#", "http://", "https://", "mailto:")):
                continue
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), ref.split("#")[0]))
            if resolved not in members:
                raise FanPackError(f"{name} references {resolved}, which is not in the fan pack")
    manifest = {
        **meta,
        "members": [{"name": n, "bytes": len(members[n]), "sha256": sha256_bytes(members[n])} for n in sorted(members)],
        "unavailable": [{"member": m, "reason": r} for m, r in sorted(unavailable.items())],
    }
    files = dict(members)
    files["manifest.json"] = canonical_json_bytes(manifest)
    files["checksums.sha256"] = "".join(f"{sha256_bytes(files[n])}  {n}\n" for n in sorted(files)).encode()
    return deterministic_zip(files)


def legacy_unknown_report(summary: Mapping[str, Any]) -> AccuracyReport | None:
    """Public report for the imported legacy forecast archive (origin ``legacy_unknown``).

    Descriptive only; never pooled with prospective accuracy. ``None`` when nothing joined.
    """
    metrics = summary.get("metrics")
    if not metrics:
        return None
    rows, joined = int(summary.get("rows", 0)), int(summary.get("joined", 0))
    reasons = {str(k): int(v) for k, v in (summary.get("unjoined_reasons") or {}).items()}
    return AccuracyReport(
        model_id="legacy_unknown",
        baseline_id=None,
        season=None,
        origin="legacy_unknown",
        label=str(summary.get("label", "legacy_unknown archive")),
        headline=MetricBlock(**{k: metrics[k] for k in MetricBlock.model_fields}),
        baseline_headline=None,
        mean_of_rounds_mae=None,
        populations=Populations(
            intended=rows,
            predicted=rows,
            joined=joined,
            played=joined,
            missing=rows - joined,
            excluded=0,
            exclusion_reasons=reasons,
        ),
        cohorts=[],
        interval=IntervalInfo(available=False, level=None, method=None, calibrated=False, reason="legacy archive"),
        interval_coverage=None,
        interval_median_width=None,
        promotion="Archive only: never merged with prospective accuracy (target and cutoff unverified).",
        notes=[str(summary["note"])] if summary.get("note") else [],
        rows_resource=None,
    )


# ---------------------------------------------------------------------------
# Build state
# ---------------------------------------------------------------------------


@dataclass
class _LoadedPrediction:
    artifact: PredictionArtifact
    model: BundleManifest


@dataclass
class _Forecast:
    status: Literal["available", "unavailable", "expired"]
    reason: str | None
    current_path: str | None
    current: PredictionSet | None
    model: ModelInfo | None
    model_id: str | None
    baseline_model_id: str | None


@dataclass
class _Build:
    writer: ReleaseWriter
    snapshot_id: str
    manifest: Any
    data_root: Path
    generated_at: datetime
    season: int
    demo: bool
    base: str
    eras: CoverageEras
    index: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    summaries: dict[str, MatchSummary] = field(default_factory=dict)
    current_teams: list[TeamSeason] = field(default_factory=list)
    downloads: list[DownloadItem] = field(default_factory=list)

    def put_json(self, path: str, model: Any, index_key: str | None = None) -> None:
        self.writer.put_json(path, model)
        if index_key:
            self.index[index_key] = path

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n


def _has(manifest: Any, name: str) -> bool:
    return name in manifest.tables


def _records(q: SnapshotQuery, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    cur = q.con.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Input loading (before the release ID exists)
# ---------------------------------------------------------------------------


def _load_predictions(inputs: ReleaseInputs, warnings: list[str]) -> tuple[list[_LoadedPrediction], int]:
    from pydantic import ValidationError

    from supercoach_via.ml.predict import ArtifactIntegrityError, load_artifact

    models = {m.bundle_id: m for m in inputs.model_manifests}
    loaded: dict[str, _LoadedPrediction] = {}
    rejected = 0
    for d in inputs.prediction_dirs:
        try:
            art = load_artifact(d)
        except (ArtifactIntegrityError, OSError, ValueError, ValidationError) as exc:
            warnings.append(f"prediction artifact {d.name} rejected: {type(exc).__name__}")
            rejected += 1
            continue
        m = art.manifest
        if m.origin != "prospective":
            warnings.append(f"prediction artifact {m.prediction_run_id} rejected: origin {m.origin} is not published")
            rejected += 1
            continue
        model = models.get(m.model_id)
        if model is None:
            warnings.append(f"prediction artifact {m.prediction_run_id} rejected: model manifest {m.model_id} missing")
            rejected += 1
            continue
        loaded[m.prediction_run_id] = _LoadedPrediction(art, model)  # same run twice -> once
    return [loaded[k] for k in sorted(loaded)], rejected


def _load_live(live_root: Path | None, warnings: list[str]) -> list[tuple[str, LiveIndexEntry, LiveSnapshot, bytes]]:
    from supercoach_via.live.monitor import MonitorState

    if live_root is None:
        return []
    out = []
    for state_path in sorted(live_root.glob("*/state.json")):  # persisted state, never directory mtimes
        state = MonitorState(**json.loads(state_path.read_text(encoding="utf-8")))
        if not state.accepted:
            continue
        rel = f"live/{state.source_game_id}/latest.json"
        try:
            check_release_path(rel)
        except ValueError:
            warnings.append("live snapshot skipped: unsafe source game id")
            continue
        raw = (state_path.parent / "snapshots" / f"{state.accepted[-1]}.json").read_bytes()
        snap = LiveSnapshot.model_validate_json(raw)
        entry = LiveIndexEntry(
            source_game_id=state.source_game_id,
            match_id=state.match_id,
            label=state.label or state.source_game_id,
            last_fetched_at=datetime.fromisoformat(state.accepted_at[-1]) if state.accepted_at else snap.fetched_at,
            final=state.final or snap.final,
            resource=rel,
        )
        out.append((rel, entry, snap, raw))
    return out


def _articles(
    inputs: ReleaseInputs, *, base: str, asset_prefix: str, generated_at: datetime, snapshot_id: str
) -> tuple[list[Article], dict[str, Path]]:
    from supercoach_via.publish.articles import build_articles
    from supercoach_via.publish.content import render_markdown

    articles: list[Article] = []
    assets: dict[str, Path] = {}
    if inputs.content_manifest is not None:
        if inputs.content_root is None:
            raise ValueError("content_manifest requires content_root")
        built = build_articles(inputs.content_root, inputs.content_manifest, base=base, asset_prefix=asset_prefix)
        articles.extend(built.articles)
        assets.update(built.assets)
    for draft in inputs.editorial:
        out = draft.outcome
        if out.status != "publishable" or out.rendered_markdown is None:
            continue
        if not _SLUG.match(draft.slug):
            raise ValueError(f"unsafe editorial slug {draft.slug!r}")
        text = out.rendered_markdown
        excerpt = next(
            (" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip() and not p.lstrip().startswith("#")),
            "",
        )[:280]
        summary = ArticleSummary(
            slug=draft.slug,
            title=draft.title[:200],
            category=draft.category,
            published=generated_at.date(),
            as_of=generated_at.date().isoformat(),
            scope="live",
            excerpt=excerpt,
            editorial_state="generated",
            original_path=f"editorial/{draft.slug}",
            resource=f"articles/{draft.slug}.json",
        )
        articles.append(
            Article(
                summary=summary,
                html=render_markdown(text, base=base, link_map={}, source_path=None),
                sources=[Source(label="Deterministic claim-checked evidence packet", url=None, note=snapshot_id)],
                provenance=(
                    "Generated draft that passed deterministic numeric verification against snapshot "
                    f"{snapshot_id}; numbers are rendered from claim objects, not free prose."
                ),
            )
        )
    slugs = [a.summary.slug for a in articles]
    if len(set(slugs)) != len(slugs):
        raise ValueError("article slugs collide between curated and editorial content")
    return articles, assets


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retained(output_root: Path, generated_at: datetime, keep: int) -> list[RetainedRelease]:
    """Previously built releases strictly older than this one (by manifest, never mtime)."""
    out = []
    for rid in list_releases(output_root):
        path = output_root / "releases" / rid / "public" / "release.json"
        if not path.is_file():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        at = datetime.fromisoformat(str(doc["generated_at"]).replace("Z", "+00:00"))
        if at >= generated_at:
            continue
        out.append(
            RetainedRelease(release_id=rid, generated_at=at, snapshot_id=doc["snapshot_id"], path=f"releases/{rid}")
        )
    out.sort(key=lambda r: (r.generated_at, r.release_id), reverse=True)
    return out[: max(0, keep - 1)]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_release(snapshot: SnapshotRef, inputs: ReleaseInputs, context: RunContext) -> ReleaseCandidate:
    """Build, stage, atomically publish into ``<output_root>/releases/<id>`` and validate."""
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    mark = t0

    def tick(label: str) -> None:
        nonlocal mark
        now = time.perf_counter()
        timings[label] = round(now - mark, 4)
        mark = now

    warnings: list[str] = []
    clock = inputs.clock or context.clock
    generated_at = clock().astimezone(UTC)
    base = inputs.public_base or context.settings.public_base
    data_root = context.data_root
    output_root = context.output_root
    config_dir = inputs.config_dir or DEFAULT_CONFIG_DIR
    template_dir = inputs.template_dir or DEFAULT_TEMPLATE_DIR
    unknown = sorted(set(inputs.report_overlays) - set(REPORTS))
    if unknown:
        raise ValueError(f"unknown report overlay(s): {unknown}; known report names are {sorted(REPORTS)}")
    for name, text in inputs.report_overlays.items():
        replace_marked_section(text, name, "")  # fail closed on missing/duplicate markers before any work

    manifest = load_snapshot(data_root, snapshot.snapshot_id, verify=True)
    predictions, rejected = _load_predictions(inputs, warnings)
    live = _load_live(inputs.live_root, warnings)
    probe, probe_assets = _articles(
        inputs, base=base, asset_prefix="data/_/", generated_at=generated_at, snapshot_id=manifest.snapshot_id
    )
    retained = _retained(output_root, generated_at, context.settings.retain_releases)
    digest_parts = {
        "builder": BUILDER_VERSION,
        "snapshot": manifest.snapshot_id,
        "generated_at": _iso(generated_at),
        "demo": inputs.demo,
        "base": base,
        "season": context.settings.season,
        "predictions": [
            [
                p.artifact.manifest.prediction_run_id,
                p.artifact.manifest.rows_sha256,
                p.artifact.manifest.omissions_sha256,
                p.model.manifest_sha256,
            ]
            for p in predictions
        ],
        "rejected_predictions": rejected,
        "models": sorted(m.manifest_sha256 for m in inputs.model_manifests),
        "evaluations": sorted(
            [e.evaluation_id, sha256_bytes(e.scored_rows.to_csv(index=False).encode())] for e in inputs.evaluations
        ),
        "articles": sha256_bytes(canonical_json_bytes([a.model_dump(mode="json") for a in probe])),
        "article_assets": {k: _file_sha(v) for k, v in sorted(probe_assets.items())},
        "live": [[rel, sha256_bytes(raw), e.model_dump(mode="json")] for rel, e, _s, raw in live],
        "overlays": {k: sha256_bytes(v.encode()) for k, v in sorted(inputs.report_overlays.items())},
        "templates": {p.name: _file_sha(p) for p in sorted(template_dir.glob("*.j2"))},
        "config": {n: _file_sha(config_dir / n) for n in ("coverage.yaml", "ranking_legacy_v1.toml")},
        "retained": [r.release_id for r in retained],
    }
    digest = sha256_bytes(canonical_json_bytes(digest_parts))
    release_id = f"{generated_at:%Y%m%dT%H%M%SZ}-{'demo-' if inputs.demo else ''}{digest[:12]}"
    tick("snapshot_s")

    final = output_root / "releases" / release_id
    if final.exists():
        report = validate_release(final, write=False)
        if not report.ok:
            raise ReleaseBuildError(f"existing release {release_id} failed validation; it is immutable, not rebuilt")
        doc = json.loads((final / "public" / "release.json").read_text(encoding="utf-8"))
        return ReleaseCandidate(
            release_id=release_id,
            release_dir=final,
            snapshot_id=manifest.snapshot_id,
            validation=report,
            forecast_status=doc["forecast"]["status"],
            forecast_reason=doc["forecast"]["reason"],
            reused=True,
            timings={
                "reuse_check_s": round(time.perf_counter() - mark, 4),
                "total_s": round(time.perf_counter() - t0, 4),
            },
            warnings=warnings,
        )

    with SnapshotQuery(data_root, manifest) as q:
        season = context.settings.season or int(q.scalar("SELECT max(season) FROM matches"))
    writer = ReleaseWriter(output_root, release_id)
    b = _Build(
        writer=writer,
        snapshot_id=manifest.snapshot_id,
        manifest=manifest,
        data_root=data_root,
        generated_at=generated_at,
        season=season,
        demo=inputs.demo,
        base=base,
        eras=CoverageEras.load(config_dir / "coverage.yaml"),
        warnings=warnings,
    )
    try:
        forecast = _build_all(b, inputs, predictions, rejected, live, retained, template_dir, tick)
        release_dir = writer.finish(
            snapshot_id=manifest.snapshot_id,
            generated_at=generated_at,
            season=season,
            demo=inputs.demo,
            coverage_status="demo" if inputs.demo else manifest.status.value,
            coverage_through=_coverage_through(b),
            forecast_status=forecast.status,
            forecast_reason=forecast.reason,
            forecast_artifact=forecast.current_path,
            index_resources=b.index,
            forecast_model_id=forecast.model_id,
        )
    except BaseException:
        shutil.rmtree(writer.staging, ignore_errors=True)
        raise
    tick("finish_s")
    report = validate_release(release_dir)
    tick("validate_s")
    timings["total_s"] = round(time.perf_counter() - t0, 4)
    b.counts["files"] = len(writer.files)
    b.counts["bytes"] = sum(int(v["bytes"]) for v in writer.files.values())
    return ReleaseCandidate(
        release_id=release_id,
        release_dir=release_dir,
        snapshot_id=manifest.snapshot_id,
        validation=report,
        forecast_status=forecast.status,
        forecast_reason=forecast.reason,
        reused=False,
        timings=timings,
        counts=b.counts,
        warnings=warnings,
    )


def _build_all(
    b: _Build,
    inputs: ReleaseInputs,
    predictions: list[_LoadedPrediction],
    rejected: int,
    live: list[tuple[str, LiveIndexEntry, LiveSnapshot, bytes]],
    retained: list[RetainedRelease],
    template_dir: Path,
    tick: Callable[[str], None],
) -> _Forecast:
    live_by_match: dict[str, list[str]] = {}
    for rel, entry, snap, _raw in live:
        b.put_json(rel, snap)
        if entry.match_id:
            live_by_match.setdefault(entry.match_id, []).append(rel)
    b.put_json(
        "live/index.json", LiveIndex(delivery=LIVE_DELIVERY, matches=[e for _r, e, _s, _b in live]), "live_index"
    )

    with SnapshotQuery(b.data_root, b.manifest) as q:
        seasons = resources.seasons(q)
        team_index = _team_index(q)
        b.put_json("teams/index.json", team_index, "team_index")
    _season_resources(b, seasons, live_by_match)
    tick("season_resources_s")

    with SnapshotQuery(b.data_root, b.manifest) as q:
        history = _history(b, q)
        tick("history_s")
        light = _light_history(b, q) if predictions or _legacy_prediction_rows(b) else None
        forecast = _predictions(b, q, predictions, rejected, light)
        tick("predictions_s")
        pidx = resources.player_index(q)
        b.put_json("players/index.json", pidx, "player_index")
        player_rows = _player_details(b, q, pidx, forecast)
        names = {
            pid: (str(f or ""), str(la or ""))
            for pid, f, la in q.rows("SELECT player_id, first_name, last_name FROM players")
        }
        tick("players_s")
        accuracy_rows = _accuracy(b, inputs, forecast, light)
        _lists(b, q)
        articles = _publish_articles(b, inputs)
        leaders = {
            stat: player_analytics.season_leaders(q, stat, b.season, n=LEADER_N) for stat, _label in LEADER_STATS
        }
        overview = _overview(b, q, forecast, articles, leaders)
        b.put_json("overview.json", overview, "overview")
        b.put_json("quality.json", _quality(b, q, forecast), "quality")
        tick("resources_s")
        _downloads(
            b,
            inputs,
            template_dir,
            forecast,
            overview,
            leaders,
            pidx,
            player_rows,
            names,
            history,
            accuracy_rows,
            retained,
        )
        tick("downloads_s")
    return forecast


# ---------------------------------------------------------------------------
# Season-scoped resources: matches, player-season logs, team pages
# ---------------------------------------------------------------------------


def _team_index(q: SnapshotQuery) -> TeamIndex:
    rows = q.rows(
        """WITH s AS (SELECT home_club_id AS club_id, season FROM matches
                      UNION SELECT away_club_id, season FROM matches)
           SELECT c.club_id, c.name, c.lineage_id, c.first_season, c.last_season, c.active,
                  list(DISTINCT s.season ORDER BY s.season)
           FROM clubs c JOIN s USING (club_id) GROUP BY ALL ORDER BY c.name, c.club_id"""
    )
    return TeamIndex(
        teams=[
            TeamIndexEntry(
                club_id=cid,
                name=name,
                lineage_id=lineage,
                first_season=first if first is not None else min(ss),
                last_season=last if last is not None else max(ss),
                active=bool(active),
                seasons=[int(s) for s in ss],
            )
            for cid, name, lineage, first, last, active, ss in rows
        ]
    )


SEASON_WORKERS = min(3, os.cpu_count() or 1)  # ~400 MiB per worker; keeps the build under 2 GiB


@dataclass
class _SeasonResult:
    season: int
    files: dict[str, dict[str, Any]]
    counts: dict[str, int]
    summaries: list[MatchSummary]
    current_teams: list[TeamSeason]
    match_frags: int
    pg_frags: int


_SEASON_TABLES = {"matches", "player_games", "clubs", "players", "seasons"}


def _season_query(b: _Build, season: int) -> SnapshotQuery:
    parts = {
        "matches": {str(s) for s in range(season - FIVE_YEARS, season + 1)},
        "player_games": {str(season)},
    }
    return SnapshotQuery(b.data_root, b.manifest, tables=_SEASON_TABLES, partitions=parts)


def _one_season(
    b: _Build, season: int, ladders: dict[int, list[LadderRow]], live_by_match: dict[str, list[str]]
) -> _SeasonResult:
    """Every season-scoped resource of one season (match index/detail, game logs, team pages)."""
    start = dict(b.counts)
    with _season_query(b, season) as q:
        if SEASON_WORKERS > 1:
            q.con.execute("SET threads TO 1")  # workers run side by side: no oversubscription
        b.bump("season_contexts")
        mindex = resources.match_index(q, season)
        b.put_json(f"matches/{season}/index.json", mindex)
        for detail in resources.match_details(q, season):
            links = live_by_match.get(detail.summary.match_id)
            if links:
                detail = detail.model_copy(update={"live_snapshots": links})
            b.put_json(f"matches/detail/{resources.match_key(detail.summary.match_id)}.json", detail)
            b.bump("match_details")
        for log in resources.player_season_games(q, season):
            b.put_json(f"player-games/{resources.public_key(log.player_id)}/{season}.json", log)
            b.bump("player_season_logs")
        teams = []
        for ts in _team_seasons(b, q, season, ladders, mindex.matches):
            b.put_json(f"teams/{ts.club.club_id}/{season}.json", ts)
            b.bump("team_pages")
            if season == b.season:
                teams.append(ts)
        frags = (q.fragments_registered.get("matches", 0), q.fragments_registered.get("player_games", 0))
    counts = {k: v - start.get(k, 0) for k, v in b.counts.items() if v != start.get(k, 0)}
    return _SeasonResult(season, {}, counts, mindex.matches, teams, *frags)


def _isolated(b: _Build, writer: ReleaseWriter | None = None) -> _Build:
    """A copy of ``b`` with its own counters/collections (and optionally another writer)."""
    return replace(b, writer=writer or b.writer, counts={}, index={}, summaries={}, current_teams=[], downloads=[],
                   warnings=[])  # fmt: skip


def _season_worker(
    b: _Build, season: int, ladders: dict[int, list[LadderRow]], live_by_match: dict[str, list[str]]
) -> _SeasonResult:
    """Process-pool entry: same work as the sequential path, through an attached writer."""
    wb = _isolated(b, ReleaseWriter.attach(b.writer.output_root, b.writer.release_id))
    res = _one_season(wb, season, ladders, live_by_match)
    res.files = wb.writer.files
    return res


def _season_ladders(b: _Build, seasons: list[int]) -> dict[int, list[LadderRow]]:
    ladders: dict[int, list[LadderRow]] = {}
    for season in seasons:
        with _season_query(b, season) as q:
            ladders[season] = team_analytics.ladder(q, season)
    return ladders


def _season_resources(b: _Build, seasons: list[int], live_by_match: dict[str, list[str]]) -> None:
    """Season resources, in a small process pool when SEASON_WORKERS > 1 (byte-identical either way)."""
    results: list[_SeasonResult] = []
    if SEASON_WORKERS <= 1:
        ladders: dict[int, list[LadderRow]] = {}
        own = _isolated(b)
        for season in seasons:  # ascending, so prior ladders are cached for the five-year view
            results.append(_one_season(own, season, ladders, live_by_match))  # files go to b.writer directly
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        all_ladders = _season_ladders(b, seasons)  # the only cross-season dependency (five-year view)
        base = _isolated(b, cast(ReleaseWriter, _DetachedWriter(b.writer)))

        def args(s: int) -> tuple[dict[int, list[LadderRow]], dict[str, list[str]]]:
            prior = {y: all_ladders[y] for y in range(s - FIVE_YEARS, s + 1) if y in all_ladders}
            return prior, {m: v for m, v in live_by_match.items() if m.startswith(f"m:{s}:")}

        ctx = multiprocessing.get_context("spawn")  # never fork an open DuckDB/process state
        with ProcessPoolExecutor(max_workers=SEASON_WORKERS, mp_context=ctx) as pool:
            futures = [pool.submit(_season_worker, base, s, *args(s)) for s in seasons]
            results = [f.result() for f in futures]
        for r in results:
            b.writer.merge(r.files)
    match_frags = pg_frags = 0
    for r in sorted(results, key=lambda r: r.season):
        for k, v in r.counts.items():
            b.bump(k, v)
        b.index[f"match_index:{r.season}"] = f"matches/{r.season}/index.json"
        b.summaries.update({m.match_id: m for m in r.summaries})
        b.current_teams.extend(r.current_teams)
        match_frags, pg_frags = max(match_frags, r.match_frags), max(pg_frags, r.pg_frags)
    b.counts["season_context_match_fragments_max"] = match_frags
    b.counts["season_context_player_game_fragments_max"] = pg_frags


@dataclass(frozen=True)
class _DetachedWriter:
    """Picklable stand-in carrying only what a worker needs to attach to the staging directory."""

    output_root: Path
    release_id: str

    def __init__(self, writer: ReleaseWriter) -> None:
        object.__setattr__(self, "output_root", writer.output_root)
        object.__setattr__(self, "release_id", writer.release_id)


TEAM_LEADER_STATS = ("disposals", "goals", "tackles")  # = analytics.teams.club_season_leaders default


@dataclass(frozen=True)
class TeamParts:
    leaders: list[LeaderRow]
    fixtures: list[MatchSummary]
    form: list[FormEntry]


def _chrono_key(m: MatchSummary) -> tuple[Any, ...]:
    """``analytics.teams.CHRONO_ORDER`` (nulls last) as a Python sort key."""
    return (
        m.match_date is None,
        m.match_date or date.min,
        m.local_start is None,
        m.local_start or "",
        m.stage_order,
        m.replay_occurrence,
        m.match_id,
    )


def season_team_parts(
    q: SnapshotQuery, season: int, clubs: Sequence[str], matches: Sequence[MatchSummary] | None = None
) -> dict[str, TeamParts]:
    """Per-club leaders, fixtures and form for one season in O(stats) queries, not O(clubs).

    Equal to ``club_season_leaders`` / ``club_fixtures`` / ``team_form(n=FORM_WINDOW)`` of
    ``analytics.teams`` evaluated per club in the same (season-scoped) context; a parity test
    holds this. ``matches`` (the season's summaries) avoids re-reading the fixture.
    """
    names = dict(q.rows("SELECT club_id, name FROM clubs"))
    players = dict(q.rows("SELECT player_id, display_name FROM players"))
    leaders: dict[str, list[LeaderRow]] = {c: [] for c in clubs}
    for stat in TEAM_LEADER_STATS:
        col = '"' + stat + '"'  # stat names are a module constant
        for club, pid, v, obs in q.rows(
            f"""WITH t AS (SELECT club_id, player_id, SUM({col}) AS v, COUNT({col}) AS obs FROM player_games
                           WHERE season = ? GROUP BY club_id, player_id HAVING COUNT({col}) > 0),
                     r AS (SELECT *, row_number() OVER (PARTITION BY club_id ORDER BY v DESC, player_id) AS rn FROM t)
                SELECT club_id, player_id, v, obs FROM r WHERE rn = 1""",  # noqa: S608 - constant identifier
            [season],
        ):
            if club in leaders:
                leaders[club].append(
                    LeaderRow(
                        player_id=pid, name=players.get(pid, pid), stat=stat, value=float(v), observed_games=int(obs)
                    )
                )
    if matches is None:
        matches = resources.match_index(q, season).matches
    ordered = sorted(matches, key=_chrono_key)
    fixtures = {c: [m for m in ordered if c in (m.home.club_id, m.away.club_id)] for c in clubs}
    done: dict[str, list[tuple[Any, ...]]] = {c: [] for c in clubs}
    for mid, mdate, home, away, hs, as_ in q.rows(
        f"""SELECT match_id, match_date, home_club_id, away_club_id, home_score, away_score FROM matches
            WHERE status = 'complete' AND home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY {team_analytics.CHRONO_ORDER}"""  # noqa: S608 - module constant
    ):
        for club in (home, away):
            if club in done:
                done[club].append((mid, mdate, home, away, hs, as_))
    form: dict[str, list[FormEntry]] = {}
    for club in clubs:
        out = []
        for mid, mdate, home, away, hs, as_ in done[club][-FORM_WINDOW:]:
            own, opp, opp_id = (hs, as_, away) if home == club else (as_, hs, home)
            margin = int(own - opp)
            result: Literal["W", "L", "D"] = "W" if margin > 0 else "L" if margin < 0 else "D"
            out.append(
                FormEntry(
                    match_id=mid, match_date=mdate, opponent=names.get(opp_id, opp_id), result=result, margin=margin
                )
            )
        form[club] = out
    return {c: TeamParts(leaders[c], fixtures[c], form[c]) for c in clubs}


def _team_seasons(
    b: _Build, q: SnapshotQuery, season: int, ladders: dict[int, list[LadderRow]], matches: list[MatchSummary]
) -> Iterable[TeamSeason]:
    """All club pages of one season from ONE scoped context; season-level work is done once."""
    clubs = q.rows(
        """SELECT c.club_id, c.name FROM clubs c JOIN (
               SELECT home_club_id AS club_id FROM matches WHERE season = ?
               UNION SELECT away_club_id FROM matches WHERE season = ?) s USING (club_id)
           ORDER BY c.club_id""",
        [season, season],
    )
    ladder = ladders.get(season)
    if ladder is None:  # the parallel path passes every season's ladder precomputed
        ladder = ladders[season] = team_analytics.ladder(q, season)
    pathway = team_analytics.finals_pathway(q, season)
    tg = team_analytics.team_games(q, stats=TEAM_STATS, seasons=[season])
    parts = season_team_parts(q, season, [c for c, _n in clubs], matches)
    b.bump("team_leader_queries", len(TEAM_LEADER_STATS))
    positions = {r.club_id: r.position for r in ladder}
    for club_id, name in clubs:
        part = parts[club_id]
        yield TeamSeason(
            club=ClubRef(club_id=club_id, name=name),
            season=season,
            ladder=ladder,
            ladder_note=team_analytics.LADDER_METHOD,
            position=positions.get(club_id),
            form_window=FORM_WINDOW,
            form=part.form,  # the context ends at this season, so form never includes later matches
            fixtures=part.fixtures,
            team_stats=team_analytics.team_season_stats(tg, club_id, season, stats=TEAM_STATS),
            leaders=part.leaders,
            five_year=[
                r for s in range(season - FIVE_YEARS, season) for r in ladders.get(s, []) if r.club_id == club_id
            ],
            heuristics=pathway.get(club_id, []),
        )


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PlayerRow:
    player_id: str
    name: str
    clubs: str
    first_season: int | None
    last_season: int | None
    career_games: int
    active: bool
    stats: tuple[tuple[float | None, int], ...]


def _nn(v: Any) -> float | None:
    if v is None:
        return None
    f = float(v)
    return None if f != f else f


def player_stat_lines(
    bundle: player_analytics.PlayerStatsBundle, games_resource: Callable[[str, int], str]
) -> dict[str, tuple[list[StatValue], list[SeasonLine]]]:
    """Career ``StatValue`` and per-season ``SeasonLine`` lists for EVERY player in one pass.

    Equal to ``career_stat_values`` / ``season_lines`` of ``analytics.players`` per player (a
    parity test holds this) without filtering the whole-corpus frames once per player, which
    is quadratic on the real corpus.
    """

    def sv(r: Mapping[Any, Any]) -> StatValue:
        # Trusted typed analytics rows (1.6M on the real corpus): skip per-object validation.
        # to_stat_columns re-checks every mean/coverage derivation before anything is published.
        return StatValue.model_construct(
            stat=str(r["stat"]),
            total=_nn(r["total"]),
            mean=_nn(r["mean"]),
            observed_games=int(r["observed_games"]),
            eligible_games=int(r["eligible_games"]),
            coverage=_nn(r["coverage"]),
        )

    careers: dict[str, list[StatValue]] = {}
    for r in bundle.careers.to_dict("records"):
        careers.setdefault(str(r["player_id"]), []).append(sv(r))
    seasons: dict[str, dict[int, tuple[int, list[StatValue]]]] = {}
    for r in bundle.seasons.to_dict("records"):
        per = seasons.setdefault(str(r["player_id"]), {})
        s = int(r["season"])
        if s not in per:
            per[s] = (int(r["games"]), [])
        per[s][1].append(sv(r))
    clubs = {
        (str(r["player_id"]), int(r["season"])): [str(c) for c in r["clubs"]]
        for r in bundle.season_clubs.to_dict("records")
    }
    out = {}
    for pid in (str(p) for p in bundle.games["player_id"]):
        lines = [
            SeasonLine(
                season=s, clubs=clubs.get((pid, s), []), games=g, stats=stats, games_resource=games_resource(pid, s)
            )
            for s, (g, stats) in sorted(seasons.get(pid, {}).items())
        ]
        out[pid] = (careers.get(pid, []), lines)
    return out


class PlayerStatColumns:
    """Per-player compact stats straight from the bundle frames, one player at a time.

    Equal to packing ``player_stat_lines`` with ``to_stat_columns`` (a test holds this) but never
    materialises the ~1.7M per-stat objects of the real corpus. Mean and coverage are verified to
    be the derived values once, vectorised, before anything is returned (fails closed).
    """

    def __init__(self, bundle: player_analytics.PlayerStatsBundle, games_resource: Callable[[str, int], str]):
        self._res = games_resource
        self._c, self._c_idx = self._arrays(bundle.careers, ["player_id"], "career_games")
        self._s, s_idx = self._arrays(bundle.seasons, ["player_id", "season"], "games")
        self._s_by_player: dict[str, list[tuple[int, Any]]] = {}
        for (pid, season), idx in sorted(s_idx.items()):
            self._s_by_player.setdefault(str(pid), []).append((int(season), idx))
        sc = bundle.season_clubs
        self._clubs = {
            (str(p), int(s)): [str(c) for c in clubs]
            for p, s, clubs in zip(sc["player_id"], sc["season"], sc["clubs"], strict=True)
        }

    @staticmethod
    def _arrays(frame: Any, keys: list[str], scope: str) -> tuple[dict[str, Any], dict[Any, Any]]:
        import numpy as np

        total = frame["total"].to_numpy(dtype="float64")
        obs = frame["observed_games"].to_numpy(dtype="int64")
        den = frame[scope].to_numpy(dtype="int64")
        mean = frame["mean"].to_numpy(dtype="float64")
        cov = frame["coverage"].to_numpy(dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            want_mean = np.where((obs > 0) & ~np.isnan(total), total / np.maximum(obs, 1), np.nan)
            want_cov = np.where(den > 0, np.minimum(1.0, obs / np.maximum(den, 1)), np.nan)
        mean_ok = (mean == want_mean) | (np.isnan(mean) & np.isnan(want_mean))
        cov_ok = (cov == want_cov) | (np.isnan(cov) & np.isnan(want_cov))
        if not (mean_ok.all() and cov_ok.all()):
            raise ValueError("bundle mean/coverage is not derived from total/observed/scope games")
        arrays = {"stat": frame["stat"].to_numpy(dtype=object), "total": total, "observed": obs,
                  "eligible": frame["eligible_games"].to_numpy(dtype="int64"), "scope": den}  # fmt: skip
        key = keys[0] if len(keys) == 1 else keys
        return arrays, frame.groupby(key, sort=False).indices

    def _cols(self, a: dict[str, Any], idx: Any) -> StatColumns:
        t = a["total"][idx]
        return StatColumns(
            total=[None if v != v else float(v) for v in t.tolist()],
            observed_games=a["observed"][idx].tolist(),
            eligible_games=a["eligible"][idx].tolist(),
        )

    def get(self, pid: str) -> tuple[list[str], StatColumns, list[PlayerSeason]]:
        idx = self._c_idx.get(pid)
        if idx is None:
            return [], StatColumns(total=[], observed_games=[], eligible_games=[]), []
        names = [str(x) for x in self._c["stat"][idx].tolist()]
        seasons = []
        for season, sidx in self._s_by_player.get(pid, []):
            if [str(x) for x in self._s["stat"][sidx].tolist()] != names:
                raise ValueError(f"{pid} {season}: season stat order differs from career order")
            seasons.append(PlayerSeason(season=season, clubs=self._clubs.get((pid, season), []),
                                        games=int(self._s["scope"][sidx[0]]), stats=self._cols(self._s, sidx),
                                        games_resource=self._res(pid, season)))  # fmt: skip
        return names, self._cols(self._c, idx), seasons


def _player_details(b: _Build, q: SnapshotQuery, pidx: PlayerIndex, forecast: _Forecast) -> dict[str, _PlayerRow]:
    """Write every canonical player's detail page; returns the rows for the players CSV."""
    import pandas as pd

    def games_resource(pid: str, season: int) -> str:
        return f"player-games/{resources.public_key(pid)}/{season}.json"

    bundle = player_analytics.player_stats_bundle(q, b.eras)
    stat_cols = PlayerStatColumns(bundle, games_resource)
    games = {str(r["player_id"]): r for r in bundle.games.to_dict("records")}
    club_names = dict(q.rows("SELECT club_id, name FROM clubs"))
    people = {r["player_id"]: r for r in _records(q, "SELECT * FROM players")}
    aliases: dict[str, list[str]] = {}
    if _has(b.manifest, "player_aliases"):
        for pid, alias in q.rows("SELECT player_id, alias FROM player_aliases ORDER BY player_id, alias"):
            aliases.setdefault(pid, []).append(alias)
    forecast_rows = {r.player_id: r for r in (forecast.current.rows if forecast.current else [])}
    out_rows: dict[str, _PlayerRow] = {}
    for entry in pidx.players:
        pid = entry.id
        names, career_cols, player_seasons = stat_cols.get(pid)
        club_ids: list[str] = []
        for line in player_seasons:
            club_ids.extend(c for c in line.clubs if c not in club_ids)
        g = games.get(pid)
        counter = None if g is None or pd.isna(g["counter_max"]) else int(g["counter_max"])
        p = people[pid]
        career_games = int(g["career_games"]) if g is not None else 0
        urls = json.loads(p["source_urls"]) if p.get("source_urls") else []
        sources = [Source(label="Legacy player CSV import", url=None, note=p.get("source_path"))]
        sources += [Source(label="Verified source page", url=u) for u in urls if str(u).startswith("https://")]
        detail = PlayerDetail(
            id=pid,
            key=entry.key,
            name=entry.name,
            first_name=p.get("first_name"),
            last_name=p.get("last_name"),
            birth_date=p.get("birth_date"),
            birth_date_quality=p["birth_date_quality"],
            debut_date=p.get("debut_date"),
            height_cm=p.get("height_cm"),
            weight_kg=p.get("weight_kg"),
            identity_status=p["identity_status"],
            aliases=aliases.get(pid, []),
            clubs=[ClubRef(club_id=c, name=club_names.get(c, c)) for c in club_ids],
            career_games=career_games,
            career_counter_max=counter,
            stat_names=names,
            career=career_cols,
            seasons=player_seasons,
            forecast=forecast_rows.get(pid),
            sources=sources,
            coverage_note=player_analytics.COVERAGE_NOTE,
        )
        b.put_json(f"players/{entry.key}.json", detail)
        b.bump("player_pages")
        by_stat = {n: (t, o) for n, t, o in zip(names, career_cols.total, career_cols.observed_games, strict=True)}
        out_rows[pid] = _PlayerRow(
            player_id=pid,
            name=entry.name,
            clubs="; ".join(entry.clubs),
            first_season=entry.first_season,
            last_season=entry.last_season,
            career_games=detail.career_games,
            active=entry.active,
            stats=tuple(
                by_stat.get(s, (None, 0)) for s in PLAYER_CSV_STATS
            ),
        )
    return out_rows


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


@dataclass
class _History:
    ranking: Any
    all_time: HistoryTable


def _history(b: _Build, q: SnapshotQuery) -> _History:
    lo, hi = q.rows("SELECT min(season), max(season) FROM player_games")[0]
    eras = [(n, s, e) for n, s, e in era_analytics.era_bounds(q) if lo is not None and s <= hi and e >= lo]
    entries: list[HistoryIndexEntry] = []

    def publish(category: str, title: str, scope: Any, tables: list[tuple[str, HistoryTable]]) -> None:
        res = {}
        for era, table in tables:
            path = f"history/{category}/{era}.json"
            b.put_json(path, table)
            res[era] = path
        entries.append(HistoryIndexEntry(category=category, title=title, scope=scope, eras=list(res), resources=res))

    for stat in CAREER_STATS:
        tables = [("all", player_analytics.career_leaders(q, stat, eras=b.eras, n=HISTORY_N))]
        for name, s, e in eras:
            t = player_analytics.career_leaders(q, stat, eras=b.eras, n=HISTORY_N, seasons=(s, e))
            if t.rows:
                tables.append((name, t))
        publish(tables[0][1].category, tables[0][1].title, "career", tables)
    games = player_analytics.games_leaders(q, n=HISTORY_N)
    publish(games.category, games.title, "career", [("all", games)])
    for stat in SEASON_STATS:
        t = player_analytics.single_season_leaders(q, stat, eras=b.eras, n=HISTORY_N)
        publish(t.category, t.title, "single_season", [("all", t)])
    ranking = ranking_analytics.run_legacy_v1(q)
    all_time = ranking_analytics.all_time_history_table(q, ranking)
    publish(all_time.category, all_time.title, "ranking", [("all", all_time)])
    yearly = [(str(s), ranking_analytics.yearly_history_table(q, ranking, s)) for s in sorted(ranking.yearly)]
    if yearly:
        publish("yearly_top_100", "Yearly top 100", "ranking", yearly)
    summary: list[dict[str, float | int | str | None]] = []
    for name, s, e in eras:
        players_n, games_n, matches_n = q.rows(
            """SELECT count(DISTINCT player_id), count(*),
                      count(DISTINCT match_id) FROM player_games WHERE season BETWEEN ? AND ?""",
            [s, e],
        )[0]
        summary.append(
            {
                "era": name,
                "first_season": s,
                "last_season": e,
                "players": int(players_n),
                "player_games": int(games_n),
                "matches": int(matches_n),
            }
        )
    b.put_json("history/index.json", HistoryIndex(tables=entries, era_summary=summary), "history_index")
    return _History(ranking=ranking, all_time=all_time)


# ---------------------------------------------------------------------------
# Predictions and accuracy
# ---------------------------------------------------------------------------


def _legacy_prediction_rows(b: _Build) -> int:
    entry = b.manifest.tables.get("legacy_predictions")
    return int(entry.row_count) if entry is not None else 0


def _light_history(b: _Build, q: SnapshotQuery) -> History:
    """Only the columns ``to_prediction_set`` / ``legacy_unknown_summary`` read (no full load)."""
    import pandas as pd

    from supercoach_via.ml.features import History

    def df(name: str, sql: str) -> pd.DataFrame:
        return q.df(sql) if _has(b.manifest, name) else pd.DataFrame()

    extra = {}
    if _legacy_prediction_rows(b):
        extra["legacy_predictions"] = df("legacy_predictions", "SELECT * FROM legacy_predictions")
    return History(
        snapshot_id=b.snapshot_id,
        matches=df("matches", "SELECT * FROM matches ORDER BY match_id"),
        player_games=df("player_games", "SELECT match_id, player_id, club_id, season, disposals FROM player_games"),
        players=df("players", "SELECT player_id, display_name FROM players"),
        clubs=df("clubs", "SELECT club_id, name FROM clubs"),
        venues=df("venues", "SELECT venue_id, name FROM venues"),
        lineups=pd.DataFrame(),
        extra=extra,
    )


def _manifest_only(m: BundleManifest) -> ModelBundle:
    """Publishing reads only ``bundle.manifest``; the trusted-local model payload is never loaded."""
    from supercoach_via.ml.bundles import ModelBundle

    return ModelBundle(manifest=m, predictor=None)


def _predictions(
    b: _Build, q: SnapshotQuery, loaded: list[_LoadedPrediction], rejected: int, light: History | None
) -> _Forecast:
    from supercoach_via.ml.predict import to_prediction_set

    status = {
        mid: (st, s, stage) for mid, st, s, stage in q.rows("SELECT match_id, status, season, stage_id FROM matches")
    }
    sets: dict[tuple[int, str], tuple[PredictionSet, _LoadedPrediction]] = {}
    unavailable_reasons: list[str] = []
    for lp in loaded:
        m = lp.artifact.manifest
        if m.status != "available" or m.season is None or not m.stage_id:
            unavailable_reasons.append(m.reason or m.status)
            continue
        targets = []
        for mid, claim in m.target_matches.items():
            cur = status.get(mid)
            ok = cur is not None and int(cur[1]) == int(claim["season"]) and cur[2] == claim["stage_id"]
            targets.append(cur[0] if ok and cur is not None else None)
        state = classify_prediction_targets(targets)
        if state is None:
            b.warnings.append(f"prediction artifact {m.prediction_run_id} rejected: targets absent or re-staged")
            rejected += 1
            continue
        try:
            check_release_path(f"predictions/{int(m.season)}/{m.stage_id}.json")
        except ValueError:
            b.warnings.append(f"prediction artifact {m.prediction_run_id} rejected: unsafe stage id")
            rejected += 1
            continue
        assert light is not None
        pset: PredictionSet = to_prediction_set(lp.artifact, light, _manifest_only(lp.model))
        update: dict[str, Any] = {"target_matches": [b.summaries[mid] for mid in m.target_matches]}
        if state == "expired":
            update.update(status="expired", reason="targets_started_or_played")
        pset = pset.model_copy(update=update)
        key = (int(m.season), str(m.stage_id))
        prev = sets.get(key)
        if prev is not None:
            pm = prev[1].artifact.manifest
            keep_new = (m.generated_at, m.prediction_run_id) > (pm.generated_at, pm.prediction_run_id)
            loser = pm.prediction_run_id if keep_new else m.prediction_run_id
            b.warnings.append(f"prediction artifact {loser} superseded for {key[0]} {key[1]}")
            if not keep_new:
                continue
        sets[key] = (pset, lp)

    def order(item: tuple[tuple[int, str], tuple[PredictionSet, _LoadedPrediction]]) -> tuple[Any, ...]:
        pset = item[1][0]
        first = min((t.match_date or date.max for t in pset.target_matches), default=date.max)
        return (pset.season, first, item[0][1])

    entries: list[PredictionIndexEntry] = []
    current: tuple[str, PredictionSet, _LoadedPrediction] | None = None
    any_expired = False
    for (season, stage), (pset, lp) in sorted(sets.items(), key=order):
        path = f"predictions/{season}/{stage}.json"
        b.put_json(path, pset)
        entries.append(
            PredictionIndexEntry(
                season=season,
                stage_id=stage,
                stage_label=pset.stage_label,
                status=pset.status,
                resource=path,
                rows=len(pset.rows),
            )
        )
        if pset.status == "available" and current is None:
            current = (path, pset, lp)
        any_expired = any_expired or pset.status == "expired"
    if current is not None:
        path, pset, lp = current
        fc = _Forecast(
            "available", None, path, pset, pset.model, lp.model.bundle_id, lp.artifact.manifest.baseline_model_id
        )
    else:
        if any_expired:
            st: Literal["unavailable", "expired"] = "expired"
            reason = "prediction_targets_started_or_played"
        elif rejected:
            st, reason = "unavailable", "no_valid_prediction_artifact"
        elif unavailable_reasons:
            st, reason = "unavailable", sorted(unavailable_reasons)[0]
        else:
            scheduled = q.scalar("SELECT count(*) FROM matches WHERE status = 'scheduled'")
            st, reason = "unavailable", "no_prediction_artifact" if scheduled else "no_valid_future_fixture"
        fc = _Forecast(st, reason, None, None, None, None, None)
    b.put_json(
        "predictions/index.json",
        PredictionIndex(current=fc.current_path, status=fc.status, reason=fc.reason, sets=entries),
        "prediction_index",
    )
    b.counts["prediction_sets"] = len(entries)
    return fc


def _accuracy(b: _Build, inputs: ReleaseInputs, forecast: _Forecast, light: History | None) -> list[list[Any]]:
    from supercoach_via.ml.evaluate import legacy_unknown_summary, to_accuracy_report
    from supercoach_via.ml.predict import interval_info

    models = {m.bundle_id: m for m in inputs.model_manifests}
    reports: dict[str, AccuracyReport] = {}
    rows: list[list[Any]] = []
    for ev in sorted(inputs.evaluations, key=lambda e: e.evaluation_id):
        seasons_ = sorted({int(s) for s in ev.scored_rows["season"]}) if len(ev.scored_rows) else []
        season = seasons_[0] if len(seasons_) == 1 else None
        path = f"accuracy/{ev.model_id}/{season if season is not None else 'all'}-{ev.origin}.json"
        check_release_path(path)
        if path in reports:
            raise ValueError(f"two evaluations map to {path}; pool them or pass one")
        bm = models.get(ev.model_id)
        rows_path = "downloads/accuracy-rows.csv" if len(ev.scored_rows) else None
        reports[path] = to_accuracy_report(
            ev,
            season=season,
            promotion=bm.promotion_note if bm else "",
            interval_info=interval_info(_manifest_only(bm)) if bm else None,
            rows_resource=rows_path,
        )
        sr = ev.scored_rows
        base = sr["baseline_prediction"] if "baseline_prediction" in sr else [None] * len(sr)
        for r, bp in zip(sr.to_dict("records"), base, strict=True):
            pred, actual = float(r["predicted_disposals"]), float(r["actual"])
            rows.append(
                [
                    ev.evaluation_id,
                    ev.origin,
                    ev.model_id,
                    r["prediction_id"],
                    r["prediction_run_id"],
                    r["player_id"],
                    r["club_id"],
                    r["match_id"],
                    int(r["season"]),
                    r["stage_id"],
                    pred,
                    actual,
                    pred - actual,
                    abs(pred - actual),
                    None if bp is None or bp != bp else float(bp),
                ]
            )
    if light is not None and _legacy_prediction_rows(b):
        legacy = legacy_unknown_report(legacy_unknown_summary(light))
        if legacy is not None:
            reports["accuracy/legacy_unknown/all-legacy_unknown.json"] = legacy
    for path, rep in reports.items():
        b.put_json(path, rep)
    champion = forecast.model_id or next((m.bundle_id for m in inputs.model_manifests if m.promoted), None)
    card_model = models.get(champion) if champion else None
    if card_model is None and inputs.model_manifests:
        card_model = inputs.model_manifests[0]
    card = "No model bundle is attached to this release."
    if card_model is not None:
        from supercoach_via.ml.bundles import model_card_facts

        card = "\n".join(model_card_facts(card_model))
    baseline = forecast.baseline_model_id or next(
        (e.baseline_model_id for e in inputs.evaluations if e.baseline_model_id), None
    )
    b.put_json(
        "accuracy/index.json",
        AccuracyIndex(
            champion_model_id=champion,
            baseline_model_id=baseline,
            reports=[
                AccuracyIndexEntry(model_id=r.model_id, season=r.season, origin=r.origin, label=r.label, resource=p)
                for p, r in sorted(reports.items())
            ],
            model_card=card,
        ),
        "accuracy_index",
    )
    return rows


# ---------------------------------------------------------------------------
# Lists, articles, overview, quality
# ---------------------------------------------------------------------------


def _lists(b: _Build, q: SnapshotQuery) -> None:
    parts = []
    for table, col in (
        ("draft_events", "season"),
        ("contract_observations", "contract_end"),
        ("school_observations", "draft_year"),
    ):
        if _has(b.manifest, table):
            parts.append(f"SELECT {col} AS s FROM {table} WHERE {col} IS NOT NULL")  # noqa: S608 - constant identifiers
    seasons: list[int] = []
    if parts:
        seasons = [int(s) for (s,) in q.rows(f"SELECT DISTINCT s FROM ({' UNION '.join(parts)}) ORDER BY s")]  # noqa: S608 - constant identifiers
    res = {}
    for s in seasons:
        if not 1000 <= s <= 9999:
            continue
        path = f"lists/{s}.json"
        b.put_json(path, list_analytics.lists_season(q, s))
        res[str(s)] = path
    b.put_json("lists/index.json", ListsIndex(seasons=[int(k) for k in res], resources=res), "lists_index")


def _publish_articles(b: _Build, inputs: ReleaseInputs) -> list[ArticleSummary]:
    articles, assets = _articles(
        inputs,
        base=b.base,
        asset_prefix=f"data/{b.writer.release_id}/",
        generated_at=b.generated_at,
        snapshot_id=b.snapshot_id,
    )
    for rel, src in sorted(assets.items()):
        b.writer.put_bytes(rel, src.read_bytes())
    for a in articles:
        b.put_json(a.summary.resource, a)
    summaries = sorted(
        (a.summary for a in articles), key=lambda s: ((s.published or date.min).toordinal() * -1, s.slug)
    )
    b.put_json("articles/index.json", ArticleIndex(articles=summaries), "article_index")
    return summaries


def _latest_complete(b: _Build) -> MatchSummary | None:
    done = [m for m in b.summaries.values() if m.season == b.season and m.status == "complete"]
    return max(done, key=_chrono) if done else None


def _chrono(m: MatchSummary) -> tuple[Any, ...]:
    return (m.match_date or date.min, m.local_start or "", m.stage_order, m.replay_occurrence, m.match_id)


def _set_label(pset: PredictionSet) -> str:
    stype = pset.target_matches[0].stage_type if pset.target_matches else "other"
    return f"{pset.season} {stage_text(pset.stage_label, stype)}"


def _coverage_through(b: _Build) -> str | None:
    last = _latest_complete(b)
    return None if last is None else f"{last.season} {stage_text(last.stage_label, last.stage_type)}"


def _form_highlights(b: _Build, q: SnapshotQuery) -> list[LeaderRow]:
    rows = q.rows(
        """WITH g AS (
             SELECT player_id, disposals, row_number() OVER (
               PARTITION BY player_id ORDER BY match_date DESC NULLS LAST, match_id DESC) AS rn
             FROM player_games WHERE season = ?)
           SELECT g.player_id, p.display_name, avg(disposals), count(disposals)
           FROM g JOIN players p USING (player_id) WHERE rn <= 3
           GROUP BY ALL HAVING count(disposals) = 3 ORDER BY 3 DESC, g.player_id LIMIT 5""",
        [b.season],
    )
    return [
        LeaderRow(
            player_id=pid, name=name, stat="disposals (mean of last 3 games)", value=float(v), observed_games=int(n)
        )
        for pid, name, v, n in rows
    ]


def _overview(
    b: _Build, q: SnapshotQuery, fc: _Forecast, articles: list[ArticleSummary], leaders: dict[str, list[LeaderRow]]
) -> Overview:
    season_matches = sorted((m for m in b.summaries.values() if m.season == b.season), key=_chrono)
    checked = None
    season_active = None
    if _has(b.manifest, "seasons"):
        checked = q.scalar("SELECT max(fixture_checked_at) FROM seasons")
        sc = q.scalar("SELECT schedule_complete FROM seasons WHERE season = ?", [b.season])
        season_active = None if sc is None else not bool(sc)
    if isinstance(checked, datetime) and checked.tzinfo is None:
        checked = checked.replace(tzinfo=UTC)
    stale = False
    stale_reason: str | None = "source check time is not recorded in this snapshot"
    if isinstance(checked, datetime):
        age = b.generated_at - checked
        stale = age > STALE_AFTER
        stale_reason = f"sources last checked {age.days} days before this build" if stale else None
    last = _latest_complete(b)
    warnings = []
    if b.demo:
        warnings.append("DEMO release: synthetic data for testing the product; not AFL statistics.")
    if b.manifest.status.value == "legacy_unverified" and not b.demo:
        warnings.append("Imported from legacy CSV files; not yet independently re-verified against the sources.")
    if fc.status != "available":
        warnings.append(f"No current forecast: {fc.status} ({fc.reason}).")
    top = sorted(fc.current.rows, key=lambda r: (-r.predicted_disposals, r.prediction_id))[:5] if fc.current else []
    return Overview(
        release_id=b.writer.release_id,
        snapshot_id=b.snapshot_id,
        season=b.season,
        demo=b.demo,
        freshness=Freshness(
            source_checked_at=checked if isinstance(checked, datetime) else None,
            latest_completed_match_at=None,
            latest_completed_match_date=last.match_date if last else None,
            coverage_through=_coverage_through(b),
            generated_at=b.generated_at,
            published_at=None,
            validation_state="PASS" if _is_current(b) else "UNKNOWN",
            dataset_status="demo" if b.demo else b.manifest.status.value,
            season_active=season_active,
            stale=stale,
            stale_reason=stale_reason,
        ),
        next_fixture_status=fc.status,
        next_fixture_reason=fc.reason,
        upcoming=[m for m in season_matches if m.status == "scheduled"][:9],
        prediction_highlights=top,
        form_highlights=_form_highlights(b, q),
        recent_results=[m for m in reversed(season_matches) if m.status == "complete"][:6],
        latest_articles=articles[:2],
        leaders=[r for stat, _label in LEADER_STATS for r in leaders[stat]],
        model_status=ModelStatus(forecast_status=fc.status, reason=fc.reason, model=fc.model),
        warnings=warnings,
    )


def _is_current(b: _Build) -> bool:
    pointer = read_current(b.data_root)
    return pointer is not None and pointer.snapshot_id == b.snapshot_id


def _quality(b: _Build, q: SnapshotQuery, fc: _Forecast) -> QualityReport:
    issues = []
    if _has(b.manifest, "quality_issues"):
        for rule, sev, status, n in q.rows(
            "SELECT rule_id, severity, status, count(*) FROM quality_issues GROUP BY ALL ORDER BY 1, 2, 3"
        ):
            issues.append(
                QualityIssue(
                    rule_id=rule,
                    severity=sev,
                    count=int(n),
                    description=f"{n} row(s) flagged by rule {rule} (status: {status}).",
                )
            )
    status = "demo" if b.demo else b.manifest.status.value
    limitations = [
        "Missing statistics are shown as not recorded, never as zero; aggregates disclose observed games.",
        f"Forecast status for this release: {fc.status}" + (f" ({fc.reason})." if fc.reason else "."),
    ]
    if b.demo:
        limitations.insert(0, "DEMO: this release is synthetic and exists only to test the product.")
    if status == "legacy_unverified":
        limitations.append("Dataset imported from legacy CSV files; dates without fixture evidence are inferred.")
    quarantine = b.manifest.tables.get("quarantine")
    return QualityReport(
        dataset_status=status,
        table_counts={k: int(v.row_count) for k, v in sorted(b.manifest.tables.items())},
        quarantined_rows=int(quarantine.row_count) if quarantine is not None else 0,
        issues=issues,
        limitations=limitations,
        sources=[Source(label="Canonical snapshot", url=None, note=b.snapshot_id)],
    )


# ---------------------------------------------------------------------------
# Downloads: CSV/JSON exports, charts, Markdown reports, fan pack
# ---------------------------------------------------------------------------


def _jinja(template_dir: Path) -> Any:
    import jinja2

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        undefined=jinja2.StrictUndefined,
        autoescape=False,  # noqa: S701 - Markdown (not HTML) output; cell text is escaped by the md filter
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(
        md=_md,
        num=_num,
        dec=lambda v: "-" if v is None else f"{float(v):.1f}",
        pct=lambda v: "-" if v is None else f"{float(v):.1f}",
        share=lambda v: "-" if v is None else f"{100 * float(v):.0f}%",
        stage=stage_text,
    )
    return env


@dataclass(frozen=True)
class _Chart:
    key: str
    rel: str  # path relative to downloads/
    alt_text: str
    png: bytes
    title: str
    unit: str
    plotted: list[tuple[str, float | None]]


def _charts(b: _Build, fc: _Forecast, leaders: dict[str, list[LeaderRow]]) -> tuple[list[_Chart], dict[str, str]]:
    from supercoach_via.publish.charts import BarChartSpec, render_bar_chart

    specs: list[tuple[str, BarChartSpec | None, str]] = []
    disp = leaders.get("disposals", [])
    specs.append(
        (
            "season_disposal_leaders",
            BarChartSpec(
                title=f"{b.season} disposal leaders (season totals)",
                unit="disposals",
                labels=[r.name for r in disp],
                values=[r.value for r in disp],
            )
            if disp
            else None,
            f"no observed disposals in {b.season}",
        )
    )
    ladder = b.current_teams[0].ladder if b.current_teams else []
    specs.append(
        (
            "ladder_points",
            BarChartSpec(
                title=f"{b.season} ladder: premiership points",
                unit="points",
                labels=[r.name for r in ladder],
                values=[float(r.premiership_points) for r in ladder],
            )
            if ladder
            else None,
            f"no completed regular-season matches in {b.season}",
        )
    )
    top = sorted(fc.current.rows, key=lambda r: (-r.predicted_disposals, r.prediction_id))[:10] if fc.current else []
    specs.append(
        (
            "top_predictions",
            BarChartSpec(
                title=f"Top predicted disposals, {_set_label(fc.current) if fc.current else ''}".strip(", "),
                unit="disposals",
                labels=[r.player_name for r in top],
                values=[r.predicted_disposals for r in top],
            )
            if top
            else None,
            f"forecast {fc.status}: {fc.reason}",
        )
    )
    charts, missing = [], {}
    for key, spec, why in specs:
        rel = f"charts/{key.replace('_', '-')}.png"
        if spec is None:
            missing[rel] = why
            continue
        r = render_bar_chart(spec)
        charts.append(_Chart(key, rel, r.alt_text, r.png, spec.title, spec.unit, r.plotted))
    return charts, missing


def _downloads(
    b: _Build,
    inputs: ReleaseInputs,
    template_dir: Path,
    fc: _Forecast,
    overview: Overview,
    leaders: dict[str, list[LeaderRow]],
    pidx: PlayerIndex,
    player_rows: dict[str, _PlayerRow],
    names: dict[str, tuple[str, str]],
    history: _History,
    accuracy_rows: list[list[Any]],
    retained: list[RetainedRelease],
) -> None:
    as_of = f"{'DEMO ' if b.demo else ''}{_coverage_through(b) or b.season}"
    pack: dict[str, bytes] = {}
    labels: dict[str, str] = {}
    unavailable: dict[str, str] = {}

    def add(key: str, label: str, name: str, data: bytes, *, rows: int | None = None, in_pack: bool = True) -> None:
        rel = f"downloads/{name}"
        b.writer.put_bytes(rel, data)
        b.downloads.append(b.writer.download_item(key, label, rel, as_of=as_of, rows=rows))
        if in_pack:
            pack[name] = data
            labels[name] = label

    # players (rich)
    header = ["player_id", "name", "clubs", "first_season", "last_season", "career_games", "active"]
    for s in PLAYER_CSV_STATS:
        header += [f"{s}_total", f"{s}_observed_games"]
    prow = []
    for entry in pidx.players:
        pr = player_rows[entry.id]
        line: list[Any] = [
            pr.player_id,
            pr.name,
            pr.clubs,
            pr.first_season,
            pr.last_season,
            pr.career_games,
            "true" if pr.active else "false",
        ]
        for total, obs in pr.stats:
            line += [total, obs]
        prow.append(line)
    add(
        "players_csv",
        "Player directory with career totals (all rows)",
        "players.csv",
        safe_csv_bytes(header, prow),
        rows=len(prow),
    )

    # predictions: rich CSV (full precision), legacy 3-column CSV + sidecar manifest
    meta_forecast: dict[str, Any] = {
        "status": fc.status,
        "reason": fc.reason,
        "stage_label": None,
        "cutoff": None,
        "model_name": None,
        "rows": 0,
        "top": [],
    }
    if fc.current is not None:
        cur = fc.current
        fields = list(PredictionRow.model_fields)
        dumped = [r.model_dump(mode="json") for r in cur.rows]
        rows = [[";".join(d[f]) if f == "warnings" else d[f] for f in fields] for d in dumped]
        add(
            "predictions_csv",
            f"Current predictions, {_set_label(cur)} (all rows, full precision)",
            "predictions.csv",
            safe_csv_bytes(fields, rows),
            rows=len(rows),
        )
        legacy_rows, side_rows = [], []
        for i, r in enumerate(sorted(cur.rows, key=lambda r: (-r.predicted_disposals, r.prediction_id))):
            first, last = names.get(r.player_id, ("", ""))
            legacy_name = f"{last} {first}".strip() if first and last else r.player_name
            legacy_rows.append([legacy_name, r.club_name, round(r.predicted_disposals)])
            side_rows.append(
                {
                    "line": i + 2,
                    "player": legacy_name,
                    "team": r.club_name,
                    "prediction_id": r.prediction_id,
                    "player_id": r.player_id,
                    "club_id": r.club_id,
                    "match_id": r.match_id,
                    "predicted_disposals": r.predicted_disposals,
                }
            )
        add(
            "predictions_legacy_csv",
            "Legacy three-column forecast CSV (rounded display values)",
            "predictions-legacy.csv",
            safe_csv_bytes(["player", "team", "predicted_disposals"], legacy_rows),
            rows=len(legacy_rows),
        )
        side = {
            "kind": "legacy_forward_prediction_csv_sidecar",
            "legacy_file": "downloads/predictions-legacy.csv",
            "columns": ["player", "team", "predicted_disposals"],
            "display_rounding": "predicted_disposals rounded to the nearest integer for the legacy file only",
            "release_id": b.writer.release_id,
            "snapshot_id": b.snapshot_id,
            "prediction_run_id": cur.rows[0].prediction_run_id if cur.rows else None,
            "model_id": fc.model_id,
            "season": cur.season,
            "stage_id": cur.stage_id,
            "forecast_cutoff": _iso(cur.forecast_cutoff) if cur.forecast_cutoff else None,
            "rows": side_rows,
        }
        add(
            "predictions_legacy_manifest",
            "Sidecar manifest for the legacy forecast CSV (full IDs and precision)",
            "predictions-legacy.manifest.json",
            canonical_json_bytes(side),
        )
        meta_forecast.update(
            stage_label=_set_label(cur),
            cutoff=side["forecast_cutoff"],
            model_name=cur.model.name if cur.model else fc.model_id,
            rows=len(cur.rows),
            top=overview.prediction_highlights,
        )
    else:
        why = f"forecast {fc.status}: {fc.reason}"
        for name in ("predictions.csv", "predictions-legacy.csv", "predictions-legacy.manifest.json"):
            unavailable[name] = why

    # accuracy rows
    if accuracy_rows:
        acc_header = [
            "evaluation_id",
            "origin",
            "model_id",
            "prediction_id",
            "prediction_run_id",
            "player_id",
            "club_id",
            "match_id",
            "season",
            "stage_id",
            "predicted_disposals",
            "actual",
            "error",
            "abs_error",
            "baseline_prediction",
        ]
        add(
            "accuracy_rows_csv",
            "Scored accuracy rows (all rows, every evaluation in this release)",
            "accuracy-rows.csv",
            safe_csv_bytes(acc_header, accuracy_rows),
            rows=len(accuracy_rows),
        )
    else:
        unavailable["accuracy-rows.csv"] = "no evaluation attached to this release"

    # both legacy all-time shapes
    with SnapshotQuery(b.data_root, b.manifest) as q:
        bio_rows = ranking_analytics.biography_export_rows(q, history.ranking)
    add(
        "all_time_biography_csv",
        "All-time top 100, legacy biography shape (root all_time_top_100.csv)",
        "all-time-top-100.csv",
        _quote_all_csv(["Serial Number", "Player Name", "Footy Teams", "Comment"], bio_rows),
        rows=len(bio_rows),
    )
    numeric = ranking_analytics.numeric_export_rows(history.ranking)
    add(
        "all_time_numeric_csv",
        "All-time top 100, legacy numeric ranking shape (data/top100)",
        "all-time-top-100-scores.csv",
        safe_csv_bytes(["player", "all_time_score"], numeric),
        rows=len(numeric),
    )

    # yearly top 100, legacy data/top100/yearly shape, only once a season is final (legacy cadence)
    final_seasons = [s for s, y in history.ranking.yearly.items() if not y.provisional]
    if final_seasons:
        season = max(final_seasons)
        yearly_rows = ranking_analytics.yearly_export_rows(history.ranking.yearly[season])
        add(
            "yearly_top100_csv",
            f"Top 100 of the {season} season, legacy yearly shape (final season)",
            f"yearly-top-100-{season}.csv",
            safe_csv_bytes(["player", "score", "percentile_rank", "games_played"], yearly_rows),
            rows=len(yearly_rows),
        )
    for s in sorted(history.ranking.provisional_seasons):
        unavailable[f"yearly-top-100-{s}.csv"] = f"{s} season not final; yearly CSV is published at season end"

    # era summaries and the Brownlow proxy (labelled analytics; never presented as observed votes)
    import math

    from supercoach_via.analytics import awards as award_analytics

    def _cell(v: Any) -> Any:
        return None if isinstance(v, float) and math.isnan(v) else v

    era_cols = ["era", "metric", "legacy_metric", "n_player_games", "n_with_metric", "mean_per_game", "std_per_game",
                "median_per_game", "mean_per_100pct_played", "recorded_from", "recording_status"]  # fmt: skip
    with SnapshotQuery(b.data_root, b.manifest) as q:
        q.con.execute("SET threads TO 1")  # parallel float aggregation is not bit-reproducible
        era_df = era_analytics.era_stats(q)
        proxy = award_analytics.brownlow_proxy(q, b.season)
    era_rows = [[_cell(r[c]) for c in era_cols] for r in era_df.to_dict("records")]
    add("era_summary_csv", "Era summaries per stat (observed games; unrecorded stats are blank, not zero)",
        "era-summary.csv", safe_csv_bytes(era_cols, era_rows), rows=len(era_rows))  # fmt: skip
    if proxy.rows:
        bl_cols = ["rank", "player_id", "name", "club", "games", "disposals_pg", "clearances_pg",
                   "contested_possessions_pg", "goals_pg", "effective_disposals_pg", "tackles_pg", "proxy_per_game",
                   "season_proxy_scaled", "observed_votes", "votes_observed_games", "ineligible",
                   "ineligible_reason", "ineligible_source"]  # fmt: skip
        bl_rows = [[*(getattr(r, c) for c in bl_cols), proxy.label, proxy.version, proxy.scale_label]
                   for r in proxy.rows]  # fmt: skip
        add("brownlow_proxy_csv", f"{proxy.label} {proxy.season} ({proxy.version}; an index, not predicted votes)",
            f"brownlow-proxy-{proxy.season}.csv",
            safe_csv_bytes([*bl_cols, "label", "version", "scale"], bl_rows), rows=len(bl_rows))  # fmt: skip
    else:
        unavailable[f"brownlow-proxy-{b.season}.csv"] = f"no player met the {proxy.min_games}-game minimum"

    # charts (values and alt text from the same spec)
    charts, missing_charts = _charts(b, fc, leaders)
    unavailable.update(missing_charts)
    for c in charts:
        add(f"chart_{c.key}", c.title, c.rel, c.png)
    chart_doc = {
        "release_id": b.writer.release_id,
        "snapshot_id": b.snapshot_id,
        "charts": [
            {
                "key": c.key,
                "status": "available",
                "path": f"downloads/{c.rel}",
                "title": c.title,
                "unit": c.unit,
                "alt_text": c.alt_text,
                "plotted": [[lbl, v] for lbl, v in c.plotted],
            }
            for c in charts
        ]
        + [
            {"key": Path(rel).stem.replace("-", "_"), "status": "unavailable", "reason": why}
            for rel, why in sorted(missing_charts.items())
        ],
    }
    add("charts_manifest", "Chart values and alt text (JSON)", "charts.json", canonical_json_bytes(chart_doc))
    by_key = {c.key: c for c in charts}

    # Markdown compatibility reports
    env = _jinja(template_dir)
    meta = {
        "release_id": b.writer.release_id,
        "snapshot_id": b.snapshot_id,
        "season": b.season,
        "coverage_through": _coverage_through(b),
        "generated_at": _iso(b.generated_at),
        "demo": b.demo,
        "dataset_status": "demo" if b.demo else b.manifest.status.value,
    }
    groups = [
        {
            "title": f"{label} ({b.season} season totals)",
            "value_header": f"{label} (season total)",
            "rows": list(zip(_competition_ranks([r.value for r in leaders[stat]]), leaders[stat], strict=True)),
        }
        for stat, label in LEADER_STATS
    ]
    ctx: dict[str, dict[str, Any]] = {
        "season-summary": {
            "forecast": meta_forecast,
            "recent": overview.recent_results,
            "upcoming": overview.upcoming,
            "chart": by_key.get("top_predictions"),
        },
        "stat-leaders": {"leader_groups": groups, "chart": by_key.get("season_disposal_leaders")},
        "team-analysis": {
            "ladder": b.current_teams[0].ladder if b.current_teams else [],
            "ladder_note": team_analytics.LADDER_METHOD,
            "teams": b.current_teams,
            "chart": by_key.get("ladder_points"),
        },
    }
    for name, title in REPORTS.items():
        body = env.get_template(f"{name}.md.j2").render(meta=meta, **ctx[name]).strip("\n")
        overlay = inputs.report_overlays.get(name)
        if overlay is not None:
            doc = replace_marked_section(overlay, name, body)
        else:
            doc = env.get_template("document.md.j2").render(
                title=title,
                name=name,
                body=body,
                intro="Generated from the release view models; the section between GEN markers is regenerated "
                "on every build and curated text outside it is preserved.",
            )
        add(f"report_{name.replace('-', '_')}", f"{title} report (Markdown)", f"reports/{name}.md", doc.encode())

    # README and fan pack
    files = [{"name": n, "label": labels[n]} for n in sorted(pack)]
    readme = (
        env.get_template("readme.md.j2")
        .render(
            meta=meta,
            forecast=meta_forecast,
            files=files,
            unavailable=[{"member": m, "reason": r} for m, r in sorted(unavailable.items())],
        )
        .encode()
    )
    add("readme", "Release README", "README.md", readme)
    required = [
        "README.md",
        "players.csv",
        "all-time-top-100.csv",
        "all-time-top-100-scores.csv",
        "charts.json",
        *(f"reports/{n}.md" for n in REPORTS),
    ]
    zip_bytes = build_fan_pack(
        pack,
        required=required,
        unavailable=unavailable,
        meta={
            "kind": "supercoach_via_fan_pack",
            "release_id": b.writer.release_id,
            "snapshot_id": b.snapshot_id,
            "generated_at": meta["generated_at"],
            "demo": b.demo,
            "dataset_status": meta["dataset_status"],
            "forecast_status": fc.status,
            "forecast_reason": fc.reason,
            "forecast_cutoff": meta_forecast["cutoff"],
        },
    )
    add(
        "fan_pack",
        "Fan pack ZIP (README, manifest, checksums, CSVs, reports, charts)",
        "fan-pack.zip",
        zip_bytes,
        in_pack=False,
    )
    b.put_json(
        "downloads.json", Downloads(release_id=b.writer.release_id, items=b.downloads, retained=retained), "downloads"
    )


def _quote_all_csv(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> bytes:
    """Legacy root-CSV shape: every field quoted (matches ``all_time_top_100.csv``)."""
    import csv
    import io

    from supercoach_via.publish.bundle import neutralise_cell

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_ALL)
    w.writerow([neutralise_cell(h) for h in header])
    for row in rows:
        w.writerow(["" if v is None else neutralise_cell(v) for v in row])
    return buf.getvalue().encode("utf-8")
