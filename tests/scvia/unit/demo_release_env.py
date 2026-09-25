"""Shared end-to-end fixture for the release builder tests (test helper, not a test module).

Demo path: write_demo_corpus -> import_legacy -> validate_dataset -> promote, then a small
model is trained on the promoted snapshot, one prospective forecast is archived, a replay
evaluation is scored, and tiny curated-article / live-snapshot inputs are written. Every
root is under the pytest temp directory; nothing reads the real ``data/`` corpus.

The demo corpus validates under the DEFAULT policy (player goals/behinds are allocated
from the team totals), so the snapshot is promoted exactly as the operator path would.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from supercoach_via.demo import write_demo_corpus
from supercoach_via.domain.schemas import CheckOutcome, Origin
from supercoach_via.editorial.verify import EditorialOutcome, NumericCheck
from supercoach_via.ingest.legacy import import_legacy
from supercoach_via.ingest.reconcile import load_policy, validate_dataset
from supercoach_via.ml.bundles import BundleManifest
from supercoach_via.ml.evaluate import EvaluationArtifact
from supercoach_via.publish.builder import EditorialDraft, ReleaseCandidate, ReleaseInputs, build_release
from supercoach_via.publish.view_models import LivePlayerRow, LiveSnapshot, TeamScore
from supercoach_via.settings import RunContext, Settings
from supercoach_via.storage.snapshots import SnapshotRef, promote

IMPORT_CLOCK = datetime(2026, 5, 1, 6, 0, tzinfo=UTC)
RELEASE_CLOCK = datetime(2026, 5, 2, 9, 30, tzinfo=UTC)
FORECAST_CUTOFF = datetime(2026, 5, 1, tzinfo=UTC)
# 1x1 transparent PNG
PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)
ARTICLE_MD = """# Demo curated article

<!-- verify-asof: 2026-05-01 -->

This DEMO article is curated editorial content. It is imported unchanged.

![Demo chart](../../assets/demo-chart.png)

<script>alert(1)</script>
"""


@dataclass
class DemoEnv:
    root: Path
    data_root: Path
    snapshot: SnapshotRef
    prediction_dir: Path
    model_manifest: BundleManifest
    evaluation: EvaluationArtifact
    content_root: Path
    content_manifest: Path
    live_root: Path
    live_match_id: str
    target_stage: str


_CACHE: dict[Path, DemoEnv] = {}
_BUILT: dict[Path, ReleaseCandidate] = {}


def context_for(env: DemoEnv, output_root: Path) -> RunContext:
    settings = Settings(data_root=env.data_root, output_root=output_root, source_root=env.root / "src")
    return RunContext(settings=settings, clock=lambda: RELEASE_CLOCK)


def full_inputs(env: DemoEnv, **overrides: Any) -> ReleaseInputs:
    base: dict[str, Any] = dict(
        prediction_dirs=(env.prediction_dir,),
        model_manifests=(env.model_manifest,),
        evaluations=(env.evaluation,),
        content_root=env.content_root,
        content_manifest=env.content_manifest,
        live_root=env.live_root,
        editorial=(
            EditorialDraft(
                slug="demo-generated-note",
                title="DEMO generated note",
                category="analysis",
                outcome=EditorialOutcome(
                    "publishable",
                    "draft",
                    "# DEMO generated note\n\nGenerated from claim-checked evidence.\n",
                    NumericCheck(outcome=CheckOutcome.PASS, numbers_checked=0),
                    None,
                ),
            ),
            EditorialDraft(
                slug="demo-failed-note",
                title="DEMO failed note",
                category="analysis",
                outcome=EditorialOutcome(
                    "unpublished_draft",
                    "draft",
                    None,
                    NumericCheck(outcome=CheckOutcome.FAIL, numbers_checked=1),
                    None,
                    ["numeric check failed"],
                ),
            ),
        ),
        demo=True,
    )
    base.update(overrides)
    return ReleaseInputs(**base)


def _write_content(root: Path) -> tuple[Path, Path]:
    news = root / "docs" / "news"
    news.mkdir(parents=True)
    (news / "2026-05-01-demo-article.md").write_text(ARTICLE_MD, encoding="utf-8")
    (root / "assets").mkdir()
    (root / "assets" / "demo-chart.png").write_bytes(PNG_1PX)
    manifest = root / "public_content.toml"
    manifest.write_text(
        'schema_version = 1\n\n[[article]]\npath = "docs/news/2026-05-01-demo-article.md"\n'
        'category = "news"\nscope = "frozen"\n',
        encoding="utf-8",
    )
    return root, manifest


def _write_live(root: Path, match_id: str) -> None:
    game = root / "demo-live-1"
    (game / "snapshots").mkdir(parents=True)
    digest = "a" * 64
    snap = LiveSnapshot(
        source_game_id="demo-live-1",
        match_id=match_id,
        fetched_at=datetime(2026, 5, 1, 5, 0, tzinfo=UTC),
        status="in_progress",
        quarter="Q2",
        home=TeamScore(club_id="src:demo_a", name="Demo A", goals=3, behinds=2, score=20),
        away=TeamScore(club_id="src:demo_b", name="Demo B", goals=2, behinds=1, score=13),
        reliable_fields=["disposals"],
        unavailable_fields=["goals"],
        players=[LivePlayerRow(player_id="fanfooty:1", name="Demo Live", stats={"disposals": 7.0})],
        timeline=[],
        reads=["DEMO read"],
        anomalies=[],
        final=False,
    )
    (game / "snapshots" / f"{digest}.json").write_text(snap.model_dump_json(), encoding="utf-8")
    state = {
        "source_game_id": "demo-live-1",
        "match_id": match_id,
        "accepted": [digest],
        "accepted_at": ["2026-05-01T05:00:00+00:00"],
        "label": "DEMO live match",
    }
    (game / "state.json").write_text(json.dumps(state), encoding="utf-8")


def demo_env(base: Path) -> DemoEnv:
    if base in _CACHE:
        return _CACHE[base]
    from supercoach_via.ml import evaluate as E
    from supercoach_via.ml import features as F
    from supercoach_via.ml import predict as P
    from supercoach_via.ml import train as T

    root = base / "demo-env"
    write_demo_corpus(root / "src")
    settings = Settings(data_root=root / "var", output_root=root / "dist", source_root=root / "src")
    ctx = RunContext(settings=settings, clock=lambda: IMPORT_CLOCK)
    cand = import_legacy(root / "src", ctx)
    report = validate_dataset(cand, load_policy())
    assert report.ok, report.issues[:3]
    ref = promote(cand.data_root, cand.candidate, report)

    hist = F.load_history(root / "var", ref.snapshot_id)
    cfg = T.TrainingConfig(
        train_cutoff=date(2025, 6, 1),
        calibration_end=date(2026, 1, 1),
        target_seasons_from=2024,
        candidates=("hgb",),
        n_folds=2,
        threads=1,
        min_calibration=50,
        cohort_min_n=20,
        params={"hgb": {"max_iter": 10, "learning_rate": 0.1, "max_leaf_nodes": 7, "min_samples_leaf": 5}},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trained = T.train_model(hist, cfg, bundle_root=root / "models", clock=lambda: IMPORT_CLOCK)
        req = P.ForecastRequest(forecast_cutoff=FORECAST_CUTOFF, generated_at=IMPORT_CLOCK, origin=Origin.PROSPECTIVE)
        art = P.forecast(hist, trained.bundle, req)
        assert art.manifest.status == "available"
        pdir = P.write_artifact(art, root / "predictions")
        _arts, ev = E.replay(
            hist, trained.bundle, season=2026, generated_at=IMPORT_CLOCK, stage_ids=("r08", "r09"), min_cohort=20
        )
    content_root, content_manifest = _write_content(root / "content")
    live_match = str(art.rows["match_id"].iloc[0])
    _write_live(root / "live", live_match)
    env = DemoEnv(
        root=root,
        data_root=root / "var",
        snapshot=ref,
        prediction_dir=pdir,
        model_manifest=trained.bundle.manifest,
        evaluation=ev,
        content_root=content_root,
        content_manifest=content_manifest,
        live_root=root / "live",
        live_match_id=live_match,
        target_stage=str(art.manifest.stage_id),
    )
    _CACHE[base] = env
    return env


def full_release(base: Path) -> ReleaseCandidate:
    """The complete demo release (every optional input), built once per test session."""
    if base not in _BUILT:
        env = demo_env(base)
        _BUILT[base] = build_release(env.snapshot, full_inputs(env), context_for(env, base / "dist-full"))
    return _BUILT[base]
