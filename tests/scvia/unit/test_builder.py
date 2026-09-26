"""Release builder: complete staged release from one snapshot (+ optional ML/editorial inputs).

Covers the PLAN 8.1 resource table, forecast status, downloads/fan pack, Markdown reports,
R05-R07 (hash/incomplete rejection, no mtime selection, determinism/reuse), S04/S05
(allowlist: no private/operator/model content), and the team-page operation budget.

Builds are shared to keep the fast tier fast: ``built`` (every optional input, shared with
test_reconcile_outputs), ``bare`` (no optional inputs), ``variant`` (rejected predictions,
a report overlay, instrumented ladder) and one determinism rebuild.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import zipfile
from pathlib import Path
from dataclasses import replace
from typing import Any

import pytest

from supercoach_via.domain.schemas import CheckOutcome
from supercoach_via.publish import builder as B
from supercoach_via.publish.release import validate_release
from supercoach_via.publish.reports import MarkerError
from tests.scvia.unit.demo_release_env import (
    RELEASE_CLOCK,
    DemoEnv,
    context_for,
    demo_env,
    full_inputs,
    full_release,
)

CURATED = (
    "# Stat leaders\n\nCurated intro kept verbatim.\n\n"
    "<!-- GEN:stat-leaders START -->\nold\n<!-- GEN:stat-leaders END -->\n\nCurated footer.\n"
)


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> DemoEnv:
    return demo_env(tmp_path_factory.getbasetemp())


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> B.ReleaseCandidate:
    return full_release(tmp_path_factory.getbasetemp())


@pytest.fixture(scope="module")
def bare(env: DemoEnv, tmp_path_factory: pytest.TempPathFactory) -> B.ReleaseCandidate:
    out = tmp_path_factory.mktemp("dist-bare")
    return B.build_release(env.snapshot, B.ReleaseInputs(demo=True), context_for(env, out))


@pytest.fixture(scope="module")
def variant(env: DemoEnv, tmp_path_factory: pytest.TempPathFactory) -> tuple[B.ReleaseCandidate, list[int]]:
    """Tampered + manifest-less predictions, a curated overlay, and a counting ladder."""
    from supercoach_via.analytics import teams

    root = tmp_path_factory.mktemp("variant")
    bad = root / "bad-run"
    shutil.copytree(env.prediction_dir, bad)
    rows = next(p for p in bad.iterdir() if p.suffix == ".parquet")
    rows.chmod(0o644)
    rows.write_bytes(rows.read_bytes() + b"x")
    calls: list[int] = []
    real = teams.ladder

    def counting(q: Any, season: int, **kw: Any) -> Any:
        calls.append(season)
        return real(q, season, **kw)

    mp = pytest.MonkeyPatch()
    mp.setattr(teams, "ladder", counting)
    try:
        inputs = B.ReleaseInputs(
            demo=True,
            prediction_dirs=(bad, env.prediction_dir),
            model_manifests=(),
            report_overlays={"stat-leaders": CURATED},
        )
        cand = B.build_release(env.snapshot, inputs, context_for(env, root / "out"))
    finally:
        mp.undo()
    return cand, calls


def _json(cand: B.ReleaseCandidate, rel: str) -> Any:
    return json.loads((cand.public_dir / rel).read_text(encoding="utf-8"))


def _files(cand: B.ReleaseCandidate) -> list[str]:
    return sorted(p.relative_to(cand.public_dir).as_posix() for p in cand.public_dir.rglob("*") if p.is_file())


# ---------------------------------------------------------------------------
# Resource table (PLAN 8.1) and layout contract (web/tests/fixtures/demo-release)
# ---------------------------------------------------------------------------


def test_release_validates_and_is_listed(built: B.ReleaseCandidate) -> None:
    assert built.ok and built.validation.outcome is CheckOutcome.PASS, built.validation.issues[:5]
    assert built.release_dir.name == built.release_id
    assert json.loads((built.release_dir / "validation.json").read_text())["outcome"] == "PASS"
    assert not built.reused
    assert built.release_id.startswith(RELEASE_CLOCK.strftime("%Y%m%dT%H%M%SZ") + "-demo-")


def test_every_resource_family_is_produced(built: B.ReleaseCandidate, env: DemoEnv) -> None:
    files = set(_files(built))
    for fixed in (
        "release.json",
        "overview.json",
        "predictions/index.json",
        "players/index.json",
        "teams/index.json",
        "history/index.json",
        "accuracy/index.json",
        "lists/index.json",
        "live/index.json",
        "articles/index.json",
        "quality.json",
        "downloads.json",
    ):
        assert fixed in files, fixed
    assert f"predictions/2026/{env.target_stage}.json" in files
    assert {f"matches/{s}/index.json" for s in (2024, 2025, 2026)} <= files
    assert any(f.startswith("matches/detail/") for f in files)
    assert any(f.startswith("players/legacy__") for f in files)
    assert any(f.startswith("player-games/legacy__") and f.endswith("/2026.json") for f in files)
    assert {f"teams/demo_harbour/{s}.json" for s in (2024, 2025, 2026)} <= files
    assert "live/demo-live-1/latest.json" in files
    assert {"articles/2026-05-01-demo-article.json", "articles/demo-generated-note.json"} <= files
    assert "articles/demo-failed-note.json" not in files  # unpublished drafts never ship
    assert any(f.startswith("history/career_disposals_total/") for f in files)
    assert any(f.startswith("accuracy/") and f.endswith("-replay.json") for f in files)

    manifest = _json(built, "release.json")
    keys = set(manifest["resources"])
    assert {
        "overview",
        "prediction_index",
        "player_index",
        "team_index",
        "history_index",
        "accuracy_index",
        "lists_index",
        "live_index",
        "article_index",
        "quality",
        "downloads",
    } <= keys
    assert {"match_index:2024", "match_index:2025", "match_index:2026"} <= keys  # web matchSeasons() contract
    assert manifest["snapshot_id"] == env.snapshot.snapshot_id
    assert manifest["demo"] is True and manifest["base_label"] == "DEMO"
    assert manifest["coverage"]["status"] == "demo"


def test_forecast_available_with_full_precision(built: B.ReleaseCandidate, env: DemoEnv) -> None:
    from supercoach_via.ml.predict import load_artifact

    manifest = _json(built, "release.json")
    path = f"predictions/2026/{env.target_stage}.json"
    assert manifest["forecast"] == {
        "status": "available",
        "reason": None,
        "artifact": path,
        "model_id": env.model_manifest.bundle_id,
    }
    pset = _json(built, path)
    art = load_artifact(env.prediction_dir)
    assert len(pset["rows"]) == len(art.rows) > 0
    by_id = {r["prediction_id"]: r["predicted_disposals"] for r in pset["rows"]}
    for r in art.rows.itertuples():
        assert by_id[r.prediction_id] == float(r.predicted_disposals)  # exact, not rounded
    idx = _json(built, "predictions/index.json")
    assert idx["current"] == path and idx["status"] == "available"
    # target matches come from the same match summaries as the match index
    mindex = {m["match_id"]: m for m in _json(built, "matches/2026/index.json")["matches"]}
    for m in pset["target_matches"]:
        assert m == mindex[m["match_id"]]
    # the player detail carries that player's current forecast row
    row = pset["rows"][0]
    detail = _json(built, f"players/{row['player_id'].replace(':', '__')}.json")
    assert detail["forecast"] == row


def test_no_prediction_inputs_still_builds_with_honest_status(bare: B.ReleaseCandidate, built: Any) -> None:
    assert bare.ok, bare.validation.issues[:5]
    assert bare.release_id != built.release_id  # different inputs, different release
    manifest = _json(bare, "release.json")
    assert manifest["forecast"] == {
        "status": "unavailable",
        "reason": "no_prediction_artifact",
        "artifact": None,
        "model_id": None,
    }
    ov = _json(bare, "overview.json")
    assert ov["next_fixture_status"] == "unavailable" and ov["prediction_highlights"] == []
    assert _json(bare, "predictions/index.json")["sets"] == []
    assert _json(bare, "live/index.json")["matches"] == [] and _json(bare, "articles/index.json")["articles"] == []
    keys = {i["key"] for i in _json(bare, "downloads.json")["items"]}
    assert "predictions_csv" not in keys and "players_csv" in keys and "fan_pack" in keys
    with zipfile.ZipFile(bare.public_dir / "downloads/fan-pack.zip") as zf:
        pack = json.loads(zf.read("manifest.json"))
    assert {"predictions.csv", "accuracy-rows.csv"} <= {u["member"] for u in pack["unavailable"]}


def test_rejected_prediction_artifacts_are_not_published(variant: tuple[B.ReleaseCandidate, list[int]]) -> None:
    cand, _calls = variant
    assert cand.ok
    manifest = _json(cand, "release.json")
    assert manifest["forecast"]["status"] == "unavailable"
    assert manifest["forecast"]["reason"] == "no_valid_prediction_artifact"
    assert not any(p.startswith("predictions/2026/") for p in _files(cand))
    assert any("rejected: ArtifactIntegrityError" in w for w in cand.warnings), cand.warnings
    assert any("model manifest" in w for w in cand.warnings), cand.warnings


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (["scheduled", "scheduled"], "available"),
        (["scheduled", "complete"], "expired"),
        (["in_progress"], "expired"),
        (["scheduled", None], None),  # target missing from the snapshot -> rejected
        ([], None),
    ],
)
def test_classify_prediction_targets(statuses: list[str | None], expected: str | None) -> None:
    assert B.classify_prediction_targets(statuses) == expected


def test_legacy_unknown_report_is_descriptive_only() -> None:
    assert B.legacy_unknown_report({"metrics": None}) is None
    summary = {
        "label": "legacy_unknown archive",
        "rows": 10,
        "joined": 7,
        "unjoined_reasons": {"unresolved_identity": 3},
        "metrics": {"n": 7, "mae": 3.5, "rmse": 4.0, "bias": -0.5, "median_ae": 3.0, "within_5": 0.7, "within_10": 1.0},
        "note": "rounded integers",
    }
    rep = B.legacy_unknown_report(summary)
    assert rep is not None and rep.origin == "legacy_unknown" and rep.headline.mae == 3.5
    assert rep.populations.missing == 3 and rep.populations.exclusion_reasons == {"unresolved_identity": 3}
    assert "never merged" in rep.promotion and rep.rows_resource is None


# ---------------------------------------------------------------------------
# Other public resources
# ---------------------------------------------------------------------------


def test_overview_quality_and_team_index(built: B.ReleaseCandidate, env: DemoEnv) -> None:
    ov = _json(built, "overview.json")
    manifest = _json(built, "release.json")
    assert ov["release_id"] == manifest["release_id"] and ov["snapshot_id"] == env.snapshot.snapshot_id
    assert ov["season"] == 2026 and ov["demo"] is True
    assert ov["freshness"]["generated_at"].startswith("2026-05-02T09:30")
    assert ov["freshness"]["validation_state"] == "PASS"  # the promoted current snapshot
    assert ov["freshness"]["latest_completed_match_date"] is not None
    assert all(m["status"] == "scheduled" for m in ov["upcoming"]) and ov["upcoming"]
    assert all(m["status"] == "complete" for m in ov["recent_results"]) and ov["recent_results"]
    assert ov["leaders"] and ov["prediction_highlights"] and ov["form_highlights"]
    assert [a["slug"] for a in ov["latest_articles"]] == ["demo-generated-note", "2026-05-01-demo-article"]
    assert any("DEMO" in w for w in ov["warnings"])
    q = _json(built, "quality.json")
    assert q["dataset_status"] == "demo"
    assert q["table_counts"]["matches"] == 95
    assert {i["rule_id"] for i in q["issues"]} >= {"venue_unmapped"}
    teams = {t["club_id"]: t for t in _json(built, "teams/index.json")["teams"]}
    assert teams["demo_harbour"]["seasons"] == [2024, 2025, 2026]
    assert "adelaide" not in teams  # clubs with no matches in the snapshot get no pages
    ts = _json(built, "teams/demo_harbour/2026.json")
    assert ts["club"]["club_id"] == "demo_harbour" and ts["ladder"] and ts["heuristics"] and ts["leaders"]
    assert [r["club_id"] for r in ts["five_year"]] == ["demo_harbour", "demo_harbour"]  # 2024, 2025


def test_live_and_match_detail_link(built: B.ReleaseCandidate, env: DemoEnv) -> None:
    live = _json(built, "live/index.json")
    assert [m["resource"] for m in live["matches"]] == ["live/demo-live-1/latest.json"]
    detail = _json(built, f"matches/detail/{env.live_match_id.replace(':', '__')}.json")
    assert detail["live_snapshots"] == ["live/demo-live-1/latest.json"]


def test_articles_are_sanitized_and_assets_contained(built: B.ReleaseCandidate) -> None:
    art = _json(built, "articles/2026-05-01-demo-article.json")
    assert "<script" not in art["html"]
    assert art["summary"]["editorial_state"] == "published_archive"
    assert f"/data/{built.release_id}/assets/demo-chart.png" in art["html"]
    assert (built.public_dir / "assets/demo-chart.png").is_file()
    gen = _json(built, "articles/demo-generated-note.json")
    assert gen["summary"]["editorial_state"] == "generated"


def test_article_manifest_cannot_publish_operator_docs(env: DemoEnv, tmp_path: Path) -> None:
    (tmp_path / "docs" / "rewrite").mkdir(parents=True)
    (tmp_path / "docs" / "rewrite" / "PLAN.md").write_text("# private plan\n")
    manifest = tmp_path / "m.toml"
    manifest.write_text(
        'schema_version = 1\n[[article]]\npath = "docs/rewrite/PLAN.md"\ncategory = "x"\nscope = "frozen"\n'
    )
    with pytest.raises(ValueError, match="operator/agent"):
        B.build_release(
            env.snapshot,
            B.ReleaseInputs(content_root=tmp_path, content_manifest=manifest),
            context_for(env, tmp_path / "out"),
        )
    assert not (tmp_path / "out" / "releases").exists()  # refused before anything is staged


# ---------------------------------------------------------------------------
# Downloads, legacy shapes, fan pack, reports
# ---------------------------------------------------------------------------


def test_downloads_list_every_asset_with_checksums(built: B.ReleaseCandidate) -> None:
    dl = _json(built, "downloads.json")
    assert dl["release_id"] == built.release_id
    items = {i["key"]: i for i in dl["items"]}
    for key in (
        "players_csv",
        "predictions_csv",
        "predictions_legacy_csv",
        "predictions_legacy_manifest",
        "accuracy_rows_csv",
        "all_time_numeric_csv",
        "all_time_biography_csv",
        "charts_manifest",
        "report_season_summary",
        "report_stat_leaders",
        "report_team_analysis",
        "fan_pack",
        "readme",
    ):
        assert key in items, key
    assert any(k.startswith("chart_") and items[k]["kind"] == "png" for k in items)
    for item in dl["items"]:
        data = (built.public_dir / item["path"]).read_bytes()
        assert item["bytes"] == len(data) and item["sha256"] == hashlib.sha256(data).hexdigest()
        assert item["as_of"].startswith("DEMO 2026 Round")
    assert items["players_csv"]["rows"] == _json(built, "players/index.json")["count"]


def test_legacy_shapes(built: B.ReleaseCandidate) -> None:
    text = (built.public_dir / "downloads/all-time-top-100.csv").read_text()
    assert text.splitlines()[0] == '"Serial Number","Player Name","Footy Teams","Comment"'
    assert (built.public_dir / "downloads/all-time-top-100-scores.csv").read_text().splitlines()[0] == (
        "player,all_time_score"
    )
    legacy = list(csv.reader(io.StringIO((built.public_dir / "downloads/predictions-legacy.csv").read_text())))
    assert legacy[0] == ["player", "team", "predicted_disposals"]
    assert all(r[2].isdigit() for r in legacy[1:]) and len(legacy) > 1
    assert legacy[1][0].startswith("Player")  # legacy "Surname First" order
    side = _json(built, "downloads/predictions-legacy.manifest.json")
    assert side["kind"] == "legacy_forward_prediction_csv_sidecar"
    assert len(side["rows"]) == len(legacy) - 1
    for line, row in zip(legacy[1:], side["rows"], strict=True):
        assert int(line[2]) == round(row["predicted_disposals"]) and row["prediction_id"] and row["match_id"]


def test_fan_pack_is_complete_and_checksummed(built: B.ReleaseCandidate) -> None:
    with zipfile.ZipFile(built.public_dir / "downloads/fan-pack.zip") as zf:
        names = set(zf.namelist())
        assert {"README.md", "manifest.json", "checksums.sha256", "players.csv", "reports/stat-leaders.md"} <= names
        for info in zf.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
        sums = {}
        for line in zf.read("checksums.sha256").decode().splitlines():
            digest, name = line.split("  ", 1)
            sums[name] = digest
        for name, digest in sums.items():
            assert hashlib.sha256(zf.read(name)).hexdigest() == digest
        assert set(sums) == names - {"checksums.sha256"}
        pack = json.loads(zf.read("manifest.json"))
    assert pack["release_id"] == built.release_id and pack["kind"] == "supercoach_via_fan_pack"


def test_fan_pack_missing_required_file_blocks_packaging() -> None:
    members = {"README.md": b"x", "reports/a.md": b"![c](../charts/c.png)\n"}
    with pytest.raises(B.FanPackError, match=r"players\.csv"):
        B.build_fan_pack(members, required=("README.md", "players.csv"), unavailable={}, meta={})
    with pytest.raises(B.FanPackError, match=r"charts/c\.png"):  # dependency closure from report links
        B.build_fan_pack(members, required=("README.md",), unavailable={}, meta={})
    with pytest.raises(ValueError, match="unsafe archive member"):
        B.build_fan_pack({"../x.csv": b"x"}, required=(), unavailable={}, meta={})


def test_reports_keep_curated_text_outside_generated_markers(
    variant: tuple[B.ReleaseCandidate, list[int]], built: B.ReleaseCandidate
) -> None:
    cand, _calls = variant
    text = (cand.public_dir / "downloads/reports/stat-leaders.md").read_text()
    assert "Curated intro kept verbatim." in text and "Curated footer." in text and "\nold\n" not in text
    assert text.count("<!-- GEN:stat-leaders START -->") == 1 and text.count("<!-- GEN:stat-leaders END -->") == 1
    for name in B.REPORTS:
        doc = (built.public_dir / f"downloads/reports/{name}.md").read_text()
        assert doc.count(f"<!-- GEN:{name} START -->") == 1, name


def test_bad_overlays_fail_before_anything_is_staged(env: DemoEnv, tmp_path: Path) -> None:
    with pytest.raises(MarkerError):
        B.build_release(
            env.snapshot,
            B.ReleaseInputs(report_overlays={"stat-leaders": "# no markers\n"}),
            context_for(env, tmp_path),
        )
    with pytest.raises(ValueError, match="unknown report"):
        B.build_release(env.snapshot, B.ReleaseInputs(report_overlays={"nope": CURATED}), context_for(env, tmp_path))
    assert not (tmp_path / "releases").exists()


# ---------------------------------------------------------------------------
# R05-R07: integrity, no mtime selection, determinism and reuse
# ---------------------------------------------------------------------------


def test_identical_input_is_byte_identical_and_mtime_independent(
    built: B.ReleaseCandidate, env: DemoEnv, tmp_path: Path
) -> None:
    # perturb every mtime the builder could possibly consult, and repeat an input
    for p in [*env.data_root.rglob("*"), *env.prediction_dir.rglob("*"), *env.live_root.rglob("*")]:
        os.utime(p, (1_000_000_000, 1_000_000_000), follow_symlinks=False)
    inputs = full_inputs(env, prediction_dirs=(env.prediction_dir, env.prediction_dir))
    again = B.build_release(env.snapshot, inputs, context_for(env, tmp_path))
    assert again.release_id == built.release_id
    assert (again.release_dir / "checksums.json").read_bytes() == (built.release_dir / "checksums.json").read_bytes()
    assert _files(again) == _files(built)
    for rel in _files(built):
        assert (again.public_dir / rel).read_bytes() == (built.public_dir / rel).read_bytes(), rel


def test_rebuild_into_same_root_reuses_the_immutable_release(bare: B.ReleaseCandidate, env: DemoEnv) -> None:
    stamp = (bare.release_dir / "checksums.json").stat().st_mtime_ns
    ctx = context_for(env, bare.release_dir.parent.parent)
    again = B.build_release(env.snapshot, B.ReleaseInputs(demo=True), ctx)
    assert again.reused and again.release_id == bare.release_id and again.ok
    assert (again.release_dir / "checksums.json").stat().st_mtime_ns == stamp
    assert set(again.timings) == {"reuse_check_s", "total_s"}


def test_tampered_or_incomplete_release_is_rejected(bare: B.ReleaseCandidate, env: DemoEnv, tmp_path: Path) -> None:
    copy = tmp_path / "releases" / bare.release_id
    shutil.copytree(bare.release_dir, copy)
    target = copy / "public/players/index.json"
    target.write_bytes(target.read_bytes().replace(b'"count":', b'"count": '))
    report = validate_release(copy, write=False)
    assert report.checks["hashes"] is CheckOutcome.FAIL
    # a rebuild that maps to this immutable release must refuse the tampered bytes, not reuse them
    with pytest.raises(B.ReleaseBuildError, match="failed validation"):
        B.build_release(env.snapshot, B.ReleaseInputs(demo=True), context_for(env, tmp_path))
    (copy / "public/quality.json").unlink()
    report = validate_release(copy, write=False)
    assert report.checks["closure"] is CheckOutcome.FAIL


def test_corrupt_snapshot_fragment_refuses_to_build(env: DemoEnv, tmp_path: Path) -> None:
    from supercoach_via.storage.snapshots import IntegrityError, SnapshotRef

    data = tmp_path / "var"
    shutil.copytree(env.data_root, data)
    frag = next((data / "fragments").rglob("*.parquet"))
    frag.chmod(0o644)
    raw = bytearray(frag.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    frag.write_bytes(bytes(raw))
    ctx = context_for(env, tmp_path / "out")
    ctx.settings = ctx.settings.model_copy(update={"data_root": data})
    with pytest.raises(IntegrityError):
        B.build_release(SnapshotRef(env.snapshot.snapshot_id, data / "unused.json"), B.ReleaseInputs(), ctx)
    assert not (tmp_path / "out" / "releases").exists()


# ---------------------------------------------------------------------------
# S04/S05: release allowlist
# ---------------------------------------------------------------------------


def test_release_contains_no_private_operator_or_model_content(built: B.ReleaseCandidate, env: DemoEnv) -> None:
    forbidden_parts = ("docs/rewrite", ".claude", ".env", "models", "raw", "fragments", "snapshots")
    forbidden_ext = (".pkl", ".joblib", ".parquet", ".duckdb", ".py", ".sh", ".log", ".jsonl")
    blobs: list[tuple[str, bytes]] = []
    for rel in _files(built):
        assert not any(part in rel.split("/") or rel.startswith(part) for part in forbidden_parts), rel
        assert not rel.endswith(forbidden_ext), rel
        data = (built.public_dir / rel).read_bytes()
        blobs.append((rel, data))
        if rel.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for name in zf.namelist():
                    assert not name.endswith(forbidden_ext) and ".claude" not in name and ".env" not in name
                    blobs.append((f"{rel}!{name}", zf.read(name)))
    secrets = (str(env.root).encode(), b"/tmp/", b"/home/", b"ANTHROPIC_API_KEY", b"payload.joblib")
    for rel, data in blobs:
        for s in secrets:
            assert s not in data, (rel, s)
    manifest = _json(built, "release.json")
    assert all(not ref["path"].startswith(("/", "..")) for ref in manifest["resources"].values())


# ---------------------------------------------------------------------------
# Performance: season-scoped query contexts (operation counts, not timings)
# ---------------------------------------------------------------------------


def test_team_pages_use_one_partition_scoped_context_per_season(
    variant: tuple[B.ReleaseCandidate, list[int]],
) -> None:
    cand, calls = variant
    seasons = [2024, 2025, 2026]
    assert cand.counts["season_contexts"] == len(seasons)
    assert cand.counts["team_pages"] == 6 * len(seasons)
    # per season: that season's player_games fragment and at most six match partitions
    assert cand.counts["season_context_player_game_fragments_max"] == 1
    assert cand.counts["season_context_match_fragments_max"] <= 1 + B.FIVE_YEARS
    # ladder work is per season, independent of the number of clubs (6 per season here)
    assert sorted(set(calls)) == seasons
    assert all(calls.count(s) == 2 for s in seasons), calls  # builder + finals_pathway
    # one query per stat per season for club leaders, not one per club
    assert cand.counts["team_leader_queries"] == len(seasons) * len(B.TEAM_LEADER_STATS)


def test_batched_team_parts_equal_the_per_club_analytics(env: DemoEnv) -> None:
    """The season-batched leaders/fixtures/form equal analytics' per-club functions exactly."""
    from supercoach_via.analytics import teams
    from supercoach_via.storage.queries import SnapshotQuery
    from supercoach_via.storage.snapshots import load_snapshot

    manifest = load_snapshot(env.data_root, env.snapshot.snapshot_id)
    for season in (2024, 2025, 2026):
        parts = {"matches": {str(s) for s in range(season - 5, season + 1)}, "player_games": {str(season)}}
        with SnapshotQuery(env.data_root, manifest, partitions=parts) as q:
            clubs = sorted(c for (c,) in q.rows("SELECT DISTINCT home_club_id FROM matches WHERE season = ?", [season]))
            got = B.season_team_parts(q, season, clubs)
            assert set(got) == set(clubs)
            for club in clubs:
                assert got[club].leaders == teams.club_season_leaders(q, club, season)
                assert got[club].fixtures == teams.club_fixtures(q, club, season)
                assert got[club].form == teams.team_form(q, club, n=B.FORM_WINDOW)


def test_player_stat_lines_equal_the_analytics_bundle_helpers(env: DemoEnv) -> None:
    from supercoach_via.analytics import players
    from supercoach_via.domain.metrics import CoverageEras
    from supercoach_via.storage.queries import SnapshotQuery
    from supercoach_via.storage.snapshots import load_snapshot

    manifest = load_snapshot(env.data_root, env.snapshot.snapshot_id)
    eras = CoverageEras.load(B.DEFAULT_CONFIG_DIR / "coverage.yaml")
    with SnapshotQuery(env.data_root, manifest) as q:
        bundle = players.player_stats_bundle(q, eras)

    def res(pid: str, s: int) -> str:
        return f"player-games/{pid}/{s}.json"

    lines = B.player_stat_lines(bundle, res)
    ids = sorted(str(p) for p in bundle.games["player_id"])
    assert set(lines) == set(ids)
    for pid in ids:
        career, seasons = lines[pid]
        assert career == players.career_stat_values(bundle, pid)
        assert seasons == players.season_lines(bundle, pid, res)


def test_stage_timings_are_reported(built: B.ReleaseCandidate) -> None:
    for stage in (
        "snapshot_s",
        "season_resources_s",
        "players_s",
        "history_s",
        "predictions_s",
        "downloads_s",
        "finish_s",
        "validate_s",
        "total_s",
    ):
        assert stage in built.timings and built.timings[stage] >= 0
    assert built.counts["files"] == len(_files(built))


def test_yearly_top100_csv_only_for_the_latest_final_season(built: B.ReleaseCandidate) -> None:
    # Legacy cadence: data/top100/yearly/year_<season>.csv appears once a season is final.
    # The DEMO corpus's 2026 season is in progress, so the CSV is 2025's and 2026 is listed unavailable.
    files = sorted(p.name for p in (built.public_dir / "downloads").glob("yearly-top-100-*.csv"))
    assert files == ["yearly-top-100-2025.csv"]
    rows = list(csv.reader(io.StringIO((built.public_dir / "downloads" / files[0]).read_text())))
    assert rows[0] == ["player", "score", "percentile_rank", "games_played"] and len(rows) > 1
    table = _json(built, "history/yearly_top_100/2025.json")
    assert [r[0] for r in rows[1:4]] and len(rows) - 1 == len(table["rows"])


def test_era_summary_and_brownlow_proxy_downloads(built: B.ReleaseCandidate) -> None:
    era = list(csv.DictReader(io.StringIO((built.public_dir / "downloads/era-summary.csv").read_text())))
    assert era and {"era", "metric", "n_player_games", "n_with_metric", "mean_per_game", "recording_status"} <= set(
        era[0]
    )
    for r in era:  # an unrecorded metric is blank (unknown), never a zero mean
        assert r["n_with_metric"] != "0" or r["mean_per_game"] == ""
    bl = list(csv.DictReader(io.StringIO((built.public_dir / "downloads/brownlow-proxy-2026.csv").read_text())))
    assert bl and bl[0]["rank"] == "1"
    assert {r["label"] for r in bl} == {"Brownlow stat-profile proxy"} and {r["version"] for r in bl} == {
        "brownlow_proxy_v1.1"
    }
    items = {i["path"] for i in _json(built, "downloads.json")["items"]}
    assert {"downloads/era-summary.csv", "downloads/brownlow-proxy-2026.csv"} <= items


def test_player_pages_are_positional_and_expand_to_the_analytics_values(built: B.ReleaseCandidate, env: DemoEnv) -> None:
    from supercoach_via.analytics import players
    from supercoach_via.domain.metrics import CoverageEras
    from supercoach_via.publish import resources
    from supercoach_via.publish.view_models import PlayerDetail, expand_stats
    from supercoach_via.storage.queries import SnapshotQuery
    from supercoach_via.storage.snapshots import load_snapshot

    manifest = load_snapshot(env.data_root, env.snapshot.snapshot_id)
    eras = CoverageEras.load(B.DEFAULT_CONFIG_DIR / "coverage.yaml")
    with SnapshotQuery(env.data_root, manifest) as q:
        bundle = players.player_stats_bundle(q, eras)

    def res(pid: str, s: int) -> str:
        return f"player-games/{resources.public_key(pid)}/{s}.json"

    lines = B.player_stat_lines(bundle, res)
    checked = 0
    for pid, (career, seasons) in lines.items():
        raw = _json(built, f"players/{resources.public_key(pid)}.json")
        assert "mean" not in json.dumps(raw["career"]) and "coverage" not in json.dumps(raw["seasons"])
        d = PlayerDetail.model_validate(raw)
        assert expand_stats(d.stat_names, d.career, d.career_games) == career
        assert [s.season for s in d.seasons] == [s.season for s in seasons]
        for got, want in zip(d.seasons, seasons, strict=True):
            assert (got.clubs, got.games, got.games_resource) == (want.clubs, want.games, want.games_resource)
            assert expand_stats(d.stat_names, got.stats, got.games) == want.stats
        checked += 1
    assert checked > 10


def _bundle(env: DemoEnv) -> Any:
    from supercoach_via.analytics import players
    from supercoach_via.domain.metrics import CoverageEras
    from supercoach_via.storage.queries import SnapshotQuery
    from supercoach_via.storage.snapshots import load_snapshot

    manifest = load_snapshot(env.data_root, env.snapshot.snapshot_id)
    with SnapshotQuery(env.data_root, manifest) as q:
        return players.player_stats_bundle(q, CoverageEras.load(B.DEFAULT_CONFIG_DIR / "coverage.yaml"))


def test_streamed_player_columns_equal_the_object_path(env: DemoEnv) -> None:
    from supercoach_via.publish.view_models import PlayerSeason, to_stat_columns

    bundle = _bundle(env)

    def res(pid: str, s: int) -> str:
        return f"player-games/{pid}/{s}.json"

    lines = B.player_stat_lines(bundle, res)
    cols = B.PlayerStatColumns(bundle, res)
    games = {str(r["player_id"]): int(r["career_games"]) for r in bundle.games.to_dict("records")}
    for pid, (career, seasons) in lines.items():
        names = [v.stat for v in career]
        want = (names, to_stat_columns(names, career, games[pid]),
                [PlayerSeason(season=s.season, clubs=s.clubs, games=s.games,
                              stats=to_stat_columns(names, s.stats, s.games), games_resource=s.games_resource)
                 for s in seasons])  # fmt: skip
        assert cols.get(pid) == want, pid
    assert cols.get("legacy:nobody") == ([], B.StatColumns(total=[], observed_games=[], eligible_games=[]), [])


def test_streamed_player_columns_refuse_a_mean_that_is_not_derived(env: DemoEnv) -> None:
    bundle = _bundle(env)
    bad = bundle.seasons.copy()
    i = bad.index[bad["observed_games"] > 0][0]
    bad.loc[i, "mean"] = float(bad.loc[i, "mean"]) + 1.0
    with pytest.raises(ValueError, match="derived"):
        B.PlayerStatColumns(replace(bundle, seasons=bad), lambda p, s: f"x/{p}/{s}.json")
