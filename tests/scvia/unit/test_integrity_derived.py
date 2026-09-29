"""Integrity checker: team, history, list, summary, download, forecast and content resources.

Every test edits a copy of the sealed demo release, recalculates every structural hash
(``integrity_fixtures.reseal``) and requires the responsible SEMANTIC check to report the
discrepancy: byte and seal verification alone never catch these.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from supercoach_via.integrity.runner import AuditOptions, AuditResult, run_audit
from supercoach_via.publish.web_data import canonical_json_bytes, sha256_bytes
from tests.scvia.unit import integrity_fixtures as fx

AS_OF = "2026-05-03T00:00:00Z"
DERIVED = ("release.derived", "release.forecast", "release.content", "release.coverage")


@pytest.fixture
def env(integrity_demo: Any, tmp_path: Path) -> dict[str, Path]:
    demo = integrity_demo.env
    out = {
        "data": tmp_path / "var",
        "release": tmp_path / "releases" / integrity_demo.release_dir.name,
        "models": tmp_path / "models",
        "predictions": tmp_path / "predictions",
        "evaluation": tmp_path / "evaluations" / demo.evaluation_dir.name,
        "live": tmp_path / "live",
        "content": tmp_path / "content",
    }
    shutil.copytree(integrity_demo.data_root, out["data"], symlinks=True)
    shutil.copytree(integrity_demo.release_dir, out["release"], symlinks=True)
    shutil.copytree(demo.root / "models", out["models"])
    shutil.copytree(demo.root / "predictions", out["predictions"])
    shutil.copytree(demo.evaluation_dir, out["evaluation"])
    shutil.copytree(demo.live_root, out["live"])
    shutil.copytree(demo.content_root, out["content"])
    for tree in ("models", "predictions", "evaluation", "live", "content"):
        for p in out[tree].rglob("*"):
            p.chmod(0o755 if p.is_dir() else 0o644)
    return out


def audit(env: dict[str, Path], **kw: Any) -> AuditResult:
    opts: dict[str, Any] = dict(
        data_root=env["data"],
        release_dir=env["release"],
        scope="full",
        as_of=AS_OF,
        families=("release",),
        keep_all=True,
        models_root=env["models"],
        predictions_root=env["predictions"],
        evaluation_dirs=(env["evaluation"],),
        live_root=env["live"],
        content_root=env["content"],
        content_manifest=env["content"] / "public_content.toml",
    )
    opts.update(kw)
    return run_audit(AuditOptions(**opts))


def found(res: AuditResult, rule_id: str) -> list[Any]:
    return [f for f in res.findings if f.rule_id == rule_id and f.status == "open"]


def status(res: AuditResult, check_id: str) -> str:
    return next(c["status"] for c in res.report["checks"] if c["check_id"] == check_id)


def public(env: dict[str, Path]) -> Path:
    return env["release"] / "public"


def _team_page(env: dict[str, Path]) -> Path:
    return sorted(public(env).glob("teams/*/2026.json"))[0]


def _csv(path: Path) -> list[list[str]]:
    return list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))


def _rewrite_download(env: dict[str, Path], name: str, data: bytes) -> None:
    """Replace a download and keep downloads.json's bytes/sha256/rows consistent with it."""
    (public(env) / "downloads" / name).write_bytes(data)

    def fix(doc: Any) -> None:
        for item in doc["items"]:
            if item["path"] == f"downloads/{name}":
                item.update(bytes=len(data), sha256=sha256_bytes(data))
                if item["kind"] == "csv":
                    item["rows"] = len(_csv(public(env) / "downloads" / name)) - 1

    fx.edit_json(public(env) / "downloads.json", fix)


# ---------------------------------------------------------------------------
# The four payload mutations: team totals, history ranks/values, download row
# membership and stale summaries, in cold, warm-cache and changed-since modes
# ---------------------------------------------------------------------------


def _mutate_team_total(env: dict[str, Path]) -> Path:
    """The payload's diagnostic: first team-season statistic total changed, mean adjusted consistently."""
    path = _team_page(env)

    def edit(d: Any) -> None:
        sv = d["team_stats"][0]
        sv["total"] += 7
        sv["mean"] = sv["total"] / sv["observed_games"]

    fx.edit_json(path, edit)
    return path


def _mutate_history(env: dict[str, Path]) -> Path:
    path = public(env) / "history" / "career_disposals_total" / "all.json"

    def edit(d: Any) -> None:
        rows = d["rows"]
        rows[0]["player_id"], rows[1]["player_id"] = rows[1]["player_id"], rows[0]["player_id"]
        rows[0]["name"], rows[1]["name"] = rows[1]["name"], rows[0]["name"]
        rows[2]["value"] += 1

    fx.edit_json(path, edit)
    return path


def _mutate_download_rows(env: dict[str, Path]) -> str:
    rows = _csv(public(env) / "downloads" / "players.csv")
    dropped = rows.pop(3)
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    _rewrite_download(env, "players.csv", buf.getvalue().encode())
    return dropped[0]


def _mutate_summaries(env: dict[str, Path]) -> None:
    def overview(d: Any) -> None:
        d["leaders"][0]["value"] += 1

    def quality(d: Any) -> None:
        d["table_counts"]["matches"] += 1

    fx.edit_json(public(env) / "overview.json", overview)
    fx.edit_json(public(env) / "quality.json", quality)


@pytest.mark.parametrize("mode", ["cold", "warm", "changed_since"])
def test_semantic_mutations_are_detected_in_every_mode(env: dict[str, Path], tmp_path: Path, mode: str) -> None:
    cache = tmp_path / "cache"
    kw: dict[str, Any] = {"checks": DERIVED}
    if mode != "cold":
        clean = audit(env, cache_dir=cache, **kw)
        # restricted to the derived checks, so UNKNOWN (match/player pages uncompared), never FAIL
        assert clean.outcome.value == "UNKNOWN" and not clean.findings, [f.as_dict() for f in clean.findings][:3]
        prior = tmp_path / "prior.json"
        prior.write_bytes(canonical_json_bytes(clean.report))
        kw["cache_dir"] = cache
        if mode == "changed_since":
            kw["changed_since"] = prior
    team = _mutate_team_total(env)
    _mutate_history(env)
    dropped = _mutate_download_rows(env)
    _mutate_summaries(env)
    fx.reseal(env["release"])
    res = audit(env, **kw)
    assert res.outcome.value == "FAIL"
    rel_team = team.relative_to(public(env)).as_posix()
    assert any(f.field.startswith("team_stats[0].") for f in found(res, "release.team_value") if rel_team in f.entity)
    hist_fields = {f.field for f in found(res, "release.history_value")}
    assert {"rows[0].player_id", "rows[1].player_id", "rows[2].value"} <= hist_fields
    assert any(dropped in json.dumps(f.as_dict()) for f in found(res, "release.download_value"))
    summaries = {(f.entity, f.field) for f in found(res, "release.summary_value")}
    assert ("resource:overview.json", "leaders[0].value") in summaries
    assert ("resource:quality.json", "table_counts.matches") in summaries


# ---------------------------------------------------------------------------
# Clean release and coverage truthfulness
# ---------------------------------------------------------------------------


def test_clean_release_has_semantic_coverage_for_every_resource(env: dict[str, Path]) -> None:
    res = audit(env)
    assert res.outcome.value == "PASS", [f.as_dict() for f in res.findings][:5]
    sem = res.report["coverage"]["semantic"]
    assert sem["uncompared"] == {}
    by_type = sem["by_type"]
    for kind in (
        "team_index",
        "team_season",
        "history_index",
        "history_table",
        "lists_index",
        "overview",
        "quality",
        "downloads",
        "prediction_index",
        "prediction_set",
        "accuracy_index",
        "accuracy_report",
        "live_index",
        "live_snapshot",
        "article_index",
    ):
        assert by_type[kind]["compared"] == by_type[kind]["resources"] > 0, kind
    # prose is verified for provenance only and says so
    assert by_type["article"]["provenance_only"] == by_type["article"]["resources"] > 0
    assert any("prose" in u for u in sem["unaudited"])


def test_missing_comparator_inputs_make_the_audit_unknown_not_pass(env: dict[str, Path]) -> None:
    res = audit(env, evaluation_dirs=(), live_root=None, content_root=None, content_manifest=None)
    assert res.outcome.value == "UNKNOWN"
    assert status(res, "release.forecast") == "UNKNOWN"
    assert status(res, "release.content") == "UNKNOWN"
    assert status(res, "release.coverage") == "UNKNOWN"
    assert not res.report["scope"]["complete"]
    uncompared = res.report["coverage"]["semantic"]["uncompared"]
    assert {"accuracy_report", "live_snapshot"} <= set(uncompared)


def test_a_comparator_that_did_not_run_leaves_coverage_unknown(env: dict[str, Path]) -> None:
    res = audit(env, checks=("release.coverage",))
    assert status(res, "release.coverage") == "UNKNOWN"
    assert "team_season" in res.report["coverage"]["semantic"]["uncompared"]


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------


def test_ladder_row_and_form_changed(env: dict[str, Path]) -> None:
    path = _team_page(env)

    def edit(d: Any) -> None:
        d["ladder"][0]["premiership_points"] += 4
        d["form"][-1]["margin"] += 1
        d["leaders"][0]["value"] += 1

    fx.edit_json(path, edit)
    fx.reseal(env["release"])
    res = audit(env, checks=("release.derived",))
    fields = {f.field for f in found(res, "release.team_value")}
    last = len(json.loads(path.read_text())["form"]) - 1
    assert {"ladder[0].premiership_points", f"form[{last}].margin", "leaders[0].value"} <= fields


def test_heuristic_that_misstates_the_ladder_position(env: dict[str, Path]) -> None:
    path = _team_page(env)

    def edit(d: Any) -> None:
        pos = d["position"]
        d["heuristics"][0]["text"] = d["heuristics"][0]["text"].replace(f" are {pos} on ", f" are {pos + 1} on ")

    fx.edit_json(path, edit)
    fx.reseal(env["release"])
    res = audit(env, checks=("release.derived",))
    assert any(f.field == "heuristics[0].text" for f in found(res, "release.team_value"))


def test_team_index_season_membership(env: dict[str, Path]) -> None:
    fx.edit_json(public(env) / "teams" / "index.json", lambda d: d["teams"][0]["seasons"].pop())
    fx.reseal(env["release"])
    res = audit(env, checks=("release.derived",))
    assert any(f.field.startswith("teams[0].seasons") for f in found(res, "release.team_value"))


def test_stale_team_page_for_a_season_without_the_club(env: dict[str, Path]) -> None:
    src = _team_page(env)
    extra = src.parent / "1901.json"
    shutil.copy(src, extra)
    fx.reseal(env["release"], expect_valid=True)
    res = audit(env, checks=("release.derived",))
    assert any("1901.json" in f.entity for f in found(res, "release.derived_resource"))


# ---------------------------------------------------------------------------
# History and rankings
# ---------------------------------------------------------------------------


def test_yearly_ranking_score_changed(env: dict[str, Path]) -> None:
    """(The demo has no 150-game player, so its all-time table is empty; the yearly lists are not.)"""
    path = sorted((public(env) / "history" / "yearly_top_100").glob("*.json"))[-1]
    fx.edit_json(path, lambda d: d["rows"][0].update(value=d["rows"][0]["value"] + 1))
    fx.reseal(env["release"])
    res = audit(env, checks=("release.derived",))
    assert any(f.field == "rows[0].value" for f in found(res, "release.history_value"))


def test_history_index_era_summary_changed(env: dict[str, Path]) -> None:
    fx.edit_json(public(env) / "history" / "index.json", lambda d: d["era_summary"][0].update(players=1))
    fx.reseal(env["release"])
    res = audit(env, checks=("release.derived",))
    assert any(f.field == "era_summary[0].players" for f in found(res, "release.history_value"))


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def test_download_metadata_that_disagrees_with_the_file(env: dict[str, Path]) -> None:
    fx.edit_json(public(env) / "downloads.json", lambda d: d["items"][0].update(rows=d["items"][0]["rows"] + 1))
    fx.reseal(env["release"])
    res = audit(env, checks=("release.derived",))
    assert any(f.field == "items[0].rows" for f in found(res, "release.download_metadata"))


def test_ranking_download_value_changed(env: dict[str, Path]) -> None:
    name = next(public(env).glob("downloads/yearly-top-100-*.csv")).name
    rows = _csv(public(env) / "downloads" / name)
    rows[1][2] = str(float(rows[1][2]) + 0.5)
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    _rewrite_download(env, name, buf.getvalue().encode())
    fx.reseal(env["release"])
    res = audit(env, checks=("release.derived",))
    assert found(res, "release.download_value")


def test_fan_pack_member_that_differs_from_its_download(env: dict[str, Path]) -> None:
    import zipfile

    pack = public(env) / "downloads" / "fan-pack.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(pack) as src, zipfile.ZipFile(buf, "w") as dst:
        for info in src.infolist():
            data = src.read(info)
            if info.filename.endswith("players.csv"):
                data += b"extra,row\n"
            dst.writestr(info, data)
    _rewrite_download(env, "fan-pack.zip", buf.getvalue())
    fx.reseal(env["release"])
    res = audit(env, checks=("release.derived",))
    assert any("players.csv" in (f.field or "") for f in found(res, "release.download_value"))


def test_era_summary_value_changed(env: dict[str, Path]) -> None:
    rows = _csv(public(env) / "downloads" / "era-summary.csv")
    col = rows[0].index("n_with_metric")
    rows[1][col] = str(int(rows[1][col]) + 1)
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    _rewrite_download(env, "era-summary.csv", buf.getvalue().encode())
    fx.reseal(env["release"])
    res = audit(env, checks=("release.derived",))
    assert any("n_with_metric" in (f.field or "") for f in found(res, "release.download_value"))


# ---------------------------------------------------------------------------
# Forecast, accuracy and live resources against their own pinned inputs
# ---------------------------------------------------------------------------


def test_published_prediction_differs_from_its_artifact(env: dict[str, Path]) -> None:
    path = next(public(env).glob("predictions/2026/*.json"))
    fx.edit_json(path, lambda d: d["rows"][0].update(predicted_disposals=d["rows"][0]["predicted_disposals"] + 1))
    fx.reseal(env["release"])
    res = audit(env, checks=("release.forecast",))
    assert any(f.field == "rows[0].predicted_disposals" for f in found(res, "release.forecast_value"))


def test_accuracy_headline_differs_from_the_scored_rows(env: dict[str, Path]) -> None:
    path = next(p for p in public(env).glob("accuracy/*/*.json") if "legacy_unknown" not in p.name)
    fx.edit_json(path, lambda d: d["headline"].update(mae=d["headline"]["mae"] * 0.9))
    fx.reseal(env["release"])
    res = audit(env, checks=("release.forecast",))
    assert any(f.field == "headline.mae" for f in found(res, "release.forecast_value"))


def test_live_snapshot_differs_from_the_accepted_capture(env: dict[str, Path]) -> None:
    path = next(public(env).glob("live/*/latest.json"))
    fx.edit_json(path, lambda d: d["home"].update(score=d["home"]["score"] + 6))
    fx.reseal(env["release"])
    res = audit(env, checks=("release.forecast",))
    assert any(f.field == "home.score" for f in found(res, "release.forecast_value"))


# ---------------------------------------------------------------------------
# Content: provenance only
# ---------------------------------------------------------------------------


def test_article_whose_source_changed_after_publication(env: dict[str, Path]) -> None:
    src = env["content"] / "docs" / "news" / "2026-05-01-demo-article.md"
    src.write_text(src.read_text() + "\nA later edit.\n")
    res = audit(env, checks=("release.content",))
    assert found(res, "release.content_provenance")


def test_generated_article_bound_to_another_snapshot(env: dict[str, Path]) -> None:
    path = public(env) / "articles" / "demo-generated-note.json"
    fx.edit_json(path, lambda d: d.update(provenance=d["provenance"].replace("sha256:", "sha256:0")))
    fx.reseal(env["release"])
    res = audit(env, checks=("release.content",))
    assert found(res, "release.content_provenance")


# ---------------------------------------------------------------------------
# Independent readings of the ranking and list contracts on tiny inputs
# ---------------------------------------------------------------------------


def test_checker_ranking_reading_on_a_hand_computed_cohort() -> None:
    """Two post-2010 players: scores, percentile ranks and the single-stat cap by hand."""
    from supercoach_via.integrity import derived_expect as dx

    cfg = dx.ranking_config(None)
    # A: 10 goals x 55 = 550, 100 kicks x 4.5 = 450 -> uncapped 1000; cap 0.55 -> no excess
    # B: 20 goals x 55 = 1100 alone -> excess 1100 - 0.55*1100 = 495 -> int(605)
    season_rows = [
        ("a", "a", 2020, 20, {"goals": 10.0, "kicks": 100.0}),
        ("b", "b", 2020, 18, {"goals": 20.0}),
    ]
    yearly = dx.rank_season(2020, season_rows, cfg)
    assert [(e["player_id"], e["score"], e["percentile_rank"]) for e in yearly] == [
        ("a", 1000, 100.0),
        ("b", 605, 50.0),
    ]


def test_checker_lists_reading_orders_and_counts() -> None:
    import duckdb
    import pyarrow as pa

    from supercoach_via.integrity import derived_expect as dx

    con = duckdb.connect()
    con.register("clubs", pa.table({"club_id": ["c1"], "name": ["Club One"]}))
    con.register(
        "draft_events",
        pa.table(
            {
                "draft_event_id": ["d2", "d1"],
                "season": [2020, 2020],
                "event_type": ["national", "national"],
                "draft_round": [1, 1],
                "pick": [2, 1],
                "club_id": ["c1", None],
                "club_source_name": ["C1", "Other"],
                "player_name": ["B", "A"],
                "player_id": ["p:b", None],
                "recruited_from": [None, None],
                "grade": [None, None],
                "source_family": ["draftguru", "draftguru"],
            }
        ),
    )
    docs = dx.lists_expected(con, {"clubs", "draft_events"})
    season = docs["lists/2020.json"]
    assert [d["player_name"] for d in season["drafts"]] == ["A", "B"]
    assert season["drafts"][1]["club"] == "Club One" and season["drafts"][0]["club"] == "Other"
    assert docs["lists/index.json"] == {"seasons": [2020], "resources": {"2020": "lists/2020.json"}}
