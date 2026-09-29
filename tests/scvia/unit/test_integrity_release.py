"""Integrity checker, release -> canonical agreement and artifact binding (family E, release side)."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from supercoach_via.integrity.runner import AuditOptions, AuditResult, run_audit
from supercoach_via.publish.resources import public_key
from tests.scvia.unit import integrity_fixtures as fx

FAMILIES = ("release",)
AS_OF = "2026-05-03T00:00:00Z"


@pytest.fixture
def pristine(integrity_demo: Any) -> tuple[Path, Path]:
    return integrity_demo.data_root, integrity_demo.release_dir


@pytest.fixture
def rel(pristine: tuple[Path, Path], tmp_path: Path) -> tuple[Path, Path]:
    data, release = pristine
    d2 = tmp_path / "var"
    shutil.copytree(data, d2, symlinks=True)
    r2 = tmp_path / "releases" / release.name
    shutil.copytree(release, r2, symlinks=True)
    return d2, r2


def audit(data: Path, release: Path, **kw: Any) -> AuditResult:
    opts = dict(
        data_root=data,
        release_dir=release,
        snapshot="current",
        scope="full",
        as_of=AS_OF,
        families=FAMILIES,
        keep_all=True,
    )
    opts.update(kw)
    return run_audit(AuditOptions(**opts))  # type: ignore[arg-type]


def found(res: AuditResult, rule_id: str) -> list[Any]:
    return [f for f in res.findings if f.rule_id == rule_id and f.status == "open"]


def rules(res: AuditResult) -> set[str]:
    return {f.rule_id for f in res.findings if f.status == "open"}


def _first(release: Path, pattern: str, pred: Any = lambda d: True) -> Path:
    for p in sorted((release / "public").glob(pattern)):
        if pred(json.loads(p.read_text())):
            return p
    raise AssertionError(f"no {pattern} matched")


def _detail_with_players(release: Path) -> Path:
    return _first(release, "matches/detail/*.json", lambda d: len(d["home_players"]["player_id"]) >= 2)


def test_clean_release_passes(rel: tuple[Path, Path], integrity_demo: Any) -> None:
    demo = integrity_demo.env
    res = audit(
        *rel,
        models_root=demo.root / "models",
        predictions_root=demo.root / "predictions",
        evaluation_dirs=(demo.evaluation_dir,),
        live_root=demo.live_root,
        content_root=demo.content_root,
        content_manifest=demo.content_manifest,
    )
    assert res.outcome.value == "PASS", [f.as_dict() for f in res.findings][:5]
    cov = res.report["coverage"]["public_compare"]
    assert cov["match_details"] > 0 and cov["player_logs"] > 0 and cov["player_details"] > 0
    assert res.report["counts"]["cells_compared"] > 1000
    assert res.report["scope"]["semantic_complete"] is True


def test_release_without_comparator_inputs_is_not_a_pass(rel: tuple[Path, Path]) -> None:
    """Byte and seal checks alone are not a semantic comparison: missing inputs leave it UNKNOWN."""
    res = audit(*rel)
    assert res.outcome.value == "UNKNOWN" and not res.findings
    assert res.report["scope"]["semantic_complete"] is False


def test_value_attached_to_the_wrong_player_with_unchanged_totals(rel: tuple[Path, Path]) -> None:
    data, release = rel
    path = _detail_with_players(release)

    def swap(d: Any) -> None:
        s = d["home_players"]["stats"]
        s[0], s[1] = s[1], s[0]

    fx.edit_json(path, swap)
    fx.reseal(release)
    res = audit(data, release, checks=('release.public',))
    hits = {f.entity for f in found(res, "release.cell_mismatch")}
    assert len({e.rsplit("|", 1)[0] for e in hits}) >= 1 and len(hits) >= 2
    assert res.outcome.value == "FAIL"


def test_truncated_compact_row_with_hashes_recalculated(rel: tuple[Path, Path]) -> None:
    data, release = rel
    path = _detail_with_players(release)
    fx.edit_json(path, lambda d: d["home_players"]["stats"][0].pop())
    fx.reseal(release, expect_valid=False)  # O55-04: validate_release now refuses the short row too
    res = audit(data, release, checks=("release.public", "release.validate", "release.artifact"))
    assert found(res, "release.compact_row_width")
    assert found(res, "release.schema_invalid")  # the producer's own rule, mapped
    assert found(res, "release.validation_outcome")


def test_null_converted_to_zero_in_public_data(rel: tuple[Path, Path]) -> None:
    """A canonical null published as 0 is a contradiction, not a harmless equivalent."""
    from supercoach_via.domain.schemas import CheckOutcome, ValidationReport
    from supercoach_via.storage import snapshots
    from supercoach_via.storage.queries import SnapshotQuery

    data, release = rel
    m = snapshots.load_snapshot(data)
    with SnapshotQuery(data, m, tables={"player_games"}) as q:
        row = q.arrow(
            "SELECT * FROM player_games WHERE season = 2026 ORDER BY match_id, player_id LIMIT 1"
        ).to_pylist()[0]
    row["bounces"] = None
    cand = snapshots.apply_upserts(
        data, m, {"player_games": [row]}, clock=lambda: m.created_at, code_version="t", status=m.status
    )
    snapshots.promote(data, cand, ValidationReport(outcome=CheckOutcome.PASS))
    path = release / "public" / "matches" / "detail" / f"{public_key(row['match_id'])}.json"

    def zero(d: Any) -> None:
        for side in ("home_players", "away_players"):
            if row["player_id"] in d[side]["player_id"]:
                i = d[side]["player_id"].index(row["player_id"])
                d[side]["stats"][i][d["stat_columns"].index("bounces")] = 0

    fx.edit_json(path, zero)
    fx.reseal(release)
    res = audit(data, release, checks=('release.public',))
    entity = f"player_game:{row['match_id']}|{row['player_id']}|{row['club_id']}"
    got = {
        (f.evidence["resource"].split("/")[0], f.expected, f.actual)
        for f in found(res, "release.cell_mismatch")
        if f.entity == entity and f.field == "bounces"
    }
    assert ("matches", None, 0) in got, got  # the detail now says 0
    assert any(src == "player-games" and exp is None and act is not None for src, exp, act in got), got  # stale log


def test_mismatched_shared_match_facts(rel: tuple[Path, Path]) -> None:
    data, release = rel
    path = _first(release, "player-games/*/2026.json")
    fx.edit_json(path, lambda d: d.update(match_facts="matches/2025/index.json"))
    fx.reseal(release)
    res = audit(data, release, checks=('release.public',))
    assert found(res, "release.shared_facts")


def test_altered_embedded_site_data(rel: tuple[Path, Path]) -> None:
    data, release = rel
    target = release / "site" / "data" / release.name / "matches" / "2026" / "index.json"
    raw = bytearray(target.read_bytes())
    raw[-2] = ord(" ")
    target.write_bytes(bytes(raw))
    res = audit(data, release, checks=('release.artifact',))
    assert {"release.embedded_mismatch", "release.seal_mismatch"} <= rules(res)
    assert res.outcome.value == "FAIL"


def test_extra_public_file(rel: tuple[Path, Path]) -> None:
    data, release = rel
    (release / "public" / "players" / "stray.json").write_text("{}\n")
    res = audit(data, release, checks=('release.artifact',))
    assert "release.public_extra" in rules(res)


def test_missing_and_truncated_public_files(rel: tuple[Path, Path]) -> None:
    data, release = rel
    victim = _first(release, "player-games/*/2025.json")
    victim.unlink()
    trunc = _first(release, "players/*.json", lambda d: "stat_names" in d)
    trunc.write_bytes(trunc.read_bytes()[:40])
    res = audit(data, release, checks=('release.artifact',))
    assert {"release.public_missing", "release.public_hash"} <= rules(res)


def test_unsafe_symlink_in_the_site(rel: tuple[Path, Path]) -> None:
    data, release = rel
    os.symlink("/etc/hostname", release / "site" / "leak.txt")
    res = audit(data, release, checks=('release.artifact',))
    assert "release.tree_entry_refused" in rules(res)


def test_validation_record_names_another_artifact(rel: tuple[Path, Path]) -> None:
    data, release = rel
    fx.edit_json(release / "validation.json", lambda d: d.update(checksums_sha256="0" * 64))
    res = audit(data, release, checks=('release.artifact',))
    assert "release.validation_binding" in rules(res)


def test_wrong_snapshot_in_release(rel: tuple[Path, Path]) -> None:
    from supercoach_via.domain.schemas import CheckOutcome, ValidationReport
    from supercoach_via.storage import snapshots

    data, release = rel
    m = snapshots.load_snapshot(data)
    venue = {"venue_id": "demo_extra_ground", "name": "Extra Ground", "source_names": '["Extra Ground"]'}
    cand = snapshots.apply_upserts(
        data, m, {"venues": [venue]}, clock=lambda: m.created_at, code_version="t", status=m.status
    )
    snapshots.promote(data, cand, ValidationReport(outcome=CheckOutcome.PASS))
    res = audit(data, release, checks=('release.artifact',))
    [f] = found(res, "release.snapshot_mismatch")
    assert f.actual == m.snapshot_id and f.expected == cand.manifest.snapshot_id


def test_stale_player_aggregate(rel: tuple[Path, Path]) -> None:
    data, release = rel
    path = _first(release, "players/*.json", lambda d: d.get("career_games", 0) > 3)

    def bump(d: Any) -> None:
        i = d["stat_names"].index("disposals")
        d["career"]["total"][i] = (d["career"]["total"][i] or 0) + 1

    fx.edit_json(path, bump)
    fx.reseal(release)
    res = audit(data, release, checks=('release.public',))
    assert any(f.field == "career.disposals.total" for f in found(res, "release.aggregate_mismatch"))


def test_player_index_membership_and_season_gap(rel: tuple[Path, Path]) -> None:
    data, release = rel
    idx = release / "public" / "players" / "index.json"

    def edit(d: Any) -> None:
        d["players"].pop(0)
        d["count"] -= 1
        p = next(p for p in d["players"] if len(p["seasons"]) >= 2)
        p["seasons"] = list(range(p["seasons"][0] - 1, p["seasons"][-1] + 1))  # adds a season never played

    fx.edit_json(idx, edit)
    fx.reseal(release)
    res = audit(data, release, checks=('release.public',))
    assert "release.index_membership" in rules(res)
    assert any(f.field == "seasons" for f in found(res, "release.player_index_value"))


def test_failing_audit_leaves_every_input_byte_identical(rel: tuple[Path, Path]) -> None:
    data, release = rel
    fx.edit_json(release / "validation.json", lambda d: d.update(checksums_sha256="0" * 64))
    before = fx.tree_digest(data, release)
    res = audit(data, release, checks=('release.artifact',))
    assert res.outcome.value == "FAIL"
    assert fx.tree_digest(data, release) == before


def test_key_encoding_used_for_paths() -> None:
    assert public_key("m:2026:gf:a:b:0").startswith("k.")


def test_boolean_and_number_are_not_interchangeable(rel: tuple[Path, Path]) -> None:
    """JSON true is not 1: Python's True == 1 must not hide a type change in published data."""
    data, release = rel
    idx = release / "public" / "players" / "index.json"

    def edit(d: Any) -> None:
        p = next(p for p in d["players"] if p["active"] is True)
        p["active"] = 1

    fx.edit_json(idx, edit)
    fx.reseal(release, expect_valid=False)  # O55-05: validate_release no longer coerces 1 -> true either
    res = audit(data, release, checks=("release.public", "release.validate"))
    assert any(f.field == "active" for f in found(res, "release.player_index_value"))
    assert found(res, "release.schema_invalid")


def test_validate_release_rules_are_mapped(rel: tuple[Path, Path]) -> None:
    """A private path leaked into a public file, with every hash recomputed: validate_release's rule catches it."""
    data, release = rel
    path = _first(release, "articles/*.json", lambda d: "body_html" in d or "html" in d)
    fx.edit_json(path, lambda d: d.update({k: v + " /home/someone/secret" for k, v in d.items() if k in ("body_html", "html")}))
    # validate_release refuses it too (its record is not PASS); the checker must say so either way
    with contextlib.suppress(AssertionError):
        fx.reseal(release)
    res = audit(data, release, checks=("release.validate", "release.artifact"))
    assert "release.private_content" in rules(res)
    assert res.outcome.value == "FAIL"
