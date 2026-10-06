"""Plan building, input pinning and output-path safety (DESIGN section 5; T22, T23, T29)."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import pytest

from supercoach_via.reconciliation import inventory as inv
from supercoach_via.reconciliation.schema import Plan, canonical_dump
from tests.scvia.unit import integrity_fixtures as fx


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    fx.build(root)
    return root


@pytest.fixture
def legacy_root(tmp_path: Path) -> Path:
    root = tmp_path / "legacy"
    (root / "data" / "player_data").mkdir(parents=True)
    (root / "data" / "matches").mkdir(parents=True)
    (root / "data" / "player_data" / "able_ann_01011990_performance_details.csv").write_text("team,year\nA,2020\n")
    (root / "data" / "player_data" / "able_ann_01011990_personal_details.csv").write_text("first_name\nAnn\n")
    (root / "data" / "matches" / "matches_2020.csv").write_text("round_num\n1\n")
    (root / "data" / "unrelated.csv").write_text("x\n")
    return root


def _plan(data_root: Path, legacy_root: Path | None, run_dir: Path, **kw: object) -> Plan:
    args: dict[str, object] = {"through_date": "2026-09-30", "scope": "all", **kw}
    return inv.build_plan(
        data_root=data_root,
        snapshot="current",
        legacy_root=legacy_root,
        run_dir=run_dir,
        **args,  # type: ignore[arg-type]
    )


def test_plan_pins_the_resolved_snapshot_and_every_fragment(data_root: Path, tmp_path: Path) -> None:
    plan = _plan(data_root, None, tmp_path / "run")
    snap = plan.inputs.snapshot
    assert snap.snapshot_id.startswith("sha256:") and snap.selector == "current"
    assert snap.fragments["player_games"].keys() >= {"1970", "2026"}
    assert all(len(h) == 64 for parts in snap.fragments.values() for h in parts.values())
    assert plan.inputs.legacy is None
    assert plan.scope.full_population and plan.scope.through_date == "2026-09-30"
    assert plan.source.reference_mode == "observed_current"


def test_plan_identity_ignores_operational_paths_and_is_deterministic(data_root: Path, tmp_path: Path) -> None:
    a = _plan(data_root, None, tmp_path / "run-a")
    b = _plan(data_root, None, tmp_path / "run-b")
    assert a.operational.run_dir != b.operational.run_dir
    assert a.plan_id == b.plan_id and a.capture_identity == b.capture_identity
    assert a.operational.python == platform.python_version()


def test_plan_identity_changes_with_scope_inputs_and_policies(
    data_root: Path, legacy_root: Path, tmp_path: Path
) -> None:
    base = _plan(data_root, legacy_root, tmp_path / "r")
    other_date = inv.build_plan(
        data_root=data_root, snapshot="current", legacy_root=legacy_root, through_date="2026-08-30",
        scope="all", run_dir=tmp_path / "r",
    )  # fmt: skip
    assert other_date.plan_id != base.plan_id and other_date.capture_identity != base.capture_identity
    (legacy_root / "data" / "matches" / "matches_2020.csv").write_text("round_num\n2\n")
    changed = _plan(data_root, legacy_root, tmp_path / "r")
    assert changed.inputs.legacy.content_sha256 != base.inputs.legacy.content_sha256  # type: ignore[union-attr]
    assert changed.inputs.legacy.membership_sha256 == base.inputs.legacy.membership_sha256  # type: ignore[union-attr]
    assert changed.plan_id != base.plan_id
    assert changed.capture_identity == base.capture_identity  # comparison inputs do not change which pages are fetched


def test_legacy_inventory_counts_only_the_files_the_audit_reads(legacy_root: Path) -> None:
    pin = inv.pin_legacy(legacy_root)
    assert (pin.player_files, pin.personal_files, pin.match_files) == (1, 1, 1)
    (legacy_root / "data" / "player_data" / "late_bob_02021991_performance_details.csv").write_text("team\nB\n")
    assert inv.pin_legacy(legacy_root).membership_sha256 != pin.membership_sha256  # membership, not just content


def test_symlinked_legacy_file_is_refused(legacy_root: Path, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.csv"
    target.write_text("x\n")
    os.symlink(target, legacy_root / "data" / "player_data" / "evil_sam_03031992_performance_details.csv")
    with pytest.raises(inv.PlanError, match="symlink"):
        inv.pin_legacy(legacy_root)


def test_unknown_or_unverifiable_snapshot_is_refused(data_root: Path, tmp_path: Path) -> None:
    with pytest.raises(inv.PlanError, match="snapshot"):
        inv.build_plan(
            data_root=data_root, snapshot="sha256:" + "0" * 64, legacy_root=None, through_date="2026-09-30",
            scope="all", run_dir=tmp_path / "r",
        )  # fmt: skip
    frag = next((data_root / "fragments" / "player_games").rglob("*.parquet"))
    frag.chmod(0o644)
    frag.write_bytes(frag.read_bytes() + b"x")
    with pytest.raises(inv.PlanError, match="snapshot"):
        _plan(data_root, None, tmp_path / "r2")


def test_write_plan_is_atomic_idempotent_and_refuses_an_incompatible_plan(data_root: Path, tmp_path: Path) -> None:
    run = tmp_path / "run"
    plan = _plan(data_root, None, run)
    path = inv.write_plan(plan)
    assert path == run / "plan.json" and path.read_bytes() == canonical_dump(plan)
    inv.write_plan(plan)  # same plan again is a no-op
    incompatible = inv.build_plan(
        data_root=data_root, snapshot="current", legacy_root=None, through_date="2026-08-01", scope="all", run_dir=run
    )
    with pytest.raises(inv.PlanError, match="incompatible"):
        inv.write_plan(incompatible)
    assert json.loads(path.read_text())["plan_id"] == plan.plan_id


def test_replanning_comparison_inputs_keeps_history_when_capture_identity_matches(
    data_root: Path, legacy_root: Path, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    first = _plan(data_root, legacy_root, run)
    inv.write_plan(first)
    (legacy_root / "data" / "matches" / "matches_2020.csv").write_text("round_num\n9\n")
    second = _plan(data_root, legacy_root, run)
    inv.write_plan(second)
    assert json.loads((run / "plan.json").read_text())["plan_id"] == second.plan_id
    assert (run / "plans" / f"plan-{first.plan_id[:16]}.json").is_file()


def test_run_dir_inside_an_input_root_is_refused(data_root: Path, legacy_root: Path) -> None:
    with pytest.raises(inv.PlanError, match="inside an input root"):
        _plan(data_root, legacy_root, data_root / "fragments" / "run")
    with pytest.raises(inv.PlanError, match="inside an input root"):
        _plan(data_root, legacy_root, legacy_root / "data" / "player_data" / "audit")
    _plan(data_root, legacy_root, legacy_root / "var" / "run")  # elsewhere under the legacy root is fine


def test_run_dir_symlinked_into_an_input_root_is_refused(data_root: Path, tmp_path: Path) -> None:
    link = tmp_path / "link"
    os.symlink(data_root, link)
    with pytest.raises(inv.PlanError, match="inside an input root"):
        _plan(data_root, None, link / "run")


def test_hardlinked_output_alias_of_an_input_file_is_refused(data_root: Path, tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    victim = next((data_root / "fragments").rglob("*.parquet"))
    os.link(victim, run / "plan.json")
    plan = _plan(data_root, None, run)
    with pytest.raises(inv.PlanError, match=r"hard link|alias"):
        inv.write_plan(plan)
    assert victim.stat().st_nlink == 2 and victim.read_bytes()[:4] == b"PAR1"  # input bytes untouched


def test_sample_scope_is_never_a_full_population(data_root: Path, tmp_path: Path) -> None:
    plan = inv.build_plan(
        data_root=data_root, snapshot="current", legacy_root=None, through_date="2026-09-30", scope="sample",
        run_dir=tmp_path / "r", sample_profiles=["https://afltables.com/afl/stats/players/S/Scott_Pendlebury.html"],
    )  # fmt: skip
    assert plan.scope.population == "sample" and plan.scope.full_population is False
    with pytest.raises(inv.PlanError, match="sample"):
        inv.build_plan(
            data_root=data_root, snapshot="current", legacy_root=None, through_date="2026-09-30", scope="sample",
            run_dir=tmp_path / "r",
        )  # fmt: skip


def test_through_date_must_be_a_real_calendar_date(data_root: Path, tmp_path: Path) -> None:
    with pytest.raises(inv.PlanError, match="through-date"):
        inv.build_plan(
            data_root=data_root, snapshot="current", legacy_root=None, through_date="2026-02-30", scope="all",
            run_dir=tmp_path / "r",
        )  # fmt: skip


def test_a_plan_written_before_fields_were_appended_keeps_its_identity(
    data_root: Path, legacy_root: Path, tmp_path: Path
) -> None:  # the frozen full-capture plan must stay loadable and keep its capture identity
    import hashlib

    from supercoach_via.integrity.report import canonical_bytes

    plan = _plan(data_root, legacy_root, tmp_path / "run")
    path = inv.write_plan(plan)
    doc = json.loads(path.read_text())
    doc["scope"].pop("seasons", None)
    doc["code"].pop("capture_files_current", None)
    doc["inputs"]["legacy"].pop("award_files", None)
    # the identities exactly as the older code computed them: over a body that never had those fields
    body = {k: v for k, v in doc.items() if k not in ("plan_id", "operational")}
    cap = {
        "schema_version": doc["schema_version"],
        "capture_files": doc["code"]["capture_files"],
        "scope": doc["scope"],
        "source": doc["source"],
        "source_policy_sha256": doc["policies"]["source_policy_sha256"],
    }
    doc["plan_id"] = hashlib.sha256(canonical_bytes(body)).hexdigest()
    doc["capture_identity"] = hashlib.sha256(canonical_bytes(cap)).hexdigest()
    path.write_text(json.dumps(doc))
    again = inv.load_plan(path)
    assert again.plan_id == doc["plan_id"] and again.capture_identity == doc["capture_identity"]


def test_a_seasons_scope_names_its_seasons_and_is_never_a_full_population(data_root: Path, tmp_path: Path) -> None:
    plan = _plan(data_root, None, tmp_path / "run", scope="seasons", seasons=[2026, 2025, 2026])
    assert plan.scope.population == "seasons" and plan.scope.seasons == [2025, 2026]
    assert plan.scope.full_population is False
    with pytest.raises(inv.PlanError, match="season"):
        _plan(data_root, None, tmp_path / "run2", scope="seasons", seasons=[])
    other = _plan(data_root, None, tmp_path / "run3", scope="seasons", seasons=[2024])
    assert other.capture_identity != plan.capture_identity  # different seasons are a different capture


def test_a_plan_can_bind_to_the_verified_capture_code_of_an_existing_archive(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # DESIGN section 15, A11
    old = _plan(data_root, None, tmp_path / "old")
    old_path = inv.write_plan(old)
    real = inv.code_identity

    def changed() -> object:
        ident = real()
        files = dict(ident.capture_files)
        files["reconciliation/capture.py"] = "0" * 64  # the capture code has since been edited
        return ident.model_copy(update={"capture_files": files})

    monkeypatch.setattr(inv, "code_identity", changed)
    fresh = _plan(data_root, None, tmp_path / "new")
    assert fresh.capture_identity != old.capture_identity
    bound = _plan(data_root, None, tmp_path / "bound", capture_plan=old_path)
    assert bound.capture_identity == old.capture_identity
    assert bound.code.capture_files == old.code.capture_files
    assert bound.code.capture_files_current["reconciliation/capture.py"] == "0" * 64  # both are recorded
    assert inv.load_plan(inv.write_plan(bound)).capture_identity == old.capture_identity
    with pytest.raises(inv.PlanError, match="scope"):
        _plan(data_root, None, tmp_path / "bad", capture_plan=old_path, through_date="2026-08-31")
