"""Integrity checker, stored bytes and contracts (family A). Every negative case has a clean control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from supercoach_via.integrity.runner import AuditOptions, AuditResult, run_audit
from tests.scvia.unit import integrity_fixtures as fx

FAMILIES = ("storage", "contract")


def audit(root: Path, **kw: Any) -> AuditResult:
    opts = dict(data_root=root, snapshot="current", scope="data", as_of=fx.AS_OF, families=FAMILIES, keep_all=True)
    opts.update(kw)
    return run_audit(AuditOptions(**opts))  # type: ignore[arg-type]


def hits(res: AuditResult) -> set[tuple[str, str]]:
    return {(f.rule_id, f.entity) for f in res.findings if f.status == "open"}


def rules(res: AuditResult) -> set[str]:
    return {r for r, _ in hits(res)}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    r = tmp_path / "var"
    fx.build(r)
    return r


def test_clean_corpus_passes_with_no_findings(root: Path) -> None:
    res = audit(root)
    assert res.outcome.value == "PASS", [f.as_dict() for f in res.findings]
    assert res.findings == []
    checks = {c["check_id"]: c["status"] for c in res.report["checks"]}
    assert set(checks.values()) == {"PASS"}
    assert res.report["counts"]["rows_examined"] > 0


def test_missing_fragment(root: Path) -> None:
    m = fx.build(root)
    path = fx.fragment_path(root, m, "matches", "2026")
    path.chmod(0o644)
    path.unlink()
    res = audit(root)
    assert res.outcome.value == "FAIL"
    assert ("storage.fragment_missing", "fragment:matches/2026") in hits(res)


def test_tampered_fragment_same_size(root: Path) -> None:
    m = fx.build(root)
    path = fx.fragment_path(root, m, "players")
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.chmod(0o644)
    path.write_bytes(bytes(data))
    res = audit(root)
    assert ("storage.fragment_hash", "fragment:players") in hits(res)


def test_bad_manifest_identity(root: Path) -> None:
    fx.rewrite_manifest(root, lambda d: d.update(notes=["edited after sealing"]), fix_id=False)
    res = audit(root)
    assert "storage.manifest_identity" in rules(res)
    assert res.outcome.value == "FAIL"


def test_declared_row_count_disagrees_with_parquet(root: Path) -> None:
    def edit(d: dict[str, Any]) -> None:
        frag = d["tables"]["players"]["fragments"][0]
        frag["rows"] += 1
        d["tables"]["players"]["row_count"] += 1

    fx.rewrite_manifest(root, edit)
    res = audit(root)
    assert ("storage.fragment_rows", "fragment:players") in hits(res)


def test_table_row_count_disagrees_with_fragments(root: Path) -> None:
    fx.rewrite_manifest(root, lambda d: d["tables"]["clubs"].update(row_count=99))
    res = audit(root)
    assert ("storage.table_row_count", "table:clubs") in hits(res)


def test_malformed_schema_with_valid_hashes(root: Path) -> None:
    """A fragment whose column type changed, re-hashed and re-declared, still fails the contract."""
    m = fx.build(root)
    frag = next(f for f in m.tables["clubs"].fragments)
    t = pq.read_table(root / "fragments" / frag.path)
    bad = t.set_column(t.schema.get_field_index("first_season"), "first_season", pa.array(["1900", "1900"]))
    from supercoach_via.storage.snapshots import write_fragment

    ref = write_fragment(root, "clubs", bad, None)
    fx.rewrite_manifest(root, lambda d: d["tables"]["clubs"].update(fragments=[ref.model_dump()], row_count=2))
    res = audit(root)
    assert ("storage.fragment_schema", "fragment:clubs") in hits(res)


def test_misplaced_partition_record(root: Path) -> None:
    m = fx.build(root)
    frag = next(f for f in m.tables["matches"].fragments if f.partition == "2026")
    t = pq.read_table(root / "fragments" / frag.path)
    moved = t.set_column(t.schema.get_field_index("season"), "season", pa.array([2025, 2026], pa.int32()))
    from supercoach_via.storage.snapshots import write_fragment

    ref = write_fragment(root, "matches", moved, "2026")

    def edit(d: dict[str, Any]) -> None:
        d["tables"]["matches"]["fragments"] = [
            ref.model_dump() if f["partition"] == "2026" else f for f in d["tables"]["matches"]["fragments"]
        ]

    fx.rewrite_manifest(root, edit)
    res = audit(root)
    assert ("storage.partition_mismatch", "fragment:matches/2026") in hits(res)


def test_duplicate_key_across_fragments(root: Path) -> None:
    fx.rehash(root, lambda r: r["players"].append(dict(r["players"][0])))
    res = audit(root)
    assert ("contract.key_duplicate", "players:legacy:p1") in hits(res)


def test_null_required_value(root: Path) -> None:
    # the factory fills required nulls with a default; write the null directly instead
    m = fx.build(root)
    frag = next(f for f in m.tables["matches"].fragments if f.partition == "1970")
    t = pq.read_table(root / "fragments" / frag.path)
    i = t.schema.get_field_index("stage_id")
    nulls = t.set_column(i, pa.field("stage_id", pa.string(), nullable=True), pa.array([None, "r02"], pa.string()))
    from supercoach_via.storage.snapshots import write_fragment

    ref = write_fragment(root, "matches", nulls, "1970")
    fx.rewrite_manifest(
        root,
        lambda d: d["tables"]["matches"].update(
            fragments=[ref.model_dump() if f["partition"] == "1970" else f for f in d["tables"]["matches"]["fragments"]]
        ),
    )
    res = audit(root)
    assert ("contract.null_required", "table:matches") in hits(res)


def test_invalid_enum(root: Path) -> None:
    fx.rehash(root, lambda r: r["player_games"][0].update(date_quality="guessed"))
    res = audit(root)
    assert ("contract.enum", "table:player_games") in hits(res)


def test_non_finite_statistic(root: Path) -> None:
    fx.rehash(root, lambda r: r["player_games"][-1].update(time_on_ground_pct=float("nan")))
    res = audit(root)
    assert ("contract.non_finite", "table:player_games") in hits(res)


def test_conflicting_json_keys_in_json_column(root: Path) -> None:
    fx.rehash(root, lambda r: r["venues"][0].update(source_names='{"a": 1, "a": 2}'))
    res = audit(root)
    assert ("contract.json_column", "table:venues") in hits(res)


def test_unknown_table(root: Path) -> None:
    fx.rewrite_manifest(root, lambda d: d["tables"].update(mystery=d["tables"]["clubs"]))
    res = audit(root)
    assert ("storage.table_unknown", "table:mystery") in hits(res)


def test_pointer_and_manifest_path_mismatch(root: Path) -> None:
    ptr = json.loads((root / "current.json").read_text())
    ptr["manifest_path"] = "snapshots/" + "f" * 64 + ".json"
    (root / "current.json").write_text(json.dumps(ptr))
    res = audit(root)
    assert "storage.pointer" in rules(res)


def test_missing_snapshot_is_incomplete_not_pass(tmp_path: Path) -> None:
    res = audit(tmp_path / "empty")
    assert res.outcome.value in {"FAIL", "UNKNOWN"}
    assert res.outcome.value != "PASS"


def test_audit_never_writes_to_inputs(root: Path) -> None:
    fx.rehash(root, lambda r: r["player_games"][0].update(date_quality="guessed"))
    before = fx.tree_digest(root)
    res = audit(root)
    assert res.outcome.value == "FAIL"
    assert fx.tree_digest(root) == before
