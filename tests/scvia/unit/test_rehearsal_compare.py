"""The rehearsal comparison reads the snapshot the candidate already promoted."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from supercoach_via.cli import app
from supercoach_via.demo import write_demo_corpus

ROOT = Path(__file__).resolve().parents[3]
runner = CliRunner()


def _compare():
    path = ROOT / "docs/rewrite/evidence/rehearsal_compare.py"
    spec = importlib.util.spec_from_file_location("rehearsal_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_promoted_compare_uses_the_built_snapshot(tmp_path: Path) -> None:
    src, var = tmp_path / "src", tmp_path / "var"
    write_demo_corpus(src)
    imported = runner.invoke(app, ["import-legacy", "--source", str(src), "--data-root", str(var), "--json"])
    assert imported.exit_code == 0, imported.output
    snapshot_id = json.loads((var / "current.json").read_text())["snapshot_id"]
    legacy = tmp_path / "legacy"
    table = legacy / "data" / "top100"
    table.mkdir(parents=True)
    (table / "all_time_top_100.csv").write_text("player,all_time_score\n")
    public = tmp_path / "public"
    public.mkdir()
    module = _compare()

    (public / "release.json").write_text(json.dumps({"snapshot_id": "sha256:" + "ab" * 32}))
    with pytest.raises(SystemExit, match="promoted snapshot"):
        module.compare_promoted_snapshot(legacy, var, public)

    (public / "release.json").write_text(json.dumps({"snapshot_id": snapshot_id}))
    compared = module.compare_promoted_snapshot(legacy, var, public)
    assert compared["snapshot_id"] == snapshot_id
    assert compared["release_snapshot_id"] == snapshot_id
    assert compared["data_root"] == str(var)


def _base_report(module, season_row: dict) -> dict:
    return {
        "current_matches_selected": True,
        "all_time_numeric": {
            "same_players": True,
            "same_order": True,
            "max_abs_score_delta": 0,
            "n_legacy": 2,
            "n_new": 2,
        },
        "yearly": {"seasons": 1, "disagreeing": {"1897": season_row} if season_row else {}},
        "biography_csv": {"rows_legacy": 1, "rows_new": 1, "identical_rows": 1, "content_matches": True},
    }


def test_input_inventory_ignores_ranking_exports_and_sees_source_edits(tmp_path: Path) -> None:
    module = _compare()
    left, right = tmp_path / "left", tmp_path / "right"
    for root, text in ((left, "1"), (right, "1")):
        data = root / "data"
        (data / "top100").mkdir(parents=True)
        (data / "players.csv").write_text(text)
        (data / "top100" / "all_time_top_100.csv").write_text("stale-export")
    assert module.raw_input_inventory(left) == module.raw_input_inventory(right)
    (right / "data" / "players.csv").write_text("2")
    assert module.raw_input_inventory(left)["sha256"] != module.raw_input_inventory(right)["sha256"]
    dest = tmp_path / "already"
    dest.mkdir()
    with pytest.raises(SystemExit, match="existing"):
        module.regenerate_legacy_exports(left, dest)


def test_identical_biography_rows_match_when_a_name_is_repeated() -> None:
    module = _compare()
    rows = [["1", "Alex", "Club", "same prose"], ["2", "Alex", "Club", "other prose"]]
    assert module._biography_content_matches(rows, rows) is True
    changed = [["1", "Alex", "Club", "same prose"], ["2", "Alex", "Club", "edited prose"]]
    assert module._biography_content_matches(rows, changed) is False


def test_documented_repair_appends_only_on_the_isolated_copy(tmp_path: Path) -> None:
    module = _compare()
    source = tmp_path / "source" / "data" / "player_data"
    source.mkdir(parents=True)
    original = source / "perez_flynn_25082001_performance_details.csv"
    original.write_text("team,year\nHawthorn,2025\n")
    dest = tmp_path / "copy"
    (dest / "data" / "player_data").mkdir(parents=True)
    copied = dest / "data" / "player_data" / original.name
    copied.write_text(original.read_text())
    evidence = tmp_path / "b1"
    evidence.mkdir()
    (evidence / "fetch-manifest.json").write_text(json.dumps({"outcome": "pass"}))
    (evidence / "rows.jsonl").write_text(
        json.dumps(
            {
                "table": "player_games",
                "row": {"player_id": "legacy:perez_flynn_25082001", "season": 2026, "club_source_name": "Hawthorn",
                        "career_game_counter": 25, "goals": None},
            }
        )
        + "\n"
    )
    (evidence / "rows.jsonl").write_text(
        (evidence / "rows.jsonl").read_text()
        + json.dumps(
            {
                "table": "player_games",
                "row": {"player_id": "src:afltables:J.Jack_Dalton1", "season": 2026, "kicks": 5, "goals": 1},
            }
        )
        + "\n"
    )
    record = module.apply_documented_repair(dest, evidence, 2026)
    assert record["rows_appended"] == 2
    assert original.read_text() == "team,year\nHawthorn,2025\n"
    assert "2026" in copied.read_text()
    created = list((dest / "data" / "player_data").glob("repair_*_performance_details.csv"))
    assert len(created) == 1
    assert "5" in created[0].read_text()


def test_legacy_ranker_imports_when_the_checkout_root_is_not_on_the_path(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util; from pathlib import Path; "
            f"root = Path({str(ROOT)!r}); "
            "spec = importlib.util.spec_from_file_location('rehearsal_compare', root / 'docs/rewrite/evidence/rehearsal_compare.py'); "
            "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); "
            "ranker = mod._import_legacy_ranker(); "
            "print(ranker.__file__)",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("top_players_comprehensive.py")


def test_legacy_scan_skips_the_documented_duplicate_files(tmp_path: Path) -> None:
    module = _compare()
    folder = tmp_path / "player_data"
    folder.mkdir()
    kept = folder / "ablett_gary_01011984_performance_details.csv"
    kept.write_text("c")
    (folder / "green_william_08092005_performance_details.csv").write_text("a")
    (folder / "steele_roan_19092002_performance_details.csv").write_text("b")
    found = module.legacy_scan_glob(str(folder / "*performance_details.csv"))
    assert found == [str(kept)]


def test_nontied_replacement_and_empty_tables_fail() -> None:
    module = _compare()
    swapped = module._compare_scored([("a", 100.0), ("b", 90.0)], [("a", 100.0), ("c", 1.0)])
    verdict = module.comparison_verdict(_base_report(module, swapped))
    assert verdict["ok"] is False
    assert any("1897" in reason for reason in verdict["reasons"])
    empty_year = _base_report(module, {})
    empty_year["yearly"] = {"seasons": 0, "disagreeing": {}}
    assert module.comparison_verdict(empty_year)["ok"] is False
    bare = {
        "all_time_numeric": {"same_players": False, "max_abs_score_delta": 0, "only_legacy": ["b"], "only_new": ["c"]},
        "yearly": {"seasons": 1, "disagreeing": {}},
    }
    assert module.comparison_verdict(bare)["ok"] is False


def test_tied_cutoff_swap_passes_and_a_biography_change_fails() -> None:
    module = _compare()
    tied = module._compare_scored([("a", 100.0), ("b", 10.0)], [("a", 100.0), ("c", 10.0)])
    assert module.comparison_verdict(_base_report(module, tied))["ok"] is True
    changed = _base_report(module, tied)
    changed["biography_csv"] = {**changed["biography_csv"], "content_matches": False, "identical_rows": 0}
    assert module.comparison_verdict(changed)["ok"] is False


def test_common_score_change_and_nontied_reorder_fail() -> None:
    module = _compare()
    changed_common = module._compare_scored([("a", 100.0), ("b", 90.0)], [("a", 200.0), ("c", 90.0)])
    assert module._season_ok(changed_common) is False
    assert module.comparison_verdict(_base_report(module, changed_common))["ok"] is False
    reordered = module._compare_scored([("a", 10.0), ("b", 9.0)], [("b", 9.0), ("a", 10.0)])
    assert module._season_ok(reordered) is False
    assert module.comparison_verdict(_base_report(module, reordered))["ok"] is False


def test_a_short_legacy_table_and_a_reordered_cut_swap_fail(tmp_path: Path) -> None:
    module = _compare()
    yearly = tmp_path / "yearly"
    yearly.mkdir()
    (yearly / "year_1897.csv").write_text("player,score\na,100\nb,90\n")
    summary = module._yearly_from_files(
        yearly, {1897}, {1897: [("a", 100.0), ("b", 90.0), ("c", 80.0)]}
    )
    assert "1897" in summary["disagreeing"]
    report = _base_report(module, {})
    report["yearly"] = summary
    assert module.comparison_verdict(report)["ok"] is False
    hidden = module._compare_scored(
        [("a", 100.0), ("b", 90.0), ("c", 80.0)],
        [("b", 90.0), ("a", 100.0), ("d", 80.0)],
    )
    assert module._season_ok(hidden) is False
    assert module.comparison_verdict(_base_report(module, hidden))["ok"] is False


def test_one_missing_expected_season_and_empty_tables_fail(tmp_path: Path) -> None:
    module = _compare()
    yearly = tmp_path / "yearly"
    yearly.mkdir()
    (yearly / "year_1897.csv").write_text("player,score\na,1\n")
    summary = module._yearly_from_files(yearly, {1897, 1898}, {1897: [("a", 1.0)], 1898: [("a", 1.0)]})
    assert summary["missing_expected"] == ["1898"]
    report = _base_report(module, {})
    report["yearly"] = summary
    verdict = module.comparison_verdict(report)
    assert verdict["ok"] is False
    assert any("missing expected season" in reason or "1898" in reason for reason in verdict["reasons"])
    empty = {
        "current_matches_selected": True,
        "all_time_numeric": {"n_legacy": 0, "n_new": 0, "same_players": True, "max_abs_score_delta": 0},
        "yearly": {"seasons": 0, "disagreeing": {}, "missing_expected": []},
        "biography_csv": {"rows_legacy": 0, "rows_new": 0, "content_matches": False},
    }
    assert module.comparison_verdict(empty)["ok"] is False


def _corrected_replay(tmp_path: Path, suspect_goals: int = 0, *, missing_metadata: bool = False):
    import hashlib
    import shutil
    from datetime import UTC, datetime

    from supercoach_via import pipeline
    from supercoach_via.settings import RunContext, Settings
    from supercoach_via.storage.queries import SnapshotQuery
    from supercoach_via.storage.snapshots import load_snapshot
    from tests.scvia.unit import integrity_fixtures as fx
    from tests.scvia.unit.test_corrections import DRAW, _replay_corpus

    module = _compare()
    source, dest, root = tmp_path / "source", tmp_path / "legacy", tmp_path / "data"
    _replay_corpus(root, suspect_goals)
    with SnapshotQuery(root, load_snapshot(root)) as q:
        rows = {name: q.arrow(f'SELECT * FROM "{name}"').to_pylist() for name in load_snapshot(root).tables}
    suspect = next(g for g in rows["player_games"] if g["match_id"] == DRAW and g["player_id"] == "legacy:p5")
    if missing_metadata:
        suspect.update(match_date=None, career_game_counter=None, career_game_counter_token=None)
    rel = "data/player_data/p5_performance_details.csv"
    file = source / rel
    file.parent.mkdir(parents=True)
    cells = ["" if suspect.get(field) is None else str(suspect[field]) for _, field in module._REPAIR_CSV]
    line = ",".join(cells)
    file.write_text(",".join(column for column, _ in module._REPAIR_CSV) + "\n" + line + "\n")
    suspect.update(source_path=rel, source_row=1, source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                   revision_id="rev:" + hashlib.sha256(line.encode()).hexdigest()[:32])
    before = fx.build(root, rows)
    result = pipeline.apply_corrections(
        RunContext(settings=Settings(data_root=root), clock=lambda: datetime(2026, 9, 30, tzinfo=UTC)), seasons=[],
    )
    assert result.exit_code == 0 and result.promoted, result.message
    shutil.copytree(source / "data", dest / "data")
    return module, source, dest, root, before.snapshot_id, result.snapshot_id


def test_legacy_copy_removes_only_a_verified_replay_quarantine(tmp_path: Path) -> None:
    module, source, dest, root, before, after = _corrected_replay(tmp_path)
    original = (source / "data/player_data/p5_performance_details.csv").read_bytes()
    record = module.apply_verified_replay_corrections(source, dest, root, before, after)
    assert record["source_snapshot_id"] == before and record["corrected_snapshot_id"] == after
    assert record["rows_removed"] == 1 and record["rows_relinked"] == 0
    assert record["removals"][0]["source_row"] == 1
    assert (source / "data/player_data/p5_performance_details.csv").read_bytes() == original
    assert len((dest / "data/player_data/p5_performance_details.csv").read_text().splitlines()) == 1


def test_statistic_preserving_replay_relink_keeps_legacy_csv_bytes(tmp_path: Path) -> None:
    module, source, dest, root, before, after = _corrected_replay(tmp_path, suspect_goals=2)
    file = dest / "data/player_data/p5_performance_details.csv"
    original = file.read_bytes()
    record = module.apply_verified_replay_corrections(source, dest, root, before, after)
    assert record["rows_removed"] == 0 and record["rows_relinked"] == 1
    assert file.read_bytes() == original


def test_replay_quarantine_can_preserve_missing_csv_date_and_career_counter(tmp_path: Path) -> None:
    module, source, dest, root, before, after = _corrected_replay(tmp_path, missing_metadata=True)
    record = module.apply_verified_replay_corrections(source, dest, root, before, after)
    assert record["rows_removed"] == 1


@pytest.mark.parametrize("defect", [
    "extra_delete", "changed_stat", "source_edit", "scratch_edit", "quarantine_raw", "source_alias",
    "quarantine_candidates", "quarantine_table", "quarantine_source", "unrelated_snapshot",
])
def test_legacy_correction_alignment_refuses_unverified_changes(tmp_path: Path, defect: str) -> None:
    from datetime import UTC, datetime

    from supercoach_via.storage import snapshots
    from supercoach_via.storage.queries import SnapshotQuery

    module, source, dest, root, before, after = _corrected_replay(tmp_path)
    selected = snapshots.load_snapshot(root, after)
    with SnapshotQuery(root, selected, tables={"player_games", "quarantine"}) as q:
        game = q.arrow("SELECT * FROM player_games ORDER BY match_id, player_id LIMIT 1").to_pylist()[0]
        quarantined = q.arrow("SELECT * FROM quarantine WHERE reason = 'replayed_draw_link_unresolved'").to_pylist()[0]
    if defect == "unrelated_snapshot":
        builder = snapshots.SnapshotBuilder(root, clock=lambda: datetime(2026, 9, 30, tzinfo=UTC), code_version="test")
        for name, entry in selected.tables.items():
            for fragment in entry.fragments:
                builder.reuse(name, fragment)
        after = builder.finish(status=selected.status).manifest.snapshot_id
    elif defect in ("extra_delete", "changed_stat", "quarantine_raw", "quarantine_candidates",
                    "quarantine_table", "quarantine_source"):
        ups, deletes = {"player_games": []}, None
        if defect == "extra_delete":
            deletes = {"player_games": [game]}
        elif defect == "changed_stat":
            game["kicks"] += 1
            ups["player_games"] = [game]
        else:
            field, value = {
                "quarantine_raw": ("raw", "{}"),
                "quarantine_candidates": ("candidates", '["not-a-replay"]'),
                "quarantine_table": ("table_name", "matches"),
                "quarantine_source": ("source_sha256", "0" * 64),
            }[defect]
            quarantined[field] = value
            ups["quarantine"] = [quarantined]
        selected = snapshots.apply_upserts(
            root, selected, ups, deletes=deletes, allow_empty=True,
            clock=lambda: datetime(2026, 9, 30, tzinfo=UTC), code_version="test", status=selected.status,
        ).manifest
        after = selected.snapshot_id
    else:
        file = (source if defect == "source_edit" else dest) / "data/player_data/p5_performance_details.csv"
        if defect == "source_alias":
            file.unlink()
            file.hardlink_to(source / "data/player_data/p5_performance_details.csv")
        else:
            file.write_text(file.read_text() + "modified\n")
    scratch_before = {str(p): p.read_bytes() for p in dest.rglob("*") if p.is_file()}
    with pytest.raises(SystemExit, match="compare:"):
        module.apply_verified_replay_corrections(source, dest, root, before, after)
    assert scratch_before == {str(p): p.read_bytes() for p in dest.rglob("*") if p.is_file()}
