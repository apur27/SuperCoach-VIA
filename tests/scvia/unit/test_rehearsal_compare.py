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
