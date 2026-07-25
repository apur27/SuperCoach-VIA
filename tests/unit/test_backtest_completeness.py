"""Backtest artifacts from a FATALed cycle must never count as complete.

The incident this encodes (2026-07-20): the weekly cycle ran, wrote backtest
artifacts at 15:57, then FATALed at 16:04 when the phantom-row gate caught a
mass-duplicated player corpus. The artifacts stayed on disk. The retry run picked
the newest `backtest_summary_*.csv`, concluded "last complete backtest: round 20",
and skipped re-scoring — so figures computed on a corpus we KNOW was bad were
committed and published.

Two consumers were fooled by the same orphan: the harness's start-round detector
(newest summary by filename) and the doc generators (newest summary by mtime).
Quarantining unmarked artifacts out of the directory fixes both at once, because
neither can glob what is no longer there.

The invariant: an artifact counts as complete only if a cycle explicitly marked
it complete. Absence of a mark means incomplete — fail closed, re-score.
"""
import json
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts import backtest_completeness as bc


def _write_run(bt_dir, ts, rounds, year=2026):
    """Create the full artifact family a single backtest invocation emits."""
    (bt_dir / f"backtest_summary_{ts}.csv").write_text(
        "year,round,n_players,mae\n"
        + "".join(f"{year},{r},300,3.9\n" for r in rounds)
    )
    (bt_dir / f"backtest_by_team_{ts}.csv").write_text("year,round,team,n,bias\n")
    (bt_dir / f"backtest_by_position_{ts}.csv").write_text("year,round,position,n\n")
    (bt_dir / f"backtest_run_{ts}.log").write_text("run log\n")
    for r in rounds:
        (bt_dir / f"prediction_vs_actual_round_{r}_{year}_{ts}.csv").write_text(
            "player,predicted,actual\n"
        )


@pytest.fixture
def bt_dir(tmp_path):
    d = tmp_path / "backtest"
    d.mkdir()
    return d


# ---------------------------------------------------------------- discovery


def test_timestamps_on_disk_finds_every_run(bt_dir):
    _write_run(bt_dir, "20260713_205008", [18, 19])
    _write_run(bt_dir, "20260720_155725", [20])
    assert bc.timestamps_on_disk(bt_dir) == {"20260713_205008", "20260720_155725"}


def test_no_manifest_means_nothing_is_complete(bt_dir):
    """Fail closed: an absent manifest must not be read as blanket approval."""
    _write_run(bt_dir, "20260720_155725", [20])
    assert bc.completed_timestamps(bt_dir) == set()
    assert bc.incomplete_timestamps(bt_dir) == {"20260720_155725"}


# ------------------------------------------------------- the core regression


def test_last_complete_round_ignores_unmarked_run(bt_dir):
    """THE BUG: round 20 exists on disk from a FATALed run; round 19 is complete.

    The detector must report 19, so the next cycle re-scores 20 on the fixed corpus.
    """
    _write_run(bt_dir, "20260713_205008", [18, 19])
    bc.mark_complete(bt_dir, ["20260713_205008"])
    _write_run(bt_dir, "20260720_155725", [20])  # FATALed run — never marked

    assert bc.last_complete_round(bt_dir, 2026) == 19


def test_last_complete_round_counts_marked_run(bt_dir):
    _write_run(bt_dir, "20260713_205008", [18, 19])
    _write_run(bt_dir, "20260720_155725", [20])
    bc.mark_complete(bt_dir, ["20260713_205008", "20260720_155725"])
    assert bc.last_complete_round(bt_dir, 2026) == 20


def test_last_complete_round_is_none_when_nothing_complete(bt_dir):
    _write_run(bt_dir, "20260720_155725", [20])
    assert bc.last_complete_round(bt_dir, 2026) is None


def test_last_complete_round_is_year_scoped(bt_dir):
    _write_run(bt_dir, "20250801_120000", [23], year=2025)
    bc.mark_complete(bt_dir, ["20250801_120000"])
    assert bc.last_complete_round(bt_dir, 2026) is None
    assert bc.last_complete_round(bt_dir, 2025) == 23


# ------------------------------------------------------------- quarantining


def test_sweep_moves_unmarked_artifacts_out_of_the_directory(bt_dir):
    _write_run(bt_dir, "20260713_205008", [19])
    bc.mark_complete(bt_dir, ["20260713_205008"])
    _write_run(bt_dir, "20260720_155725", [20])

    moved = bc.quarantine_incomplete(bt_dir)

    # Every member of the orphan family moves, including by_position and the log.
    assert len(moved) == 5
    q = bt_dir / "quarantine"
    assert (q / "backtest_summary_20260720_155725.csv").exists()
    assert (q / "backtest_by_position_20260720_155725.csv").exists()
    assert (q / "prediction_vs_actual_round_20_2026_20260720_155725.csv").exists()
    # The complete run is untouched.
    assert (bt_dir / "backtest_summary_20260713_205008.csv").exists()


def test_sweep_hides_orphans_from_the_doc_generators(bt_dir):
    """update_team_analysis.py / update_eval_surface.sh glob this directory.

    After a sweep the orphan must be invisible to that glob — this is what stops
    a tainted artifact being published.
    """
    _write_run(bt_dir, "20260713_205008", [19])
    bc.mark_complete(bt_dir, ["20260713_205008"])
    _write_run(bt_dir, "20260720_155725", [20])

    bc.quarantine_incomplete(bt_dir)

    visible = sorted(p.name for p in bt_dir.glob("backtest_summary_*.csv"))
    assert visible == ["backtest_summary_20260713_205008.csv"]


def test_sweep_bootstraps_instead_of_quarantining_everything(bt_dir):
    """A fresh clone has artifacts but no manifest yet.

    "No manifest" means never initialised — NOT "every run is an orphan". Treating
    it as the latter would quarantine the entire committed backtest history and
    reset the pipeline to round 1. Bootstrap adopts what is on disk instead.
    """
    _write_run(bt_dir, "20260713_205008", [18, 19])
    _write_run(bt_dir, "20260720_155725", [20])
    assert not (bt_dir / bc.MANIFEST_NAME).exists()

    moved = bc.quarantine_incomplete(bt_dir)

    assert moved == [], "bootstrap must not quarantine anything"
    assert (bt_dir / "backtest_summary_20260713_205008.csv").exists()
    assert bc.last_complete_round(bt_dir, 2026) == 20


def test_orphan_is_quarantined_once_initialised(bt_dir):
    """After bootstrap the protection is live: a later unmarked run IS an orphan."""
    _write_run(bt_dir, "20260713_205008", [19])
    bc.quarantine_incomplete(bt_dir)          # bootstrap
    _write_run(bt_dir, "20260720_155725", [20])  # a FATALed cycle's output

    moved = bc.quarantine_incomplete(bt_dir)

    assert len(moved) == 5
    assert bc.last_complete_round(bt_dir, 2026) == 19


def test_sweep_is_a_noop_when_everything_is_marked(bt_dir):
    _write_run(bt_dir, "20260713_205008", [19])
    bc.mark_complete(bt_dir, ["20260713_205008"])
    assert bc.quarantine_incomplete(bt_dir) == []
    assert not (bt_dir / "quarantine").exists() or not list(
        (bt_dir / "quarantine").iterdir()
    )


def test_sweep_does_not_collide_on_repeat(bt_dir):
    """A second orphan with the same name must not crash the sweep."""
    bc.mark_complete(bt_dir, [])  # initialise the manifest: past bootstrap
    _write_run(bt_dir, "20260720_155725", [20])
    bc.quarantine_incomplete(bt_dir)
    _write_run(bt_dir, "20260720_155725", [20])
    moved = bc.quarantine_incomplete(bt_dir)
    assert len(moved) == 5


def test_sweep_leaves_unrelated_files_alone(bt_dir):
    (bt_dir / "README.md").write_text("notes\n")
    (bt_dir / "completed_runs.json").write_text("{}")
    _write_run(bt_dir, "20260720_155725", [20])
    bc.quarantine_incomplete(bt_dir)
    assert (bt_dir / "README.md").exists()


# ------------------------------------------------------------------ marking


def test_mark_complete_is_additive_and_idempotent(bt_dir):
    _write_run(bt_dir, "20260713_205008", [19])
    _write_run(bt_dir, "20260720_155725", [20])
    bc.mark_complete(bt_dir, ["20260713_205008"])
    bc.mark_complete(bt_dir, ["20260713_205008", "20260720_155725"])
    assert bc.completed_timestamps(bt_dir) == {"20260713_205008", "20260720_155725"}


def test_mark_all_present_marks_every_on_disk_run(bt_dir):
    _write_run(bt_dir, "20260713_205008", [19])
    _write_run(bt_dir, "20260720_155725", [20])
    bc.mark_complete(bt_dir, bc.timestamps_on_disk(bt_dir))
    assert bc.incomplete_timestamps(bt_dir) == set()


def test_manifest_is_readable_json(bt_dir):
    _write_run(bt_dir, "20260713_205008", [19])
    bc.mark_complete(bt_dir, ["20260713_205008"])
    payload = json.loads((bt_dir / "completed_runs.json").read_text())
    assert "20260713_205008" in payload["completed"]


def test_corrupt_manifest_fails_closed(bt_dir):
    """A damaged manifest must not be silently read as 'everything complete'."""
    _write_run(bt_dir, "20260720_155725", [20])
    (bt_dir / "completed_runs.json").write_text("{not json")
    assert bc.completed_timestamps(bt_dir) == set()
    assert bc.last_complete_round(bt_dir, 2026) is None


# ---------------------------------------------------------------------- CLI


def _cli(bt_dir, *args):
    import subprocess

    return subprocess.run(
        [sys.executable, os.path.join(_REPO_ROOT, "scripts", "backtest_completeness.py"),
         "--dir", str(bt_dir), *args],
        capture_output=True, text=True,
    )


def test_cli_last_round_prints_empty_when_nothing_complete(bt_dir):
    _write_run(bt_dir, "20260720_155725", [20])
    res = _cli(bt_dir, "last-round", "--year", "2026")
    assert res.returncode == 0
    assert res.stdout.strip() == ""


def test_cli_last_round_prints_marked_round(bt_dir):
    _write_run(bt_dir, "20260713_205008", [18, 19])
    bc.mark_complete(bt_dir, ["20260713_205008"])
    _write_run(bt_dir, "20260720_155725", [20])
    res = _cli(bt_dir, "last-round", "--year", "2026")
    assert res.returncode == 0
    assert res.stdout.strip() == "19"


def test_cli_mark_and_sweep_roundtrip(bt_dir):
    _write_run(bt_dir, "20260713_205008", [19])
    assert _cli(bt_dir, "mark").returncode == 0
    _write_run(bt_dir, "20260720_155725", [20])
    res = _cli(bt_dir, "sweep")
    assert res.returncode == 0
    assert not (bt_dir / "backtest_summary_20260720_155725.csv").exists()
    assert (bt_dir / "backtest_summary_20260713_205008.csv").exists()
