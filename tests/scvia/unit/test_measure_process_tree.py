"""scripts/measure_process_tree.py: wall time, process-tree RSS and exit-status preservation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "measure_process_tree.py"


def test_records_peak_tree_rss_including_children_and_preserves_exit_status(tmp_path: Path) -> None:
    out = tmp_path / "m.json"
    child = "import subprocess,sys,time; subprocess.run([sys.executable,'-c','b=bytearray(60*2**20); import time; time.sleep(1)']); sys.exit(3)"
    res = subprocess.run([sys.executable, str(SCRIPT), str(out), "--", sys.executable, "-c", child], check=False)
    doc = json.loads(out.read_text())
    assert res.returncode == 3 == doc["exit_code"]
    assert doc["peak_tree_rss_mib"] >= 60 and doc["processes_at_peak"] >= 2 and doc["wall_seconds"] >= 1.0


def test_usage_error_is_exit_2(tmp_path: Path) -> None:
    assert (
        subprocess.run(
            [sys.executable, str(SCRIPT), str(tmp_path / "x.json")], check=False, capture_output=True
        ).returncode
        == 2
    )
