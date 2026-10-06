"""The weekly AFL Tables reconciliation gate (scripts/reconciliation_gate.py): which seasons it audits, what blocks a
cycle, and the full flow (plan, capture, compare, optional fix and re-audit) on a fake site."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "reconciliation_gate.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("recon_gate", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DIFF = """diff --git a/data/player_data/able_ann_01071990_performance_details.csv b/data/player_data/able_ann_01071990_performance_details.csv
@@ -10,0 +11 @@
+Alpha,2026,11,Beta,24,W,5,10,3,4,14,1,,,,,,,,,,,,,,,,,,,2026-08-29
diff --git a/data/matches/matches_2026.csv b/data/matches/matches_2026.csv
@@ -200,0 +201 @@
+24,M.C.G.,2026-08-29 19:30,2026,50000,Alpha,1,1,2,2,3,3,4,4,Beta,1,1,2,2,3,3,4,4
diff --git a/data/player_data/old_bob_01011950_performance_details.csv b/data/player_data/old_bob_01011950_performance_details.csv
@@ -3 +3 @@
-Gamma,1970,3,Delta,3,L,9,1,1,2,1,,,,,,,,,,,,,,,,,,,,1970-04-18
+Gamma,1970,3,Delta,3,L,9,2,1,3,1,,,,,,,,,,,,,,,,,,,,1970-04-18
diff --git a/README.md b/README.md
+2031 is not a season row
"""


def test_changed_seasons_come_from_the_added_and_removed_rows_of_the_legacy_data_only() -> None:
    assert _load().changed_seasons(DIFF) == [1970, 2026]


def _report(**layers: Any) -> dict[str, Any]:
    return {"layers": layers, "source": {"capture_complete": True}}


def _layer(verdict: str, **counts: int) -> dict[str, Any]:
    return {"verdict": verdict, "finding_counts": {f"{k}|x": v for k, v in counts.items()}}


def test_a_legacy_fail_or_an_identity_problem_blocks_and_a_pass_or_a_capture_outage_does_not() -> None:
    g = _load()
    assert g.decide(_report(legacy_csv=_layer("PASS")))[0] == "pass"
    assert g.decide(_report(legacy_csv=_layer("FAIL", CELL_MISMATCH=1)))[0] == "block"
    assert g.decide(_report(legacy_csv=_layer("UNKNOWN", IDENTITY_CONFLICT=1)))[0] == "block"
    outage = _report(legacy_csv=_layer("UNKNOWN", CAPTURE_GAP=3))
    outage["source"]["capture_complete"] = False
    assert g.decide(outage)[0] == "warn"  # the network failed us, not the data: fail open, loudly
    assert g.decide(_report(legacy_csv=_layer("UNKNOWN", CELL_UNRESOLVED=2)))[0] == "warn"  # the source is incomplete


def test_the_full_gate_flow_blocks_a_bad_row_and_fix_corrects_it_then_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.scvia.unit import recon_e2e as E
    from tests.scvia.unit import recon_inputs as RI
    from tests.scvia.unit.test_reconciliation_e2e import two_season_world

    g = _load()
    world = two_season_world()

    def rows(layer: str, season: int) -> list[Any]:
        base = RI.local_rows(world, season, layer)
        return RI.with_cell(base, "a", "041520260305", "kicks", 99) if layer == "legacy_csv" else base

    e2e = E.build(tmp_path, world, monkeypatch=monkeypatch, local_rows=rows, capture=False)
    assert e2e.legacy_root is not None
    common = dict(data_root=e2e.data_root, legacy_root=e2e.legacy_root, through_date="2026-09-30",
                  client=e2e.site.client(), clock=e2e.site.clock)  # fmt: skip
    blocked = g.run_gate(seasons=[2026], run_dir=tmp_path / "gate1", fix=False, **common)
    assert blocked["decision"] == "block" and blocked["layers"]["legacy_csv"] == "FAIL"
    fixed = g.run_gate(seasons=[2026], run_dir=tmp_path / "gate2", fix=True, **common)
    assert fixed["decision"] == "pass" and fixed["fixed"]["cells_changed"] >= 1, fixed
    assert fixed["layers"]["legacy_csv"] == "PASS"
    summary = json.loads((tmp_path / "gate2" / "gate.json").read_text())
    assert summary["decision"] == "pass" and summary["seasons"] == [2026]


def test_a_missing_snapshot_data_root_skips_loudly_instead_of_breaking_the_cycle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A clone without the untracked var/ data root cannot pin a snapshot: warn and exit 0 (fail open, like a
    capture outage), never crash the weekly cycle on a machine-local precondition."""
    g = _load()
    rc = g.main(["gate", "--data-root", str(tmp_path / "absent"), "--season", "2026", "--runs-root", str(tmp_path),
                 "--status-file", str(tmp_path.parent / f"{tmp_path.name}-status.json")])  # fmt: skip
    assert rc == 0
    assert "WARN" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []  # no run directory was started


def test_seasons_are_read_from_the_working_tree_against_the_base_so_staged_but_uncommitted_rows_count(
    tmp_path: Path,
) -> None:
    """Production commits the scrape before the gate; a smoke run suppresses that commit but stages the rows. Both
    must see the same seasons, so the diff is base -> working tree, never base -> HEAD."""
    import subprocess

    g = _load()
    repo = tmp_path / "r"
    (repo / "data" / "matches").mkdir(parents=True)
    csv = repo / "data" / "matches" / "matches_1990.csv"
    csv.write_text("round_num,venue,date,year\n1,M.C.G.,1990-04-01 14:10,1990\n")
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    assert g.seasons_since(repo, "HEAD") == []
    with csv.open("a") as fh:
        fh.write("2,M.C.G.,1990-04-08 14:10,1990\n")
    assert g.seasons_since(repo, "HEAD") == [1990]  # unstaged
    run("add", "-A")
    assert g.seasons_since(repo, "HEAD") == [1990]  # staged, commit suppressed


def _status(root: Path) -> dict[str, Any]:
    return json.loads((root / "status.json").read_text())


def test_every_gate_exit_is_persisted_and_unverified_seasons_carry_to_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gaffer H1/M4: a skipped or warned gate used to log "passed" and leave nothing behind, so a re-scrape that
    reverted a correction in a fail-open week stayed reverted. Every exit now writes a status record, and seasons
    that were not verified are audited again on the next run until one passes."""
    g = _load()
    data = tmp_path / "data"
    status = ["--status-file", str(tmp_path / "status.json"), "--runs-root", str(tmp_path / "runs")]
    changed: list[int] = [2026]
    calls: list[list[int]] = []
    outcome = {"decision": "warn", "reason": "capture outage"}
    monkeypatch.setattr(g, "seasons_since", lambda root, base: list(changed))

    def fake_gate(*, seasons: list[int], **kw: Any) -> dict[str, Any]:
        calls.append(seasons)
        return {**outcome, "seasons": seasons}

    monkeypatch.setattr(g, "run_gate", fake_gate)
    # no accepted snapshot: skipped, and 2026 is pending
    assert g.main(["gate", "--data-root", str(data), *status]) == 0
    assert _status(tmp_path)["decision"] == "skip" and _status(tmp_path)["pending"] == [2026]
    data.mkdir()
    (data / "current.json").write_text("{}")
    # nothing changed this week, but the pending season is audited; the capture fails open: still pending
    changed[:] = []
    assert g.main(["gate", "--data-root", str(data), *status]) == 0
    assert calls == [[2026]] and _status(tmp_path)["decision"] == "warn" and _status(tmp_path)["pending"] == [2026]
    # a new season changes too: both are audited, and a pass clears the queue
    changed[:] = [2025]
    outcome.update(decision="pass", reason="agree")
    assert g.main(["gate", "--data-root", str(data), *status]) == 0
    assert calls[-1] == [2025, 2026] and _status(tmp_path)["pending"] == []
    # nothing changed and nothing pending: recorded as such, no audit run
    changed[:] = []
    assert g.main(["gate", "--data-root", str(data), *status]) == 0
    assert len(calls) == 2 and _status(tmp_path)["decision"] == "noop"
    # a block exits 1 and keeps its seasons pending
    changed[:] = [2026]
    outcome.update(decision="block", reason="mismatch")
    assert g.main(["gate", "--data-root", str(data), *status]) == 1
    assert _status(tmp_path)["decision"] == "block" and _status(tmp_path)["pending"] == [2026]
