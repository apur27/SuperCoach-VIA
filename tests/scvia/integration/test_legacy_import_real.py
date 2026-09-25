"""Real-corpus legacy import + validation (reads data/ read-only; writes only under tmp).

What an operator does if this fires: it means the checked-in corpus and the importer's
accounting disagree (a file/row was dropped, a new family appeared, or a new
current-season defect exists). Read the failing assertion's path/season, inspect the
import report, and fix the importer or repair the data through a refresh -- never relax
the accounting assertions.
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest

from supercoach_via.domain import ids
from supercoach_via.domain.schemas import CheckOutcome, DatasetStatus
from supercoach_via.ingest import legacy, reconcile
from supercoach_via.settings import RunContext, Settings
from supercoach_via.storage.queries import SnapshotQuery

REPO = Path(__file__).resolve().parents[3]

#: Current-season corpus gaps known at 2026-09-25 (players named in 2026 lineups whose
#: 2026 player rows are absent from data/player_data). The check below allows the set to
#: SHRINK (data repaired) but fails if any new blocking issue appears.
KNOWN_2026_GAP_FILES = {
    "data/lineups/team_lineups_hawthorn.csv",  # Flynn Perez, Jack Dalton
    "data/lineups/team_lineups_port_adelaide.csv",  # Will Brodie
}
KNOWN_2026_GOAL_KEYS = {
    "m:2026:r17:greater_western_sydney:hawthorn:0|hawthorn",
    "m:2026:r18:hawthorn:melbourne:0|hawthorn",
}


@pytest.fixture(scope="module")
def real(tmp_path_factory: pytest.TempPathFactory) -> legacy.DatasetCandidate:
    root = tmp_path_factory.mktemp("real")
    return legacy.import_legacy(REPO, RunContext(settings=Settings(data_root=root / "var")))


@pytest.fixture(scope="module")
def report(real: legacy.DatasetCandidate) -> reconcile.ValidationReport:
    return reconcile.validate_dataset(real, reconcile.load_policy())


def _q(c: legacy.DatasetCandidate, sql: str) -> list[tuple[object, ...]]:
    with SnapshotQuery(c.data_root, c.candidate.manifest) as q:
        return q.rows(sql)


def test_every_data_file_accounted(real: legacy.DatasetCandidate) -> None:
    on_disk = set()
    for dirpath, _dirs, files in os.walk(REPO / "data"):
        for fn in files:
            on_disk.add((Path(dirpath) / fn).relative_to(REPO).as_posix())
    files = {f["path"]: f for f in real.report["files"]}
    assert on_disk <= set(files)
    for f in files.values():
        assert f["disposition"] in {"imported", "ignored", "quarantined"}, f
        assert f["disposition"] != "ignored" or f["reason"], f
        if f["rows_read"] is not None:
            assert f["rows_read"] == f["rows_imported"] + f["rows_quarantined"], f["path"]
    assert not [f for f in files.values() if f["family"] == "unknown"], "new unrecognized legacy files"


def test_per_season_rows_reconcile(real: legacy.DatasetCandidate) -> None:
    acc = real.report["row_accounting"]["player_games"]
    assert acc["read"] == acc["imported"] + acc["quarantined"]
    for season, c in acc["by_season"].items():
        assert c["read"] == c["imported"] + c["quarantined"], season
    m = real.report["row_accounting"]["matches"]
    assert m["read"] == m["imported"] + m["quarantined"]
    lu = real.report["row_accounting"]["lineups"]
    assert lu["tokens_read"] == lu["tokens_resolved"] + lu["tokens_quarantined"]
    assert lu["token_resolution_rate"] > 0.999
    (n,) = _q(real, "SELECT count(*) FROM player_games")[0]
    assert n == acc["imported"]


def test_2026_season_matches_refresh_report(real: legacy.DatasetCandidate) -> None:
    # DATA_REFRESH.md: 217 completed 2026 matches, latest 19 September
    rows = _q(
        real,
        "SELECT count(*), max(match_date), count(*) FILTER (WHERE status='complete') FROM matches WHERE season=2026",
    )
    assert rows == [(217, date(2026, 9, 19), 217)]
    wf = _q(real, "SELECT count(*) FROM matches WHERE season=2026 AND stage_id='wf' AND stage_type='final'")
    assert wf == [(2,)]


def test_extra_time_finals_repaired(real: legacy.DatasetCandidate) -> None:
    # 1994 QF, 2007 SF, 2017 EF: score written into team_2 name; opponent from two-way player evidence
    rows = _q(
        real,
        "SELECT match_id, status, home_score, away_score, (SELECT count(*) FROM player_games g "
        "WHERE g.match_id = m.match_id) FROM matches m WHERE match_id IN ("
        "'m:1994:qf:hawthorn:north_melbourne:0', 'm:2007:sf:collingwood:west_coast:0', "
        "'m:2017:ef:port_adelaide:west_coast:0') ORDER BY 1",
    )
    assert [(r[0], r[1], r[2], r[3]) for r in rows] == [
        ("m:1994:qf:hawthorn:north_melbourne:0", "unknown", None, None),
        ("m:2007:sf:collingwood:west_coast:0", "unknown", None, None),
        ("m:2017:ef:port_adelaide:west_coast:0", "unknown", None, None),
    ]
    assert [r[4] for r in rows] == [42, 44, 44]
    assert real.report["row_accounting"]["matches"]["quarantined"] == 0
    assert _q(real, "SELECT count(*) FROM quality_issues WHERE rule_id='source_row_malformed'") == [(3,)]


def test_duplicate_identities_quarantined(real: legacy.DatasetCandidate) -> None:
    dups = dict(
        _q(real, "SELECT player_id, canonical_player_id FROM players WHERE identity_status='quarantined_duplicate'")
    )
    assert dups == {
        ids.player_id_for_slug(d.duplicate_slug): ids.player_id_for_slug(d.canonical_slug) for d in ids.KNOWN_DUPLICATES
    }
    assert _q(
        real,
        "SELECT count(*) FROM player_games WHERE player_id IN ('legacy:green_william_08092005', "
        "'legacy:steele_roan_19092002')",
    ) == [(0,)]
    (nq,) = _q(real, "SELECT count(*) FROM quarantine WHERE reason='duplicate_identity'")[0]
    assert nq == 3  # green_william 1 row + steele_roan_19092002 2 rows


def test_registry_matches_evidence() -> None:
    ev = json.loads((REPO / "docs/rewrite/evidence/refresh-sources.json").read_text())
    dup_ev = {
        (Path(d["path"]).name, Path(d["canonical_path"]).name, d["source"]) for d in ev["legacy_duplicates_preserved"]
    }
    dup_reg = {
        (f"{d.duplicate_slug}_performance_details.csv", f"{d.canonical_slug}_performance_details.csv", d.source_url)
        for d in ids.KNOWN_DUPLICATES
    }
    assert dup_ev == dup_reg
    alias_ev = {
        (Path(p["path"]).name, p["verified_source_name"], p["source"])
        for p in ev["players"]
        if "verified_source_name" in p
    }
    alias_reg = {
        (f"{a.slug}_performance_details.csv", a.source_name, a.source_url) for a in ids.VERIFIED_SOURCE_ALIASES
    }
    assert alias_ev == alias_reg


def test_validation_structure_and_known_gaps(real: legacy.DatasetCandidate, report: reconcile.ValidationReport) -> None:
    assert real.candidate.manifest.status is DatasetStatus.LEGACY_UNVERIFIED
    for check in ("schema", "keys", "foreign_keys", "stages", "scores", "player_stats"):
        assert report.checks[check] is CheckOutcome.PASS, check
    blocking = [i for i in report.issues if i["severity"] == "blocking" and i["status"] != "accepted"]
    for i in blocking:
        if i["rule_id"] == "current_season_row_quarantined":
            assert i["row_key"].rsplit(":", 1)[0] in KNOWN_2026_GAP_FILES, i
        elif i["rule_id"] == "match_player_goals_mismatch":
            assert i["row_key"] in KNOWN_2026_GOAL_KEYS, i
        else:
            pytest.fail(f"new blocking issue: {i}")
    assert (report.outcome is CheckOutcome.PASS) == (not blocking)


def test_rerun_idempotent(real: legacy.DatasetCandidate, tmp_path: Path) -> None:
    again = legacy.import_legacy(REPO, RunContext(settings=Settings(data_root=tmp_path / "var")))
    assert again.snapshot_id == real.snapshot_id
