"""Re-linking quarantined lineup tokens after a refresh/repair adds player rows (PLAN 4.1, 4.4).

A lineup token the importer could not resolve (no player row for that match) stays in
quarantine. Once a verified source repair adds the missing player-game rows,
``reconcile.relink_quarantined_lineups`` links the token with the importer's pass-1 rule
(exact normalized name, unique among that match+club's participants), marks the
quarantine row resolved while keeping its raw evidence, and validation accepts a
resolved row only when the linked lineup row really exists.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from supercoach_via.domain.schemas import TABLES, DatasetStatus
from supercoach_via.ingest import reconcile
from supercoach_via.ingest.legacy import DatasetCandidate, _to_table
from supercoach_via.storage import snapshots

FIXED = datetime(2026, 9, 25, tzinfo=UTC)
SRC = "data/lineups/team_lineups_hawthorn.csv"


def clock() -> datetime:
    return FIXED


def _row(table: str, **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {c: None for c in TABLES[table].column_names}
    row.update(values)
    return row


def _pg(pid: str) -> dict[str, Any]:
    return _row("player_games", match_id="m1", player_id=pid, club_id="hawthorn", season=2026, stage_label="1",
                stage_id="r01", club_source_name="Hawthorn", link_method="key", date_quality="fixture_verified",
                revision_id="r", provenance="legacy_import")  # fmt: skip


def _player(pid: str, name: str) -> dict[str, Any]:
    return _row("players", player_id=pid, display_name=name, birth_date_quality="source",
                identity_status="canonical", provenance="legacy_import")  # fmt: skip


def _q(qid: str, token: str, reason: str = "lineup_token_unresolved") -> dict[str, Any]:
    return _row("quarantine", quarantine_id=qid, table_name="lineups", reason=reason, season=2026,
                raw=json.dumps({"token": token, "row": {"team_name": "Hawthorn"}}),
                candidates=json.dumps({"match": [], "club_season": []}), provenance="legacy_import",
                source_path=SRC, source_sha256="ab" * 32, source_row=7)  # fmt: skip


def _snapshot(root: Path, *, with_new_player_row: bool) -> snapshots.SnapshotCandidate:
    b = snapshots.SnapshotBuilder(root, clock=clock, code_version="t")
    pgs = [_pg("legacy:hawk_one")] + ([_pg("src:afltables:J.Jack_Dalton")] if with_new_player_row else [])
    b.add_partitioned("player_games", _to_table("player_games", pgs), "season")
    b.add("players", _to_table("players", [
        _player("legacy:hawk_one", "Hawk One"), _player("src:afltables:J.Jack_Dalton", "Jack Dalton"),
        _player("legacy:dalton_jack_15041876", "Jack Dalton")]))  # fmt: skip
    lineup = _row("lineups", match_id="m1", club_id="hawthorn", player_id="legacy:hawk_one", season=2026,
                  role="played", confidence="high", name_token="Hawk One", resolution="match_participation",
                  provenance="legacy_import", source_path=SRC, source_sha256="ab" * 32, source_row=7)  # fmt: skip
    b.add_partitioned("lineups", _to_table("lineups", [lineup]), "season")
    b.add("quarantine", _to_table("quarantine", [_q("q1", "Jack Dalton"), _q("q2", "Nobody Here")]))
    match = _row("matches", match_id="m1", season=2026, stage_label="1", stage_type="regular", round_number=1,
                 stage_order=1, stage_id="r01", replay_occurrence=0, home_club_id="hawthorn", away_club_id="carlton",
                 home_source_name="Hawthorn", away_source_name="Carlton", date_precision="date",
                 status="complete", provenance="legacy_import")  # fmt: skip
    b.add_partitioned("matches", _to_table("matches", [match]), "season")
    return b.finish(status=DatasetStatus.LEGACY_UNVERIFIED)


def _quarantine_issues(root: Path, cand: snapshots.SnapshotCandidate) -> list[tuple[str, str]]:
    report = reconcile.validate_dataset(DatasetCandidate(cand, {}, root), reconcile.load_policy(current_season=2026))
    return sorted(
        (i["rule_id"], i["explanation"]) for i in report.issues
        if i["rule_id"] in ("current_season_row_quarantined", "quarantine_resolution_unverified")
    )  # fmt: skip


def test_links_token_to_the_unique_participant_and_marks_quarantine_resolved(tmp_path: Path) -> None:
    base = _snapshot(tmp_path, with_new_player_row=True)
    ups = reconcile.relink_quarantined_lineups(tmp_path, base.manifest, season=2026)
    (lu,) = ups["lineups"]
    assert (lu["match_id"], lu["club_id"], lu["player_id"]) == ("m1", "hawthorn", "src:afltables:J.Jack_Dalton")
    assert lu["name_token"] == "Jack Dalton" and lu["resolution"] == "match_participation"
    assert (lu["source_path"], lu["source_row"], lu["provenance"]) == (SRC, 7, "legacy_import")
    (q,) = ups["quarantine"]
    assert q["quarantine_id"] == "q1" and q["reason"] == "resolved_by_repair:lineup_token_unresolved"
    assert json.loads(q["raw"])["token"] == "Jack Dalton"  # evidence preserved
    merged = snapshots.apply_upserts(tmp_path, base.manifest, ups, clock=clock, code_version="t",
                                     status=DatasetStatus.PARTIAL)  # fmt: skip
    issues = _quarantine_issues(tmp_path, merged)
    assert len(issues) == 1 and "lineup_token_unresolved" in issues[0][1]  # only "Nobody Here" remains


def test_no_link_without_a_player_row_and_no_link_to_a_namesake(tmp_path: Path) -> None:
    base = _snapshot(tmp_path, with_new_player_row=False)
    ups = reconcile.relink_quarantined_lineups(tmp_path, base.manifest, season=2026)
    assert ups == {"lineups": [], "quarantine": []}


def test_resolved_marker_without_lineup_row_blocks(tmp_path: Path) -> None:
    base = _snapshot(tmp_path, with_new_player_row=True)
    forged = snapshots.apply_upserts(
        tmp_path, base.manifest, {"quarantine": [_q("q1", "Jack Dalton", "resolved_by_repair:lineup_token_unresolved")]},
        clock=clock, code_version="t", status=DatasetStatus.PARTIAL,
    )  # fmt: skip
    rules = [r for r, _ in _quarantine_issues(tmp_path, forged)]
    assert "quarantine_resolution_unverified" in rules
