"""Source-backed corrections derived from audit findings (DESIGN section 14 "Corrections").

A correction is proposed only for a confirmed discrepancy whose correct value the frozen source states (a
double-sourced cell, a proven zero, the match page's date); every change carries the finding id, rule and source
evidence. Applying refuses a row whose current value is not the audited ``old`` value, so a correction can never
land on data the audit did not see.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from supercoach_via.reconciliation import corrections as CO


def _f(category: str, **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": f"id-{category}-{kw.get('field')}",
        "category": category,
        "severity": "fail",
        "layer": "legacy_csv",
        "rule_id": "R-X",
        "player": {"source_url": "https://afltables.com/afl/stats/players/A/Ann_Able.html"},
        "season": 2026,
        "match": {"source_url": "https://afltables.com/afl/stats/games/2026/041520260305.html"},
        "field": None,
        "expected": None,
        "actual": None,
        "evidence": {"source_url": "https://afltables.com/x.html", "body_sha256": "ab" * 32, "locator": "row 3"},
        "local": {"origin": "data/player_data/able_ann_01071990_performance_details.csv#1"},
        "detail": "",
    }
    base.update(kw)
    return base


def test_findings_become_changes_with_their_evidence_and_unsupported_classes_are_counted_not_guessed() -> None:
    findings = [
        _f("CELL_LOCAL_NULL", layer="snapshot", field="tackles", expected="0", actual=None,
           local={"origin": "abcdabcdabcdabcd#7"}, rule_id="R-TOTAL-NONBLANK"),
        _f("CELL_MISMATCH", field="kicks", expected="10", actual="11", rule_id="R-BOTH-PAGES"),
        _f("APPEARANCE_ATTR_MISMATCH", field="jersey", expected="14", actual="5", rule_id="R-ATTR-JERSEY"),
        _f("APPEARANCE_DATE_MISMATCH", field="date", rule_id="R-ATTR-DATE",
           local={"origin": "o#1", "rows": 2, "date_quality": None,
                  "changes": [["f.csv#1", "2026-03-05", "2026-02-01"], ["f.csv#2", "2026-03-12", "2026-02-08"]]}),
        _f("APPEARANCE_MISSING_LOCAL"),
        _f("CELL_UNRESOLVED", severity="unknown"),
    ]  # fmt: skip
    changes, unsupported = CO.changes_from_findings(findings)
    assert [(c.layer, c.target, c.field, c.old, c.new) for c in changes] == [  # stable order: layer, file, row, field
        ("legacy_csv", "data/player_data/able_ann_01071990_performance_details.csv#1", "jersey", "5", "14"),
        ("legacy_csv", "data/player_data/able_ann_01071990_performance_details.csv#1", "kicks", 11, 10),
        ("legacy_csv", "f.csv#1", "date", "2026-02-01", "2026-03-05"),
        ("legacy_csv", "f.csv#2", "date", "2026-02-08", "2026-03-12"),
        ("snapshot", "abcdabcdabcdabcd#7", "tackles", None, 0),
    ]
    assert all(c.finding_id and c.body_sha256 == "ab" * 32 for c in changes)
    assert unsupported == {"APPEARANCE_MISSING_LOCAL": 1}  # unknowns are never corrections; misses are counted


def test_a_mismatch_is_only_corrected_from_a_double_sourced_cell() -> None:
    single = _f("CELL_MISMATCH", field="kicks", expected="10", actual="11", rule_id="R-PROFILE-ONLY")
    changes, unsupported = CO.changes_from_findings([single])
    assert changes == [] and unsupported == {"CELL_MISMATCH:R-PROFILE-ONLY": 1}


def _legacy(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "player_data"
    d.mkdir(parents=True)
    rows = [
        ["team", "year", "games_played", "opponent", "round", "result", "jersey_num", "kicks", "tackles", "date"],
        ["Alpha", "2026", "1", "Beta", "1", "W", "5", "11.0", "", "2026-02-01"],
        ["Alpha", "2026", "2", "Beta", "2", "L", "5", "7.0", "3.0", "2026-02-08"],
    ]
    with (d / "able_ann_01071990_performance_details.csv").open("w", newline="") as fh:
        csv.writer(fh, lineterminator="\n").writerows(rows)
    return tmp_path


def test_legacy_changes_edit_only_the_named_cells_in_the_files_own_number_style(tmp_path: Path) -> None:
    root = _legacy(tmp_path)
    rel = "data/player_data/able_ann_01071990_performance_details.csv"
    path = root / rel
    before = path.read_bytes()
    changes = [
        CO.Change("legacy_csv", f"{rel}#1", "kicks", 11, 10, "R-BOTH-PAGES", "f1", "u", "s"),
        CO.Change("legacy_csv", f"{rel}#1", "jersey", "5", "14", "R-ATTR-JERSEY", "f2", "u", "s"),
        CO.Change("legacy_csv", f"{rel}#2", "date", "2026-02-08", "2026-03-12", "R-ATTR-DATE", "f3", "u", "s"),
    ]
    summary = CO.apply_legacy(root, changes)
    assert summary == {"files_changed": 1, "cells_changed": 3}
    lines = path.read_text().splitlines()
    assert lines[1] == "Alpha,2026,1,Beta,1,W,14,10.0,,2026-02-01"
    assert lines[2] == "Alpha,2026,2,Beta,2,L,5,7.0,3.0,2026-03-12"
    assert len(path.read_bytes()) == len(before) + 1  # "5" -> "14"; nothing else moved


def test_a_legacy_change_whose_old_value_is_not_what_the_file_holds_is_refused_and_nothing_is_written(
    tmp_path: Path,
) -> None:
    root = _legacy(tmp_path)
    rel = "data/player_data/able_ann_01071990_performance_details.csv"
    before = (root / rel).read_bytes()
    stale = CO.Change("legacy_csv", f"{rel}#1", "kicks", 99, 10, "R-BOTH-PAGES", "f1", "u", "s")
    with pytest.raises(CO.CorrectionConflict, match="kicks"):
        CO.apply_legacy(root, [stale])
    assert (root / rel).read_bytes() == before


def test_a_legacy_change_outside_the_player_data_directory_is_refused(tmp_path: Path) -> None:
    root = _legacy(tmp_path)
    bad = CO.Change("legacy_csv", "../evil.csv#1", "kicks", 1, 2, "R", "f", "u", "s")
    with pytest.raises(CO.CorrectionConflict, match="outside"):
        CO.apply_legacy(root, [bad])


def test_changes_round_trip_through_jsonl_in_a_stable_order(tmp_path: Path) -> None:
    changes = [
        CO.Change("snapshot", "b#1", "kicks", 1, 2, "R", "f2", "u", "s"),
        CO.Change("legacy_csv", "a.csv#3", "date", "2026-01-01", "2026-01-02", "R", "f1", "u", "s"),
    ]
    out = tmp_path / "c.jsonl"
    digest = CO.write_changes(out, changes)
    again = CO.read_changes(out)
    assert again == sorted(changes, key=CO.change_key) and len(digest) == 64
    assert [json.loads(line)["layer"] for line in out.read_text().splitlines()] == ["legacy_csv", "snapshot"]


def consistent_world() -> Any:
    """One home-and-away match whose cells satisfy the production validator: disposals = kicks + handballs,
    player goals and behinds sum to the printed score, six Brownlow votes."""
    from datetime import date

    from tests.scvia.unit import recon_inputs as RI
    from tests.scvia.unit import recon_world as rw

    def cells(kicks: int, handballs: int, goals: int, behinds: int, tackles: int, br: int | None) -> dict[str, Any]:
        c = RI.full_cells(1, kicks=kicks, handballs=handballs, disposals=kicks + handballs, goals=goals,
                          behinds=behinds, tackles=tackles, brownlow_votes=br)  # fmt: skip
        return c

    players = RI.modern_world().players
    apps = (
        rw.A("a", "Alpha", "1", cells(8, 4, 2, 1, 3, 3)),
        rw.A("b", "Alpha", "2", cells(5, 6, 1, 1, 0, None)),
        rw.A("c", "Beta", "3", cells(9, 2, 2, 1, 4, 3)),
    )
    m = rw.M("041520260305", 2026, "1", "Alpha", "Beta", date(2026, 3, 5),
             hq=((1, 0), (2, 1), (3, 1), (3, 2)), aq=((0, 0), (1, 0), (2, 1), (2, 1)), apps=apps)  # fmt: skip
    return rw.World(players, (m,), (2026,))


def test_snapshot_changes_promote_a_validated_child_snapshot_that_a_re_audit_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from supercoach_via import pipeline
    from supercoach_via.ingest.http import RawArchive
    from supercoach_via.reconciliation import inventory as inv
    from supercoach_via.settings import RunContext, load_settings
    from supercoach_via.storage import snapshots
    from tests.scvia.unit import recon_e2e as E
    from tests.scvia.unit import recon_inputs as RI

    world = consistent_world()

    def rows(layer: str, season: int) -> list[Any]:
        base = RI.local_rows(world, season, layer)
        if layer != "snapshot":
            return base
        base = RI.with_cell(base, "b", "041520260305", "tackles", None)  # the source proves 0 (team total printed)
        return RI.with_cell(base, "a", "041520260305", "kicks", 99)  # both source pages say otherwise

    e2e = E.build(tmp_path, world, monkeypatch=monkeypatch, local_rows=rows)
    result, code, out = e2e.compare()
    assert code == 4 and result.report["result"]["layers"]["snapshot"] == "FAIL"
    findings = [json.loads(line) for line in (out / "findings.jsonl").read_text().splitlines()]
    changes, _ = CO.changes_from_findings(findings)
    snap = [c for c in changes if c.layer == "snapshot"]
    assert {(c.field, c.old, c.new) for c in snap} == {("tackles", None, 0), ("kicks", 99, 8)}
    changes_file = tmp_path / "changes.jsonl"
    CO.write_changes(changes_file, snap)
    before = snapshots.load_snapshot(e2e.data_root).snapshot_id
    ctx = RunContext(settings=load_settings(None, overrides={"data_root": e2e.data_root}))
    res = pipeline.apply_reconciliation_corrections(ctx, changes_file=changes_file, expected_snapshot=before)
    assert res.exit_code == 0 and res.promoted and res.snapshot_id != before
    assert res.outputs["corrections"]["player_games_rows"] == 2
    # a second application is refused: the audited old values are gone
    again = pipeline.apply_reconciliation_corrections(ctx, changes_file=changes_file, expected_snapshot=before)
    assert again.exit_code != 0 and not again.promoted

    run2 = tmp_path / "run2"
    plan = inv.build_plan(data_root=e2e.data_root, snapshot="current", legacy_root=e2e.legacy_root,
                          through_date="2026-09-30", scope="all", run_dir=run2)  # fmt: skip
    plan_path = inv.write_plan(plan)
    shutil.copytree(e2e.root / "run" / "evidence", run2 / "evidence")
    assert RawArchive(run2 / "evidence") is not None
    from supercoach_via.reconciliation import compare as CP

    opts = CP.CompareOptions(plan=plan_path, capture_manifest=e2e.manifest_path, out=run2 / "r", cache=run2 / "c")
    after, _audit = CP.run_audit(opts)
    CP.cleanup(_audit)
    assert after.report["result"]["layers"]["snapshot"] == "PASS"


def test_a_zero_the_local_layer_invented_where_the_notes_say_not_recorded_becomes_null_per_field() -> None:
    f = _f("LOCAL_UNSUPPORTED_NUMERIC", severity="unknown", layer="snapshot", field="frees_for,frees_against",
           rule_id="R-NOTES-EXCEPTION", local={"origin": "abcdabcdabcdabcd#3"})  # fmt: skip
    changes, unsupported = CO.changes_from_findings([f])
    assert [(c.field, c.old, c.new) for c in changes] == [("frees_against", 0, None), ("frees_for", 0, None)]
    assert not unsupported


def test_the_snapshot_career_counter_token_is_corrected_with_the_counter() -> None:
    f = _f("APPEARANCE_ATTR_MISMATCH", layer="snapshot", field="counter_token", expected="162", actual=None,
           local={"origin": "abcdabcdabcdabcd#3"})  # fmt: skip
    changes, _ = CO.changes_from_findings([f])
    assert [(c.field, c.old, c.new) for c in changes] == [("counter_token", None, "162")]
    legacy = _f("APPEARANCE_ATTR_MISMATCH", field="counter_token", expected="162", actual="161")
    assert CO.changes_from_findings([legacy])[1] == {"APPEARANCE_ATTR_MISMATCH": 1}  # no legacy token column


def test_identity_proposals_become_player_changes_with_the_audited_old_values() -> None:
    from supercoach_via.reconciliation.idfix import Proposal

    url = "https://afltables.com/afl/stats/players/P/Paul_Vander_Haar.html"
    props = [
        Proposal("repair", "legacy:haar_paul_07031958", url, {"display_name": "Paul Vander Haar", "last_name": "Vander Haar",
                 "birth_date": "1958-03-08"}),
        Proposal("duplicate", "legacy:ross_jonathan_03111973", "u2", {"canonical": "legacy:ross_jonathon_03111973"}),
        Proposal("bind", "legacy:ross_jonathon_03111973", "u2", {}),
    ]  # fmt: skip
    current = {
        "legacy:haar_paul_07031958": {"display_name": "Paul Haar", "first_name": "Paul", "last_name": "Haar",
                                      "birth_date": "1958-03-07", "source_urls": None},
        "legacy:ross_jonathan_03111973": {"identity_status": "canonical", "canonical_player_id": None},
        "legacy:ross_jonathon_03111973": {"source_urls": None},
    }  # fmt: skip
    snap = CO.identity_changes(props, layer="snapshot", current=current)
    got = {(c.target, c.field, c.old, c.new) for c in snap}
    assert ("player:legacy:haar_paul_07031958", "last_name", "Haar", "Vander Haar") in got
    assert ("player:legacy:haar_paul_07031958", "birth_date", "1958-03-07", "1958-03-08") in got
    assert ("player:legacy:haar_paul_07031958", "source_urls", None, json.dumps([url])) in got
    assert ("player:legacy:ross_jonathan_03111973", "identity_status", "canonical", "quarantined_duplicate") in got
    assert ("player:legacy:ross_jonathan_03111973", "delete_games", None, "all") in got
    legacy_current = {
        "haar_paul_07031958": {"first_name": "Paul", "last_name": "Haar", "born_date": "07-03-1958"},
        "ross_jonathan_03111973": {},
    }
    leg = CO.identity_changes(props, layer="legacy_csv", current=legacy_current)
    got = {(c.target, c.field, c.old, c.new) for c in leg}
    assert ("data/player_data/haar_paul_07031958_personal_details.csv#1", "last_name", "Haar", "Vander Haar") in got
    assert (
        "data/player_data/haar_paul_07031958_personal_details.csv#1",
        "born_date",
        "07-03-1958",
        "08-03-1958",
    ) in got
    # the removal names its canonical record, so applying it can repoint the duplicate's lineup tokens
    assert ("data/player_data/ross_jonathan_03111973", "delete_files", "present", "ross_jonathon_03111973") in got
    assert not [c for c in leg if c.field == "source_urls"]  # the legacy layer has no source URL column


def test_legacy_identity_edits_and_duplicate_removal(tmp_path: Path) -> None:
    root = _legacy(tmp_path)
    d = root / "data" / "player_data"
    (d / "able_ann_01071990_personal_details.csv").write_text(
        "first_name,last_name,born_date,debut_date,height,weight\nAnn,Able,01-07-1990,01-04-2010,180,80\n"
    )
    (d / "dup_x_01011990_performance_details.csv").write_text("team\n")
    (d / "dup_x_01011990_personal_details.csv").write_text("first_name\n")
    changes = [
        CO.Change("legacy_csv", "data/player_data/able_ann_01071990_personal_details.csv#1", "last_name", "Able",
                  "De Able", "R-ID-REPAIR", "", "u", None),
        CO.Change("legacy_csv", "data/player_data/dup_x_01011990", "delete_files", "present", None, "R-ID-DUPLICATE",
                  "", "u", None),
    ]  # fmt: skip
    CO.apply_legacy(root, changes)
    assert (d / "able_ann_01071990_personal_details.csv").read_text().splitlines()[
        1
    ] == "Ann,De Able,01-07-1990,01-04-2010,180,80"
    assert not (d / "dup_x_01011990_performance_details.csv").exists()
    assert not (d / "dup_x_01011990_personal_details.csv").exists()


def test_player_changes_and_a_duplicates_game_deletion_promote_a_validated_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supercoach_via import pipeline
    from supercoach_via.settings import RunContext, load_settings
    from supercoach_via.storage import snapshots
    from supercoach_via.storage.queries import SnapshotQuery
    from tests.scvia.unit import recon_e2e as E

    e2e = E.build(tmp_path, consistent_world(), monkeypatch=monkeypatch)
    root = e2e.data_root
    base = snapshots.load_snapshot(root)
    with SnapshotQuery(root, base, tables={"players", "player_games"}) as q:
        pids = sorted(r[0] for r in q.rows("SELECT player_id FROM players"))
        before_games = q.rows("SELECT count(*) FROM player_games")[0][0]
    a, b = pids[0], pids[1]
    changes = [
        CO.Change("snapshot", f"player:{a}", "last_name", "Able", "De Able", "R-ID-REPAIR", "", "u", None),
        CO.Change("snapshot", f"player:{a}", "source_urls", "[]", '["u"]', "R-ID-REPAIR", "", "u", None),
    ]  # fmt: skip
    # deleting a duplicate's games is checked at the upsert level: in this world b is a real player, so removing
    # his game would (rightly) fail validation of the team goal totals
    ups = CO.snapshot_upserts(
        root, base, [CO.Change("snapshot", f"player:{b}", "delete_games", None, "all", "R-ID-DUPLICATE", "", "u", None)]
    )
    assert [r["player_id"] for r in ups["_delete_player_games"]] == [b] and ups["players"] == []
    f = tmp_path / "id.jsonl"
    CO.write_changes(f, changes)
    ctx = RunContext(settings=load_settings(None, overrides={"data_root": root}))
    res = pipeline.apply_reconciliation_corrections(ctx, changes_file=f, expected_snapshot=base.snapshot_id)
    assert res.exit_code == 0 and res.promoted, res
    after = snapshots.load_snapshot(root)
    with SnapshotQuery(root, after, tables={"players", "player_games"}) as q:
        rows = {r[0]: r[1:] for r in q.rows(
            "SELECT player_id, last_name, source_urls, identity_status, canonical_player_id FROM players")}  # fmt: skip
        games = q.rows("SELECT count(*) FROM player_games")[0][0]
        b_games = q.rows("SELECT count(*) FROM player_games WHERE player_id = ?", [b])[0][0]
    assert rows[a][0] == "De Able" and rows[a][1] == '["u"]'
    assert rows[b][2] == "canonical" and b_games == 1 and games == before_games  # nothing else moved


def test_cli_propose_then_apply_both_layers_then_a_re_audit_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from typer.testing import CliRunner

    from supercoach_via.reconciliation import inventory as inv
    from supercoach_via.reconciliation.cli import reconcile_app
    from supercoach_via.storage import snapshots
    from tests.scvia.unit import recon_e2e as E
    from tests.scvia.unit import recon_inputs as RI

    world = consistent_world()

    def rows(layer: str, season: int) -> list[Any]:
        base = RI.local_rows(world, season, layer)
        base = RI.with_cell(base, "a", "041520260305", "kicks", 99)  # both layers hold a wrong value
        return RI.with_cell(base, "b", "041520260305", "tackles", None) if layer == "snapshot" else base

    e2e = E.build(tmp_path, world, monkeypatch=monkeypatch, local_rows=rows)
    _, code, report = e2e.compare("reports/before")
    assert code == 4
    run = CliRunner()
    out = tmp_path / "proposal"
    r = run.invoke(reconcile_app, ["propose-corrections", "--plan", str(e2e.plan_path), "--capture-manifest",
                                   str(e2e.manifest_path), "--report", str(report), "--out", str(out),
                                   "--cache", str(tmp_path / "run" / "cache")])  # fmt: skip
    assert r.exit_code == 0, r.output
    summary = json.loads((out / "summary.json").read_text())
    assert (
        summary["by_layer_rule"]["snapshot:R-BOTH-PAGES"] == 1
        and summary["by_layer_rule"]["legacy_csv:R-BOTH-PAGES"] == 1
    )
    snap_id = snapshots.load_snapshot(e2e.data_root).snapshot_id
    r = run.invoke(reconcile_app, ["apply-corrections", "--changes", str(out / "changes.jsonl"), "--layer", "snapshot",
                                   "--data-root", str(e2e.data_root), "--expected-snapshot", snap_id])  # fmt: skip
    assert r.exit_code == 0, r.output
    assert e2e.legacy_root is not None
    r = run.invoke(reconcile_app, ["apply-corrections", "--changes", str(out / "changes.jsonl"), "--layer",
                                   "legacy_csv", "--legacy-root", str(e2e.legacy_root)])  # fmt: skip
    assert r.exit_code == 0, r.output
    run2 = tmp_path / "run2"
    plan = inv.build_plan(data_root=e2e.data_root, snapshot="current", legacy_root=e2e.legacy_root,
                          through_date="2026-09-30", scope="all", run_dir=run2)  # fmt: skip
    plan_path = inv.write_plan(plan)
    shutil.copytree(e2e.root / "run" / "evidence", run2 / "evidence")
    from supercoach_via.reconciliation import compare as CP

    after, audit = CP.run_audit(CP.CompareOptions(plan=plan_path, capture_manifest=e2e.manifest_path,
                                                  out=run2 / "r", cache=run2 / "c"))  # fmt: skip
    CP.cleanup(audit)
    assert after.report["result"] == {"layers": {"legacy_csv": "PASS", "snapshot": "PASS"}, "overall": "PASS"}


def test_a_quarantined_row_is_relinked_to_the_match_the_source_proves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supercoach_via.storage import snapshots
    from supercoach_via.storage.queries import SnapshotQuery
    from tests.scvia.unit import recon_e2e as E

    e2e = E.build(tmp_path, consistent_world(), monkeypatch=monkeypatch)
    root = e2e.data_root
    base = snapshots.load_snapshot(root)
    with SnapshotQuery(root, base, tables={"player_games"}) as q:
        row = q.arrow("SELECT * FROM player_games ORDER BY player_id LIMIT 1").to_pylist()[0]
    raw = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()}
    q_row = {"quarantine_id": "q:test", "table_name": "player_games", "reason": "replayed_draw_link_unresolved",
             "candidates": json.dumps([row["match_id"], "m:other"]), "raw": json.dumps(raw, sort_keys=True),
             "season": row["season"], "provenance": row["provenance"], "source_path": row["source_path"],
             "source_sha256": row["source_sha256"], "source_row": row["source_row"]}  # fmt: skip
    quarantined = snapshots.apply_upserts(root, base, {"quarantine": [q_row]}, clock=lambda: base.created_at,
                                          code_version="t", status=base.status, deletes={"player_games": [row]})  # fmt: skip
    ups = CO.snapshot_upserts(root, quarantined.manifest, [CO.Change("snapshot", "quarantine:q:test", "relink",
                              None, json.dumps({"match_id": row["match_id"]}), "R-QUARANTINE", "f", "u", "s")])  # fmt: skip
    assert [r["match_id"] for r in ups["player_games"]] == [row["match_id"]]
    assert ups["player_games"][0]["link_method"] == "source_url"
    assert [r["quarantine_id"] for r in ups["_delete_quarantine"]] == ["q:test"]
    with pytest.raises(CO.CorrectionConflict, match="candidates"):
        CO.snapshot_upserts(root, quarantined.manifest, [CO.Change("snapshot", "quarantine:q:test", "relink", None,
                            json.dumps({"match_id": "m:nowhere"}), "R", "f", "u", "s")])  # fmt: skip
    # a relink onto a key that already holds an accepted row would silently overwrite it: refused
    both = snapshots.apply_upserts(root, base, {"quarantine": [q_row]}, clock=lambda: base.created_at,
                                   code_version="t", status=base.status)  # fmt: skip
    with pytest.raises(CO.CorrectionConflict, match="already"):
        CO.snapshot_upserts(root, both.manifest, [CO.Change("snapshot", "quarantine:q:test", "relink", None,
                            json.dumps({"match_id": row["match_id"]}), "R", "f", "u", "s")])  # fmt: skip
    with pytest.raises(CO.CorrectionConflict, match="no quarantine"):
        CO.snapshot_upserts(root, quarantined.manifest, [CO.Change("snapshot", "quarantine:q:missing", "relink", None,
                            json.dumps({"match_id": row["match_id"]}), "R", "f", "u", "s")])  # fmt: skip


def test_missing_legacy_rows_a_missing_legacy_player_and_a_missing_match_row_are_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from supercoach_via.reconciliation import compare as CP
    from supercoach_via.reconciliation import inventory as inv
    from supercoach_via.reconciliation.propose import propose
    from tests.scvia.unit import recon_e2e as E
    from tests.scvia.unit import recon_inputs as RI

    world = RI.modern_world()

    def rows(layer: str, season: int) -> list[Any]:
        base = RI.local_rows(world, season, layer)
        return [r for r in base if not (layer == "legacy_csv" and r.origin.endswith(":a:041520260312"))]

    e2e = E.build(tmp_path, world, monkeypatch=monkeypatch, local_rows=rows)
    assert e2e.legacy_root is not None
    pdir = e2e.legacy_root / "data" / "player_data"
    for p in pdir.glob("cole_cy_*"):  # the legacy layer has no file for c at all
        p.unlink()
    mfile = e2e.legacy_root / "data" / "matches" / "matches_2026.csv"
    lines = mfile.read_text().splitlines(True)
    mfile.write_text("".join(lines[:-1]))  # nor the Grand Final row
    plan = inv.build_plan(data_root=e2e.data_root, snapshot="current", legacy_root=e2e.legacy_root,
                          through_date="2026-09-30", scope="all", run_dir=tmp_path / "run")  # fmt: skip
    plan_path = inv.write_plan(plan)
    opts = CP.CompareOptions(plan=plan_path, capture_manifest=e2e.manifest_path, out=tmp_path / "run" / "r1",
                             cache=tmp_path / "run" / "c")  # fmt: skip
    res, audit = CP.run_audit(opts)
    from supercoach_via.reconciliation import report as RP

    RP.write_report_dir(opts.out, res, CP.input_roots_of(audit.plan, audit.capture_dir))
    CP.cleanup(audit)
    assert res.report["result"]["layers"]["legacy_csv"] == "FAIL"
    summary = propose(CP.CompareOptions(plan=plan_path, capture_manifest=e2e.manifest_path,
                                        out=tmp_path / "prop" / "_a", cache=tmp_path / "run" / "c"),
                      opts.out, tmp_path / "prop")  # fmt: skip
    assert summary["unsupported_fail_findings"] == {}, summary
    changes = CO.read_changes(tmp_path / "prop" / "changes.jsonl")
    CO.apply_legacy(e2e.legacy_root, [c for c in changes if c.layer == "legacy_csv"])
    assert len(list(pdir.glob("cole_cy_*"))) == 2
    run2 = tmp_path / "run2"
    plan2 = inv.build_plan(data_root=e2e.data_root, snapshot="current", legacy_root=e2e.legacy_root,
                           through_date="2026-09-30", scope="all", run_dir=run2)  # fmt: skip
    p2 = inv.write_plan(plan2)
    shutil.copytree(tmp_path / "run" / "evidence", run2 / "evidence")
    after, a2 = CP.run_audit(CP.CompareOptions(plan=p2, capture_manifest=e2e.manifest_path, out=run2 / "r",
                                               cache=run2 / "c"))  # fmt: skip
    CP.cleanup(a2)
    assert after.report["result"]["layers"]["legacy_csv"] == "PASS", [
        (f["category"], f["detail"][:80]) for f in after.findings if f["severity"] != "info"
    ][:6]


def test_a_missing_season_summary_value_becomes_an_award_row_in_each_layer(tmp_path: Path) -> None:
    f = _f("LOCAL_MISSING_SUMMARY_VALUE", layer="snapshot", rule_id="R-AGG-STINT", field="brownlow_votes",
           season=1935, expected="13", actual="0", player={"source_url": "u-able", "local_id": "legacy:able"},
           local={"origins": [], "club": "Alpha"})  # fmt: skip
    leg = {**f, "id": "f2", "layer": "legacy_csv", "player": {"source_url": "u-able", "local_id": "able_ann_01071990"}}
    career = {**f, "id": "f3", "rule_id": "R-AGG-CAREER", "season": None, "local": {"origins": [], "club": None}}
    changes, unsupported = CO.changes_from_findings([f, leg, career])
    assert {(c.layer, c.target, c.field, c.new) for c in changes} == {
        ("snapshot", "award:legacy:able|1935|Alpha", "brownlow_votes", 13),
        (
            "legacy_csv",
            "data/awards/brownlow_season_votes.csv#award:able_ann_01071990|1935|Alpha",
            "brownlow_votes",
            13,
        ),
    }
    assert unsupported == {}  # the career finding is derived from the stint ones
    root = _legacy(tmp_path)
    CO.apply_legacy(root, [c for c in changes if c.layer == "legacy_csv"])
    lines = (root / "data" / "awards" / "brownlow_season_votes.csv").read_text().splitlines()
    assert lines[0] == "slug,year,team,award,value,source_url,source_sha256"
    assert lines[1].startswith("able_ann_01071990,1935,Alpha,brownlow_votes,13,")
    with pytest.raises(CO.CorrectionConflict, match="already"):
        CO.apply_legacy(root, [c for c in changes if c.layer == "legacy_csv"])


def test_a_snapshot_award_row_takes_the_club_id_from_the_players_own_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supercoach_via.storage import snapshots
    from tests.scvia.unit import recon_e2e as E

    e2e = E.build(tmp_path, consistent_world(), monkeypatch=monkeypatch)
    base = snapshots.load_snapshot(e2e.data_root)
    ch = CO.Change("snapshot", "award:legacy:a|2026|Alpha", "brownlow_votes", None, 13, "R-AGG-STINT", "f",
                   "https://afltables.com/p", "ab" * 32)  # fmt: skip
    ups = CO.snapshot_upserts(e2e.data_root, base, [ch])
    (row,) = ups["player_season_awards"]
    assert (row["player_id"], row["season"], row["club_id"], row["club_source_name"], row["value"]) == (
        "legacy:a",
        2026,
        "alpha",
        "Alpha",
        13,
    )
    assert row["source_path"] == "https://afltables.com/p" and row["provenance"] == "source_fetch"
    with pytest.raises(CO.CorrectionConflict, match="no rows"):
        CO.snapshot_upserts(e2e.data_root, base, [CO.Change("snapshot", "award:legacy:a|1999|Alpha",
                            "brownlow_votes", None, 1, "R", "f", "u", "s")])  # fmt: skip


def test_a_local_number_the_source_revised_to_a_proven_zero_is_corrected_to_zero() -> None:
    """Live gate, 2026: AFL Tables revised e.g. a one-percenter 1 to blank (team total non-blank, so a proven zero).
    The evidence is the same as for a null proven zero; any other single-sourced mismatch stays unsupported."""
    proven = _f("CELL_MISMATCH", field="one_percenters", expected=0, actual=1, rule_id="R-TOTAL-NONBLANK")
    changes, unsupported = CO.changes_from_findings([proven])
    assert [(c.field, c.old, c.new, c.rule_id) for c in changes] == [("one_percenters", 1, 0, "R-TOTAL-NONBLANK")]
    assert not unsupported
    # a proven-zero rule never rewrites a local number to a NON-zero value, and other rules stay unsupported
    nonzero = _f("CELL_MISMATCH", field="kicks", expected=3, actual=4, rule_id="R-TOTAL-NONBLANK")
    single = _f("CELL_MISMATCH", field="kicks", expected=0, actual=4, rule_id="R-PROFILE-ONLY")
    changes, unsupported = CO.changes_from_findings([nonzero, single])
    assert changes == [] and sum(unsupported.values()) == 2


def test_removing_a_duplicate_repoints_its_lineup_tokens_to_the_canonical_name(tmp_path: Path) -> None:
    """2026-10-06: three duplicate removals (William Green, Jonathan Ross, Henry Paternoster) left 23 lineup tokens
    naming a player that no longer existed; the import quarantined them, and the 2026 one blocked the numeric
    pipeline. The removal now rewrites the duplicate's exact name to the canonical's name, only in lineup rows of
    the duplicate's own club-seasons, and leaves every other byte alone."""
    root = _legacy(tmp_path)
    d = root / "data" / "player_data"
    pers = "first_name,last_name,born_date,debut_date,height,weight\n"
    perf = "team,year,games_played,opponent,round,result,jersey_num,kicks,date\n"
    (d / "ross_jonathan_03111973_personal_details.csv").write_text(pers + "Jonathan,Ross,03-11-1973,,,\n")
    (d / "ross_jonathan_03111973_performance_details.csv").write_text(
        perf + "Adelaide,1992,1,Geelong,5,W,9,3,1992-04-19\n"
    )
    (d / "ross_jonathon_03111973_personal_details.csv").write_text(pers + "Jonathon,Ross,03-11-1973,,,\n")
    (d / "ross_jonathon_03111973_performance_details.csv").write_text(
        perf + "Adelaide,1992,1,Geelong,5,W,9,3,1992-04-19\n"
    )
    lu = root / "data" / "lineups"
    lu.mkdir(parents=True)
    body = (
        "year,date,round_num,team_name,players\n"
        "1992,1992-04-19 13:40,5,Adelaide,Ann Able;Jonathan Ross;Bob Baker\n"
        "1993,1993-04-19 13:40,5,Adelaide,Jonathan Ross;Bob Baker\n"  # not a season the duplicate played: untouched
    )
    (lu / "team_lineups_adelaide.csv").write_text(body)
    (lu / "team_lineups_geelong.csv").write_text(
        "year,date,round_num,team_name,players\n1992,x,5,Geelong,Jonathan Ross\n"
    )
    change = CO.Change("legacy_csv", "data/player_data/ross_jonathan_03111973", "delete_files", "present",
                       "ross_jonathon_03111973", "R-ID-DUPLICATE", "", "u", None)  # fmt: skip
    CO.apply_legacy(root, [change])
    assert not (d / "ross_jonathan_03111973_performance_details.csv").exists()
    assert (lu / "team_lineups_adelaide.csv").read_text() == body.replace(
        "Ann Able;Jonathan Ross;Bob Baker", "Ann Able;Jonathon Ross;Bob Baker"
    )
    assert "Jonathan Ross" in (lu / "team_lineups_geelong.csv").read_text()  # another club's row: untouched
    # an old change file (no canonical named) still only removes the files
    (d / "ross_jonathan_03111973_personal_details.csv").write_text(pers + "Jonathan,Ross,03-11-1973,,,\n")
    (d / "ross_jonathan_03111973_performance_details.csv").write_text(
        perf + "Adelaide,1993,1,Geelong,5,W,9,3,1993-04-19\n"
    )
    before = (lu / "team_lineups_adelaide.csv").read_text()
    CO.apply_legacy(root, [CO.Change("legacy_csv", "data/player_data/ross_jonathan_03111973", "delete_files",
                                     "present", None, "R-ID-DUPLICATE", "", "u", None)])  # fmt: skip
    assert (lu / "team_lineups_adelaide.csv").read_text() == before
