"""Legacy import: dtypes, nulls, identities, stages, linking, quarantine, accounting (D01-D12)."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from supercoach_via.domain.schemas import TABLES, DatasetStatus, is_safe_id
from supercoach_via.ingest import legacy
from supercoach_via.settings import RunContext, Settings
from supercoach_via.storage.queries import SnapshotQuery

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "legacy" / "corpus"
FIXED = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)


def _ctx(data_root: Path) -> RunContext:
    return RunContext(settings=Settings(data_root=data_root), clock=lambda: FIXED)


def _copy_corpus(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    shutil.copytree(FIXTURE, src)
    return src


@pytest.fixture(scope="module")
def imported(tmp_path_factory: pytest.TempPathFactory) -> legacy.DatasetCandidate:
    root = tmp_path_factory.mktemp("imp")
    src = root / "src"
    shutil.copytree(FIXTURE, src)
    return legacy.import_legacy(src, _ctx(root / "var"))


_OPEN: dict[tuple[str, str], SnapshotQuery] = {}


def _conn(cand: legacy.DatasetCandidate) -> SnapshotQuery:
    """One read-only query connection per imported candidate (view registration is the slow part)."""
    key = (str(cand.data_root), cand.snapshot_id)
    if key not in _OPEN:
        _OPEN[key] = SnapshotQuery(cand.data_root, cand.candidate.manifest).__enter__()
    return _OPEN[key]


def teardown_module() -> None:
    for q in _OPEN.values():
        q.__exit__(None, None, None)
    _OPEN.clear()


def _q(cand: legacy.DatasetCandidate, sql: str) -> list[tuple[Any, ...]]:
    return _conn(cand).rows(sql)


def _rows(cand: legacy.DatasetCandidate, sql: str) -> list[dict[str, Any]]:
    df = _conn(cand).df(sql)
    import pandas as pd

    return [{k: (None if pd.isna(v) else v) for k, v in rec.items()} for rec in df.to_dict(orient="records")]


class TestContract:
    def test_status_and_schema(self, imported: legacy.DatasetCandidate) -> None:
        m = imported.candidate.manifest
        assert m.status is DatasetStatus.LEGACY_UNVERIFIED
        import pyarrow.parquet as pq

        for name, entry in m.tables.items():
            assert name in TABLES, name
            for frag in entry.fragments:
                schema = pq.read_schema(imported.data_root / "fragments" / frag.path)
                expected = TABLES[name].arrow_schema()
                assert schema.names == expected.names, name
                assert [f.type for f in schema] == [f.type for f in expected], name

    def test_partitioned_by_season(self, imported: legacy.DatasetCandidate) -> None:
        m = imported.candidate.manifest
        parts = {f.partition for f in m.tables["player_games"].fragments}
        assert parts == {"1977", "2025", "2026"}
        assert {f.partition for f in m.tables["matches"].fragments} == {"1977", "2025", "2026"}

    def test_all_ids_safe(self, imported: legacy.DatasetCandidate) -> None:
        for table, col in [
            ("players", "player_id"),
            ("matches", "match_id"),
            ("clubs", "club_id"),
            ("venues", "venue_id"),
            ("quality_issues", "issue_id"),
            ("quarantine", "quarantine_id"),
        ]:
            for (v,) in _q(imported, f"SELECT {col} FROM {table}"):
                assert is_safe_id(v), (table, v)


class TestDtypesAndNulls:
    def test_d01_normalization(self, imported: legacy.DatasetCandidate) -> None:
        rows = _rows(
            imported,
            "SELECT season, kicks, handballs, goals, career_game_counter, career_game_counter_token "
            "FROM player_games WHERE player_id='legacy:mover_max_01011995' ORDER BY season",
        )
        assert rows[0]["kicks"] == 5  # '5.0' -> 5
        assert rows[0]["career_game_counter"] == 14 and rows[0]["career_game_counter_token"] == "14↓"
        assert rows[1]["career_game_counter"] == 15 and rows[1]["career_game_counter_token"] == "15↑"

    def test_d02_missing_is_not_zero(self, imported: legacy.DatasetCandidate) -> None:
        (row,) = _rows(
            imported,
            "SELECT handballs, goals, tackles FROM player_games "
            "WHERE player_id='legacy:mover_max_01011995' AND season=2025",
        )
        assert row["handballs"] == 0
        assert row["goals"] is None and row["tackles"] is None

    def test_d03_historic_rows_keep_unrecorded_stats_null(self, imported: legacy.DatasetCandidate) -> None:
        rows = _rows(imported, "SELECT tackles, clearances, goals FROM player_games WHERE season=1977")
        assert rows and all(r["tackles"] is None and r["clearances"] is None for r in rows)

    def test_d04_malformed_rows_quarantined_with_evidence(self, imported: legacy.DatasetCandidate) -> None:
        q = _rows(
            imported,
            "SELECT reason, raw, source_row FROM quarantine WHERE source_path LIKE '%hawk_harry%' ORDER BY source_row",
        )
        reasons = {r["reason"] for r in q}
        assert {"malformed_value", "malformed_column_count", "duplicate_player_game"} <= reasons
        bad = next(r for r in q if r["reason"] == "malformed_value")
        assert json.loads(bad["raw"])["kicks"] == "x"
        short = next(r for r in q if r["reason"] == "malformed_column_count")
        assert json.loads(short["raw"]) == {"line": "Hawthorn,2025,5,Sydney"}


class TestIdentities:
    def test_d05_same_name_players_distinct(self, imported: legacy.DatasetCandidate) -> None:
        rows = _rows(imported, "SELECT player_id, display_name FROM players WHERE display_name='Tom Lynch'")
        assert {r["player_id"] for r in rows} == {"legacy:lynch_tom_31101990", "legacy:lynch_tom_09101992"}
        lineup = dict(_q(imported, "SELECT club_id, player_id FROM lineups WHERE name_token='Tom Lynch'"))
        assert lineup == {"sydney": "legacy:lynch_tom_31101990", "carlton": "legacy:lynch_tom_09101992"}

    def test_d06_multiword_alias(self, imported: legacy.DatasetCandidate) -> None:
        (p,) = _rows(imported, "SELECT * FROM players WHERE player_id='legacy:wyk_alex_01072004'")
        assert p["display_name"] == "Alex Van Wyk" and p["last_name"] == "Wyk"
        aliases = _rows(imported, "SELECT * FROM player_aliases WHERE player_id='legacy:wyk_alex_01072004'")
        assert any(a["alias"] == "Alex Van Wyk" and a["source_url"].endswith("Alex_Van_Wyk.html") for a in aliases)
        (res,) = _q(imported, "SELECT resolution FROM lineups WHERE name_token='Alex Van Wyk'")
        assert res[0] == "match_participation"

    def test_d07_transferred_player(self, imported: legacy.DatasetCandidate) -> None:
        clubs = _q(
            imported,
            "SELECT season, club_id FROM player_games WHERE player_id='legacy:mover_max_01011995' ORDER BY season",
        )
        assert clubs == [(2025, "fremantle"), (2026, "demo_harbour")]

    def test_d08_entities_and_unknown_club(self, imported: legacy.DatasetCandidate) -> None:
        clubs = dict(_q(imported, "SELECT club_id, lineage_id FROM clubs"))
        assert clubs["western_bulldogs"] == clubs["footscray"] == "bulldogs"
        assert clubs["fitzroy"] != clubs["brisbane_lions"]
        assert clubs["demo_harbour"] == "demo_harbour"
        issues = _q(imported, "SELECT severity FROM quality_issues WHERE rule_id='club_auto_created'")
        assert issues == [("warning",)]

    def test_duplicate_identities_quarantined(self, imported: legacy.DatasetCandidate) -> None:
        dups = dict(
            _q(
                imported,
                "SELECT player_id, canonical_player_id FROM players WHERE identity_status='quarantined_duplicate'",
            )
        )
        assert dups == {
            "legacy:green_william_08092005": "legacy:green_will_08092005",
            "legacy:steele_roan_19092002": "legacy:steele_roan_22102001",
        }
        pg = _q(
            imported,
            "SELECT count(*) FROM player_games WHERE player_id IN (SELECT player_id FROM players WHERE identity_status<>'canonical')",
        )
        assert pg == [(0,)]
        q = _q(imported, "SELECT count(*) FROM quarantine WHERE reason='duplicate_identity'")
        assert q == [(2,)]
        diffs = _rows(imported, "SELECT * FROM quality_issues WHERE rule_id='duplicate_identity_cell_diff'")
        by_key = {d["row_key"].split("|")[0]: d["explanation"] for d in diffs}
        assert '"goals"' in by_key["legacy:green_will_08092005"]
        assert '"games_played": ["1", "24"]' in by_key["legacy:steele_roan_22102001"]
        files = dict(_q(imported, "SELECT path, disposition FROM source_files WHERE path LIKE '%green_william%'"))
        assert files["data/player_data/green_william_08092005_performance_details.csv"] == "quarantined"

    def test_birth_dates(self, imported: legacy.DatasetCandidate) -> None:
        rows = {
            r["player_id"]: r
            for r in _rows(
                imported, "SELECT player_id, birth_date, birth_date_quality, height_cm, weight_kg FROM players"
            )
        }
        casey = rows["legacy:casey_dick_01011900"]
        assert casey["birth_date"] is None and casey["birth_date_quality"] == "unknown"
        assert casey["height_cm"] is None and casey["weight_kg"] is None
        ghost = rows["legacy:ghost_gary_02021960"]
        assert ghost["birth_date_quality"] == "legacy_filename"


class TestMatchesAndStages:
    def test_d09_drawn_final_and_replay(self, imported: legacy.DatasetCandidate) -> None:
        gf = _q(
            imported,
            "SELECT match_id, replay_occurrence, match_date FROM matches WHERE season=1977 AND stage_id='gf' ORDER BY 2",
        )
        assert [r[1] for r in gf] == [0, 1] and gf[0][0] != gf[1][0]
        dench = _q(
            imported,
            "SELECT match_id, link_method, date_quality FROM player_games "
            "WHERE player_id='legacy:dench_david_05091951' AND stage_id='gf' ORDER BY career_game_counter",
        )
        assert [d[0] for d in dench] == [gf[0][0], gf[1][0]]
        assert {d[1] for d in dench} == {"row_order"} and {d[2] for d in dench} == {"inferred"}

    def test_d10_postponed_low_round_keeps_chronological_order(self, imported: legacy.DatasetCandidate) -> None:
        order = dict(_q(imported, "SELECT DISTINCT stage_id, stage_order FROM matches WHERE season=2025"))
        assert order["r01"] < order["r02"]
        late = _q(imported, "SELECT max(match_date) FROM matches WHERE season=2025 AND stage_id='r01'")
        assert late == [(date(2025, 8, 27),)]  # the postponed round-1 match stays round 1

    def test_d12_same_day_matches_distinct(self, imported: legacy.DatasetCandidate) -> None:
        rows = _q(imported, "SELECT match_id FROM matches WHERE match_date = DATE '2025-03-15'")
        assert len(rows) == 2 and rows[0] != rows[1]

    def test_wildcard_final_linked_and_verified(self, imported: legacy.DatasetCandidate) -> None:
        (row,) = _rows(
            imported,
            "SELECT m.stage_type, m.round_number, p.date_quality FROM player_games p JOIN matches m USING (match_id) "
            "WHERE p.player_id='legacy:green_will_08092005'",
        )
        assert row == {"stage_type": "final", "round_number": None, "date_quality": "fixture_verified"}

    def test_scores_status_and_attendance(self, imported: legacy.DatasetCandidate) -> None:
        rows = {
            r["stage_label"] + ":" + r["home_source_name"]: r
            for r in _rows(imported, "SELECT * FROM matches WHERE season IN (2025, 2026)")
        }
        geel = rows["2:Geelong"]
        assert geel["attendance"] is None and geel["away_score"] == 4 and geel["away_final_goals"] == 0
        assert rows["1:Sydney"]["home_score"] == 20 * 6 + 12
        assert rows["Grand Final:Sydney"]["status"] == "scheduled" and rows["Grand Final:Sydney"]["home_score"] is None
        assert rows["3:Brisbane Lions"]["status"] == "unknown"
        challenge = rows["Challenge Final:Sydney"]
        assert challenge["stage_type"] == "other" and challenge["round_number"] is None

    def test_extra_time_final_row_repaired_from_player_evidence(self, imported: legacy.DatasetCandidate) -> None:
        # legacy extra-time finals put a 'g.b.p' score in the team_2 name cell; team 2's cells are
        # shifted. The match entity is created with the opponent taken from unanimous two-way
        # player-row evidence; ALL scores are null (never guessed) and the raw row is kept.
        (m,) = _rows(imported, "SELECT * FROM matches WHERE season=2025 AND stage_id='sf'")
        assert m["match_id"] == "m:2025:sf:collingwood:west_coast:0"
        assert (m["home_club_id"], m["away_club_id"]) == ("west_coast", "collingwood")
        score_cols = [c for c in m if c.endswith(("_goals", "_behinds", "_score"))]
        assert len(score_cols) == 18 and all(m[c] is None for c in score_cols)
        assert m["status"] == "unknown" and str(m["match_date"])[:10] == "2025-09-12"
        (issue,) = _rows(imported, "SELECT * FROM quality_issues WHERE rule_id='source_row_malformed'")
        assert issue["severity"] == "error" and issue["row_key"] == m["match_id"] and "10.14.74" in issue["explanation"]
        linked = _q(
            imported,
            "SELECT player_id FROM player_games WHERE match_id='m:2025:sf:collingwood:west_coast:0' ORDER BY 1",
        )
        assert linked == [("legacy:eagle_ed_04041998",), ("legacy:pie_pete_05051997",)]
        (ev,) = _rows(imported, "SELECT reason, raw FROM quarantine WHERE table_name='matches'")
        assert ev["reason"] == "source_row_malformed_evidence"
        assert json.loads(ev["raw"])["team_2_team_name"] == "10.14.74"
        assert imported.report["row_accounting"]["matches"]["quarantined"] == 0  # imported; evidence retained

    def test_extra_time_row_without_two_way_evidence_stays_quarantined(self, tmp_path: Path) -> None:
        src = _copy_corpus(tmp_path)
        for f in (src / "data" / "player_data").glob("pie_pete_*"):
            f.unlink()
        cand = legacy.import_legacy(src, _ctx(tmp_path / "var"))
        q = _rows(cand, "SELECT reason, raw FROM quarantine WHERE table_name='matches'")
        assert len(q) == 1 and json.loads(q[0]["raw"])["team_2_team_name"] == "10.14.74"
        eagle = _q(cand, "SELECT reason FROM quarantine WHERE source_path LIKE '%eagle_ed%'")
        assert eagle == [("unlinked_no_match",)]

    def test_date_quality(self, imported: legacy.DatasetCandidate) -> None:
        dq = dict(_q(imported, "SELECT player_id, date_quality FROM player_games WHERE season=2026 AND stage_id='r01'"))
        assert dq["legacy:lynch_tom_09101992"] == "inferred"  # 2026-03-01 placeholder
        assert dq["legacy:lynch_tom_31101990"] == "fixture_verified"


class TestDedupAndLineups:
    def test_duplicate_player_game_keeps_verified_row(self, imported: legacy.DatasetCandidate) -> None:
        rows = _q(
            imported,
            "SELECT date_quality, source_row FROM player_games WHERE player_id='legacy:hawk_harry_03031999'",
        )
        assert rows == [("fixture_verified", 2)]

    def test_duplicate_display_name_resolves_to_canonical(self, imported: legacy.DatasetCandidate) -> None:
        rows = _q(imported, "SELECT player_id, resolution FROM lineups WHERE name_token='William Green'")
        assert rows == [("legacy:green_will_08092005", "match_participation")]

    def test_multipart_surname_resolved_within_match(self, imported: legacy.DatasetCandidate) -> None:
        rows = _q(imported, "SELECT player_id, resolution FROM lineups WHERE name_token='Matt de Boer'")
        assert rows == [("legacy:boer_matt_10031990", "match_first_last_word")]

    def test_lineup_resolution(self, imported: legacy.DatasetCandidate) -> None:
        nk = _rows(imported, "SELECT reason, candidates FROM quarantine WHERE table_name='lineups' ORDER BY reason")
        reasons = [r["reason"] for r in nk]
        assert reasons == ["lineup_row_unlinked", "lineup_token_unresolved"]
        rep = imported.report["row_accounting"]["lineups"]
        assert rep["tokens_read"] == rep["tokens_resolved"] + rep["tokens_quarantined"]
        assert rep["rows_read"] == rep["rows_linked"] + rep["rows_quarantined"]
        dench = _q(
            imported, "SELECT count(DISTINCT match_id) FROM lineups WHERE player_id='legacy:dench_david_05091951'"
        )
        assert dench == [(3,)]


class TestOtherFamilies:
    def test_predictions_are_legacy_unknown(self, imported: legacy.DatasetCandidate) -> None:
        rows = _rows(imported, "SELECT * FROM legacy_predictions ORDER BY source_path, source_row")
        assert {r["origin"] for r in rows} == {"legacy_unknown"}
        nr = {r["player_display_name"]: r for r in rows if r["artifact_kind"] == "next_round"}
        assert nr["Lynch Tom"]["player_id"] == "legacy:lynch_tom_31101990"
        assert nr["Lynch Tom"]["claimed_round_label"] == "2" and nr["Lynch Tom"]["claimed_timestamp"] == "20260430_1459"
        assert nr["Nobody Known"]["player_id"] is None
        pva = next(r for r in rows if r["artifact_kind"] == "prediction_vs_actual")
        assert pva["claimed_season"] == 2025 and pva["claimed_actual"] == 5.0

    def test_two_top100_schemas_distinct(self, imported: legacy.DatasetCandidate) -> None:
        scores = _rows(imported, "SELECT * FROM legacy_rank_scores ORDER BY rank")
        assert scores[0]["player_id"] == "legacy:dench_david_05091951" and scores[1]["player_id"] is None
        bios = _rows(imported, "SELECT * FROM legacy_top100_bios")
        assert bios[0]["player_name"] == "David Dench" and bios[0]["comment"] == "Bio text, with comma."
        yearly = _rows(imported, "SELECT * FROM legacy_rank_yearly")
        assert yearly[0]["season"] == 1977 and yearly[0]["games_played"] == 22

    def test_drafts_contracts_schools(self, imported: legacy.DatasetCandidate) -> None:
        ev = {r["player_name"]: r for r in _rows(imported, "SELECT * FROM draft_events")}
        assert ev["Someone Else"]["club_id"] == "brisbane_lions"  # 1996 draft serves the 1997 season
        assert ev["Sam Schulz"]["club_id"] is None and ev["Sam Schulz"]["event_type"] == "rookie_end_season"
        con = {r["player_name"]: r for r in _rows(imported, "SELECT * FROM contract_observations")}
        assert con["Tom Lynch"]["source_name"] == "AFL.com.au" and con["Tom Lynch"]["observed_at"] is None
        assert con["Tom Lynch"]["source_type"] == "legacy" and con["Max Mover"]["club_id"] == "greater_western_sydney"
        sch = _rows(imported, "SELECT * FROM school_observations")
        assert {s["classifier_version"] for s in sch} == {"legacy"}

    def test_live_snapshots(self, imported: legacy.DatasetCandidate) -> None:
        rows = {r["payload_kind"]: r for r in _rows(imported, "SELECT * FROM live_snapshots")}
        state = rows["state_json"]
        assert state["home_score"] == 65 and state["match_id"] == "m:2026:r01:carlton:sydney:0"
        players = rows["players_csv"]
        assert players["home_score"] is None and players["source_game_id"] == "9999"


class TestAccounting:
    def test_every_file_accounted(self, imported: legacy.DatasetCandidate) -> None:
        on_disk = sorted(str(p.relative_to(FIXTURE)) for p in FIXTURE.rglob("*") if p.is_file())
        files = {f["path"]: f for f in imported.report["files"]}
        assert sorted(files) == on_disk
        assert {f["disposition"] for f in files.values()} <= {"imported", "ignored", "quarantined"}
        assert files["data/notes.txt"]["disposition"] == "quarantined"
        assert files["data/era_stats.csv"]["disposition"] == "ignored" and files["data/era_stats.csv"]["reason"]
        assert files["data/live_snapshots/.gitkeep"]["disposition"] == "ignored"
        for f in files.values():
            assert len(f["sha256"]) == 64 and (f["bytes"] > 0 or f["path"].endswith(".gitkeep"))
            if f["rows_read"] is not None:
                assert f["rows_read"] == f["rows_imported"] + f["rows_quarantined"], f["path"]
        assert imported.report["families"]["conceded_stats"]["status"] == "absent"

    def test_per_season_player_rows_reconcile(self, imported: legacy.DatasetCandidate) -> None:
        acc = imported.report["row_accounting"]["player_games"]
        for season, c in acc["by_season"].items():
            assert c["read"] == c["imported"] + c["quarantined"], season
        (n,) = _q(imported, "SELECT count(*) FROM player_games")[0]
        assert acc["imported"] == n
        (nq,) = _q(imported, "SELECT count(*) FROM quarantine WHERE table_name='player_games'")[0]
        assert acc["quarantined"] == nq

    def test_absent_optional_families(self, tmp_path: Path) -> None:
        src = _copy_corpus(tmp_path)
        for sub in ["drafts", "contracts", "live_snapshots", "prediction", "top100"]:
            shutil.rmtree(src / "data" / sub)
        (src / "all_time_top_100.csv").unlink()
        cand = legacy.import_legacy(src, _ctx(tmp_path / "var"))
        fam = cand.report["families"]
        assert fam["drafts_national"]["status"] == "absent" and fam["matches"]["status"] == "present"

    def test_missing_mandatory_family_fails(self, tmp_path: Path) -> None:
        src = _copy_corpus(tmp_path)
        shutil.rmtree(src / "data" / "matches")
        with pytest.raises(legacy.LegacyImportError):
            legacy.import_legacy(src, _ctx(tmp_path / "var"))

    def test_report_written(self, imported: legacy.DatasetCandidate, tmp_path: Path) -> None:
        out = tmp_path / "r.json"
        legacy.write_import_report(imported, out)
        data = json.loads(out.read_text())
        assert data["snapshot_id"] == imported.candidate.manifest.snapshot_id


class TestIdempotence:
    def test_rerun_same_snapshot(self, tmp_path: Path) -> None:
        src = _copy_corpus(tmp_path)
        a = legacy.import_legacy(src, _ctx(tmp_path / "var1"))
        b = legacy.import_legacy(src, _ctx(tmp_path / "var2"))
        c = legacy.import_legacy(src, _ctx(tmp_path / "var1"))
        assert a.candidate.manifest.snapshot_id == b.candidate.manifest.snapshot_id == c.candidate.manifest.snapshot_id

    def test_d12_corrected_fixture_date_keeps_match_id(self, tmp_path: Path) -> None:
        src = _copy_corpus(tmp_path)
        a = legacy.import_legacy(src, _ctx(tmp_path / "var1"))
        path = src / "data" / "matches" / "matches_2025.csv"
        path.write_text(path.read_text().replace("2025-03-07 19:40", "2025-03-08 19:40"), encoding="utf-8")
        b = legacy.import_legacy(src, _ctx(tmp_path / "var2"))
        ids_a = _q(a, "SELECT match_id FROM matches ORDER BY 1")
        ids_b = _q(b, "SELECT match_id FROM matches ORDER BY 1")
        assert ids_a == ids_b
        assert a.candidate.manifest.snapshot_id != b.candidate.manifest.snapshot_id
        moved = _q(b, "SELECT match_date FROM matches WHERE match_id='m:2025:r01:hawthorn:sydney:0'")
        assert moved == [(date(2025, 3, 8),)]


def test_arrow_types_import() -> None:
    assert pa.int32() == TABLES["player_games"].arrow_schema().field("kicks").type
