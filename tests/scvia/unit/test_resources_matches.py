"""Match and game-log public resources from canonical tables (zero vs missing, replays)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from supercoach_via.publish import resources
from supercoach_via.publish.view_models import box_rows, game_rows
from supercoach_via.storage.queries import SnapshotQuery

from .snapshot_factory import build

NOW = datetime(2026, 5, 1, tzinfo=UTC)


def _m(mid: str, stage: str, occ: int, hg: int | None, ag: int | None, status: str, d: date | None) -> dict:
    return {
        "match_id": mid,
        "season": 2025,
        "stage_label": stage,
        "stage_type": "final" if stage == "GF" else "regular",
        "round_number": None if stage == "GF" else int(stage),
        "stage_order": 99 if stage == "GF" else int(stage),
        "stage_id": "gf" if stage == "GF" else f"r{int(stage):02d}",
        "replay_occurrence": occ,
        "home_club_id": "demo-harbour",
        "away_club_id": "demo-ridge",
        "home_source_name": "Demo Harbour",
        "away_source_name": "Demo Ridge",
        "venue_source_name": "Demo Oval 1",
        "local_start": None if d is None else f"{d} 14:30",
        "match_date": d,
        "date_precision": "minute" if d else "unknown",
        "status": status,
        "home_final_goals": hg,
        "home_final_behinds": None if hg is None else 0,
        "away_final_goals": ag,
        "away_final_behinds": None if ag is None else 0,
        "home_score": None if hg is None else hg * 6,
        "away_score": None if ag is None else ag * 6,
        "provenance": "demo",
    }


def _snap(root: Path):
    matches = [
        _m("m:2025:r01:a", "1", 0, 0, 3, "complete", date(2025, 3, 20)),
        _m("m:2025:gf:a:0", "GF", 0, 5, 5, "complete", date(2025, 9, 27)),
        _m("m:2025:gf:a:1", "GF", 1, 7, 5, "complete", date(2025, 10, 4)),
        _m("m:2025:r02:a", "2", 0, None, None, "scheduled", date(2025, 3, 27)),
    ]
    clubs = [
        {"club_id": "demo-harbour", "name": "Demo Harbour", "lineage_id": "demo-harbour", "active": True},
        {"club_id": "demo-ridge", "name": "Demo Ridge", "lineage_id": "demo-ridge", "active": True},
    ]
    players = [
        {
            "player_id": "legacy:demo_a_01011990",
            "display_name": "Demo A",
            "birth_date_quality": "source",
            "identity_status": "canonical",
            "provenance": "demo",
        }
    ]
    games = [
        {
            "match_id": "m:2025:r01:a",
            "player_id": "legacy:demo_a_01011990",
            "club_id": "demo-harbour",
            "season": 2025,
            "opponent_club_id": "demo-ridge",
            "stage_label": "1",
            "stage_id": "r01",
            "match_date": date(2025, 3, 3),
            "date_quality": "inferred",
            "career_game_counter": 1,
            "kicks": 0,
            "handballs": 0,
            "disposals": 0,
            "tackles": None,
            "revision_id": "r",
            "provenance": "demo",
        }
    ]
    return build(root, lambda: NOW, {"matches": matches, "clubs": clubs, "players": players, "player_games": games})


def test_match_summaries_zero_missing_replay(tmp_path: Path) -> None:
    manifest = _snap(tmp_path)
    with SnapshotQuery(tmp_path, manifest) as q:
        idx = resources.match_index(q, 2025)
    by = {m.match_id: m for m in idx.matches}
    assert [m.match_id for m in idx.matches][:2] == ["m:2025:r01:a", "m:2025:r02:a"]  # chronological
    assert by["m:2025:r01:a"].home.score == 0 and by["m:2025:r01:a"].winner_club_id == "demo-ridge"
    assert by["m:2025:r02:a"].home.score is None and by["m:2025:r02:a"].status == "scheduled"
    assert by["m:2025:gf:a:0"].winner_club_id is None  # draw
    assert by["m:2025:gf:a:1"].replay_occurrence == 1 and by["m:2025:gf:a:1"].winner_club_id == "demo-harbour"


def test_match_detail_and_game_logs_keep_nulls(tmp_path: Path) -> None:
    manifest = _snap(tmp_path)
    with SnapshotQuery(tmp_path, manifest) as q:
        details = {d.summary.match_id: d for d in resources.match_details(q, 2025)}
        logs = list(resources.player_season_games(q, 2025))
    d = details["m:2025:r01:a"]
    cols = d.stat_columns
    # only stats observed somewhere in the file are listed; tackles was never observed
    assert "tackles" not in cols and {"kicks", "handballs", "disposals"} <= set(cols)
    row = dict(zip(cols, box_rows(d.home_players)[0].stats, strict=True))
    assert row["disposals"] == 0  # zero stays zero
    assert logs[0].player_id == "legacy:demo_a_01011990"
    shown = game_rows(logs[0].games)[0]
    assert shown.match_date == date(2025, 3, 20) and shown.date_quality == "source"
    g = dict(zip(logs[0].stat_columns, game_rows(logs[0].games)[0].stats, strict=True))
    assert g["disposals"] == 0 and "tackles" not in g
    assert game_rows(logs[0].games)[0].opponent_name == "Demo Ridge"


def test_match_source_label_follows_recorded_provenance(tmp_path: Path) -> None:
    url = "https://afltables.com/afl/stats/games/2026/081920260926.html"
    matches = [
        _m("m:2025:r01:a", "1", 0, 0, 3, "complete", date(2025, 3, 20)),
        _m("m:2025:r02:a", "2", 0, None, None, "complete", date(2025, 3, 27)),
    ]
    matches[0]["provenance"] = "legacy_import"
    matches[0]["source_path"] = "data/matches/matches_2025.csv"
    matches[1]["provenance"] = "source_fetch"
    matches[1]["source_path"] = url
    games = [
        {
            "match_id": "m:2025:r02:a",
            "player_id": "legacy:demo_a_01011990",
            "club_id": "demo-harbour",
            "season": 2025,
            "opponent_club_id": "demo-ridge",
            "stage_label": "2",
            "stage_id": "r02",
            "match_date": date(2025, 3, 27),
            "date_quality": "fixture_verified",
            "kicks": 2,
            "handballs": None,
            "disposals": 2,
            "revision_id": "r",
            "provenance": "source_fetch",
            "source_path": url,
        }
    ]
    players = [
        {
            "player_id": "legacy:demo_a_01011990",
            "display_name": "Demo A",
            "birth_date_quality": "source",
            "identity_status": "canonical",
            "provenance": "source_fetch",
        }
    ]
    clubs = [
        {"club_id": "demo-harbour", "name": "Demo Harbour", "lineage_id": "demo-harbour", "active": True},
        {"club_id": "demo-ridge", "name": "Demo Ridge", "lineage_id": "demo-ridge", "active": True},
    ]
    manifest = build(tmp_path, lambda: NOW, {"matches": matches, "clubs": clubs, "players": players, "player_games": games})
    with SnapshotQuery(tmp_path, manifest) as q:
        details = {d.summary.match_id: d for d in resources.match_details(q, 2025)}
    legacy = details["m:2025:r01:a"].sources[0]
    fetched = details["m:2025:r02:a"].sources[0]
    assert legacy.label == "Legacy match/player CSV import" and legacy.url is None
    assert legacy.note == "data/matches/matches_2025.csv"
    assert fetched.label == "AFL Tables source page" and fetched.url == url and fetched.note == url
    cols = details["m:2025:r02:a"].stat_columns
    assert "handballs" not in cols
    row = dict(zip(cols, box_rows(details["m:2025:r02:a"].home_players)[0].stats, strict=True))
    assert row["kicks"] == 2 and row["disposals"] == 2


def test_stat_arrays_keep_partial_nulls() -> None:
    cols, rows = resources.compact_stats([{"kicks": 3, "marks": None}, {"kicks": None, "marks": 2}])
    assert cols == ["kicks", "marks"]
    assert rows == [[3.0, None], [None, 2.0]]


def test_public_key_roundtrip() -> None:
    canonical = resources.public_key("legacy:x_y_01011990")
    assert canonical.startswith("k.") and "__" not in canonical.split(".", 1)[0]
    assert ":" not in canonical
    assert resources.parse_public_key(canonical) == "legacy:x_y_01011990"
    # Historical links keep the colon-to-double-underscore alias. New paths do not emit it.
    assert resources.parse_public_key("legacy__x_y_01011990") == "legacy:x_y_01011990"
    assert resources.public_key("legacy:x_y_01011990") != "legacy__x_y_01011990"
    assert resources.parse_public_key(resources.match_key("m:2025:gf:a:1")) == "m:2025:gf:a:1"
    assert resources.public_key("a__b") != resources.public_key("id.a_x_b")
    assert resources.parse_public_key(resources.public_key("a__b")) == "a__b"
    assert resources.parse_public_key(resources.public_key("id.a_x_b")) == "id.a_x_b"
    assert resources.parse_public_key(resources.public_key("legacy:a___b")) == "legacy:a___b"


def test_player_index_search_terms_and_active(tmp_path: Path) -> None:
    manifest = _snap(tmp_path)
    with SnapshotQuery(tmp_path, manifest) as q:
        idx = resources.player_index(q)
    assert idx.count == 1
    e = idx.players[0]
    assert e.key == resources.public_key("legacy:demo_a_01011990") and e.clubs == ["Demo Harbour"]
    assert e.first_season == 2025 and e.last_season == 2025 and e.seasons == [2025]
    assert e.games == 1 and e.active is True


def test_normalise_search_strips_diacritics() -> None:
    assert resources.normalise_search("Zoë  O'Brien-Émile") == "zoe o'brien-emile"
