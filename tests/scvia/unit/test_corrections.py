"""Bounded, source-backed corrections of an accepted snapshot (fixture fields, venues, blanks)."""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from supercoach_via import pipeline
from supercoach_via.ingest import corrections
from supercoach_via.settings import RunContext, Settings
from supercoach_via.storage.queries import SnapshotQuery
from supercoach_via.storage.snapshots import load_snapshot, read_current
from tests.scvia.unit import integrity_fixtures as fx

R02 = "m:2026:r02:alpha:beta:0"
NOW = datetime(2026, 9, 29, tzinfo=UTC)


def _resolver(name: str, _season: int) -> str | None:
    return name.lower()


def _match(root: Path, mid: str) -> dict[str, Any]:
    with SnapshotQuery(root, load_snapshot(root), tables={"matches"}) as q:
        return q.arrow("SELECT * FROM matches WHERE match_id = ?", [mid]).to_pylist()[0]


def _built(root: Path, *, snapshot_attendance: int) -> None:
    """Pinned page says 1000 for R02; the snapshot holds ``snapshot_attendance``."""
    src = copy.deepcopy(fx.tables())

    def mutate(r: dict[str, list[dict[str, Any]]]) -> None:
        next(m for m in r["matches"] if m["match_id"] == R02).update(attendance=snapshot_attendance, venue_id=None)

    fx.with_sources(root, page_rows=src, mutate=mutate)


def test_attendance_is_corrected_from_the_pinned_capture_with_provenance(tmp_path: Path) -> None:
    _built(tmp_path, snapshot_attendance=0)
    m = load_snapshot(tmp_path)
    ups = corrections.fixture_corrections(tmp_path, m, 2026, club_resolver=_resolver,
                                          venue_resolver=lambda n: "oval" if n == "Oval" else None)  # fmt: skip
    [row] = [r for r in ups["matches"] if r["match_id"] == R02]
    assert row["attendance"] == 1000 and row["venue_id"] == "oval"
    assert row["provenance"] == "legacy_import"  # the row is still the legacy row, corrected field by field
    [issue] = [i for i in ups["quality_issues"] if i["row_key"] == R02 and "attendance" in i["explanation"]]
    assert issue["status"] == "resolved" and issue["rule_id"] == "fixture_field_corrected"
    assert m.source_revisions["afltables:season:2026"] in issue["explanation"]
    assert issue["source_path"] == fx.SEASON_URL


def test_matching_values_and_unpinned_seasons_are_untouched(tmp_path: Path) -> None:
    _built(tmp_path, snapshot_attendance=1000)
    m = load_snapshot(tmp_path)
    ups = corrections.fixture_corrections(tmp_path, m, 2026, club_resolver=_resolver, venue_resolver=lambda n: "oval")
    assert [r["match_id"] for r in ups["matches"]] == [R02]  # venue_id only
    assert all("attendance" not in i["explanation"] for i in ups["quality_issues"])
    assert corrections.fixture_corrections(tmp_path, m, 1970, club_resolver=_resolver,
                                           venue_resolver=lambda n: None) == {"matches": [], "quality_issues": []}  # fmt: skip


def test_score_disagreement_is_not_patched_field_by_field(tmp_path: Path) -> None:
    def mutate(r: dict[str, list[dict[str, Any]]]) -> None:
        next(m for m in r["matches"] if m["match_id"] == R02).update(attendance=0, home_q3_goals=0)

    fx.with_sources(tmp_path, mutate=mutate)
    ups = corrections.fixture_corrections(tmp_path, load_snapshot(tmp_path), 2026, club_resolver=_resolver,
                                          venue_resolver=lambda n: "oval")  # fmt: skip
    assert not [r for r in ups["matches"] if r["match_id"] == R02 and r["attendance"] != 0]
    assert any(i["rule_id"] == "fixture_correction_refused" for i in ups["quality_issues"])


def test_missing_capture_is_an_error_not_a_silent_skip(tmp_path: Path) -> None:
    shas = fx.with_sources(tmp_path)
    (tmp_path / "raw" / "objects" / shas["season"][:2] / shas["season"]).unlink()
    with pytest.raises(corrections.CorrectionError):
        corrections.fixture_corrections(tmp_path, load_snapshot(tmp_path), 2026, club_resolver=_resolver,
                                        venue_resolver=lambda n: None)  # fmt: skip


def test_apply_corrections_stage_promotes_a_validated_child_snapshot(tmp_path: Path, monkeypatch: Any) -> None:
    _built(tmp_path / "var", snapshot_attendance=0)
    parent = read_current(tmp_path / "var").snapshot_id
    monkeypatch.setattr(corrections, "default_resolvers", lambda _cfg: (_resolver, lambda n: "oval"))
    ctx = RunContext(settings=Settings(data_root=tmp_path / "var", output_root=tmp_path / "dist"), clock=lambda: NOW)
    res = pipeline.apply_corrections(ctx, seasons=[2026])
    assert res.exit_code == 0 and res.promoted, res.message
    child = load_snapshot(tmp_path / "var")
    assert child.parent == parent and child.snapshot_id != parent
    assert _match(tmp_path / "var", R02)["attendance"] == 1000
    again = pipeline.apply_corrections(ctx, seasons=[2026])
    assert again.exit_code == 0 and not again.promoted  # idempotent: nothing left to correct
    assert read_current(tmp_path / "var").snapshot_id == child.snapshot_id


def test_refresh_applies_fixture_corrections_from_the_page_it_pinned(tmp_path: Path, monkeypatch: Any) -> None:
    """Root cause: a refresh pinned the season page but only upserted matches whose score changed."""
    from supercoach_via.domain.schemas import CheckOutcome, DatasetStatus
    from supercoach_via.ingest import refresh as rf

    root = tmp_path / "var"
    _built(root, snapshot_attendance=0)
    base = load_snapshot(root)
    sha = base.source_revisions["afltables:season:2026"]
    obs = {"source_ref": "src:again", "adapter": "afltables.season_fixture", "adapter_version": "1",
           "url": fx.SEASON_URL, "fetched_at": fx.CHECKED, "content_sha256": sha, "http_status": 200, "etag": None,
           "last_modified": None, "bytes": 1, "source_mode": "live", "outcome": "PASS"}  # fmt: skip

    def fake_refresh(_base: object, plan: rf.RefreshPlan, _ctx: object, **_kw: object) -> rf.RefreshResult:
        return rf.RefreshResult(
            plan=plan, outcome=CheckOutcome.PASS, dataset_status=DatasetStatus.VERIFIED, exit_code=0,
            source_checked_at=fx.CHECKED, counts={}, upserts={"source_observations": [obs]},
            revisions={"afltables:season:2026": sha}, corrections=[], superseded=[], interior_gaps=[],
            stale_fixtures=[], work_log=[], issues=[], latest_completed_match_date=None, request_counts={},
            bytes_received=0,
        )  # fmt: skip

    monkeypatch.setattr(rf, "refresh_sources", fake_refresh)
    monkeypatch.setattr(corrections, "default_resolvers", lambda _cfg: (_resolver, lambda n: "oval"))
    ctx = RunContext(settings=Settings(data_root=root, output_root=tmp_path / "dist"), clock=lambda: NOW)
    res = pipeline.refresh(ctx, season=2026)
    assert res.exit_code == 0 and res.promoted, res.message
    assert _match(root, R02)["attendance"] == 1000


def test_cli_apply_corrections(tmp_path: Path, monkeypatch: Any) -> None:
    import json

    from typer.testing import CliRunner

    from supercoach_via.cli import app

    root = tmp_path / "var"
    _built(root, snapshot_attendance=0)
    monkeypatch.setattr(corrections, "default_resolvers", lambda _cfg: (_resolver, lambda n: "oval"))
    res = CliRunner().invoke(app, ["apply-corrections", "--data-root", str(root), "--season", "2026", "--json"])
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["promoted"] is True and out["outputs"]["corrections"]["matches"] == 1
    missing = CliRunner().invoke(app, ["apply-corrections", "--data-root", str(tmp_path / "none"), "--season", "2026"])
    assert missing.exit_code != 0


# -- O55-06: player rows of a replayed drawn final --------------------------------------------

DRAW, REPLAY = "m:1970:sf:alpha:beta:0", "m:1970:sf:alpha:beta:1"


def _replay_corpus(root: Path, suspect_goals: int, *, conflict: bool = False) -> None:
    from datetime import date

    def mutate(r: dict[str, list[dict[str, Any]]]) -> None:
        draw = fx._match(DRAW, 1970, "Semi Final", 20, date(1970, 9, 5), "alpha", "beta", (2, 5), (2, 5))
        replay = fx._match(REPLAY, 1970, "Semi Final", 20, date(1970, 9, 12), "alpha", "beta", (4, 4), (2, 2))
        for m in (draw, replay):
            m.update(stage_type="final", stage_id="sf", round_number=None)
        replay["replay_occurrence"] = 1
        r["matches"] += [draw, replay]
        r["players"].append({**r["players"][0], "player_id": "legacy:p5", "display_name": "Player 5", "last_name": "5"})
        rows = []
        for m, who in ((draw, [("legacy:p1", "alpha", 1), ("legacy:p2", "alpha", 1), ("legacy:p3", "beta", 1),
                                ("legacy:p4", "beta", 1)]),
                       (replay, [("legacy:p1", "alpha", 2), ("legacy:p3", "beta", 1), ("legacy:p4", "beta", 1)])):  # fmt: skip
            for pid, club, goals in who:
                g = fx._game(m, pid, club, 50, dict(fx.OLD_STATS))
                g["goals"] = goals
                rows.append(g)
        # p5 played the replay (alpha won it) but the legacy row-order link put the row on the draw
        suspect = fx._game(draw, "legacy:p5", "alpha", 51, dict(fx.OLD_STATS))
        suspect.update(goals=suspect_goals, result="W", link_method="row_order", date_quality="inferred")
        rows.append(suspect)
        if conflict:
            other = fx._game(replay, "legacy:p5", "alpha", 52, dict(fx.OLD_STATS))
            other.update(goals=0)
            rows.append(other)
        # replay alpha scored 4: p1's 2 plus the suspect's; with suspect_goals=0 the totals cannot place it
        r["player_games"] += rows
        r["seasons"][0].update(matches_complete=4, last_match_date=date(1970, 9, 12))

    fx.rehash(root, mutate)


def _pg(root: Path, player: str) -> list[dict[str, Any]]:
    with SnapshotQuery(root, load_snapshot(root), tables={"player_games"}) as q:
        return q.arrow("SELECT * FROM player_games WHERE player_id = ? ORDER BY match_id", [player]).to_pylist()


def test_replay_row_is_relinked_when_official_totals_prove_it(tmp_path: Path, monkeypatch: Any) -> None:
    root = tmp_path / "var"
    _replay_corpus(root, suspect_goals=2)
    ups = corrections.replay_link_corrections(root, load_snapshot(root))
    assert [(d["match_id"], d["player_id"]) for d in ups["deletes"]] == [(DRAW, "legacy:p5")]
    [moved] = ups["player_games"]
    assert (moved["match_id"], moved["link_method"], moved["result"]) == (REPLAY, "score_reconciled", "W")
    assert any(i["rule_id"] == "replay_link_corrected" and i["row_key"] == f"{DRAW}|legacy:p5|alpha"
               for i in ups["quality_issues"])  # fmt: skip
    monkeypatch.setattr(corrections, "default_resolvers", lambda _cfg: (_resolver, lambda n: "oval"))
    ctx = RunContext(settings=Settings(data_root=root, output_root=tmp_path / "dist"), clock=lambda: NOW)
    res = pipeline.apply_corrections(ctx, seasons=[])
    assert res.exit_code == 0 and res.promoted, res.message
    assert [r["match_id"] for r in _pg(root, "legacy:p5")] == [REPLAY]


def test_row_the_totals_cannot_place_is_quarantined(tmp_path: Path) -> None:
    root = tmp_path / "var"
    _replay_corpus(root, suspect_goals=0)
    ups = corrections.replay_link_corrections(root, load_snapshot(root))
    assert ups["player_games"] == []
    assert [(d["match_id"], d["player_id"]) for d in ups["deletes"]] == [(DRAW, "legacy:p5")]
    [q] = ups["quarantine"]
    assert q["reason"] == "replayed_draw_link_unresolved" and DRAW in q["candidates"] and REPLAY in q["candidates"]
    assert any(i["rule_id"] == "replay_link_quarantined" and i["remediation"] for i in ups["quality_issues"])


def test_player_already_on_the_replay_is_quarantined_not_merged(tmp_path: Path) -> None:
    root = tmp_path / "var"
    _replay_corpus(root, suspect_goals=2, conflict=True)
    ups = corrections.replay_link_corrections(root, load_snapshot(root))
    assert ups["player_games"] == [] and [q["reason"] for q in ups["quarantine"]] == ["replayed_draw_link_unresolved"]
