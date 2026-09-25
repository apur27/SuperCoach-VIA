"""Draft/contract/school list views: stable ordering, source disclosure, unresolved names."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from supercoach_via.analytics import lists
from supercoach_via.storage.queries import SnapshotQuery
from tests.scvia.fixtures.analytics import synthetic as syn


@pytest.fixture
def snap(tmp_path: Path):  # type: ignore[no-untyped-def]
    tables = {
        "clubs": [syn.club("GEE", "Geelong")],
        "draft_events": [
            {"draft_event_id": "d2", "season": 2004, "event_type": "national", "draft_round": 1, "pick": 2,
             "club_id": "GEE", "club_source_name": "Geelong", "player_name": "B Two", "player_id": "legacy:b",
             "source_family": "afl_draft_history"},
            {"draft_event_id": "d1", "season": 2004, "event_type": "national", "draft_round": 1, "pick": 1,
             "club_id": None, "club_source_name": "Kangaroos", "player_name": "A One", "player_id": None,
             "source_family": "afl_draft_history"},
            {"draft_event_id": "r1", "season": 2004, "event_type": "rookie_end_season", "pick": 1,
             "club_source_name": "Richmond", "player_name": "C Three", "source_family": "afl_rookie_draft_history"},
        ],
        "contract_observations": [
            {"observation_id": "c1", "player_name": "Chayce Jones", "club_id": "GEE", "contract_end": 2026,
             "fa_category": "Unrestricted FA", "source_type": "manual_fixture", "confidence": "medium",
             "observed_at": date(2026, 7, 1), "notes": "Unrestricted FA (AFL.com.au)"},
            {"observation_id": "c2", "player_name": "Old Guy", "player_id": "legacy:o", "contract_end": 2027,
             "source_type": "legacy", "confidence": "low"},
        ],
        "school_observations": [
            {"observation_id": "s1", "draft_year": 2004, "pick": 1, "player_name": "A One",
             "school": None, "school_type": "unknown", "classifier_version": "v1", "confidence": "low"},
        ],
    }
    return tmp_path, syn.build(tmp_path, tables)


def test_draft_rows_are_ordered_and_named(snap) -> None:  # type: ignore[no-untyped-def]
    root, manifest = snap
    with SnapshotQuery(root, manifest) as q:
        rows = lists.draft_rows(q, 2004)
    assert [(r.event_type, r.pick) for r in rows] == [("national", 1), ("national", 2), ("rookie_end_season", 1)]
    assert rows[1].club == "Geelong" and rows[0].club == "Kangaroos"
    assert rows[0].player_id is None


def test_lists_season_discloses_source_mode(snap) -> None:  # type: ignore[no-untyped-def]
    root, manifest = snap
    with SnapshotQuery(root, manifest) as q:
        season = lists.lists_season(q, 2026)
    assert [c.player_name for c in season.contracts] == ["Chayce Jones"]
    assert season.contracts[0].source_type == "manual_fixture"
    assert "manual_fixture" in season.source_note and "not guaranteed current" in season.source_note
    assert any("contract" in s.label.lower() for s in season.sources)


def test_unresolved_names_listed(snap) -> None:  # type: ignore[no-untyped-def]
    root, manifest = snap
    with SnapshotQuery(root, manifest) as q:
        un = lists.unresolved_names(q)
    assert sorted(zip(un["table_name"], un["record_id"], strict=True)) == [
        ("contract_observations", "c1"), ("draft_events", "d1"), ("draft_events", "r1"),
        ("school_observations", "s1"),
    ]


def test_missing_tables_give_empty_views(tmp_path: Path) -> None:
    manifest = syn.build(tmp_path, {"clubs": [syn.club("A", "A")]})
    with SnapshotQuery(tmp_path, manifest) as q:
        season = lists.lists_season(q, 2026)
        assert lists.unresolved_names(q).empty
    assert season.drafts == [] and season.contracts == [] and season.schools == []
    assert "no contract observations" in season.source_note.lower()
