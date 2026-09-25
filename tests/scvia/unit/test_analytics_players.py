"""Coverage-aware player aggregates and leaders (A03; null-vs-zero; counter vs rows)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from supercoach_via.analytics import players
from supercoach_via.domain.metrics import CoverageEras
from supercoach_via.storage.queries import SnapshotQuery
from tests.scvia.fixtures.analytics import synthetic as syn

ERAS = CoverageEras({"tackles": 1987, "disposals": 1965, "goals": 1897})


@pytest.fixture(scope="module")
def snap(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    root = tmp_path_factory.mktemp("players")
    games = [
        # A: 1985-1988, tackles only recorded from 1987; counter leads rows (missing finals row)
        syn.pg("m1", "legacy:a", "CAR", 1985, 1, disposals=20, goals=1, tackles=None, match_date=date(1985, 4, 1)),
        syn.pg("m2", "legacy:a", "CAR", 1986, 2, disposals=None, goals=0, tackles=None, match_date=date(1986, 4, 1)),
        syn.pg("m3", "legacy:a", "CAR", 1987, 3, disposals=30, goals=2, tackles=5, match_date=date(1987, 4, 1)),
        syn.pg("m4", "legacy:a", "ESS", 1988, 5, disposals=10, goals=0, tackles=0, match_date=date(1988, 4, 1)),
        # B: a single game with every stat blank except goals
        syn.pg("m3", "legacy:b", "ESS", 1987, 1, disposals=None, goals=3, tackles=None, match_date=date(1987, 4, 1)),
        # C: 1988 only
        syn.pg("m4", "legacy:c", "CAR", 1988, 1, disposals=40, goals=0, tackles=2, match_date=date(1988, 4, 1)),
        syn.pg("m5", "legacy:c", "CAR", 1988, 2, disposals=35, goals=1, tackles=None, match_date=date(1988, 4, 8)),
    ]
    tables = {
        "players": [syn.player("legacy:a", "Alan Able"), syn.player("legacy:b", "Bob Baker"),
                    syn.player("legacy:c", "Cal Cole")],
        "clubs": [syn.club("CAR", "Carlton"), syn.club("ESS", "Essendon")],
        "player_games": games,
    }
    manifest = syn.build(root, tables)
    return root, manifest


def _q(snap):  # type: ignore[no-untyped-def]
    root, manifest = snap
    return SnapshotQuery(root, manifest)


class TestBundle:
    def test_career_values_use_observed_denominators(self, snap) -> None:  # type: ignore[no-untyped-def]
        with _q(snap) as q:
            bundle = players.player_stats_bundle(q, ERAS, stats=("disposals", "goals", "tackles"))
        vals = {v.stat: v for v in players.career_stat_values(bundle, "legacy:a")}
        d = vals["disposals"]
        assert d.total == 60 and d.observed_games == 3 and d.mean == 20.0
        # canonical career games = max(4 rows, counter 5) = 5
        assert d.coverage == pytest.approx(3 / 5)
        t = vals["tackles"]
        assert t.total == 5 and t.observed_games == 2 and t.eligible_games == 2
        assert t.mean == 2.5  # 0 tackles is an observed zero
        games = bundle.games.set_index("player_id").loc["legacy:a"]
        assert games["row_games"] == 4 and games["counter_max"] == 5 and games["career_games"] == 5

    def test_no_observed_games_gives_null_total(self, snap) -> None:  # type: ignore[no-untyped-def]
        with _q(snap) as q:
            bundle = players.player_stats_bundle(q, ERAS, stats=("disposals", "tackles"))
        vals = {v.stat: v for v in players.career_stat_values(bundle, "legacy:b")}
        assert vals["disposals"].total is None and vals["disposals"].mean is None
        assert vals["disposals"].observed_games == 0
        assert vals["tackles"].total is None

    def test_season_lines(self, snap) -> None:  # type: ignore[no-untyped-def]
        with _q(snap) as q:
            bundle = players.player_stats_bundle(q, ERAS, stats=("disposals",))
        lines = players.season_lines(bundle, "legacy:a", lambda pid, s: f"players/{s}.json")
        assert [ln.season for ln in lines] == [1985, 1986, 1987, 1988]
        assert lines[1].stats[0].total is None and lines[1].games == 1
        assert lines[3].clubs == ["ESS"]
        assert lines[0].games_resource == "players/1985.json"


class TestLeaders:
    def test_career_total_leaders_disclose_coverage(self, snap) -> None:  # type: ignore[no-untyped-def]
        with _q(snap) as q:
            table = players.career_leaders(q, "disposals", eras=ERAS, n=10)
        assert [r.player_id for r in table.rows] == ["legacy:c", "legacy:a"]  # b has no observed
        a = table.rows[1]
        assert a.value == 60 and a.observed_games == 3 and a.coverage == pytest.approx(0.6)
        assert a.name == "Alan Able" and a.clubs == ["Carlton", "Essendon"]
        assert "observed" in table.coverage_note
        assert table.scope == "career"

    def test_career_mean_leaders_require_minimum_observed(self, snap) -> None:  # type: ignore[no-untyped-def]
        with _q(snap) as q:
            table = players.career_leaders(q, "disposals", eras=ERAS, basis="mean", min_observed_games=3)
        assert [r.player_id for r in table.rows] == ["legacy:a"]
        assert table.rows[0].value == 20.0

    def test_games_leaders_show_counter_and_rows(self, snap) -> None:  # type: ignore[no-untyped-def]
        with _q(snap) as q:
            table = players.games_leaders(q, n=5)
        top = table.rows[0]
        assert top.player_id == "legacy:a" and top.value == 5 and top.observed_games == 4
        assert "counter" in table.method

    def test_single_season_leaders(self, snap) -> None:  # type: ignore[no-untyped-def]
        with _q(snap) as q:
            table = players.single_season_leaders(q, "disposals", eras=ERAS, n=3)
        assert table.rows[0].player_id == "legacy:c" and table.rows[0].value == 75
        assert table.rows[0].seasons == "1988"
        assert table.scope == "single_season"

    def test_season_leaders_by_club(self, snap) -> None:  # type: ignore[no-untyped-def]
        with _q(snap) as q:
            rows = players.season_leaders(q, "goals", 1987, n=5)
            car = players.season_leaders(q, "goals", 1988, n=5, club_id="CAR")
        assert [(r.player_id, r.value) for r in rows] == [("legacy:b", 3.0), ("legacy:a", 2.0)]
        assert [r.player_id for r in car] == ["legacy:c"]

    def test_leader_ties_break_by_player_id(self, tmp_path: Path) -> None:
        tables = {
            "player_games": [
                syn.pg("m1", "legacy:z", "X", 2020, 1, goals=2),
                syn.pg("m1", "legacy:y", "X", 2020, 1, goals=2),
            ]
        }
        manifest = syn.build(tmp_path, tables)
        with SnapshotQuery(tmp_path, manifest) as q:
            table = players.career_leaders(q, "goals", eras=ERAS)
        assert [r.player_id for r in table.rows] == ["legacy:y", "legacy:z"]
        assert [r.rank for r in table.rows] == [1, 1]

    def test_unknown_stat_rejected(self, snap) -> None:  # type: ignore[no-untyped-def]
        with _q(snap) as q, pytest.raises(ValueError):
            players.career_leaders(q, "goals; DROP TABLE x", eras=ERAS)
