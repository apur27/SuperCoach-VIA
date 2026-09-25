"""Fixture-based ladder, finals separation, form chronology, team-game aggregates (A05)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from supercoach_via.analytics import teams
from supercoach_via.storage.queries import SnapshotQuery
from tests.scvia.fixtures.analytics import synthetic as syn

D = date


def _season_matches() -> list[dict]:  # type: ignore[type-arg]
    m = syn.match
    return [
        m("r1a", 2025, 1, "A", "B", 100, 50, day=D(2025, 3, 1)),
        m("r1b", 2025, 1, "C", "D", 60, 60, day=D(2025, 3, 1)),  # draw
        m("r2a", 2025, 2, "A", "C", 80, 90, day=D(2025, 3, 8)),
        m("r2b", 2025, 2, "B", "D", 70, 0, day=D(2025, 3, 8)),
        # round 3: A scheduled (future fixture), B v D postponed, C bye
        m("r3a", 2025, 3, "A", "D", None, None, day=D(2025, 3, 15), status="scheduled"),
        m("r3b", 2025, 3, "B", "C", None, None, day=D(2025, 3, 15), status="postponed"),
        # a final must not enter the ladder
        m("qf1", 2025, None, "A", "B", 200, 10, day=D(2025, 9, 1), stage_label="Qualifying Final",
          stage_type="final", stage_order=200),
    ]


@pytest.fixture
def snap(tmp_path: Path):  # type: ignore[no-untyped-def]
    tables = {
        "clubs": [syn.club(c, f"Club {c}") for c in "ABCD"],
        "matches": _season_matches(),
    }
    return tmp_path, syn.build(tmp_path, tables)


class TestLadder:
    def test_ladder_uses_completed_regular_matches_only(self, snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snap
        with SnapshotQuery(root, manifest) as q:
            ladder = teams.ladder(q, 2025)
        rows = {r.club_id: r for r in ladder}
        assert rows["A"].played == 2 and rows["A"].won == 1 and rows["A"].lost == 1
        assert rows["A"].points_for == 180 and rows["A"].points_against == 140  # QF excluded
        assert rows["C"].drawn == 1 and rows["C"].premiership_points == 6
        assert rows["D"].premiership_points == 2
        assert sum(r.played for r in ladder) == 2 * 4  # 4 completed regular matches
        assert [r.position for r in ladder] == [1, 2, 3, 4]
        assert rows["A"].name == "Club A"

    def test_percentage_and_order(self, snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snap
        with SnapshotQuery(root, manifest) as q:
            ladder = teams.ladder(q, 2025)
        assert [r.club_id for r in ladder] == ["C", "A", "B", "D"]
        a = next(r for r in ladder if r.club_id == "A")
        assert a.percentage == pytest.approx(100 * 180 / 140)

    def test_zero_points_against_gives_null_percentage(self, tmp_path: Path) -> None:
        tables = {"matches": [syn.match("x", 2024, 1, "A", "B", 50, 0, day=D(2024, 3, 1))]}
        manifest = syn.build(tmp_path, tables)
        with SnapshotQuery(tmp_path, manifest) as q:
            ladder = teams.ladder(q, 2024)
        a = next(r for r in ladder if r.club_id == "A")
        b = next(r for r in ladder if r.club_id == "B")
        assert a.percentage is None and a.points_against == 0
        assert b.percentage == 0.0
        assert [r.club_id for r in ladder] == ["A", "B"]

    def test_missing_score_is_not_a_zero_score(self, tmp_path: Path) -> None:
        bad = syn.match("x", 2024, 1, "A", "B", 50, None, day=D(2024, 3, 1))
        manifest = syn.build(tmp_path, {"matches": [bad]})
        with SnapshotQuery(tmp_path, manifest) as q:
            ladder = teams.ladder(q, 2024)
        assert all(r.played == 0 for r in ladder)

    def test_finals_displayed_separately(self, snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snap
        with SnapshotQuery(root, manifest) as q:
            finals = teams.finals_matches(q, 2025)
        assert [f.match_id for f in finals] == ["qf1"]
        assert finals[0].stage_type == "final" and finals[0].winner_club_id == "A"


class TestStructureAndPathway:
    def test_season_structure_from_data(self, snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snap
        with SnapshotQuery(root, manifest) as q:
            s = teams.season_structure(q, 2025)
        assert s.regular_rounds == (1, 2, 3)
        assert s.remaining_regular["A"] == 1
        assert s.remaining_regular["C"] == 1  # postponed B v C still owed
        assert s.scheduled_regular["A"] == 3
        assert s.has_future_fixture is True
        assert s.stage_labels[-1] == "Qualifying Final"

    def test_pathway_is_labelled_heuristic_and_fixture_aware(self, snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snap
        with SnapshotQuery(root, manifest) as q:
            paths = teams.finals_pathway(q, 2025, finals_places=2)
        a = paths["A"]
        assert all("heuristic" in h.label.lower() for h in a)
        assert all(h.method for h in a)
        text = " ".join(h.text for h in a)
        assert "1 regular-season" in text
        assert "12 wins" not in text

    def test_pathway_elimination_and_clinch_logic(self) -> None:
        # pts now, remaining games; 2 places
        table = {"A": (16, 0), "B": (12, 0), "C": (4, 1), "D": (0, 1)}
        status = teams.pathway_status(table, finals_places=2)
        assert status["A"] == "clinched" and status["B"] == "clinched"
        assert status["C"] == "eliminated" and status["D"] == "eliminated"
        table = {"A": (8, 1), "B": (8, 1), "C": (8, 1)}
        assert set(teams.pathway_status(table, finals_places=2).values()) == {"in_contention"}

    def test_unknown_remaining_games_is_disclosed(self, tmp_path: Path) -> None:
        tables = {"matches": [syn.match("x", 2025, 1, "A", "B", 50, 40, day=D(2025, 3, 1))]}
        manifest = syn.build(tmp_path, tables)
        with SnapshotQuery(tmp_path, manifest) as q:
            paths = teams.finals_pathway(q, 2025, finals_places=8)
        text = " ".join(h.text for h in paths["A"])
        assert "unknown" in text.lower()

    def test_started_finals_mean_the_regular_season_is_over(self, tmp_path: Path) -> None:
        m = syn.match
        tables = {"matches": [
            m("x", 2025, 1, "A", "B", 50, 40, day=D(2025, 3, 1)),
            m("f", 2025, None, "A", "B", 50, 40, day=D(2025, 9, 1), stage_label="Qualifying Final",
              stage_type="final", stage_order=200),
        ]}
        manifest = syn.build(tmp_path, tables)
        with SnapshotQuery(tmp_path, manifest) as q:
            assert teams.season_structure(q, 2025).finals_started is True
            paths = teams.finals_pathway(q, 2025, finals_places=1)
        text = paths["A"][0].text
        assert "unknown" not in text.lower() and "0 regular-season games" in text

    def test_finals_places_rule(self, snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snap
        with SnapshotQuery(root, manifest) as q:
            places, basis = teams.finals_places(q, 2025)
        assert places == 8 and "rule" in basis


class TestForm:
    def test_same_day_matches_order_by_start_then_stable_id(self, tmp_path: Path) -> None:
        m = syn.match
        tables = {
            "matches": [
                m("zz", 2025, 1, "A", "B", 10, 20, day=D(2025, 3, 1)),
                m("aa", 2025, 1, "A", "C", 30, 20, day=D(2025, 3, 1)),
                m("mm", 2025, 2, "A", "D", 50, 50, day=D(2025, 3, 1), local_start="2025-03-01 19:00"),
                m("old", 2025, 1, "A", "E", 1, 2, day=D(2025, 2, 1)),
            ]
        }
        manifest = syn.build(tmp_path, tables)
        with SnapshotQuery(tmp_path, manifest) as q:
            form = teams.team_form(q, "A", n=3)
        # chronological, last three: same-day ordering = local_start (nulls last), then match_id
        assert [f.match_id for f in form] == ["mm", "aa", "zz"]
        assert [f.result for f in form] == ["D", "W", "L"]
        assert [f.margin for f in form] == [0, 10, -10]


class TestTeamGames:
    @pytest.fixture
    def tg_snap(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        m = syn.match
        tables = {
            "matches": [m("g1", 2024, 1, "A", "B", 60, 40, day=D(2024, 3, 1)),
                        m("g2", 2024, 2, "B", "A", 50, 70, day=D(2024, 3, 8))],
            "player_games": [
                syn.pg("g1", "p1", "A", 2024, 1, disposals=20, kicks=12, handballs=8, tackles=None),
                syn.pg("g1", "p2", "A", 2024, 1, disposals=10, kicks=5, handballs=5, tackles=None),
                syn.pg("g1", "p3", "B", 2024, 1, disposals=15, kicks=10, handballs=5, tackles=3),
                syn.pg("g2", "p1", "A", 2024, 2, disposals=25, kicks=15, handballs=10, tackles=4),
                syn.pg("g2", "p3", "B", 2024, 2, disposals=None, kicks=None, handballs=None, tackles=None),
            ],
        }
        return tmp_path, syn.build(tmp_path, tables)

    def test_team_game_sums_keep_nulls(self, tg_snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest = tg_snap
        with SnapshotQuery(root, manifest) as q:
            tg = teams.team_games(q, stats=("disposals", "tackles"), seasons=[2024])
        a1 = tg[(tg.match_id == "g1") & (tg.club_id == "A")].iloc[0]
        assert a1.disposals == 30 and pd.isna(a1.tackles) and a1.tackles_observed == 0
        assert a1.player_rows == 2 and a1.points_for == 60 and a1.result == "W"
        b2 = tg[(tg.match_id == "g2") & (tg.club_id == "B")].iloc[0]
        assert pd.isna(b2.disposals)

    def test_conceded_is_opponent_team_game(self, tg_snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest = tg_snap
        with SnapshotQuery(root, manifest) as q:
            tg = teams.team_games(q, stats=("disposals", "tackles"), seasons=[2024])
        con = teams.conceded_stats(tg, stats=("disposals", "tackles"))
        a1 = con[(con.match_id == "g1") & (con.club_id == "A")].iloc[0]
        assert a1.disposals_conceded == 15 and a1.tackles_conceded == 3
        assert a1.opponent_player_rows == 1

    def test_team_season_stat_values(self, tg_snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest = tg_snap
        with SnapshotQuery(root, manifest) as q:
            tg = teams.team_games(q, stats=("disposals", "tackles"), seasons=[2024])
        vals = {v.stat: v for v in teams.team_season_stats(tg, "A", 2024, stats=("disposals", "tackles"))}
        assert vals["disposals"].mean == 27.5 and vals["disposals"].observed_games == 2
        assert vals["tackles"].mean == 4 and vals["tackles"].observed_games == 1
        assert vals["tackles"].coverage == 0.5

    def test_club_season_leaders(self, tg_snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest = tg_snap
        with SnapshotQuery(root, manifest) as q:
            leaders = teams.club_season_leaders(q, "A", 2024, stats=("disposals",))
        assert leaders[0].player_id == "p1" and leaders[0].value == 45


class TestConcededRotation:
    COLS: ClassVar[list[str]] = ["year", "round", "team", "opponent", "disposals_conceded", "kicks_conceded",
            "handballs_conceded", "marks_conceded", "goals_conceded", "behinds_conceded",
            "tackles_conceded", "hitouts_conceded", "inside_50s_conceded", "clearances_conceded"]

    def _row(self, disp: int, kicks: int, hb: int, marks: int) -> list[object]:
        return [2025, 1, "A", "B", disp, kicks, hb, marks, 10, 5, 40, 30, 50, 35]

    def test_rotation_repair_is_idempotent(self) -> None:
        # true: disposals 300 = kicks 200 + handballs 100, marks 80
        corrupt = pd.DataFrame([self._row(100, 200, 80, 300)], columns=self.COLS)
        fixed = teams.fix_conceded_rotation(corrupt)
        assert fixed.iloc[0][["disposals_conceded", "handballs_conceded", "marks_conceded"]].tolist() == [300, 100, 80]
        assert teams.fix_conceded_rotation(fixed).equals(fixed)

    def test_rotation_repair_refuses_unexplained_rows(self) -> None:
        with pytest.raises(ValueError):
            teams.fix_conceded_rotation(pd.DataFrame([self._row(1, 2, 3, 4)], columns=self.COLS))


class TestFiveYear:
    def test_profile_means_over_season_means(self, tmp_path: Path) -> None:
        m = syn.match
        matches, games = [], []
        for season in (2021, 2022):
            matches.append(m(f"x{season}", season, 1, "A", "B", 10, 5, day=D(season, 3, 1)))
            matches.append(m(f"y{season}", season, 2, "A", "B", 10, 5, day=D(season, 3, 8)))
            for mid, disp in ((f"x{season}", 100 if season == 2021 else 200), (f"y{season}", 300)):
                games.append(syn.pg(mid, "p", "A", season, 1, disposals=disp, handballs=disp // 2))
                games.append(syn.pg(mid, "q", "B", season, 1, disposals=50, handballs=10))
        manifest = syn.build(tmp_path, {"matches": matches, "player_games": games})
        with SnapshotQuery(tmp_path, manifest) as q:
            prof = teams.five_year_profile(q, end_season=2023, years=2)
        a = prof.set_index("club_id").loc["A"]
        assert a["disposals"] == pytest.approx(((100 + 300) / 2 + (200 + 300) / 2) / 2)
        assert a["seasons"] == 2
        assert a["handball_ratio"] == pytest.approx(0.5)
        assert a["disposals_rank"] == 1
        assert list(prof.attrs["window"]) == [2021, 2022]
