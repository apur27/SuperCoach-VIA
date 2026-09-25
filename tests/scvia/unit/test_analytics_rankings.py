"""legacy_v1 ranking parity (A01) and top-100 export distinction (A02)."""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path

import pytest

from supercoach_via.analytics import rankings
from supercoach_via.storage.queries import SnapshotQuery
from tests.scvia.fixtures.analytics import synthetic as syn

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def legacy(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """The legacy module with its file scan made deterministic (sorted glob).

    Legacy leaves exact score ties in filesystem glob order; legacy_v1 breaks them by
    player_id, which equals sorted slug order, so a sorted scan is the comparable input.
    """
    import glob as _glob
    from types import SimpleNamespace

    import top_players_comprehensive as tpc

    monkeypatch.setattr(tpc, "glob", SimpleNamespace(glob=lambda p: sorted(_glob.glob(p))))
    return tpc


@pytest.fixture(scope="module")
def corpus() -> list[dict]:  # type: ignore[type-arg]
    return syn.ranking_corpus()


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory, corpus: list[dict]):  # type: ignore[no-untyped-def,type-arg]
    root = tmp_path_factory.mktemp("rank_snap")
    manifest = syn.build(root, syn.canonical_ranking_tables(corpus))
    return root, manifest


class TestConfig:
    def test_config_matches_executable_legacy_constants(self, legacy) -> None:  # type: ignore[no-untyped-def]
        cfg = rankings.RankingConfig.load()
        assert cfg.formula_version == "legacy_v1"
        assert cfg.z_cap == legacy.Z_CAP
        assert cfg.top_n_seasons == legacy.TOP_N_SEASONS == 11
        assert cfg.rank_gamma == legacy.RANK_GAMMA == 0.37
        assert cfg.z_blend == legacy.Z_BLEND
        assert cfg.single_stat_cap == legacy.SINGLE_STAT_CAP
        assert cfg.min_position_group == legacy.MIN_POSITION_GROUP
        assert cfg.active_player_discount == legacy.ACTIVE_PLAYER_DISCOUNT
        assert set(cfg.recent_years) == legacy.RECENT_YEARS
        assert dict(cfg.era_completeness) == legacy.ERA_COMPLETENESS
        rename = {"goal_assists": "goal_assist"}
        assert {rename.get(k, k): v for k, v in cfg.weight_map.items()} == legacy.WEIGHTS
        legacy_eras = {
            e.name: (e.start, e.end, [rename.get(s, s) for s in e.stats]) for e in cfg.eras
        }
        assert legacy_eras == {k: (a, b, list(s)) for k, (a, b, s) in legacy.ERAS.items()}

    def test_config_hash_is_stable_and_sensitive(self) -> None:
        cfg = rankings.RankingConfig.load()
        assert cfg.config_hash() == rankings.RankingConfig.load().config_hash()
        assert len(cfg.config_hash()) == 64
        assert replace(cfg, rank_gamma=0.2).config_hash() != cfg.config_hash()

    def test_era_lookup(self) -> None:
        cfg = rankings.RankingConfig.load()
        assert cfg.era_for(1964).name == "pre_1965"
        assert cfg.era_for(1991).name == "1990_2010"
        assert cfg.era_for(1890) is None


class TestYearlyParity:
    def test_rank_season_equals_legacy_generate_yearly(self, legacy, tmp_path: Path, snapshot) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snapshot
        cfg = rankings.RankingConfig.load()
        with SnapshotQuery(root, manifest) as q:
            inputs = rankings.load_ranking_inputs(q, cfg)
        legacy_dir = tmp_path / "pd"
        syn.write_legacy_csvs(legacy_dir, syn.ranking_corpus())
        (tmp_path / "out" / "yearly").mkdir(parents=True)
        by_year, _ = legacy._aggregate_all_players(str(legacy_dir))
        for season in sorted(by_year):
            want = legacy._generate_yearly_from_memory(
                season, by_year[season], legacy.WEIGHTS, str(tmp_path / "out")
            )
            got = rankings.rank_season(season, inputs.by_season.get(season, []), cfg)
            assert [e.player_key for e in got] == [w[0] for w in want]
            for e, w in zip(got, want, strict=True):
                assert e.score == w[1]
                assert e.games == w[3]
                assert e.percentile_rank == pytest.approx(w[4], abs=1e-9)
                assert e.z_adj == pytest.approx(w[5], abs=1e-9)

    def test_legacy_zero_fill_is_applied_to_blank_cells(self) -> None:
        cfg = rankings.RankingConfig.load()
        rows = [
            rankings.SeasonTotals("a", "a", 2015, {"goals": 2.0, "kicks": 0.0}, 3),
            rankings.SeasonTotals("b", "b", 2015, {"goals": 1.0}, 3),
        ]
        got = rankings.rank_season(2015, rows, cfg)
        assert [e.player_id for e in got] == ["a", "b"]

    def test_non_positive_scores_excluded_and_ties_break_by_games_then_id(self) -> None:
        cfg = rankings.RankingConfig.load()
        rows = [
            rankings.SeasonTotals("z", "z", 1960, {"goals": 1.0}, 2),
            rankings.SeasonTotals("a", "a", 1960, {"goals": 1.0}, 2),
            rankings.SeasonTotals("b", "b", 1960, {"goals": 1.0}, 5),
            rankings.SeasonTotals("n", "n", 1960, {"goals": 0.0, "behinds": 0.0}, 9),
        ]
        got = rankings.rank_season(1960, rows, cfg)
        assert [e.player_id for e in got] == ["b", "a", "z"]
        assert [e.rank for e in got] == [1, 2, 3]

    def test_single_stat_cap(self) -> None:
        cfg = rankings.RankingConfig.load()
        # goals 10*55=550, behinds 10*1.5=15 -> uncapped 565, cap 0.55*565=310.75
        rows = [rankings.SeasonTotals("a", "a", 1960, {"goals": 10.0, "behinds": 10.0}, 1)]
        (e,) = rankings.rank_season(1960, rows, cfg)
        assert e.score == int(565 - (550 - 0.55 * 565))


class TestAllTimeParity:
    def test_run_legacy_v1_equals_legacy_pipeline(self, legacy, tmp_path: Path, snapshot) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snapshot
        legacy_dir = tmp_path / "pd"
        syn.write_legacy_csvs(legacy_dir, syn.ranking_corpus())
        out = tmp_path / "out"
        (out / "yearly").mkdir(parents=True)
        by_year, career = legacy._aggregate_all_players(str(legacy_dir))
        yearly = {}
        for season in sorted(by_year):
            top = legacy._generate_yearly_from_memory(season, by_year[season], legacy.WEIGHTS, str(out))
            if top:
                yearly[season] = top
        legacy.compile_all_time_top_100(yearly, str(out), career_games=career)
        with (out / "all_time_top_100.csv").open() as fh:
            want = [(r["player"], float(r["all_time_score"])) for r in csv.DictReader(fh)]
        assert want, "fixture must produce an all-time list"

        with SnapshotQuery(root, manifest) as q:
            result = rankings.run_legacy_v1(q)
        got = rankings.numeric_export_rows(result)
        assert [p for p, _ in got] == [p for p, _ in want]
        assert max(abs(a - b) for (_, a), (_, b) in zip(got, want, strict=True)) < 1e-9
        assert result.formula_version == "legacy_v1"
        assert result.config_hash == rankings.RankingConfig.load().config_hash()
        assert result.snapshot_id == manifest.snapshot_id
        assert result.career_games == {f"legacy:{k}": v for k, v in career.items()}

    def test_eligibility_uses_canonical_games_counter(self) -> None:
        cfg = rankings.RankingConfig.load()
        e = rankings.YearlyEntry(1960, 1, "a", "a", 100, 1, 50.0, 0.0)
        assert rankings.compile_all_time({1960: [e]}, {"a": 149}, cfg) == []
        (row,) = rankings.compile_all_time({1960: [e]}, {"a": 150}, cfg)
        rank_score = 1.0
        z_signal = 0.5
        year_score = 0.8 * rank_score + 0.2 * z_signal
        adj = year_score * 0.84
        assert row.all_time_score == pytest.approx(adj * (1 + 2.0 * (1 / 18)))
        assert not row.active

    def test_active_discount(self) -> None:
        cfg = rankings.RankingConfig.load()
        e = rankings.YearlyEntry(2025, 1, "a", "a", 100, 1, 50.0, 0.0)
        (row,) = rankings.compile_all_time({2025: [e]}, {"a": 200}, cfg)
        assert row.active
        assert row.all_time_score == pytest.approx(0.9 * 0.89 * (1 + 2.0 / 18) * 0.95)


class TestLifecycle:
    def test_in_progress_season_is_provisional(self, tmp_path: Path) -> None:
        tables = syn.canonical_ranking_tables(syn.ranking_corpus(n_players=20))
        tables["seasons"] = [
            {"season": s, "matches_complete": 1, "matches_scheduled": 1, "schedule_complete": True}
            for s in (1960, 1961, 1980, 2005, 2015, 2025)
        ] + [{"season": 2026, "matches_complete": 5, "matches_scheduled": 9, "schedule_complete": False}]
        manifest = syn.build(tmp_path, tables)
        with SnapshotQuery(tmp_path, manifest) as q:
            result = rankings.run_legacy_v1(q)
        assert result.yearly[2026].provisional is True
        assert result.yearly[2026].lifecycle == "provisional"
        assert result.yearly[2025].provisional is False
        assert result.yearly[2025].lifecycle == "final"

    def test_unknown_season_status_defaults_to_provisional_only_for_latest(self, snapshot) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snapshot
        with SnapshotQuery(root, manifest) as q:
            result = rankings.run_legacy_v1(q)
        assert result.yearly[2026].provisional is True
        assert result.yearly[2025].provisional is False

    def test_grand_final_completes_a_season_without_season_metadata(self, tmp_path: Path) -> None:
        tables = syn.canonical_ranking_tables(syn.ranking_corpus(n_players=20))
        tables["matches"] = [
            syn.match("gf", 2026, None, "C1", "C2", 80, 70, stage_label="Grand Final", stage_type="final")
        ]
        manifest = syn.build(tmp_path, tables)
        with SnapshotQuery(tmp_path, manifest) as q:
            result = rankings.run_legacy_v1(q)
        assert result.yearly[2026].provisional is False


class TestExports:
    def test_numeric_and_biography_exports_are_distinct_shapes(self, snapshot) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snapshot
        with SnapshotQuery(root, manifest) as q:
            result = rankings.run_legacy_v1(q)
            bio = rankings.biography_export_rows(q, result)
        assert rankings.NUMERIC_EXPORT_COLUMNS == ("player", "all_time_score")
        assert rankings.BIOGRAPHY_EXPORT_COLUMNS == (
            "Serial Number", "Player Name", "Footy Teams", "Comment"
        )
        numeric = rankings.numeric_export_rows(result)
        assert all(isinstance(s, float) and math.isfinite(s) for _, s in numeric)
        assert len(bio) == len(numeric)
        assert [r[0] for r in bio] == list(range(1, len(numeric) + 1))
        assert all(isinstance(r[1], str) for r in bio)

    def test_root_biography_file_is_not_a_numeric_ranking(self) -> None:
        with (REPO / "all_time_top_100.csv").open() as fh:
            header = next(csv.reader(fh))
        assert tuple(header) == rankings.BIOGRAPHY_EXPORT_COLUMNS
        with (REPO / "data" / "top100" / "all_time_top_100.csv").open() as fh:
            header = next(csv.reader(fh))
        assert tuple(header) == rankings.NUMERIC_EXPORT_COLUMNS

    def test_yearly_export_rows_match_legacy_columns(self, snapshot) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snapshot
        with SnapshotQuery(root, manifest) as q:
            result = rankings.run_legacy_v1(q)
        rows = rankings.yearly_export_rows(result.yearly[2015])
        assert rankings.YEARLY_EXPORT_COLUMNS == ("player", "score", "percentile_rank", "games_played")
        assert rows[0][1] >= rows[-1][1]

    def test_history_table_view_model(self, snapshot) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snapshot
        with SnapshotQuery(root, manifest) as q:
            result = rankings.run_legacy_v1(q)
            table = rankings.all_time_history_table(q, result)
            yearly = rankings.yearly_history_table(q, result, 2026)
        assert table.method_version == "legacy_v1"
        assert result.config_hash[:12] in table.method
        assert table.scope == "ranking"
        assert [r.rank for r in table.rows] == list(range(1, len(table.rows) + 1))
        assert yearly.warning is not None and "provisional" in yearly.warning.lower()
