"""Real-corpus analytics parity against the legacy code on the same input bytes.

Snapshot: ``SCVIA_SNAPSHOT_ROOT`` (default ``var/agent-import``) and ``SCVIA_SNAPSHOT``
(default ``current``; may be a ``sha256:`` id). Legacy functions run IN-PROCESS: they only
read ``data/`` and every legacy output path is redirected into ``tmp_path``; nothing is
written to the repository.

Intentional input difference: the canonical snapshot quarantines the duplicate legacy files
``green_william_08092005`` and ``steele_roan_19092002`` (docs/rewrite/DATA_REFRESH.md), so the
legacy scan here excludes them too. Any other difference is a defect to explain.
"""

from __future__ import annotations

import glob as _glob
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from supercoach_via.analytics import awards, eras, players, rankings, teams
from supercoach_via.domain.metrics import CoverageEras
from supercoach_via.domain.schemas import SnapshotManifest
from supercoach_via.storage.queries import SnapshotQuery
from supercoach_via.storage.snapshots import load_snapshot

REPO = Path(__file__).resolve().parents[3]
QUARANTINED_DUPLICATES = ("green_william_08092005", "steele_roan_19092002")
SCORE_TOL = 1e-9


@pytest.fixture(scope="module")
def snap() -> tuple[Path, SnapshotManifest]:
    root = Path(os.environ.get("SCVIA_SNAPSHOT_ROOT", REPO / "var" / "agent-import"))
    selector = os.environ.get("SCVIA_SNAPSHOT", "current")
    return root, load_snapshot(root, selector)


@pytest.fixture(scope="module")
def eras_cfg() -> CoverageEras:
    return CoverageEras.load(REPO / "config" / "stat_coverage_eras.yaml")


def _dedup_sorted_glob(pattern: str) -> list[str]:
    return sorted(p for p in _glob.glob(pattern) if not any(d in p for d in QUARANTINED_DUPLICATES))


@pytest.fixture(scope="module")
def legacy_ranking(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    import top_players_comprehensive as tpc

    out = tmp_path_factory.mktemp("legacy_rank")
    (out / "yearly").mkdir()
    orig = tpc.glob
    tpc.glob = SimpleNamespace(glob=_dedup_sorted_glob)  # type: ignore[assignment]
    try:
        by_year, career = tpc._aggregate_all_players(str(REPO / "data" / "player_data"))
    finally:
        tpc.glob = orig
    yearly = {}
    for season in sorted(by_year):
        top = tpc._generate_yearly_from_memory(season, by_year[season], tpc.WEIGHTS, str(out))
        if top:
            yearly[season] = top
    tpc.compile_all_time_top_100(yearly, str(out), career_games=career)
    return {"all_time": pd.read_csv(out / "all_time_top_100.csv"), "yearly": yearly, "career": career}


class TestRankingParity:
    def test_all_time_top_100_identical_order_and_scores(self, snap, legacy_ranking) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snap
        with SnapshotQuery(root, manifest) as q:
            result = rankings.run_legacy_v1(q)
        got = rankings.numeric_export_rows(result)
        want = legacy_ranking["all_time"]
        assert [p for p, _ in got] == want["player"].tolist()
        assert max(abs(s - w) for (_, s), w in zip(got, want["all_time_score"], strict=True)) < SCORE_TOL

    def test_yearly_lists_identical(self, snap, legacy_ranking) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snap
        with SnapshotQuery(root, manifest) as q:
            result = rankings.run_legacy_v1(q)
        differing = []
        for season, want in legacy_ranking["yearly"].items():  # type: ignore[attr-defined]
            got = result.yearly[season].entries
            if [(e.player_key, e.score, e.games) for e in got] != [(w[0], w[1], w[3]) for w in want]:
                differing.append(season)
        assert differing == []

    def test_current_season_is_provisional(self, snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snap
        with SnapshotQuery(root, manifest) as q:
            result = rankings.run_legacy_v1(q)
            latest = q.scalar("SELECT MAX(season) FROM player_games")
            gf = q.scalar(
                "SELECT COUNT(*) FROM matches WHERE season = ? AND status = 'complete' AND stage_id = 'gf'",
                [latest],
            )
        if not gf:
            assert result.yearly[latest].provisional


class TestLadder:
    @pytest.mark.parametrize("season", [2025, 2026])
    def test_played_equals_completed_regular_matches(self, snap, season: int) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snap
        with SnapshotQuery(root, manifest) as q:
            ladder = teams.ladder(q, season)
            counts = dict(
                q.rows(
                    """SELECT c, COUNT(*) FROM (
                         SELECT home_club_id c FROM matches WHERE season = ? AND stage_type = 'regular'
                           AND status = 'complete' AND home_score IS NOT NULL AND away_score IS NOT NULL
                         UNION ALL SELECT away_club_id FROM matches WHERE season = ? AND stage_type = 'regular'
                           AND status = 'complete' AND home_score IS NOT NULL AND away_score IS NOT NULL)
                       GROUP BY c""",
                    [season, season],
                )
            )
        legacy = pd.read_csv(REPO / "data" / "matches" / f"matches_{season}.csv")
        regular = legacy[legacy["round_num"].astype(str).str.fullmatch(r"\d+")]
        assert len(ladder) == 18
        assert {r.club_id: r.played for r in ladder} == counts
        assert sum(r.played for r in ladder) == 2 * len(regular)
        assert sum(r.won for r in ladder) == sum(r.lost for r in ladder)
        assert sum(r.premiership_points for r in ladder) == 4 * sum(r.won for r in ladder) + 2 * sum(
            r.drawn for r in ladder
        )
        assert all(r.played == 23 for r in ladder)

    def test_finals_separate_and_wildcard_2026(self, snap) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snap
        with SnapshotQuery(root, manifest) as q:
            finals = teams.finals_matches(q, 2026)
            places, basis = teams.finals_places(q, 2026)
            paths = teams.finals_pathway(q, 2026)
        assert finals and all(f.stage_type == "final" for f in finals)
        assert places == 10 and "Wildcard" in basis
        assert all("heuristic" in h.label for hs in paths.values() for h in hs)


class TestBrownlowParity:
    def test_proxy_matches_legacy_for_2026(self, snap, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
        import update_team_analysis as uta

        monkeypatch.setattr(uta, "glob", SimpleNamespace(glob=_dedup_sorted_glob))
        want = uta._build_brownlow_proxy_table(uta._load_player_games_with_names(2026), min_games=3)
        root, manifest = snap
        with SnapshotQuery(root, manifest) as q:
            got = awards.brownlow_proxy(q, 2026)
        assert [r.player_id.removeprefix("legacy:") for r in got.rows] == want["player_stem"].tolist()
        assert max(
            abs(r.proxy_per_game - w) for r, w in zip(got.rows, want["brownlow_proxy_pg"], strict=True)
        ) < SCORE_TOL
        assert got.season_games == 23 and got.scale_basis.startswith("fixture")
        assert [r.player_id for r in got.rows if r.ineligible] == ["legacy:xerri_tristan_15031999"]


class TestErasParity:
    def test_era_stats_match_legacy(self, snap, eras_cfg, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
        import supercoach.era_based_statistical_analysis as era

        for name in ("OUT_ERA_STATS", "OUT_TEAM_SCORING", "OUT_SIG_TESTS", "OUT_YEARLY", "OUT_SUMMARY_JSON"):
            monkeypatch.setattr(era, name, str(tmp_path / f"{name}.csv"))
        monkeypatch.setattr(era, "glob", SimpleNamespace(glob=_dedup_sorted_glob))
        want = era.summarise_per_era(era.load_player_games())
        root, manifest = snap
        with SnapshotQuery(root, manifest) as q:
            got = eras.era_stats(q, eras=eras_cfg)
        got = got.set_index(["era", "legacy_metric"]).loc[list(zip(want.era, want.metric, strict=True))]
        got = got.reset_index(drop=True)
        assert (got["n_player_games"] == want["n_player_games"]).all()
        assert (got["n_with_metric"] == want["n_with_metric"]).all()
        for col in ("mean_per_game", "std_per_game", "median_per_game", "mean_per_100pct_played"):
            a, b = got[col].astype(float), want[col].astype(float)
            assert ((a - b).abs() <= 1e-9 * b.abs().clip(lower=1) | (a.isna() & b.isna())).all(), col


class TestPerformance:
    def test_whole_corpus_player_analytics_is_fast(self, snap, eras_cfg) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snap
        t0 = time.perf_counter()
        with SnapshotQuery(root, manifest) as q:
            bundle = players.player_stats_bundle(q, eras_cfg)
            t_bundle = time.perf_counter() - t0
            for stat in ("disposals", "goals", "tackles", "brownlow_votes"):
                players.career_leaders(q, stat, eras=eras_cfg)
                players.single_season_leaders(q, stat, eras=eras_cfg)
            players.games_leaders(q)
        total = time.perf_counter() - t0
        print(f"bundle {t_bundle:.2f}s, bundle+leaders {total:.2f}s, players {len(bundle.games)}")
        assert len(bundle.games) > 13000
        assert total < 20.0

    def test_counter_vs_rows_disclosed(self, snap, eras_cfg) -> None:  # type: ignore[no-untyped-def]
        root, manifest = snap
        with SnapshotQuery(root, manifest) as q:
            table = players.games_leaders(q, n=20000)
        assert any(r.observed_games != r.value for r in table.rows)
        assert all(r.value >= (r.observed_games or 0) for r in table.rows)
