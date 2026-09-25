"""Era summaries and adjacent-era Welch tests: parity with the legacy definitions."""

from __future__ import annotations

import random
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from supercoach_via.analytics import eras
from supercoach_via.domain.metrics import CoverageEras
from supercoach_via.storage.queries import SnapshotQuery
from tests.scvia.fixtures.analytics import synthetic as syn

REPO = Path(__file__).resolve().parents[3]
CANON_TO_LEGACY = {"hitouts": "hit_outs", "frees_for": "free_kicks_for", "frees_against": "free_kicks_against",
                   "goal_assists": "goal_assist", "time_on_ground_pct": "percentage_of_game_played"}


@pytest.fixture
def legacy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    """Legacy era module with every output path redirected into tmp_path (never the repo)."""
    import supercoach.era_based_statistical_analysis as era

    for name in ("OUT_ERA_STATS", "OUT_TEAM_SCORING", "OUT_SIG_TESTS", "OUT_YEARLY", "OUT_SUMMARY_JSON"):
        monkeypatch.setattr(era, name, str(tmp_path / f"{name}.out"))
    return era


@pytest.fixture
def corpus(tmp_path: Path):  # type: ignore[no-untyped-def]
    rng = random.Random(11)
    rows, matches = [], []
    for season in (1960, 1961, 1975, 1988, 1995, 2012, 2026):
        for g in range(20):
            mid = f"m{season}_{g}"
            hs, as_ = rng.randint(30, 150), rng.randint(30, 150)
            matches.append(syn.match(mid, season, g + 1, "A", "B", hs, as_, day=date(season, 4, 1)))
            for p in range(3):
                def v(lo: int = 0, hi: int = 30, start: int = 1965, s: int = season) -> int | None:
                    return None if s < start or rng.random() < 0.05 else rng.randint(lo, hi)

                tog = None if season < 2003 else float(rng.choice([10, 60, 80, 100]))
                rows.append(syn.pg(mid, f"p{p}", "AB"[p % 2], season, g + 1,
                                   kicks=v(), handballs=v(), disposals=v(), goals=rng.randint(0, 5),
                                   tackles=v(start=1987), time_on_ground_pct=tog))
    root = tmp_path / "snap"
    manifest = syn.build(root, {"player_games": rows, "matches": matches})
    legacy_games = pd.DataFrame(
        [{"year": r["season"], **{CANON_TO_LEGACY.get(k, k): r.get(k) for k in eras.PLAYER_METRICS}} for r in rows]
    )
    legacy_matches = pd.DataFrame(
        [{"year": m["season"], "team_1_team_name": m["home_club_id"], "team_2_team_name": m["away_club_id"],
          "team_1_final_goals": m["home_final_goals"], "team_1_final_behinds": m["home_final_behinds"],
          "team_2_final_goals": m["away_final_goals"], "team_2_final_behinds": m["away_final_behinds"]}
         for m in matches]
    )
    return root, manifest, legacy_games, legacy_matches


def _prep_legacy_games(era, games: pd.DataFrame) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    g = games.copy()
    for m in era.PLAYER_METRICS:
        g[m] = pd.to_numeric(g[m], errors="coerce") if m in g else np.nan
    g["era"] = g["year"].map(era.assign_era)
    return g


def _close(a: pd.Series, b: pd.Series) -> bool:
    return bool(np.allclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True))


class TestEraParity:
    def test_era_stats_match_legacy(self, legacy, corpus) -> None:  # type: ignore[no-untyped-def]
        root, manifest, games, _ = corpus
        want = legacy.summarise_per_era(_prep_legacy_games(legacy, games))
        with SnapshotQuery(root, manifest) as q:
            got = eras.era_stats(q, eras=CoverageEras({"tackles": 1987}))
        got = got.set_index(["era", "legacy_metric"]).loc[list(zip(want.era, want.metric, strict=True))]
        for col in ("n_player_games", "n_with_metric", "mean_per_game", "std_per_game",
                    "median_per_game", "mean_per_100pct_played"):
            assert _close(got[col].reset_index(drop=True), want[col]), col

    def test_significance_tests_match_legacy(self, legacy, corpus) -> None:  # type: ignore[no-untyped-def]
        root, manifest, games, _ = corpus
        want = legacy.adjacent_era_tests(_prep_legacy_games(legacy, games))
        with SnapshotQuery(root, manifest) as q:
            got = eras.adjacent_era_tests(q, eras=CoverageEras({"tackles": 1987}))
        got = got.set_index(["era_a", "legacy_metric"]).loc[list(zip(want.era_a, want.metric, strict=True))]
        got = got.reset_index(drop=True)
        for col in ("n_a", "n_b", "mean_a", "mean_b", "delta", "cohens_d", "welch_t", "p_value"):
            assert _close(got[col], want[col]), col
        assert (got["note"].fillna("") == want["note"].fillna("")).all()

    def test_recording_boundary_is_flagged(self, corpus) -> None:  # type: ignore[no-untyped-def]
        root, manifest, _, _ = corpus
        with SnapshotQuery(root, manifest) as q:
            tests = eras.adjacent_era_tests(q, eras=CoverageEras({"tackles": 1987}))
            stats = eras.era_stats(q, eras=CoverageEras({"tackles": 1987}))
        row = tests[(tests.metric == "tackles") & (tests.era_a == "1965-1990")].iloc[0]
        assert "recording" in row["boundary_note"]
        t = stats[(stats.metric == "tackles")].set_index("era")
        assert t.loc["1965-1990", "recording_status"] == "partial_era"
        assert t.loc["1991-2010", "recording_status"] == "recorded"
        assert t.loc["pre-1965", "recording_status"] == "not_recorded"

    def test_yearly_trends_match_legacy(self, legacy, corpus) -> None:  # type: ignore[no-untyped-def]
        root, manifest, games, _ = corpus
        want = legacy.yearly_trends(_prep_legacy_games(legacy, games))
        with SnapshotQuery(root, manifest) as q:
            got = eras.yearly_trends(q)
        assert got["year"].tolist() == want["year"].tolist()
        for m in legacy.PLAYER_METRICS:
            assert _close(got[m], want[m]), m
        assert got["n_player_games"].tolist() == want["n_player_games"].tolist()

    def test_team_scoring_matches_legacy(self, legacy, corpus) -> None:  # type: ignore[no-untyped-def]
        root, manifest, _, matches = corpus
        want = legacy.summarise_match_scoring(matches)
        with SnapshotQuery(root, manifest) as q:
            got = eras.team_scoring(q)
        want = want.dropna(subset=["n_team_games"]).reset_index(drop=True)
        got = got.set_index("era").loc[want.era].reset_index()
        for col in want.columns.drop("era"):
            assert _close(got[col], want[col]), col

    def test_era_bounds_extend_to_latest_season(self, corpus) -> None:  # type: ignore[no-untyped-def]
        root, manifest, _, _ = corpus
        with SnapshotQuery(root, manifest) as q:
            bounds = eras.era_bounds(q)
        assert bounds[-1] == ("2011-present", 2011, 2026)
