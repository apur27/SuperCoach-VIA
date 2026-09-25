"""Feature engine contracts (PLAN 7.2; M02, M03 cutoff parts, property tests)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from supercoach_via.ml import features as F
from tests.scvia.fixtures.ml.synthetic import build_corpus, write_corpus


@pytest.fixture(scope="module")
def history(tmp_path_factory: pytest.TempPathFactory) -> F.History:
    root = tmp_path_factory.mktemp("snap")
    write_corpus(root)
    return F.load_history(root)


def _naive(history: F.History, player_id: str, cutoff: datetime, season: int) -> dict[str, float]:
    """Independent reference: filter, sort, compute with plain pandas."""
    pg = history.player_games
    m = history.matches.set_index("match_id")
    ok = pg.date_quality.isin(["fixture_verified", "source"]) | (pg.link_method == "key")
    rows = pg[(pg.player_id == player_id) & ok].copy()
    rows["day"] = rows.match_id.map(m["match_date"])
    rows = rows[rows.match_id.map(m["status"]) == "complete"]
    rows = rows[pd.to_datetime(rows.day) < pd.Timestamp(cutoff.date())]
    rows = rows.sort_values(["day", "match_id"])
    out: dict[str, float] = {"history_games": float(len(rows))}
    for s in F.BASE_STATS:
        v = rows[s].astype(float)
        out[f"{s}_prior5_mean"] = v.tail(5).mean() if v.tail(5).notna().any() else np.nan
        sv = rows[rows.season == season][s].astype(float)
        out[f"{s}_season3_mean"] = sv.tail(3).mean() if sv.tail(3).notna().any() else np.nan
        out[f"{s}_season_mean"] = sv.mean() if sv.notna().any() else np.nan
        e = np.nan
        for x in v:
            if np.isnan(x):
                continue
            e = x if np.isnan(e) else 0.5 * e + 0.5 * x
        out[f"{s}_ewm3"] = e
    out["days_since_last_game"] = (
        float((cutoff.date() - rows.day.iloc[-1]).days) if len(rows) else np.nan
    )
    return out


def _targets_for(history: F.History, season: int, stage: str) -> pd.DataFrame:
    t = F.historical_targets(history, seasons=(season,))
    return t[t.stage_label == stage].reset_index(drop=True)


class TestContract:
    def test_base_stats_are_the_legacy_six(self) -> None:
        # supercoach/prediction.py base_rolling_features (reference, not imported)
        assert F.BASE_STATS == (
            "disposals", "kicks", "handballs", "tackles", "clearances", "inside_50s",
        )

    def test_feature_order_is_fixed_and_includes_indicators(self, history: F.History) -> None:
        t = _targets_for(history, 2025, "4")
        ff = F.build_features(history, t, F.FeatureSpec())
        assert list(ff.X.columns) == list(ff.feature_names)
        assert "disposals_prior5_mean__missing" in ff.feature_names
        assert ff.categorical_features == ("club_id", "opponent_club_id", "venue_id")
        again = F.build_features(history, t.sample(frac=1, random_state=1), F.FeatureSpec())
        assert list(again.X.columns) == list(ff.X.columns)

    def test_spec_fingerprint_changes_with_spec(self) -> None:
        assert F.FeatureSpec().fingerprint() != F.FeatureSpec(window_long=6).fingerprint()
        assert F.FeatureSpec().fingerprint() == F.FeatureSpec().fingerprint()


class TestValuesMatchReference:
    def test_all_targets_of_a_round_match_naive(self, history: F.History) -> None:
        for season, stage in [(2025, "4"), (2024, "2"), (2025, "GF"), (2023, "1")]:
            t = _targets_for(history, season, stage)
            assert len(t)
            ff = F.build_features(history, t, F.FeatureSpec())
            for i, row in t.iterrows():
                ref = _naive(history, row.player_id, row.forecast_cutoff.to_pydatetime(), season)
                for k, v in ref.items():
                    got = ff.X.iloc[i][k]
                    assert (np.isnan(v) and np.isnan(got)) or got == pytest.approx(v, rel=1e-12), (
                        season, stage, row.player_id, k, got, v,
                    )

    def test_inferred_date_with_key_link_uses_match_date_not_row_date(self, history: F.History) -> None:
        # 2023/2024 rows carry a synthesized row date 17 days early but a key link.
        t = _targets_for(history, 2024, "1")
        ff = F.build_features(history, t, F.FeatureSpec())
        pg = history.player_games
        shifted = pg.assign(match_date=pg.match_date.map(lambda d: d - timedelta(days=300)))
        import dataclasses as dc

        ff2 = F.build_features(dc.replace(history, player_games=shifted), t, F.FeatureSpec())
        pd.testing.assert_frame_equal(ff.X, ff2.X)
        assert (ff.history_games > 0).any()
        assert ff.diagnostics["date_disagrees_with_match"] > 0

    def test_inferred_date_rows_without_key_link_are_never_history(self, history: F.History) -> None:
        t = _targets_for(history, 2025, "5")
        ff = F.build_features(history, t, F.FeatureSpec())
        i = t.index[t.player_id == "legacy:bris_p1"]
        assert (ff.history_games[i] == 0).all()
        assert ff.diagnostics["excluded_unverified_date"] > 0

    def test_cold_start_player_has_zero_history_and_missing_indicators(
        self, history: F.History
    ) -> None:
        t = _targets_for(history, 2025, "4")
        ff = F.build_features(history, t, F.FeatureSpec())
        i = int(t.index[t.player_id == "legacy:adel_rookie"][0])
        assert ff.history_games[i] == 0
        assert np.isnan(ff.X.iloc[i]["disposals_prior5_mean"])
        assert ff.X.iloc[i]["disposals_prior5_mean__missing"] == 1.0

    def test_postponed_round_ordered_by_date_not_round_number(self, history: F.History) -> None:
        # round 2 (postponed) is played after round 3: a round-3 target must not see round 2
        t = _targets_for(history, 2025, "3")
        ff = F.build_features(history, t, F.FeatureSpec())
        i = int(t.index[t.player_id == "legacy:adel_p0"][0])
        assert ff.X.iloc[i]["season_games_prior"] == 1.0  # only round 1


class TestLeakage:
    def test_M02_mutating_target_and_future_fields_leaves_features_unchanged(
        self, tmp_path: Path
    ) -> None:
        corpus = build_corpus()
        a = F.load_history(corpus.write(tmp_path / "a"))
        t = _targets_for(a, 2025, "4")
        cutoff_day = t.forecast_cutoff.min().date()
        base = F.build_features(a, t, F.FeatureSpec())
        rng = np.random.default_rng(0)
        for r in corpus.player_games:
            if r["match_date"] >= cutoff_day:
                for s in ("disposals", "kicks", "handballs", "tackles", "clearances",
                          "inside_50s", "time_on_ground_pct", "goals", "marks"):
                    r[s] = None if rng.random() < 0.2 else int(rng.integers(0, 60))
                r["result"] = "W"
                r["career_game_counter"] = 999
        for m in corpus.matches:
            if m["match_date"] >= cutoff_day:
                m["home_score"], m["away_score"] = 1, 200
        b = F.load_history(corpus.write(tmp_path / "b"))
        assert a.snapshot_id != b.snapshot_id
        again = F.build_features(b, t, F.FeatureSpec())
        pd.testing.assert_frame_equal(base.X, again.X)

    def test_M03_same_day_ambiguous_observation_excluded(self, history: F.History) -> None:
        pg = history.player_games
        row = pg[(pg.player_id == "legacy:adel_p0") & (pg.stage_label == "4")
                 & (pg.season == 2025)].iloc[0]
        m = history.matches.set_index("match_id").loc[row.match_id]
        day = m.match_date
        tgt = pd.DataFrame([{
            "player_id": "legacy:adel_p0", "club_id": "adel", "match_id": "future_x",
            "season": 2025, "stage_id": "r05", "stage_label": "5", "stage_type": "regular",
            "stage_order": 5, "match_date": day + timedelta(days=1),
            "opponent_club_id": "carl", "venue_id": None,
        }])
        # cutoff late on the same UTC day as the observation.
        late = tgt.assign(forecast_cutoff=pd.Timestamp(datetime.combine(day, datetime.min.time(), UTC) + timedelta(hours=23)))
        ff = F.build_features(history, late, F.FeatureSpec())
        # adel's match that day is the minute-precision MCG game (13:10 Melbourne): ended
        # well before 23:00Z, so it is verifiably prior and included.
        assert m.local_start is not None
        assert ff.X.iloc[0]["season_games_prior"] == 4.0
        # the same observation made day-precision (unknown start) is ambiguous -> excluded
        h2 = dataclasses.replace(
            history,
            matches=history.matches.assign(
                local_start=lambda d: d.local_start.where(d.match_id != row.match_id, None),
                date_precision=lambda d: d.date_precision.where(d.match_id != row.match_id, "day"),
            ),
        )
        ff2 = F.build_features(h2, late, F.FeatureSpec())
        assert ff2.X.iloc[0]["season_games_prior"] == 3.0
        assert ff2.diagnostics["excluded_same_day_ambiguous"] >= 1

    def test_cutoff_after_target_start_is_rejected(self, history: F.History) -> None:
        t = _targets_for(history, 2025, "4")
        bad = t.assign(forecast_cutoff=t.forecast_cutoff + pd.Timedelta(days=2))
        with pytest.raises(F.TargetCutoffError):
            F.build_features(history, bad, F.FeatureSpec())

    def test_available_at_after_cutoff_excludes_observation(self, history: F.History) -> None:
        t = _targets_for(history, 2025, "4")
        pid = "legacy:carl_p0"
        pg = history.player_games.copy()
        pg["available_at"] = pd.Series(pd.NaT, index=pg.index, dtype="datetime64[us, UTC]")
        last = pg[(pg.player_id == pid) & (pg.season == 2025) & (pg.stage_label == "3")].index
        pg.loc[last, "available_at"] = pd.Timestamp("2025-12-01", tz="UTC")
        h2 = dataclasses.replace(history, player_games=pg)
        a = F.build_features(history, t, F.FeatureSpec())
        b = F.build_features(h2, t, F.FeatureSpec())
        i = int(t.index[t.player_id == pid][0])
        assert b.history_games[i] == a.history_games[i] - 1


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    offset=st.integers(min_value=0, max_value=900),
    hour=st.integers(min_value=0, max_value=23),
    pidx=st.integers(min_value=0, max_value=4),
)
def test_property_features_depend_only_on_pre_cutoff_rows(
    history: F.History, offset: int, hour: int, pidx: int
) -> None:
    cutoff = datetime(2023, 3, 1, hour, tzinfo=UTC) + timedelta(days=offset)
    pid = f"legacy:carl_p{pidx}"
    tgt = pd.DataFrame([{
        "player_id": pid, "club_id": "carl", "match_id": "hyp", "season": cutoff.year,
        "stage_id": "r01", "stage_label": "1", "stage_type": "regular", "stage_order": 1,
        "match_date": cutoff.date() + timedelta(days=1), "opponent_club_id": None,
        "venue_id": None, "forecast_cutoff": pd.Timestamp(cutoff),
    }])
    full = F.build_features(history, tgt, F.FeatureSpec())
    m = history.matches.set_index("match_id")
    days = history.player_games.match_id.map(m["match_date"])
    keep = days.map(lambda d: d is not None and d <= cutoff.date())
    pruned = dataclasses.replace(history, player_games=history.player_games[keep.astype(bool)])
    small = F.build_features(pruned, tgt, F.FeatureSpec())
    pd.testing.assert_frame_equal(full.X, small.X)
    assert full.history_games[0] == small.history_games[0]


def test_match_day_cutoff_is_utc_midnight() -> None:
    assert F.match_day_cutoff(date(2026, 5, 1)) == datetime(2026, 5, 1, tzinfo=UTC)
