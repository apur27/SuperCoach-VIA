"""Evaluation contracts: M07 metric recomputation, M08 origin isolation, M09 intervals."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from supercoach_via.domain.schemas import Origin
from supercoach_via.ml import evaluate as E
from supercoach_via.ml import features as F
from supercoach_via.ml import predict as P
from supercoach_via.ml import train as T
from supercoach_via.publish.view_models import AccuracyReport
from tests.scvia.fixtures.ml.synthetic import build_corpus

SMALL = {"hgb": {"max_iter": 20, "learning_rate": 0.1, "max_leaf_nodes": 7, "min_samples_leaf": 5}}


def clock() -> datetime:
    return datetime(2026, 9, 25, tzinfo=UTC)


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("e")
    hist = F.load_history(build_corpus().write(root / "snap"))
    cfg = T.TrainingConfig(train_cutoff=date(2024, 1, 1), calibration_end=date(2024, 6, 1),
                           target_seasons_from=2023, candidates=("hgb",), n_folds=2, params=SMALL,
                           threads=1)
    bundle = T.train_model(hist, cfg, bundle_root=root / "models", clock=clock).bundle
    return {"root": root, "history": hist, "bundle": bundle}


# ---------------------------------------------------------------------------
# M07
# ---------------------------------------------------------------------------


class TestM07Metrics:
    def test_independent_recompute_full_precision(self) -> None:
        p = np.array([10.25, 0.0, 33.3333333, 7.1])
        a = np.array([12.0, 0.0, 30.0, 7.1])
        m = E.metric_block(p, a)
        e = p - a
        assert m.mae == pytest.approx(np.abs(e).mean(), rel=0, abs=1e-15)
        assert m.rmse == pytest.approx(math.sqrt((e**2).mean()), abs=1e-15)
        assert m.bias == pytest.approx(e.mean(), abs=1e-15)
        assert m.median_ae == pytest.approx(np.median(np.abs(e)))
        assert m.within_5 == 1.0 and m.within_10 == 1.0 and m.n == 4

    def test_unknown_actuals_must_be_excluded_not_zero_filled(self) -> None:
        with pytest.raises(ValueError):
            E.metric_block(np.array([1.0]), np.array([np.nan]))

    def test_unequal_rounds_reconcile_to_pooled(self) -> None:
        rng = np.random.default_rng(1)
        sizes = [3, 50, 7]
        groups = np.repeat(["r1", "r2", "r3"], sizes)
        p = rng.random(sum(sizes)) * 30
        a = rng.integers(0, 35, sum(sizes)).astype(float)
        pooled = E.metric_block(p, a).mae
        per = [E.metric_block(p[groups == g], a[groups == g]) for g in ("r1", "r2", "r3")]
        weighted = sum(x.mae * x.n for x in per) / sum(x.n for x in per)  # type: ignore[operator]
        assert pooled == pytest.approx(weighted, rel=1e-12)
        mor = E.mean_of_groups_mae(p, a, groups)
        assert mor == pytest.approx(np.mean([x.mae for x in per]), rel=1e-12)
        assert mor != pytest.approx(pooled, rel=1e-6)


@settings(max_examples=50, deadline=None)
@given(st.lists(st.tuples(st.floats(0, 60), st.integers(0, 60), st.integers(0, 4)), min_size=1, max_size=80))
def test_property_pooled_mae_is_row_weighted(rows: list[tuple[float, int, int]]) -> None:
    p = np.array([r[0] for r in rows])
    a = np.array([float(r[1]) for r in rows])
    g = np.array([r[2] for r in rows])
    pooled = E.metric_block(p, a).mae
    num = sum(E.metric_block(p[g == k], a[g == k]).mae * (g == k).sum() for k in np.unique(g))  # type: ignore[operator]
    assert pooled == pytest.approx(num / len(rows), rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# M09
# ---------------------------------------------------------------------------


class TestM09Intervals:
    def test_quantile_rank_boundary(self) -> None:
        r = np.arange(1, 201, dtype=float)  # n=200 -> rank ceil(201*0.8)=161
        cal = E.conformal_quantile(r, 0.8, 200)
        assert cal.available and cal.q == 161.0 and cal.n_calibration == 200
        r2 = np.arange(1, 251, dtype=float)  # n=250 -> ceil(251*0.8)=201
        assert E.conformal_quantile(r2, 0.8).q == 201.0

    def test_insufficient_calibration_is_null_with_reason(self) -> None:
        cal = E.conformal_quantile(np.ones(199), 0.8, 200)
        assert not cal.available and cal.q is None
        assert cal.reason is not None and cal.reason.startswith("insufficient_calibration")
        lo, hi = E.apply_interval(np.array([10.0]), cal)
        assert np.isnan(lo).all() and np.isnan(hi).all()

    def test_rank_beyond_sample_is_unavailable(self) -> None:
        cal = E.conformal_quantile(np.ones(200), 0.999, 200)
        assert not cal.available and cal.reason == "quantile_rank_exceeds_sample"

    def test_lower_bound_clipped_at_zero_and_not_mae(self) -> None:
        resid = np.concatenate([np.zeros(150), np.full(100, 10.0)])  # MAE=4, q80=10
        cal = E.conformal_quantile(resid)
        assert cal.q == 10.0 and cal.q != pytest.approx(resid.mean())
        lo, hi = E.apply_interval(np.array([3.0, 20.0]), cal)
        assert list(lo) == [0.0, 10.0] and list(hi) == [13.0, 30.0]

    def test_calibrated_label_only_inside_band_with_enough_outcomes(self) -> None:
        a = np.arange(300, dtype=float)
        lo, hi = a - 1, a + 1
        good = E.evaluate_interval(lo, hi, np.where(np.arange(300) < 240, a, a + 5))  # 80%
        assert good.calibrated and good.coverage == pytest.approx(0.8)
        over = E.evaluate_interval(lo, hi, a)  # 100% coverage -> outside band
        assert not over.calibrated and over.reason and "outside_band" in over.reason
        few = E.evaluate_interval(lo[:50], hi[:50], a[:50])
        assert not few.calibrated and few.reason and few.reason.startswith("insufficient_holdout")
        assert good.median_width == 2.0


# ---------------------------------------------------------------------------
# Scoring / replay / origin isolation (M08)
# ---------------------------------------------------------------------------


class TestScoring:
    def _replay_art(self, env: dict[str, Any]) -> P.PredictionArtifact:
        req = P.ForecastRequest(forecast_cutoff=datetime(2025, 3, 16, tzinfo=UTC),
                                generated_at=datetime(2026, 9, 25, tzinfo=UTC), origin=Origin.REPLAY,
                                season=2025, stage_id="r01")
        return P.forecast(env["history"], env["bundle"], req)

    def test_populations_and_reconciliation(self, env: dict[str, Any]) -> None:
        art = self._replay_art(env)
        ev = E.evaluate(art, env["history"])
        pop = ev.populations
        assert pop["predicted"] == len(art.rows)
        assert pop["joined"] + pop["missing"] == pop["predicted"]
        assert ev.headline.n == pop["joined"] - pop["excluded"]
        s = ev.scored_rows
        assert ev.headline.mae == pytest.approx(np.abs(s.predicted_disposals - s.actual).mean(), abs=1e-12)
        assert ev.baseline_headline is not None and ev.baseline_headline.n == ev.headline.n
        assert ev.origin == "replay"
        assert any(c.dimension == "actual_band" and c.post_hoc for c in ev.cohorts)

    def test_zero_actual_is_valid_and_unknown_is_excluded(self, env: dict[str, Any]) -> None:
        art = self._replay_art(env)
        pg = env["history"].player_games.copy()
        ids = art.rows.player_id.tolist()
        m0 = art.rows.match_id.iloc[0]
        pg.loc[(pg.match_id == m0) & (pg.player_id == ids[0]), "disposals"] = 0
        pg.loc[(pg.match_id == m0) & (pg.player_id == ids[1]), "disposals"] = None
        ev = E.evaluate(art, replace(env["history"], player_games=pg))
        assert ev.populations["exclusion_reasons"].get("unknown_actual") == 1
        assert 0.0 in set(ev.scored_rows.actual)

    def test_wrong_target_manifest_rejected(self, env: dict[str, Any]) -> None:
        art = self._replay_art(env)
        m = env["history"].matches.copy()
        mid = art.rows.match_id.iloc[0]
        m.loc[m.match_id == mid, "stage_id"] = "r09"
        with pytest.raises(E.TargetMismatchError):
            E.evaluate(art, replace(env["history"], matches=m))
        bad_rows = art.rows.copy()
        bad_rows.loc[0, "season"] = 1999
        with pytest.raises(E.TargetMismatchError):
            E.evaluate(art.with_rows(bad_rows), env["history"])

    def test_rescoring_never_rewrites_forecast_bytes(self, env: dict[str, Any], tmp_path: Path) -> None:
        art = self._replay_art(env)
        d = P.write_artifact(art, tmp_path / "pred")
        before = {p.name: p.read_bytes() for p in d.iterdir()}
        e1 = E.score_archive(d, env["history"], allow_origins=(Origin.REPLAY,))
        E.write_evaluation(e1, tmp_path / "eval")
        pg = env["history"].player_games.copy()
        pg["disposals"] = pg["disposals"].fillna(0) + 1  # corrected actuals -> new version
        e2 = E.score_archive(d, replace(env["history"], snapshot_id="sha256:" + "1" * 64,
                                        player_games=pg), allow_origins=(Origin.REPLAY,))
        E.write_evaluation(e2, tmp_path / "eval")
        assert e1.evaluation_id != e2.evaluation_id
        assert {p.name: p.read_bytes() for p in d.iterdir()} == before

    def test_written_evaluation_reads_back_identically(self, env: dict[str, Any], tmp_path: Path) -> None:
        art = self._replay_art(env)
        ev = E.evaluate(art, env["history"])
        d = E.write_evaluation(ev, tmp_path / "eval")
        back = E.read_evaluation(d)
        assert back.evaluation_id == ev.evaluation_id and back.origin == ev.origin
        assert back.headline == ev.headline and back.baseline_headline == ev.baseline_headline
        assert back.cohorts == ev.cohorts and back.interval == ev.interval
        assert back.populations == ev.populations and back.notes == ev.notes
        pd.testing.assert_frame_equal(back.scored_rows, ev.scored_rows)

    def test_read_evaluation_rejects_mismatched_directory(self, env: dict[str, Any], tmp_path: Path) -> None:
        d = E.write_evaluation(E.evaluate(self._replay_art(env), env["history"]), tmp_path / "eval")
        moved = d.rename(d.parent / "not-the-id")
        with pytest.raises(ValueError, match="evaluation_id"):
            E.read_evaluation(moved)

    def test_score_requires_prospective_by_default(self, env: dict[str, Any], tmp_path: Path) -> None:
        d = P.write_artifact(self._replay_art(env), tmp_path / "p2")
        with pytest.raises(E.OriginError):
            E.score_archive(d, env["history"])

    def test_origins_are_never_pooled(self, env: dict[str, Any]) -> None:
        ev = E.evaluate(self._replay_art(env), env["history"])
        fake = replace(ev, origin="prospective")
        with pytest.raises(E.OriginError):
            E.pool_evaluations([ev, fake])
        leg = replace(ev, origin="legacy_unknown")
        with pytest.raises(E.OriginError):
            E.pool_evaluations([fake, leg])
        pooled = E.pool_evaluations([ev, ev])
        assert pooled.headline.n == 2 * ev.headline.n

    def test_replay_season_and_accuracy_report(self, env: dict[str, Any]) -> None:
        arts, ev = E.replay(env["history"], env["bundle"], season=2025,
                            generated_at=datetime(2026, 9, 25, tzinfo=UTC))
        assert len(arts) == len({(a.manifest.stage_id) for a in arts})
        assert all(a.manifest.origin == "replay" for a in arts)
        cut = {a.manifest.stage_id: a.manifest.forecast_cutoff for a in arts}
        assert cut["r02"] > cut["r03"]  # postponed round 2 played after round 3
        rep = E.to_accuracy_report(ev, model_id=env["bundle"].bundle_id)
        assert isinstance(rep, AccuracyReport) and rep.origin == "replay"
        assert "replay" in rep.label.lower() and rep.populations.predicted == ev.populations["predicted"]
        assert rep.mean_of_rounds_mae is not None


class TestLegacyUnknown:
    def test_legacy_archive_kept_separate(self) -> None:
        hist = F.History("sha256:" + "0" * 64, pd.DataFrame({
            "match_id": ["m1"], "season": [2026], "stage_label": ["5"], "stage_id": ["r05"],
            "home_club_id": ["adel"], "away_club_id": ["carl"], "status": ["complete"],
            "match_date": [date(2026, 4, 1)]}),
            pd.DataFrame({"match_id": ["m1", "m1"], "player_id": ["legacy:a", "legacy:b"],
                          "club_id": ["adel", "carl"], "disposals": [20, 0]}),
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            extra={"legacy_predictions": pd.DataFrame({
                "prediction_row_id": ["x1", "x2", "x3"], "artifact_kind": ["next_round"] * 3,
                "claimed_round_label": ["5", "5", "5"], "claimed_season": [2026] * 3,
                "claimed_timestamp": ["20260401_1000"] * 3, "player_display_name": ["A", "B", "C"],
                "club_id": ["adel", "carl", "adel"], "player_id": ["legacy:a", "legacy:b", None],
                "predicted_value": [18.0, 3.0, 10.0], "origin": ["legacy_unknown"] * 3,
                "source_path": ["data/prediction/next_round_5_prediction_20260401_1000.csv"] * 3})})
        s = E.legacy_unknown_summary(hist)
        assert s["origin"] == "legacy_unknown" and s["rows"] == 3
        assert s["joined"] == 2 and s["unjoined_reasons"] == {"unresolved_identity": 1}
        assert s["metrics"]["mae"] == pytest.approx((2 + 3) / 2)
        assert "not prospective" in s["label"]

    def test_null_claimed_season_derived_from_filename_timestamp(self) -> None:
        hist = F.History("sha256:" + "0" * 64, pd.DataFrame({
            "match_id": ["m1"], "season": [2026], "stage_label": ["5"], "stage_id": ["r05"],
            "home_club_id": ["adel"], "away_club_id": ["carl"], "status": ["complete"],
            "match_date": [date(2026, 4, 1)]}),
            pd.DataFrame({"match_id": ["m1"], "player_id": ["legacy:a"], "club_id": ["adel"],
                          "disposals": [20]}),
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            extra={"legacy_predictions": pd.DataFrame({
                "prediction_row_id": ["x1"], "artifact_kind": ["next_round"],
                "claimed_round_label": ["5"], "claimed_season": [np.nan],
                "claimed_timestamp": ["20260401_1000"], "player_display_name": ["A"],
                "club_id": ["adel"], "player_id": ["legacy:a"], "predicted_value": [18.0],
                "origin": ["legacy_unknown"], "source_path": ["data/prediction/f.csv"]})})
        s = E.legacy_unknown_summary(hist)
        assert s["joined"] == 1 and s["season_from_timestamp"] == 1
