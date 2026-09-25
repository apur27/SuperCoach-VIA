"""Forecast contracts: M01, M05, M06 and artifact immutability."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from supercoach_via.domain.schemas import Origin
from supercoach_via.ml import features as F
from supercoach_via.ml import models as M
from supercoach_via.ml import predict as P
from supercoach_via.ml import train as T
from supercoach_via.publish.view_models import PredictionSet
from tests.scvia.fixtures.ml.synthetic import build_corpus

SMALL = {"hgb": {"max_iter": 20, "learning_rate": 0.1, "max_leaf_nodes": 7, "min_samples_leaf": 5}}
GEN = datetime(2026, 3, 1, 12, tzinfo=UTC)


def clock() -> datetime:
    return datetime(2026, 9, 25, tzinfo=UTC)


def _cfg() -> T.TrainingConfig:
    return T.TrainingConfig(train_cutoff=date(2025, 1, 1), calibration_end=date(2025, 4, 1),
                            target_seasons_from=2023, candidates=("hgb",), n_folds=2,
                            params=SMALL, threads=1)


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("m")
    corpus = build_corpus(future_rounds=2)
    hist = F.load_history(corpus.write(root / "snap"))
    res = T.train_model(hist, _cfg(), bundle_root=root / "models", clock=clock)
    return {"root": root, "corpus": corpus, "history": hist, "bundle": res.bundle}


def _prospective(**kw: Any) -> P.ForecastRequest:
    base: dict[str, Any] = dict(forecast_cutoff=datetime(2026, 3, 10, tzinfo=UTC),
                                generated_at=GEN, origin=Origin.PROSPECTIVE)
    base.update(kw)
    return P.ForecastRequest(**base)


class TestM01FutureFixture:
    def test_completed_history_creates_explicit_future_rows(self, env: dict[str, Any]) -> None:
        art = P.forecast(env["history"], env["bundle"], _prospective())
        assert art.manifest.status == "available"
        rows = art.rows
        assert set(rows.match_id) == {"m2026_r1_0", "m2026_r1_1"}
        assert (rows.season == 2026).all() and (rows.stage_id == "r01").all()
        assert (pd.to_datetime(rows.forecast_cutoff, utc=True) == pd.Timestamp("2026-03-10", tz="UTC")).all()
        assert set(rows.origin) == {"prospective"}
        # a completed historic player-game row is never relabelled as the target
        hist_keys = set(zip(env["history"].player_games.match_id, env["history"].player_games.player_id, strict=True))
        assert not hist_keys & set(zip(rows.match_id, rows.player_id, strict=True))
        assert rows.predicted_disposals.dtype == np.float64
        assert (rows.predicted_disposals >= 0).all()
        for c in P.CANONICAL_FIELDS:
            assert c in rows.columns

    def test_never_manufactures_next_round(self, env: dict[str, Any]) -> None:
        req = _prospective(season=2025)  # only completed matches in 2025
        art = P.forecast(env["history"], env["bundle"], req)
        assert art.manifest.status == "unavailable"
        assert art.manifest.reason == "no_valid_future_fixture"
        assert art.rows.empty


class TestM06:
    def test_no_future_fixtures_yields_unavailable(self, env: dict[str, Any], tmp_path: Path) -> None:
        hist = F.load_history(build_corpus().write(tmp_path / "s"))
        art = P.forecast(hist, env["bundle"], _prospective())
        assert art.manifest.status == "unavailable"
        assert art.manifest.reason == "no_valid_future_fixture"
        ps = P.to_prediction_set(art, hist, env["bundle"])
        assert isinstance(ps, PredictionSet) and ps.status == "unavailable" and ps.rows == []

    def test_zero_history_gets_labelled_cold_start_and_selection_unconfirmed(self, env: dict[str, Any]) -> None:
        req = _prospective(candidates=(("legacy:brand_new", "adel"), ("legacy:adel_p0", "adel")))
        hist = env["history"]
        players = pd.concat([hist.players, pd.DataFrame([{
            "player_id": "legacy:brand_new", "display_name": "Brand New",
            "birth_date_quality": "unknown", "identity_status": "canonical", "provenance": "demo"}])])
        from dataclasses import replace

        art = P.forecast(replace(hist, players=players), env["bundle"], req)
        r = art.rows.set_index("player_id")
        assert r.loc["legacy:brand_new", "history_games"] == 0
        assert r.loc["legacy:brand_new", "eligibility_basis"] == M.COLD_START_NAME
        assert M.COLD_START_NAME in r.loc["legacy:brand_new", "warnings"]
        assert (art.rows.selection_status == "unconfirmed").all()

    def test_omission_reasons_for_every_intended_candidate(self, env: dict[str, Any]) -> None:
        req = _prospective(candidates=(
            ("legacy:coll_ambig", "coll"),       # ambiguous identity
            ("legacy:nobody", "adel"),           # not in players table
            ("legacy:adel_p1", "adel"),          # fine
            ("legacy:carl_p0", "no_such_club"),  # club not in fixture
        ), allow_cold_start=False)
        art = P.forecast(env["history"], env["bundle"], req)
        om = dict(zip(art.omissions.player_id, art.omissions.reason, strict=True))
        assert om == {"legacy:coll_ambig": "unresolved_identity", "legacy:nobody": "unresolved_identity",
                      "legacy:carl_p0": "not_in_fixture"}
        assert list(art.rows.player_id) == ["legacy:adel_p1"]
        assert art.manifest.intended == 4

    def test_insufficient_history_when_cold_start_disallowed(self, env: dict[str, Any]) -> None:
        from dataclasses import replace

        hist = env["history"]
        players = pd.concat([hist.players, pd.DataFrame([{
            "player_id": "legacy:brand_new", "display_name": "Brand New",
            "birth_date_quality": "unknown", "identity_status": "canonical", "provenance": "demo"}])])
        req = _prospective(candidates=(("legacy:brand_new", "adel"),), allow_cold_start=False)
        art = P.forecast(replace(hist, players=players), env["bundle"], req)
        assert art.rows.empty
        assert list(art.omissions.reason) == ["insufficient_history"]

    def test_unconfirmed_stays_unconfirmed_and_late_lineup_ignored(self, env: dict[str, Any], tmp_path: Path) -> None:
        corpus = build_corpus(future_rounds=1)
        corpus.lineups = [
            {"match_id": "m2026_r1_0", "club_id": "adel", "player_id": "legacy:adel_p0", "season": 2026,
             "role": "named", "announced_at": datetime(2026, 3, 12, tzinfo=UTC),  # after cutoff
             "confidence": "high", "name_token": "x", "resolution": "match_participation",
             "provenance": "demo"},
            {"match_id": "m2026_r1_0", "club_id": "carl", "player_id": "legacy:carl_p0", "season": 2026,
             "role": "named", "announced_at": datetime(2026, 3, 9, tzinfo=UTC),  # before cutoff
             "confidence": "high", "name_token": "x", "resolution": "match_participation",
             "provenance": "demo"},
        ]
        hist = F.load_history(corpus.write(tmp_path / "l"))
        art = P.forecast(hist, env["bundle"], _prospective())
        adel = art.rows[art.rows.club_id == "adel"]
        carl = art.rows[art.rows.club_id == "carl"]
        assert (adel.selection_status == "unconfirmed").all() and len(adel) > 1
        assert list(carl.player_id) == ["legacy:carl_p0"]
        assert (carl.selection_status == "confirmed").all()


class TestM05ReplayProspectiveParity:
    def test_identical_features_and_predictions(self, env: dict[str, Any], tmp_path: Path) -> None:
        # prospective world: 2026 r1 scheduled (no stats). replay world: same fixture completed.
        corpus = build_corpus(future_rounds=1)
        pro_hist = F.load_history(corpus.write(tmp_path / "pro"))
        done = build_corpus(future_rounds=1)
        rng = np.random.default_rng(3)
        for m in done.matches:
            if m["season"] == 2026:
                m["status"] = "complete"
                for club, opp in ((m["home_club_id"], m["away_club_id"]), (m["away_club_id"], m["home_club_id"])):
                    for k in range(5):
                        done.player_games.append({
                            "match_id": m["match_id"], "player_id": f"legacy:{club}_p{k}", "club_id": club,
                            "season": 2026, "opponent_club_id": opp, "stage_label": "1", "stage_id": "r01",
                            "club_source_name": club.upper(), "link_method": "key",
                            "match_date": m["match_date"], "date_quality": "fixture_verified",
                            "revision_id": "r1", "provenance": "demo",
                            "disposals": int(rng.integers(0, 40)), "kicks": 1, "handballs": 1,
                        })
        rep_hist = F.load_history(done.write(tmp_path / "rep"))
        cutoff = datetime(2026, 3, 10, tzinfo=UTC)
        pro = P.forecast(pro_hist, env["bundle"], _prospective(forecast_cutoff=cutoff))
        rep = P.forecast(rep_hist, env["bundle"], _prospective(forecast_cutoff=cutoff, origin=Origin.REPLAY,
                                                               season=2026, stage_id="r01"))
        assert rep.manifest.origin == "replay" and pro.manifest.origin == "prospective"
        key = ["match_id", "player_id"]
        a = pro.rows.sort_values(key).reset_index(drop=True)
        b = rep.rows.sort_values(key).reset_index(drop=True)
        assert list(a.player_id) == list(b.player_id)
        np.testing.assert_array_equal(a.predicted_disposals.to_numpy(), b.predicted_disposals.to_numpy())
        pd.testing.assert_frame_equal(pro.features.X, rep.features.X)


class TestArtifactIO:
    def test_write_is_immutable_and_hash_verified(self, env: dict[str, Any], tmp_path: Path) -> None:
        art = P.forecast(env["history"], env["bundle"], _prospective())
        d = P.write_artifact(art, tmp_path / "pred")
        before = {p.name: p.read_bytes() for p in d.iterdir()}
        with pytest.raises(P.ArtifactExistsError):
            P.write_artifact(art.with_rows(art.rows.assign(predicted_disposals=1.0)), tmp_path / "pred")
        assert {p.name: p.read_bytes() for p in d.iterdir()} == before
        back = P.load_artifact(d)
        pd.testing.assert_frame_equal(back.rows, art.rows, check_dtype=False)
        (d / "rows.parquet").chmod(0o644)
        (d / "rows.parquet").write_bytes(b"x" + (d / "rows.parquet").read_bytes())
        with pytest.raises(P.ArtifactIntegrityError):
            P.load_artifact(d)

    def test_to_prediction_set_view_model(self, env: dict[str, Any]) -> None:
        art = P.forecast(env["history"], env["bundle"], _prospective())
        ps = P.to_prediction_set(art, env["history"], env["bundle"])
        assert ps.status == "available" and len(ps.rows) == len(art.rows)
        assert ps.model is not None and ps.model.model_id == env["bundle"].bundle_id
        assert {m.match_id for m in ps.target_matches} == set(art.rows.match_id)
        r0 = ps.rows[0]
        assert r0.player_name and r0.club_name and r0.origin == "prospective"
