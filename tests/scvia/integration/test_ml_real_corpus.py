"""Real-corpus ML invariants (auto-marked integration).

Uses the session ``real_snapshot_root`` fixture (``SCVIA_SNAPSHOT_ROOT`` overrides the root,
``SCVIA_SNAPSHOT`` the selector; default ``current``).
Writes only under pytest's tmp_path. Fails (never skips) if the snapshot is absent.
"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from supercoach_via.domain.schemas import Origin
from supercoach_via.ml import evaluate as E
from supercoach_via.ml import features as F
from supercoach_via.ml import predict as P
from supercoach_via.ml import train as T

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def history(real_snapshot_root: Path) -> F.History:
    root = real_snapshot_root
    sel = os.environ.get("SCVIA_SNAPSHOT", "current")
    return F.load_history(root, sel, extra_tables=("legacy_predictions",))


def test_time_verification_policy_keeps_nearly_all_rows(history: F.History) -> None:
    t = F.historical_targets(history, seasons=(2026,))
    ff = F.build_features(history, t.head(500))
    d = ff.diagnostics
    assert d["excluded_unverified_date"] / d["observations_total"] < 0.01
    assert d["excluded_same_day_ambiguous"] == 0


def test_features_match_naive_recompute_on_sample(history: F.History) -> None:
    t = F.historical_targets(history, seasons=(2026,)).sample(40, random_state=0).reset_index(drop=True)
    ff = F.build_features(history, t)
    pg = history.player_games
    mdate = history.matches.set_index("match_id")["match_date"]
    # eligibility rule 2 (ml/features.py): the match must be status=complete; the three
    # extra-time finals with malformed source scores stay status=unknown and are excluded
    complete = pg.match_id.map(history.matches.set_index("match_id")["status"]).eq("complete")
    ok = F.time_verified(pg, F.FeatureSpec()) & complete
    for i, r in t.iterrows():
        rows = pg[(pg.player_id == r.player_id) & ok].copy()
        rows["day"] = rows.match_id.map(mdate)
        rows = rows[rows.day < r.forecast_cutoff.date()].sort_values(["day", "match_id"])
        assert ff.history_games[i] == len(rows)
        v = rows["disposals"].astype(float).tail(5)
        ref = v.mean() if v.notna().any() else np.nan
        got = ff.X.loc[i, "disposals_prior5_mean"]
        assert (np.isnan(ref) and np.isnan(got)) or got == pytest.approx(ref, rel=1e-12)


def test_target_outcomes_do_not_move_features(history: F.History) -> None:
    t = F.historical_targets(history, seasons=(2026,))
    gf_day = t.match_date.max()
    tt = t[t.match_date == gf_day].reset_index(drop=True)
    a = F.build_features(history, tt)
    pg = history.player_games.copy()
    fut = pg.match_id.map(history.matches.set_index("match_id")["match_date"]) >= gf_day
    pg.loc[fut, "disposals"] = 99
    b = F.build_features(replace(history, player_games=pg), tt)
    pd.testing.assert_frame_equal(a.X, b.X)


def test_prospective_forecast_is_honestly_unavailable_without_fixture(
    history: F.History, tmp_path: Path
) -> None:
    cfg = T.TrainingConfig(train_cutoff=date(2025, 1, 1), calibration_end=date(2025, 7, 1),
                           holdout_end=date(2026, 1, 1), target_seasons_from=2023, candidates=(),
                           n_folds=2, threads=2)
    bundle = T.train_model(history, cfg, bundle_root=tmp_path, clock=lambda: datetime(2026, 9, 25, tzinfo=UTC)).bundle
    assert bundle.manifest.name == "baseline_prior5"
    scheduled = (history.matches.status == "scheduled").sum()
    art = P.forecast(history, bundle, P.ForecastRequest(
        forecast_cutoff=datetime(2026, 9, 25, tzinfo=UTC), generated_at=datetime(2026, 9, 25, tzinfo=UTC)))
    if scheduled == 0:
        assert art.manifest.status == "unavailable" and art.manifest.reason == "no_valid_future_fixture"
    else:
        assert art.manifest.status in ("available", "unavailable")
    # a single replay stage reconciles its populations
    m26 = history.matches[(history.matches.season == 2026) & (history.matches.status == "complete")]
    last_stage = m26.sort_values("match_date").stage_id.iloc[-1]
    arts, ev = E.replay(history, bundle, season=2026, stage_ids=(last_stage,),
                        generated_at=datetime(2026, 9, 25, tzinfo=UTC))
    assert all(x.manifest.origin == Origin.REPLAY.value for x in arts)
    p = ev.populations
    assert p["joined"] + p["missing"] == p["predicted"]
    assert ev.headline.n == p["joined"] - p["excluded"]


def test_legacy_archive_summary_is_separate(history: F.History) -> None:
    s = E.legacy_unknown_summary(history)
    assert s["origin"] == "legacy_unknown" and "not prospective" in s["label"]
    assert s["rows"] > 0
