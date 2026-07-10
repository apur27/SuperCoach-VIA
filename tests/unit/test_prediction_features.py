"""Phantom-feature guard for the disposal predictor (Task S1b).

The model declares a set of feature columns (``base_rolling_features`` and
``extra_features``) plus a ``RENAMES`` map that normalises raw CSV column
names to the model's canonical names. If a declared feature has no matching
column after RENAMES are applied, it is *silently dropped* during feature
engineering (see ``_engineer_features``:
``extra_feats = [f for f in self.extra_features if f in df.columns]``).
Silent drops are dangerous: the model trains on fewer features than the
author believes, with no error.

Two historical phantoms motivated this guard:
  * ``percentage_time_played`` — the raw CSV column is
    ``percentage_of_game_played`` (time-on-ground %). Without a RENAMES entry
    the declared feature never resolved and was dropped every run.
  * ``cba_percent`` — centre-bounce-attendance %. No such column (nor any
    equivalent) exists anywhere in ``data/player_data/`` (verified 0/200
    sampled CSVs), so it was declared but never present.

These tests pin the invariant: every declared feature must resolve to a real
column after RENAMES are applied to the canonical raw CSV schema.
"""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MODULE_PATH = REPO_ROOT / "supercoach" / "prediction.py"

# The canonical raw column schema of a *_performance_details.csv, as written
# by the scraper. This is the data contract the model consumes. Verified
# against 200/13350 real files on 2026-07-09 (all identical).
RAW_PERFORMANCE_COLUMNS = [
    "team", "year", "games_played", "opponent", "round", "result",
    "jersey_num", "kicks", "marks", "handballs", "disposals", "goals",
    "behinds", "hit_outs", "tackles", "rebound_50s", "inside_50s",
    "clearances", "clangers", "free_kicks_for", "free_kicks_against",
    "brownlow_votes", "contested_possessions", "uncontested_possessions",
    "contested_marks", "marks_inside_50", "one_percenters", "bounces",
    "goal_assist", "percentage_of_game_played", "date",
]


def _load_module():
    spec = importlib.util.spec_from_file_location("prediction_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


@pytest.fixture
def predictor(mod, tmp_path):
    """A predictor instance with an explicit target_year so construction does
    NOT scan the 13k-file data dir to auto-detect the year (keeps tests fast).
    """
    return mod.AFLDisposalPredictor(data_dir=str(tmp_path), target_year=2026)


def _renamed_columns(mod):
    """The set of column names the model actually sees after clean_columns."""
    return {mod.RENAMES.get(c, c) for c in RAW_PERFORMANCE_COLUMNS}


def test_base_rolling_features_all_resolve(mod, predictor):
    available = _renamed_columns(mod)
    missing = [f for f in predictor.base_rolling_features if f not in available]
    assert not missing, (
        f"base_rolling_features declared but absent after RENAMES: {missing}. "
        f"Add a RENAMES entry or remove the feature."
    )


def test_extra_features_all_resolve(mod, predictor):
    """The core phantom guard: every extra_feature must map to a real column."""
    available = _renamed_columns(mod)
    missing = [f for f in predictor.extra_features if f not in available]
    assert not missing, (
        f"extra_features declared but absent after RENAMES: {missing}. "
        f"Each must either map from a raw column via RENAMES or be removed."
    )


def test_percentage_time_played_is_wired_from_tog(mod):
    """The TOG column (percentage_of_game_played) must be renamed to the
    declared feature name, not silently dropped."""
    assert mod.RENAMES.get("percentage_of_game_played") == "percentage_time_played"


def test_cba_percent_not_declared_without_data(mod, predictor):
    """cba_percent has no source column anywhere in the data. It must not be
    reintroduced as a declared feature (it would silently drop every run)."""
    assert "cba_percent" not in predictor.extra_features
    assert "cba_percent" not in predictor.base_rolling_features
    # And it must not be in RENAMES targets either.
    assert "cba_percent" not in set(mod.RENAMES.values())


def _synthetic_engineered_df():
    """A minimal cleaned (post-RENAMES) frame with two players over four
    rounds each — enough history for the shift(1)/rolling features to
    resolve on at least one surviving row. Uses canonical column names
    (the model consumes data after clean_columns has applied RENAMES)."""
    rows = []
    for player in ("Alpha Player", "Beta Player"):
        for r in range(1, 5):
            rows.append({
                "player": player,
                "year": 2025,
                "round": f"Round {r}",
                "date": pd.Timestamp("2025-03-01") + pd.Timedelta(days=7 * r),
                "disposals": 20 + r,
                "kicks": 10 + r,
                "handballs": 10,
                "tackles": 4,
                "clearances": 3,
                "inside_50s": 5,
                "percentage_time_played": 80 + r,
            })
    return pd.DataFrame(rows)


def test_no_unlagged_raw_feature_in_feature_columns(predictor):
    """Guard against target leakage / train-serve skew: no raw same-game
    feature may enter feature_columns. percentage_time_played must appear
    only in its shift(1) lagged form; cba_percent must never reappear."""
    predictor._engineer_features(_synthetic_engineered_df())

    # Raw same-game TOG% must NOT be a feature (would leak game-i info into
    # the game-i prediction; also causes train/serve skew in production).
    assert "percentage_time_played" not in predictor.feature_columns

    # cba_percent regression guard — phantom feature, must stay out.
    assert "cba_percent" not in predictor.feature_columns

    # The lagged (prior-round) TOG% IS a legitimate conditioning signal.
    assert "percentage_time_played_lag1" in predictor.feature_columns


def test_tog_lag_uses_prior_game_only(predictor):
    """The lagged TOG% column must equal the player's previous-game value
    (strictly prior information), never the same game's value."""
    df = predictor._engineer_features(_synthetic_engineered_df())
    df = df.sort_values(["player", "round"])
    alpha = df[df["player"] == "Alpha Player"]
    # Round 4 row's lag must be Round 3's raw TOG% (83), not Round 4's (84).
    r4 = alpha[alpha["round"] == "Round 4"].iloc[0]
    assert r4["percentage_time_played_lag1"] == 83
    assert r4["percentage_time_played"] == 84


def test_existing_renames_still_resolve(mod):
    """Guard the pre-existing renames (hit_outs, free_kicks_*) so this change
    doesn't regress them."""
    available = _renamed_columns(mod)
    for canonical in ("hitouts", "frees_for", "frees_against"):
        assert canonical in available, f"{canonical} no longer resolves after RENAMES"
