"""Pin the real _stat_leaders.json schema so QA's embedded snippets can't regress (A-02 / CR-2).

QA.md previously read `leaders['career_games']['leaders'][0]` and `rank1['player_id']`,
which KeyError against the real schema: top-level keys are meta/categories/single_season,
career stats live under `categories.<stat>.leaders`, and leader objects carry no
`player_id`. These tests assert the schema QA now relies on, and execute QA's corrected
rank-1 games cross-check end-to-end against the live JSON + player CSVs.
"""
import glob
import json
import pathlib

import pandas as pd
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
JSON_PATH = REPO_ROOT / "docs" / "hall-of-fame" / "_stat_leaders.json"


@pytest.fixture(scope="module")
def doc():
    if not JSON_PATH.exists():
        pytest.skip("_stat_leaders.json not present (gitignored / not yet generated)")
    return json.loads(JSON_PATH.read_text())


def test_top_level_keys(doc):
    assert set(doc.keys()) == {"meta", "categories", "single_season"}


def test_career_games_lives_under_categories(doc):
    # The exact path QA's snippets now use.
    assert "career_games" in doc["categories"]
    assert "leaders" in doc["categories"]["career_games"]
    assert len(doc["categories"]["career_games"]["leaders"]) > 0


def test_leader_object_has_no_player_id(doc):
    # The bug: QA globbed by rank1['player_id'], which does not exist.
    leader = doc["categories"]["career_games"]["leaders"][0]
    assert "player_id" not in leader
    for key in ("rank", "name", "teams", "games", "total", "per_game"):
        assert key in leader, f"leader object missing expected key {key!r}"


def test_qa_rank1_cross_check_runs(doc):
    """QA's corrected rank-1 games verification must run without exception and match."""
    rank1 = doc["categories"]["career_games"]["leaders"][0]
    parts = rank1["name"].split()
    surname, first = parts[-1].lower(), parts[0].lower()
    files = glob.glob(
        str(REPO_ROOT / "data" / "player_data" / f"{surname}_{first}_*_performance_details.csv")
    )
    if not files:
        pytest.skip(f"no player CSV for rank-1 leader {rank1['name']}")
    df = pd.read_csv(files[0])
    csv_games = max(len(df), int(df["games_played"].max()))  # canonical games metric
    assert csv_games == int(rank1["total"]), (
        f"HOF JSON says {rank1['total']}, CSV says {csv_games} for {rank1['name']}"
    )
