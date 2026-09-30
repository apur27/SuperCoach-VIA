"""Independent source denominator oracle, anchored by annotated and hand-calculated cases."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from supercoach_via.analytics import eras
from supercoach_via.domain import blanks
from supercoach_via.domain.schemas import LEGACY_PLAYER_COLUMN_MAP, PLAYER_STAT_COLUMNS
from supercoach_via.ingest.reconcile import load_policy
from supercoach_via.storage.queries import SnapshotQuery
from tests.scvia.era_source_oracle import resolve_source_blanks, source_era_stats
from tests.scvia.fixtures.analytics import synthetic as syn

REPO = Path(__file__).resolve().parents[3]
ANNOTATIONS = json.loads((REPO / "tests/scvia/fixtures/zero_semantics/annotations.json").read_text())
RECORDED = load_policy(REPO / "config").coverage.recorded_from
CANON_TO_LEGACY = {v: k for k, v in LEGACY_PLAYER_COLUMN_MAP.items()}


def source_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Express the annotated cases using raw legacy column names."""
    return pd.DataFrame([
        {**dict.fromkeys(CANON_TO_LEGACY.get(s, s) for s in PLAYER_STAT_COLUMNS),
         **{CANON_TO_LEGACY.get(k, k): v for k, v in r.items()}}
        for r in rows
    ])


@pytest.mark.parametrize("case", ANNOTATIONS["row_cases"], ids=[c["name"] for c in ANNOTATIONS["row_cases"]])
def test_oracle_matches_independent_source_annotations(case: dict) -> None:
    rows = [{"match_id": "m1", "year": case["season"], "stage_type": case["stage_type"],
             **{"time_on_ground_pct" if k == "tog" else k: v for k, v in r.items()}} for r in case["match"]]
    got = resolve_source_blanks(source_rows(rows), RECORDED).set_index("id")
    for player, expected in case["expect"].items():
        for stat, value in expected.items():
            name = CANON_TO_LEGACY.get("time_on_ground_pct" if stat == "tog" else stat, stat)
            cell = got.loc[player, name]
            assert pd.isna(cell) if value is None else cell == value


def test_imported_zeros_have_hand_calculated_counts_median_and_per100(tmp_path: Path) -> None:
    rows = [
        {"match_id": "old", "year": 1950, "stage_type": "regular", "goals": 2},
        {"match_id": "old", "year": 1950, "stage_type": "regular"},
        {"match_id": "new", "year": 2021, "stage_type": "regular", "goals": 4, "time_on_ground_pct": 50},
        {"match_id": "new", "year": 2021, "stage_type": "regular", "time_on_ground_pct": 100},
        {"match_id": "new", "year": 2021, "stage_type": "regular"},
        {"match_id": "new", "year": 2021, "stage_type": "regular", "time_on_ground_pct": 10},
        {"match_id": "new", "year": 2021, "stage_type": "regular", "goals": 2},
        {"match_id": "unreported", "year": 2021, "stage_type": "regular", "kicks": 9, "time_on_ground_pct": 80},
    ]
    source = source_rows(rows)
    expected = source_era_stats(resolve_source_blanks(source, RECORDED)).set_index(["era", "metric"])
    golden = {
        "pre-1965": (2, 2, 1, math.sqrt(2), 1, math.nan),
        # Source values [4, blank, unknown row, blank, 2, unreported match] -> [4, 0, null, 0, 2, null].
        # Only the 50% and 100% rows enter per100: mean([8, 0]) = 4.
        "2011-present": (6, 4, 1.5, math.sqrt(11 / 3), 1, 4),
    }
    columns = ("n_player_games", "n_with_metric", "mean_per_game", "std_per_game",
               "median_per_game", "mean_per_100pct_played")
    canonical = [{**dict.fromkeys(PLAYER_STAT_COLUMNS), "season": r["year"],
                  **{k: v for k, v in r.items() if k not in {"year", "stage_type"}}} for r in rows]
    resolved, counts = blanks.resolve_blanks(pa.Table.from_pylist(canonical),
                                             {"old": "regular", "new": "regular", "unreported": "regular"}, RECORDED)
    assert counts["goals"] == 3
    games = [syn.pg(r["match_id"], f"p{i}", "A", r["season"], i + 1,
                    **{s: r[s] for s in PLAYER_STAT_COLUMNS}) for i, r in enumerate(resolved.to_pylist())]
    manifest = syn.build(tmp_path, {"player_games": games})
    with SnapshotQuery(tmp_path, manifest) as q:
        actual = eras.era_stats(q).set_index(["era", "legacy_metric"])
    for era, values in golden.items():
        for column, value in zip(columns, values, strict=True):
            for frame in (expected, actual):
                got = frame.loc[(era, "goals"), column]
                assert pd.isna(got) if math.isnan(value) else got == pytest.approx(value)


@pytest.mark.parametrize("corruption", ["self_consistent_count", "median_per_game", "mean_per_100pct_played"])
def test_source_oracle_rejects_count_and_affected_stat_mutations(corruption: str) -> None:
    from tests.scvia.era_source_oracle import assert_source_summary

    source = source_rows([
        {"match_id": "m1", "year": 2021, "stage_type": "regular", "goals": 4, "time_on_ground_pct": 50},
        {"match_id": "m1", "year": 2021, "stage_type": "regular", "time_on_ground_pct": 100},
    ])
    expected = source_era_stats(resolve_source_blanks(source, RECORDED))
    actual = expected.rename(columns={"metric": "legacy_metric"}).copy()
    index = actual.index[(actual["era"] == "2011-present") & (actual["legacy_metric"] == "goals")][0]
    if corruption == "self_consistent_count":
        # Invent another zero: mean and SD still satisfy the former parity test's equations.
        actual.loc[index, ["n_with_metric", "mean_per_game", "std_per_game"]] = [3, 4 / 3, math.sqrt(16 / 3)]
    else:
        actual.loc[index, corruption] = 123
    with pytest.raises(AssertionError):
        assert_source_summary(actual, expected)


def test_source_mapping_refuses_missing_or_duplicate_rows(tmp_path: Path) -> None:
    from tests.scvia.era_source_oracle import read_source_games

    (tmp_path / "one.csv").write_text("year,goals\n2021,4\n2021,\n")
    links = pd.DataFrame([{"source_path": "one.csv", "source_row": 1, "season": 2021,
                           "match_id": "m1", "stage_type": "regular"}])
    with pytest.raises(AssertionError):
        read_source_games(tmp_path, ["one.csv"], links)
    with pytest.raises(pd.errors.MergeError):
        read_source_games(tmp_path, ["one.csv"], pd.concat([links, links]))


@pytest.mark.parametrize("token", ["oops", "NaN", "inf"])
def test_source_oracle_refuses_malformed_nonblank_statistic(token: str) -> None:
    source = source_rows([
        {"match_id": "m1", "year": 2021, "stage_type": "regular", "goals": 4, "kicks": 5},
        {"match_id": "m1", "year": 2021, "stage_type": "regular", "goals": token, "kicks": 7},
    ])
    with pytest.raises(ValueError):
        resolve_source_blanks(source, RECORDED)
