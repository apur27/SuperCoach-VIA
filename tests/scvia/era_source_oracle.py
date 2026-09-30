"""Test-only era expectations from raw source cells and annotated blank semantics.

No production blank resolver, analytics, canonical statistic values or reported counts are
used. Snapshot provenance supplies only the accepted row-to-match mapping. The rules are
anchored by fixtures/zero_semantics/annotations.json and hand-calculated unit cases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from supercoach_via.domain.schemas import LEGACY_PLAYER_COLUMN_MAP, PLAYER_STAT_COLUMNS

CANON_TO_LEGACY = {v: k for k, v in LEGACY_PLAYER_COLUMN_MAP.items()}
METRICS = tuple(CANON_TO_LEGACY.get(s, s) for s in PLAYER_STAT_COLUMNS)
BOUNDS = (("pre-1965", 1897, 1964), ("1965-1990", 1965, 1990),
          ("1991-2010", 1991, 2010), ("2011-present", 2011, 2026))
TOG = "percentage_of_game_played"


def read_source_games(repo: Path, paths: Sequence[str], links: pd.DataFrame) -> pd.DataFrame:
    """Read original cells and require complete, unique provenance on both sides."""
    frames = []
    for path in paths:
        frame = pd.read_csv(repo / path, usecols=lambda c: c in {*METRICS, "year"},
                            dtype=str, keep_default_na=False, na_values=[""], low_memory=False)
        for column in ("year", *METRICS):
            frame[column] = _source_numbers(frame[column]) if column in frame else float("nan")
        frame["source_path"] = path
        frame["source_row"] = range(1, len(frame) + 1)
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    mapped = source.merge(links, on=["source_path", "source_row"], how="outer", validate="one_to_one", indicator=True)
    assert mapped["_merge"].eq("both").all(), mapped.loc[mapped["_merge"].ne("both"), ["source_path", "source_row", "_merge"]]
    assert len(mapped) == len(source) == len(links)
    assert mapped["year"].eq(mapped["season"]).all()
    assert mapped["match_id"].notna().all() and mapped["stage_type"].notna().all()
    return mapped.drop(columns="_merge")


def _source_numbers(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    if not np.isfinite(numeric.dropna().astype(float)).all():
        raise ValueError("non-finite statistic in source cells")
    return numeric


def resolve_source_blanks(source: pd.DataFrame, recorded_from: Mapping[str, int]) -> pd.DataFrame:
    """Apply the annotated rules to source observations using pandas match groups."""
    out = source.copy()
    for metric in METRICS:
        out[metric] = _source_numbers(out[metric]) if metric in out else float("nan")
    observed = out[list(METRICS)].notna()
    universal = observed[["kicks", "marks", "handballs", "disposals", TOG]].any(axis=1)
    universal_in_match = universal.groupby(out["match_id"]).transform("any")
    played = observed.drop(columns="brownlow_votes").any(axis=1) | ~universal_in_match
    for canonical in PLAYER_STAT_COLUMNS:
        metric = CANON_TO_LEGACY.get(canonical, canonical)
        if metric == TOG:
            continue
        reported = observed[metric].groupby(out["match_id"]).transform("any")
        eligible = played & reported & out["year"].ge(recorded_from.get(canonical, 0))
        if metric == "brownlow_votes":
            eligible &= out["stage_type"].ne("final")
        out.loc[~observed[metric] & eligible, metric] = 0
    return out


def source_era_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Pandas statistics over independently resolved source values (all denominators)."""
    rows = []
    latest = int(games["year"].max())
    for era, lo, hi in BOUNDS:
        sub = games[games["year"].between(lo, max(hi, latest) if era == "2011-present" else hi)]
        for metric in METRICS:
            vals = sub[metric].dropna()
            pct = sub.loc[vals.index, TOG]
            normalized = vals if metric == TOG else vals[pct.ge(25)] * (100 / pct[pct.ge(25)])
            rows.append({"era": era, "metric": metric, "n_player_games": len(sub), "n_with_metric": len(vals),
                         "mean_per_game": vals.mean(), "std_per_game": vals.std(), "median_per_game": vals.median(),
                         "mean_per_100pct_played": normalized.mean()})
    return pd.DataFrame(rows)


def assert_source_summary(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Compare every row, with exact counts and all four floating statistics checked."""
    actual = actual.set_index(["era", "legacy_metric"]).sort_index()
    expected = expected.set_index(["era", "metric"]).sort_index()
    assert actual.index.tolist() == expected.index.tolist()
    for column in ("n_player_games", "n_with_metric"):
        assert actual[column].eq(expected[column]).all(), column
    for column in ("mean_per_game", "std_per_game", "median_per_game", "mean_per_100pct_played"):
        assert np.allclose(actual[column], expected[column], rtol=1e-9, atol=1e-9, equal_nan=True), column
