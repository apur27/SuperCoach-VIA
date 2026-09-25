"""Tiny canonical snapshots built with ``SnapshotBuilder`` following ``TABLES``.

Only the columns a test cares about need to be given; every other column is filled with a
schema-valid default (null when nullable). Partitioned tables are split by season exactly
as the importer does, so analytics read them through the same manifest path.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa

from supercoach_via.domain.schemas import TABLES, DatasetStatus, SnapshotManifest
from supercoach_via.storage.snapshots import SnapshotBuilder

FIXED = datetime(2026, 9, 24, 0, 0, tzinfo=UTC)

_DEFAULTS: dict[str, Any] = {
    "provenance": "legacy_import",
    "birth_date_quality": "unknown",
    "identity_status": "canonical",
    "date_quality": "fixture_verified",
    "date_precision": "day",
    "status": "complete",
    "replay_occurrence": 0,
    "revision_id": "r0",
    "stage_type": "regular",
    "active": True,
    "matches_complete": 0,
    "matches_scheduled": 0,
    "source_type": "legacy",
    "confidence": "high",
    "classifier_version": "v0",
    "role": "played",
}


_TYPE_DEFAULTS: dict[str, Any] = {
    "string": "unknown",
    "json": "[]",
    "int32": 0,
    "int64": 0,
    "float64": 0.0,
    "bool": False,
    "date32": date(1900, 1, 1),
    "timestamp_utc": FIXED,
}


def table(name: str, rows: list[dict[str, Any]]) -> pa.Table:
    spec = TABLES[name]
    schema = spec.arrow_schema()
    unknown = {k for r in rows for k in r} - set(spec.column_names)
    if unknown:
        raise KeyError(f"{name}: unknown columns {sorted(unknown)}")
    cols: dict[str, list[Any]] = {}
    for col in spec.columns:
        vals = []
        for r in rows:
            if col.name in r:
                vals.append(r[col.name])
            elif col.nullable:
                vals.append(None)
            elif col.name in _DEFAULTS:
                vals.append(_DEFAULTS[col.name])
            else:  # schema-valid placeholder for required columns a test does not use
                vals.append(_TYPE_DEFAULTS[col.type])
        cols[col.name] = vals
    return pa.table(cols, schema=schema)


def build(root: Path, tables: dict[str, list[dict[str, Any]]]) -> SnapshotManifest:
    builder = SnapshotBuilder(root, clock=lambda: FIXED, code_version="test")
    for name, rows in tables.items():
        t = table(name, rows)
        part = TABLES[name].partition_by
        if part and rows:
            builder.add_partitioned(name, t, part)
        else:
            builder.add(name, t)
    return builder.finish(status=DatasetStatus.LEGACY_UNVERIFIED).manifest


def match(
    match_id: str,
    season: int,
    rnd: int | None,
    home: str,
    away: str,
    hs: int | None,
    as_: int | None,
    *,
    day: date | None = None,
    stage_label: str | None = None,
    stage_type: str = "regular",
    status: str = "complete",
    stage_order: int | None = None,
    local_start: str | None = None,
) -> dict[str, Any]:
    label = stage_label or str(rnd)
    return {
        "match_id": match_id,
        "season": season,
        "stage_label": label,
        "stage_type": stage_type,
        "round_number": rnd,
        "stage_order": stage_order if stage_order is not None else (rnd or 100),
        "stage_id": label.lower().replace(" ", "_") if rnd is None else f"r{rnd:02d}",
        "home_club_id": home,
        "away_club_id": away,
        "home_source_name": home,
        "away_source_name": away,
        "match_date": day,
        "local_start": local_start,
        "status": status,
        "home_score": hs,
        "away_score": as_,
        "home_final_goals": None if hs is None else hs // 6,
        "home_final_behinds": None if hs is None else hs % 6,
        "away_final_goals": None if as_ is None else as_ // 6,
        "away_final_behinds": None if as_ is None else as_ % 6,
    }


def pg(
    match_id: str,
    player_id: str,
    club_id: str,
    season: int,
    counter: int | None = None,
    **stats: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "match_id": match_id,
        "player_id": player_id,
        "club_id": club_id,
        "season": season,
        "stage_label": stats.pop("stage_label", "1"),
        "stage_id": stats.pop("stage_id", "r01"),
        "club_source_name": club_id,
        "career_game_counter": counter,
    }
    row.update(stats)
    return row


def player(player_id: str, name: str, **extra: Any) -> dict[str, Any]:
    first, _, last = name.partition(" ")
    return {
        "player_id": player_id,
        "legacy_slug": player_id.removeprefix("legacy:"),
        "display_name": name,
        "first_name": first,
        "last_name": last,
        **extra,
    }


def club(club_id: str, name: str) -> dict[str, Any]:
    return {"club_id": club_id, "name": name, "lineage_id": club_id, "active": True}


# ---------------------------------------------------------------------------
# Paired legacy-CSV / canonical corpus for ranking parity tests
# ---------------------------------------------------------------------------

LEGACY_STAT_ORDER = (
    "kicks", "marks", "handballs", "disposals", "goals", "behinds", "hit_outs", "tackles",
    "rebound_50s", "inside_50s", "clearances", "clangers", "free_kicks_for",
    "free_kicks_against", "brownlow_votes", "contested_possessions", "uncontested_possessions",
    "contested_marks", "marks_inside_50", "one_percenters", "bounces", "goal_assist",
    "percentage_of_game_played",
)
_TO_CANON = {
    "hit_outs": "hitouts",
    "free_kicks_for": "frees_for",
    "free_kicks_against": "frees_against",
    "goal_assist": "goal_assists",
    "percentage_of_game_played": "time_on_ground_pct",
}


def ranking_corpus(seed: int = 7, n_players: int = 130) -> list[dict[str, Any]]:
    """Deterministic pseudo-random careers across all four legacy_v1 eras."""
    import random

    rng = random.Random(seed)
    seasons = [1960, 1961, 1980, 2005, 2015, 2025, 2026]
    corpus = []
    for i in range(n_players):
        slug = f"p{i:03d}_x_0101{1930 + i % 70}"
        chosen = sorted(rng.sample(seasons, k=rng.randint(1, 4)))
        rows = []
        counter = rng.randint(0, 150)
        for season in chosen:
            for g in range(rng.randint(2, 6)):
                counter += 1
                stats: dict[str, int | None] = {}
                for s in LEGACY_STAT_ORDER:
                    stats[s] = None if rng.random() < 0.15 else rng.randint(0, 12)
                stats["goals"] = None if rng.random() < 0.1 else rng.choice([0, 0, 1, 2, 3, 5, 8])
                rows.append(
                    {
                        "season": season,
                        "match_id": f"m{season}_{g}_{i % 9}",
                        "club": f"C{i % 9}",
                        "counter": counter,
                        "stats": stats,
                    }
                )
        # a deliberate counter lead (missing rows) for some careers
        if i % 5 == 0:
            rows[-1]["counter"] = counter + rng.randint(1, 3)
        corpus.append({"slug": slug, "rows": rows})
    return corpus


def write_legacy_csvs(directory: Path, corpus: list[dict[str, Any]]) -> None:
    import csv

    directory.mkdir(parents=True, exist_ok=True)
    header = ["team", "year", "games_played", "opponent", "round", "result", "jersey_num",
              *LEGACY_STAT_ORDER, "date"]
    for p in corpus:
        path = directory / f"{p['slug']}_performance_details.csv"
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for r in p["rows"]:
                stats = [("" if r["stats"][s] is None else r["stats"][s]) for s in LEGACY_STAT_ORDER]
                w.writerow([r["club"], r["season"], r["counter"], "X", 1, "W", 1, *stats,
                            f"{r['season']}-04-01"])


def canonical_ranking_tables(corpus: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    players, games = [], []
    for p in corpus:
        pid = f"legacy:{p['slug']}"
        players.append(player(pid, p["slug"]))
        for r in p["rows"]:
            stats = {_TO_CANON.get(k, k): v for k, v in r["stats"].items()}
            if stats.get("time_on_ground_pct") is not None:
                stats["time_on_ground_pct"] = float(stats["time_on_ground_pct"])
            games.append(pg(r["match_id"], pid, r["club"], r["season"], r["counter"], **stats))
    return {"players": players, "player_games": games}
