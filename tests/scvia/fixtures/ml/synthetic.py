"""Synthetic multi-season canonical snapshots for hermetic ML tests.

Builds tables that follow ``domain.schemas.TABLES`` column specs exactly and writes them
through ``storage.snapshots.SnapshotBuilder`` into a caller-owned ``tmp_path``.

Deliberate edge cases (all tiny, deterministic):
- finals (QF/PF/GF) after the home-and-away rounds, including a drawn final + replay;
- a postponed low-number round played after a later round (stage order != date order);
- two matches on the same calendar day (one with minute-precision start + venue tz);
- zero-disposal games, missing (null) stat cells, missing time-on-ground;
- a cold-start player who debuts mid-way through the final season;
- a player whose rows carry ``date_quality='inferred'`` (never time-verified);
- optionally scheduled future fixtures with no player-game rows.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa

from supercoach_via.domain.schemas import (
    PLAYER_STAT_COLUMNS,
    TABLES,
    CheckOutcome,
    DatasetStatus,
    ValidationReport,
)
from supercoach_via.storage import snapshots

FIXED = datetime(2026, 9, 25, 0, 0, tzinfo=UTC)
CLUBS = ("adel", "bris", "carl", "coll")


def _clock() -> datetime:
    return FIXED


def table_from_rows(name: str, rows: list[dict[str, Any]]) -> pa.Table:
    spec = TABLES[name]
    schema = spec.arrow_schema()
    cols: dict[str, list[Any]] = {c: [] for c in spec.column_names}
    for r in rows:
        extra = set(r) - set(cols)
        if extra:
            raise KeyError(f"{name}: unknown columns {sorted(extra)}")
        for c in cols:
            cols[c].append(r.get(c))
    return pa.table(cols, schema=schema)


@dataclass
class Corpus:
    matches: list[dict[str, Any]] = field(default_factory=list)
    player_games: list[dict[str, Any]] = field(default_factory=list)
    players: list[dict[str, Any]] = field(default_factory=list)
    clubs: list[dict[str, Any]] = field(default_factory=list)
    venues: list[dict[str, Any]] = field(default_factory=list)
    lineups: list[dict[str, Any]] = field(default_factory=list)

    def write(self, root: Path) -> Path:
        """Write every table as a snapshot under ``root`` and promote it; return root."""
        builder = snapshots.SnapshotBuilder(root, clock=_clock, code_version="test")
        builder.add("players", table_from_rows("players", self.players))
        builder.add("clubs", table_from_rows("clubs", self.clubs))
        builder.add("venues", table_from_rows("venues", self.venues))
        builder.add_partitioned("matches", table_from_rows("matches", self.matches), "season")
        if self.player_games:
            builder.add_partitioned(
                "player_games", table_from_rows("player_games", self.player_games), "season"
            )
        if self.lineups:
            builder.add_partitioned("lineups", table_from_rows("lineups", self.lineups), "season")
        cand = builder.finish(status=DatasetStatus.DEMO)
        snapshots.promote(root, cand, ValidationReport(outcome=CheckOutcome.PASS), promoted_at=FIXED)
        return root


def _match(
    match_id: str,
    season: int,
    stage_label: str,
    stage_order: int,
    home: str,
    away: str,
    day: date,
    *,
    status: str = "complete",
    local_start: str | None = None,
    venue: str = "v_mcg",
    replay: int = 0,
) -> dict[str, Any]:
    final = not stage_label.isdigit()
    return {
        "match_id": match_id,
        "season": season,
        "stage_label": stage_label,
        "stage_type": "final" if final else "regular",
        "round_number": None if final else int(stage_label),
        "stage_order": stage_order,
        "stage_id": stage_label.lower() if final else f"r{int(stage_label):02d}",
        "replay_occurrence": replay,
        "home_club_id": home,
        "away_club_id": away,
        "home_source_name": home.upper(),
        "away_source_name": away.upper(),
        "venue_id": venue,
        "venue_source_name": venue,
        "local_start": local_start,
        "match_date": day,
        "date_precision": "minute" if local_start else "day",
        "status": status,
        "provenance": "demo",
    }


def _stats(rng: random.Random, level: float) -> dict[str, Any]:
    disp = max(0, round(rng.gauss(level, 4)))
    kicks = rng.randint(0, disp)
    out: dict[str, Any] = {s: rng.randint(0, 6) for s in PLAYER_STAT_COLUMNS}
    out.update(
        disposals=disp,
        kicks=kicks,
        handballs=disp - kicks,
        time_on_ground_pct=float(rng.randint(60, 95)),
        brownlow_votes=None,
    )
    return out


def build_corpus(
    seasons: tuple[int, ...] = (2023, 2024, 2025),
    rounds: int = 6,
    players_per_club: int = 5,
    *,
    future_rounds: int = 0,
    seed: int = 7,
) -> Corpus:
    """Deterministic corpus. Each round: two matches (c0 v c1, c2 v c3), rotated pairings."""
    rng = random.Random(seed)
    corpus = Corpus()
    corpus.clubs = [
        {"club_id": c, "name": c.title(), "lineage_id": c, "active": True} for c in CLUBS
    ]
    corpus.venues = [
        {"venue_id": "v_mcg", "name": "MCG", "source_names": '["M.C.G."]', "timezone": "Australia/Melbourne"},
        {"venue_id": "v_gabba", "name": "Gabba", "source_names": '["Gabba"]', "timezone": None},
    ]
    roster: dict[str, list[str]] = {}
    levels: dict[str, float] = {}
    for ci, club in enumerate(CLUBS):
        roster[club] = []
        for k in range(players_per_club):
            pid = f"legacy:{club}_p{k}"
            roster[club].append(pid)
            levels[pid] = 10 + 4 * k + ci
            corpus.players.append(
                {
                    "player_id": pid,
                    "display_name": f"{club.title()} Player {k}",
                    "birth_date": date(1995 + k, 1 + ci, 10),
                    "birth_date_quality": "source" if k % 2 == 0 else "legacy_filename",
                    "identity_status": "canonical",
                    "provenance": "demo",
                }
            )
    # cold-start debutant (joins adel in the last season, round 4) and an ambiguous identity
    rookie = "legacy:adel_rookie"
    levels[rookie] = 15
    corpus.players.append(
        {"player_id": rookie, "display_name": "Adel Rookie", "birth_date_quality": "unknown",
         "identity_status": "canonical", "provenance": "demo"}
    )
    corpus.players.append(
        {"player_id": "legacy:coll_ambig", "display_name": "Coll Ambiguous",
         "birth_date_quality": "unknown", "identity_status": "ambiguous", "provenance": "demo"}
    )
    levels["legacy:coll_ambig"] = 18
    roster["coll"].append("legacy:coll_ambig")

    last_season = seasons[-1]
    for season in seasons:
        start = date(season, 3, 16)
        order = 0
        schedule: list[tuple[str, int, date, list[tuple[str, str]]]] = []
        for r in range(1, rounds + 1):
            day = start + timedelta(days=7 * (r - 1))
            pairs = [(CLUBS[0], CLUBS[1 + (r % 3)]), tuple(c for c in CLUBS[1:] if c != CLUBS[1 + (r % 3)])]
            schedule.append((str(r), r, day, [(a, b) for a, b in pairs]))  # type: ignore[misc]
        # postponed round 2: played after round 3 (low-number round, later date)
        lbl, so, day, pairs = schedule[1]
        schedule[1] = (lbl, so, schedule[2][2] + timedelta(days=3), pairs)
        fin_day = start + timedelta(days=7 * rounds)
        schedule.append(("QF", rounds + 1, fin_day, [(CLUBS[0], CLUBS[1]), (CLUBS[2], CLUBS[3])]))
        schedule.append(("GF", rounds + 3, fin_day + timedelta(days=14), [(CLUBS[0], CLUBS[2])]))
        for label, stage_order, day, pairs in schedule:
            order = stage_order
            for mi, (home, away) in enumerate(pairs):
                mid = f"m{season}_{label.lower()}_{mi}"
                local = f"{day.isoformat()} {13 + 3 * mi:02d}:10" if mi == 0 else None
                corpus.matches.append(
                    _match(mid, season, label, order, home, away, day, local_start=local,
                           venue="v_mcg" if mi == 0 else "v_gabba")
                )
        # drawn QF replay the following week (same stage, replay_occurrence=1)
        rp_day = fin_day + timedelta(days=7)
        corpus.matches.append(
            _match(f"m{season}_qf_0r", season, "QF", rounds + 2, CLUBS[0], CLUBS[1], rp_day, replay=1)
        )
    # player games for every complete match
    for m in corpus.matches:
        for club, opp in ((m["home_club_id"], m["away_club_id"]), (m["away_club_id"], m["home_club_id"])):
            members = list(roster[club])
            if club == "adel" and m["season"] == last_season and (m["round_number"] or 99) >= 4:
                members.append(rookie)
            for pid in members:
                row = {
                    "match_id": m["match_id"],
                    "player_id": pid,
                    "club_id": club,
                    "season": m["season"],
                    "opponent_club_id": opp,
                    "stage_label": m["stage_label"],
                    "stage_id": m["stage_id"],
                    "club_source_name": club.upper(),
                    "opponent_source_name": opp.upper(),
                    "link_method": "key",
                    "match_date": m["match_date"],
                    "date_quality": "fixture_verified",
                    "revision_id": "r1",
                    "provenance": "demo",
                    **_stats(rng, levels[pid]),
                }
                if pid.endswith("_p1") and club == "bris":
                    row.update(date_quality="inferred", link_method="row_order")
                elif m["season"] < last_season:
                    # legacy-style: synthesized row date, but a deterministic match-key link
                    row.update(date_quality="inferred",
                               match_date=m["match_date"] - timedelta(days=17))
                if pid.endswith("_p2") and club == "carl" and m["stage_label"] == "3":
                    row.update(disposals=0, kicks=0, handballs=0)
                if pid.endswith("_p3") and m["stage_label"] == "5":
                    row.update(tackles=None, time_on_ground_pct=None)
                corpus.player_games.append(row)
    # future scheduled fixtures (no player-game rows)
    if future_rounds:
        fseason = last_season + 1
        base = date(fseason, 3, 15)
        for r in range(1, future_rounds + 1):
            day = base + timedelta(days=7 * (r - 1))
            corpus.matches.append(_match(f"m{fseason}_r{r}_0", fseason, str(r), r, "adel", "carl", day, status="scheduled"))
            corpus.matches.append(_match(f"m{fseason}_r{r}_1", fseason, str(r), r, "bris", "coll", day, status="scheduled", venue="v_gabba"))
    return corpus


def write_corpus(root: Path, **kwargs: Any) -> Path:
    return build_corpus(**kwargs).write(root)
