"""Read-only adapters for the audited local data (DESIGN sections 2, 5).

* ``LocalSnapshot`` reads the pinned canonical snapshot through the integrity checker's verified
  capture (every fragment is hashed against the manifest before use) and yields typed rows.
* ``LocalLegacy`` reads the legacy performance and personal CSV files as RAW STRINGS with their
  row numbers. A blank stays blank here; the layer's declared representation (a blank counting
  statistic is a zero in a played game) is applied only at comparison time and reported
  separately, so nothing "verifies stored zeros" that the CSV never stored.

Neither adapter uses the production importer, blank resolver or analytics builders.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from supercoach_via.integrity.capture import SnapshotCapture
from supercoach_via.reconciliation.schema import STAT_FIELDS, SnapshotPin

#: legacy performance CSV column -> canonical field (written independently of the importer)
LEGACY_COLUMNS: dict[str, str] = {
    "kicks": "kicks",
    "marks": "marks",
    "handballs": "handballs",
    "disposals": "disposals",
    "goals": "goals",
    "behinds": "behinds",
    "hit_outs": "hitouts",
    "tackles": "tackles",
    "rebound_50s": "rebound_50s",
    "inside_50s": "inside_50s",
    "clearances": "clearances",
    "clangers": "clangers",
    "free_kicks_for": "frees_for",
    "free_kicks_against": "frees_against",
    "brownlow_votes": "brownlow_votes",
    "contested_possessions": "contested_possessions",
    "uncontested_possessions": "uncontested_possessions",
    "contested_marks": "contested_marks",
    "marks_inside_50": "marks_inside_50",
    "one_percenters": "one_percenters",
    "bounces": "bounces",
    "goal_assist": "goal_assists",
    "percentage_of_game_played": "time_on_ground_pct",
}
LEGACY_FIXED = ("team", "year", "games_played", "opponent", "round", "result", "jersey_num", "date")
_FLOAT_INT = re.compile(r"^(0|[1-9][0-9]*)(\.0+)?$")
_PERF_SUFFIX = "_performance_details.csv"
_PERS_SUFFIX = "_personal_details.csv"


class LocalDataError(RuntimeError):
    """Local input cannot be read as pinned (reported as UNKNOWN coverage, never swallowed)."""


@dataclass(frozen=True, slots=True)
class LocalGame:
    layer: str
    player_key: str
    season: int
    club: str
    opponent: str | None
    #: snapshot: the player-row stage label (``"1"``, ``"QF"``); legacy: the CSV ``round`` token
    stage: str
    match_key: str | None
    result: str | None
    jersey: str | None
    counter: int | None
    counter_token: str | None
    match_date: str | None
    #: 23 values in canonical order: int/float/None (snapshot) or int/None/raw-str (legacy; str = unreadable)
    cells: tuple[object, ...]
    #: legacy only: the raw cell text as read (``""`` = blank), for representation reporting
    raw_cells: tuple[str, ...]
    #: where the row lives: ``<fragment sha>#<row>`` or ``<relative csv>#<data row>``
    origin: str
    #: snapshot only: the row's own declaration of how its date was obtained ("fixture_verified" | "inferred")
    date_quality: str | None = None


@dataclass(frozen=True)
class LocalPlayer:
    layer: str
    key: str
    display_name: str
    first_name: str | None
    last_name: str | None
    birth_date: str | None  # ISO date
    birth_quality: str
    identity_status: str
    canonical_player_id: str | None
    source_urls: tuple[str, ...]
    legacy_slug: str | None


def _iso(d: object) -> str | None:
    if d is None:
        return None
    if isinstance(d, date):
        return d.isoformat()
    text = str(d)
    return text[:10] if text else None


class LocalSnapshot:
    def __init__(self, cap: SnapshotCapture) -> None:
        self.cap = cap
        self._partitions = sorted(
            {f.ref.partition for f in cap.fragments if f.table == "player_games" and f.ref.partition is not None}
        )

    @classmethod
    def open(cls, data_root: Path, pin: SnapshotPin) -> LocalSnapshot:
        cap = SnapshotCapture.open(data_root, pin.snapshot_id)
        if cap.problems or cap.manifest is None:
            raise LocalDataError(f"snapshot {pin.snapshot_id}: {cap.problems}")
        bad = [f.entity for f in cap.fragments if not f.verified]
        if bad:
            raise LocalDataError(f"snapshot fragments failed verification: {bad[:3]}")
        have: dict[str, dict[str, str]] = {}
        for f in cap.fragments:
            have.setdefault(f.table, {})[f.ref.partition or ""] = f.ref.sha256
        if have != pin.fragments:
            raise LocalDataError("snapshot fragments differ from the plan's pinned hashes")
        return cls(cap)

    @property
    def seasons(self) -> list[int]:
        return sorted(int(p) for p in self._partitions)

    def _rows(self, table: str, partition: str | None = None) -> list[dict[str, Any]]:
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows: list[dict[str, Any]] = []
        for f in sorted(self.cap.fragments, key=lambda x: (x.ref.partition or "", x.ref.path)):
            if f.table != table or (partition is not None and f.ref.partition != partition):
                continue
            assert f.data is not None
            for i, row in enumerate(pq.read_table(pa.BufferReader(f.data)).to_pylist()):
                row["_origin"] = f"{f.ref.sha256[:16]}#{i}"
                rows.append(row)
        return rows

    def players(self) -> list[LocalPlayer]:
        out = []
        for r in self._rows("players"):
            urls = json.loads(r["source_urls"]) if r.get("source_urls") else []
            out.append(
                LocalPlayer(
                    layer="snapshot",
                    key=r["player_id"],
                    display_name=r["display_name"],
                    first_name=r.get("first_name"),
                    last_name=r.get("last_name"),
                    birth_date=_iso(r.get("birth_date")),
                    birth_quality=r["birth_date_quality"],
                    identity_status=r["identity_status"],
                    canonical_player_id=r.get("canonical_player_id"),
                    source_urls=tuple(sorted(urls)),
                    legacy_slug=r.get("legacy_slug"),
                )
            )
        return sorted(out, key=lambda p: p.key)

    def aliases(self) -> list[dict[str, Any]]:
        return self._rows("player_aliases")

    def season_awards(self) -> list[tuple[str, int, str, str, int]]:
        """(player id, season, club source name, award, value) from ``player_season_awards`` (absent: none)."""
        return [
            (r["player_id"], int(r["season"]), r["club_source_name"], r["award"], int(r["value"]))
            for r in self._rows("player_season_awards")
        ]

    def matches(self) -> list[dict[str, Any]]:
        return self._rows("matches")

    def quarantine(self) -> list[dict[str, Any]]:
        return self._rows("quarantine")

    def games(self, season: int) -> list[LocalGame]:
        out = []
        for r in self._rows("player_games", str(season)):
            counter_token = r.get("career_game_counter_token")
            out.append(
                LocalGame(
                    layer="snapshot",
                    player_key=r["player_id"],
                    season=int(r["season"]),
                    club=r["club_source_name"],
                    opponent=r.get("opponent_source_name"),
                    stage=r["stage_label"],
                    match_key=r["match_id"],
                    result=r.get("result"),
                    jersey=None if r.get("jersey_number") is None else str(r["jersey_number"]),
                    counter=r.get("career_game_counter"),
                    counter_token=None if counter_token is None else str(counter_token),
                    match_date=_iso(r.get("match_date")),
                    cells=tuple(r.get(f) for f in STAT_FIELDS),
                    raw_cells=(),
                    origin=r["_origin"],
                    date_quality=r.get("date_quality"),
                )
            )
        return sorted(out, key=lambda g: (g.match_key or "", g.player_key, g.club, g.origin))

    def drift(self) -> list[str]:
        return self.cap.drift()


@dataclass(frozen=True)
class LegacyPlayer:
    slug: str
    first_name: str | None
    last_name: str | None
    birth_date: str | None
    games: tuple[LocalGame, ...]
    problems: tuple[str, ...]


def _leading_int(token: str) -> int | None:
    """The career counter's digits; a trailing sub-on/off arrow is not part of the count."""
    m = re.match(r"\d+", token)
    return int(m.group(0)) if m else None


def parse_legacy_number(raw: str) -> int | str | None:
    """Strict: ``""`` is blank (None), digits or digits with a ``.0`` suffix are an integer, anything else
    is returned unchanged as the unreadable raw text (never silently coerced or nulled)."""
    if raw == "":
        return None
    if _FLOAT_INT.match(raw):
        return int(raw.split(".", 1)[0])
    return raw


class LocalLegacy:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.player_dir = root / "data" / "player_data"

    def slugs(self) -> list[str]:
        names = [p.name for p in self.player_dir.iterdir() if p.name.endswith(_PERF_SUFFIX)]
        return sorted(n[: -len(_PERF_SUFFIX)] for n in names)

    def _born(self, slug: str) -> tuple[str | None, str | None, str | None, list[str]]:
        path = self.player_dir / f"{slug}{_PERS_SUFFIX}"
        if not path.is_file():
            return None, None, None, ["personal details file missing"]
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != 1:
            return None, None, None, [f"personal details has {len(rows)} rows"]
        r = rows[0]
        born = (r.get("born_date") or "").strip()
        iso: str | None = None
        problems: list[str] = []
        if born:
            m = re.fullmatch(r"(\d{2})-(\d{2})-(\d{4})", born)
            if m:
                try:
                    iso = date(int(m.group(3)), int(m.group(2)), int(m.group(1))).isoformat()
                except ValueError:
                    problems.append(f"born_date {born!r} is not a calendar date")
            else:
                problems.append(f"born_date {born!r} is not DD-MM-YYYY")
        return (r.get("first_name") or "").strip() or None, (r.get("last_name") or "").strip() or None, iso, problems

    def read(self, slug: str) -> LegacyPlayer:
        first, last, born, problems = self._born(slug)
        rel = f"data/player_data/{slug}{_PERF_SUFFIX}"
        games: list[LocalGame] = []
        with (self.player_dir / f"{slug}{_PERF_SUFFIX}").open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []
            unknown = [c for c in header if c not in LEGACY_COLUMNS and c not in LEGACY_FIXED]
            missing = [c for c in LEGACY_FIXED if c not in header]
            if unknown:
                problems.append(f"unexpected columns {unknown}")
            if missing:
                problems.append(f"missing columns {missing}")
            for n, row in enumerate(reader, 1):
                raw = {f: (row.get(c) or "").strip() for c, f in LEGACY_COLUMNS.items()}
                counter_raw = (row.get("games_played") or "").strip()
                games.append(
                    LocalGame(
                        layer="legacy_csv",
                        player_key=slug,
                        season=int(row["year"]) if (row.get("year") or "").strip().isdigit() else -1,
                        club=(row.get("team") or "").strip(),
                        opponent=(row.get("opponent") or "").strip() or None,
                        stage=(row.get("round") or "").strip(),
                        match_key=None,
                        result=(row.get("result") or "").strip() or None,
                        jersey=(row.get("jersey_num") or "").strip() or None,
                        counter=_leading_int(counter_raw),
                        counter_token=counter_raw or None,
                        match_date=(row.get("date") or "").strip() or None,
                        cells=tuple(parse_legacy_number(raw[f]) for f in STAT_FIELDS),
                        raw_cells=tuple(raw[f] for f in STAT_FIELDS),
                        origin=f"{rel}#{n}",
                    )
                )
        return LegacyPlayer(slug, first, last, born, tuple(games), tuple(problems))

    def players(self) -> Iterator[LegacyPlayer]:
        for slug in self.slugs():
            yield self.read(slug)

    def season_awards(self) -> list[tuple[str, int, str, str, int]]:
        """(slug, season, club, award, value) from ``data/awards/*.csv`` (the layer's season-level values)."""
        out: list[tuple[str, int, str, str, int]] = []
        for path in sorted((self.root / "data" / "awards").glob("*.csv")):
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    out.append((row["slug"], int(row["year"]), row["team"], row["award"], int(row["value"])))
        return out


# ---------------------------------------------------------------------------
# Season shards for the legacy layer (derived working data, never an input)
# ---------------------------------------------------------------------------


def game_to_json(g: LocalGame) -> dict[str, Any]:
    return {
        "layer": g.layer,
        "player_key": g.player_key,
        "season": g.season,
        "club": g.club,
        "opponent": g.opponent,
        "stage": g.stage,
        "match_key": g.match_key,
        "result": g.result,
        "jersey": g.jersey,
        "counter": g.counter,
        "counter_token": g.counter_token,
        "match_date": g.match_date,
        "cells": list(g.cells),
        "raw_cells": list(g.raw_cells),
        "origin": g.origin,
        "date_quality": g.date_quality,
    }


def game_from_json(d: dict[str, Any]) -> LocalGame:
    return LocalGame(
        layer=d["layer"],
        player_key=d["player_key"],
        season=d["season"],
        club=d["club"],
        opponent=d["opponent"],
        stage=d["stage"],
        match_key=d["match_key"],
        result=d["result"],
        jersey=d["jersey"],
        counter=d["counter"],
        counter_token=d["counter_token"],
        match_date=d["match_date"],
        cells=tuple(d["cells"]),
        raw_cells=tuple(d["raw_cells"]),
        origin=d["origin"],
        date_quality=d.get("date_quality"),
    )
