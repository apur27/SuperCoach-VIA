"""Parse the captured bodies into cached facts (comparison phase 1; parallel and cache-keyed).

Every usable resource is read from the immutable object store, verified against its recorded
SHA-256, parsed by the independent readers and written to the facts cache. Workers return only a
small header; the comparison loads full facts lazily, season by season. A reader exception is
recorded as an explicit PARSER_ERROR header (that page's facts are then unavailable and reported),
never swallowed into an empty result.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from supercoach_via.reconciliation import discover as D
from supercoach_via.reconciliation import source as R
from supercoach_via.reconciliation.cache import FactsCache
from supercoach_via.reconciliation.schema import Resource

_GAMES = TypeAdapter(list[R.ProfileGame])
PARSE_ERRORS = (ValueError, IndexError, KeyError, AttributeError, TypeError, UnicodeError)


@dataclass(frozen=True)
class ParseTask:
    kind: str
    url: str
    final_url: str
    sha256: str
    objects_root: str
    cache_root: str
    phash: str
    notes_labels: tuple[tuple[str, str], ...]


def object_bytes(root: Path, sha: str) -> bytes | None:
    """The stored body if it exists and still hashes to ``sha`` (a corrupt object is a missing one)."""
    path = root / "objects" / sha[:2] / sha
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    return data if hashlib.sha256(data).hexdigest() == sha else None


def _dump(model: Any) -> bytes:
    text: str = model.model_dump_json()
    return text.encode()


def parse_one(task: ParseTask) -> dict[str, Any]:
    """Parse (or reuse) one resource; returns its header."""
    cache = FactsCache(Path(task.cache_root), task.phash)
    hdr_raw = cache.read(task.kind, task.sha256, "hdr")
    if hdr_raw is not None:
        hdr: dict[str, Any] = json.loads(hdr_raw)
        hdr["cache"] = "hit"
        return hdr
    data = object_bytes(Path(task.objects_root), task.sha256)
    if data is None:
        return {
            "url": task.url,
            "kind": task.kind,
            "problems": ["OBJECT_MISSING: stored body missing or corrupt"],
            "cache": "error",
        }
    try:
        hdr = _parse(task, data, cache)
    except PARSE_ERRORS as exc:
        return {
            "url": task.url,
            "kind": task.kind,
            "problems": [f"PARSER_ERROR: {type(exc).__name__}: {exc}"[:300]],
            "parser_error": True,
            "cache": "error",
        }
    cache.write(task.kind, task.sha256, json.dumps(hdr, sort_keys=True).encode(), "hdr")
    hdr["cache"] = "miss"
    return hdr


def _parse(task: ParseTask, data: bytes, cache: FactsCache) -> dict[str, Any]:
    kind, sha, final = task.kind, task.sha256, task.final_url
    if kind == "profile":
        p = R.read_profile(data, final)
        core = p.model_copy(update={"games": ()})
        cache.write("profile", sha, _dump(core), "core")
        by_season: dict[int, list[R.ProfileGame]] = {}
        for g in p.games:
            by_season.setdefault(g.season, []).append(g)
        for season, games in sorted(by_season.items()):
            cache.write("profile", sha, json.dumps([g.model_dump(mode="json") for g in games]).encode(), f"s{season}")
        return {
            "url": task.url,
            "kind": kind,
            "h1": p.h1,
            "born": p.born,
            "born_raw": p.born_raw,
            "problems": list(p.problems),
            "seasons": sorted(by_season),
            "appearances": [[g.match_url, g.club, g.season, g.rd_token, g.opponent, g.counter] for g in p.games],
            "club_seasons": [[cs.club, cs.season] for cs in p.club_seasons],
        }
    if kind == "match":
        m = R.read_match(data, final)
        cache.write("match", sha, _dump(m), "main")
        teams = sorted(t.name for t in m.teams)
        pts = [t.points[-1] for t in m.teams]
        return {
            "url": task.url,
            "kind": kind,
            "stage_text": m.stage_text,
            "date": m.match_date,
            "teams": teams,
            "drawn": len(pts) == 2 and pts[0] == pts[1],
            "problems": list(m.problems),
        }
    if kind == "season":
        info = D.read_season_page(data, final)
        hdr = {
            "url": task.url,
            "kind": kind,
            "year": info.year,
            "problems": list(info.problems),
            "fixtures": [
                [
                    f.stage_text,
                    f.home,
                    f.away,
                    f.match_date.isoformat() if f.match_date else None,
                    f.game_url,
                    f.has_score,
                ]
                for f in info.fixtures
            ],
        }
        return hdr
    if kind == "letter":
        page = D.read_census_page(data, final)
        return {
            "url": task.url,
            "kind": kind,
            "letter": page.letter,
            "problems": list(page.problems),
            "profiles": [[p.display_name, p.url] for p in page.profiles],
        }
    if kind == "notes":
        notes = R.read_notes(data, dict(task.notes_labels))
        cache.write("notes", sha, _dump(notes), "main")
        return {"url": task.url, "kind": kind, "problems": list(notes.problems)}
    return {"url": task.url, "kind": kind, "problems": [f"UNSUPPORTED_KIND: {kind}"]}


def parse_all(
    resources: list[Resource],
    *,
    objects_root: Path,
    cache_root: Path,
    phash: str,
    notes_labels: dict[str, str],
    workers: int,
) -> dict[str, dict[str, Any]]:
    """Header per usable resource URL, in deterministic order; ``workers`` only changes the wall clock."""
    tasks = [
        ParseTask(
            r.kind,
            r.url,
            r.final_url or r.url,
            r.sha256 or "",
            str(objects_root),
            str(cache_root),
            phash,
            tuple(sorted(notes_labels.items())),
        )
        for r in sorted(resources, key=lambda x: (x.kind, x.url))
        if r.status == "usable" and r.sha256 and r.kind in ("profile", "match", "season", "letter", "notes")
    ]
    if workers <= 1 or len(tasks) < 8:
        headers = [parse_one(t) for t in tasks]
    else:
        # spawn, not fork: workers start from a clean interpreter instead of inheriting the parent's heap
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            headers = list(pool.map(parse_one, tasks, chunksize=32))
    return {t.url: h for t, h in zip(tasks, headers, strict=True)}


def load_core(cache: FactsCache, sha: str) -> R.ProfileFacts | None:
    raw = cache.read("profile", sha, "core")
    return None if raw is None else R.ProfileFacts.model_validate_json(raw)


def load_profile_games(cache: FactsCache, sha: str, season: int) -> list[R.ProfileGame] | None:
    raw = cache.read("profile", sha, f"s{season}")
    if raw is None:
        return None
    return _GAMES.validate_json(raw)


def load_match(cache: FactsCache, sha: str) -> R.MatchFacts | None:
    raw = cache.read("match", sha, "main")
    return None if raw is None else R.MatchFacts.model_validate_json(raw)


def load_notes(cache: FactsCache, sha: str) -> R.NotesFacts | None:
    raw = cache.read("notes", sha, "main")
    return None if raw is None else R.NotesFacts.model_validate_json(raw)
