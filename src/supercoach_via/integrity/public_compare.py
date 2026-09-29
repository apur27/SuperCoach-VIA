"""Public release resources vs canonical facts, one self-contained unit at a time.

A unit is a season (its match index, match details and player season logs) or a group of
players (their detail pages). Each unit receives the canonical rows it needs and the
inventory of the public files it reads, and returns raw findings plus examined counts.
Units are pure functions of their payload, so they can run in worker processes and be
cached by content identity (``integrity.cache``).

Expected values are derived here from the canonical rows, independently of the release
builder: ordering, compact stat columns, shared match facts and observed-denominator
aggregates follow the published contract, not the builder's code.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from supercoach_via.domain.schemas import PLAYER_STAT_COLUMNS
from supercoach_via.integrity.capture import strict_json
from supercoach_via.integrity.report import Collector, Finding, RuleSpec

_CANON_INDEX = {s: i for i, s in enumerate(PLAYER_STAT_COLUMNS)}
#: float columns summed in float64; totals may differ in the last place with summation order
_FLOAT_STATS = frozenset({"time_on_ground_pct"})
_FLOAT_REL_TOL = 1e-9


@dataclass
class UnitResult:
    unit_id: str
    findings: list[Finding] = field(default_factory=list)
    examined: dict[str, int] = field(default_factory=dict)
    expected_paths: list[str] = field(default_factory=list)


class _Out:
    def __init__(self, rules: dict[str, RuleSpec], current_season: int | None, unit_id: str):
        self.col = Collector(rules, sample_limit=1, current_season=current_season, keep_all=True)
        self.res = UnitResult(unit_id)

    def add(self, rule_id: str, entity: str, **kw: Any) -> None:
        self.col.add(rule_id, entity, **kw)

    def count(self, key: str, n: int = 1) -> None:
        self.res.examined[key] = self.res.examined.get(key, 0) + n

    def done(self) -> UnitResult:
        self.res.findings = self.col.all_findings()
        return self.res


def _list(v: Any) -> list[Any]:
    """A JSON array, or an empty list for anything else (the schema checks report the shape)."""
    return v if isinstance(v, list) else []


def _obj(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _iso(v: Any) -> Any:
    return v.isoformat() if isinstance(v, date) else v


def _num(v: Any) -> Any:
    """A canonical value as the public JSON carries it (integral floats become ints)."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return v
        return int(v) if v.is_integer() else v
    return v


def _equal(want: Any, have: Any) -> bool:
    """JSON equality that keeps booleans apart from numbers (Python's ``True == 1``)."""
    if isinstance(want, bool) or isinstance(have, bool):
        return type(want) is type(have) and want == have
    if isinstance(want, int | float) and isinstance(have, int | float):
        return float(want) == float(have)
    if isinstance(want, list) and isinstance(have, list):
        return len(want) == len(have) and all(_equal(a, b) for a, b in zip(want, have, strict=True))
    if isinstance(want, dict) and isinstance(have, dict):
        return want.keys() == have.keys() and all(_equal(v, have[k]) for k, v in want.items())
    return bool(want == have)


def _same_number(want: Any, have: Any, stat: str | None = None) -> bool:
    if want is None or have is None:
        return want is None and have is None
    if isinstance(have, bool) or not isinstance(have, int | float):
        return False
    if stat in _FLOAT_STATS:
        return abs(float(want) - float(have)) <= _FLOAT_REL_TOL * max(1.0, abs(float(want)))
    return float(want) == float(have)


def public_key(identifier: str) -> str:
    from supercoach_via.publish.resources import public_key as codec

    return codec(identifier)


def normalise_search(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return " ".join("".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower().split())


def _read(out: _Out, release_dir: Path, inventory: dict[str, tuple[str, int]], rel: str) -> Any:
    """Parse one public file after checking it still has the inventoried bytes."""
    info = inventory.get(rel)
    if info is None:
        return None
    path = release_dir / "public" / rel
    try:
        with path.open("rb") as fh:
            data = fh.read()
    except OSError:
        data = b""
    if len(data) != info[1] or hashlib.sha256(data).hexdigest() != info[0]:
        out.add("release.changed_during_audit", f"resource:{rel}", evidence={"sha256": info[0]})
        return None
    out.count("resources")
    try:
        return strict_json(data)
    except (ValueError, UnicodeDecodeError) as exc:
        rid = "release.json_duplicate_key" if "duplicate JSON key" in str(exc) else "release.json_invalid"
        out.add(rid, f"resource:{rel}", message=str(exc)[:300])
        return None


# ---------------------------------------------------------------------------
# Season unit: match index, match details, player season logs
# ---------------------------------------------------------------------------


def _order_key(m: dict[str, Any]) -> tuple[Any, ...]:
    d, ls = m["match_date"], m["local_start"]
    return (d is None, d or date.min, ls is None, ls or "", m["stage_order"], m["match_id"])


def expected_summary(m: dict[str, Any], club_names: dict[str, str]) -> dict[str, Any]:
    def team(side: str) -> dict[str, Any]:
        cid = m[f"{side}_club_id"]
        return {
            "club_id": cid,
            "name": club_names.get(cid, m[f"{side}_source_name"]),
            "goals": m[f"{side}_final_goals"],
            "behinds": m[f"{side}_final_behinds"],
            "score": m[f"{side}_score"],
        }

    home, away = team("home"), team("away")
    winner = None
    if (
        m["status"] == "complete"
        and home["score"] is not None
        and away["score"] is not None
        and home["score"] != away["score"]
    ):
        winner = home["club_id"] if home["score"] > away["score"] else away["club_id"]
    return {
        "match_id": m["match_id"],
        "season": m["season"],
        "stage_id": m["stage_id"],
        "stage_label": m["stage_label"],
        "stage_type": m["stage_type"],
        "round_number": m["round_number"],
        "stage_order": m["stage_order"],
        "replay_occurrence": m["replay_occurrence"],
        "local_start": m["local_start"],
        "match_date": _iso(m["match_date"]),
        "date_precision": m["date_precision"],
        "status": m["status"],
        "venue": m["venue_source_name"],
        "home": home,
        "away": away,
        "winner_club_id": winner,
    }


def _compact(rows: list[dict[str, Any]]) -> list[str]:
    return [s for s in PLAYER_STAT_COLUMNS if any(r[s] is not None for r in rows)]


def _source_label(m: dict[str, Any]) -> dict[str, Any]:
    path = m.get("source_path")
    note = path if isinstance(path, str) else None
    if m.get("provenance") == "source_fetch":
        url = note if note and note.startswith(("http://", "https://")) else None
        return {"label": "AFL Tables source page", "url": url, "note": note}
    return {"label": "Legacy match/player CSV import", "url": None, "note": note}


def _diff_dict(
    out: _Out, rule_id: str, entity: str, want: dict[str, Any], have: Any, season: int, prefix: str = "", **ev: Any
) -> None:
    if not isinstance(have, dict):
        out.add(rule_id, entity, field=prefix or None, season=season, expected="object", actual=type(have).__name__)
        return
    for k, w in want.items():
        h = have.get(k)
        if isinstance(w, dict):
            _diff_dict(out, rule_id, entity, w, h, season, f"{prefix}{k}.", **ev)
        elif not _equal(w, h):
            out.add(rule_id, entity, field=f"{prefix}{k}", season=season, expected=w, actual=h, evidence=ev or None)
    out.count("cells", len(want))


def _cells(out: _Out, entity: str, season: int, cols: list[str], values: Any, canon: dict[str, Any], rel: str) -> None:
    if not isinstance(values, list) or len(values) != len(cols):
        out.add(
            "release.compact_row_width",
            entity,
            season=season,
            expected=len(cols),
            actual=len(values) if isinstance(values, list) else None,
            evidence={"resource": rel},
        )
        return
    out.count("cells", len(cols))
    for stat, v in zip(cols, values, strict=True):
        want = canon.get(stat) if stat in _CANON_INDEX else v
        if want == v and type(want) is type(v):  # fast path: identical int/None/float
            continue
        if stat not in _CANON_INDEX:
            continue
        want = _num(want)
        if not _same_number(want, v, stat):
            out.add(
                "release.cell_mismatch",
                entity,
                table="player_games",
                field=stat,
                season=season,
                expected=want,
                actual=v,
                evidence={"resource": rel},
            )


def _check_columns(out: _Out, rel: str, season: int, want: list[str], have: Any) -> list[str] | None:
    if not isinstance(have, list) or not all(isinstance(c, str) for c in have):
        out.add("release.stat_columns", f"resource:{rel}", season=season, expected=want, actual=have)
        return None
    bad = [c for c in have if c not in _CANON_INDEX]
    ordered = sorted(have, key=lambda c: _CANON_INDEX.get(c, 999))
    if bad or ordered != have or len(set(have)) != len(have) or have != want:
        out.add("release.stat_columns", f"resource:{rel}", season=season, expected=want, actual=have)
    return have


def season_unit(p: dict[str, Any]) -> UnitResult:
    """Compare one season's match index, match details and player season logs."""
    season: int = p["season"]
    out = _Out(p["rules"], p["current_season"], f"season:{season}")
    release_dir: Path = p["release_dir"]
    inv: dict[str, tuple[str, int]] = p["inventory"]
    clubs: dict[str, str] = p["club_names"]
    names: dict[str, str] = p["player_names"]
    matches = sorted(p["matches"].to_pylist(), key=_order_key)
    games: list[dict[str, Any]] = p["player_games"].to_pylist()
    by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for g in games:
        by_match[g["match_id"]].append(g)
    expected_paths: list[str] = []

    # -- match index --------------------------------------------------------
    index_rel = f"matches/{season}/index.json"
    expected_paths.append(index_rel)
    summaries = {m["match_id"]: expected_summary(m, clubs) for m in matches}
    doc = _read(out, release_dir, inv, index_rel)
    if index_rel not in inv:
        out.add("release.resource_missing", f"resource:{index_rel}", season=season)
    elif isinstance(doc, dict):
        have = _list(doc.get("matches"))
        have_ids = [m.get("match_id") for m in have if isinstance(m, dict)]
        want_ids = [m["match_id"] for m in matches]
        for mid in sorted(set(want_ids) - set(have_ids)):
            out.add(
                "release.index_membership",
                f"match:{mid}",
                season=season,
                expected="listed",
                actual="absent",
                evidence={"resource": index_rel},
            )
        for mid in sorted({str(x) for x in have_ids} - set(want_ids)):
            out.add(
                "release.index_membership",
                f"match:{mid}",
                season=season,
                expected="absent",
                actual="listed",
                evidence={"resource": index_rel},
            )
        common = [x for x in have_ids if x in summaries]
        if common != [x for x in want_ids if x in set(have_ids)]:
            out.add("release.index_order", f"resource:{index_rel}", season=season)
        for m in have:
            if isinstance(m, dict) and m.get("match_id") in summaries:
                _diff_dict(
                    out,
                    "release.match_summary_mismatch",
                    f"match:{m['match_id']}",
                    summaries[m["match_id"]],
                    m,
                    season,
                    resource=index_rel,
                )
    index_ids = set(summaries)

    # -- match details ------------------------------------------------------
    for m in matches:
        mid = m["match_id"]
        rel = f"matches/detail/{public_key(mid)}.json"
        expected_paths.append(rel)
        if rel not in inv:
            out.add("release.resource_missing", f"resource:{rel}", season=season, evidence={"match_id": mid})
            continue
        d = _read(out, release_dir, inv, rel)
        if not isinstance(d, dict):
            continue
        rows = by_match.get(mid, [])
        _diff_dict(
            out,
            "release.match_summary_mismatch",
            f"match:{mid}",
            summaries[mid],
            d.get("summary"),
            season,
            resource=rel,
        )
        quarters = [
            {
                "quarter": q,
                "home_goals": m[f"home_{q}_goals"],
                "home_behinds": m[f"home_{q}_behinds"],
                "away_goals": m[f"away_{q}_goals"],
                "away_behinds": m[f"away_{q}_behinds"],
            }
            for q in ("q1", "q2", "q3", "final")
        ]
        if d.get("quarters") != quarters:
            out.add(
                "release.match_summary_mismatch",
                f"match:{mid}",
                field="quarters",
                season=season,
                expected=quarters,
                actual=d.get("quarters"),
                evidence={"resource": rel},
            )
        if d.get("attendance") != m["attendance"]:
            out.add(
                "release.match_summary_mismatch",
                f"match:{mid}",
                field="attendance",
                season=season,
                expected=m["attendance"],
                actual=d.get("attendance"),
                evidence={"resource": rel},
            )
        if d.get("sources") != [_source_label(m)]:
            out.add(
                "release.source_label",
                f"match:{mid}",
                field="sources",
                season=season,
                expected=[_source_label(m)],
                actual=d.get("sources"),
                evidence={"resource": rel},
            )
        cols = _check_columns(out, rel, season, _compact(rows), d.get("stat_columns"))
        if cols is None:
            continue
        for side in ("home", "away"):
            club = m[f"{side}_club_id"]
            side_rows = sorted(
                (g for g in rows if g["club_id"] == club), key=lambda g: (names.get(g["player_id"], ""), g["player_id"])
            )
            box = d.get(f"{side}_players") or {}
            pids = _list(box.get("player_id"))
            want_ids = [g["player_id"] for g in side_rows]
            if pids != want_ids:
                for pid in sorted(set(want_ids) - set(pids)):
                    out.add(
                        "release.box_membership",
                        f"player_game:{mid}|{pid}|{club}",
                        season=season,
                        expected="listed",
                        actual="absent",
                        evidence={"resource": rel},
                    )
                for pid in sorted({str(x) for x in pids} - set(want_ids)):
                    out.add(
                        "release.box_membership",
                        f"player_game:{mid}|{pid}|{club}",
                        season=season,
                        expected="absent",
                        actual="listed",
                        evidence={"resource": rel},
                    )
                if set(pids) == set(want_ids):
                    out.add("release.box_order", f"match:{mid}|{club}", season=season, evidence={"resource": rel})
            stats = _list(box.get("stats"))
            names_col = _list(box.get("name"))
            by_pid = {g["player_id"]: g for g in side_rows}
            for i, pid in enumerate(pids):
                hit = by_pid.get(pid)
                if hit is None:
                    continue
                entity = f"player_game:{mid}|{pid}|{club}"
                if i < len(names_col) and names_col[i] != names.get(pid):
                    out.add(
                        "release.box_membership",
                        entity,
                        field="name",
                        season=season,
                        expected=names.get(pid),
                        actual=names_col[i],
                        evidence={"resource": rel},
                    )
                _cells(out, entity, season, cols, stats[i] if i < len(stats) else None, hit, rel)
        out.count("match_details")

    # -- player season logs -------------------------------------------------
    by_player: dict[str, list[dict[str, Any]]] = defaultdict(list)
    match_date = {m["match_id"]: m["match_date"] for m in matches}
    match_start = {m["match_id"]: m["local_start"] for m in matches}
    for g in games:
        by_player[g["player_id"]].append(g)
    for pid, prows in sorted(by_player.items()):
        rel = f"player-games/{public_key(pid)}/{season}.json"
        expected_paths.append(rel)
        if rel not in inv:
            out.add("release.resource_missing", f"resource:{rel}", season=season, evidence={"player_id": pid})
            continue
        doc = _read(out, release_dir, inv, rel)
        if not isinstance(doc, dict):
            continue

        def okey(g: dict[str, Any]) -> tuple[Any, ...]:
            ev = match_date.get(g["match_id"]) or g["match_date"]
            ls = match_start.get(g["match_id"])
            return (ev is None, ev or date.min, ls is None, ls or "", g["match_id"])

        prows = sorted(prows, key=okey)
        if doc.get("player_id") != pid or doc.get("season") != season:
            out.add(
                "release.log_value_mismatch",
                f"resource:{rel}",
                field="player_id",
                season=season,
                expected=[pid, season],
                actual=[doc.get("player_id"), doc.get("season")],
            )
        cols = _check_columns(out, rel, season, _compact(prows), doc.get("stat_columns"))
        gm = _obj(doc.get("games"))
        ids = _list(gm.get("match_id"))
        want_ids = [g["match_id"] for g in prows]
        if ids != want_ids:
            for mid in sorted(set(want_ids) - {str(x) for x in ids}):
                out.add(
                    "release.log_membership",
                    f"player_game:{mid}|{pid}",
                    season=season,
                    expected="listed",
                    actual="absent",
                    evidence={"resource": rel},
                )
            for mid in sorted({str(x) for x in ids} - set(want_ids)):
                out.add(
                    "release.log_membership",
                    f"player_game:{mid}|{pid}",
                    season=season,
                    expected="absent",
                    actual="listed",
                    evidence={"resource": rel},
                )
            if sorted(ids) == sorted(want_ids):
                out.add("release.log_order", f"resource:{rel}", season=season)
        facts = doc.get("match_facts")
        shared = ("match_date", "stage_label", "opponent_club_id", "opponent_name")
        if facts is not None:
            if facts != index_rel or index_rel not in inv:
                out.add(
                    "release.shared_facts",
                    f"resource:{rel}",
                    field="match_facts",
                    season=season,
                    expected=index_rel,
                    actual=facts,
                )
            if any(gm.get(f) for f in shared):
                out.add(
                    "release.shared_facts",
                    f"resource:{rel}",
                    field="shared arrays",
                    season=season,
                    expected="empty when match_facts is set",
                    actual={f: len(gm.get(f) or []) for f in shared},
                )
            for mid in sorted({str(x) for x in ids} - index_ids):
                out.add(
                    "release.shared_facts",
                    f"player_game:{mid}|{pid}",
                    field="match_id",
                    season=season,
                    expected=f"listed in {index_rel}",
                    actual="absent",
                )
        by_mid = {g["match_id"]: g for g in prows}
        stats = _list(gm.get("stats"))
        for i, mid in enumerate(ids):
            hit = by_mid.get(mid)
            if hit is None:
                continue
            entity = f"player_game:{mid}|{pid}|{hit['club_id']}"
            event = match_date.get(mid)
            row_date = hit["match_date"]
            want = {
                "club_id": hit["club_id"],
                "result": hit["result"],
                "career_game_counter": hit["career_game_counter"],
                "date_quality": (hit["date_quality"] if row_date == event else "source")
                if event is not None
                else hit["date_quality"],
            }
            if facts is None:
                want["match_date"] = _iso(event if event is not None else row_date)
            for k, w in want.items():
                col = _list(gm.get(k))
                h = col[i] if i < len(col) else None
                out.count("cells")
                if not _equal(w, h):
                    out.add(
                        "release.log_value_mismatch",
                        entity,
                        field=k,
                        season=season,
                        expected=w,
                        actual=h,
                        evidence={"resource": rel},
                    )
            if cols is not None:
                _cells(out, entity, season, cols, stats[i] if i < len(stats) else None, hit, rel)
        out.count("player_logs")
    out.res.expected_paths = expected_paths
    return out.done()


# ---------------------------------------------------------------------------
# Player units: index and detail pages (career and season aggregates)
# ---------------------------------------------------------------------------


def expected_index_entry(pid: str, name: str, agg: dict[str, Any], latest: int | None) -> dict[str, Any]:
    clubs = sorted(agg.get("club_names") or [])
    last = agg.get("last_season")
    return {
        "id": pid,
        "key": public_key(pid),
        "name": name,
        "clubs": clubs,
        "first_season": agg.get("first_season"),
        "last_season": last,
        "seasons": sorted(agg.get("seasons") or []),
        "games": int(agg.get("games") or 0),
        "active": last is not None and last == latest,
        "search": normalise_search(f"{name} {' '.join(clubs)}"),
    }


def index_unit(p: dict[str, Any]) -> UnitResult:
    out = _Out(p["rules"], p["current_season"], "players:index")
    rel = "players/index.json"
    inv = p["inventory"]
    out.res.expected_paths = [rel]
    if rel not in inv:
        out.add("release.resource_missing", f"resource:{rel}")
        return out.done()
    doc = _read(out, p["release_dir"], inv, rel)
    if not isinstance(doc, dict):
        return out.done()
    want = [expected_index_entry(pid, name, agg, p["latest"]) for pid, name, agg in p["players"]]
    have = _list(doc.get("players"))
    if doc.get("count") != len(have) or len(have) != len(want):
        out.add(
            "release.player_index_value", f"resource:{rel}", field="count", expected=len(want), actual=doc.get("count")
        )
    want_by = {w["id"]: w for w in want}
    have_by = {h.get("id"): h for h in have if isinstance(h, dict)}
    for pid in sorted(set(want_by) - set(have_by)):
        out.add(
            "release.index_membership", f"player:{pid}", expected="listed", actual="absent", evidence={"resource": rel}
        )
    for pid in sorted({str(x) for x in have_by} - set(want_by)):
        out.add(
            "release.index_membership", f"player:{pid}", expected="absent", actual="listed", evidence={"resource": rel}
        )
    order_have = [h.get("id") for h in have if isinstance(h, dict) and h.get("id") in want_by]
    if order_have != [w["id"] for w in want if w["id"] in have_by]:
        out.add("release.index_order", f"resource:{rel}")
    for pid, w in want_by.items():
        h = have_by.get(pid)
        if h is None:
            continue
        for k, v in w.items():
            out.count("cells")
            if not _equal(v, h.get(k)):
                out.add(
                    "release.player_index_value",
                    f"player:{pid}",
                    field=k,
                    expected=v,
                    actual=h.get(k),
                    season=None,
                    evidence={"resource": rel},
                )
    out.count("player_index_entries", len(have))
    return out.done()


def detail_unit(p: dict[str, Any]) -> UnitResult:
    """Compare a group of player detail pages with aggregates recomputed from the facts."""
    out = _Out(p["rules"], p["current_season"], p["unit_id"])
    inv = p["inventory"]
    stat_names = sorted(PLAYER_STAT_COLUMNS)
    for pid, person, career, seasons in p["players"]:
        rel = f"players/{public_key(pid)}.json"
        out.res.expected_paths.append(rel)
        if rel not in inv:
            out.add("release.resource_missing", f"resource:{rel}", evidence={"player_id": pid})
            continue
        d = _read(out, p["release_dir"], inv, rel)
        if not isinstance(d, dict):
            continue
        entity = f"player:{pid}"
        head = {
            "id": pid,
            "key": public_key(pid),
            "name": person["display_name"],
            "identity_status": person["identity_status"],
            "birth_date": _iso(person["birth_date"]),
            "birth_date_quality": person["birth_date_quality"],
            "career_games": career["career_games"],
            "career_counter_max": career["counter_max"],
            # a canonical identity with no fact rows has no statistics at all, not 23 empty ones
            "stat_names": stat_names if career["stats"] else [],
        }
        for k, v in head.items():
            out.count("cells")
            if not _equal(v, d.get(k)):
                out.add(
                    "release.aggregate_mismatch",
                    entity,
                    field=k,
                    expected=v,
                    actual=d.get(k),
                    evidence={"resource": rel},
                )
        names = _list(d.get("stat_names"))
        _stat_block(out, entity, rel, "career", names, d.get("career"), career["stats"], None)
        have_seasons = {s.get("season"): s for s in d.get("seasons") or [] if isinstance(s, dict)}
        for sv in sorted(set(seasons) - set(have_seasons)):
            out.add(
                "release.aggregate_mismatch",
                entity,
                field=f"seasons.{sv}",
                expected="listed",
                actual="absent",
                evidence={"resource": rel},
            )
        for sv in sorted({x for x in have_seasons if x not in seasons}, key=str):
            out.add(
                "release.aggregate_mismatch",
                entity,
                field=f"seasons.{sv}",
                expected="absent",
                actual="listed",
                evidence={"resource": rel},
            )
        listed = [x for x in have_seasons if isinstance(x, int)]
        if listed != sorted(listed):
            out.add("release.aggregate_mismatch", entity, field="seasons.order", evidence={"resource": rel})
        for sv, sw in sorted(seasons.items()):
            h = have_seasons.get(sv)
            if h is None:
                continue
            for k, v in (
                ("games", sw["games"]),
                ("clubs", sw["clubs"]),
                ("games_resource", f"player-games/{public_key(pid)}/{sv}.json"),
            ):
                out.count("cells")
                if not _equal(v, h.get(k)):
                    out.add(
                        "release.aggregate_mismatch",
                        entity,
                        field=f"seasons.{sv}.{k}",
                        season=sv,
                        expected=v,
                        actual=h.get(k),
                        evidence={"resource": rel},
                    )
            _stat_block(out, entity, rel, f"seasons.{sv}", names, h.get("stats"), sw["stats"], sv)
        out.count("player_details")
    return out.done()


def _stat_block(
    out: _Out,
    entity: str,
    rel: str,
    where: str,
    names: list[str],
    cols: Any,
    want: dict[str, tuple[Any, int, int]],
    season: int | None,
) -> None:
    if not isinstance(cols, dict):
        out.add(
            "release.aggregate_mismatch",
            entity,
            field=where,
            expected="StatColumns",
            actual=None,
            evidence={"resource": rel},
        )
        return
    total, obs, elig = cols.get("total"), cols.get("observed_games"), cols.get("eligible_games")
    if not all(isinstance(x, list) and len(x) == len(names) for x in (total, obs, elig)):
        out.add(
            "release.compact_row_width",
            entity,
            field=where,
            expected=len(names),
            actual=[len(x) if isinstance(x, list) else None for x in (total, obs, elig)],
            evidence={"resource": rel},
        )
        return
    assert isinstance(total, list) and isinstance(obs, list) and isinstance(elig, list)
    for i, stat in enumerate(names):
        w = want.get(stat)
        if w is None:
            continue
        for part, have, expect in (
            ("total", total[i], _num(w[0])),
            ("observed_games", obs[i], w[1]),
            ("eligible_games", elig[i], w[2]),
        ):
            out.count("cells")
            if not _same_number(expect, have, stat if part == "total" else None):
                out.add(
                    "release.aggregate_mismatch",
                    entity,
                    field=f"{where}.{stat}.{part}",
                    season=season,
                    expected=expect,
                    actual=have,
                    evidence={"resource": rel},
                )


def run_units(fn: Any, payloads: list[dict[str, Any]], workers: int) -> list[UnitResult]:
    """Run units in order; with ``workers > 1`` in a spawn process pool. Order never matters."""
    if workers <= 1 or len(payloads) < 2:
        return [fn(p) for p in payloads]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(fn, payloads, chunksize=1))
