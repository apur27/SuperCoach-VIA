"""Source-backed corrections derived from completed audit findings (DESIGN section 14, "Corrections").

The audit says WHAT is wrong; this module turns the confirmed discrepancies whose correct value the frozen source
states into explicit, reviewable changes:

* ``CELL_LOCAL_NULL``: the source proves a recorded zero (team totals, a header score, the Brownlow award sum);
* ``CELL_MISMATCH`` from a double-sourced cell (``R-BOTH-PAGES``: the profile and the match page agree), or a local
  number where the source proves a recorded zero (the same rules as ``CELL_LOCAL_NULL``);
* ``APPEARANCE_ATTR_MISMATCH`` for jersey and career counter, ``APPEARANCE_DATE_MISMATCH`` per row.

Every change names its finding, rule and evidence (source URL and body SHA-256). Anything else (an UNKNOWN, a
single-sourced value, a missing appearance) is never guessed: it is counted as ``unsupported`` for a dedicated,
separately tested handler. Applying a change first checks that the stored value is the audited ``old`` value, so
a correction never lands on data the audit did not see; a conflict aborts before anything is written.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from supercoach_via.reconciliation.local import LEGACY_COLUMNS, parse_legacy_number

#: canonical attribute -> legacy CSV column
_LEGACY_ATTR = {"jersey": "jersey_num", "counter": "games_played", "date": "date"}
_LEGACY_FIELD = {v: k for k, v in LEGACY_COLUMNS.items()}
_PLAYER_DIR = "data/player_data/"
_MATCH_DIR = "data/matches/"
_AWARDS_FILE = "data/awards/brownlow_season_votes.csv"
AWARDS_HEADER = ("slug", "year", "team", "award", "value", "source_url", "source_sha256")


#: rules under which the source proves a cell was not recorded
_NOT_RECORDED_RULES = frozenset({"R-NOTES-EXCEPTION", "R-AVG-EXCLUDED"})
#: the cell rules that prove a blank source cell is a recorded zero (``cells.py``, ``season.py``)
_PROVEN_ZERO_RULES = frozenset(
    {"R-TOTAL-NONBLANK", "R-HEADER-ZERO", "R-BR-AWARD-SUM", "R-AVG-COUNTED", "R-BR-SEASON-TOTAL"}
)


class CorrectionConflict(RuntimeError):
    """A change cannot be applied as audited (stale old value, unknown target, unsafe path); nothing was written."""


@dataclass(frozen=True)
class Change:
    layer: str  # snapshot | legacy_csv
    target: str  # the audited local origin: "<fragment sha16>#<row>" or "<relative csv>#<data row>"
    field: str  # canonical statistic, or an attribute: date | jersey | counter
    old: Any
    new: Any
    rule_id: str
    finding_id: str
    source_url: str | None
    body_sha256: str | None


def change_key(c: Change) -> tuple[str, str, str]:
    file, _, row = c.target.rpartition("#")
    return (c.layer, f"{file}#{int(row):09d}" if row.isdigit() else c.target, c.field)


def _num(v: Any) -> int | float | None:
    if v is None or v == "":
        return None
    d = float(v)
    return int(d) if d.is_integer() else d


def changes_from_findings(findings: Iterable[dict[str, Any]]) -> tuple[list[Change], Counter[str]]:
    """The corrections the findings support, and a count of the fail findings they do not."""
    out: list[Change] = []
    unsupported: Counter[str] = Counter()
    for f in findings:
        if f["severity"] != "fail" and f["category"] != "LOCAL_UNSUPPORTED_NUMERIC":
            continue
        ev = f.get("evidence") or {}
        url, sha = ev.get("source_url"), ev.get("body_sha256")
        cat, rule, layer = f["category"], f["rule_id"], f["layer"]
        origin = (f.get("local") or {}).get("origin")

        def add(
            target: str,
            field: str,
            old: Any,
            new: Any,
            *,
            _ctx: tuple[str, str, str, str | None, str | None] = (layer, rule, f["id"], url, sha),
        ) -> None:
            lay, rid, fid, u, s = _ctx
            out.append(Change(lay, target, field, old, new, rid, fid, u, s))

        if cat == "CELL_LOCAL_NULL" and origin and f.get("field"):
            add(origin, f["field"], None, _num(f["expected"]))
        elif cat == "CELL_MISMATCH" and rule == "R-BOTH-PAGES" and origin and f.get("field"):
            add(origin, f["field"], _num(f["actual"]), _num(f["expected"]))
        elif (
            cat == "CELL_MISMATCH"
            and rule in _PROVEN_ZERO_RULES
            and _num(f["expected"]) == 0
            and origin
            and f.get("field")
        ):
            # the source revised a number to a blank its own totals prove zero: the same proof as a null's
            add(origin, f["field"], _num(f["actual"]), 0)
        elif (
            cat == "APPEARANCE_ATTR_MISMATCH"
            and origin
            and (f.get("field") in ("jersey", "counter") or (f.get("field") == "counter_token" and layer == "snapshot"))
        ):
            old, new = f["actual"], f["expected"]
            if f["field"] in ("jersey", "counter_token"):
                old, new = (None if old is None else str(old)), str(new)
            add(origin, f["field"], old, new)
        elif cat == "LOCAL_UNSUPPORTED_NUMERIC" and rule in _NOT_RECORDED_RULES and origin and f.get("field"):
            # the source proves the statistic was not recorded for this team-match (the notes page, or the players'
            # printed averages): a local zero there was invented. ``propose`` keeps only cells that hold 0.
            for fld in sorted(f["field"].split(",")):
                add(origin, fld, 0, None)
        elif cat == "LOCAL_MISSING_SUMMARY_VALUE" and rule == "R-AGG-STINT" and f.get("field") == "brownlow_votes":
            club = (f.get("local") or {}).get("club")
            pid = ((f.get("player") or {}).get("local_id") or "").split(",")[0]
            if not club or not pid or f.get("season") is None:
                unsupported[cat] += 1
                continue
            key = f"{pid}|{f['season']}|{club}"
            target = f"award:{key}" if layer == "snapshot" else f"{_AWARDS_FILE}#award:{key}"
            add(target, "brownlow_votes", None, _num(f["expected"]))
        elif cat == "LOCAL_MISSING_SUMMARY_VALUE" and rule == "R-AGG-CAREER":
            pass  # the career total is the sum of the stint values corrected above: re-judged, never edited
        elif cat == "APPEARANCE_DATE_MISMATCH":
            for target, src, local in (f.get("local") or {}).get("changes") or []:
                add(target, "date", local, src)
        else:
            unsupported[cat if cat != "CELL_MISMATCH" else f"{cat}:{rule}"] += 1
    return sorted(out, key=change_key), unsupported


# ---------------------------------------------------------------------------
# Change files
# ---------------------------------------------------------------------------


def write_changes(path: Path, changes: list[Change]) -> str:
    """Canonical JSON lines in a stable order; returns the file's SHA-256."""
    body = b"".join(
        json.dumps(asdict(c), sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for c in sorted(changes, key=change_key)
    )
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(body)
    os.replace(tmp, path)
    return hashlib.sha256(body).hexdigest()


def read_changes(path: Path) -> list[Change]:
    return [Change(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ---------------------------------------------------------------------------
# Legacy CSV layer
# ---------------------------------------------------------------------------


_PERSONAL_FIELDS = ("first_name", "last_name", "born_date")


def _legacy_column(field: str) -> str:
    if field in _PERSONAL_FIELDS:
        return field
    if field in _LEGACY_ATTR:
        return _LEGACY_ATTR[field]
    if field in _LEGACY_FIELD:
        return _LEGACY_FIELD[field]
    raise CorrectionConflict(f"no legacy column for field {field!r}")


def _same(field: str, raw: str, old: Any) -> bool:
    if field in ("date", "jersey", *_PERSONAL_FIELDS):
        return (raw or None) == (None if old is None else str(old))
    if field == "counter":
        return (raw or None) == (None if old is None else str(old)) or bool(parse_legacy_number(raw) == old)
    return bool(parse_legacy_number(raw) == old)


def _format(field: str, raw: str, new: Any) -> str:
    if new is None:
        return ""
    if field in ("date", "jersey", "counter", *_PERSONAL_FIELDS):
        return str(new)
    return f"{new}.0" if raw.endswith(".0") and isinstance(new, int) else str(new)


def apply_legacy(root: Path, changes: list[Change]) -> dict[str, int]:
    """Apply ``legacy_csv`` changes to ``root/data/player_data`` (all checked before any file is written).

    Each file is re-serialised only if its unchanged parse reproduces its original bytes exactly, so the edit
    touches the named cells and nothing else."""
    by_file: dict[str, list[Change]] = defaultdict(list)
    deletes: list[Path] = []
    creates: dict[Path, bytes] = {}
    awards: list[Change] = []
    for c in changes:
        if c.layer != "legacy_csv":
            continue
        if c.target.startswith(f"{_AWARDS_FILE}#award:"):
            awards.append(c)
            continue
        if c.field == "create_player":
            slug = c.target[len(_PLAYER_DIR) :] if c.target.startswith(_PLAYER_DIR) else ""
            if not slug or "/" in slug or ".." in slug or "#" in slug:
                raise CorrectionConflict(f"target {c.target!r} is outside {_PLAYER_DIR}")
            spec = json.loads(c.new)
            perf, pers = root / f"{c.target}_performance_details.csv", root / f"{c.target}_personal_details.csv"
            if perf.exists() or pers.exists():
                raise CorrectionConflict(f"{c.target}: a legacy file for this player already exists")
            from supercoach_via.reconciliation.legacy_rows import PERF_HEADER, PERSONAL_HEADER

            creates[pers] = _serialise([list(PERSONAL_HEADER), [spec["personal"].get(k, "") for k in PERSONAL_HEADER]])
            creates[perf] = _serialise([list(PERF_HEADER), *spec["rows"]])
            continue
        if c.field == "delete_files":
            slug = c.target[len(_PLAYER_DIR) :] if c.target.startswith(_PLAYER_DIR) else ""
            if not slug or "/" in slug or ".." in slug or "#" in slug:
                raise CorrectionConflict(f"target {c.target!r} is outside {_PLAYER_DIR}")
            pair = [root / f"{c.target}{sfx}" for sfx in ("_performance_details.csv", "_personal_details.csv")]
            if not all(p.is_file() for p in pair):
                raise CorrectionConflict(f"{c.target}: both legacy files must exist to remove a duplicate player")
            deletes.extend(pair)
            continue
        rel, _, row = c.target.rpartition("#")
        base_dir = _PLAYER_DIR if rel.startswith(_PLAYER_DIR) else _MATCH_DIR if rel.startswith(_MATCH_DIR) else ""
        if (
            not base_dir
            or "/" in rel[len(base_dir) :]
            or ".." in rel
            or not (row.isdigit() or (row == "insert" and c.field == "row"))
        ):
            raise CorrectionConflict(f"target {c.target!r} is outside {_PLAYER_DIR} and {_MATCH_DIR}")
        by_file[rel].append(c)
    staged: dict[Path, bytes] = {}
    cells = 0
    for rel, cs in sorted(by_file.items()):
        path = root / rel
        original = path.read_bytes()
        rows = list(csv.reader(io.StringIO(original.decode("utf-8"), newline="")))
        if _serialise(rows) != original:
            raise CorrectionConflict(f"{rel}: the CSV does not round-trip byte for byte; refusing to rewrite it")
        header = rows[0]
        inserts: list[list[str]] = []
        for c in sorted(cs, key=lambda x: x.field != "row"):
            if c.field == "row":
                new_row = json.loads(c.new)
                if len(new_row) != len(header):
                    raise CorrectionConflict(f"{rel}: a replacement row needs {len(header)} cells")
                pos = c.target.rpartition("#")[2]
                if pos == "insert":
                    inserts.append(new_row)
                else:
                    n = int(pos)
                    if n < 1 or n >= len(rows) or rows[n] != json.loads(c.old):
                        raise CorrectionConflict(f"{rel}#{pos} row: does not hold the audited row")
                    rows[n] = new_row
                cells += 1
                continue
            n = int(c.target.rpartition("#")[2])
            col = header.index(_legacy_column(c.field)) if _legacy_column(c.field) in header else -1
            if col < 0 or n < 1 or n >= len(rows):
                raise CorrectionConflict(f"{rel}#{n}: no cell for {c.field}")
            raw = rows[n][col]
            if not _same(c.field, raw, c.old):
                raise CorrectionConflict(f"{rel}#{n} {c.field}: holds {raw!r}, the audit saw {c.old!r}")
            rows[n][col] = _format(c.field, raw, c.new)
            cells += 1
        if inserts:
            rows = [header, *_ordered(header, rows[1:], inserts)]
        staged[path] = _serialise(rows)
    if awards:
        staged[root / _AWARDS_FILE] = _awards_file(root / _AWARDS_FILE, awards)
        cells += len(awards)
    if awards:
        (root / _AWARDS_FILE).parent.mkdir(parents=True, exist_ok=True)
    for path, body in staged.items():
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(body)
        os.replace(tmp, path)
    for path, body in creates.items():
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(body)
        os.replace(tmp, path)
    for path in deletes:
        path.unlink()
    out = {"files_changed": len(staged), "cells_changed": cells}
    if creates:
        out["files_created"] = len(creates)
    if deletes:
        out["files_removed"] = len(deletes)
    return out


def _awards_file(path: Path, changes: list[Change]) -> bytes:
    """The legacy awards file with one row per (slug, year, team, award); an existing row is never overwritten."""
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"), newline=""))) if path.is_file() else []
    if rows and tuple(rows[0]) != AWARDS_HEADER:
        raise CorrectionConflict(f"{_AWARDS_FILE}: unexpected header {rows[0]}")
    body = rows[1:]
    have = {(r[0], r[1], r[2], r[3]) for r in body}
    for c in changes:
        slug, year, team = c.target.partition("#award:")[2].split("|", 2)
        key = (slug, year, team, c.field)
        if key in have:
            raise CorrectionConflict(f"{_AWARDS_FILE}: {key} already holds a value")
        have.add(key)
        body.append([slug, year, team, c.field, str(c.new), c.source_url or "", c.body_sha256 or ""])
    return _serialise([list(AWARDS_HEADER), *sorted(body)])


def _ordered(header: list[str], body: list[list[str]], inserts: list[list[str]]) -> list[list[str]]:
    """``body`` with each insert placed after the last row that precedes it (career counter for a player file, start
    time for a match file). Existing rows never move, whatever their tokens (a counter may carry a sub arrow)."""
    from supercoach_via.reconciliation.local import _leading_int

    if "games_played" in header:
        i = header.index("games_played")

        def key(r: list[str]) -> tuple[int, str]:
            n = _leading_int(r[i])
            return (n if n is not None else -1, "")
    else:
        i = header.index("date")

        def key(r: list[str]) -> tuple[int, str]:
            return (0, r[i])

    out = list(body)
    for new in sorted(inserts, key=key):
        k = key(new)
        pos = max((j + 1 for j, r in enumerate(out) if key(r) <= k), default=0)
        out.insert(pos, new)
    return out


def _serialise(rows: list[list[str]]) -> bytes:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Snapshot layer
# ---------------------------------------------------------------------------

#: canonical attribute -> player_games column
_SNAP_ATTR = {
    "jersey": "jersey_number",
    "counter": "career_game_counter",
    "counter_token": "career_game_counter_token",
    "date": "match_date",
}


def _snap_value(field: str, new: Any) -> Any:
    from datetime import date as _date

    if new is None:
        return None
    if field == "date":
        return _date.fromisoformat(str(new))
    if field == "counter_token":
        return str(new)
    if field in ("jersey", "counter"):
        return int(new)
    return float(new) if field == "time_on_ground_pct" else int(new)


def _snap_same(field: str, stored: Any, old: Any) -> bool:
    if field == "date":
        return bool((None if stored is None else stored.isoformat()) == old)
    if field in ("jersey", "counter_token"):
        return bool((None if stored is None else str(stored)) == old)
    if stored is None or old is None:
        return stored is None and old is None
    return float(stored) == float(old)


def snapshot_upserts(data_root: Path, manifest: Any, changes: list[Change]) -> dict[str, list[dict[str, Any]]]:
    """Full ``player_games`` rows with the audited changes applied (checked against the audited values) plus one
    resolved quality issue per (season, field class) naming the evidence. Raises ``CorrectionConflict``."""
    import pyarrow.parquet as pq

    from supercoach_via.domain.schemas import TABLES
    from supercoach_via.storage.snapshots import contained_path

    frags = {f.sha256[:16]: f for f in manifest.tables["player_games"].fragments}
    by_frag: dict[str, list[Change]] = defaultdict(list)
    player_changes: list[Change] = []
    relinks: list[Change] = []
    award_changes: list[Change] = []
    for c in changes:
        if c.layer != "snapshot":
            continue
        if c.target.startswith("player:"):
            player_changes.append(c)
            continue
        if c.target.startswith("quarantine:"):
            relinks.append(c)
            continue
        if c.target.startswith("award:"):
            award_changes.append(c)
            continue
        sha16, _, row_no = c.target.partition("#")
        if sha16 not in frags or not row_no.isdigit():
            raise CorrectionConflict(f"{c.target}: not a player_games row of snapshot {manifest.snapshot_id}")
        by_frag[sha16].append(c)
    names = TABLES["player_games"].column_names
    rows_out: list[dict[str, Any]] = []
    issues: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    for sha16, cs in sorted(by_frag.items()):
        frag = frags[sha16]
        rows = pq.read_table(contained_path(data_root / "fragments", frag.path)).to_pylist()
        touched: dict[int, dict[str, Any]] = {}
        for c in cs:
            i = int(c.target.partition("#")[2])
            if i >= len(rows):
                raise CorrectionConflict(f"{c.target}: row {i} is outside the fragment")
            row = touched.setdefault(i, dict(rows[i]))
            col = _SNAP_ATTR.get(c.field, c.field)
            if col not in names:
                raise CorrectionConflict(f"{c.target}: no player_games column for {c.field!r}")
            if not _snap_same(c.field, row.get(col), c.old):
                raise CorrectionConflict(f"{c.target} {c.field}: holds {row.get(col)!r}, the audit saw {c.old!r}")
            row[col] = _snap_value(c.field, c.new)
            if c.field == "date":
                row["date_quality"] = "source"  # stated by the source match page
            klass = "date" if c.field == "date" else "attribute" if c.field in _SNAP_ATTR else "statistic"
            issues[(int(row["season"]), klass)][c.rule_id] += 1
        rows_out.extend({k: r.get(k) for k in names} for _i, r in sorted(touched.items()))
    quality = [
        {
            "issue_id": "qc:" + hashlib.sha256(json.dumps([s, k, sorted(cnt.items())]).encode()).hexdigest()[:24],
            "severity": "info",
            "status": "resolved",
            "table_name": "player_games",
            "row_key": None,
            "source_path": "https://afltables.com/afl/stats/",
            "rule_id": f"reconciliation.afltables_{k}_corrected",
            "explanation": f"{sum(cnt.values())} {k} values corrected from the frozen AFL Tables capture: "
            + ", ".join(f"{r}={n}" for r, n in sorted(cnt.items())),
            "remediation": None,
            "acceptance_basis": (
                "reconciliation findings + frozen capture; every change lists its source URL and SHA-256"
            ),
            "season": s,
        }
        for (s, k), cnt in sorted(issues.items())
    ]
    players, deletes = _player_changes(data_root, manifest, player_changes)
    moved, q_deletes = _relinks(data_root, manifest, relinks)
    done = {(r["match_id"], r["player_id"], r["club_id"]) for r in rows_out}
    clash = [r for r in moved if (r["match_id"], r["player_id"], r["club_id"]) in done]
    if clash:
        raise CorrectionConflict(f"a relinked row and a corrected row share the key {clash[0]['match_id']}")
    out: dict[str, list[dict[str, Any]]] = {
        "player_games": rows_out + moved,
        "quality_issues": quality,
        "players": players,
        "player_season_awards": _award_rows(data_root, manifest, award_changes),
    }
    if deletes:
        out["_delete_player_games"] = deletes
    if q_deletes:
        out["_delete_quarantine"] = q_deletes
    return out


def _award_rows(data_root: Path, manifest: Any, changes: list[Change]) -> list[dict[str, Any]]:
    """``player_season_awards`` rows; the club id is the one the player's own rows of that season carry for that
    club name, and an existing award row is never overwritten."""
    if not changes:
        return []
    from supercoach_via.domain.schemas import TABLES
    from supercoach_via.storage.queries import SnapshotQuery

    want = sorted({c.target.removeprefix("award:").split("|", 1)[0] for c in changes})
    tables = {"player_games"} | ({"player_season_awards"} if "player_season_awards" in manifest.tables else set())
    with SnapshotQuery(data_root, manifest, tables=tables) as q:
        clubs = {
            (r[0], int(r[1]), r[2]): r[3]
            for r in q.rows(
                "SELECT DISTINCT player_id, season, club_source_name, club_id FROM player_games "
                "WHERE list_contains(?, player_id)",
                [want],
            )
        }
        have = (
            {
                (r[0], int(r[1]), r[2], r[3])
                for r in q.rows("SELECT player_id, season, club_id, award FROM player_season_awards")
            }
            if "player_season_awards" in tables
            else set()
        )
    names = TABLES["player_season_awards"].column_names
    out: list[dict[str, Any]] = []
    for c in changes:
        pid, season_s, club = c.target.removeprefix("award:").split("|", 2)
        season = int(season_s)
        club_id = clubs.get((pid, season, club))
        if club_id is None:
            raise CorrectionConflict(f"{c.target}: the player has no rows for {club} in {season}")
        if (pid, season, club_id, c.field) in have:
            raise CorrectionConflict(f"{c.target}: an award row already exists")
        have.add((pid, season, club_id, c.field))
        row = dict.fromkeys(names)
        row.update(player_id=pid, season=season, club_id=club_id, club_source_name=club, award=c.field,
                   value=int(c.new), provenance="source_fetch", source_path=c.source_url or "",
                   source_sha256=c.body_sha256 or "")  # fmt: skip
        out.append(row)
    return out


def _relinks(
    data_root: Path, manifest: Any, changes: list[Change]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A quarantined player-game row restored as an accepted row of the candidate match the source proves (its own
    recorded cells, unchanged), and the quarantine rows to remove."""
    if not changes:
        return [], []
    from datetime import date as _date
    from datetime import datetime as _dt

    from supercoach_via.domain.schemas import TABLES
    from supercoach_via.storage.queries import SnapshotQuery

    if "quarantine" not in manifest.tables:
        raise CorrectionConflict(f"snapshot {manifest.snapshot_id} has no quarantine table")
    with SnapshotQuery(data_root, manifest, tables={"quarantine", "player_games"}) as q:
        qrows = {r["quarantine_id"]: r for r in q.arrow("SELECT * FROM quarantine").to_pylist()}
        keys = {tuple(r) for r in q.rows("SELECT match_id, player_id, club_id FROM player_games")}
    spec = TABLES["player_games"]
    moved: list[dict[str, Any]] = []
    gone: list[dict[str, Any]] = []
    for c in changes:
        qid = c.target.removeprefix("quarantine:")
        qr = qrows.get(qid)
        if qr is None or qr["table_name"] != "player_games":
            raise CorrectionConflict(f"{c.target}: no quarantine player-game row with that id")
        target = json.loads(c.new)["match_id"]
        if target not in json.loads(qr["candidates"] or "[]"):
            raise CorrectionConflict(f"{c.target}: {target} is not one of the quarantine row's candidates")
        raw = json.loads(qr["raw"])
        row = {k: raw.get(k) for k in spec.column_names}
        row.update(match_id=target, link_method="source_url")
        for col in spec.columns:
            v = row.get(col.name)
            if isinstance(v, str) and col.type == "date32":
                row[col.name] = _date.fromisoformat(v)
            elif isinstance(v, str) and col.type == "timestamp_utc":
                row[col.name] = _dt.fromisoformat(v)
        key = (row["match_id"], row["player_id"], row["club_id"])
        if key in keys:
            raise CorrectionConflict(f"{c.target}: {key} already holds an accepted row")
        keys.add(key)
        moved.append(row)
        gone.append({"quarantine_id": qid})
    return moved, gone


_PLAYER_FIELDS = ("display_name", "first_name", "last_name", "birth_date", "source_urls", "identity_status",
                  "canonical_player_id")  # fmt: skip


def _player_changes(
    data_root: Path, manifest: Any, changes: list[Change]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Full ``players`` rows with the audited identity changes applied, and the ``player_games`` keys of every
    player marked a duplicate (its rows restate the canonical player's games)."""
    if not changes:
        return [], []
    from datetime import date as _date

    from supercoach_via.domain.schemas import TABLES
    from supercoach_via.storage.queries import SnapshotQuery

    want = sorted({c.target.removeprefix("player:") for c in changes})
    with SnapshotQuery(data_root, manifest, tables={"players", "player_games"}) as q:
        rows = {
            r["player_id"]: r
            for r in q.arrow("SELECT * FROM players WHERE list_contains(?, player_id)", [want]).to_pylist()
        }
        dup_ids = sorted({c.target.removeprefix("player:") for c in changes if c.field == "delete_games"})
        deletes = (
            q.arrow(
                "SELECT match_id, player_id, club_id, season FROM player_games WHERE list_contains(?, player_id)",
                [dup_ids],
            ).to_pylist()
            if dup_ids
            else []
        )
    touched: dict[str, dict[str, Any]] = {}
    for c in changes:
        pid = c.target.removeprefix("player:")
        if pid not in rows:
            raise CorrectionConflict(f"{c.target}: no such player in snapshot {manifest.snapshot_id}")
        if c.field == "delete_games":
            continue
        if c.field not in _PLAYER_FIELDS:
            raise CorrectionConflict(f"{c.target}: {c.field!r} is not a correctable player field")
        row = touched.setdefault(pid, dict(rows[pid]))
        stored = row.get(c.field)
        shown = stored.isoformat() if isinstance(stored, _date) else stored
        if shown != c.old:
            raise CorrectionConflict(f"{c.target} {c.field}: holds {shown!r}, the audit saw {c.old!r}")
        row[c.field] = _date.fromisoformat(c.new) if c.field == "birth_date" and c.new else c.new
        if c.field == "birth_date":
            row["birth_date_quality"] = "source"  # stated by the source profile page
    names = TABLES["players"].column_names
    return [{k: r.get(k) for k in names} for _p, r in sorted(touched.items())], deletes


# ---------------------------------------------------------------------------
# Identity changes (``idfix`` proposals)
# ---------------------------------------------------------------------------


def _dmy(iso: str) -> str:
    y, m, d = iso.split("-")
    return f"{d}-{m}-{y}"


def identity_changes(proposals: list[Any], *, layer: str, current: dict[str, dict[str, Any]]) -> list[Change]:
    """``idfix.Proposal`` objects as changes for one layer. ``current`` maps the local key (snapshot player id;
    legacy slug) to its stored values, which become each change's audited ``old``."""
    out: list[Change] = []
    for p in proposals:
        key = p.key if layer == "snapshot" else p.key.removeprefix("legacy:")
        cur = current.get(key)
        if cur is None:
            continue
        rule = {"bind": "R-ID-BIND", "repair": "R-ID-REPAIR", "duplicate": "R-ID-DUPLICATE"}[p.kind]

        def add(target: str, field: str, new: Any, *, _cur: dict[str, Any] = cur, _rule: str = rule,
                _url: str = p.url) -> None:  # fmt: skip
            out.append(Change(layer, target, field, _cur.get(field), new, _rule, "", _url, None))

        if layer == "snapshot":
            target = f"player:{key}"
            if p.kind == "duplicate":
                add(target, "identity_status", "quarantined_duplicate")
                add(target, "canonical_player_id", p.fields["canonical"])
                out.append(Change(layer, target, "delete_games", None, "all", rule, "", p.url, None))
                continue
            for f in ("display_name", "first_name", "last_name", "birth_date"):
                if f in p.fields:
                    add(target, f, p.fields[f])
            if cur.get("source_urls") != json.dumps([p.url]):
                add(target, "source_urls", json.dumps([p.url]))
        else:
            if p.kind == "duplicate":
                out.append(Change(layer, f"{_PLAYER_DIR}{key}", "delete_files", "present", None, rule, "", p.url, None))
                continue
            target = f"{_PLAYER_DIR}{key}_personal_details.csv#1"
            for f in ("first_name", "last_name"):
                if f in p.fields:
                    add(target, f, p.fields[f])
            if "birth_date" in p.fields:
                add(target, "born_date", _dmy(p.fields["birth_date"]))
    return sorted(out, key=change_key)
