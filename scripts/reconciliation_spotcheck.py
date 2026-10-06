#!/usr/bin/env python3
"""Independent spot-check of applied reconciliation corrections (snapshot layer).

The corrections were derived with the reconciliation reader; this check re-reads the SAME frozen page bytes with
the production ingest parser (``supercoach_via.ingest.afltables``), which shares no code with that reader, and
requires every sampled corrected value to agree:

* a statistic: the player page's cell for that game equals the corrected value (a proven zero or a nulled cell
  is consistent with the page's blank, which the ingest parser reads as ``None``);
* a match date: the season fixture's date for that ``match_id`` equals the corrected row date.

The sample is stratified by rule and seeded, so it is reproducible. Usage::

    reconciliation_spotcheck.py CHANGES BASE_ROOT CORRECTED_ROOT CAPTURE_MANIFEST OUT_JSON [--per-rule N] [--seed S]

Exit 0 when every sampled value agrees, 1 when any disagrees, 2 on invalid input.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from supercoach_via.ingest import afltables as at

STAT_RULES = ("R-TOTAL-NONBLANK", "R-BR-AWARD-SUM", "R-HEADER-ZERO", "R-AVG-COUNTED", "R-BR-SEASON-TOTAL",
              "R-BOTH-PAGES", "R-NOTES-EXCEPTION", "R-AVG-EXCLUDED")  # fmt: skip


def parse_page(body: bytes, url: str) -> at.PlayerPage:
    return at.parse_player_page(body, page_url=url)


def find_game(page: at.PlayerPage, *, season: int, team: str, opponent: str, stage: str) -> Any:
    hits = [
        g
        for g in page.games
        if g.season == season and g.team == team and g.opponent == opponent and g.round_token == stage
    ]
    return hits[0] if len(hits) == 1 else None


def cell_verdict(game: Any, field: str, corrected: Any) -> str:
    page = game.stats.get(field)
    if corrected is None or corrected == 0:
        return "agrees" if page in (None, 0) else "disagrees"
    return "agrees" if page is not None and float(page) == float(corrected) else "disagrees"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("changes")
    ap.add_argument("base_root")
    ap.add_argument("corrected_root")
    ap.add_argument("capture_manifest")
    ap.add_argument("out")
    ap.add_argument("--per-rule", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261005)
    a = ap.parse_args(argv[1:])

    import pyarrow.parquet as pq

    from supercoach_via.reconciliation import corrections as CO
    from supercoach_via.storage import snapshots
    from supercoach_via.storage.queries import SnapshotQuery

    changes = [c for c in CO.read_changes(Path(a.changes)) if c.layer == "snapshot" and "#" in c.target]
    by_rule: dict[str, list[CO.Change]] = defaultdict(list)
    for c in changes:
        if c.rule_id in STAT_RULES or c.field == "date":
            by_rule["R-ATTR-DATE" if c.field == "date" else c.rule_id].append(c)
    rng = random.Random(a.seed)  # noqa: S311 - a reproducible audit sample, not cryptography
    sample = {r: rng.sample(cs, min(a.per_rule, len(cs))) for r, cs in sorted(by_rule.items())}

    base_root, corr_root = Path(a.base_root), Path(a.corrected_root)
    base_snaps = sorted((base_root / "snapshots").glob("*.json"))
    frags: dict[str, Path] = {}
    for p in base_snaps:  # every snapshot of the corrected root's history: an origin names a fragment of one
        m = json.loads(p.read_text())
        for f in m["tables"].get("player_games", {}).get("fragments", []):
            frags[f["sha256"][:16]] = base_root / "fragments" / f["path"]
    tables: dict[str, list[dict[str, Any]]] = {}

    def key_of(c: CO.Change) -> tuple[str, str, str]:
        sha16, _, i = c.target.partition("#")
        if sha16 not in tables:
            tables[sha16] = pq.read_table(frags[sha16]).to_pylist()
        r = tables[sha16][int(i)]
        return (r["match_id"], r["player_id"], r["club_id"])

    manifest = json.loads(Path(a.capture_manifest).read_text())
    objects = Path(a.capture_manifest).parent / "objects"
    sha_of = {r["url"]: r["sha256"] for r in manifest["resources"] if r.get("sha256")}

    def body(url: str) -> bytes | None:
        s = sha_of.get(url)
        return (objects / s[:2] / s).read_bytes() if s else None

    corrected = snapshots.load_snapshot(corr_root)
    pages: dict[str, Any] = {}
    fixtures: dict[int, dict[str, Any]] = {}
    out: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    with SnapshotQuery(corr_root, corrected, tables={"player_games", "players", "matches"}) as q:
        urls = {
            pid: json.loads(u)[0] for pid, u in q.rows("SELECT player_id, source_urls FROM players") if u and u != "[]"
        }
        for rule, cs in sample.items():
            for c in cs:
                mid, pid, club = key_of(c)
                row = q.arrow("SELECT * FROM player_games WHERE match_id = ? AND player_id = ? AND club_id = ?",
                              [mid, pid, club]).to_pylist()  # fmt: skip
                if len(row) != 1:
                    out[rule]["row_missing"] += 1
                    continue
                r = row[0]
                if c.field == "date":
                    season = int(r["season"])
                    if season not in fixtures:
                        b = body(at.season_url(season))
                        fx = at.parse_season_page(b, season=season) if b else None
                        fixtures[season] = {
                            (frozenset((m.home_name, m.away_name)), m.stage.label, m.replay_occurrence): m
                            for m in (fx.matches if fx else [])
                        }
                    mrow = q.arrow("SELECT * FROM matches WHERE match_id = ?", [mid]).to_pylist()
                    fm = None
                    if mrow:
                        mr = mrow[0]
                        fm = fixtures[season].get(
                            (frozenset((mr["home_source_name"], mr["away_source_name"])), mr["stage_label"],
                             mr["replay_occurrence"])
                        )  # fmt: skip
                    if fm is None or fm.match_date is None:
                        out[rule]["unverifiable"] += 1
                        continue
                    ok = r["match_date"] == fm.match_date
                    out[rule]["agrees" if ok else "disagrees"] += 1
                    if not ok:
                        examples.append({"rule": rule, "key": [mid, pid], "row": str(r["match_date"]),
                                         "fixture": str(fm.match_date)})  # fmt: skip
                    continue
                url = urls.get(pid)
                b = body(url) if url else None
                if b is None:
                    out[rule]["unverifiable"] += 1
                    continue
                if url not in pages:
                    pages[url] = parse_page(b, url)
                g = find_game(pages[url], season=int(r["season"]), team=r["club_source_name"],
                              opponent=r["opponent_source_name"], stage=r["stage_label"])  # fmt: skip
                if g is None:
                    out[rule]["unverifiable"] += 1
                    continue
                v = cell_verdict(g, c.field, r[c.field])
                out[rule][v] += 1
                if v == "disagrees":
                    examples.append({"rule": rule, "key": [mid, pid], "field": c.field, "row": r[c.field],
                                     "page": g.stats.get(c.field)})  # fmt: skip
    doc = {
        "kind": "afltables-reconciliation-spotcheck",
        "parser": "supercoach_via.ingest.afltables (independent of the reconciliation reader)",
        "seed": a.seed,
        "per_rule": a.per_rule,
        "population": {r: len(cs) for r, cs in sorted(by_rule.items())},
        "results": {r: dict(sorted(v.items())) for r, v in sorted(out.items())},
        "disagreements": examples[:50],
    }
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(doc["results"], sort_keys=True))
    return 1 if any(v.get("disagrees") for v in out.values()) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
