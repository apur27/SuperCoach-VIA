"""B1 repair: append source-verified 2026 rows to the legacy player CSVs (input v0 amendment).

Reads only docs/rewrite/evidence/b1-repair-sources.json (written by b1_repair_fetch.py); makes
no network requests. Refuses to run unless every pre-existing file still has the hash it had
when this repair was prepared, the file's last career counter immediately precedes the first
new row, and every new row is fixture-verified with disposals == kicks + handballs.
Writes docs/rewrite/evidence/b1-repair-manifest.json (paths, before/after sha256, rows added).

Run: uv run --locked python docs/rewrite/evidence/b1_apply_rows.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "docs/rewrite/evidence"
PLAYER_DIR = ROOT / "data/player_data"

HEADER = (
    "team,year,games_played,opponent,round,result,jersey_num,kicks,marks,handballs,disposals,goals,behinds,"
    "hit_outs,tackles,rebound_50s,inside_50s,clearances,clangers,free_kicks_for,free_kicks_against,"
    "brownlow_votes,contested_possessions,uncontested_possessions,contested_marks,marks_inside_50,"
    "one_percenters,bounces,goal_assist,percentage_of_game_played,date"
).split(",")
# legacy column -> canonical stat key produced by the rewrite's AFLTables parser
LEGACY_TO_CANONICAL = {
    "hit_outs": "hitouts",
    "free_kicks_for": "frees_for",
    "free_kicks_against": "frees_against",
    "goal_assist": "goal_assists",
    "percentage_of_game_played": "time_on_ground_pct",
}

TARGETS = {
    # key in sources json -> (performance file, expected sha256 before, personal file to create or None)
    "perez_flynn": ("perez_flynn_25082001_performance_details.csv", None, None),
    "brodie_will": ("brodie_will_23081998_performance_details.csv", None, None),
    "dalton_jack_2026": ("dalton_jack_05042007_performance_details.csv", None, "dalton_jack_05042007_personal_details.csv"),
}


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def fmt(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def main() -> None:
    src = json.loads((EVIDENCE / "b1-repair-sources.json").read_text())
    v0 = {f["path"]: f["sha256"] for f in json.loads((EVIDENCE / "input-v0-manifest.json").read_text())["files"]}
    manifest: dict[str, object] = {"source_evidence": "b1-repair-sources.json", "files": []}
    for key, (perf_name, _unused, personal_name) in TARGETS.items():
        page = src["players"][key]
        assert page["outcome"] == "PASS" and not page["issues"], (key, page["issues"])
        perf = PLAYER_DIR / perf_name
        rel = str(perf.relative_to(ROOT))
        before = sha(perf)
        if before is not None:
            assert v0.get(rel) == before, f"{rel} changed since input-v0 manifest; refusing"
            rows = list(csv.DictReader(io.StringIO(perf.read_text())))
            assert not any(r["year"] == "2026" for r in rows), f"{rel} already has 2026 rows"
            last_counter = int(rows[-1]["games_played"])
        else:
            assert rel not in v0, rel
            last_counter = 0
        games = page["games_2026"]
        assert int(games[0]["counter"]) == last_counter + 1, (key, games[0]["counter"], last_counter)
        new_rows = []
        for i, g in enumerate(games):
            assert g["date_quality"] == "fixture_verified" and g["date"], g
            assert int(g["counter"]) == last_counter + 1 + i, g
            st = g["stats"]
            assert st["disposals"] == (st["kicks"] or 0) + (st["handballs"] or 0), g
            row = {
                "team": g["team"], "year": "2026", "games_played": g["counter"], "opponent": g["opponent"],
                "round": g["round"], "result": g["result"] or "", "jersey_num": g["jersey"], "date": g["date"],
            }
            for col in HEADER[7:-1]:
                row[col] = fmt(st[LEGACY_TO_CANONICAL.get(col, col)])
            new_rows.append(row)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=HEADER, lineterminator="\n")
        if before is None:
            w.writeheader()
        else:
            raw = perf.read_bytes()
            assert raw.endswith(b"\n"), f"{rel} lacks a trailing newline"
        w.writerows(new_rows)
        with perf.open("a" if before is not None else "x", newline="") as fh:
            fh.write(buf.getvalue())
        entry = {"path": rel, "before_sha256": before, "after_sha256": sha(perf), "rows_added": len(new_rows),
                 "source_url": page["url"], "rounds": [g["round"] for g in games]}
        manifest["files"].append(entry)  # type: ignore[union-attr]
        if personal_name:
            p = PLAYER_DIR / personal_name
            bd = page["birth_date"]  # ISO yyyy-mm-dd -> legacy dd-mm-yyyy
            debut = games[0]["date"]
            text = (
                "first_name,last_name,born_date,debut_date,height,weight\n"
                f"Jack,Dalton,{bd[8:10]}-{bd[5:7]}-{bd[0:4]},{debut[8:10]}-{debut[5:7]}-{debut[0:4]},,\n"
            )
            with p.open("x") as fh:
                fh.write(text)
            manifest["files"].append(  # type: ignore[union-attr]
                {"path": str(p.relative_to(ROOT)), "before_sha256": None, "after_sha256": sha(p), "rows_added": 1,
                 "source_url": page["url"], "note": "height/weight not parsed from source; left blank (unknown)"}
            )
    (EVIDENCE / "b1-repair-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
