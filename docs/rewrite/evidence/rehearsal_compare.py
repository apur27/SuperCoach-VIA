"""Old-vs-new rehearsal: compare legacy harness outputs with the new package on identical input bytes.

Usage:
  uv run --locked python docs/rewrite/evidence/rehearsal_compare.py <legacy-worktree> <scratch-dir> [out.json]

<legacy-worktree> is a scripts/smoke_harness.sh worktree after its run (commit/push stubbed). The
new side imports that worktree's data/ (the same bytes the legacy run read), WITHOUT archived
repairs, into <scratch-dir>, runs the pipeline's validation (expected to refuse promotion on
the known 2026 gaps) and computes legacy_v1 rankings from the candidate. Nothing is written
outside <scratch-dir>.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

from supercoach_via.analytics import rankings
from supercoach_via.ingest import legacy, reconcile
from supercoach_via.settings import RunContext, Settings
from supercoach_via.storage.queries import SnapshotQuery

TOL = 1e-9
EXCLUDED = ("green_william_08092005", "steele_roan_19092002")  # quarantined duplicates (DATA_REFRESH.md)
# The parity test excludes these from the legacy scan. Filtering them only from the
# finished CSV leaves their games in the normalization and moves other scores.
SCAN_EXCLUSIONS = (
    {
        "player_id": "green_william_08092005",
        "policy": "excluded from the legacy scan",
        "provenance": "docs/rewrite/DATA_REFRESH.md; tests/scvia/integration/test_analytics_parity.py",
    },
    {
        "player_id": "steele_roan_19092002",
        "policy": "excluded from the legacy scan",
        "provenance": "docs/rewrite/DATA_REFRESH.md; tests/scvia/integration/test_analytics_parity.py",
    },
)


def _rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as fh:
        return list(csv.reader(fh))[1:]


def _compare_scored(legacy_rows: list[tuple[str, float]], new_rows: list[tuple[str, float]]) -> dict[str, Any]:
    lp, np_ = [p for p, _ in legacy_rows], [p for p, _ in new_rows]
    ls, ns = dict(legacy_rows), dict(new_rows)
    common = set(ls) & set(ns)
    max_delta = max((abs(ls[p] - ns[p]) for p in common), default=0.0)
    same_order = lp == np_
    # legacy breaks exact ties non-deterministically; legacy_v1 breaks them by player id
    tie_only = not same_order and set(lp) == set(np_) and all(
        abs(ls[a] - ls[b]) <= TOL for a, b in zip(lp, np_, strict=True) if a != b
    )
    only_l = sorted(set(lp) - set(np_))
    only_n = sorted(set(np_) - set(lp))
    common_left = [p for p in lp if p in ns]
    common_right = [p for p in np_ if p in ls]
    common_order_ok = common_left == common_right or all(
        abs(ls[a] - ls[b]) <= TOL for a, b in zip(common_left, common_right, strict=True) if a != b
    )
    return {
        "n_legacy": len(lp),
        "n_new": len(np_),
        "same_players": set(lp) == set(np_),
        "same_order": same_order,
        "order_differs_only_within_exact_ties": tie_only,
        "max_abs_score_delta": max_delta,
        "only_legacy": only_l[:10],
        "only_new": only_n[:10],
        "only_legacy_count": len(only_l),
        "only_new_count": len(only_n),
        "only_legacy_scores": [[p, ls[p]] for p in only_l[:10]],
        "only_new_scores": [[p, ns[p]] for p in only_n[:10]],
        "cutoff_legacy": legacy_rows[-1][1] if legacy_rows else None,
        "cutoff_new": new_rows[-1][1] if new_rows else None,
        "common_order_ok": common_order_ok,
    }


def _rank_against_legacy(wt: Path, data_root: Path, manifest: Any, snapshot_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {"legacy_worktree": str(wt), "snapshot_id": snapshot_id, "data_root": str(data_root)}
    with SnapshotQuery(data_root, manifest) as q:
        res = rankings.run_legacy_v1(q)
        bio_new = rankings.biography_export_rows(q, res)
    legacy_all = [(p, float(s)) for p, s in _rows(wt / "data/top100/all_time_top_100.csv")]
    legacy_all = [r for r in legacy_all if r[0] not in EXCLUDED]
    out["all_time_numeric"] = _compare_scored(legacy_all, rankings.numeric_export_rows(res))
    out["yearly"] = _yearly_from_files(
        wt / "data/top100/yearly",
        set(res.yearly),
        {
            season: [(player, float(score)) for player, score, _pct, _games in rankings.yearly_export_rows(table)]
            for season, table in res.yearly.items()
        },
    )
    legacy_bio = _rows(wt / "all_time_top_100.csv") if (wt / "all_time_top_100.csv").is_file() else []
    new_bio = [[str(a), b, c, d] for a, b, c, d in bio_new]
    same = sum(1 for a, b in zip(legacy_bio, new_bio, strict=False) if a == b)
    out["biography_csv"] = {
        "rows_legacy": len(legacy_bio),
        "rows_new": len(new_bio),
        "identical_rows": same,
        "content_matches": _biography_content_matches(legacy_bio, new_bio),
        "first_difference": next(([a, b] for a, b in zip(legacy_bio, new_bio, strict=False) if a != b), None),
    }
    return out


def _biography_content_matches(legacy_rows: list[list[str]], new_rows: list[list[str]]) -> bool:
    """Same prose per player. Rank order may change; a changed career line may not."""
    if not legacy_rows or len(legacy_rows) != len(new_rows):
        return False
    if all(left == right for left, right in zip(legacy_rows, new_rows, strict=True)):
        return True

    def keyed(rows: list[list[str]]) -> dict[str, tuple[str, ...]] | None:
        found: dict[str, tuple[str, ...]] = {}
        for row in rows:
            if len(row) < 2 or row[1] in found:
                return None
            found[row[1]] = tuple(row[2:])
        return found

    left, right = keyed(legacy_rows), keyed(new_rows)
    return left is not None and left == right


def _tied_cut_swap(row: dict[str, Any]) -> bool:
    """One player each side, equal scores, and those scores are the last-place cutoff."""
    if int(row.get("only_legacy_count") or 0) != 1 or int(row.get("only_new_count") or 0) != 1:
        return False
    left = row.get("only_legacy_scores") or []
    right = row.get("only_new_scores") or []
    if len(left) != 1 or len(right) != 1:
        return False
    left_score, right_score = float(left[0][1]), float(right[0][1])
    if abs(left_score - right_score) > TOL:
        return False
    if row.get("cutoff_legacy") is None or row.get("cutoff_new") is None:
        return False
    return abs(left_score - float(row["cutoff_legacy"])) <= TOL and abs(right_score - float(row["cutoff_new"])) <= TOL


def _season_ok(row: dict[str, Any]) -> bool:
    if row.get("missing_in_new") or row.get("missing_legacy"):
        return False
    if int(row.get("n_legacy") or 0) == 0 or int(row.get("n_new") or 0) == 0:
        return False
    if float(row.get("max_abs_score_delta", 1)) > TOL:
        return False
    if row.get("same_players"):
        return bool(row.get("same_order")) or bool(row.get("order_differs_only_within_exact_ties"))
    if row.get("common_order_ok") is not True:
        return False
    return _tied_cut_swap(row)


def _yearly_summary(yearly: dict[str, Any], missing_expected: list[str]) -> dict[str, Any]:
    agree = [season for season, row in yearly.items() if _season_ok(row)]
    return {
        "seasons": len(yearly),
        "same_players_and_scores": len(agree),
        "exact_order": sum(1 for row in yearly.values() if row.get("same_order")),
        "order_differs_only_within_exact_ties": sum(
            1 for row in yearly.values() if row.get("order_differs_only_within_exact_ties")
        ),
        "missing_expected": missing_expected,
        "disagreeing": {season: row for season, row in yearly.items() if season not in agree},
    }


def _yearly_from_files(
    yearly_dir: Path, expected: set[int], new_rows: dict[int, list[tuple[str, float]]]
) -> dict[str, Any]:
    """Compare every expected season, including one whose legacy CSV is absent."""
    yearly: dict[str, Any] = {}
    present: set[int] = set()
    if yearly_dir.is_dir():
        for path in sorted(yearly_dir.glob("year_*.csv")):
            season = int(path.stem.removeprefix("year_"))
            present.add(season)
            if season not in expected:
                yearly[str(season)] = {"missing_in_new": True, "n_legacy": 0, "n_new": 0}
                continue
            legacy_rows = [(row[0], float(row[1])) for row in _rows(path) if row[0] not in EXCLUDED]
            yearly[str(season)] = _compare_scored(legacy_rows, list(new_rows.get(season, [])))
    missing = [str(season) for season in sorted(expected - present)]
    for season in missing:
        yearly[season] = {"missing_legacy": True, "n_legacy": 0, "n_new": len(new_rows.get(int(season), []))}
    return _yearly_summary(yearly, missing)


def comparison_verdict(report: dict[str, Any]) -> dict[str, Any]:
    """PASS only for identical scores, exact-tie reordering, or a scored tied cut swap.

    A non-tied replacement, an empty table, a missing season, or a biography
    content change fails. Names without cutoff scores are not a tie.
    """
    reasons: list[str] = []
    if report.get("current_matches_selected") is not True:
        reasons.append("selected snapshot is not the current pointer")
    all_time = report.get("all_time_numeric") or {}
    if int(all_time.get("n_legacy") or 0) == 0 or int(all_time.get("n_new") or 0) == 0:
        reasons.append("all-time export empty")
    elif not all_time.get("same_players"):
        reasons.append("all-time players differ")
    elif float(all_time.get("max_abs_score_delta", 1)) > TOL:
        reasons.append("all-time scores differ")
    elif not all_time.get("same_order") and not all_time.get("order_differs_only_within_exact_ties"):
        reasons.append("all-time order differs")
    yearly = report.get("yearly") if isinstance(report.get("yearly"), dict) else None
    if yearly is None or int(yearly.get("seasons") or 0) <= 0:
        reasons.append("yearly comparison missing")
    else:
        if yearly.get("missing_expected"):
            reasons.append("missing expected season")
        for season, row in (yearly.get("disagreeing") or {}).items():
            if not _season_ok(row):
                reasons.append(f"{season} differs")
    bio = report.get("biography_csv") if isinstance(report.get("biography_csv"), dict) else None
    if bio is None or int(bio.get("rows_legacy") or 0) == 0 or int(bio.get("rows_new") or 0) == 0:
        reasons.append("biography export missing")
    elif bio.get("content_matches") is not True:
        reasons.append("biography rows differ")
    return {"ok": not reasons, "reasons": reasons[:20]}


def compare_promoted_snapshot(
    wt: Path, data_root: Path, release_public: Path, *, snapshot_id: str | None = None
) -> dict[str, Any]:
    """Rank ``snapshot_id``, or the release manifest's id. Do not follow a moved pointer."""
    from supercoach_via.storage.snapshots import IntegrityError, load_snapshot, read_current

    built = json.loads((release_public / "release.json").read_text())
    selected = snapshot_id or built.get("snapshot_id")
    if not selected or built.get("snapshot_id") != selected:
        raise SystemExit("compare: release snapshot is not the selected snapshot")
    try:
        manifest = load_snapshot(data_root, selected, verify=True)
    except (OSError, ValueError, IntegrityError) as exc:
        raise SystemExit(f"compare: selected snapshot is not a promoted snapshot: {exc}") from exc
    pointer = read_current(data_root)
    out = _rank_against_legacy(wt, data_root, manifest, selected)
    out["release_snapshot_id"] = built["snapshot_id"]
    out["selected_snapshot_id"] = selected
    out["current_snapshot_id"] = None if pointer is None else pointer.snapshot_id
    out["current_matches_selected"] = pointer is not None and pointer.snapshot_id == selected
    return out


def raw_input_inventory(source: Path) -> dict[str, Any]:
    """Fingerprint captured inputs. Ranking exports are outputs, not inputs."""
    import hashlib

    data = source / "data"
    if not data.is_dir():
        raise SystemExit(f"compare: {source} has no data/ directory")
    rows: list[str] = []
    for path in sorted(p for p in data.rglob("*") if p.is_file()):
        rel = path.relative_to(source).as_posix()
        if rel.startswith("data/top100/") or rel == "all_time_top_100.csv":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {rel}")
    return {"files": len(rows), "sha256": hashlib.sha256("\n".join(rows).encode()).hexdigest()}


def legacy_scan_glob(pattern: str) -> list[str]:
    """Legacy file scan with the documented duplicate identities left out."""
    import glob

    excluded = tuple(item["player_id"] for item in SCAN_EXCLUSIONS)
    return sorted(path for path in glob.glob(pattern) if not any(name in path for name in excluded))  # noqa: PTH207


def _import_legacy_ranker():
    """The legacy ranker lives at the checkout root, not on the package path."""
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import top_players_comprehensive as legacy_ranker

    return legacy_ranker


def _run_legacy_ranker(dest: Path) -> None:
    """Run the legacy ranker in ``dest``. Its scan uses ``legacy_scan_glob``."""
    import os
    from types import SimpleNamespace

    legacy_ranker = _import_legacy_ranker()

    original = legacy_ranker.glob
    previous = Path.cwd()
    legacy_ranker.glob = SimpleNamespace(glob=legacy_scan_glob)
    try:
        os.chdir(dest)
        legacy_ranker.main()
    finally:
        legacy_ranker.glob = original
        os.chdir(previous)


_REPAIR_CSV = (
    ("team", "club_source_name"),
    ("year", "season"),
    ("games_played", "career_game_counter"),
    ("opponent", "opponent_source_name"),
    ("round", "stage_label"),
    ("result", "result"),
    ("jersey_num", "jersey_number"),
    ("kicks", "kicks"),
    ("marks", "marks"),
    ("handballs", "handballs"),
    ("disposals", "disposals"),
    ("goals", "goals"),
    ("behinds", "behinds"),
    ("hit_outs", "hitouts"),
    ("tackles", "tackles"),
    ("rebound_50s", "rebound_50s"),
    ("inside_50s", "inside_50s"),
    ("clearances", "clearances"),
    ("clangers", "clangers"),
    ("free_kicks_for", "frees_for"),
    ("free_kicks_against", "frees_against"),
    ("brownlow_votes", "brownlow_votes"),
    ("contested_possessions", "contested_possessions"),
    ("uncontested_possessions", "uncontested_possessions"),
    ("contested_marks", "contested_marks"),
    ("marks_inside_50", "marks_inside_50"),
    ("one_percenters", "one_percenters"),
    ("bounces", "bounces"),
    ("goal_assist", "goal_assists"),
    ("percentage_of_game_played", "time_on_ground_pct"),
    ("date", "match_date"),
)


def apply_documented_repair(dest: Path, evidence_dir: Path, season: int) -> dict[str, Any]:
    """Append archived repair games onto the isolated copy. The captured source stays unchanged."""
    import hashlib

    manifest_path = evidence_dir / "fetch-manifest.json"
    rows_path = evidence_dir / "rows.jsonl"
    if not manifest_path.is_file() or not rows_path.is_file():
        raise SystemExit(f"compare: {evidence_dir} is not repair evidence")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("outcome", "")).lower() != "pass":
        raise SystemExit(f"compare: repair evidence outcome is {manifest.get('outcome')}")
    appended = 0
    skipped: list[str] = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("table") != "player_games":
            continue
        row = record["row"]
        if int(row["season"]) != season:
            continue
        player_id = str(row["player_id"])
        if player_id.startswith("legacy:"):
            slug = player_id.removeprefix("legacy:")
        else:
            safe = "".join(ch if ch.isalnum() else "_" for ch in player_id)
            slug = f"repair_{safe}_00000000"
            skipped.append(player_id)
        path = dest / "data" / "player_data" / f"{slug}_performance_details.csv"
        cells = ["" if row.get(field) is None else str(row.get(field)) for _column, field in _REPAIR_CSV]
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(",".join(column for column, _field in _REPAIR_CSV) + "\n", encoding="utf-8")
        with path.open("a", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(cells)
        appended += 1
    return {
        "evidence": str(evidence_dir),
        "season": season,
        "evidence_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "rows_appended": appended,
        "players_without_legacy_file": skipped,
        "created_files_for_new_identities": sorted(set(skipped)),
        "policy": "archived repair rows appended to the isolated copy only",
    }


def regenerate_legacy_exports(
    source: Path, dest: Path, *, repair: tuple[Path, int] | None = None
) -> dict[str, Any]:
    """Copy the captured tree and rewrite its ranking CSVs with the legacy script."""
    import shutil

    if dest.exists():
        raise SystemExit(f"compare: refusing to replace existing legacy exports at {dest}")
    inventory = raw_input_inventory(source)
    shutil.copytree(source / "data", dest / "data")
    top = dest / "data" / "top100"
    if top.exists():
        shutil.rmtree(top)
    bio = dest / "all_time_top_100.csv"
    if bio.exists():
        bio.unlink()
    corrections: list[dict[str, Any]] = list(SCAN_EXCLUSIONS)
    if repair is not None:
        corrections.append(apply_documented_repair(dest, repair[0], repair[1]))
    _run_legacy_ranker(dest)
    if not (dest / "data" / "top100" / "all_time_top_100.csv").is_file():
        raise SystemExit("compare: legacy regeneration did not write rankings")
    inventory["destination"] = str(dest)
    inventory["corrections"] = corrections
    return inventory


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--regenerate":
        args = sys.argv[2:]
        repair_spec: tuple[Path, int] | None = None
        if "--repair" in args:
            at = args.index("--repair")
            evidence, _sep, season = args[at + 1].rpartition(":")
            if not _sep or not season.isdigit():
                raise SystemExit("compare: --repair expects EVIDENCE_DIR:SEASON")
            repair_spec = (Path(evidence), int(season))
            del args[at : at + 2]
        if len(args) != 2:
            raise SystemExit("usage: rehearsal_compare.py --regenerate SOURCE DEST [--repair EVIDENCE_DIR:SEASON]")
        text = json.dumps(regenerate_legacy_exports(Path(args[0]), Path(args[1]), repair=repair_spec), indent=1)
        print(text)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--promoted":
        args = sys.argv[2:]
        snapshot_flag: str | None = None
        if "--snapshot" in args:
            at = args.index("--snapshot")
            snapshot_flag = args[at + 1]
            del args[at : at + 2]
        if len(args) not in (3, 4):
            raise SystemExit(
                "usage: rehearsal_compare.py --promoted LEGACY DATA_ROOT RELEASE_PUBLIC [out.json] --snapshot ID"
            )
        out = compare_promoted_snapshot(Path(args[0]), Path(args[1]), Path(args[2]), snapshot_id=snapshot_flag)
        out["verdict"] = comparison_verdict(out)
        text = json.dumps(out, indent=1, default=str)
        if len(args) == 4:
            Path(args[3]).write_text(text + "\n")
        print(text[:6000])
        if not out["verdict"]["ok"]:
            raise SystemExit("compare: " + "; ".join(out["verdict"]["reasons"]))
        return
    wt, scratch = Path(sys.argv[1]), Path(sys.argv[2])
    out: dict[str, Any] = {"legacy_worktree": str(wt)}
    cand = legacy.import_legacy(wt, RunContext(settings=Settings(data_root=scratch / "var")))
    report = reconcile.validate_dataset(cand, reconcile.load_policy())
    blocking = [i for i in report.issues if i["severity"] == "blocking" and i["status"] != "accepted"]
    out["new_validation"] = {"outcome": report.outcome.value, "blocking": len(blocking),
                             "blocking_rules": sorted({i["rule_id"] for i in blocking}),
                             "would_promote": report.ok}  # fmt: skip
    with SnapshotQuery(cand.data_root, cand.candidate.manifest) as q:
        res = rankings.run_legacy_v1(q)
        bio_new = rankings.biography_export_rows(q, res)

    legacy_all = [(p, float(s)) for p, s in _rows(wt / "data/top100/all_time_top_100.csv")]
    legacy_all = [r for r in legacy_all if r[0] not in EXCLUDED]
    out["all_time_numeric"] = _compare_scored(legacy_all, rankings.numeric_export_rows(res))

    out["yearly"] = _yearly_from_files(
        wt / "data/top100/yearly",
        set(res.yearly),
        {
            season: [(player, float(score)) for player, score, _pct, _games in rankings.yearly_export_rows(table)]
            for season, table in res.yearly.items()
        },
    )

    legacy_bio = _rows(wt / "all_time_top_100.csv")
    new_bio = [[str(a), b, c, d] for a, b, c, d in bio_new]
    same = sum(1 for a, b in zip(legacy_bio, new_bio, strict=False) if a == b)
    out["biography_csv"] = {"rows_legacy": len(legacy_bio), "rows_new": len(new_bio), "identical_rows": same,
                            "first_difference": next(([a, b] for a, b in zip(legacy_bio, new_bio, strict=False)
                                                      if a != b), None)}  # fmt: skip
    text = json.dumps(out, indent=1, default=str)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(text + "\n")
    print(text[:6000])


if __name__ == "__main__":
    main()
