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
    return {"n_legacy": len(lp), "n_new": len(np_), "same_players": set(lp) == set(np_), "same_order": same_order,
            "order_differs_only_within_exact_ties": tie_only, "max_abs_score_delta": max_delta,
            "only_legacy": sorted(set(lp) - set(np_))[:10], "only_new": sorted(set(np_) - set(lp))[:10]}  # fmt: skip


def main() -> None:
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
    out["all_time_numeric"] = _compare_scored(legacy_all, rankings.numeric_export_rows(res)[: len(legacy_all)])

    yearly: dict[str, Any] = {}
    for f in sorted((wt / "data/top100/yearly").glob("year_*.csv")):
        season = int(f.stem.removeprefix("year_"))
        if season not in res.yearly:
            yearly[str(season)] = {"missing_in_new": True}
            continue
        lrows = [(r[0], float(r[1])) for r in _rows(f) if r[0] not in EXCLUDED]
        nrows = [(p, float(s)) for p, s, _pct, _g in rankings.yearly_export_rows(res.yearly[season])]
        yearly[str(season)] = _compare_scored(lrows, nrows[: len(lrows)])
    agree = [s for s, v in yearly.items() if v.get("same_players") and v.get("max_abs_score_delta", 1) <= TOL]
    out["yearly"] = {"seasons": len(yearly), "same_players_and_scores": len(agree),
                     "exact_order": sum(1 for v in yearly.values() if v.get("same_order")),
                     "order_differs_only_within_exact_ties": sum(1 for v in yearly.values()
                                                                 if v.get("order_differs_only_within_exact_ties")),
                     "disagreeing": {s: v for s, v in yearly.items() if s not in agree}}  # fmt: skip

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
