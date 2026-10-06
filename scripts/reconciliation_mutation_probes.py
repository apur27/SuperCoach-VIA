#!/usr/bin/env python3
"""Mutation probes for the AFL Tables reconciliation (DESIGN section 10). Works ONLY on copies.

Given one finished audit run directory (``plan.json``, ``capture/``, ``reports/cold-4/report.json`` and the
warm unit cache ``cache-4/``) this script makes hard-linked copies, applies ONE change to each copy, re-runs
``compare`` with ``--previous`` and records which units were recomputed and what the verdict became:

* source-cell    - one game-row statistic on one captured player profile
* availability   - one cell of the captured notes availability grid (1975 hit-outs)
* rule           - the first season of the Brownlow sum rule in a copied rules file
* local-row      - one cell of one legacy CSV row (copied legacy tree, re-issued plan)

Every mutated file is a NEW file (the hard link is broken first), so the original capture, cache, plan,
legacy tree and reports are never written through. The script re-hashes the originals at the end and exits
1 if any changed. Usage: ``reconciliation_mutation_probes.py RUN_DIR OUT_DIR``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from supercoach_via.reconciliation import compare as CP
from supercoach_via.reconciliation import inventory as inv
from supercoach_via.reconciliation import report as RP
from supercoach_via.reconciliation import schema as S
from supercoach_via.settings import default_config_dir


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def link_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, copy_function=os.link)


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(f"{p.relative_to(root)}\0{sha(p.read_bytes())}\n".encode())
    return h.hexdigest()


def replace_once(body: bytes, old: bytes, new: bytes, *, after: bytes = b"") -> bytes:
    start = body.index(after) if after else 0
    i = body.index(old, start)
    return body[:i] + new + body[i + len(old) :]


class Probe:
    def __init__(self, base: Path, out: Path, capture: Path, report: str, cache: str) -> None:
        self.base, self.out, self.capture, self.cache = base, out, capture, base / cache
        self.previous = base / "reports" / report / "report.json"
        self.plan = S.Plan.model_validate_json((base / "plan.json").read_bytes())
        self.manifest = json.loads((capture / "manifest.json").read_text())

    def resource(self, pred: Any) -> dict[str, Any]:
        return next(r for r in self.manifest["resources"] if pred(r))

    def body(self, res: dict[str, Any]) -> bytes:
        s: str = res["sha256"]
        return (self.capture / "objects" / s[:2] / s).read_bytes()

    def retarget(self, work: Path, res: dict[str, Any], edited: bytes) -> None:
        digest = sha(edited)
        path = work / "capture" / "objects" / digest[:2] / digest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(edited)
        manifest = json.loads((work / "capture" / "manifest.json").read_text())
        for r in manifest["resources"]:
            if r["url"] == res["url"]:
                r["sha256"], r["bytes"] = digest, len(edited)
        (work / "capture" / "manifest.json").unlink()  # break the hard link before writing
        (work / "capture" / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))

    # ---- mutations: each returns (description, legacy_root | None, config_dir | None)
    def source_cell(self, work: Path) -> tuple[str, Path | None, Path | None]:
        res = self.resource(lambda r: r["kind"] == "profile" and r["url"].endswith("/Scott_Pendlebury.html"))
        edited = replace_once(
            self.body(res),
            b"<td align=center>5</td><td align=center>3</td>",
            b"<td align=center>9</td><td align=center>3</td>",
            after=b'<a name="20060">',
        )
        self.retarget(work, res, edited)
        return f"{res['url']}: 2006 game 1 kicks 5 -> 9 (disposals stay 11)", None, None

    def availability(self, work: Path) -> tuple[str, Path | None, Path | None]:
        res = self.resource(lambda r: r["kind"] == "notes")
        body = self.body(res)
        row_start = body.index(b"1975")
        row_end = body.index(b"</tr>", row_start)
        row = body[row_start:row_end]
        cells = row.split(b"<td")
        if len(cells) < 8 or b"X" not in cells[6]:
            raise RuntimeError(f"unexpected 1975 notes row: {row[:200]!r}")
        cells[6] = cells[6].replace(b"X", b"&nbsp;", 1)  # cells[0] is the year, then KI MK HB GL BH HO -> 6
        self.retarget(work, res, body[:row_start] + b"<td".join(cells) + body[row_end:])
        return "notes grid 1975 hit-outs availability X -> blank", None, None

    def rule(self, work: Path) -> tuple[str, Path | None, Path | None]:
        cfg = work / "config"
        shutil.copytree(default_config_dir(), cfg)
        rules = cfg / "reconciliation_rules.toml"
        rules.write_text(
            rules.read_text()
            + '\n[[notes_club_alias]]\nseason = 1990\nname = "Probe"\nclub = "Footscray"\n'
            + 'evidence_locator = "mutation probe"\nreason = "mutation probe"\n'
        )
        return "rules: a notes-club alias for 1990 added (only 1990 units may be recomputed)", None, cfg

    def local_row(self, work: Path) -> tuple[str, Path | None, Path | None]:
        legacy = self.out / "_legacy-copy"  # outside the run dir: plan refuses a run dir holding an input root
        shutil.rmtree(legacy, ignore_errors=True)
        src = Path(self.plan.operational.legacy_root or "")
        for sub in ("player_data", "matches", "awards"):  # every directory the legacy layer is pinned from
            if (src / "data" / sub).is_dir():
                link_tree(src / "data" / sub, legacy / "data" / sub)
        victim = sorted((legacy / "data" / "player_data").glob("pendlebury_scott_*_performance_details.csv"))[0]
        with victim.open(newline="") as fh:
            rows = list(csv.reader(fh))
        col = rows[0].index("kicks")
        before = rows[10][col]
        rows[10][col] = str(int(float(before or 0)) + 7)
        victim.unlink()  # break the hard link: the original is never written through
        with victim.open("w", newline="") as fh:
            csv.writer(fh, lineterminator="\n").writerows(rows)
        return f"{victim.name}: data row 10 kicks {before!r} -> {rows[10][col]!r}", legacy, None

    def run(self, name: str, mutate: Any) -> dict[str, Any]:
        work = self.out / name
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        link_tree(self.capture, work / "capture")
        if (self.base / "evidence").is_dir():
            link_tree(self.base / "evidence", work / "evidence")  # the pinned Brownlow award evidence
        for stale in ("checkpoint.sqlite", "checkpoint.sqlite-wal", "checkpoint.sqlite-shm"):
            (work / "capture" / stale).unlink(missing_ok=True)  # compare does not need the checkpoint
        link_tree(self.cache, work / "cache")
        description, legacy, cfg = mutate(work)
        orig_legacy = Path(self.plan.operational.legacy_root) if self.plan.operational.legacy_root else None
        (work / "MUTATION.txt").write_text(description + "\n")
        plan = inv.build_plan(
            data_root=Path(self.plan.operational.data_root),
            snapshot=self.plan.inputs.snapshot.snapshot_id,
            legacy_root=legacy or orig_legacy,
            through_date=self.plan.scope.through_date,
            scope="all",
            run_dir=work,
            config_dir=cfg,
        )
        plan_path = inv.write_plan(plan)
        opts = CP.CompareOptions(
            plan=plan_path,
            capture_manifest=work / "capture" / "manifest.json",
            out=work / "report",
            workers=2,
            cache=work / "cache",
            previous=self.previous,
            config_dir=cfg,
        )
        result, audit = CP.run_audit(opts)
        RP.write_report_dir(opts.out, result, CP.input_roots_of(audit.plan, audit.capture_dir))
        CP.cleanup(audit)
        prev = result.execution.get("previous", {})
        changed = prev.get("units_changed_or_new")
        return {
            "probe": name,
            "mutation": description,
            "overall": result.report["result"]["overall"],
            "layers": result.report["result"]["layers"],
            "units_total": prev.get("units_total"),
            "units_reused": prev.get("units_unchanged_reused_evidence"),
            "units_changed": changed,
            "findings": result.report["findings"]["count"],
            "report_sha256": sha((work / "report" / "report.json").read_bytes()),
            "plan_id": plan.plan_id,
            "capture_identity": plan.capture_identity,
        }


def main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--capture", help="capture directory (default RUN_DIR/capture)")
    ap.add_argument("--report", default="cold-4", help="report directory name under RUN_DIR/reports")
    ap.add_argument("--cache", default="cache-4", help="warm unit cache directory under RUN_DIR")
    a = ap.parse_args(argv[1:])
    base, out = Path(a.run_dir), Path(a.out_dir)
    capture = Path(a.capture) if a.capture else base / "capture"
    out.mkdir(parents=True, exist_ok=True)
    probe = Probe(base, out, capture, a.report, a.cache)

    def originals() -> dict[str, str]:
        return {
            "capture_objects": tree_digest(capture / "objects"),
            "capture_manifest": sha((capture / "manifest.json").read_bytes()),
            "plan": sha((base / "plan.json").read_bytes()),
            "previous_report": sha(probe.previous.read_bytes()),
            "cache": tree_digest(probe.cache),
        }

    before = originals()
    results = []
    for name, fn in (
        ("source-cell", probe.source_cell),
        ("availability", probe.availability),
        ("rule", probe.rule),
        ("local-row", probe.local_row),
    ):
        results.append(probe.run(name, fn))
        print(json.dumps(results[-1], sort_keys=True), flush=True)
    after = originals()
    summary = {"originals_untouched": before == after, "before": before, "after": after, "probes": results}
    (out / "probes-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print("originals untouched:", summary["originals_untouched"])
    return 0 if summary["originals_untouched"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
