#!/usr/bin/env python3
"""Weekly AFL Tables reconciliation gate for the seasons a scrape changed (legacy CSV layer).

Runs in the weekly harness after the Phase 1 scrape is committed locally and before it is pushed:

1. the seasons whose rows in ``data/player_data`` or ``data/matches`` (working tree) differ from ``--base``
   (default ``origin/main``);
2. a seasons-scoped plan, a polite capture of those seasons' AFL Tables pages (one request at a time, at least two
   seconds apart), and an offline comparison;
3. a decision on the legacy layer:

   * ``block`` (exit 1): a confirmed discrepancy (FAIL), or an identity conflict/unresolved player (a stub or
     duplicate player file);
   * ``warn`` (exit 0): the capture could not complete (network, rate limit) or AFL Tables itself is incomplete,
     so nothing can be confirmed either way. Like the match-completeness gate, this fails OPEN, loudly;
   * ``pass`` (exit 0).

With ``--fix`` a blocking result first gets the source-backed corrections (``reconciliation.corrections``) applied
to the legacy CSVs, and the seasons are audited again on the corrected files; the gate passes only if that second
audit passes. The changed files are left for the harness to commit; nothing here commits or pushes.

The pinned Brownlow award evidence is the committed fixture whose SHA-256 the rules file pins, so no extra request
is made for it.

Usage: ``reconciliation_gate.py --legacy-root REPO --data-root DATA [--season N ...] [--base origin/main] [--fix]``
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EVIDENCE_FIXTURE = REPO / "tests" / "scvia" / "fixtures" / "reconciliation" / "brownlow_idx.html"
_ROW_YEAR = {"player_data": 1, "matches": 3}  # column holding the season in each legacy CSV kind
#: finding categories whose UNKNOWN still blocks: a player the local layer represents twice or not uniquely
_BLOCKING_UNKNOWN = ("IDENTITY_CONFLICT", "IDENTITY_UNRESOLVED")


def changed_seasons(diff: str) -> list[int]:
    """Seasons of every added or removed row in a ``git diff -U0`` of the legacy CSV directories."""
    seasons: set[int] = set()
    kind: str | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            m = re.search(r" b/data/(player_data|matches)/", line)
            kind = m.group(1) if m else None
            continue
        if kind is None or line.startswith(("+++", "---")) or line[:1] not in ("+", "-"):
            continue
        cells = line[1:].split(",")
        col = _ROW_YEAR[kind]
        if len(cells) > col and re.fullmatch(r"(18|19|20)\d{2}", cells[col].strip()):
            seasons.add(int(cells[col]))
    return sorted(seasons)


def seasons_since(legacy_root: Path, base: str) -> list[int]:
    """Seasons changed between ``base`` and the WORKING TREE (committed, staged or not), so a run whose commit was
    suppressed (the smoke harness) audits exactly what a production run that committed would."""
    argv = ["git", "-C", str(legacy_root), "diff", "-U0", base, "--", "data/player_data", "data/matches"]
    diff = subprocess.run(argv, capture_output=True, text=True, check=True).stdout  # noqa: S603 - fixed argv
    return changed_seasons(diff)


def decide(report: dict[str, Any]) -> tuple[str, str]:
    """(pass | warn | block, reason) for the legacy CSV layer of a completed report."""
    layer = report["layers"].get("legacy_csv")
    if layer is None:
        return "warn", "no legacy layer in the report"
    verdict = layer["verdict"]
    counts = layer.get("finding_counts", {})
    if verdict == "FAIL":
        fails = {k.split("|")[0]: n for k, n in counts.items() if k.endswith("|fail")}
        return "block", f"confirmed discrepancies: {fails}"
    blocking = {k.split("|")[0]: n for k, n in counts.items() if k.split("|")[0] in _BLOCKING_UNKNOWN}
    if blocking:
        return "block", f"identity problems: {blocking}"
    if verdict == "PASS":
        return "pass", "the changed seasons agree with AFL Tables"
    if not report.get("source", {}).get("capture_complete", True):
        return "warn", "the capture did not complete; nothing could be confirmed (fails open)"
    return "warn", f"AFL Tables records are incomplete for some cells: {layer.get('unknown_reasons', [])}"


def _audit(plan_path: Path, run_dir: Path, out: str) -> dict[str, Any]:
    from supercoach_via.reconciliation import compare as CP
    from supercoach_via.reconciliation import report as RP

    opts = CP.CompareOptions(plan=plan_path, capture_manifest=run_dir / "capture" / "manifest.json",
                             out=run_dir / "reports" / out, cache=run_dir / "cache")  # fmt: skip
    result, audit = CP.run_audit(opts)
    try:
        RP.write_report_dir(opts.out, result, CP.input_roots_of(audit.plan, audit.capture_dir))
    finally:
        CP.cleanup(audit)
    report: dict[str, Any] = json.loads((opts.out / "report.json").read_text())
    return report


def run_gate(
    *,
    seasons: list[int],
    run_dir: Path,
    data_root: Path,
    legacy_root: Path,
    through_date: str,
    fix: bool,
    client: Any = None,
    clock: Any = None,
) -> dict[str, Any]:
    from supercoach_via.ingest.http import RawArchive
    from supercoach_via.reconciliation import corrections as CO
    from supercoach_via.reconciliation import inventory as inv
    from supercoach_via.reconciliation.capture import Capture, SystemClock

    run_dir.mkdir(parents=True, exist_ok=False)
    plan = inv.build_plan(data_root=data_root, snapshot="current", legacy_root=legacy_root, through_date=through_date,
                          scope="seasons", seasons=seasons, run_dir=run_dir)  # fmt: skip
    plan_path = inv.write_plan(plan)
    RawArchive(run_dir / "evidence").put(EVIDENCE_FIXTURE.read_bytes())
    if client is None:
        from supercoach_via.reconciliation.cli import make_client
        from supercoach_via.reconciliation.urls import load_reconciliation_policies

        client = make_client(load_reconciliation_policies())
    cap = Capture(plan, run_dir, client, clock=clock or SystemClock()).run()
    out: dict[str, Any] = {"seasons": seasons, "plan_id": plan.plan_id, "capture": cap.state, "fixed": None}
    if not (run_dir / "capture" / "manifest.json").exists():
        out.update(decision="warn", reason=f"capture produced no manifest ({cap.state}); fails open", layers={})
        return _write(run_dir, out)
    report = _audit(plan_path, run_dir, "first")
    decision, reason = decide(report)
    if decision == "block" and fix:
        from supercoach_via.reconciliation import compare as CP
        from supercoach_via.reconciliation.propose import propose

        prop = run_dir / "corrections"
        opts = CP.CompareOptions(plan=plan_path, capture_manifest=run_dir / "capture" / "manifest.json",
                                 out=prop / "_audit", cache=run_dir / "cache")  # fmt: skip
        summary = propose(opts, run_dir / "reports" / "first", prop)
        changes = [c for c in CO.read_changes(prop / "changes.jsonl") if c.layer == "legacy_csv"]
        out["fixed"] = CO.apply_legacy(legacy_root, changes)
        out["unsupported"] = summary["unsupported_fail_findings"]
        plan2 = inv.build_plan(data_root=data_root, snapshot="current", legacy_root=legacy_root,
                               through_date=through_date, scope="seasons", seasons=seasons,
                               run_dir=run_dir)  # fmt: skip
        report = _audit(inv.write_plan(plan2), run_dir, "after-fix")
        decision, reason = decide(report)
        reason = f"after source-backed corrections: {reason}"
    out.update(decision=decision, reason=reason, layers=report["result"]["layers"])
    return _write(run_dir, out)


def _write(run_dir: Path, out: dict[str, Any]) -> dict[str, Any]:
    (run_dir / "gate.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    return out


def _record(
    path: Path, decision: str, reason: str, seasons: list[int], pending: list[int], run_dir: str | None
) -> None:
    """Persist every exit (Gaffer H1): the harness logs from this, and the next run reads ``pending``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"decision": decision, "reason": reason, "seasons": seasons, "pending": sorted(set(pending)),
           "run_dir": run_dir, "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}  # fmt: skip
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def _pending(path: Path) -> list[int]:
    """Seasons an earlier run could not verify (skip, warn or block): audited again until one passes (M4)."""
    if not path.is_file():
        return []
    return [int(x) for x in json.loads(path.read_text()).get("pending", [])]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--legacy-root", type=Path, default=REPO)
    ap.add_argument("--data-root", type=Path, required=True, help="accepted snapshot data root (pinned, read only)")
    ap.add_argument("--season", type=int, action="append", help="audit these seasons (default: what changed)")
    ap.add_argument("--base", default="origin/main", help="git tree-ish the scraped working tree is compared against")
    ap.add_argument("--runs-root", type=Path, default=REPO / "var" / "reconciliations" / "afltables" / "gate")
    ap.add_argument("--through-date", default=date.today().isoformat())
    ap.add_argument("--fix", action="store_true", help="apply source-backed legacy corrections, then re-audit")
    ap.add_argument(
        "--status-file", type=Path, help="default: <legacy-root>/.claude/audit/reconciliation_gate_status.json"
    )
    a = ap.parse_args(argv[1:])
    status = a.status_file or a.legacy_root / ".claude" / "audit" / "reconciliation_gate_status.json"
    pending = _pending(status)
    changed = sorted(set(a.season or [])) or seasons_since(a.legacy_root, a.base)
    seasons = sorted(set(changed) | set(pending))
    if not (a.data_root / "current.json").is_file():
        reason = f"no accepted snapshot at {a.data_root}; gate skipped (fails open)"
        _record(status, "skip", reason, seasons, seasons, None)
        print(f"reconciliation gate: WARN {reason}; seasons {seasons} stay pending")
        return 0
    if not seasons:
        _record(status, "noop", "no legacy season changed and none pending", [], [], None)
        print("reconciliation gate: no legacy season changed; nothing to audit")
        return 0
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = a.runs_root / f"{stamp}-{'-'.join(map(str, seasons))}"
    out = run_gate(seasons=seasons, run_dir=run_dir, data_root=a.data_root, legacy_root=a.legacy_root,
                   through_date=a.through_date, fix=a.fix)  # fmt: skip
    _record(status, out["decision"], out["reason"], seasons, [] if out["decision"] == "pass" else seasons, str(run_dir))
    print(f"reconciliation gate: {out['decision'].upper()} for seasons {seasons}: {out['reason']} ({run_dir})")
    return 1 if out["decision"] == "block" else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
