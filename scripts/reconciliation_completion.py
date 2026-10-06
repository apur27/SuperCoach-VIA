#!/usr/bin/env python3
"""Write ``implementation-validation.json`` and ``completion.json`` for a reconciliation run (Surveyor B5).

Nothing here is typed in by hand: every validation step is RUN, its status computed from the exit code or from
pytest's JUnit report (a failure or error can never read PASS; a failure is BASELINE_FAIL only when the same test is
listed with evidence that it fails identically at the base commit); the implementation status is
``schema.implementation_status`` of those steps; the data verdicts are the final report's per-layer verdicts; timings,
memory and hashes are read from the measure files and output manifests. Both records validate against the
repository's own pydantic models before they are written.

Usage::

    reconciliation_completion.py RUN_DIR --base-commit SHA [--prefix final-] [--integrity DIR/stdout.json]
        [--baseline TEST_ID=EVIDENCE ...] [--model-requested TEXT --model-resolved ID] [--skip-checks]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from supercoach_via.reconciliation import schema as S

REPO = Path(__file__).resolve().parents[1]
RUNS = (
    ("cold-1", 1, "cold (empty)"),
    ("cold-4", 4, "cold (empty)"),
    ("warm-4", 4, "warm (cold-4 units)"),
    ("changed", 2, "warm + --previous cold-4"),
)


def pytest_step(
    name: str, command: str, junit_xml: str, summary: str, *, baseline: dict[str, str] | None = None
) -> S.ValidationStep:
    """A test step from pytest's JUnit XML (counts, skip reasons, failing ids) and its terminal summary (deselected)."""
    try:
        root = ET.fromstring(junit_xml)  # noqa: S314 - pytest's own JUnit output
    except ET.ParseError as exc:
        raise ValueError(f"{name}: not a JUnit report: {exc}") from exc
    cases = root.iter("testcase")
    counts: Counter[str] = Counter()
    skips: Counter[str] = Counter()
    failing: list[str] = []
    for case in cases:
        tid = f"{case.get('classname', '')}::{case.get('name', '')}"
        if case.find("failure") is not None:
            counts["failed"] += 1
            failing.append(tid)
        elif case.find("error") is not None:
            counts["errors"] += 1
            failing.append(tid)
        elif (sk := case.find("skipped")) is not None:
            counts["skipped"] += 1
            skips[sk.get("message") or "no reason given"] += 1
        else:
            counts["passed"] += 1
    if root.tag not in ("testsuites", "testsuite") or not sum(counts.values()):
        raise ValueError(f"{name}: the JUnit report holds no test cases; a tier that ran nothing is not a pass")
    m = re.search(r"(\d+) deselected", summary)
    tc = S.TestCounts(
        passed=counts["passed"],
        failed=counts["failed"],
        errors=counts["errors"],
        skipped=counts["skipped"],
        deselected=int(m.group(1)) if m else 0,
    )
    detail = (
        f"{tc.passed} passed, {tc.failed} failed, {tc.errors} errors, {tc.skipped} skipped, {tc.deselected} deselected"
    )
    base = baseline or {}
    if not failing:
        return S.ValidationStep(
            name=name, command=command, status="PASS", detail=detail, counts=tc, skip_reasons=dict(skips)
        )
    if all(t in base for t in failing):
        evidence = "; ".join(f"{t}: {base[t]}" for t in failing)
        return S.ValidationStep(
            name=name,
            command=command,
            status="BASELINE_FAIL",
            detail=detail,
            counts=tc,
            skip_reasons=dict(skips),
            baseline_evidence=evidence,
        )
    new = [t for t in failing if t not in base]
    return S.ValidationStep(
        name=name,
        command=command,
        status="FAIL",
        detail=f"{detail}; new: {new[:5]}",
        counts=tc,
        skip_reasons=dict(skips),
    )


def command_step(name: str, command: str, returncode: int, output: str) -> S.ValidationStep:
    """A non-test step: PASS exactly when the command exits 0; the detail is its last output line."""
    lines = [ln for ln in output.strip().splitlines() if ln.strip()]
    return S.ValidationStep(
        name=name,
        command=command,
        status="PASS" if returncode == 0 else "FAIL",
        detail=lines[-1] if lines else f"exit {returncode}",
    )


def validation(steps: list[S.ValidationStep], *, branch: str, base_commit: str) -> S.ImplementationValidation:
    libs = {"python": sys.version.split()[0]}
    for n in ("pydantic", "pytest", "ruff", "mypy", "typer", "pandas", "pyarrow", "duckdb", "lxml", "httpx"):
        try:
            libs[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            continue
    return S.ImplementationValidation(
        created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        branch=branch,
        base_commit=base_commit,
        libraries=libs,
        steps=steps,
        baseline_failures=[s.baseline_evidence or "" for s in steps if s.status == "BASELINE_FAIL"],
        new_failures=[f"{s.name}: {s.detail}" for s in steps if s.status == "FAIL"],
    )


def derived_statuses(
    iv: S.ImplementationValidation, report: dict[str, Any]
) -> tuple[Literal["COMPLETE", "INCOMPLETE"], dict[str, Literal["PASS", "FAIL", "UNKNOWN"]], str]:
    """(implementation status, per-layer data verdicts, overall data verdict), all computed."""
    data: dict[str, Literal["PASS", "FAIL", "UNKNOWN"]] = {}
    for layer, v in report["layers"].items():
        verdict = v["verdict"]
        if verdict not in ("PASS", "FAIL", "UNKNOWN"):
            raise ValueError(f"layer {layer}: unexpected verdict {verdict!r}")
        data[layer] = verdict
    overall = "FAIL" if "FAIL" in data.values() else "UNKNOWN" if "UNKNOWN" in data.values() else "PASS"
    return S.implementation_status(iv), data, overall


def _run(name: str, argv: list[str], *, junit: Path | None = None, baseline: dict[str, str]) -> S.ValidationStep:
    cmd = " ".join(argv)
    res = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)  # noqa: S603 - fixed argv
    if junit is None:
        return command_step(name, cmd, res.returncode, res.stdout + res.stderr)
    return pytest_step(
        name, cmd, junit.read_text(), res.stdout.strip().splitlines()[-1] if res.stdout else "", baseline=baseline
    )


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--base-commit", required=True)
    ap.add_argument("--branch", default="work/afltables-reconciliation")
    ap.add_argument("--prefix", default="final-", help="report/measure name prefix of the four compare runs")
    ap.add_argument("--integrity", type=Path, help="stdout.json of the separate scvia check-integrity run")
    ap.add_argument("--baseline", action="append", default=[], metavar="TEST_ID=EVIDENCE")
    ap.add_argument("--model-requested", default="unrecorded")
    ap.add_argument("--model-resolved", default=None)
    ap.add_argument("--note", action="append", default=[], help="an open item (stated, not a status)")
    a = ap.parse_args(argv[1:])
    run: Path = a.run_dir
    baseline = dict(b.split("=", 1) for b in a.baseline)
    py = sys.executable
    out = run / "validation"
    out.mkdir(exist_ok=True)
    steps = [
        _run("ruff", [py, "-m", "ruff", "check", "src", "tests", "scripts"], baseline=baseline),
        _run(
            "ruff format",
            [py, "-m", "ruff", "format", "--check", "src/supercoach_via/reconciliation", "scripts"],
            baseline=baseline,
        ),
        _run("mypy", [py, "-m", "mypy"], baseline=baseline),
        _run(
            "scvia fast tier",
            [
                py,
                "-m",
                "pytest",
                "tests/scvia",
                "-m",
                "not integration",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={out / 'scvia.xml'}",
            ],
            junit=out / "scvia.xml",
            baseline=baseline,
        ),
        _run(
            "legacy fast tier",
            [
                py,
                "-m",
                "pytest",
                "tests/unit",
                "-m",
                "not integration",
                "-q",
                "-p",
                "no:cacheprovider",
                f"--junitxml={out / 'legacy.xml'}",
            ],
            junit=out / "legacy.xml",
            baseline=baseline,
        ),
    ]
    iv = validation(steps, branch=a.branch, base_commit=a.base_commit)
    (run / "implementation-validation.json").write_bytes(S.canonical_dump(iv))

    reports: dict[str, S.ReportRef] = {}
    for name, workers, cache in RUNS:
        rdir, meas = run / "reports" / f"{a.prefix}{name}", run / "measure" / f"{a.prefix}{name}.json"
        m, man = json.loads(meas.read_text()), json.loads((rdir / "output-manifest.json").read_text())
        reports[name] = S.ReportRef(
            path=str(rdir),
            report_sha256=man["report_sha256"],
            findings_sha256=man["files"]["findings.jsonl"]["sha256"],
            workers=workers,
            cache=cache,
            wall_seconds=m["wall_seconds"],
            peak_rss_mib=m["peak_tree_rss_mib"],
        )
    identical = (
        len({r.report_sha256 for r in reports.values()}) == 1
        and len({r.findings_sha256 for r in reports.values()}) == 1
    )
    primary = json.loads((Path(reports["cold-4"].path) / "report.json").read_text())
    impl, data, overall = derived_statuses(iv, primary)
    rep_sha = reports["cold-4"].report_sha256
    layers = {
        k: S.LayerOutcome(status=v, report_sha256=rep_sha, note=f"report.json layers.{k}") for k, v in data.items()
    }
    inputs = {
        "snapshot_id": primary["snapshot_id"],
        "plan_id": primary["plan_id"],
        "capture_identity": primary["capture_identity"],
        "capture_manifest_sha256": primary["capture_manifest_sha256"],
        "legacy_content_sha256": primary["legacy_inputs"]["content_sha256"],
    }
    if a.integrity:
        integ = json.loads(a.integrity.read_text().splitlines()[0])
        layers["release"] = S.LayerOutcome(
            status=integ["outcome"],
            report_sha256=integ["report_sha256"],
            note="scvia check-integrity --scope full; a separate claim",
        )
        inputs["release_id"] = integ["release_id"]
    probes = run / "probes" / "probes-summary.json"
    completion = S.Completion(
        created_utc=iv.created_utc,
        model=S.ModelUse(requested=a.model_requested, resolved=a.model_resolved, resolution_source="--model-* flags"),
        design=S.LayerOutcome(
            status="COMPLETE",
            report_sha256=None,
            note="docs/reviews/AFLTABLES_RECONCILIATION_DESIGN_REVIEW.md; commit 09ac3a750",
        ),
        implementation=S.LayerOutcome(
            status=impl,
            report_sha256=_sha(run / "implementation-validation.json"),
            note="derived from implementation-validation.json steps",
        ),
        execution=S.LayerOutcome(
            status="COMPLETE" if identical else "INCOMPLETE",
            report_sha256=rep_sha,
            note=f"four compare runs byte-identical: {identical}",
        ),
        data=layers,
        overall_data_verdict=S.Verdict(overall),
        inputs=inputs,
        reports=reports,
        reproducibility={
            "runs_byte_identical": str(identical).lower(),
            "report_sha256": rep_sha,
            "findings_sha256": reports["cold-4"].findings_sha256,
            "mutation_probes": _sha(probes) if probes.exists() else "not run",
            "seed": "none: the program has no stochastic step",
        },
        acceptance=S.LayerOutcome(status="PENDING", report_sha256=None, note="Opus Surveyor + QA + Gaffer"),
        open_items=a.note,
    )
    (run / "completion.json").write_bytes(S.canonical_dump(completion))
    print(
        json.dumps(
            {
                "implementation": impl,
                "data": data,
                "overall": overall,
                "identical": identical,
                "steps": {s.name: s.status for s in steps},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
