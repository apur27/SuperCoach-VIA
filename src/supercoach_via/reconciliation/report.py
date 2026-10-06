"""Report writing: stable streams, spreadsheet-safe CSV, hashed outputs, completion marker (DESIGN section 9).

Canonical JSON (report, findings stream) uses sorted keys, compact separators and UTF-8; records are
written in a stable order and the report binds the stream's hash and count. Timings, cache state,
PIDs, absolute paths and worker counts live only in ``execution.json``. The ``output-manifest.json``
is written last and is the completion marker: if any write fails, no marker exists and the error
names exactly what was and was not written.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import stat
from pathlib import Path
from typing import Any

from supercoach_via.integrity.report import canonical_bytes
from supercoach_via.storage.snapshots import atomic_write_bytes

FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")
OUTPUTS = ("report.json", "findings.jsonl", "players.csv", "coverage.csv", "summary.md", "execution.json")


class OutputWriteError(OSError):
    def __init__(self, message: str, written: list[str], not_written: list[str]) -> None:
        super().__init__(message)
        self.written = written
        self.not_written = not_written


def neutralise(value: Any) -> Any:
    """Defuse a spreadsheet formula without changing the canonical JSON the CSV was derived from."""
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def atomic_copy(src: Path, dst: Path) -> None:
    import shutil

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _digest(payload: bytes | None, file: Path) -> dict[str, Any]:
    if payload is not None:
        return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    h = hashlib.sha256()
    size = 0
    with file.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return {"sha256": h.hexdigest(), "bytes": size}


def csv_bytes(rows: list[dict[str, Any]], columns: list[str]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for r in rows:
        w.writerow([neutralise(r.get(c, "")) for c in columns])
    return buf.getvalue().encode("utf-8")


def jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_bytes(r) for r in records)


def check_destination(out: Path, input_roots: list[Path]) -> None:
    """Refuse an output directory that aliases an input or already holds a completed report (T23)."""
    resolved = out.resolve()
    for root in input_roots:
        if resolved == root or root in resolved.parents or resolved in root.parents:
            raise OutputWriteError(f"output {out} overlaps the input root {root}; refused", [], list(OUTPUTS))
    if os.path.lexists(out):
        st = os.lstat(out)
        if stat.S_ISLNK(st.st_mode):
            raise OutputWriteError(f"output {out} is a symlink; refused", [], list(OUTPUTS))
        if (out / "output-manifest.json").exists():
            raise OutputWriteError(f"{out} already holds a completed report; refused", [], list(OUTPUTS))
        for name in (*OUTPUTS, "output-manifest.json"):
            p = out / name
            if os.path.lexists(p):
                pst = os.lstat(p)
                if stat.S_ISLNK(pst.st_mode) or pst.st_nlink > 1:
                    raise OutputWriteError(f"{p} is a symlink or hard-link alias; refused", [], list(OUTPUTS))


def summary_text(report: dict[str, Any], execution: dict[str, Any]) -> str:
    res = report["result"]
    lines = [
        "# AFL Tables reconciliation",
        "",
        f"Overall result: **{res['overall']}**",
        "",
        "| Layer | Verdict |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(res["layers"].items())],
        "",
        f"Plan `{report['plan_id'][:16]}`, snapshot `{report['snapshot_id'][:23]}`, "
        f"boundary {report['scope']['through_date']} (reference mode {report['scope']['reference_mode']}).",
        f"Full population: {report['scope']['full_population']}; "
        f"capture complete: {report['source']['capture_complete']}.",
        f"Findings: {report['findings']['count']} (stream sha256 `{report['findings']['sha256'][:16]}`).",
        "",
    ]
    for name, doc in sorted(report["layers"].items()):
        lines.append(f"## {name}: {doc['verdict']}")
        for r in doc["unknown_reasons"]:
            lines.append(f"- unresolved: {r}")
        fails = {k: n for k, n in doc["finding_counts"].items() if k.endswith("|fail")}
        for k, n in sorted(fails.items()):
            lines.append(f"- {k.split('|')[0]}: {n}")
        lines.append("")
    lf = report.get("latest_completed_final")
    if lf:
        lines.append(
            f"Latest completed final before the boundary: {lf['stage']} on {lf['date']}; "
            f"present locally: {lf['present_locally']}."
        )
    lines.append("")
    lines.append(f"Accounting identities hold: {execution.get('accounting_identities_hold')}.")
    return "\n".join(lines) + "\n"


PLAYER_COLUMNS = [
    "layer",
    "source_url",
    "name",
    "local_ids",
    "status",
    "appearances_expected",
    "appearances_matched",
    "appearances_missing",
    "appearances_unresolved",
    "appearances_local_only",
    "cells_expected",
    "cells_equal",
    "cells_mismatch",
    "cells_source_unavailable",
    "cells_not_applicable",
    "cells_unresolved",
    "findings_fail",
    "findings_unknown",
]
COVERAGE_COLUMNS = ["layer", "season", "statistic", "metric", "count"]


def write_report_dir(out: Path, result: Any, input_roots: list[Path]) -> dict[str, Any]:
    """Write every output, then the completion manifest. Returns the manifest document."""
    check_destination(out, input_roots)
    out.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, bytes] = {
        "report.json": canonical_bytes(result.report),
        "players.csv": csv_bytes(sorted(result.players, key=lambda r: (r["layer"], r["source_url"])), PLAYER_COLUMNS),
        "coverage.csv": csv_bytes(
            sorted(result.coverage, key=lambda r: (r["layer"], str(r["season"]).zfill(6), r["statistic"], r["metric"])),
            COVERAGE_COLUMNS,
        ),
        "summary.md": summary_text(result.report, result.execution).encode("utf-8"),
        "execution.json": canonical_bytes(result.execution),
    }
    findings_file = Path(result.findings_path)
    written: list[str] = []
    try:
        for name in OUTPUTS:
            if name == "findings.jsonl":
                atomic_copy(findings_file, out / name)
            else:
                atomic_write_bytes(out / name, payloads[name])
            written.append(name)
    except OSError as exc:
        raise OutputWriteError(
            f"writing {name} failed: {exc}", written, [n for n in OUTPUTS if n not in written]
        ) from exc
    result.findings_path = out / "findings.jsonl"  # the work copy is disposable once the output exists
    manifest = {
        "kind": "afltables-reconciliation-output-manifest",
        "files": {n: _digest(payloads.get(n), findings_file) for n in OUTPUTS},
        "records": {
            "findings.jsonl": result.findings_count,
            "players.csv": len(result.players),
            "coverage.csv": len(result.coverage),
        },
        "report_sha256": hashlib.sha256(payloads["report.json"]).hexdigest(),
        "canonical": [n for n in OUTPUTS if n != "execution.json"],
    }
    try:
        atomic_write_bytes(out / "output-manifest.json", canonical_bytes(manifest))
    except OSError as exc:
        raise OutputWriteError(
            f"writing output-manifest.json failed: {exc}", written, ["output-manifest.json"]
        ) from exc
    return manifest


def read_previous_units(path: Path) -> dict[str, str] | None:
    try:
        doc = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None
    units = doc.get("unit_digests")
    return units if isinstance(units, dict) else None
