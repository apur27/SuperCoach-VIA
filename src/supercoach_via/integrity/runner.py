"""Run an integrity audit: pin inputs, run the requested checks, build the canonical report.

``run_audit`` never writes into its inputs. It returns the sealed canonical report and a
separate execution-metadata document; ``write_outputs`` writes both atomically outside
every input root.
"""

from __future__ import annotations

import hashlib
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from supercoach_via.integrity import checks_storage
from supercoach_via.integrity.capture import EvidenceStore, ReleaseCapture, SnapshotCapture
from supercoach_via.integrity.context import AuditContext, CheckSkipped, CheckSpec
from supercoach_via.integrity.policy import IntegrityPolicy, load_policy
from supercoach_via.integrity.report import (
    REPORT_SCHEMA,
    Collector,
    Finding,
    Outcome,
    RuleSpec,
    Status,
    canonical_bytes,
    exit_code,
    overall,
    seal_report,
)

CHECKER_VERSION = "scvia-integrity/1"
SCOPES = ("full", "data")
#: modules whose logic decides verdicts; their bytes form the checker's code identity
_CODE_FILES = (
    "integrity",
    "ingest/reconcile.py",
    "publish/release.py",
    "publish/view_models.py",
    "domain/schemas.py",
    "storage/snapshots.py",
)


class UsageError(ValueError):
    """Invalid options (exit 2)."""


class CheckerError(RuntimeError):
    """The checker itself failed (exit 9). Never reported as a verdict."""


@dataclass(frozen=True)
class AuditOptions:
    data_root: Path
    snapshot: str = "current"
    release_dir: Path | None = None
    scope: str = "full"
    as_of: str | None = None
    evidence_dirs: tuple[Path, ...] = ()
    models_root: Path | None = None
    predictions_root: Path | None = None
    workers: int = 1
    sample_limit: int | None = None
    keep_all: bool = False
    #: restrict to these families (tests, diagnosis); the report then says it is partial
    families: tuple[str, ...] | None = None
    #: restrict to these check ids (tests target one check); the report then says it is partial
    checks: tuple[str, ...] | None = None
    config_dir: Path | None = None
    cache_dir: Path | None = None
    #: produce the complete findings stream (every finding, not only the capped sample)
    findings_stream: bool = False
    #: a prior report whose inputs this run is compared with (changed-data mode; needs cache_dir)
    changed_since: Path | None = None


@dataclass
class AuditResult:
    outcome: Outcome
    report: dict[str, Any]
    findings: list[Finding]
    execution: dict[str, Any]
    exit_code: int
    all_findings: list[Finding] | None = field(default=None, repr=False)
    stream_bytes: bytes | None = field(default=None, repr=False)


def registry() -> list[CheckSpec]:
    from supercoach_via.integrity import checks_data, checks_models, checks_release, checks_source

    return [
        *checks_storage.CHECKS,
        *checks_data.CHECKS,
        *checks_source.CHECKS,
        *checks_release.CHECKS,
        *checks_models.CHECKS,
    ]


def rule_catalog() -> dict[str, RuleSpec]:
    from supercoach_via.integrity import checks_data, checks_models, checks_release, checks_source

    out: dict[str, RuleSpec] = {}
    for spec in [
        *checks_storage.RULES,
        *checks_data.RULES,
        *checks_source.RULES,
        *checks_release.RULES,
        *checks_models.RULES,
    ]:
        if spec.rule_id in out:
            raise CheckerError(f"duplicate rule id {spec.rule_id}")
        out[spec.rule_id] = spec
    return out


def code_identity() -> dict[str, str]:
    base = Path(__file__).resolve().parents[1]
    files: list[Path] = []
    for rel in _CODE_FILES:
        p = base / rel
        files.extend(sorted(p.glob("*.py")) if p.is_dir() else [p])
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(base).as_posix().encode() + b"\0" + hashlib.sha256(p.read_bytes()).digest())
    return {"version": CHECKER_VERSION, "code_sha256": h.hexdigest()}


def _rules_sha256(rules: dict[str, RuleSpec]) -> str:
    body = [
        [r.rule_id, r.check_id, r.severity.value, r.kind.value, r.current_blocks, r.summary, r.action]
        for r in sorted(rules.values(), key=lambda r: r.rule_id)
    ]
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def parse_as_of(text: str | None) -> datetime | None:
    if text is None:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise UsageError(f"--as-of must be an ISO 8601 instant, got {text!r}") from exc
    if dt.tzinfo is None:
        raise UsageError("--as-of must carry a UTC offset, e.g. 2026-09-28T12:00:00Z")
    return dt.astimezone(UTC)


def _current_season(snap: SnapshotCapture | None) -> int | None:
    if snap is None:
        return None
    t = snap.table("matches", ["season", "status"])
    if t is None:
        return None
    seasons = [s for s, st in zip(t["season"].to_pylist(), t["status"].to_pylist(), strict=True) if st == "complete"]
    return max(seasons) if seasons else None


def _requested(options: AuditOptions, checks: list[CheckSpec]) -> list[CheckSpec]:
    out = []
    for c in checks:
        if options.scope == "data" and "release" in c.needs:
            continue
        if options.families is not None and c.family not in options.families:
            continue
        if options.checks is not None and c.check_id not in options.checks:
            continue
        out.append(c)
    return out


def _supplied(ctx: AuditContext, need: str) -> bool:
    return {
        "snapshot": ctx.snapshot is not None,
        "release": ctx.release is not None,
    }.get(need, True)


def run_audit(options: AuditOptions) -> AuditResult:
    t0 = time.perf_counter()
    started = datetime.now(UTC)
    if options.scope not in SCOPES:
        raise UsageError(f"--scope must be one of {', '.join(SCOPES)}")
    if options.workers < 1:
        raise UsageError("--workers must be at least 1")
    as_of = parse_as_of(options.as_of)
    try:
        policy = load_policy(options.config_dir)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc
    rules = rule_catalog()
    snapshot = SnapshotCapture.open(options.data_root, options.snapshot)
    release = ReleaseCapture.open(options.release_dir) if options.release_dir is not None else None
    evidence = EvidenceStore([options.data_root / "raw", *options.evidence_dirs])
    current = _current_season(snapshot)
    if options.sample_limit is not None and options.sample_limit < 1:
        raise UsageError("--sample-limit must be at least 1")
    keep_all = options.keep_all or options.findings_stream
    collector = Collector(
        rules,
        sample_limit=options.sample_limit or policy.sample_limit,
        current_season=current,
        exceptions=policy.exceptions,
        keep_all=keep_all,
    )
    ctx = AuditContext(
        collector=collector,
        policy=policy,
        snapshot=snapshot,
        release=release,
        evidence=evidence,
        as_of=as_of,
        current_season=current,
        models_root=options.models_root or options.data_root / "models",
        predictions_root=options.predictions_root or options.data_root / "predictions",
        workers=options.workers,
    )
    salt = _cache_salt(policy, rules, options, as_of)
    changed = _baseline(options, snapshot, release, salt) if options.changed_since is not None else None
    if options.cache_dir is not None and (changed is None or changed["compatible"]):
        from supercoach_via.integrity.cache import SemanticCache

        ctx.cache = SemanticCache(options.cache_dir, salt)
    results = []
    seconds: dict[str, float] = {"capture": round(time.perf_counter() - t0, 3)}
    try:
        for spec in _requested(options, registry()):
            ctx.check_id = spec.check_id
            t_check = time.perf_counter()
            unknown: list[str] = []
            missing = [n for n in spec.needs if not _supplied(ctx, n)]
            if missing:
                status, reason = Status.UNKNOWN, f"input not supplied: {', '.join(missing)}"
            else:
                try:
                    unknown = list(spec.fn(ctx) or [])
                    status = collector.check_status(spec.check_id, unknown=unknown)
                    reason = "; ".join(unknown)[:500] if unknown else ""
                except CheckSkipped as skip:
                    status, reason = skip.status, skip.reason
                    if collector.check_status(spec.check_id, unknown=[]) is Status.FAIL:
                        status = Status.FAIL
            results.append((spec, status, reason))
            seconds[spec.check_id] = round(time.perf_counter() - t_check, 3)
        collector.finish_exceptions()
    except CheckSkipped as exc:  # pragma: no cover - raised outside a check body
        raise CheckerError(str(exc)) from exc
    except (UsageError, CheckerError):
        raise
    except Exception as exc:
        raise CheckerError(f"check {ctx.check_id} crashed: {type(exc).__name__}: {exc}") from exc
    finally:
        ctx.close()
    t_drift = time.perf_counter()
    drift = snapshot.drift() + (release.drift() if release is not None else [])
    seconds["inputs.stable"] = round(time.perf_counter() - t_drift, 3)
    stability = CheckSpec("inputs.stable", "storage", "inputs unchanged on disk for the whole audit", lambda _c: None)
    results.append(
        (
            stability,
            Status.UNKNOWN if drift else Status.PASS,
            f"{len(drift)} input(s) changed during the audit" if drift else "",
        )
    )
    report = _build_report(options, policy, rules, ctx, results, as_of)
    stream_bytes = None
    if options.findings_stream:
        every = collector.all_findings()
        stream_bytes = b"".join(canonical_bytes(f.as_dict()) for f in every)
        report["findings_stream"] = {
            "count": len(every),
            "sha256": hashlib.sha256(stream_bytes).hexdigest(),
            "complete": collector.stream_complete,
            "unlisted_by_producer": sum(r["unlisted_by_producer"] for r in report["rules"]),
        }
        report = seal_report(report)
    from supercoach_via.integrity.report_schema import validate

    try:
        validate(report)
    except ValueError as exc:
        raise CheckerError(f"report breaks its own contract: {exc}") from exc
    outcome = Outcome(report["outcome"])
    rss_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    execution = {
        "schema": "scvia.integrity-execution/1",
        "report_sha256": report["report_sha256"],
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "pid": os.getpid(),
        "workers": options.workers,
        "paths": {
            "data_root": str(options.data_root.resolve()),
            "release_dir": str(options.release_dir.resolve()) if options.release_dir else None,
            "evidence_dirs": [str(p.resolve()) for p in options.evidence_dirs],
            "config_dir": str(policy.config_dir),
        },
        "peak_rss_mib": {"self": round(rss_self / 1024, 1), "children": round(rss_children / 1024, 1)},
        "input_drift": drift,
        "cache": ctx.cache.stats() if ctx.cache is not None else None,
        "changed": changed,
        "seconds": seconds,
    }
    if drift:
        # the verdict above is about the pinned bytes; those bytes are no longer on disk
        execution["warning"] = "inputs changed during the audit; re-run against a stable copy"
    findings = collector.samples()
    return AuditResult(
        outcome=outcome,
        report=report,
        findings=collector.all_findings() if options.keep_all else findings,
        execution=execution,
        exit_code=exit_code(outcome),
        all_findings=collector.all_findings() if keep_all else None,
        stream_bytes=stream_bytes,
    )


def _baseline(
    options: AuditOptions, snapshot: SnapshotCapture, release: ReleaseCapture | None, salt: str
) -> dict[str, Any]:
    """Compare with a prior report. Only a verified, compatible baseline enables cache reuse.

    The result of a changed-data run is always the full verdict: the baseline only decides
    whether semantic results cached under identical inputs may be reused. A missing,
    corrupt or incompatible baseline falls back to recomputing everything.
    """
    from supercoach_via.integrity.capture import strict_json
    from supercoach_via.integrity.report import verify_report_digest

    assert options.changed_since is not None
    out: dict[str, Any] = {"baseline": str(options.changed_since), "compatible": False, "reason": None}
    if options.cache_dir is None:
        raise UsageError("--changed-since needs --cache DIR (the semantic results to reuse)")
    try:
        prior = strict_json(options.changed_since.read_bytes())
    except (OSError, ValueError) as exc:
        out["reason"] = f"baseline unreadable: {exc}"[:300]
        return out
    if not isinstance(prior, dict) or not verify_report_digest(prior):
        out["reason"] = "baseline report digest does not verify"
        return out
    out["baseline_report_sha256"] = prior["report_sha256"]
    here = code_identity()
    if (prior.get("checker") or {}).get("code_sha256") != here["code_sha256"]:
        out["reason"] = "baseline was produced by different checker code"
        return out
    if (prior.get("scope") or {}).get("name") != options.scope:
        out["reason"] = "baseline scope differs"
        return out
    before = ((prior.get("inputs") or {}).get("snapshot") or {}).get("partitions") or {}
    now = snapshot.identity().get("partitions") or {}
    changed = sorted(
        f"{t}/{p}"
        for t in sorted(set(before) | set(now))
        for p in sorted(set(before.get(t, {})) | set(now.get(t, {})))
        if before.get(t, {}).get(p) != now.get(t, {}).get(p)
    )
    prior_release = (prior.get("inputs") or {}).get("release") or {}
    out.update(
        compatible=True,
        changed_partitions=changed,
        unchanged_partitions=sum(
            1 for t, parts in now.items() for p, sha in parts.items() if before.get(t, {}).get(p) == sha
        ),
        release_changed=release is not None
        and prior_release.get("inventory_sha256") != release.identity()["inventory_sha256"],
    )
    return out


def write_outputs(
    result: AuditResult,
    *,
    report: Path,
    execution: Path | None = None,
    stream: Path | None = None,
    inputs: list[Path],
) -> dict[str, str]:
    """Write the report, execution metadata and optional stream atomically, outside every input."""
    from supercoach_via.storage.snapshots import atomic_write_bytes

    outputs = {"report": report, "execution": execution or default_execution_path(report)}
    if stream is not None:
        outputs["stream"] = stream
    refuse_inside_inputs(list(outputs.values()), inputs)
    if stream is not None:
        if result.stream_bytes is None:
            raise CheckerError("a findings stream was requested but not produced")
        atomic_write_bytes(stream, result.stream_bytes)
    atomic_write_bytes(report, canonical_bytes(result.report))
    atomic_write_bytes(outputs["execution"], canonical_bytes(result.execution))
    return {k: str(v) for k, v in outputs.items()}


def default_execution_path(report: Path) -> Path:
    return report.with_name(report.stem + ".execution.json")


def refuse_inside_inputs(outputs: list[Path], inputs: list[Path]) -> None:
    roots = [p.resolve() for p in inputs if p is not None]
    for out in outputs:
        target = out.resolve()
        for root in roots:
            if target == root or root in target.parents:
                raise UsageError(f"output {out} is inside the input {root}; write reports to a separate directory")


def _cache_salt(
    policy: IntegrityPolicy, rules: dict[str, RuleSpec], options: AuditOptions, as_of: datetime | None
) -> str:
    body = {
        "code": code_identity(),
        "policy": policy.sha256,
        "rules": _rules_sha256(rules),
        "scope": options.scope,
        "as_of": as_of.isoformat() if as_of else None,
    }
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def _build_report(
    options: AuditOptions,
    policy: IntegrityPolicy,
    rules: dict[str, RuleSpec],
    ctx: AuditContext,
    results: list[tuple[CheckSpec, Status, str]],
    as_of: datetime | None,
) -> dict[str, Any]:
    col = ctx.collector
    summary = col.rule_summary()
    checks: list[dict[str, Any]] = []
    for spec, status, reason in results:
        sev: dict[str, int] = {}
        for r in summary.values():
            if r["check_id"] == spec.check_id:
                for k, v in r["by_severity"].items():
                    sev[k] = sev.get(k, 0) + v
        checks.append(
            {
                "check_id": spec.check_id,
                "family": spec.family,
                "summary": spec.summary,
                "required": spec.required,
                "status": status.value,
                "reason": reason,
                "examined": dict(sorted(ctx.examined.get(spec.check_id, {}).items())),
                "findings": dict(sorted(sev.items())),
            }
        )
    outcome = overall((Status(c["status"]), c["required"]) for c in checks)
    by = {s: sum(1 for c in checks if c["status"] == s) for s in (x.value for x in Status)}
    families_requested = sorted({c["family"] for c in checks})
    restricted = options.families is not None or options.checks is not None
    complete = not restricted and not any(
        c["status"] in (Status.UNKNOWN.value,) and c["required"] for c in checks
    )
    examined_total: dict[str, int] = {}
    for c in checks:
        for k, v in c["examined"].items():
            examined_total[k] = examined_total.get(k, 0) + v
    body: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "checker": {**code_identity(), "rules_sha256": _rules_sha256(rules), "rules": len(rules)},
        "policy": {"version": policy.version, "sha256": policy.sha256, "files": policy.files},
        "inputs": {
            "snapshot": ctx.snapshot.identity() if ctx.snapshot else None,
            "release": ctx.release.identity() if ctx.release else None,
            "evidence": ctx.evidence.identity(),
            "models": ctx.coverage.get("model_inputs"),
        },
        "scope": {
            "name": options.scope,
            "as_of": as_of.isoformat().replace("+00:00", "Z") if as_of else None,
            "families": families_requested,
            "restricted": restricted,
            "complete": complete,
            "current_season": ctx.current_season,
        },
        "outcome": outcome.value,
        "counts": {
            "checks_requested": len(checks),
            "checks_performed": by[Status.PASS.value] + by[Status.FAIL.value],
            "checks": by,
            "findings_open": col.severity_counts(open_only=True),
            "findings_total": col.severity_counts(open_only=False),
            "rows_examined": examined_total.get("rows", 0),
            "resources_examined": examined_total.get("resources", 0),
            "cells_compared": examined_total.get("cells", 0),
            "examined": dict(sorted(examined_total.items())),
        },
        "checks": checks,
        "rules": list(summary.values()),
        "findings": [f.as_dict() for f in col.samples()],
        "exceptions": col.exception_summary(),
        "coverage": {k: v for k, v in sorted(ctx.coverage.items()) if k != "model_inputs" and not k.startswith("_")},
        "report_sha256": "",
    }
    body["inputs"]["digest"] = hashlib.sha256(canonical_bytes(body["inputs"])).hexdigest()
    return seal_report(body)
