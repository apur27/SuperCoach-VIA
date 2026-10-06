"""``scvia reconcile-afltables``: ``plan``, ``capture`` and ``compare`` (DESIGN section 11).

Wiring and exit-code translation only. Exit codes: 0 PASS/complete, 2 invalid invocation or
incompatible resume, 4 FAIL, 5 busy, 8 UNKNOWN/incomplete (resumable), 9 software failure.
``plan`` and ``compare`` never touch the network; ``capture`` needs ``--allow-network``.
"""

from __future__ import annotations

import json
import signal
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

from supercoach_via.ingest.http import HttpClient
from supercoach_via.reconciliation.capture import USER_AGENT, BusyError, Capture, CaptureError, Clock, SystemClock
from supercoach_via.reconciliation.inventory import PlanError, build_plan, load_plan, write_plan
from supercoach_via.reconciliation.urls import load_reconciliation_policies

SYSTEM_CLOCK: Clock = SystemClock()

reconcile_app = typer.Typer(
    name="reconcile-afltables",
    help="Compare every local player with a captured AFL Tables reference (opt-in audit; never edits data).",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def make_client(policies: Any) -> HttpClient:
    """The reconciliation transport: its own policy, a descriptive agent and no validators."""
    return HttpClient(policies, user_agent=USER_AGENT, archive=None)


def make_evidence_client(policies: Any) -> HttpClient:
    """The evidence transport: its own exact-path policy, a descriptive agent and no validators."""
    from supercoach_via.reconciliation import evidence as EV

    return HttpClient(policies, user_agent=EV.USER_AGENT, archive=None)


def _out(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(payload, sort_keys=True, default=str))
    else:
        for k, v in payload.items():
            typer.echo(f"{k}: {json.dumps(v) if isinstance(v, bool) else v}")


def _fail(code: int, message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code)


@reconcile_app.command("plan")
def plan_cmd(
    data_root: Annotated[Path, typer.Option("--data-root", help="canonical data root (current.json, snapshots)")],
    run_dir: Annotated[Path, typer.Option("--run-dir", help="new audit directory; never inside an input root")],
    through_date: Annotated[str, typer.Option("--through-date", help="inclusive event boundary YYYY-MM-DD")],
    snapshot: Annotated[str, typer.Option("--snapshot", help="'current' (pinned once) or sha256:<id>")] = "current",
    legacy_root: Annotated[
        Path | None, typer.Option("--legacy-root", help="repo root holding data/player_data")
    ] = None,
    scope: Annotated[
        str, typer.Option("--scope", help="all; seasons (an audit of named seasons); sample (development only)")
    ] = "all",
    sample_profile: Annotated[
        list[str] | None, typer.Option("--sample-profile", help="profile URL (sample scope)")
    ] = None,
    season: Annotated[
        list[int] | None, typer.Option("--season", help="season to audit (seasons scope; repeatable)")
    ] = None,
    capture_plan: Annotated[
        Path | None,
        typer.Option("--capture-plan", help="earlier plan.json whose existing capture this plan compares against"),
    ] = None,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Pin the inputs and write plan.json (offline)."""
    try:
        plan = build_plan(
            data_root=data_root,
            snapshot=snapshot,
            legacy_root=legacy_root,
            through_date=through_date,
            scope=scope,
            run_dir=run_dir,
            sample_profiles=sample_profile,
            seasons=season,
            capture_plan=capture_plan,
        )
        path = write_plan(plan)
    except PlanError as exc:
        raise _fail(2, str(exc)) from exc
    _out(
        {
            "ok": True,
            "plan": str(path),
            "plan_id": plan.plan_id,
            "capture_identity": plan.capture_identity,
            "snapshot_id": plan.inputs.snapshot.snapshot_id,
            "full_population": plan.scope.full_population,
            "seasons": plan.scope.seasons,
            "legacy_player_files": plan.inputs.legacy.player_files if plan.inputs.legacy else 0,
            "estimate": (
                "about 30,600-30,900 requests, at least 17 hours at 2 s spacing, "
                "about 1.8 GB raw HTML (design-review estimate, not a measured count)"
            ),
        },
        json_out,
    )


@reconcile_app.command("capture")
def capture_cmd(
    plan: Annotated[Path, typer.Option("--plan", help="plan.json written by `plan`")],
    allow_network: Annotated[
        bool, typer.Option("--allow-network", help="required: this command makes requests")
    ] = False,
    resume: Annotated[bool, typer.Option("--resume", help="continue an existing checkpoint (same plan)")] = False,
    clear_block: Annotated[bool, typer.Option("--clear-block", help="resume after reviewing an access block")] = False,
    max_requests: Annotated[
        int | None, typer.Option("--max-requests", help="stop after N fetches (pilot/testing)")
    ] = None,
    seed_from: Annotated[
        list[Path] | None,
        typer.Option("--seed-from", help="earlier capture/ directory whose verified objects may be reused"),
    ] = None,
    heartbeat_seconds: Annotated[float, typer.Option("--heartbeat-seconds")] = 60.0,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Acquire the AFL Tables evidence: one request at a time, at least two seconds apart."""
    if not allow_network:
        raise _fail(2, "capture makes network requests: pass --allow-network")
    try:
        doc = load_plan(plan)
    except PlanError as exc:
        raise _fail(2, str(exc)) from exc
    run_dir = Path(doc.operational.run_dir)
    if (run_dir / "capture" / "checkpoint.sqlite").exists() and not resume:
        raise _fail(2, "a checkpoint already exists for this run; pass --resume to continue it")
    client = make_client(load_reconciliation_policies())
    cap = Capture(
        doc,
        run_dir,
        client,
        clock=SYSTEM_CLOCK,
        log=lambda m: print(m, flush=True),
        heartbeat_s=heartbeat_seconds,
        max_requests=max_requests,
        clear_block=clear_block,
        seed_from=seed_from,
    )
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _s, _f: cap.request_stop())
    try:
        result = cap.run()
    except BusyError as exc:
        raise _fail(5, str(exc)) from exc
    except CaptureError as exc:
        raise _fail(2, str(exc)) from exc
    finally:
        client.close()
    _out(
        {
            "ok": result.exit_code == 0,
            "state": result.state,
            "reason": result.reason,
            "capture_complete": result.exit_code == 0,
            "counts": result.counts,
            "requests": result.requests,
            "manifest": str(run_dir / "capture" / "manifest.json"),
        },
        json_out,
    )
    if result.exit_code:
        raise typer.Exit(result.exit_code)


@reconcile_app.command("capture-evidence")
def capture_evidence_cmd(
    run_dir: Annotated[Path, typer.Option("--run-dir", help="run directory; the evidence goes to <run-dir>/evidence")],
    allow_network: Annotated[
        bool, typer.Option("--allow-network", help="required: this command makes requests")
    ] = False,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Acquire the Brownlow award evidence (robots.txt and one page) into its own archive, politely."""
    from supercoach_via.reconciliation import evidence as EV

    if not allow_network:
        raise _fail(2, "capture-evidence makes network requests: pass --allow-network")
    client = make_evidence_client(EV.evidence_policies())
    try:
        result = EV.EvidenceCapture(run_dir, client, clock=SYSTEM_CLOCK, log=lambda m: print(m, flush=True)).run()
    except BusyError as exc:
        raise _fail(5, str(exc)) from exc
    except EV.EvidenceError as exc:
        raise _fail(2, str(exc)) from exc
    finally:
        client.close()
    _out(
        {
            "ok": result.exit_code == 0,
            "state": result.state,
            "reason": result.reason,
            "sha256": result.sha256,
            "requests": result.requests,
            "evidence": str(run_dir / EV.EVIDENCE_DIR),
        },
        json_out,
    )
    if result.exit_code:
        raise typer.Exit(result.exit_code)


@reconcile_app.command("compare")
def compare_cmd(
    plan: Annotated[Path, typer.Option("--plan")],
    capture_manifest: Annotated[Path, typer.Option("--capture-manifest")],
    out: Annotated[Path, typer.Option("--out", help="new report directory")],
    workers: Annotated[int, typer.Option("--workers")] = 1,
    cache: Annotated[Path | None, typer.Option("--cache")] = None,
    previous: Annotated[Path | None, typer.Option("--previous", help="prior report.json (changed-since mode)")] = None,
    json_out: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Compare pinned local data with the frozen capture (offline; no network)."""
    import supercoach_via.reconciliation.compare as C  # imported lazily: heavy

    code = C.run_compare_cli(
        plan=plan,
        capture_manifest=capture_manifest,
        out=out,
        workers=workers,
        cache=cache,
        previous=previous,
        json_out=json_out,
    )
    if code:
        raise typer.Exit(code)


@reconcile_app.command("propose-corrections")
def propose_cmd(
    plan: Annotated[Path, typer.Option("--plan")],
    capture_manifest: Annotated[Path, typer.Option("--capture-manifest")],
    report: Annotated[Path, typer.Option("--report", help="completed report directory of this plan")],
    out: Annotated[Path, typer.Option("--out", help="new directory for changes.jsonl and summary.json")],
    workers: Annotated[int, typer.Option("--workers")] = 1,
    cache: Annotated[Path | None, typer.Option("--cache")] = None,
) -> None:
    """Propose source-backed corrections for both local layers from a completed audit (offline; edits nothing)."""
    import supercoach_via.reconciliation.compare as C
    from supercoach_via.reconciliation.propose import ProposeError, propose

    if (out / "changes.jsonl").exists():
        raise _fail(2, f"{out} already holds a proposal; choose a new directory")
    opts = C.CompareOptions(plan=plan, capture_manifest=capture_manifest, out=out / "_audit", workers=workers,
                            cache=cache)  # fmt: skip
    try:
        summary = propose(opts, report, out)
    except (ProposeError, C.CompareError, PlanError) as exc:
        raise _fail(2, str(exc)) from exc
    typer.echo(json.dumps({k: summary[k] for k in ("changes", "changes_sha256", "unsupported_fail_findings")}))


@reconcile_app.command("apply-corrections")
def apply_cmd(
    changes: Annotated[Path, typer.Option("--changes", help="changes.jsonl from propose-corrections")],
    layer: Annotated[str, typer.Option("--layer", help="snapshot | legacy_csv")],
    data_root: Annotated[
        Path | None, typer.Option("--data-root", help="snapshot layer: the data root to promote in")
    ] = None,
    expected_snapshot: Annotated[
        str | None, typer.Option("--expected-snapshot", help="the audited sha256:<id>")
    ] = None,
    legacy_root: Annotated[Path | None, typer.Option("--legacy-root", help="legacy layer: root holding data/")] = None,
) -> None:
    """Apply proposed corrections to ONE layer. Every change is checked against the value the audit saw first;
    any mismatch aborts before a byte is written. The snapshot layer promotes a validated child snapshot."""
    from supercoach_via.reconciliation import corrections as CO

    if layer == "snapshot":
        if data_root is None or expected_snapshot is None:
            raise _fail(2, "--layer snapshot needs --data-root and --expected-snapshot")
        from supercoach_via import pipeline
        from supercoach_via.settings import RunContext, load_settings

        ctx = RunContext(settings=load_settings(None, overrides={"data_root": data_root}))
        res = pipeline.apply_reconciliation_corrections(ctx, changes_file=changes, expected_snapshot=expected_snapshot)
        doc = {"exit": res.exit_code, "promoted": res.promoted, "snapshot_id": res.snapshot_id}
        doc.update({"error": res.error_code, "message": res.message, **res.outputs})
        typer.echo(json.dumps(doc, default=str))
        if res.exit_code:
            raise typer.Exit(res.exit_code)
        return
    if layer == "legacy_csv":
        if legacy_root is None:
            raise _fail(2, "--layer legacy_csv needs --legacy-root")
        try:
            summary = CO.apply_legacy(legacy_root, [c for c in CO.read_changes(changes) if c.layer == "legacy_csv"])
        except CO.CorrectionConflict as exc:
            raise _fail(2, f"refused, nothing written: {exc}") from exc
        typer.echo(json.dumps(summary))
        return
    raise _fail(2, f"--layer must be snapshot or legacy_csv, not {layer!r}")


def main() -> None:  # pragma: no cover
    try:
        reconcile_app()
    except Exception as exc:  # noqa: BLE001 - the audit itself failed: never a verdict
        typer.echo(f"software failure: {type(exc).__name__}: {exc}", err=True)
        sys.exit(9)
