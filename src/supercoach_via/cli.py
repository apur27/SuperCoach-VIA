"""`scvia` command line: wiring and exit-code translation only.

Heavy modules (pandas, duckdb, pyarrow, scikit-learn, matplotlib) are imported inside
command bodies so ``--help`` and ``doctor`` stay fast and side-effect free.

Exit codes (PLAN 5.4): 0 ok; 2 invalid input/config; 3 unavailable source / partial
required refresh; 4 validation failure; 5 locked; 6 model/forecast unavailable;
7 publication failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

EXIT = {
    "ok": 0,
    "invalid_input": 2,
    "source_unavailable": 3,
    "validation_failed": 4,
    "locked": 5,
    "model_unavailable": 6,
    "publish_failed": 7,
}

app = typer.Typer(
    name="scvia",
    help="SuperCoach VIA: AFL disposal forecasting, analytics and static release builder.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

ConfigOpt = Annotated[Path | None, typer.Option("--config", help="TOML settings file")]
DataRootOpt = Annotated[Path | None, typer.Option("--data-root", help="dataset/run root (default var/)")]
OutputRootOpt = Annotated[Path | None, typer.Option("--output-root", help="release root (default dist/)")]
JsonOpt = Annotated[bool, typer.Option("--json", help="print one JSON result on stdout")]


class CliFailure(Exception):
    def __init__(self, code: str, message: str, recovery: str | None = None):
        super().__init__(message)
        self.code = code
        self.recovery = recovery


def _settings(config: Path | None, **overrides: Any) -> Any:
    from supercoach_via.settings import SettingsError, load_settings

    try:
        return load_settings(config, overrides=overrides)
    except SettingsError as exc:
        raise CliFailure("invalid_input", str(exc)) from exc


def _emit(result: dict[str, Any], as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(result, sort_keys=True, default=str))
    else:
        for key, value in result.items():
            if key != "details":
                typer.echo(f"{key}: {value}", err=True)


def _fail(exc: CliFailure, as_json: bool) -> NoReturn:
    payload = {"ok": False, "error_code": exc.code, "message": str(exc)[:2000], "recovery": exc.recovery}
    if as_json:
        typer.echo(json.dumps(payload, sort_keys=True))
    typer.echo(f"error [{exc.code}]: {exc}", err=True)
    if exc.recovery:
        typer.echo(f"recovery: {exc.recovery}", err=True)
    raise typer.Exit(EXIT[exc.code])


def _run(as_json: bool, fn: Any, *args: Any, **kwargs: Any) -> None:
    from supercoach_via.storage.runs import LockedError

    try:
        result = fn(*args, **kwargs)
    except CliFailure as exc:
        _fail(exc, as_json)
    except LockedError as exc:
        _fail(CliFailure("locked", str(exc), "wait for the other writer or inspect `scvia status`"), as_json)
    _emit(result, as_json)
    code = result.get("exit", "ok")
    if code != "ok":
        raise typer.Exit(EXIT[code])


# ---------------------------------------------------------------------------
# doctor / schemas / status
# ---------------------------------------------------------------------------


def doctor_checks(settings: Any) -> list[dict[str, Any]]:
    import importlib.util
    import os
    import platform
    import tomllib

    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    add("python", "pass" if sys.version_info[:2] == (3, 12) else "warn", platform.python_version())
    for label, root in (("writable_data_root", settings.data_root), ("writable_output_root", settings.output_root)):
        probe = root if root.exists() else next((p for p in root.parents if p.exists()), Path())
        add(label, "pass" if os.access(probe, os.W_OK) else "fail", str(root))
    add("public_base", "pass", settings.public_base)
    lock = Path("uv.lock")
    add(
        "lock_file",
        "pass" if lock.is_file() else "warn",
        "uv.lock present" if lock.is_file() else "uv.lock not found in cwd",
    )
    policy = settings.source_policy_path
    if policy.is_file():
        try:
            tomllib.loads(policy.read_text(encoding="utf-8"))
            add("source_policy", "pass", str(policy))
        except tomllib.TOMLDecodeError as exc:
            add("source_policy", "fail", f"{policy}: {exc}")
    else:
        add("source_policy", "warn", f"{policy} not found (needed only for live refresh)")
    ml = importlib.util.find_spec("lightgbm") is not None
    add(
        "ml_extra",
        "pass" if ml else "warn",
        "lightgbm available" if ml else "optional `ml` extra not installed; sklearn candidates only",
    )
    current = settings.data_root / "current.json"
    add(
        "current_snapshot",
        "pass" if current.is_file() else "warn",
        str(current) if current.is_file() else "no accepted snapshot yet (run import-legacy)",
    )
    if settings.editorial_enabled:
        add("editorial_adapter", "warn", "editorial enabled: adapter credentials are checked when a draft is requested")
    return checks


@app.command()
def doctor(
    config: ConfigOpt = None,
    data_root: DataRootOpt = None,
    output_root: OutputRootOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Check runtime, roots, lock, source policy and optional extras (no network)."""

    def body() -> dict[str, Any]:
        settings = _settings(config, data_root=data_root, output_root=output_root)
        checks = doctor_checks(settings)
        ok = not any(c["status"] == "fail" for c in checks)
        if not json_out:
            for c in checks:
                typer.echo(f"[{c['status']:>4}] {c['name']}: {c['detail']}", err=True)
        return {"ok": ok, "checks": checks, "exit": "ok" if ok else "invalid_input"}

    _run(json_out, body)


@app.command()
def schemas(out: Annotated[Path, typer.Option("--out")] = Path("schemas")) -> None:
    """Regenerate public JSON Schemas from the view models."""
    from supercoach_via.publish.web_data import export_json_schemas

    written = export_json_schemas(out)
    typer.echo(f"wrote {len(written)} schemas to {out}", err=True)


# ---------------------------------------------------------------------------
# release validation / publication
# ---------------------------------------------------------------------------


def _release_dir(settings: Any, release: str) -> Path:
    from supercoach_via.domain.schemas import is_safe_id

    if not is_safe_id(release) or ":" in release:
        raise CliFailure("invalid_input", f"invalid release id {release!r}")
    path: Path = settings.output_root / "releases" / release
    if not path.is_dir():
        raise CliFailure("invalid_input", f"release {release} not found under {settings.output_root}/releases")
    return path


@app.command("validate-release")
def validate_release_cmd(
    release: Annotated[str, typer.Option("--release")],
    config: ConfigOpt = None,
    output_root: OutputRootOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Validate a built release (closure, hashes, schemas, allowlist, references)."""

    def body() -> dict[str, Any]:
        from supercoach_via.publish.release import validate_release

        settings = _settings(config, output_root=output_root)
        report = validate_release(_release_dir(settings, release))
        return {
            "ok": report.ok,
            "release_id": release,
            "outcome": report.outcome.value,
            "issues": report.issues[:50],
            "exit": "ok" if report.ok else "validation_failed",
        }

    _run(json_out, body)


@app.command()
def publish(
    release: Annotated[str, typer.Option("--release")],
    destination: Annotated[Path, typer.Option("--destination", help="local static-host directory")],
    config: ConfigOpt = None,
    output_root: OutputRootOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """The only public mutation: publish a VALIDATED release to a configured destination."""

    def body() -> dict[str, Any]:
        from supercoach_via.publish.release import LocalDirectoryDestination, PublishError, publish_release
        from supercoach_via.settings import utc_now

        settings = _settings(config, output_root=output_root)
        dest = LocalDirectoryDestination("local", destination)
        try:
            receipt = publish_release(_release_dir(settings, release), dest, clock=utc_now)
        except PublishError as exc:
            raise CliFailure("publish_failed", str(exc), "fix the release and re-run validate-release") from exc
        return {"ok": True, "receipt": receipt.model_dump(mode="json")}

    _run(json_out, body)


@app.command()
def rollback(
    release: Annotated[str, typer.Option("--release", help="previously validated release to reactivate")],
    destination: Annotated[Path, typer.Option("--destination")],
    config: ConfigOpt = None,
    output_root: OutputRootOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Reactivate a previous validated release (new receipt; data stays immutable)."""

    def body() -> dict[str, Any]:
        from supercoach_via.publish.release import LocalDirectoryDestination, PublishError
        from supercoach_via.publish.release import rollback as do_rollback
        from supercoach_via.settings import utc_now

        settings = _settings(config, output_root=output_root)
        try:
            receipt = do_rollback(
                settings.output_root, LocalDirectoryDestination("local", destination), release, clock=utc_now
            )
        except PublishError as exc:
            raise CliFailure("publish_failed", str(exc)) from exc
        return {"ok": True, "receipt": receipt.model_dump(mode="json")}

    _run(json_out, body)


@app.command()
def preview(
    release: Annotated[str, typer.Option("--release")],
    config: ConfigOpt = None,
    output_root: OutputRootOpt = None,
    port: Annotated[int, typer.Option("--port", min=1024, max=65535)] = 4321,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
) -> None:
    """Serve a built release's static site read-only (localhost by default)."""
    import functools
    import http.server

    settings = _settings(config, output_root=output_root)
    rdir = _release_dir(settings, release)
    root = rdir / "site" if (rdir / "site").is_dir() else rdir / "public"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    with http.server.ThreadingHTTPServer((host, port), handler) as srv:
        typer.echo(f"serving {root} at http://{host}:{port}/ (Ctrl-C to stop)", err=True)
        srv.serve_forever()


# ---------------------------------------------------------------------------
# dataset pipeline commands (thin wrappers over supercoach_via.pipeline)
# ---------------------------------------------------------------------------

_STAGE_EXIT = {0: "ok", 2: "invalid_input", 3: "source_unavailable", 4: "validation_failed", 5: "locked",
               6: "model_unavailable", 7: "publish_failed"}  # fmt: skip


def _stage_payload(res: Any) -> dict[str, Any]:
    d: dict[str, Any] = res.as_dict()
    d["exit"] = _STAGE_EXIT.get(res.exit_code, "invalid_input")
    return d


def _parse_repair(spec: str) -> tuple[Path, int]:
    path, sep, season = spec.rpartition(":")
    if not sep or not season.isdigit():
        raise CliFailure("invalid_input", f"--repair expects EVIDENCE_DIR:SEASON, got {spec!r}")
    p = Path(path)
    if not (p / "fetch-manifest.json").is_file():
        raise CliFailure("invalid_input", f"{p} has no fetch-manifest.json")
    return p, int(season)


@app.command("import-legacy")
def import_legacy_cmd(
    source: Annotated[Path, typer.Option("--source", help="legacy repository root (read-only)")] = Path(),
    repair: Annotated[
        list[str] | None, typer.Option("--repair", help="archived repair evidence EVIDENCE_DIR:SEASON (repeatable)")
    ] = None,
    no_promote: Annotated[bool, typer.Option("--no-promote", help="validate only; leave current.json")] = False,
    config: ConfigOpt = None,
    data_root: DataRootOpt = None,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    json_out: JsonOpt = False,
) -> None:
    """Import the legacy CSV corpus (+ archived repairs), validate, and promote on PASS."""

    def body() -> dict[str, Any]:
        from supercoach_via import pipeline
        from supercoach_via.settings import RunContext

        settings = _settings(config, data_root=data_root, source_root=source)
        repairs = [_parse_repair(r) for r in repair or []]
        res = pipeline.ingest(RunContext(settings=settings), source_root=source, repairs=repairs,
                              run_id=run_id, promote=not no_promote)  # fmt: skip
        return _stage_payload(res)

    _run(json_out, body)


@app.command()
def validate(
    snapshot: Annotated[str, typer.Option("--snapshot", help="'current' or sha256:<id>")] = "current",
    config: ConfigOpt = None,
    data_root: DataRootOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Re-validate an existing snapshot (read-only)."""

    def body() -> dict[str, Any]:
        from supercoach_via import pipeline
        from supercoach_via.ingest.legacy import DatasetCandidate
        from supercoach_via.storage.snapshots import SnapshotCandidate, load_snapshot, snapshot_hex

        settings = _settings(config, data_root=data_root)
        try:
            manifest = load_snapshot(settings.data_root, snapshot, verify=True)
        except (FileNotFoundError, ValueError) as exc:
            raise CliFailure("invalid_input", str(exc), "run import-legacy first") from exc
        path = settings.data_root / "snapshots" / f"{snapshot_hex(manifest.snapshot_id)}.json"
        report = pipeline._validate(DatasetCandidate(SnapshotCandidate(manifest, path), {}, settings.data_root))
        return {"ok": report.ok, "snapshot_id": manifest.snapshot_id, "outcome": report.outcome.value,
                "checks": {k: v.value for k, v in report.checks.items()},
                "blocking": [i for i in report.issues if i.get("severity") == "blocking"][:50],
                "exit": "ok" if report.ok else "validation_failed"}  # fmt: skip

    _run(json_out, body)


@app.command()
def refresh(
    season: Annotated[int | None, typer.Option("--season")] = None,
    plan: Annotated[bool, typer.Option("--plan", help="print the offline plan; no network, no writes")] = False,
    data_only: Annotated[bool, typer.Option("--data-only", help="refresh data only (no release build)")] = False,
    repair_season: Annotated[int | None, typer.Option("--repair-season")] = None,
    allow_network: Annotated[bool, typer.Option("--allow-network", help="explicit opt-in to contact sources")] = False,
    config: ConfigOpt = None,
    data_root: DataRootOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Plan (offline) or run a bounded source refresh; the result is a candidate, never auto-published."""

    def body() -> dict[str, Any]:
        from supercoach_via.ingest import refresh as rf
        from supercoach_via.settings import RunContext

        settings = _settings(config, data_root=data_root)
        ctx = RunContext(settings=settings)
        if not plan and not allow_network:
            raise CliFailure("invalid_input", "a real refresh contacts external sources",
                             "re-run with --plan to preview, or add --allow-network to fetch")  # fmt: skip
        try:
            base = rf.base_state_from_snapshot(settings.data_root)
        except FileNotFoundError as exc:
            raise CliFailure("invalid_input", str(exc), "run import-legacy first") from exc
        try:
            req = rf.RefreshRequest(current_season=season, repair_season=repair_season)
            the_plan = rf.plan_refresh(base, req, ctx)
        except rf.RefreshConfigError as exc:
            raise CliFailure("invalid_input", str(exc)) from exc
        if plan:
            if not json_out:
                typer.echo(the_plan.describe(), err=True)
            return {**the_plan.to_dict(), "ok": True}
        from supercoach_via import pipeline
        from supercoach_via.ingest.http import HttpClient, RawArchive, load_policies

        policies = load_policies(settings.source_root / "config" / "source_policies.toml")
        with HttpClient(policies, user_agent=f"SuperCoach-VIA ({pipeline.CODE_VERSION}; operator refresh)",
                        archive=RawArchive(settings.data_root / "raw")) as http:  # fmt: skip
            ctx.http = http
            res = pipeline.refresh(ctx, season=season, repair_season=repair_season)
        return _stage_payload(res)

    _run(json_out, body)


@app.command()
def status(config: ConfigOpt = None, data_root: DataRootOpt = None, json_out: JsonOpt = False) -> None:
    """Accepted snapshot and the most recent run (read-only)."""

    def body() -> dict[str, Any]:
        from supercoach_via.storage.snapshots import read_current

        settings = _settings(config, data_root=data_root)
        cur = read_current(settings.data_root)
        runs_dir = settings.data_root / "runs"
        last = None
        if runs_dir.is_dir():
            ids = sorted(p.name for p in runs_dir.iterdir() if (p / "run.json").is_file())
            if ids:
                m = json.loads((runs_dir / ids[-1] / "run.json").read_text(encoding="utf-8"))
                last = {k: m.get(k) for k in ("run_id", "command", "state", "error_code", "recovery", "updated_at")}
        return {"ok": True, "current_snapshot": cur.snapshot_id if cur else None,
                "promoted_at": cur.promoted_at.isoformat() if cur else None, "last_run": last}  # fmt: skip

    _run(json_out, body)


@app.command()
def forecast(
    train_cutoff: Annotated[str, typer.Option("--train-cutoff", help="YYYY-MM-DD")],
    calibration_end: Annotated[str, typer.Option("--calibration-end", help="YYYY-MM-DD")],
    cutoff: Annotated[str, typer.Option("--cutoff", help="forecast cutoff instant, ISO 8601 with zone")],
    replay_season: Annotated[int | None, typer.Option("--replay-season")] = None,
    snapshot: Annotated[str, typer.Option("--snapshot")] = "current",
    config: ConfigOpt = None,
    data_root: DataRootOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Train (or reuse) the bundle, forecast the next real fixtures, optionally replay a season."""

    def body() -> dict[str, Any]:
        from datetime import date, datetime

        from supercoach_via import pipeline
        from supercoach_via.settings import RunContext

        try:
            at = datetime.fromisoformat(cutoff)
            plan = pipeline.ForecastPlan(date.fromisoformat(train_cutoff), date.fromisoformat(calibration_end), at,
                                         replay_season=replay_season)  # fmt: skip
        except ValueError as exc:
            raise CliFailure("invalid_input", str(exc)) from exc
        if at.tzinfo is None:
            raise CliFailure("invalid_input", "--cutoff needs an explicit UTC offset")
        settings = _settings(config, data_root=data_root)
        res = pipeline.forecast(RunContext(settings=settings), plan, snapshot=snapshot)
        payload = _stage_payload(res)
        if res.exit_code == 0 and res.outputs.get("forecast_status") != "available":
            payload["note"] = "no valid future fixture: forecast_status=unavailable (the release still builds)"
        return payload

    _run(json_out, body)


@app.command("build-release")
def build_release_cmd(
    snapshot: Annotated[str, typer.Option("--snapshot")] = "current",
    bundle: Annotated[list[str] | None, typer.Option("--bundle", help="model bundle ID (repeatable)")] = None,
    predictions: Annotated[list[Path] | None, typer.Option("--predictions", help="prediction run dir")] = None,
    evaluation: Annotated[list[Path] | None, typer.Option("--evaluation", help="evaluation dir")] = None,
    content_manifest: Annotated[
        Path | None, typer.Option("--content-manifest", help="public article manifest, e.g. config/public_content.toml")
    ] = None,
    content_root: Annotated[Path, typer.Option("--content-root", help="article path root")] = Path(),
    live_root: Annotated[Path | None, typer.Option("--live-root", help="live monitor state root")] = None,
    editorial: Annotated[str, typer.Option("--editorial", help="only 'off' is supported for numeric releases")] = "off",
    demo_label: Annotated[bool, typer.Option("--demo", help="label every output DEMO")] = False,
    config: ConfigOpt = None,
    data_root: DataRootOpt = None,
    output_root: OutputRootOpt = None,
    json_out: JsonOpt = False,
) -> None:
    """Build and validate a release from an accepted snapshot plus explicitly named forecast inputs."""

    def body() -> dict[str, Any]:
        from supercoach_via import pipeline
        from supercoach_via.domain.schemas import is_safe_id
        from supercoach_via.settings import RunContext

        if editorial != "off":
            raise CliFailure("invalid_input", "--editorial must be 'off'; editorial drafts never gate numeric releases")
        bundles = bundle or []
        bad = [b for b in bundles if not is_safe_id(b)]
        if bad:
            raise CliFailure("invalid_input", f"unsafe bundle id(s): {bad}")
        settings = _settings(config, data_root=data_root, output_root=output_root)
        ctx = RunContext(settings=settings)
        try:
            from supercoach_via.ml import bundles as B
            from supercoach_via.ml import evaluate as E
            from supercoach_via.publish.builder import ReleaseInputs

            inputs = ReleaseInputs(
                prediction_dirs=tuple(predictions or ()),
                model_manifests=tuple(B.read_manifest(settings.data_root / "models", b) for b in bundles),
                evaluations=tuple(E.read_evaluation(d) for d in evaluation or ()),
                content_root=content_root if content_manifest else None,
                content_manifest=content_manifest,
                live_root=live_root,
                demo=demo_label,
            )
        except (OSError, ValueError, KeyError) as exc:
            raise CliFailure("invalid_input", f"cannot load release inputs: {exc}") from exc
        return _stage_payload(pipeline.build(ctx, inputs, snapshot=snapshot))

    _run(json_out, body)


@app.command()
def demo(
    output: Annotated[Path, typer.Option("--output", help="empty directory for the DEMO build")] = Path("dist/demo"),
    json_out: JsonOpt = False,
) -> None:
    """Build the complete DEMO dataset + release offline (no network, GPU or credentials)."""

    def body() -> dict[str, Any]:
        from supercoach_via import pipeline

        if output.exists() and any(output.iterdir()):
            raise CliFailure("invalid_input", f"{output} is not empty", "choose an empty --output directory")
        res = pipeline.demo(output)
        return {**res, "exit": _STAGE_EXIT.get(int(res["exit_code"]), "invalid_input")}

    _run(json_out, body)


def main() -> None:
    try:
        app()
    except CliFailure as exc:  # pragma: no cover - defensive
        _fail(exc, False)


if __name__ == "__main__":  # pragma: no cover
    main()
