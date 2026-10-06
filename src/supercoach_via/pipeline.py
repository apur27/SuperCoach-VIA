"""Explicit staged pipeline (PLAN 5.3): ingest -> forecast/evaluate -> release.

Each public entry point takes the whole-run writer lock, creates a run directory with a
state machine (``storage.runs``), records every step's inputs/outputs/outcome, and returns a
result object with an exit code (PLAN 5.4) instead of raising for expected failures. The
accepted pointer (``current.json``) moves only after a passing validation, and a failed or
crashed step leaves it untouched. No step spawns subprocesses or touches Git; publication
is a separate command (``publish.release.publish_release``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from supercoach_via import __version__
from supercoach_via.domain.schemas import CheckOutcome, RunState, ValidationReport
from supercoach_via.settings import RunContext
from supercoach_via.storage import runs
from supercoach_via.storage.snapshots import atomic_write_bytes

CODE_VERSION = f"supercoach_via {__version__}"

EXIT_OK, EXIT_INVALID, EXIT_SOURCE, EXIT_VALIDATION, EXIT_LOCKED, EXIT_MODEL, EXIT_PUBLISH = 0, 2, 3, 4, 5, 6, 7


@dataclass
class StageResult:
    exit_code: int
    state: RunState | None
    run_id: str | None
    snapshot_id: str | None = None
    promoted: bool = False
    error_code: str | None = None
    message: str | None = None
    recovery: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    #: in-process objects handed to the next stage (not serialised)
    artifacts: dict[str, Any] = field(default_factory=dict, repr=False)

    def as_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "artifacts"}
        d["state"] = self.state.value if self.state else None
        d["ok"] = self.exit_code == 0
        return json.loads(json.dumps(d, default=str))  # type: ignore[no-any-return]


def _write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, json.dumps(payload, indent=1, sort_keys=True, default=str).encode())


def _report_json(report: ValidationReport) -> dict[str, Any]:
    return json.loads(json.dumps(dataclasses.asdict(report), default=str))  # type: ignore[no-any-return]


class _Failure(Exception):
    def __init__(self, exit_code: int, error_code: str, message: str, recovery: str | None = None) -> None:
        super().__init__(message)
        self.exit_code, self.error_code, self.recovery = exit_code, error_code, recovery


def _locked(
    ctx: RunContext, command: str, run_id: str | None, body: Callable[[runs.RunStore, StageResult], None]
) -> StageResult:
    data_root = ctx.data_root
    try:
        with runs.WriterLock(data_root):
            store = runs.RunStore.create(data_root, command, clock=ctx.clock, code_version=CODE_VERSION, run_id=run_id)
            result = StageResult(exit_code=EXIT_OK, state=RunState.CREATED, run_id=store.run_id)
            try:
                body(store, result)
            except _Failure as exc:
                store.transition(RunState.FAILED, error_code=exc.error_code, recovery=exc.recovery)
                result.exit_code, result.error_code = exc.exit_code, exc.error_code
                result.message, result.recovery = str(exc), exc.recovery
            except Exception as exc:  # noqa: BLE001 - any crash is a failed run, never a success
                store.event("crash", error=f"{type(exc).__name__}: {exc}")
                store.transition(RunState.FAILED, error_code="internal_error", recovery=f"inspect {store.directory}")
                result.exit_code, result.error_code = EXIT_INVALID, "internal_error"
                result.message = f"{type(exc).__name__}: {exc}"
            result.state = store.manifest.state
            store.event("result", **result.as_dict())
            return result
    except runs.LockedError as exc:
        return StageResult(exit_code=EXIT_LOCKED, state=None, run_id=None, error_code="locked", message=str(exc),
                           recovery="wait for the other writer to finish")  # fmt: skip


def _timed(result: StageResult, name: str, fn: Callable[[], Any]) -> Any:
    t0 = time.perf_counter()
    try:
        return fn()
    finally:
        result.timings[name] = round(time.perf_counter() - t0, 3)


# ---------------------------------------------------------------------------
# Ingest: import (+ archived repairs) -> validate -> promote
# ---------------------------------------------------------------------------


def _import(source_root: Path, ctx: RunContext) -> Any:
    from supercoach_via.ingest.legacy import import_legacy

    return import_legacy(source_root, ctx)


def _validate(candidate: Any, *, current_season: int | None = None) -> ValidationReport:
    from supercoach_via.ingest.reconcile import load_policy, validate_dataset

    return validate_dataset(candidate, load_policy(current_season=current_season))


def _with_season_aggregates(
    data_root: Path,
    merged: Any,
    match_upserts: list[dict[str, Any]],
    *,
    clock: Any,
    status: Any,
    checked_at: datetime | None,
    checked_seasons: set[int],
    run_id: str | None = None,
) -> Any:
    """Recompute season date/count rows for changed matches and checked fixtures.

    Uncertainty fields stay as recorded. A failed refresh never reaches this function.
    """
    seasons = sorted({int(row["season"]) for row in match_upserts} | set(checked_seasons))
    if not seasons:
        return merged
    from supercoach_via.ingest import refresh as rf
    from supercoach_via.storage import snapshots
    from supercoach_via.storage.queries import SnapshotQuery
    manifest = merged.manifest
    with SnapshotQuery(
        data_root, manifest, tables={"matches", "seasons"}, partitions={"matches": {str(s) for s in seasons}}
    ) as q:
        marks = ",".join("?" * len(seasons))
        match_sql = f"SELECT season, match_date, status FROM matches WHERE season IN ({marks})"  # noqa: S608
        season_sql = f"SELECT * FROM seasons WHERE season IN ({marks})"  # noqa: S608
        matches = q.arrow(match_sql, seasons).to_pylist()
        existing_rows = q.arrow(season_sql, seasons).to_pylist() if "seasons" in manifest.tables else []
    rows = rf.season_aggregate_rows(
        matches,
        {int(row["season"]): row for row in existing_rows},
        seasons=seasons,
        checked_at=checked_at,
        checked_seasons=checked_seasons,
    )
    if not rows:
        return merged
    return snapshots.apply_upserts(
        data_root, manifest, {"seasons": rows}, clock=clock, code_version=CODE_VERSION, status=status, run_id=run_id,
        notes=[f"season aggregates {', '.join(map(str, seasons))}"],
    )  # fmt: skip


def _with_resolved_blanks(
    data_root: Path, merged: Any, seasons: set[int], *, clock: Any, status: Any, run_id: str | None = None
) -> Any:
    """Apply ``domain.blanks`` to the merged player rows of ``seasons`` (source and legacy rows alike).

    Only rows whose cells change are upserted, so a no-op leaves the candidate unchanged.
    """
    if not seasons:
        return merged
    from supercoach_via.domain import blanks
    from supercoach_via.domain.metrics import CoverageEras
    from supercoach_via.domain.schemas import TABLES
    from supercoach_via.settings import default_config_dir
    from supercoach_via.storage import snapshots
    from supercoach_via.storage.queries import SnapshotQuery

    parts = {str(s) for s in seasons}
    manifest = merged.manifest
    with SnapshotQuery(
        data_root, manifest, tables={"matches", "player_games"}, partitions={"matches": parts, "player_games": parts}
    ) as q:
        marks = ",".join("?" * len(seasons))
        games = q.arrow(f"SELECT * FROM player_games WHERE season IN ({marks})", sorted(seasons))  # noqa: S608
        stages = dict(q.rows(f"SELECT match_id, stage_type FROM matches WHERE season IN ({marks})", sorted(seasons)))  # noqa: S608
    recorded = CoverageEras.load(default_config_dir() / "coverage.yaml").recorded
    resolved, counts = blanks.resolve_blanks(games, stages, recorded)
    if not sum(counts.values()):
        return merged
    before, after = games.to_pylist(), resolved.to_pylist()
    changed = [b for a, b in zip(before, after, strict=True) if a != b]
    names = TABLES["player_games"].column_names
    return snapshots.apply_upserts(
        data_root, manifest, {"player_games": [{c: r.get(c) for c in names} for r in changed]}, clock=clock,
        code_version=CODE_VERSION, status=status, run_id=run_id,
        notes=[f"{blanks.RULE_VERSION}: {sum(counts.values())} source blanks resolved to 0 in {sorted(seasons)}"],
    )  # fmt: skip


def apply_repair_evidence(ctx: RunContext, candidate: Any, evidence_dir: Path, *, season: int) -> Any:
    """Merge one archived, re-verified repair into ``candidate`` and re-link lineup tokens."""
    from supercoach_via.domain.ids import ClubRegistry
    from supercoach_via.ingest import reconcile
    from supercoach_via.ingest import refresh as rf
    from supercoach_via.ingest.legacy import DatasetCandidate
    from supercoach_via.storage import snapshots
    from supercoach_via.storage.queries import SnapshotQuery

    root = candidate.data_root
    manifest = candidate.candidate.manifest
    evidence = json.loads((evidence_dir / "fetch-manifest.json").read_text(encoding="utf-8"))
    pids = [t["player_id"] for t in evidence["targets"] if t.get("player_id")]
    with SnapshotQuery(root, manifest, tables={"players"}) as q:
        existing = {
            r["player_id"]: r
            for r in q.arrow("SELECT * FROM players WHERE list_contains(?, player_id)", [pids]).to_pylist()
        }
    base = rf.base_state_from_snapshot(root, manifest.snapshot_id)
    aliases = ctx.settings.source_root / "config" / "team_aliases.csv"
    resolver = ClubRegistry.from_csv(aliases).resolve if aliases.is_file() else None
    ups = rf.replay_repair_evidence(evidence_dir, base, season=season, context=ctx, existing_players=existing,
                                    club_resolver=resolver)  # fmt: skip
    note = f"repair evidence {evidence_dir.name}: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(ups.items()))
    merged = snapshots.apply_upserts(root, manifest, ups, clock=ctx.clock, code_version=CODE_VERSION,
                                     status=manifest.status, run_id=manifest.run_id, notes=[note])  # fmt: skip
    merged = _with_resolved_blanks(
        root, merged, {int(r["season"]) for r in ups.get("player_games") or []}, clock=ctx.clock,
        status=manifest.status, run_id=manifest.run_id,
    )  # fmt: skip
    merged = _with_season_aggregates(
        root, merged, ups.get("matches") or [], clock=ctx.clock, status=manifest.status, checked_at=None,
        checked_seasons=set(), run_id=manifest.run_id,
    )  # fmt: skip
    relink = reconcile.relink_quarantined_lineups(root, merged.manifest, season=season)
    if any(relink.values()):
        merged = snapshots.apply_upserts(
            root, merged.manifest, relink, clock=ctx.clock, code_version=CODE_VERSION, status=manifest.status,
            run_id=manifest.run_id,
            notes=[f"re-linked {len(relink['lineups'])} quarantined {season} lineup tokens"],
        )  # fmt: skip
    report = {**candidate.report, "repairs": [*candidate.report.get("repairs", []),
              {"evidence": str(evidence_dir), "upserts": {k: len(v) for k, v in ups.items()},
               "relinked_lineups": len(relink["lineups"]), "snapshot_id": merged.manifest.snapshot_id}]}  # fmt: skip
    return DatasetCandidate(candidate=merged, report=report, data_root=root)


def _with_fixture_corrections(
    data_root: Path, merged: Any, seasons: set[int], *, clock: Any, status: Any, run_id: str | None = None
) -> tuple[Any, dict[str, int]]:
    """Correct fixture fields from each season's pinned page (``ingest.corrections``)."""
    from supercoach_via.ingest import corrections
    from supercoach_via.settings import default_config_dir
    from supercoach_via.storage import snapshots

    club_resolver, venue_resolver = corrections.default_resolvers(default_config_dir())
    ups: dict[str, list[dict[str, Any]]] = {"matches": [], "quality_issues": []}
    for season in sorted(seasons):
        found = corrections.fixture_corrections(data_root, merged.manifest, season, club_resolver=club_resolver,
                                                venue_resolver=venue_resolver)  # fmt: skip
        for k, v in found.items():
            ups[k].extend(v)
    counts = {k: len(v) for k, v in ups.items()}
    if not any(ups.values()):
        return merged, counts
    keys = sorted({r["row_key"] for r in ups["quality_issues"]})
    merged = snapshots.apply_upserts(
        data_root, merged.manifest, {k: v for k, v in ups.items() if v}, clock=clock, code_version=CODE_VERSION,
        status=status, run_id=run_id,
        notes=[f"fixture corrections from pinned captures: {counts['matches']} matches ({', '.join(keys[:5])})"],
    )  # fmt: skip
    return merged, counts


def apply_corrections(
    ctx: RunContext, *, seasons: Sequence[int], run_id: str | None = None,
    evidence_root: Path | None = None, evidence_snapshot: str | None = None,
) -> StageResult:
    """Offline, bounded corrections of the accepted snapshot; promotes a child snapshot on PASS.

    1. ``ingest.corrections.fixture_corrections`` for each season with a pinned season page;
    2. ``domain.blanks`` over every season (the importer's rule, for snapshots imported before it).

    Nothing is fetched. A run that changes nothing promotes nothing.
    """

    def body(store: runs.RunStore, result: StageResult) -> None:
        from supercoach_via.ingest import corrections, reconcile
        from supercoach_via.ingest.legacy import DatasetCandidate
        from supercoach_via.storage import snapshots
        from supercoach_via.storage.queries import SnapshotQuery

        store.transition(RunState.PLANNED)
        if (evidence_root is None) != (evidence_snapshot is None):
            raise _Failure(
                EXIT_INVALID, "invalid_input", "evidence root and immutable snapshot must be supplied together"
            )
        root = ctx.data_root
        base = snapshots.load_snapshot(root, verify=True)
        store.transition(RunState.PARSED)
        manifest_path = root / "snapshots" / f"{snapshots.snapshot_hex(base.snapshot_id)}.json"
        merged = snapshots.SnapshotCandidate(base, manifest_path)
        try:
            if evidence_root is not None and evidence_snapshot is not None:
                from supercoach_via.ingest.http import RawArchive

                revisions, observations, payloads = corrections.pinned_season_evidence(
                    evidence_root, evidence_snapshot, set(seasons)
                )
                existing: dict[str, dict[str, Any]] = {}
                if "source_observations" in base.tables:
                    with SnapshotQuery(root, base, tables={"source_observations"}) as q:
                        existing = {
                            r["source_ref"]: r for r in q.arrow("SELECT * FROM source_observations").to_pylist()
                        }
                changed = [r for r in observations if existing.get(r["source_ref"]) != r]
                for sha, payload in payloads.items():
                    snapshots.contained_path(root, f"raw/objects/{sha[:2]}/{sha}")
                    RawArchive(root / "raw").put(payload)
                if changed or any(base.source_revisions.get(k) != v for k, v in revisions.items()):
                    merged = snapshots.apply_upserts(
                        root, base, {"source_observations": changed or observations},
                        source_revisions=revisions, clock=ctx.clock, code_version=CODE_VERSION,
                        status=base.status, run_id=store.run_id,
                        notes=[f"correction season evidence from {evidence_snapshot}"],
                    )
                result.outputs["evidence"] = {
                    "snapshot_id": evidence_snapshot,
                    "season_captures": {str(s): revisions[f"afltables:season:{s}"] for s in sorted(set(seasons))},
                }
            for season in sorted(set(seasons)):
                if not merged.manifest.source_revisions.get(f"afltables:season:{season}"):
                    raise corrections.CorrectionError(
                        f"accepted snapshot has no pinned season page for requested {season}"
                    )
            merged, counts = _with_fixture_corrections(
                root, merged, set(seasons), clock=ctx.clock, status=base.status, run_id=store.run_id
            )
        except corrections.CorrectionError as exc:
            hint = "supply --evidence-root and --evidence-snapshot with the requested season's verified pinned capture"
            raise _Failure(EXIT_SOURCE, "evidence_unavailable", str(exc), hint) from exc
        relinks = corrections.replay_link_corrections(root, merged.manifest)
        if relinks["deletes"]:
            merged = snapshots.apply_upserts(
                root, merged.manifest,
                {k: relinks[k] for k in ("player_games", "quarantine", "quality_issues")},
                deletes={"player_games": relinks["deletes"]}, allow_empty=True, clock=ctx.clock,
                code_version=CODE_VERSION, status=base.status, run_id=store.run_id,
                notes=[f"replayed drawn finals: {len(relinks['player_games'])} rows relinked by team scores, "
                       f"{len(relinks['quarantine'])} quarantined"],
            )  # fmt: skip
            counts["replay_relinked"] = len(relinks["player_games"])
            counts["replay_quarantined"] = len(relinks["quarantine"])
        with SnapshotQuery(root, merged.manifest, tables={"matches"}) as q:
            all_seasons = {int(s) for (s,) in q.rows("SELECT DISTINCT season FROM matches")}
        merged = _with_resolved_blanks(root, merged, all_seasons, clock=ctx.clock, status=base.status,
                                       run_id=store.run_id)  # fmt: skip
        result.snapshot_id = merged.manifest.snapshot_id
        result.outputs["corrections"] = counts
        if merged.manifest.snapshot_id == base.snapshot_id:
            store.transition(RunState.VALIDATED)
            return  # nothing to correct
        cand = DatasetCandidate(candidate=merged, report={"corrections": result.outputs["corrections"]}, data_root=root)
        report = reconcile.validate_dataset(cand, reconcile.load_policy())
        _write_json(store.directory / "validation-report.json", _report_json(report))
        if not report.ok:
            raise _Failure(EXIT_VALIDATION, "validation_failed", "corrected candidate failed validation",
                           f"read {store.directory}/validation-report.json")  # fmt: skip
        store.transition(RunState.VALIDATED)
        snapshots.promote(root, merged, report, promoted_at=ctx.clock())
        store.transition(RunState.DATASET_PROMOTED)
        result.promoted = True

    return _locked(ctx, "apply-corrections", run_id, body)


def apply_reconciliation_corrections(
    ctx: RunContext, *, changes_file: Path, expected_snapshot: str, run_id: str | None = None
) -> StageResult:
    """Offline: apply source-backed reconciliation corrections (``reconciliation.corrections``) to the accepted
    snapshot, validate, and promote a child snapshot. Refuses unless the accepted snapshot is the one the changes
    were audited against, and refuses any change whose stored value is not the audited one."""

    def body(store: runs.RunStore, result: StageResult) -> None:
        from supercoach_via.ingest import reconcile
        from supercoach_via.ingest.legacy import DatasetCandidate
        from supercoach_via.reconciliation import corrections as CO
        from supercoach_via.storage import snapshots

        store.transition(RunState.PLANNED)
        root = ctx.data_root
        base = snapshots.load_snapshot(root, verify=True)
        if base.snapshot_id != expected_snapshot:
            raise _Failure(EXIT_INVALID, "snapshot_mismatch",
                           f"accepted snapshot is {base.snapshot_id}, the changes were audited on {expected_snapshot}",
                           "re-run the audit on the accepted snapshot and propose the changes again")  # fmt: skip
        changes = [c for c in CO.read_changes(changes_file) if c.layer == "snapshot"]
        try:
            ups = CO.snapshot_upserts(root, base, changes)
        except CO.CorrectionConflict as exc:
            raise _Failure(EXIT_INVALID, "correction_conflict", str(exc),
                           "re-run the audit; a change must match the value the audit saw") from exc  # fmt: skip
        store.transition(RunState.PARSED)
        digest = hashlib.sha256(changes_file.read_bytes()).hexdigest()
        dels = ups.pop("_delete_player_games", [])
        q_dels = ups.pop("_delete_quarantine", [])
        deletes = {k: v for k, v in (("player_games", dels), ("quarantine", q_dels)) if v}
        merged = snapshots.apply_upserts(
            root, base, {k: v for k, v in ups.items() if v}, clock=ctx.clock, code_version=CODE_VERSION,
            status=base.status, run_id=store.run_id, deletes=deletes or None,
            notes=[f"reconciliation corrections {digest[:16]}: {len(changes)} changes, "
                   f"{len(ups['player_games'])} player_games rows"],
        )  # fmt: skip
        result.snapshot_id = merged.manifest.snapshot_id
        result.outputs["corrections"] = {"changes": len(changes), "player_games_rows": len(ups["player_games"]),
                                         "players_rows": len(ups["players"]), "player_games_deleted": len(dels),
                                         "changes_sha256": digest}  # fmt: skip
        cand = DatasetCandidate(candidate=merged, report={"corrections": result.outputs["corrections"]}, data_root=root)
        report = reconcile.validate_dataset(cand, reconcile.load_policy())
        _write_json(store.directory / "validation-report.json", _report_json(report))
        if not report.ok:
            raise _Failure(EXIT_VALIDATION, "validation_failed", "corrected candidate failed validation",
                           f"read {store.directory}/validation-report.json")  # fmt: skip
        store.transition(RunState.VALIDATED)
        snapshots.promote(root, merged, report, promoted_at=ctx.clock())
        store.transition(RunState.DATASET_PROMOTED)
        result.promoted = True

    return _locked(ctx, "apply-reconciliation-corrections", run_id, body)



def ingest(
    ctx: RunContext,
    *,
    source_root: Path,
    repairs: Sequence[tuple[Path, int]] = (),
    current_season: int | None = None,
    run_id: str | None = None,
    promote: bool = True,
) -> StageResult:
    """Import the legacy corpus, apply archived repairs, validate and (on PASS) promote."""
    from supercoach_via.ingest.legacy import write_import_report
    from supercoach_via.storage.snapshots import promote as do_promote

    def body(store: runs.RunStore, result: StageResult) -> None:
        store.transition(RunState.PLANNED)
        store.event("plan", source_root=str(source_root), repairs=[str(p) for p, _ in repairs])
        cand = _timed(result, "import", lambda: _import(source_root, ctx))
        store.record_step("import", input_hashes={"inputs": str(cand.report.get("input_aggregate_sha256"))},
                          outputs={"snapshot_id": cand.snapshot_id}, state="succeeded",
                          counts={k: int(v) for k, v in cand.report.get("tables", {}).items()})  # fmt: skip
        for evidence_dir, season in repairs:
            t0 = time.perf_counter()
            cand = apply_repair_evidence(ctx, cand, evidence_dir, season=season)
            result.timings[f"repair:{evidence_dir.name}"] = round(time.perf_counter() - t0, 3)
            store.record_step(f"repair:{evidence_dir.name}", input_hashes={"evidence": str(evidence_dir)},
                              outputs={"snapshot_id": cand.snapshot_id}, state="succeeded")  # fmt: skip
        write_import_report(cand, store.directory / "import-report.json")
        store.transition(RunState.PARSED)
        result.snapshot_id = cand.snapshot_id
        report = _timed(result, "validate", lambda: _validate(cand, current_season=current_season))
        _write_json(store.directory / "validation-report.json", _report_json(report))
        store.record_step("validate", input_hashes={"snapshot_id": cand.snapshot_id},
                          outputs={"outcome": report.outcome.value}, state="succeeded" if report.ok else "failed",
                          counts={k: int(v) for k, v in report.counts.items()})  # fmt: skip
        if not report.ok:
            blocking = [i for i in report.issues if i.get("severity") == "blocking"]
            raise _Failure(EXIT_VALIDATION, "validation_failed",
                           f"validation {report.outcome.value}: {len(blocking)} blocking issue(s)",
                           f"read {store.directory}/validation-report.json, repair, re-run ingest")  # fmt: skip
        store.transition(RunState.VALIDATED)
        if not promote:
            return
        _timed(result, "promote", lambda: do_promote(cand.data_root, cand.candidate, report, promoted_at=ctx.clock()))
        store.record_step("promote", input_hashes={"snapshot_id": cand.snapshot_id},
                          outputs={"current": cand.snapshot_id}, state="succeeded")  # fmt: skip
        store.transition(RunState.DATASET_PROMOTED)
        result.promoted = True

    return _locked(ctx, "ingest", run_id, body)


def refresh(
    ctx: RunContext,
    *,
    season: int | None = None,
    repair_season: int | None = None,
    run_id: str | None = None,
    request: Any = None,
) -> StageResult:
    """Networked data-only refresh (``ctx.http`` required): fetch -> merge -> validate -> promote.

    A result that is not promotable as verified (any mandatory fetch failed, quarantined or
    UNKNOWN) ends PARTIAL with exit 3 and never moves the accepted pointer.
    """
    from supercoach_via.ingest import reconcile
    from supercoach_via.ingest import refresh as rf
    from supercoach_via.ingest.legacy import DatasetCandidate
    from supercoach_via.storage import snapshots

    def body(store: runs.RunStore, result: StageResult) -> None:
        base_manifest = snapshots.load_snapshot(ctx.data_root)
        base = rf.base_state_from_snapshot(ctx.data_root, base_manifest.snapshot_id)
        req = request or rf.RefreshRequest(current_season=season, repair_season=repair_season)
        plan = rf.plan_refresh(base, req, ctx)
        _write_json(store.directory / "refresh-plan.json", plan.to_dict())
        store.transition(RunState.PLANNED)
        store.transition(RunState.FETCHING)
        res = _timed(result, "fetch", lambda: rf.refresh_sources(base, plan, ctx))
        summary = res.summary()
        _write_json(store.directory / "refresh-result.json", {**summary, "work_log": res.work_log})
        result.outputs = {"summary": summary}
        store.record_step("fetch", input_hashes={"base": base_manifest.snapshot_id},
                          outputs={"outcome": res.outcome.value}, state="succeeded" if res.promotable_as_verified
                          else "failed", counts={k: v.attempted for k, v in res.counts.items()})  # fmt: skip
        if not res.promotable_as_verified:
            store.transition(RunState.PARTIAL, error_code="source_unavailable",
                             recovery="inspect refresh-result.json; retry when the source is reachable")  # fmt: skip
            result.exit_code, result.error_code = EXIT_SOURCE, "source_unavailable"
            result.message = f"refresh {res.outcome.value}: {res.issues[:3]}"
            return
        store.transition(RunState.PARSED)
        ups = {k: v for k, v in res.upserts.items() if v}
        if not ups:
            store.transition(RunState.VALIDATED)
            result.snapshot_id = base_manifest.snapshot_id
            return  # nothing changed at the source
        merged = snapshots.apply_upserts(ctx.data_root, base_manifest, ups, clock=ctx.clock, code_version=CODE_VERSION,
                                         status=base_manifest.status, source_revisions=res.revisions,
                                         notes=[f"refresh {plan.current_season}"])  # fmt: skip
        merged = _with_resolved_blanks(
            ctx.data_root, merged, {int(r["season"]) for r in ups.get("player_games") or []}, clock=ctx.clock,
            status=base_manifest.status, run_id=merged.manifest.run_id,
        )  # fmt: skip
        pinned = {int(k.rsplit(":", 1)[1]) for k in res.revisions if k.startswith("afltables:season:")}
        from supercoach_via.ingest.corrections import CorrectionError

        try:
            merged, _fixes = _with_fixture_corrections(
                ctx.data_root, merged, pinned, clock=ctx.clock, status=base_manifest.status,
                run_id=merged.manifest.run_id,
            )  # fmt: skip
        except CorrectionError as exc:
            raise _Failure(EXIT_SOURCE, "evidence_unavailable", str(exc), "re-run the refresh") from exc
        merged = _with_season_aggregates(
            ctx.data_root, merged, ups.get("matches") or [], clock=ctx.clock, status=base_manifest.status,
            checked_at=res.source_checked_at, checked_seasons=set(plan.seasons), run_id=merged.manifest.run_id,
        )  # fmt: skip
        target = plan.current_season
        relink = reconcile.relink_quarantined_lineups(ctx.data_root, merged.manifest, season=target)
        if any(relink.values()):
            merged = snapshots.apply_upserts(ctx.data_root, merged.manifest, relink, clock=ctx.clock,
                                             code_version=CODE_VERSION, status=base_manifest.status)  # fmt: skip
        cand = DatasetCandidate(candidate=merged, report={"refresh": summary}, data_root=ctx.data_root)
        result.snapshot_id = cand.snapshot_id
        report = _validate(cand)
        _write_json(store.directory / "validation-report.json", _report_json(report))
        if not report.ok:
            raise _Failure(EXIT_VALIDATION, "validation_failed", "refreshed candidate failed validation",
                           f"read {store.directory}/validation-report.json")  # fmt: skip
        store.transition(RunState.VALIDATED)
        snapshots.promote(ctx.data_root, merged, report, promoted_at=ctx.clock())
        store.transition(RunState.DATASET_PROMOTED)
        result.promoted = True

    return _locked(ctx, "refresh", run_id, body)


# ---------------------------------------------------------------------------
# Forecast + evaluation (train reuses an identical cached bundle; never tunes implicitly)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastPlan:
    train_cutoff: date
    calibration_end: date
    forecast_cutoff: datetime
    replay_season: int | None = None
    replay_stage_ids: tuple[str, ...] | None = None
    training_overrides: dict[str, Any] = field(default_factory=dict)


def forecast(
    ctx: RunContext, plan: ForecastPlan, *, snapshot: str = "current", run_id: str | None = None
) -> StageResult:
    """Train (or reuse) the model bundle, forecast the real next fixtures, replay-evaluate."""
    import warnings

    from supercoach_via.ml import evaluate as E
    from supercoach_via.ml import features as F
    from supercoach_via.ml import predict as P
    from supercoach_via.ml import train as T

    def body(store: runs.RunStore, result: StageResult) -> None:
        store.transition(RunState.PLANNED)
        hist = _timed(result, "load_history", lambda: F.load_history(ctx.data_root, snapshot))
        result.snapshot_id = hist.snapshot_id
        cfg = T.TrainingConfig(train_cutoff=plan.train_cutoff, calibration_end=plan.calibration_end,
                               **plan.training_overrides)  # fmt: skip
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            models = ctx.data_root / "models"
            trained = _timed(result, "train", lambda: T.train_model(hist, cfg, bundle_root=models, clock=ctx.clock))
            bundle = trained.bundle
            store.record_step("train", input_hashes={"snapshot_id": hist.snapshot_id},
                              outputs={"bundle_id": bundle.bundle_id, "reused": str(trained.reused),
                                       "champion": str(bundle.manifest.name)}, state="succeeded")  # fmt: skip
            req = P.ForecastRequest(forecast_cutoff=plan.forecast_cutoff, generated_at=ctx.clock())
            try:
                art = _timed(result, "forecast", lambda: P.forecast(hist, bundle, req))
            except P.ModelEligibilityError as exc:
                raise _Failure(
                    EXIT_MODEL, "model_ineligible", str(exc),
                    "train or choose a bundle whose knowledge cutoff is on or before this forecast",
                ) from exc
            pred_root = ctx.data_root / "predictions"
            pdir = pred_root / art.manifest.prediction_run_id
            if not pdir.exists():
                pdir = P.write_artifact(art, pred_root)
            store.record_step("forecast", input_hashes={"bundle_id": bundle.bundle_id},
                              outputs={"prediction_dir": str(pdir), "status": str(art.manifest.status),
                                       "reason": str(art.manifest.reason)}, state="succeeded")  # fmt: skip
            evaluations = []
            if plan.replay_season is not None:
                kwargs: dict[str, Any] = {"season": plan.replay_season, "generated_at": ctx.clock()}
                if plan.replay_stage_ids is not None:
                    kwargs["stage_ids"] = plan.replay_stage_ids
                try:
                    _arts, ev = _timed(result, "replay", lambda: E.replay(hist, bundle, **kwargs))
                except P.ModelEligibilityError as exc:
                    raise _Failure(
                        EXIT_MODEL, "model_ineligible", str(exc),
                        "replay needs a bundle whose knowledge cutoff is on or before each stage cutoff",
                    ) from exc
                edir = E.write_evaluation(ev, ctx.data_root / "evaluations")
                evaluations.append(str(edir))
                result.artifacts.setdefault("evaluations", []).append(ev)
                store.record_step("replay", input_hashes={"bundle_id": bundle.bundle_id},
                                  outputs={"evaluation_id": ev.evaluation_id}, state="succeeded")  # fmt: skip
        store.transition(RunState.PARSED)  # nothing fetched: planned -> parsed
        result.outputs = {"bundle_id": bundle.bundle_id, "bundle_root": str(ctx.data_root / "models"),
                          "prediction_dir": str(pdir), "forecast_status": str(art.manifest.status),
                          "forecast_reason": art.manifest.reason, "evaluations": evaluations,
                          "promoted_model": bool(bundle.manifest.promoted)}  # fmt: skip
        if art.manifest.status != "available":
            result.error_code = "forecast_unavailable"
            result.message = f"forecast unavailable: {art.manifest.reason}"

    return _locked(ctx, "forecast", run_id, body)


# ---------------------------------------------------------------------------
# Release: build -> validate (publication is a separate command)
# ---------------------------------------------------------------------------


def build(ctx: RunContext, inputs: Any, *, snapshot: str = "current", run_id: str | None = None) -> StageResult:
    """Build and validate a release from the accepted snapshot; never publishes."""
    from supercoach_via.publish.builder import build_release
    from supercoach_via.storage.snapshots import SnapshotRef, load_snapshot, snapshot_hex

    def body(store: runs.RunStore, result: StageResult) -> None:
        manifest = load_snapshot(ctx.data_root, snapshot)
        path = ctx.data_root / "snapshots" / f"{snapshot_hex(manifest.snapshot_id)}.json"
        ref = SnapshotRef(manifest.snapshot_id, path)
        result.snapshot_id = manifest.snapshot_id
        store.transition(RunState.PLANNED)
        cand = _timed(result, "build_release", lambda: build_release(ref, inputs, ctx))
        store.record_step("build_release", input_hashes={"snapshot_id": manifest.snapshot_id},
                          outputs={"release_id": cand.release_id, "release_dir": str(cand.release_dir)},
                          state="succeeded" if cand.validation.ok else "failed", counts=dict(cand.counts))  # fmt: skip
        result.outputs = {"release_id": cand.release_id, "release_dir": str(cand.release_dir),
                          "forecast_status": cand.forecast_status, "forecast_reason": cand.forecast_reason,
                          "reused": cand.reused, "warnings": cand.warnings[:50], "counts": cand.counts}  # fmt: skip
        result.timings.update({f"build:{k}": v for k, v in cand.timings.items()})
        if cand.validation.outcome is not CheckOutcome.PASS:
            raise _Failure(EXIT_VALIDATION, "release_validation_failed",
                           f"release {cand.release_id} failed validation: {cand.validation.issues[:3]}",
                           "fix the failing resource and re-run build-release")  # fmt: skip
        store.transition(RunState.PARSED)

    return _locked(ctx, "build-release", run_id, body)


# ---------------------------------------------------------------------------
# Offline demo: synthetic DEMO corpus -> full release, no network/credentials
# ---------------------------------------------------------------------------

DEMO_CLOCK = datetime(2026, 5, 1, 6, 0, tzinfo=UTC)


def demo(out: Path) -> dict[str, Any]:
    """Build the complete DEMO dataset and release under ``out`` (deterministic clock).

    Layout: ``out/source`` (demo corpus), ``out/var`` (data root), ``out/releases/<id>`` (``out`` is the
    release output root, so ``validate-release``/``preview --output-root out`` find it).
    """
    from supercoach_via.demo import write_demo_corpus
    from supercoach_via.settings import Settings

    src = out / "source"
    write_demo_corpus(src)
    from supercoach_via.settings import default_config_dir

    repo_config = default_config_dir()
    ctx = RunContext(settings=Settings(data_root=out / "var", output_root=out, source_root=src),
                     clock=lambda: DEMO_CLOCK)  # fmt: skip
    steps: dict[str, Any] = {}
    ing = ingest(ctx, source_root=src)
    steps["ingest"] = ing.as_dict()
    if ing.exit_code:
        return {"ok": False, "exit_code": ing.exit_code, "steps": steps}
    fc = forecast(ctx, ForecastPlan(
        train_cutoff=date(2025, 6, 1), calibration_end=date(2026, 1, 1), forecast_cutoff=DEMO_CLOCK,
        replay_season=2026, replay_stage_ids=("r08", "r09"),
        # Hold the promotion gate at 1 May so the 06:00Z forecast and the later completed
        # rounds are eligible. Earlier 2026 rounds are inside that gate and are not replayed.
        training_overrides={"target_seasons_from": 2024, "candidates": ("hgb",), "n_folds": 2, "threads": 1,
                            "holdout_end": date(2026, 5, 1), "min_calibration": 50, "cohort_min_n": 20,
                            "params": {"hgb": {"max_iter": 10, "learning_rate": 0.1, "max_leaf_nodes": 7,
                                               "min_samples_leaf": 5}}},
    ))  # fmt: skip
    steps["forecast"] = fc.as_dict()
    if fc.exit_code:
        return {"ok": False, "exit_code": fc.exit_code, "steps": steps}
    inputs = _release_inputs(ctx, fc, demo=True, config_dir=repo_config, clock=lambda: DEMO_CLOCK)
    rel = build(ctx, inputs)
    steps["build_release"] = rel.as_dict()
    return {"ok": rel.exit_code == 0, "exit_code": rel.exit_code, "steps": steps,
            "release_dir": rel.outputs.get("release_dir"), "release_id": rel.outputs.get("release_id")}  # fmt: skip


def _release_inputs(ctx: RunContext, fc: StageResult, **kw: Any) -> Any:
    from supercoach_via.ml import bundles as B
    from supercoach_via.publish.builder import ReleaseInputs

    o = fc.outputs
    manifest = B.read_manifest(Path(o["bundle_root"]), o["bundle_id"])
    evaluations = tuple(fc.artifacts.get("evaluations", ()))
    return ReleaseInputs(prediction_dirs=(Path(o["prediction_dir"]),), model_manifests=(manifest,),
                         evaluations=evaluations, **kw)  # fmt: skip
