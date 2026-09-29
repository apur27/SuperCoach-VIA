"""Family F: model bundles and prediction artifacts, checked without loading or training a model.

Bundles are verified by manifest self-hash, streamed payload SHA-256, the persisted feature
specification (rebuilt and fingerprinted by ``ml.features``), feature order, the recorded
code fingerprint and the knowledge cutoff. Predictions are verified by their file hashes,
labels, finite values, interval order and eligibility. ``predictor.joblib`` is never
deserialized. An unavailable forecast with no future fixture is valid; one that ignores a
scheduled future fixture is not.
"""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from supercoach_via.domain.schemas import Origin, Severity
from supercoach_via.integrity.capture import _read_regular, sha256_hex, strict_json
from supercoach_via.integrity.context import AuditContext, CheckSkipped, CheckSpec, rule
from supercoach_via.integrity.report import Kind, Status

B, E, W = Severity.BLOCKING, Severity.ERROR, Severity.WARNING

RULES = [
    rule(
        "models.bundle_invalid",
        "models.bundles",
        B,
        "a bundle manifest does not parse under its contract",
        "retrain; never edit a bundle manifest",
    ),
    rule(
        "models.bundle_hash",
        "models.bundles",
        B,
        "a bundle manifest's self-hash or directory name is wrong",
        "treat the bundle as tampered; retrain",
    ),
    rule(
        "models.payload_missing",
        "models.bundles",
        B,
        "a bundle's predictor payload is missing",
        "restore or retrain the bundle",
    ),
    rule(
        "models.payload_hash",
        "models.bundles",
        B,
        "a bundle's payload bytes do not match its manifest",
        "treat the payload as untrusted; never load it; retrain",
    ),
    rule(
        "models.feature_spec",
        "models.bundles",
        B,
        "the persisted feature specification cannot be rebuilt or does not match its fingerprint",
        "retrain with the current feature code",
    ),
    rule(
        "models.feature_order",
        "models.bundles",
        B,
        "the bundle's feature order disagrees with its persisted feature list",
        "retrain; serving must use the training feature order",
    ),
    rule(
        "models.code_fingerprint",
        "models.bundles",
        W,
        "the bundle was trained with different feature/model code than this checkout",
        "retrain before the next forecast (forecast refuses a stale bundle)",
        kind=Kind.ANOMALY,
    ),
    rule(
        "models.knowledge_cutoff_missing",
        "models.bundles",
        E,
        "the bundle records no knowledge cutoff",
        "retrain so the manifest records one",
    ),
    rule(
        "models.knowledge_after_as_of",
        "models.bundles",
        B,
        "the bundle learned from outcomes after the audit's --as-of",
        "audit later or use an earlier bundle",
    ),
    rule(
        "models.bundle_missing",
        "models.predictions",
        B,
        "an available forecast names a bundle that is not present",
        "restore the bundle or rebuild the forecast",
    ),
    rule(
        "models.ineligible_bundle",
        "models.predictions",
        B,
        "a forecast cutoff precedes the bundle's knowledge cutoff (the model saw later outcomes)",
        "forecast with a bundle whose knowledge cutoff is on or before the cutoff",
    ),
    rule(
        "models.prediction_invalid",
        "models.predictions",
        B,
        "a prediction manifest does not parse",
        "rebuild the forecast",
    ),
    rule(
        "models.prediction_file_hash",
        "models.predictions",
        B,
        "a prediction file's bytes differ from its manifest",
        "rebuild the forecast",
    ),
    rule(
        "models.prediction_label",
        "models.predictions",
        B,
        "a prediction's origin/status/reason is outside the contract or inconsistent",
        "rebuild the forecast",
    ),
    rule(
        "models.row_identity",
        "models.predictions",
        B,
        "a prediction row names another run, snapshot, model, origin or cutoff than its manifest",
        "rebuild the forecast",
    ),
    rule(
        "models.prediction_value",
        "models.predictions",
        B,
        "a predicted value is missing, negative or not finite",
        "rebuild the forecast",
    ),
    rule(
        "models.interval_order",
        "models.predictions",
        B,
        "a prediction interval is not low <= prediction <= high",
        "rebuild the forecast",
    ),
    rule(
        "models.row_count",
        "models.predictions",
        B,
        "the manifest's predicted count differs from its rows",
        "rebuild the forecast",
    ),
    rule(
        "models.target_unknown",
        "models.predictions",
        E,
        "a prediction targets a match the snapshot does not hold",
        "rebuild the forecast against the audited snapshot",
    ),
    rule(
        "models.forecast_after_as_of",
        "models.predictions",
        B,
        "a forecast cutoff or generation time is after --as-of",
        "audit later or audit an earlier forecast",
    ),
    rule(
        "models.prediction_snapshot",
        "models.predictions",
        W,
        "a prediction was made from another snapshot",
        "rebuild the forecast from the audited snapshot before publishing it",
        kind=Kind.ANOMALY,
    ),
    rule(
        "models.forecast_unavailable_with_fixture",
        "models.predictions",
        B,
        "the forecast says no future fixture, but the snapshot schedules one after the cutoff",
        "rebuild the forecast; unavailability must reflect the fixture",
    ),
    rule(
        "models.release_forecast",
        "models.predictions",
        B,
        "the release's forecast status/model disagrees with the prediction artifacts",
        "rebuild the release",
    ),
]


def _dirs(root: Path | None) -> list[Path]:
    if root is None or not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".") and not p.is_symlink())


def _utc(v: Any) -> datetime | None:
    if v is None:
        return None
    dt = v if isinstance(v, datetime) else datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).astimezone(UTC)


def _stream_sha(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _load_bundles(ctx: AuditContext) -> dict[str, dict[str, Any]]:
    from supercoach_via.ml.bundles import BundleManifest

    out: dict[str, dict[str, Any]] = {}
    for d in _dirs(ctx.models_root):
        raw, _why = _read_regular(d / "manifest.json")
        if raw is None:
            continue
        entry: dict[str, Any] = {"dir": d, "raw": raw, "manifest": None}
        try:
            entry["manifest"] = BundleManifest.model_validate(strict_json(raw))
        except ValueError as exc:
            entry["error"] = str(exc)[:300]
        out[d.name] = entry
    return out


def _load_predictions(ctx: AuditContext) -> dict[str, dict[str, Any]]:
    from supercoach_via.ml.predict import PredictionManifest

    out: dict[str, dict[str, Any]] = {}
    for d in _dirs(ctx.predictions_root):
        raw, _why = _read_regular(d / "manifest.json")
        if raw is None:
            continue
        entry: dict[str, Any] = {"dir": d, "raw": raw, "manifest": None}
        try:
            entry["manifest"] = PredictionManifest.model_validate(strict_json(raw))
        except ValueError as exc:
            entry["error"] = str(exc)[:300]
        out[d.name] = entry
    return out


def _inputs(ctx: AuditContext) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if "_models" not in ctx.coverage:
        bundles, preds = _load_bundles(ctx), _load_predictions(ctx)
        ctx.coverage["_models"] = (bundles, preds)
        ctx.coverage["model_inputs"] = {
            "bundles": [{"bundle_id": k, "manifest_sha256": sha256_hex(v["raw"])} for k, v in sorted(bundles.items())],
            "predictions": [
                {
                    "prediction_run_id": k,
                    "manifest_sha256": sha256_hex(v["raw"]),
                    "status": v["manifest"].status if v["manifest"] is not None else None,
                }
                for k, v in sorted(preds.items())
            ],
        }
    return ctx.coverage["_models"]  # type: ignore[no-any-return]


def _used_by_available(preds: dict[str, dict[str, Any]]) -> set[str]:
    return {
        p["manifest"].model_id
        for p in preds.values()
        if p["manifest"] is not None and p["manifest"].status == "available"
    }


def check_bundles(ctx: AuditContext) -> list[str]:
    from supercoach_via.ml.features import FeatureSpecError, feature_spec_from_stored
    from supercoach_via.ml.predict import ModelEligibilityError, bundle_knowledge_cutoff
    from supercoach_via.ml.train import _code_hash

    bundles, preds = _inputs(ctx)
    if not bundles:
        raise CheckSkipped(Status.NOT_APPLICABLE, "no model bundles supplied")
    used = _used_by_available(preds)
    code = _code_hash()
    for bid, entry in sorted(bundles.items()):
        ctx.count("bundles")
        entity = f"bundle:{bid}"
        m = entry["manifest"]
        if m is None:
            ctx.add("models.bundle_invalid", entity, message=entry.get("error", ""))
            continue
        if m.bundle_id != bid or m.manifest_sha256 != m.self_hash():
            ctx.add("models.bundle_hash", entity, expected=m.self_hash(), actual=m.manifest_sha256)
        payload = entry["dir"] / m.payload_file
        if not payload.is_file() or payload.is_symlink():
            ctx.add("models.payload_missing", entity, field="payload_file", actual=m.payload_file)
        else:
            got = _stream_sha(payload)
            if got != m.payload_sha256:
                ctx.add("models.payload_hash", entity, expected=m.payload_sha256, actual=got)
        config = (m.cache_inputs or {}).get("config") or {}
        try:
            feature_spec_from_stored(config.get("feature_spec"), feature_version=m.feature_version)
        except FeatureSpecError as exc:
            ctx.add("models.feature_spec", entity, message=str(exc)[:300])
        persisted = (m.cache_inputs or {}).get("feature_names")
        if persisted != m.feature_names or list(m.feature_dtypes) != m.feature_names:
            ctx.add("models.feature_order", entity, expected=persisted, actual=m.feature_names)
        if (m.cache_inputs or {}).get("code") != code:
            if bid in used:
                ctx.collector.add(
                    "models.code_fingerprint",
                    entity,
                    severity=Severity.BLOCKING,
                    expected=code,
                    actual=(m.cache_inputs or {}).get("code"),
                )
            else:
                ctx.add("models.code_fingerprint", entity, expected=code, actual=(m.cache_inputs or {}).get("code"))
        try:
            cutoff = bundle_knowledge_cutoff(m)
        except ModelEligibilityError:
            ctx.add("models.knowledge_cutoff_missing", entity)
            continue
        if ctx.as_of is not None and cutoff > ctx.as_of:
            ctx.add(
                "models.knowledge_after_as_of",
                entity,
                expected=f"<= {ctx.as_of.isoformat()}",
                actual=cutoff.isoformat(),
            )
    return []


def check_predictions(ctx: AuditContext) -> list[str]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from supercoach_via.ml.predict import NO_FIXTURE, ModelEligibilityError, bundle_knowledge_cutoff

    bundles, preds = _inputs(ctx)
    if not preds:
        raise CheckSkipped(Status.NOT_APPLICABLE, "no prediction artifacts supplied")
    ctx.need("matches")
    snap_id = ctx.snapshot.snapshot_id if ctx.snapshot is not None else None
    origins = {o.value for o in Origin}
    for run_id, entry in sorted(preds.items()):
        ctx.count("predictions")
        entity = f"prediction:{run_id}"
        m = entry["manifest"]
        if m is None:
            ctx.add("models.prediction_invalid", entity, message=entry.get("error", ""))
            continue
        d: Path = entry["dir"]
        if (
            m.prediction_run_id != run_id
            or m.origin not in origins
            or m.status not in ("available", "unavailable", "expired")
        ):
            ctx.add(
                "models.prediction_label",
                entity,
                actual={"run": m.prediction_run_id, "origin": m.origin, "status": m.status},
            )
        if m.snapshot_id != snap_id:
            ctx.add("models.prediction_snapshot", entity, expected=snap_id, actual=m.snapshot_id)
        cutoff, generated = _utc(m.forecast_cutoff), _utc(m.generated_at)
        if ctx.as_of is not None and ((cutoff and cutoff > ctx.as_of) or (generated and generated > ctx.as_of)):
            ctx.add(
                "models.forecast_after_as_of",
                entity,
                expected=f"<= {ctx.as_of.isoformat()}",
                actual={"cutoff": _iso(cutoff), "generated_at": _iso(generated)},
            )
        rows_raw, _ = _read_regular(d / m.rows_file)
        omit_raw, _ = _read_regular(d / m.omissions_file)
        for name, raw, want in (
            (m.rows_file, rows_raw, m.rows_sha256),
            (m.omissions_file, omit_raw, m.omissions_sha256),
        ):
            if raw is None or sha256_hex(raw) != want:
                ctx.add(
                    "models.prediction_file_hash",
                    entity,
                    field=name,
                    expected=want,
                    actual=sha256_hex(raw) if raw is not None else None,
                )
        rows: list[dict[str, Any]] = []
        if rows_raw is not None and sha256_hex(rows_raw) == m.rows_sha256:
            try:
                rows = pq.read_table(pa.BufferReader(rows_raw)).to_pylist()
            except Exception as exc:  # noqa: BLE001 - reported
                ctx.add("models.prediction_file_hash", entity, field=m.rows_file, message=f"unreadable: {exc}"[:300])
        ctx.count("rows", len(rows))
        if m.status == "available":
            if m.predicted != len(rows) or not rows:
                ctx.add("models.row_count", entity, expected=m.predicted, actual=len(rows))
            b = bundles.get(m.model_id)
            if b is None or b["manifest"] is None:
                ctx.add("models.bundle_missing", entity, field="model_id", actual=m.model_id)
            elif cutoff is not None:
                try:
                    knowledge = bundle_knowledge_cutoff(b["manifest"])
                    if cutoff < knowledge:
                        ctx.add(
                            "models.ineligible_bundle",
                            entity,
                            expected=f">= {knowledge.isoformat()}",
                            actual=cutoff.isoformat(),
                            evidence={"bundle": m.model_id},
                        )
                except ModelEligibilityError:
                    ctx.add("models.ineligible_bundle", entity, message="bundle records no knowledge cutoff")
        elif m.status == "unavailable":
            if rows:
                ctx.add("models.row_count", entity, expected=0, actual=len(rows))
            if m.reason == NO_FIXTURE and cutoff is not None:
                future = ctx.rows(
                    "SELECT match_id FROM matches WHERE status = 'scheduled' AND match_date >= ? ORDER BY 1",
                    [cutoff.date()],
                )
                if future:
                    ctx.add(
                        "models.forecast_unavailable_with_fixture",
                        entity,
                        expected=0,
                        actual=len(future),
                        evidence={"first": future[0][0]},
                    )
        match_ids = {r.get("match_id") for r in rows}
        known = (
            {
                m_
                for (m_,) in ctx.rows(
                    "SELECT match_id FROM matches WHERE list_contains(?, match_id)",
                    [sorted(str(x) for x in match_ids if x)],
                )
            }
            if rows
            else set()
        )
        for r in rows:
            rid = f"prediction_row:{run_id}|{r.get('prediction_id')}"
            ident = {
                "prediction_run_id": run_id,
                "snapshot_id": m.snapshot_id,
                "model_id": m.model_id,
                "origin": m.origin,
            }
            bad = {k: r.get(k) for k, v in ident.items() if r.get(k) != v}
            if _utc(r.get("forecast_cutoff")) != cutoff:
                bad["forecast_cutoff"] = _iso(_utc(r.get("forecast_cutoff")))
            if bad:
                ctx.add("models.row_identity", rid, expected=ident, actual=bad)
            p, lo, hi = r.get("predicted_disposals"), r.get("interval_low"), r.get("interval_high")
            if not isinstance(p, int | float) or not math.isfinite(p) or p < 0:
                ctx.add("models.prediction_value", rid, field="predicted_disposals", actual=str(p))
            elif (lo is None) != (hi is None) or (
                lo is not None
                and hi is not None
                and not (math.isfinite(lo) and math.isfinite(hi) and 0 <= lo <= p <= hi)
            ):
                ctx.add(
                    "models.interval_order",
                    rid,
                    expected="0 <= low <= prediction <= high",
                    actual=[_finite(lo), _finite(p), _finite(hi)],
                )
            if r.get("match_id") not in known:
                ctx.add("models.target_unknown", rid, field="match_id", actual=r.get("match_id"))
    _release_binding(ctx, preds, bundles)
    return []


def _finite(v: Any) -> Any:
    return v if not isinstance(v, float) or math.isfinite(v) else str(v)


def _iso(v: datetime | None) -> str | None:
    return v.isoformat().replace("+00:00", "Z") if v is not None else None


def _release_binding(ctx: AuditContext, preds: dict[str, dict[str, Any]], bundles: dict[str, dict[str, Any]]) -> None:
    cap = ctx.release
    if cap is None or "release.json" not in cap.public:
        return
    try:
        forecast = strict_json(cap.read_public("release.json")).get("forecast") or {}
    except Exception:  # noqa: BLE001 - the release family reports unreadable metadata
        return
    status, model_id = forecast.get("status"), forecast.get("model_id")
    mine = [
        p["manifest"]
        for p in preds.values()
        if p["manifest"] is not None
        and ctx.snapshot is not None
        and p["manifest"].snapshot_id == ctx.snapshot.snapshot_id
    ]
    if status == "available":
        if model_id not in bundles or not any(m.status == "available" and m.model_id == model_id for m in mine):
            ctx.add(
                "models.release_forecast",
                f"release:{cap.release_id}",
                field="forecast",
                expected="an available prediction and bundle for this snapshot",
                actual={"status": status, "model_id": model_id},
            )
        if forecast.get("artifact") not in cap.public:
            ctx.add(
                "models.release_forecast",
                f"release:{cap.release_id}",
                field="forecast.artifact",
                actual=forecast.get("artifact"),
            )
    elif (
        status == "unavailable"
        and mine
        and not any(m.status == "unavailable" and m.reason == forecast.get("reason") for m in mine)
    ):
        ctx.add(
            "models.release_forecast",
            f"release:{cap.release_id}",
            field="forecast.reason",
            expected=sorted({str(m.reason) for m in mine}),
            actual=forecast.get("reason"),
        )


CHECKS = [
    CheckSpec(
        "models.bundles",
        "models",
        "bundle manifests, payload hashes, feature spec/order and knowledge cutoff",
        check_bundles,
        required=True,
    ),
    CheckSpec(
        "models.predictions",
        "models",
        "prediction files, labels, values, intervals, eligibility and availability",
        check_predictions,
        required=True,
    ),
]
