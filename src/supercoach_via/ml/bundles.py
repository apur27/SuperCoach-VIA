"""Trusted-local model bundle persistence (PLAN 7.3).

Layout under a bundle root (e.g. ``<data_root>/models``)::

    <bundle_id>/manifest.json     BundleManifest (self-hash in ``manifest_sha256``)
    <bundle_id>/predictor.joblib  fitted FittedPredictor (preprocessing + estimator)

``bundle_id`` is derived from the cache key, which covers every semantic training input
(snapshot, cutoffs, population, feature spec/code, folds, estimator + dependency versions,
hyperparameters, seed, device, thread budget). A bundle directory is written once
(temp dir + rename) and never overwritten.

SECURITY: ``predictor.joblib`` is Python serialization = trusted code execution. It is
loaded only (a) from a directory under the configured trusted bundle root, (b) after the
manifest self-hash verifies and (c) after the payload SHA-256 matches the manifest.
Never pass user-supplied or downloaded files to ``load_bundle``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from supercoach_via.domain.schemas import is_safe_id
from supercoach_via.storage.snapshots import atomic_write_bytes, contained_path, sha256_file

BUNDLE_SCHEMA_VERSION = 1
PAYLOAD_NAME = "predictor.joblib"


class BundleIntegrityError(RuntimeError):
    """Manifest or payload failed verification; nothing was deserialized."""


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def json_safe(obj: Any) -> Any:
    """Recursively replace NaN/inf with None so manifests round-trip exactly."""
    if isinstance(obj, float):
        return None if obj != obj or obj in (float("inf"), float("-inf")) else obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        return json_safe(obj.item())
    return obj


def cache_key(inputs: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(inputs)).hexdigest()


def dependency_versions() -> dict[str, str]:
    import numpy
    import pandas
    import sklearn

    out = {"numpy": numpy.__version__, "pandas": pandas.__version__, "scikit-learn": sklearn.__version__}
    try:
        import lightgbm

        out["lightgbm"] = lightgbm.__version__
    except ImportError:
        out["lightgbm"] = "absent"
    return out


class BundleManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: int = BUNDLE_SCHEMA_VERSION
    bundle_id: str
    kind: str  # baseline | model
    name: str
    description: str
    created_at: datetime
    cache_key: str
    cache_inputs: dict[str, Any]
    snapshot_id: str
    train_cutoff: str
    calibration_end: str
    holdout_end: str | None
    feature_version: str
    feature_names: list[str]
    feature_dtypes: dict[str, str]
    params: dict[str, Any]
    seed: int
    threads: int
    device: str = "cpu"
    versions: dict[str, str]
    cold_start: dict[str, Any]
    recalibration: dict[str, Any]
    interval: dict[str, Any]
    training: dict[str, Any] = Field(description="inner-fold OOF metrics, timings, row counts")
    holdout: dict[str, Any] = Field(description="matched holdout metrics, cohorts, gate")
    promoted: bool
    promotion_note: str
    trusted_local: bool = True
    payload_file: str = PAYLOAD_NAME
    payload_sha256: str
    manifest_sha256: str = ""

    def self_hash(self) -> str:
        data = self.model_dump(mode="json")
        data["manifest_sha256"] = ""
        return hashlib.sha256(canonical_json(data)).hexdigest()


@dataclass
class ModelBundle:
    manifest: BundleManifest
    predictor: Any  # FittedPredictor
    directory: Path | None = None

    @property
    def bundle_id(self) -> str:
        return self.manifest.bundle_id


def make_bundle_id(name: str, key: str) -> str:
    bid = f"{name}-{key[:20]}"
    if not is_safe_id(bid):
        raise ValueError(f"unsafe bundle id {bid!r}")
    return bid


def save_bundle(root: Path, manifest: BundleManifest, predictor: Any) -> ModelBundle:
    """Write a new bundle; if ``bundle_id`` already exists it is verified and reused, never
    overwritten."""
    import joblib  # type: ignore[import-untyped]

    root.mkdir(parents=True, exist_ok=True)
    final = contained_path(root, manifest.bundle_id)
    if final.exists():
        return load_bundle(root, manifest.bundle_id)
    tmp = Path(tempfile.mkdtemp(prefix=f".{manifest.bundle_id}.", dir=root))
    try:
        payload = tmp / PAYLOAD_NAME
        joblib.dump(predictor, payload, compress=3)
        m = manifest.model_copy(update={"payload_sha256": sha256_file(payload)})
        m = m.model_copy(update={"manifest_sha256": m.self_hash()})
        atomic_write_bytes(tmp / "manifest.json", m.model_dump_json(indent=1).encode())
        try:
            tmp.rename(final)  # atomic; fails if a concurrent writer created it
        except OSError:
            if final.exists():
                return load_bundle(root, manifest.bundle_id)
            raise
        for p in final.iterdir():
            p.chmod(0o444)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return ModelBundle(m, predictor, final)


def read_manifest(root: Path, bundle_id: str) -> BundleManifest:
    if not is_safe_id(bundle_id):
        raise BundleIntegrityError(f"unsafe bundle id {bundle_id!r}")
    d = contained_path(root, bundle_id)
    path = d / "manifest.json"
    if not path.is_file() or path.is_symlink():
        raise BundleIntegrityError(f"missing manifest for {bundle_id}")
    m = BundleManifest.model_validate_json(path.read_bytes())
    if m.bundle_id != bundle_id:
        raise BundleIntegrityError("manifest bundle_id does not match its directory")
    if m.manifest_sha256 != m.self_hash():
        raise BundleIntegrityError("manifest self-hash mismatch")
    if not m.trusted_local:
        raise BundleIntegrityError("bundle is not marked trusted-local")
    return m


def load_bundle(root: Path, bundle_id: str) -> ModelBundle:
    """Verify manifest + payload hash, then (trusted-local) deserialize the predictor."""
    import joblib

    m = read_manifest(root, bundle_id)
    d = contained_path(root, bundle_id)
    payload = contained_path(d, m.payload_file)
    if not payload.is_file() or payload.is_symlink():
        raise BundleIntegrityError("missing payload")
    if sha256_file(payload) != m.payload_sha256:
        raise BundleIntegrityError("payload hash does not match manifest")
    predictor = joblib.load(payload)  # trusted-local only, after verification above
    return ModelBundle(m, predictor, d)


def model_card_facts(m: BundleManifest) -> list[str]:
    """Plain-text, number-bearing facts for docs/model-card.md, all read from the manifest."""
    h = m.holdout
    t = m.training
    gate = h.get("gate") or {}
    metrics = h.get("metrics") or {}
    facts = [
        f"Shipped model: {m.name} ({'promoted ML model' if m.promoted else 'transparent baseline'}); "
        f"bundle {m.bundle_id}.",
        f"Promotion decision: {m.promotion_note}",
        f"Training snapshot {m.snapshot_id}; training rows dated before {m.train_cutoff}; "
        f"calibration block {m.train_cutoff} to {m.calibration_end}; holdout from {m.calibration_end}"
        + (f" to {m.holdout_end}" if m.holdout_end else " onward") + ".",
        f"Features: {m.feature_version}, {len(m.feature_names)} columns; outcome features use only "
        "games that finished before each target's cutoff.",
        f"Rows: {t.get('blocks')}; excluded before training: {t.get('excluded')}.",
        f"Selection: {h.get('selection_basis')}; selected candidate {h.get('selected_candidate')}.",
    ]
    for name, mb in sorted(metrics.items()):
        facts.append(f"Holdout {name}: n={mb.get('n')}, MAE={mb.get('mae')}, RMSE={mb.get('rmse')}, "
                     f"bias={mb.get('bias')}, within-5={mb.get('within_5')}.")
    if gate:
        facts.append(f"Gate: candidate MAE {gate.get('candidate_mae')} vs prior-5 baseline "
                     f"{gate.get('baseline_mae')} (relative improvement {gate.get('relative_improvement')}); "
                     f"passed={gate.get('passed')}.")
    iv = m.interval
    hold = iv.get("holdout") or {}
    if iv.get("available"):
        facts.append(f"Interval: {iv.get('method')} at level {iv.get('level')} from {iv.get('n_calibration')} "
                     f"calibration outcomes; holdout coverage {hold.get('coverage')}, median width "
                     f"{hold.get('median_width')}; calibrated={hold.get('calibrated')}. Temporal drift can "
                     "break the exchangeability this method assumes.")
    else:
        facts.append(f"Interval: unavailable ({iv.get('reason')}).")
    facts.append(f"Cold starts: {m.cold_start.get('name')} = {m.cold_start.get('value')} disposals for targets "
                 "with no eligible prior game. Outputs constrained to >= 0 only (no upper clip).")
    facts.append(f"Seed {m.seed}; threads {m.threads}; device {m.device}; versions {m.versions}.")
    return facts
