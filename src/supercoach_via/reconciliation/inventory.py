"""Input pinning and plan construction (DESIGN section 5). Offline: no network, no model.

``plan`` records exactly what an audit is about to compare: the resolved snapshot and every
fragment hash, the legacy CSV membership and content digests, the policies and rule files,
and the code that decides which pages are fetched. A plan is written atomically, refuses
output paths that alias an input, and cannot silently replace an incompatible plan.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import stat
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from supercoach_via.integrity.capture import SnapshotCapture
from supercoach_via.reconciliation import SCHEMA_VERSION
from supercoach_via.reconciliation import schema as S
from supercoach_via.reconciliation.urls import POLICY_FILE, SITE
from supercoach_via.settings import default_config_dir
from supercoach_via.storage.snapshots import atomic_write_bytes

RULES_FILE = "reconciliation_rules.toml"
OVERRIDES_FILE = "reconciliation_identity_overrides.csv"
#: package-relative files whose bytes decide which pages are fetched and how. ``schema.py`` is
#: deliberately absent: the plan/manifest models are append-only after a capture starts, and the
#: comparison-side models grow in the same file; ``schema_version`` guards their shape.
CAPTURE_FILES = (
    "reconciliation/__init__.py",
    "reconciliation/urls.py",
    "reconciliation/discover.py",
    "reconciliation/capture.py",
    "ingest/http.py",
    "integrity/sourcepages.py",
)
SEED_URLS = (
    f"{SITE}/robots.txt",
    f"{SITE}/afl/stats/stats_idx.html",
    f"{SITE}/afl/stats/notes.html",
)


class PlanError(RuntimeError):
    """The plan cannot be built or written safely (exit code 2)."""


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_identity() -> S.CodeIdentity:
    root = _package_root()
    files = {rel: _sha(root / rel) for rel in CAPTURE_FILES}
    commit: str | None = None
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        out = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        commit = out.stdout.strip() or None
    return S.CodeIdentity(repo_commit=commit, capture_files=files)


# ---------------------------------------------------------------------------
# Local inputs
# ---------------------------------------------------------------------------


def pin_snapshot(data_root: Path, selector: str) -> S.SnapshotPin:
    cap = SnapshotCapture.open(data_root, selector)
    if cap.problems or cap.manifest is None or cap.snapshot_id is None:
        detail = "; ".join(f"{rule}: {msg}" for rule, _entity, msg in cap.problems) or "no manifest"
        raise PlanError(f"snapshot {selector!r} cannot be loaded: {detail}")
    bad = [
        f"{f.table}/{f.ref.partition or ''}: {f.problem or f.parquet_error}" for f in cap.fragments if not f.verified
    ]
    if bad:
        raise PlanError(f"snapshot {cap.snapshot_id} has unverifiable fragments: {bad[:3]}")
    fragments: dict[str, dict[str, str]] = {}
    for f in cap.fragments:
        fragments.setdefault(f.table, {})[f.ref.partition or ""] = f.ref.sha256
    ident = cap.identity()
    return S.SnapshotPin(
        snapshot_id=cap.snapshot_id,
        selector=selector,
        manifest_sha256=str(ident["manifest_sha256"]),
        pointer_sha256=ident["pointer_sha256"],
        rows=int(ident["rows"]),
        fragments=fragments,
    )


def _legacy_files(root: Path) -> list[Path]:
    out: list[Path] = []
    players = root / "data" / "player_data"
    if players.is_dir():
        out += [p for p in players.iterdir() if p.name.endswith(("_performance_details.csv", "_personal_details.csv"))]
    matches = root / "data" / "matches"
    if matches.is_dir():
        out += [p for p in matches.iterdir() if p.name.startswith("matches_") and p.name.endswith(".csv")]
    awards = root / "data" / "awards"
    if awards.is_dir():
        out += [p for p in awards.iterdir() if p.name.endswith(".csv")]
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())


def pin_legacy(root: Path) -> S.LegacyPin:
    files = _legacy_files(root)
    if not files:
        raise PlanError(f"no legacy CSV files under {root}")
    members = hashlib.sha256()
    content = hashlib.sha256()
    total = 0
    counts = {"perf": 0, "pers": 0, "match": 0, "award": 0}
    for p in files:
        st = os.lstat(p)
        if stat.S_ISLNK(st.st_mode):
            raise PlanError(f"legacy input {p} is a symlink (refused)")
        if not stat.S_ISREG(st.st_mode):
            raise PlanError(f"legacy input {p} is not a regular file")
        rel = p.relative_to(root).as_posix()
        members.update(rel.encode() + b"\n")
        content.update(rel.encode() + b"\0" + _sha(p).encode() + b"\n")
        total += st.st_size
        if rel.endswith("_performance_details.csv"):
            counts["perf"] += 1
        elif rel.endswith("_personal_details.csv"):
            counts["pers"] += 1
        elif rel.startswith("data/awards/"):
            counts["award"] += 1
        else:
            counts["match"] += 1
    return S.LegacyPin(
        player_files=counts["perf"],
        personal_files=counts["pers"],
        match_files=counts["match"],
        award_files=counts["award"],
        bytes=total,
        membership_sha256=members.hexdigest(),
        content_sha256=content.hexdigest(),
    )


def input_roots(data_root: Path, legacy_root: Path | None) -> list[Path]:
    roots = [data_root.resolve()]
    if legacy_root is not None:
        roots += [(legacy_root / "data" / "player_data").resolve(), (legacy_root / "data" / "matches").resolve()]
    return roots


def check_run_dir(run_dir: Path, roots: list[Path]) -> None:
    resolved = run_dir.resolve()
    for root in roots:
        if resolved == root or root in resolved.parents:
            raise PlanError(f"run directory {run_dir} is inside an input root ({root}); refused")
        if resolved in root.parents:
            raise PlanError(f"run directory {run_dir} contains an input root ({root}); refused")


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def _policy_pins(config_dir: Path) -> S.PolicyPins:
    label_map = "\n".join(f"{lab}={field}" for lab, field in S.STAT_LABELS).encode()
    return S.PolicyPins(
        source_policy_sha256=_sha(config_dir / POLICY_FILE),
        rules_sha256=_sha(config_dir / RULES_FILE),
        identity_overrides_sha256=_sha(config_dir / OVERRIDES_FILE),
        label_map_sha256=hashlib.sha256(label_map).hexdigest(),
    )


def build_plan(
    *,
    data_root: Path,
    snapshot: str,
    legacy_root: Path | None,
    through_date: str,
    scope: str,
    run_dir: Path,
    sample_profiles: list[str] | None = None,
    config_dir: Path | None = None,
    seasons: list[int] | None = None,
    capture_plan: Path | None = None,
) -> S.Plan:
    try:
        date.fromisoformat(through_date)
    except ValueError as exc:
        raise PlanError(f"--through-date {through_date!r} is not a calendar date (YYYY-MM-DD)") from exc
    if scope not in ("all", "sample", "seasons"):
        raise PlanError(f"--scope must be 'all', 'sample' or 'seasons', not {scope!r}")
    season_list = sorted(set(seasons or []))
    if scope == "seasons" and not season_list:
        raise PlanError("a seasons scope needs at least one season (--season)")
    sample = sorted(set(sample_profiles or []))
    if scope == "sample" and not sample:
        raise PlanError("a sample scope needs explicit sample profiles; it is never a full population")
    check_run_dir(run_dir, input_roots(data_root, legacy_root))
    cfg = config_dir or default_config_dir()
    inputs = S.InputsSpec(
        snapshot=pin_snapshot(data_root, snapshot),
        legacy=pin_legacy(legacy_root) if legacy_root is not None else None,
    )
    body = {
        "kind": "afltables-reconciliation-plan",
        "schema_version": SCHEMA_VERSION,
        "code": code_identity().model_dump(mode="json"),
        "scope": S.ScopeSpec(
            population=scope,  # type: ignore[arg-type]
            league=S.LEAGUE,
            first_season=S.FIRST_SEASON,
            through_date=through_date,
            full_population=scope == "all",
            exclusions=list(S.EXCLUSIONS),
            sample_profiles=sample,
            seasons=season_list if scope == "seasons" else [],
        ).model_dump(mode="json"),
        "statistics": {field: label for label, field in S.STAT_LABELS},
        "inputs": inputs.model_dump(mode="json"),
        "policies": _policy_pins(cfg).model_dump(mode="json"),
        "source": S.SourceSpec(
            host="afltables.com", reference_mode="observed_current", seed_urls=list(SEED_URLS)
        ).model_dump(mode="json"),
        "outputs": {"plan": "plan.json", "capture": "capture", "reports": "reports"},
    }
    from supercoach_via.integrity.report import canonical_bytes

    if capture_plan is not None:
        _bind_capture(body, load_plan(capture_plan))
    body["capture_identity"] = hashlib.sha256(canonical_bytes(_capture_payload(body))).hexdigest()
    body["plan_id"] = hashlib.sha256(canonical_bytes(_identity_payload(body))).hexdigest()
    body["operational"] = S.Operational(
        run_dir=str(run_dir.resolve()),
        data_root=str(data_root.resolve()),
        legacy_root=str(legacy_root.resolve()) if legacy_root is not None else None,
        created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        python=platform.python_version(),
        platform=platform.platform(),
    ).model_dump(mode="json")
    return S.Plan.model_validate(body)


def _bind_capture(body: dict[str, Any], old: S.Plan) -> None:
    """Bind a new plan to the archive an earlier plan captured (DESIGN section 15, A11): the capture identity is
    computed from the capture code that PRODUCED that archive (recorded in the tamper-checked earlier plan), and the
    capture code current now is recorded beside it. Scope, source and source policy must be identical."""
    old_body = old.model_dump(mode="json")
    for key in ("scope", "source"):
        if _without_defaults(old_body[key]) != _without_defaults(body[key]):
            raise PlanError(f"--capture-plan: the archive's {key} differs from this plan's; it cannot be reused")
    if old_body["policies"]["source_policy_sha256"] != body["policies"]["source_policy_sha256"]:
        raise PlanError("--capture-plan: the archive was captured under a different source policy")
    current = body["code"]["capture_files"]
    body["code"]["capture_files"] = dict(old.code.capture_files)
    if current != old.code.capture_files:
        body["code"]["capture_files_current"] = current


#: fields appended to the plan after plans were first written, with their default values; leaving a default out of
#: both identities keeps every earlier plan (and the archive bound to it) verifiable
_APPENDED_DEFAULTS: tuple[tuple[str, str, object], ...] = (
    ("scope", "seasons", []),
    ("code", "capture_files_current", {}),
)


def _without_defaults(section: Any) -> Any:
    if not isinstance(section, dict):
        return section
    names = {f for _s, f, _d in _APPENDED_DEFAULTS}
    return {k: v for k, v in section.items() if not (k in names and v in ([], {}))}


def _strip_appended(body: dict[str, Any]) -> dict[str, Any]:
    out = dict(body)
    for sec, name, default in _APPENDED_DEFAULTS:
        if isinstance(out.get(sec), dict) and out[sec].get(name) == default:
            out[sec] = {k: v for k, v in out[sec].items() if k != name}
    legacy = (out.get("inputs") or {}).get("legacy") if isinstance(out.get("inputs"), dict) else None
    if isinstance(legacy, dict) and legacy.get("award_files") == 0:
        out["inputs"] = {**out["inputs"], "legacy": {k: v for k, v in legacy.items() if k != "award_files"}}
    return out


def _identity_payload(body: dict[str, object]) -> dict[str, object]:
    return {k: v for k, v in _strip_appended(dict(body)).items() if k not in ("plan_id", "operational")}


def _capture_payload(body: dict[str, object]) -> dict[str, object]:
    body = _strip_appended(dict(body))
    code = body["code"]
    assert isinstance(code, dict)
    return {
        "schema_version": body["schema_version"],
        "capture_files": code["capture_files"],  # the commit id is provenance, not behaviour
        "scope": body["scope"],
        "source": body["source"],
        "source_policy_sha256": body["policies"]["source_policy_sha256"],  # type: ignore[index]
    }


def load_plan(path: Path) -> S.Plan:
    """Read plan.json and verify its two identities against its own content (tamper check)."""
    from supercoach_via.integrity.report import canonical_bytes

    try:
        plan = S.Plan.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise PlanError(f"cannot read plan {path}: {exc}") from exc
    body = plan.model_dump(mode="json")
    plan_id = hashlib.sha256(canonical_bytes(_identity_payload(body))).hexdigest()
    capture_id = hashlib.sha256(canonical_bytes(_capture_payload(body))).hexdigest()
    if plan_id != plan.plan_id or capture_id != plan.capture_identity:
        raise PlanError(f"plan {path} does not match its own identity (edited after it was written)")
    return plan


def write_plan(plan: S.Plan) -> Path:
    from supercoach_via.integrity.report import canonical_bytes

    run_dir = Path(plan.operational.run_dir)
    target = run_dir / "plan.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    data = S.canonical_dump(plan)
    if os.path.lexists(target):
        st = os.lstat(target)
        if stat.S_ISLNK(st.st_mode) or st.st_nlink > 1:
            raise PlanError(f"{target} is a symlink or hard link alias; refused")
        try:
            existing = S.Plan.model_validate_json(target.read_bytes())
        except ValueError as exc:
            raise PlanError(f"{target} exists but is not a valid plan: {exc}") from exc
        if existing.plan_id == plan.plan_id:
            return target
        if existing.capture_identity != plan.capture_identity:
            raise PlanError(
                f"run directory already holds an incompatible plan {existing.plan_id[:16]} "
                f"(capture identity differs); use a new run directory"
            )
        atomic_write_bytes(run_dir / "plans" / f"plan-{existing.plan_id[:16]}.json", S.canonical_dump(existing))
    _ = canonical_bytes  # keep one canonical serialiser for every persisted file
    atomic_write_bytes(target, data)
    return target
