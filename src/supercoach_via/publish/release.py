"""Release staging, validation, publication and rollback.

Layout (under the output root)::

    releases/.staging-<id>/        in-progress; never listed, never publishable
    releases/<id>/public/          release.json + JSON resources + downloads/
    releases/<id>/site/            optional built static site (web/ build output)
    releases/<id>/checksums.json   every file under public/ (and site/) with sha256/bytes
    releases/<id>/validation.json  written by validate_release
    receipts/<ts>-<id>-<status>.json

A release becomes visible only by an atomic rename of its complete staging directory.
Publication consumes only a release whose validation.json is PASS and whose bytes still
match checksums.json; a failed publication leaves the previously active release in place
and writes a failed receipt. Nothing here invokes Git or the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from supercoach_via.domain.schemas import CheckOutcome, ValidationReport, is_safe_id
from supercoach_via.publish.view_models import (
    PUBLIC_MODELS,
    CoverageInfo,
    DownloadItem,
    ForecastInfo,
    PublicModel,
    ReleaseManifest,
    ResourceRef,
)
from supercoach_via.publish.web_data import canonical_json_bytes, sha256_bytes
from supercoach_via.storage.snapshots import atomic_write_bytes, sha256_file

Clock = Callable[[], datetime]

_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-.]{0,159}$")
FORBIDDEN_NAMES = re.compile(
    r"(^|/)(\.env[^/]*|\.claude|\.git|\.github|node_modules|runs|raw|__pycache__)(/|$)"
    r"|\.(pkl|pickle|joblib|npy|npz|parquet|duckdb|db|sqlite|py|sh|log|jsonl)$",
    re.I,
)
DOWNLOAD_EXTENSIONS = {".csv", ".png", ".svg", ".zip", ".json", ".md"}
PRIVATE_MARKERS = (b"/home/", b"/tmp/", b"/Users/", b".claude/audit", b"ANTHROPIC_API_KEY", b"BEGIN PRIVATE KEY")

#: (compiled path pattern, public model key) — first match wins.
_PATH_MODELS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^release\.json$"), "release"),
    (re.compile(r"^overview\.json$"), "overview"),
    (re.compile(r"^predictions/index\.json$"), "prediction_index"),
    (re.compile(r"^predictions/\d{4}/[^/]+\.json$"), "prediction_set"),
    (re.compile(r"^players/index\.json$"), "player_index"),
    (re.compile(r"^players/[^/]+\.json$"), "player_detail"),
    (re.compile(r"^player-games/[^/]+/\d{4}\.json$"), "player_season_games"),
    (re.compile(r"^teams/index\.json$"), "team_index"),
    (re.compile(r"^teams/[^/]+/\d{4}\.json$"), "team_season"),
    (re.compile(r"^matches/\d{4}/index\.json$"), "match_index"),
    (re.compile(r"^matches/detail/[^/]+\.json$"), "match_detail"),
    (re.compile(r"^history/index\.json$"), "history_index"),
    (re.compile(r"^history/[^/]+/[^/]+\.json$"), "history_table"),
    (re.compile(r"^accuracy/index\.json$"), "accuracy_index"),
    (re.compile(r"^accuracy/[^/]+/[^/]+\.json$"), "accuracy_report"),
    (re.compile(r"^lists/index\.json$"), "lists_index"),
    (re.compile(r"^lists/\d{4}\.json$"), "lists_season"),
    (re.compile(r"^articles/index\.json$"), "article_index"),
    (re.compile(r"^articles/[^/]+\.json$"), "article"),
    (re.compile(r"^live/index\.json$"), "live_index"),
    (re.compile(r"^live/[^/]+/latest\.json$"), "live_snapshot"),
    (re.compile(r"^quality\.json$"), "quality"),
    (re.compile(r"^downloads\.json$"), "downloads"),
]


class PublishError(RuntimeError):
    """Publication refused or failed (exit code 7)."""


def check_release_path(relpath: str) -> str:
    parts = relpath.split("/")
    if not relpath or relpath.startswith("/") or "\\" in relpath or any(not _SEGMENT.match(p) for p in parts):
        raise ValueError(f"unsafe release path {relpath!r}")
    if FORBIDDEN_NAMES.search(relpath):
        raise ValueError(f"forbidden release path {relpath!r}")
    return relpath


def model_for_path(relpath: str) -> str | None:
    for pattern, key in _PATH_MODELS:
        if pattern.match(relpath):
            return key
    return None


def _strict_json(data: bytes) -> Any:
    def bad_constant(name: str) -> Any:
        raise ValueError(f"non-finite JSON constant {name}")

    return json.loads(data.decode("utf-8"), parse_constant=bad_constant)


class ReleaseWriter:
    """Stages a complete release, then atomically renames it into place."""

    def __init__(self, output_root: Path, release_id: str):
        if not is_safe_id(release_id) or ":" in release_id:
            raise ValueError(f"unsafe release id {release_id!r}")
        self.output_root = output_root
        self.release_id = release_id
        self.staging = output_root / "releases" / f".staging-{release_id}"
        if self.staging.exists():
            shutil.rmtree(self.staging)
        (self.staging / "public").mkdir(parents=True)
        self.files: dict[str, dict[str, Any]] = {}

    @property
    def public(self) -> Path:
        return self.staging / "public"

    def put_bytes(self, relpath: str, data: bytes) -> ResourceRef:
        check_release_path(relpath)
        if relpath in self.files:
            raise ValueError(f"duplicate release path {relpath}")
        if relpath.startswith("downloads/") and Path(relpath).suffix not in DOWNLOAD_EXTENSIONS:
            raise ValueError(f"download type not allowed: {relpath}")
        target = self.public / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        digest = sha256_bytes(data)
        self.files[relpath] = {"sha256": digest, "bytes": len(data)}
        return ResourceRef(path=relpath, sha256=digest, bytes=len(data))

    def put_json(self, relpath: str, model: BaseModel) -> ResourceRef:
        key = model_for_path(relpath)
        if key is None or not isinstance(model, PUBLIC_MODELS[key]):
            raise ValueError(f"{relpath} must hold a {key} model, got {type(model).__name__}")
        return self.put_bytes(relpath, canonical_json_bytes(model))

    def download_item(self, key: str, label: str, relpath: str, *, as_of: str, rows: int | None = None) -> DownloadItem:
        info = self.files[relpath]
        kind = Path(relpath).suffix.lstrip(".")
        return DownloadItem(
            key=key,
            label=label,
            kind=kind,
            path=relpath,
            bytes=info["bytes"],
            sha256=info["sha256"],
            as_of=as_of,
            rows=rows,
        )

    def finish(
        self,
        *,
        snapshot_id: str,
        generated_at: datetime,
        season: int,
        demo: bool,
        coverage_status: Literal["legacy_unverified", "verified", "partial", "demo"],
        coverage_through: str | None,
        forecast_status: Literal["available", "unavailable", "expired"],
        forecast_reason: str | None,
        forecast_artifact: str | None,
        index_resources: dict[str, str],
        forecast_model_id: str | None = None,
    ) -> Path:
        resources = {}
        for key, relpath in sorted(index_resources.items()):
            info = self.files.get(relpath)
            if info is None:
                raise ValueError(f"index resource {key} -> {relpath} was not written")
            resources[key] = ResourceRef(path=relpath, sha256=info["sha256"], bytes=info["bytes"])
        manifest = ReleaseManifest(
            release_id=self.release_id,
            snapshot_id=snapshot_id,
            generated_at=generated_at,
            season=season,
            demo=demo,
            base_label="DEMO" if demo else "",
            coverage=CoverageInfo(status=coverage_status, through=coverage_through),
            forecast=ForecastInfo(
                status=forecast_status, reason=forecast_reason, artifact=forecast_artifact, model_id=forecast_model_id
            ),
            resources=resources,
        )
        self.put_bytes("release.json", canonical_json_bytes(manifest))
        checksums = {"release_id": self.release_id, "files": dict(sorted(self.files.items()))}
        (self.staging / "checksums.json").write_bytes(canonical_json_bytes(checksums))
        final = self.output_root / "releases" / self.release_id
        if final.exists():
            raise FileExistsError(f"release {self.release_id} already exists; releases are immutable")
        os.replace(self.staging, final)
        return final


def list_releases(output_root: Path) -> list[str]:
    base = output_root / "releases"
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))


def _iter_files(root: Path) -> Iterable[str]:
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            yield f"SYMLINK:{p.relative_to(root).as_posix()}"
        elif p.is_file():
            yield p.relative_to(root).as_posix()


def validate_release(release_dir: Path, *, write: bool = True) -> ValidationReport:
    """Closure, hashes, allowlist, schema and reference checks over a finished release."""
    issues: list[dict[str, Any]] = []
    checks: dict[str, CheckOutcome] = {}
    public = release_dir / "public"

    def fail(check: str, path: str, why: str) -> None:
        checks[check] = CheckOutcome.FAIL
        issues.append({"check": check, "path": path, "why": why[:300]})

    try:
        sums = _strict_json((release_dir / "checksums.json").read_bytes())["files"]
    except (OSError, ValueError, KeyError) as exc:
        fail("checksums", "checksums.json", str(exc))
        sums = {}

    present = set(_iter_files(public)) if public.is_dir() else set()
    for name in sorted(present - set(sums)):
        fail("closure", name, "file not listed in checksums.json")
    for name in sorted(set(sums) - present):
        fail("closure", name, "listed file is missing")

    listed = set(sums)
    manifest: Any = None  # only the release manifest is kept; other documents are checked then dropped
    for name in sorted(present & set(sums)):
        path = public / name
        try:
            check_release_path(name)
        except ValueError as exc:
            fail("allowlist", name, str(exc))
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != sums[name]["sha256"] or len(data) != sums[name]["bytes"]:
            fail("hashes", name, "content does not match checksums.json")
        if any(marker in data for marker in PRIVATE_MARKERS):
            fail("private_content", name, "contains a local path or secret marker")
        if name.endswith(".json"):
            key = model_for_path(name)
            if name.startswith("downloads/"):
                key = None
            try:
                doc = _strict_json(data)
            except ValueError as exc:
                fail("json", name, str(exc))
                continue
            if key is None:
                if not name.startswith("downloads/"):
                    fail("schema", name, "no public model for this path")
                continue
            try:
                model = PUBLIC_MODELS[key].model_validate(doc)
            except ValidationError as exc:
                fail("schema", name, str(exc))
                continue
            if name == "release.json":
                manifest = model
            _check_references({name: model}, listed, fail)

    if not isinstance(manifest, ReleaseManifest):
        fail("manifest", "release.json", "missing or invalid release manifest")
    else:
        if manifest.release_id != release_dir.name:
            fail("manifest", "release.json", "release_id does not match directory")
        for key, ref in manifest.resources.items():
            info = sums.get(ref.path)
            if info is None or info["sha256"] != ref.sha256 or info["bytes"] != ref.bytes:
                fail("manifest", ref.path, f"resource {key} hash/size mismatch")

    for c in (
        "checksums",
        "closure",
        "allowlist",
        "hashes",
        "private_content",
        "json",
        "schema",
        "manifest",
        "references",
    ):
        checks.setdefault(c, CheckOutcome.PASS)
    outcome = CheckOutcome.FAIL if issues else CheckOutcome.PASS
    report = ValidationReport(outcome=outcome, checks=checks, issues=issues, counts={"files": len(present)})
    if write:
        payload = {
            "release_id": release_dir.name,
            "outcome": outcome.value,
            "checks": {k: v.value for k, v in sorted(checks.items())},
            "issues": issues[:500],
            "checksums_sha256": sha256_file(release_dir / "checksums.json")
            if (release_dir / "checksums.json").exists()
            else None,
        }
        atomic_write_bytes(release_dir / "validation.json", canonical_json_bytes(payload))
    return report


def _check_references(parsed: dict[str, Any], files: set[str], fail: Callable[[str, str, str], None]) -> None:
    """Every index-referenced resource must exist in the release."""
    from supercoach_via.publish import view_models as vm

    def need(path: str | None, source: str) -> None:
        if path is not None and path not in files:
            fail("references", source, f"references missing resource {path}")

    for name, doc in parsed.items():
        if isinstance(doc, vm.PredictionIndex):
            for s in doc.sets:
                need(s.resource, name)
        elif isinstance(doc, vm.PlayerIndex):
            for p in doc.players:
                need(f"players/{p.key}.json", name)
        elif isinstance(doc, vm.PlayerDetail):
            for season in doc.seasons:
                need(season.games_resource, name)
        elif isinstance(doc, vm.HistoryIndex):
            for t in doc.tables:
                for path in t.resources.values():
                    need(path, name)
        elif isinstance(doc, vm.AccuracyIndex):
            for r in doc.reports:
                need(r.resource, name)
        elif isinstance(doc, vm.ListsIndex):
            for path in doc.resources.values():
                need(path, name)
        elif isinstance(doc, vm.ArticleIndex):
            for a in doc.articles:
                need(a.resource, name)
        elif isinstance(doc, vm.LiveIndex):
            for m in doc.matches:
                need(m.resource, name)
        elif isinstance(doc, vm.Downloads):
            for item in doc.items:
                need(item.path, name)
        elif isinstance(doc, vm.TeamIndex):
            for team in doc.teams:
                for yr in team.seasons:
                    need(f"teams/{team.club_id}/{yr}.json", name)


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------


class PublishReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    release_id: str
    destination: str
    kind: Literal["publish", "rollback"]
    status: Literal["published", "failed"]
    at: datetime
    validation_checksums_sha256: str | None
    previous_release: str | None
    message: str | None = None


class LocalDirectoryDestination:
    """Publishes into ``root/releases/<id>`` and atomically swaps the ``root/live`` symlink.

    Stands in for a static host in rehearsals; a hosted destination (e.g. GitHub Pages
    artifact upload) implements the same two hooks and is only ever run from the
    dedicated publish workflow.
    """

    def __init__(self, name: str, root: Path):
        if not is_safe_id(name):
            raise ValueError("unsafe destination name")
        self.name = name
        self.root = root

    def active_release(self) -> str | None:
        live = self.root / "live"
        return Path(os.readlink(live)).name if live.is_symlink() else None

    def upload(self, release_dir: Path) -> None:
        src = release_dir / "site" if (release_dir / "site").is_dir() else release_dir / "public"
        target = self.root / "releases" / release_dir.name
        if target.exists():
            return  # immutable: already uploaded
        tmp = self.root / "releases" / f".upload-{release_dir.name}"
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp, symlinks=False)
        os.replace(tmp, target)

    def _activate(self, release_id: str) -> None:
        link = self.root / f".live-{release_id}"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(Path("releases") / release_id, target_is_directory=True)
        os.replace(link, self.root / "live")

    def activate(self, release_id: str) -> None:
        if not (self.root / "releases" / release_id).is_dir():
            raise PublishError(f"release {release_id} was never uploaded to {self.name}")
        self._activate(release_id)


def _write_receipt(output_root: Path, receipt: PublishReceipt) -> Path:
    stamp = receipt.at.strftime("%Y%m%dT%H%M%S.%fZ")
    path = output_root / "receipts" / f"{stamp}-{receipt.release_id}-{receipt.kind}-{receipt.status}.json"
    atomic_write_bytes(path, canonical_json_bytes(receipt))
    return path


def _require_validated(release_dir: Path) -> str:
    vpath = release_dir / "validation.json"
    if not vpath.is_file():
        raise PublishError("release has not been validated (run validate-release)")
    record = _strict_json(vpath.read_bytes())
    if record.get("outcome") != "PASS":
        raise PublishError(f"release validation outcome is {record.get('outcome')}")
    if record.get("checksums_sha256") != sha256_file(release_dir / "checksums.json"):
        raise PublishError("checksums changed since validation")
    drift = integrity_issues(release_dir)
    if drift:
        raise PublishError(f"release files differ from the validated release: {drift[:3]}")
    return str(record["checksums_sha256"])


def integrity_issues(release_dir: Path) -> list[str]:
    """Files that differ from ``checksums.json`` (missing, extra or changed bytes).

    Publication and rollback check integrity only: semantic validation happened at
    validate-release time under the contract the release was built with, so a later
    schema change must not strand an older validated release.
    """
    sums = _strict_json((release_dir / "checksums.json").read_bytes())["files"]
    public = release_dir / "public"
    present = set(_iter_files(public)) if public.is_dir() else set()
    issues = [f"extra {n}" for n in sorted(present - set(sums))]
    issues += [f"missing {n}" for n in sorted(set(sums) - present)]
    for name in sorted(present & set(sums)):
        data = (public / name).read_bytes()
        if len(data) != sums[name]["bytes"] or sha256_bytes(data) != sums[name]["sha256"]:
            issues.append(f"changed {name}")
    return issues


def publish_release(
    release_dir: Path,
    destination: LocalDirectoryDestination,
    *,
    clock: Clock,
    kind: Literal["publish", "rollback"] = "publish",
) -> PublishReceipt:
    output_root = release_dir.parent.parent
    checks = _require_validated(release_dir)
    previous = destination.active_release()
    try:
        destination.upload(release_dir)
        destination.activate(release_dir.name)
    except (OSError, PublishError) as exc:
        failed = PublishReceipt(
            release_id=release_dir.name,
            destination=destination.name,
            kind=kind,
            status="failed",
            at=clock(),
            validation_checksums_sha256=checks,
            previous_release=previous,
            message=str(exc)[:500],
        )
        _write_receipt(output_root, failed)
        raise PublishError(f"publication failed; {previous or 'no release'} remains active: {exc}") from exc
    receipt = PublishReceipt(
        release_id=release_dir.name,
        destination=destination.name,
        kind=kind,
        status="published",
        at=clock(),
        validation_checksums_sha256=checks,
        previous_release=previous,
    )
    _write_receipt(output_root, receipt)
    return receipt


def rollback(
    output_root: Path, destination: LocalDirectoryDestination, release_id: str, *, clock: Clock
) -> PublishReceipt:
    """Re-activate a previously validated release; produces a new receipt."""
    release_dir = output_root / "releases" / release_id
    if not release_dir.is_dir():
        raise PublishError(f"unknown release {release_id}")
    return publish_release(release_dir, destination, clock=clock, kind="rollback")


def load_public_model(path: Path, key: str) -> PublicModel:
    return PUBLIC_MODELS[key].model_validate(_strict_json(path.read_bytes()))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
