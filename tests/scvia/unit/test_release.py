"""Release staging, validation, publication and rollback (R05, R08-R10, S04)."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from supercoach_via.domain.schemas import CheckOutcome
from supercoach_via.publish import release as rel
from supercoach_via.publish.view_models import (
    ArticleIndex,
    Downloads,
    QualityReport,
)

NOW = datetime(2026, 9, 23, 11, 0, tzinfo=UTC)
SNAP = "sha256:" + "a" * 64


def _write_minimal(out: Path, release_id: str = "20260923T110000Z-demo") -> Path:
    w = rel.ReleaseWriter(out, release_id)
    w.put_json("articles/index.json", ArticleIndex(articles=[]))
    w.put_json(
        "quality.json",
        QualityReport(
            dataset_status="demo", table_counts={}, quarantined_rows=0, issues=[], limitations=["demo"], sources=[]
        ),
    )
    w.put_bytes("downloads/players.csv", b"id,name\n1,Demo\n")
    items = [w.download_item("players_csv", "Players (all rows)", "downloads/players.csv", as_of="demo", rows=1)]
    w.put_json("downloads.json", Downloads(release_id=release_id, items=items, retained=[]))
    return w.finish(
        snapshot_id=SNAP,
        generated_at=NOW,
        season=2026,
        demo=True,
        coverage_status="demo",
        coverage_through=None,
        forecast_status="unavailable",
        forecast_reason="no_valid_future_fixture",
        forecast_artifact=None,
        index_resources={
            "article_index": "articles/index.json",
            "quality": "quality.json",
            "downloads": "downloads.json",
        },
    )


def test_release_validates_when_complete(tmp_path: Path) -> None:
    rdir = _write_minimal(tmp_path)
    report = rel.validate_release(rdir)
    assert report.outcome is CheckOutcome.PASS, report.issues
    manifest = json.loads((rdir / "public" / "release.json").read_text())
    assert set(manifest["resources"]) == {"article_index", "quality", "downloads"}
    assert (rdir / "validation.json").exists()


def test_staging_is_invisible_until_finished(tmp_path: Path) -> None:
    w = rel.ReleaseWriter(tmp_path, "r-partial")
    w.put_bytes("downloads/x.csv", b"a\n")
    assert not (tmp_path / "releases" / "r-partial").exists()
    assert rel.list_releases(tmp_path) == []


@pytest.mark.parametrize(
    "mutation",
    ["tamper", "extra", "missing", "nan", "schema", "forbidden", "private_path"],
)
def test_validation_rejects_bad_release(tmp_path: Path, mutation: str) -> None:
    rdir = _write_minimal(tmp_path)
    pub = rdir / "public"
    if mutation == "tamper":
        (pub / "downloads" / "players.csv").write_bytes(b"id,name\n1,Evil\n")
    elif mutation == "extra":
        (pub / "stray.json").write_text("{}")
    elif mutation == "missing":
        (pub / "quality.json").unlink()
    elif mutation == "nan":
        _rewrite_with_checksum(
            rdir,
            "quality.json",
            b'{"dataset_status":"demo","issues":[],"limitations":[],"quarantined_rows":NaN,"sources":[],"table_counts":{}}\n',
        )
    elif mutation == "schema":
        _rewrite_with_checksum(rdir, "quality.json", b'{"dataset_status":"bogus"}\n')
    elif mutation == "forbidden":
        _rewrite_with_checksum(rdir, ".env", b"TOKEN=x\n")
    elif mutation == "private_path":
        _rewrite_with_checksum(rdir, "downloads/players.csv", b"id,name\n1,/home/abhi/secret\n")
    assert rel.validate_release(rdir).outcome is CheckOutcome.FAIL


def _rewrite_with_checksum(rdir: Path, relpath: str, data: bytes) -> None:
    """Simulate an attacker/bug that also updates checksums (so only content rules catch it)."""
    import hashlib

    target = rdir / "public" / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    sums = json.loads((rdir / "checksums.json").read_text())
    sums["files"][relpath] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    (rdir / "checksums.json").write_text(json.dumps(sums))


@pytest.mark.parametrize("hide", ["remove", "rename"])
def test_missing_sealed_site_does_not_fall_back_to_public(tmp_path: Path, hide: str) -> None:
    out = tmp_path / "dist"
    rdir = _write_minimal(out, "r-seal-gone")
    site = rdir / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>sealed</h1>")
    _embed_public(rdir)
    rel.write_seal(rdir)
    assert rel.validate_release(rdir).ok
    record = json.loads((rdir / "validation.json").read_text())
    assert record["seal_sha256"]
    if hide == "remove":
        shutil.rmtree(site)
    else:
        site.rename(rdir / "site-renamed")
    dest = rel.LocalDirectoryDestination("local", tmp_path / "host")
    with pytest.raises(rel.PublishError, match="sealed site is missing"):
        rel.publish_release(rdir, dest, clock=lambda: NOW)
    assert dest.active_release() is None
    assert not (tmp_path / "host" / "live").exists()
    receipts = list((out / "receipts").glob("*.json")) if (out / "receipts").exists() else []
    assert not any(json.loads(p.read_text())["status"] == "published" for p in receipts)


def test_site_private_file_fails_validation_and_is_not_published(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    rdir = _write_minimal(out, "r-dotenv")
    site = rdir / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>ok</h1>")
    (site / ".env").write_text("TOKEN=synthetic\n")
    rel.write_seal(rdir)
    report = rel.validate_release(rdir)
    assert report.outcome is CheckOutcome.FAIL
    assert any(i["check"] == "private_content" and ".env" in i["why"] for i in report.issues)
    dest = rel.LocalDirectoryDestination("local", tmp_path / "host")
    with pytest.raises(rel.PublishError):
        rel.publish_release(rdir, dest, clock=lambda: NOW)
    assert dest.active_release() is None
    assert not (tmp_path / "host" / "live" / ".env").exists()


def _embed_public(rdir: Path) -> None:
    shutil.copytree(rdir / "public", rdir / "site" / "data" / rdir.name)


def _seal_public_copy(rdir: Path) -> None:
    site = rdir / "site"
    shutil.copytree(rdir / "public", site / "data" / rdir.name)
    (site / "index.html").write_text("<h1>synthetic</h1>")
    rel.write_seal(rdir)


def test_embedded_site_data_must_match_the_validated_public_inventory(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    ok = _write_minimal(out, "r-copy")
    _seal_public_copy(ok)
    assert rel.validate_release(ok).ok

    changed = _write_minimal(out, "r-changed")
    _seal_public_copy(changed)
    (changed / "site" / "data" / changed.name / "quality.json").write_text('{"not_a_quality_report":true}')
    rel.write_seal(changed)
    report = rel.validate_release(changed)
    assert report.outcome is CheckOutcome.FAIL
    assert any(i["check"] == "embedded_data" for i in report.issues)
    assert json.loads((changed / "validation.json").read_text())["outcome"] == "FAIL"

    missing = _write_minimal(out, "r-missing")
    _seal_public_copy(missing)
    (missing / "site" / "data" / missing.name / "quality.json").unlink()
    rel.write_seal(missing)
    assert rel.validate_release(missing).outcome is CheckOutcome.FAIL

    extra = _write_minimal(out, "r-extra")
    _seal_public_copy(extra)
    (extra / "site" / "data" / extra.name / "stray.json").write_text("{}")
    rel.write_seal(extra)
    assert rel.validate_release(extra).outcome is CheckOutcome.FAIL

    foreign = _write_minimal(out, "r-foreign")
    _seal_public_copy(foreign)
    shutil.copytree(foreign / "public", foreign / "site" / "data" / "other-release")
    rel.write_seal(foreign)
    foreign_report = rel.validate_release(foreign)
    assert foreign_report.outcome is CheckOutcome.FAIL
    assert any("other-release" in i["why"] for i in foreign_report.issues)

    retained = _write_minimal(out, "r-retained")
    _seal_public_copy(retained)
    shutil.copytree(retained / "public", retained / "site" / "data" / "older")
    inv = rel._inventory_from_record(json.loads((retained / "checksums.json").read_text())["files"])
    retained_dir = retained / "retained"
    retained_dir.mkdir()
    (retained_dir / "older.json").write_text(
        json.dumps({"files": {name: {"sha256": digest, "bytes": size} for name, digest, size in inv}})
    )
    rel.write_seal(retained)
    retained_report = rel.validate_release(retained)
    assert retained_report.outcome is CheckOutcome.FAIL
    assert any("older" in i["why"] for i in retained_report.issues)

    removed = _write_minimal(out, "r-removed-data")
    _seal_public_copy(removed)
    shutil.rmtree(removed / "site" / "data")
    rel.write_seal(removed)
    removed_report = rel.validate_release(removed)
    assert removed_report.outcome is CheckOutcome.FAIL
    assert any("site/data" in i["why"] for i in removed_report.issues)


def test_sealed_embedded_data_rolls_back_without_recertifying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "dist"
    rdir = _write_minimal(out, "r-old-contract")
    _seal_public_copy(rdir)
    seal = rel.write_seal(rdir)
    assert rel.validate_release(rdir).ok

    def newer_contract(*_a: object, **_k: object) -> rel.ValidationReport:
        raise AssertionError("rollback must not re-run semantic validation")

    monkeypatch.setattr(rel, "validate_release", newer_contract)
    dest = rel.LocalDirectoryDestination("local", tmp_path / "host")
    receipt = rel.rollback(out, dest, "r-old-contract", clock=lambda: NOW)
    assert receipt.status == "published" and receipt.seal_sha256 == seal
    assert (dest.root / "live" / "data" / "r-old-contract" / "quality.json").is_file()


def test_previously_validated_html_only_seal_still_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "dist"
    rdir = _write_minimal(out, "r-html-old")
    site = rdir / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>historical</h1>")
    seal = rel.write_seal(rdir)
    assert rel.validate_release(rdir).outcome is CheckOutcome.FAIL

    monkeypatch.setattr(rel, "embedded_data_problems", lambda *_a, **_k: [])
    assert rel.validate_release(rdir).ok
    monkeypatch.undo()
    assert rel.validate_release(rdir, write=False).outcome is CheckOutcome.FAIL

    def newer_contract(*_a: object, **_k: object) -> rel.ValidationReport:
        raise AssertionError("rollback must not re-run semantic validation")

    monkeypatch.setattr(rel, "validate_release", newer_contract)
    dest = rel.LocalDirectoryDestination("local", tmp_path / "host")
    receipt = rel.rollback(out, dest, "r-html-old", clock=lambda: NOW)
    assert receipt.status == "published" and receipt.seal_sha256 == seal
    assert (dest.root / "live" / "index.html").read_text() == "<h1>historical</h1>"
    assert not (dest.root / "live" / "data").exists()


def test_schema_home_pointer_is_not_a_filesystem_leak(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    rdir = _write_minimal(out, "r-pointer")
    site = rdir / "site"
    (site / "_astro").mkdir(parents=True)
    (site / "index.html").write_text("<h1>ok</h1>")
    (site / "_astro" / "score.js").write_text("instancePath:t+`/home/behinds`")
    _embed_public(rdir)
    rel.write_seal(rdir)
    assert rel.validate_release(rdir).ok


def test_home_directory_path_in_the_site_fails_validation(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    rdir = _write_minimal(out, "r-home")
    site = rdir / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>ok</h1><p>/home/abhi/secret</p>")
    rel.write_seal(rdir)
    report = rel.validate_release(rdir)
    assert report.outcome is CheckOutcome.FAIL
    assert any(i["check"] == "private_content" for i in report.issues)


def test_unsealed_site_cannot_validate_or_publish(tmp_path: Path) -> None:
    # Permanent regression for R01 in the historical reproduce_open_findings.py
    # diagnostic; that diagnostic records the old failure, not a passing test.
    out = tmp_path / "dist"
    rdir = _write_minimal(out)
    assert rel.validate_release(rdir).ok
    site = rdir / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>Unchecked synthetic replacement</h1>")
    (site / "private.json").write_text('{"synthetic_private_marker":true}')
    assert rel.validate_release(rdir, write=False).outcome is CheckOutcome.FAIL
    dest = rel.LocalDirectoryDestination("local", tmp_path / "host")
    with pytest.raises(rel.PublishError, match="rebuild and seal"):
        rel.publish_release(rdir, dest, clock=lambda: NOW)
    assert dest.active_release() is None
    assert not (tmp_path / "host" / "live" / "private.json").exists()


def test_sealed_site_is_the_only_upload_and_tamper_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    rdir = _write_minimal(out, "r-seal")
    site = rdir / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>ok</h1>")
    _embed_public(rdir)
    seal = rel.write_seal(rdir, build_inputs={"toolchain": "test"})
    assert rel.validate_release(rdir).ok
    record = json.loads((rdir / "validation.json").read_text())
    assert record["seal_sha256"] == seal
    assert not (site / "seal.json").exists()
    dest = rel.LocalDirectoryDestination("local", tmp_path / "host")
    receipt = rel.publish_release(rdir, dest, clock=lambda: NOW)
    assert receipt.status == "published" and receipt.seal_sha256 == seal
    live = tmp_path / "host" / "live"
    assert (live / "index.html").read_text() == "<h1>ok</h1>"
    assert not (live / "quality.json").exists()
    (site / "index.html").write_text("<h1>tampered</h1>")
    with pytest.raises(rel.PublishError, match="sealed"):
        rel.publish_release(rdir, dest, clock=lambda: NOW)
    assert (live / "index.html").read_text() == "<h1>ok</h1>"


def test_existing_destination_must_match_and_partial_upload_is_not_success(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    rdir = _write_minimal(out, "r1")
    site = rdir / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>ok</h1>")
    _embed_public(rdir)
    rel.write_seal(rdir)
    assert rel.validate_release(rdir).ok
    host = tmp_path / "host"
    partial = host / "releases" / ".upload-r1"
    partial.mkdir(parents=True)
    (partial / "junk.html").write_text("partial")
    dest = rel.LocalDirectoryDestination("local", host)
    rel.publish_release(rdir, dest, clock=lambda: NOW)
    assert (host / "live" / "index.html").read_text() == "<h1>ok</h1>"
    assert not (host / "releases" / ".upload-r1").exists()
    again = rel.publish_release(rdir, dest, clock=lambda: NOW)
    assert again.status == "published"
    (host / "releases" / "r1" / "index.html").write_text("<h1>altered</h1>")
    with pytest.raises(rel.PublishError, match="does not match"):
        rel.publish_release(rdir, dest, clock=lambda: NOW)
    assert dest.active_release() == "r1"


@pytest.mark.parametrize("kind", ["publish", "rollback"])
@pytest.mark.parametrize("mutation", ["html", "dotenv"])
def test_bytes_changed_after_validation_are_not_activated(tmp_path: Path, kind: str, mutation: str) -> None:
    out = tmp_path / "dist"
    previous = _write_minimal(out, "r-prev")
    assert rel.validate_release(previous).ok
    rdir = _write_minimal(out, "r-race")
    site = rdir / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>validated</h1>")
    _embed_public(rdir)
    rel.write_seal(rdir)
    assert rel.validate_release(rdir).ok
    host = tmp_path / "host"
    rel.publish_release(previous, rel.LocalDirectoryDestination("local", host), clock=lambda: NOW)

    class Mutating(rel.LocalDirectoryDestination):
        def upload(self, release_dir: Path) -> None:
            if release_dir.name == "r-race":
                if mutation == "html":
                    (release_dir / "site" / "index.html").write_text("<h1>modified after validation check</h1>")
                else:
                    (release_dir / "site" / ".env").write_text("SYNTHETIC_PRIVATE=1\n")
            super().upload(release_dir)

    with pytest.raises(rel.PublishError, match="validated inventory"):
        if kind == "publish":
            rel.publish_release(rdir, Mutating("local", host), clock=lambda: NOW)
        else:
            rel.rollback(out, Mutating("local", host), "r-race", clock=lambda: NOW)
    assert rel.LocalDirectoryDestination("local", host).active_release() == "r-prev"
    assert (host / "live" / "quality.json").is_file()
    assert not (host / "releases" / "r-race").exists()
    assert not any(p.name == ".env" for p in (host / "live").rglob("*"))
    assert not any("modified after" in p.read_text() for p in (host / "live").rglob("*.html"))


def test_existing_destination_that_matches_a_changed_source_stays_inactive(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    previous = _write_minimal(out, "r-prev")
    assert rel.validate_release(previous).ok
    rdir = _write_minimal(out, "r-race")
    site = rdir / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>validated</h1>")
    _embed_public(rdir)
    rel.write_seal(rdir)
    assert rel.validate_release(rdir).ok
    host = tmp_path / "host"
    planted = host / "releases" / "r-race"
    planted.mkdir(parents=True)
    (planted / "index.html").write_text("<h1>modified after validation check</h1>")
    rel.publish_release(previous, rel.LocalDirectoryDestination("local", host), clock=lambda: NOW)

    class Mutating(rel.LocalDirectoryDestination):
        def upload(self, release_dir: Path) -> None:
            if release_dir.name == "r-race":
                (release_dir / "site" / "index.html").write_text("<h1>modified after validation check</h1>")
            super().upload(release_dir)

    with pytest.raises(rel.PublishError, match="validated inventory"):
        rel.publish_release(rdir, Mutating("local", host), clock=lambda: NOW)
    assert rel.LocalDirectoryDestination("local", host).active_release() == "r-prev"
    assert (host / "live" / "quality.json").read_bytes()


def test_public_bytes_changed_during_upload_are_not_activated(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    previous = _write_minimal(out, "r-prev")
    rdir = _write_minimal(out, "r-public")
    assert rel.validate_release(previous).ok
    assert rel.validate_release(rdir).ok
    host = tmp_path / "host"
    rel.publish_release(previous, rel.LocalDirectoryDestination("local", host), clock=lambda: NOW)

    class Mutating(rel.LocalDirectoryDestination):
        def upload(self, release_dir: Path) -> None:
            if release_dir.name == "r-public":
                (release_dir / "public" / "quality.json").write_bytes(b'{"tampered":true}\n')
            super().upload(release_dir)

    with pytest.raises(rel.PublishError, match="validated inventory"):
        rel.publish_release(rdir, Mutating("local", host), clock=lambda: NOW)
    assert rel.LocalDirectoryDestination("local", host).active_release() == "r-prev"
    assert b"tampered" not in (host / "live" / "quality.json").read_bytes()


@pytest.mark.parametrize("kind", ["publish", "rollback"])
@pytest.mark.parametrize("mutation", ["html", "dotenv"])
def test_reseal_between_validation_and_capture_is_not_activated(
    tmp_path: Path, kind: str, mutation: str
) -> None:
    out = tmp_path / "dist"
    previous = _write_minimal(out, "previous")
    assert rel.validate_release(previous).ok
    host = tmp_path / "host"
    rel.publish_release(previous, rel.LocalDirectoryDestination("local", host), clock=lambda: NOW)
    release = _write_minimal(out, "candidate")
    site = release / "site"
    site.mkdir()
    shutil.copytree(release / "public", site / "data" / release.name)
    (site / "index.html").write_text("<h1>validated</h1>")
    rel.write_seal(release)
    assert rel.validate_release(release).ok
    validated_seal = json.loads((release / "validation.json").read_text())["seal_sha256"]

    class Resealing(rel.LocalDirectoryDestination):
        def active_release(self) -> str | None:
            if mutation == "html":
                (site / "index.html").write_text("<h1>replaced after validation</h1>")
            else:
                (site / ".env").write_text("SYNTHETIC_REVIEW_MARKER=unvalidated\n")
            rel.write_seal(release)
            return super().active_release()

    dest = Resealing("local", host)
    with pytest.raises(rel.PublishError, match="validated inventory"):
        if kind == "publish":
            rel.publish_release(release, dest, clock=lambda: NOW)
        else:
            rel.rollback(out, dest, "candidate", clock=lambda: NOW)
    assert rel.LocalDirectoryDestination("local", host).active_release() == "previous"
    assert not (host / "live" / ".env").exists()
    assert not any("replaced after validation" in p.read_text() for p in (host / "live").rglob("*.html"))
    failed = [json.loads(p.read_text()) for p in (out / "receipts").glob("*.json")]
    failed = [r for r in failed if r["release_id"] == "candidate" and r["status"] == "failed"]
    assert failed and failed[-1]["seal_sha256"] == validated_seal
    assert failed[-1]["kind"] == kind


def test_public_checksums_replaced_before_upload_keep_the_validated_digest(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    previous = _write_minimal(out, "previous")
    release = _write_minimal(out, "public-candidate")
    assert rel.validate_release(previous).ok
    assert rel.validate_release(release).ok
    original = json.loads((release / "validation.json").read_text())["checksums_sha256"]
    host = tmp_path / "host"
    rel.publish_release(previous, rel.LocalDirectoryDestination("local", host), clock=lambda: NOW)

    class Rewriting(rel.LocalDirectoryDestination):
        def active_release(self) -> str | None:
            (release / "checksums.json").write_text('{"files":{}}')
            (release / "public" / "quality.json").write_text('{"tampered":true}')
            return super().active_release()

    with pytest.raises(rel.PublishError, match="validated inventory"):
        rel.publish_release(release, Rewriting("local", host), clock=lambda: NOW)
    assert rel.LocalDirectoryDestination("local", host).active_release() == "previous"
    assert b"tampered" not in (host / "live" / "quality.json").read_bytes()
    failed = [
        json.loads(p.read_text())
        for p in (out / "receipts").glob("*public-candidate*failed.json")
    ]
    assert failed and failed[-1]["validation_checksums_sha256"] == original
    assert failed[-1]["seal_sha256"] is None


def test_publish_refuses_unvalidated_release(tmp_path: Path) -> None:
    rdir = _write_minimal(tmp_path)
    dest = rel.LocalDirectoryDestination("local", tmp_path / "site")
    with pytest.raises(rel.PublishError):
        rel.publish_release(rdir, dest, clock=lambda: NOW)


def test_publish_then_failed_publish_keeps_previous_and_rollback(tmp_path: Path) -> None:
    # The full scratch smoke exercises inject_upload_failure.py against sealed
    # releases. This unit regression covers activation failure and rollback too.
    out = tmp_path / "dist"
    r1 = _write_minimal(out, "r1")
    r2 = _write_minimal(out, "r2")
    for r in (r1, r2):
        assert rel.validate_release(r).ok
    dest = rel.LocalDirectoryDestination("local", tmp_path / "site")
    receipt1 = rel.publish_release(r1, dest, clock=lambda: NOW)
    assert receipt1.status == "published" and dest.active_release() == "r1"

    class Boom(rel.LocalDirectoryDestination):
        def _activate(self, release_id: str) -> None:
            raise OSError("host rejected upload")

    bad = Boom("local", tmp_path / "site")
    with pytest.raises(rel.PublishError):
        rel.publish_release(r2, bad, clock=lambda: NOW)
    assert dest.active_release() == "r1"
    receipts = sorted((out / "receipts").glob("*.json"))
    assert any(json.loads(p.read_text())["status"] == "failed" for p in receipts)

    rel.publish_release(r2, dest, clock=lambda: NOW)
    assert dest.active_release() == "r2"
    back = rel.rollback(out, dest, "r1", clock=lambda: NOW)
    assert back.status == "published" and back.kind == "rollback" and dest.active_release() == "r1"


@pytest.mark.parametrize("kind", ["publish", "rollback"])
@pytest.mark.parametrize("seal_field", ["missing", "null"])
@pytest.mark.parametrize("payload", ["html", "dotenv"])
def test_public_only_record_cannot_authorize_a_site(tmp_path: Path, kind: str, seal_field: str, payload: str) -> None:
    # A public-only validation inventory does not cover site/, whether the seal key is
    # absent (pre-seal records) or explicit null. Do not publish that tree or fall back to public/.
    out = tmp_path / "dist"
    rdir = _write_minimal(out, "r-old")
    assert rel.validate_release(rdir).ok
    record = json.loads((rdir / "validation.json").read_text())
    if seal_field == "missing":
        record.pop("seal_sha256", None)
        record.pop("checker", None)
    else:
        record["seal_sha256"] = None
    (rdir / "validation.json").write_text(json.dumps(record))
    site = rdir / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>unchecked</h1>")
    if payload == "dotenv":
        (site / ".env").write_text("TOKEN=synthetic\n")
    assert rel.integrity_issues(rdir) == []
    dest = rel.LocalDirectoryDestination("local", tmp_path / "host")
    with pytest.raises(rel.PublishError, match="rebuild and seal"):
        if kind == "publish":
            rel.publish_release(rdir, dest, clock=lambda: NOW)
        else:
            rel.rollback(out, dest, "r-old", clock=lambda: NOW)
    assert dest.active_release() is None
    assert not (tmp_path / "host" / "live").exists()


def test_public_only_record_without_a_seal_key_still_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "dist"
    rdir = _write_minimal(out, "r-public")
    assert rel.validate_release(rdir).ok
    record = json.loads((rdir / "validation.json").read_text())
    record.pop("seal_sha256", None)
    record.pop("checker", None)
    (rdir / "validation.json").write_text(json.dumps(record))

    def newer_contract(*_a: object, **_k: object) -> rel.ValidationReport:
        raise AssertionError("publication must not re-run semantic validation")

    monkeypatch.setattr(rel, "validate_release", newer_contract)
    dest = rel.LocalDirectoryDestination("local", tmp_path / "host")
    receipt = rel.rollback(out, dest, "r-public", clock=lambda: NOW)
    assert receipt.status == "published" and receipt.kind == "rollback"
    assert (dest.root / "live" / "quality.json").is_file()
    assert not (rdir / "site").exists()


def test_sealed_site_rolls_back_from_its_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "dist"
    rdir = _write_minimal(out, "r-sealed")
    site = rdir / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>sealed</h1>")
    _embed_public(rdir)
    seal = rel.write_seal(rdir)
    assert rel.validate_release(rdir).ok

    def newer_contract(*_a: object, **_k: object) -> rel.ValidationReport:
        raise AssertionError("publication must not re-run semantic validation")

    monkeypatch.setattr(rel, "validate_release", newer_contract)
    dest = rel.LocalDirectoryDestination("local", tmp_path / "host")
    receipt = rel.rollback(out, dest, "r-sealed", clock=lambda: NOW)
    assert receipt.status == "published" and receipt.seal_sha256 == seal
    assert (dest.root / "live" / "index.html").read_text() == "<h1>sealed</h1>"


def test_rollback_survives_a_later_contract_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A release validated under an older public contract must stay publishable/rollback-able:
    # publication checks that the bytes are exactly the validated ones, not today's schemas.
    out = tmp_path / "dist"
    r1, r2 = _write_minimal(out, "r1"), _write_minimal(out, "r2")
    for r in (r1, r2):
        assert rel.validate_release(r).ok
    dest = rel.LocalDirectoryDestination("local", tmp_path / "site")
    rel.publish_release(r1, dest, clock=lambda: NOW)
    rel.publish_release(r2, dest, clock=lambda: NOW)

    def newer_contract(*_a: object, **_k: object) -> rel.ValidationReport:
        raise AssertionError("publication must not re-run semantic validation")

    monkeypatch.setattr(rel, "validate_release", newer_contract)
    back = rel.rollback(out, dest, "r1", clock=lambda: NOW)
    assert back.status == "published" and dest.active_release() == "r1"


@pytest.mark.parametrize("tamper", ["modify", "add", "remove"])
def test_publish_refuses_bytes_that_differ_from_the_validated_release(tmp_path: Path, tamper: str) -> None:
    rdir = _write_minimal(tmp_path / "dist")
    assert rel.validate_release(rdir).ok
    public = rdir / "public"
    victim = next(p for p in sorted(public.rglob("*.json")))
    if tamper == "modify":
        victim.write_bytes(victim.read_bytes().replace(b"}", b" }", 1))
    elif tamper == "add":
        (public / "extra.json").write_text("{}")
    else:
        victim.unlink()
    dest = rel.LocalDirectoryDestination("local", tmp_path / "site")
    with pytest.raises(rel.PublishError, match="differ from the validated release"):
        rel.publish_release(rdir, dest, clock=lambda: NOW)
    assert dest.active_release() is None


def test_publish_never_runs_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    calls: list[object] = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: calls.append(a))
    r = _write_minimal(tmp_path / "dist", "r1")
    rel.validate_release(r)
    rel.publish_release(r, rel.LocalDirectoryDestination("local", tmp_path / "site"), clock=lambda: NOW)
    assert calls == []


@pytest.mark.parametrize("bad", ["../x", "/abs", "a/../../b", "x/.claude/y", "models/m.pkl"])
def test_writer_rejects_unsafe_paths(tmp_path: Path, bad: str) -> None:
    w = rel.ReleaseWriter(tmp_path, "r")
    with pytest.raises(ValueError):
        w.put_bytes(bad, b"x")


@pytest.mark.parametrize("bad", ["", "../r", "r/1", "a b"])
def test_release_id_is_validated(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        rel.ReleaseWriter(tmp_path, bad)


def test_index_referencing_a_missing_resource_fails_references(tmp_path: Path) -> None:
    rdir = _write_minimal(tmp_path)
    (rdir / "public" / "downloads" / "players.csv").unlink()
    sums = json.loads((rdir / "checksums.json").read_text())
    del sums["files"]["downloads/players.csv"]  # closure stays consistent; only the reference is dangling
    (rdir / "checksums.json").write_text(json.dumps(sums))
    report = rel.validate_release(rdir, write=False)
    assert report.checks["references"] is CheckOutcome.FAIL
    assert any(i["check"] == "references" and i["path"] == "downloads.json" for i in report.issues)
    assert report.checks["closure"] is CheckOutcome.PASS


def test_parallel_validation_reports_exactly_what_sequential_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rdir = _write_minimal(tmp_path / "dist")
    public = rdir / "public"
    (public / "quality.json").write_text('{"not": "a quality report"}')  # schema + hash
    (public / "downloads" / "players.csv").write_bytes(b"id\n/home/someone/x\n")  # hash + private marker
    monkeypatch.setattr(rel, "PARALLEL_MIN_FILES", 1)
    reports = {}
    for workers in (1, 2):
        monkeypatch.setattr(rel, "VALIDATE_WORKERS", workers)
        reports[workers] = rel.validate_release(rdir, write=False)
    assert reports[1].outcome is CheckOutcome.FAIL
    assert {i["check"] for i in reports[1].issues} >= {"schema", "hashes", "private_content"}
    assert reports[1] == reports[2]


def test_duplicate_json_keys_are_refused_at_every_release_boundary(tmp_path: Path) -> None:
    """O55-05: json.loads keeps the last duplicate silently; release metadata must not be ambiguous."""
    with pytest.raises(ValueError, match="duplicate"):
        rel._strict_json(b'{"outcome": "FAIL", "outcome": "PASS"}')
    rdir = _write_minimal(tmp_path)
    target = rdir / "public" / "release.json"
    raw = target.read_bytes()
    target.write_bytes(raw[:1] + b'"demo_extra": 1, "demo_extra": 2, ' + raw[1:])
    report = rel.validate_release(rdir, write=False)
    assert not report.ok
    assert any(i["check"] == "json" and "duplicate" in i["why"] for i in report.issues)


def test_public_booleans_are_not_coerced_from_numbers() -> None:
    """O55-05: pydantic's lax mode would turn 1 into true; the published contract says boolean."""
    from pydantic import ValidationError

    from supercoach_via.publish.view_models import PlayerIndexEntry

    ok = {"id": "legacy:x", "key": "k.x", "name": "X", "clubs": [], "first_season": None, "last_season": None,
          "seasons": [], "games": 0, "active": True, "search": "x"}  # fmt: skip
    PlayerIndexEntry.model_validate(ok)
    with pytest.raises(ValidationError):
        PlayerIndexEntry.model_validate({**ok, "active": 1})
    with pytest.raises(ValidationError):
        PlayerIndexEntry.model_validate({**ok, "active": "true"})


def test_tree_inventory_classifies_each_entry_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use one fresh no-follow type check, instead of several stat calls per path."""
    import os

    root = tmp_path / "tree"
    for branch in range(4):
        folder = root / str(branch)
        folder.mkdir(parents=True)
        for i in range(32):
            (folder / f"{i}.txt").write_bytes(b"release bytes")
    calls = 0
    real_stat, real_lstat = os.stat, os.lstat

    def count_stat(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real_stat(*args, **kwargs)

    def count_lstat(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return real_lstat(*args, **kwargs)

    monkeypatch.setattr(os, "stat", count_stat)
    monkeypatch.setattr(os, "lstat", count_lstat)
    inventory, problems = rel._walk_tree(root)
    assert problems == [] and len(inventory) == 128
    assert all(info == {"sha256": rel.sha256_bytes(b"release bytes"), "bytes": 13} for info in inventory.values())
    # One check for each of the 128files, 4directories and the root; allow one extra.
    assert calls <= 134, f"tree inventory made {calls} redundant path stats"


def test_tree_inventory_refuses_links_and_special_files(tmp_path: Path) -> None:
    import os

    root = tmp_path / "tree"
    root.mkdir()
    external = tmp_path / "outside"
    external.mkdir()
    (external / "private.txt").write_text("private")
    (root / "linked-directory").symlink_to(external, target_is_directory=True)
    (root / "linked-file").symlink_to(external / "private.txt")
    (root / "broken-link").symlink_to(tmp_path / "missing")
    os.mkfifo(root / "pipe")
    (root / "regular.txt").write_text("public")
    inventory, problems = rel._walk_tree(root)
    assert set(inventory) == {"regular.txt"}
    assert problems == ["symlink refused: broken-link", "symlink refused: linked-directory",
                        "symlink refused: linked-file", "non-regular file refused: pipe"]
    assert set(rel._iter_files(root)) == {"regular.txt", "SYMLINK:broken-link",
                                          "SYMLINK:linked-directory", "SYMLINK:linked-file"}
    assert rel._site_content_problems(root) == problems


@pytest.mark.parametrize("mutation", ["symlink", "missing"])
def test_tree_inventory_refuses_entry_changed_during_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    first, second = root / "a.txt", root / "b.txt"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    external = tmp_path / "private.txt"
    external.write_bytes(b"private")
    read_bytes = Path.read_bytes

    def replace_next(path: Path) -> bytes:
        data = read_bytes(path)
        if path == first:
            second.unlink()
            if mutation == "symlink":
                second.symlink_to(external)
        return data

    monkeypatch.setattr(Path, "read_bytes", replace_next)
    inventory, problems = rel._walk_tree(root)
    assert set(inventory) == {"a.txt"}
    label = "symlink" if mutation == "symlink" else "non-regular file"
    assert problems == [f"{label} refused: b.txt"]
