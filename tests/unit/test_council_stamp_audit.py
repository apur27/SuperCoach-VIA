"""Tests for the Q1 stamp-verifiability upgrade to scripts/check-council-stamp.sh.

The pre-commit stamp gate must not trust `DataSentinel: PASS` text on its own: a
stamp is only honoured if a content-hash-keyed sentinel audit record backs it.
These tests drive the real bash scripts via subprocess.
"""
import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CHECK = REPO / "scripts" / "check-council-stamp.sh"
RECORD = REPO / "scripts" / "record-sentinel-verdict.sh"
HASH = REPO / "scripts" / "council-content-hash.sh"

STAMP = (
    "<!-- council-pipeline: BriefBuilder@a, FootyStrategy@b, "
    "DataSentinel:PASS(pass1)@t1, DataSentinel:PASS(pass2)@t2, "
    "Skeptic:PASS@t3, Gaffer:SHIP@t4 -->"
)


def _make_doc(tmp_path: Path, body: str = "Prose with no stats here.") -> Path:
    doc = tmp_path / "docs" / "news" / "foo.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(f"# Test Doc\n\n{body}\n\n{STAMP}\n")
    return doc


def _run_check(tmp_path: Path, audit_dir: Path, enforce: str | None = "0"):
    env = {
        "COUNCIL_AUDIT_DIR": str(audit_dir),
        "PATH": "/usr/bin:/bin",
    }
    # enforce=None means "do not set AUDIT_ENFORCE at all" — exercises the script default.
    if enforce is not None:
        env["AUDIT_ENFORCE"] = enforce
    return subprocess.run(
        [str(CHECK), "docs/news/foo.md"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )


def _record(tmp_path: Path, audit_dir: Path, verdict: str = "PASS"):
    env = {"COUNCIL_AUDIT_DIR": str(audit_dir), "PATH": "/usr/bin:/bin"}
    r = subprocess.run(
        [str(RECORD), "--doc", "docs/news/foo.md", "--verdict", verdict],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    return r


def test_no_record_hard_fails_by_default(tmp_path):
    # AUDIT_ENFORCE now defaults to 1 (Sprint-1 flip): a stamp with no backing
    # audit record is a hard FAIL even when the env var is not set at all.
    _make_doc(tmp_path)
    audit = tmp_path / "audit"; audit.mkdir()
    r = _run_check(tmp_path, audit, enforce=None)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "cannot be verified" in r.stderr


def test_no_record_warns_when_enforce_explicitly_off(tmp_path):
    # AUDIT_ENFORCE=0 remains an explicit opt-out (warn, do not block) — used only
    # for controlled backfills of legacy docs that predate the record system.
    _make_doc(tmp_path)
    audit = tmp_path / "audit"; audit.mkdir()
    r = _run_check(tmp_path, audit, enforce="0")
    assert r.returncode == 0, r.stderr
    assert "WARNING" in r.stderr and "unverified" in r.stderr


def test_no_record_hard_fails_under_enforce(tmp_path):
    _make_doc(tmp_path)
    audit = tmp_path / "audit"; audit.mkdir()
    r = _run_check(tmp_path, audit, enforce="1")
    assert r.returncode == 1
    assert "cannot be verified" in r.stderr


def test_matching_pass_record_verifies(tmp_path):
    _make_doc(tmp_path)
    audit = tmp_path / "audit"; audit.mkdir()
    _record(tmp_path, audit, "PASS")
    r = _run_check(tmp_path, audit, enforce="1")
    assert r.returncode == 0, r.stderr
    assert "verified against audit record" in r.stdout


def test_tampered_doc_after_pass_fails(tmp_path):
    doc = _make_doc(tmp_path, body="Original prose.")
    audit = tmp_path / "audit"; audit.mkdir()
    _record(tmp_path, audit, "PASS")
    # Change the body (not the stamp) after the record was written.
    doc.write_text(doc.read_text().replace("Original prose.", "Sneaky edited prose."))
    r = _run_check(tmp_path, audit)  # even in warn mode, a stale record is a hard fail
    assert r.returncode == 1
    assert "changed after verification" in r.stderr


def test_fail_verdict_record_does_not_satisfy_pass_stamp(tmp_path):
    _make_doc(tmp_path)
    audit = tmp_path / "audit"; audit.mkdir()
    _record(tmp_path, audit, "FAIL")
    r = _run_check(tmp_path, audit)
    assert r.returncode == 1
    assert "no audit record matches" in r.stderr


def test_record_schema_has_required_fields(tmp_path):
    _make_doc(tmp_path)
    audit = tmp_path / "audit"; audit.mkdir()
    _record(tmp_path, audit, "PASS")
    recs = list(audit.glob("sentinel-*.json"))
    assert len(recs) == 1
    rec = json.loads(recs[0].read_text())
    assert set(rec) >= {"doc_path", "doc_hash", "verdict", "ts", "agent_id"}
    assert rec["verdict"] == "PASS"
    assert rec["doc_path"] == "docs/news/foo.md"
    assert len(rec["doc_hash"]) == 64


def test_missing_stamp_fails(tmp_path):
    doc = tmp_path / "docs" / "news" / "bar.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text("# No stamp here\n\nJust prose.\n")
    audit = tmp_path / "audit"; audit.mkdir()
    env = {"COUNCIL_AUDIT_DIR": str(audit), "PATH": "/usr/bin:/bin"}
    r = subprocess.run([str(CHECK), "docs/news/bar.md"], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "missing the <!-- council-pipeline" in r.stderr


def test_non_pass_verdict_fails(tmp_path):
    # A single-line stamp is required for the content-hash strip to work, so the
    # text gate greps the whole stamp line for PASS. A stamp carrying no PASS token
    # (DataSentinel FAIL + Skeptic BLOCK) must be rejected.
    doc = tmp_path / "docs" / "news" / "foo.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    bad_stamp = (
        "<!-- council-pipeline: BriefBuilder@a, FootyStrategy@b, "
        "DataSentinel:FAIL@t2, Skeptic:BLOCK@t3, Gaffer:SHIP@t4 -->"
    )
    doc.write_text(f"# Doc\n\nProse.\n\n{bad_stamp}\n")
    audit = tmp_path / "audit"; audit.mkdir()
    env = {"COUNCIL_AUDIT_DIR": str(audit), "PATH": "/usr/bin:/bin"}
    r = subprocess.run([str(CHECK), "docs/news/foo.md"], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 1
    assert "DataSentinel verdict is not PASS" in r.stderr


def test_non_council_file_is_skipped(tmp_path):
    (tmp_path / "README.md").write_text("# Readme, no stamp needed\n")
    audit = tmp_path / "audit"; audit.mkdir()
    env = {"COUNCIL_AUDIT_DIR": str(audit), "PATH": "/usr/bin:/bin"}
    r = subprocess.run([str(CHECK), "README.md"], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "0 failed" in r.stdout


def test_git_commit_safe_passes_through_to_git(tmp_path):
    wrapper = REPO / "scripts" / "git_commit_safe.sh"
    r = subprocess.run([str(wrapper), "--version"], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 0
    assert "git version" in r.stdout


def test_hash_ignores_the_stamp_line(tmp_path):
    """Canonical hash must be identical before and after the stamp is added."""
    doc = tmp_path / "d.md"
    doc.write_text("# T\n\nBody.\n")
    env = {"PATH": "/usr/bin:/bin"}
    h1 = subprocess.run([str(HASH), str(doc)], capture_output=True, text=True, env=env).stdout.strip()
    doc.write_text(f"# T\n\nBody.\n\n{STAMP}\n")
    h2 = subprocess.run([str(HASH), str(doc)], capture_output=True, text=True, env=env).stdout.strip()
    assert h1 == h2 and len(h1) == 64
