"""F4 — check-council-stamp.sh must verify the STAGED blob, not the working tree.

Stage-good-then-edit-bad (or the reverse) must not let unverified bytes reach
history. These drive the real script against a real git index.
"""
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CHECK = REPO / "scripts" / "check-council-stamp.sh"
RECORD = REPO / "scripts" / "record-sentinel-verdict.sh"

STAMP = (
    "<!-- council-pipeline: DataSentinel:PASS(pass2)@t, "
    "Skeptic:PASS@t, Gaffer:SHIP@t -->"
)
ENV0 = {"PATH": "/usr/bin:/bin"}


def _git(repo, *args, env):
    return subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True)


def _init(tmp_path):
    repo = tmp_path / "repo"
    (repo / "docs" / "news").mkdir(parents=True)
    env = {**ENV0, "HOME": str(tmp_path)}
    _git(repo, "init", "-q", env=env)
    _git(repo, "config", "user.email", "t@t", env=env)
    _git(repo, "config", "user.name", "t", env=env)
    return repo, env


def _run_check(repo, audit, env, path="docs/news/foo.md"):
    e = {**env, "COUNCIL_AUDIT_DIR": str(audit)}
    return subprocess.run([str(CHECK), path], cwd=repo, env=e, capture_output=True, text=True)


def test_gate_reads_staged_blob_not_working_tree(tmp_path):
    repo, env = _init(tmp_path)
    audit = tmp_path / "audit"; audit.mkdir()
    doc = repo / "docs" / "news" / "foo.md"
    doc.write_text(f"# Good\n\nOriginal verified prose.\n\n{STAMP}\n")
    _git(repo, "add", "docs/news/foo.md", env=env)
    # Record a PASS for the staged (== disk) content.
    subprocess.run([str(RECORD), "--doc", "docs/news/foo.md", "--verdict", "PASS"],
                   cwd=repo, env={**env, "COUNCIL_AUDIT_DIR": str(audit)},
                   capture_output=True, text=True, check=True)
    # Now dirty the WORKING TREE only (index still holds the recorded content).
    doc.write_text(f"# Good\n\nSNEAKY UNVERIFIED EDIT.\n\n{STAMP}\n")
    r = _run_check(repo, audit, env)
    # Gate hashes the staged blob (still the recorded content) -> PASS. If it read
    # the working tree it would find no record for the edited bytes and FAIL.
    assert r.returncode == 0, r.stdout + r.stderr
    assert "verified against audit record" in r.stdout


def test_gate_fails_when_staged_bytes_have_no_record(tmp_path):
    repo, env = _init(tmp_path)
    audit = tmp_path / "audit"; audit.mkdir()
    doc = repo / "docs" / "news" / "foo.md"
    doc.write_text(f"# Good\n\nRecorded prose.\n\n{STAMP}\n")
    _git(repo, "add", "docs/news/foo.md", env=env)
    subprocess.run([str(RECORD), "--doc", "docs/news/foo.md", "--verdict", "PASS"],
                   cwd=repo, env={**env, "COUNCIL_AUDIT_DIR": str(audit)},
                   capture_output=True, text=True, check=True)
    # Stage DIFFERENT bytes (no record for these); leave the recorded bytes only in
    # the audit log, not the index.
    doc.write_text(f"# Good\n\nUNRECORDED staged bytes.\n\n{STAMP}\n")
    _git(repo, "add", "docs/news/foo.md", env=env)
    r = _run_check(repo, audit, env)
    assert r.returncode == 1
    assert "no audit record matches" in r.stderr


def test_symlink_in_index_fails_closed(tmp_path):
    repo, env = _init(tmp_path)
    audit = tmp_path / "audit"; audit.mkdir()
    (repo / "target.md").write_text("real\n")
    os.symlink("target.md", repo / "docs" / "news" / "foo.md")
    _git(repo, "add", "docs/news/foo.md", env=env)
    r = _run_check(repo, audit, env)
    assert r.returncode == 1
    assert "symlink" in r.stderr
