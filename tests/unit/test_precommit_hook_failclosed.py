"""F14a — the pre-commit hook must fail CLOSED, not open.

An enforcement hook that `exit 0`s when its own check script is missing silently
waves every commit through. This test drives the real `.githooks/pre-commit` in a
throwaway git repo and asserts that a missing check script BLOCKS the commit.
(The corrupt-audit-record case is already covered by test_council_stamp_audit.py's
test_tampered_doc_after_pass_fails / test_no_record_hard_fails.)
"""
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / ".githooks" / "pre-commit"


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _repo_with_staged_md(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    (repo / "docs").mkdir()
    doc = repo / "docs" / "foo.md"
    doc.write_text("# doc\n")
    _git(repo, "add", "docs/foo.md")
    return repo


def _run_hook(repo: Path, extra_env: dict) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "COUNCIL_COMMIT_AUTHORIZED": "1", **extra_env}
    return subprocess.run(
        ["bash", str(HOOK)], cwd=repo, env=env, capture_output=True, text=True
    )


def test_missing_check_script_blocks_commit(tmp_path):
    """No scripts/check-council-stamp.sh in the repo -> hook must exit non-zero."""
    repo = _repo_with_staged_md(tmp_path)
    # repo has no scripts/check-council-stamp.sh at all.
    r = _run_hook(repo, {})
    assert r.returncode != 0, "hook waved the commit through with no check script (fail-open!)"
    assert "fail-closed" in r.stderr.lower() or "blocking" in r.stderr.lower()


def test_unauthorized_direct_commit_blocks(tmp_path):
    """The only-Gaffer-commits guard still blocks a commit with no authorization marker."""
    repo = _repo_with_staged_md(tmp_path)
    env = {"PATH": "/usr/bin:/bin"}  # COUNCIL_COMMIT_AUTHORIZED unset
    r = subprocess.run(
        ["bash", str(HOOK)], cwd=repo, env=env, capture_output=True, text=True
    )
    assert r.returncode != 0
    assert "direct commits are disabled" in r.stderr
