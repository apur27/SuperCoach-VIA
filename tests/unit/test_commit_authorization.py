"""Tests for the only-Gaffer-commits structural guard in .githooks/pre-commit.

The rule (harness Q6 / only-Gaffer-commits): automated commits MUST be serialised
through scripts/git_commit_safe.sh, which authorises the commit by exporting a
marker. A raw `git commit` (e.g. an agent committing directly, racing the index)
is blocked by the pre-commit hook. The human escape hatch is `git commit --no-verify`.

These drive real git against the real hook in a throwaway repo.
"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOOKS = REPO / ".githooks"
SAFE = REPO / "scripts" / "git_commit_safe.sh"

ENV = {"PATH": "/usr/bin:/bin"}


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**ENV, "HOME": str(tmp_path)}
    subprocess.run(["git", "init", "-q"], cwd=repo, env=env, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, env=env, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, env=env, check=True)
    subprocess.run(["git", "config", "core.hooksPath", str(HOOKS)], cwd=repo, env=env, check=True)
    (repo / "note.txt").write_text("hello\n")  # non-md: stamp check is irrelevant
    subprocess.run(["git", "add", "note.txt"], cwd=repo, env=env, check=True)
    return repo


def _commit(repo: Path, tmp_path: Path, *, args, extra_env=None):
    env = {**ENV, "HOME": str(tmp_path)}
    if extra_env:
        env.update(extra_env)
    return subprocess.run(args, cwd=repo, env=env, capture_output=True, text=True)


def test_raw_commit_is_blocked(tmp_path):
    repo = _init_repo(tmp_path)
    r = _commit(repo, tmp_path, args=["git", "commit", "-m", "raw"])
    assert r.returncode != 0
    assert "git_commit_safe.sh" in r.stderr


def test_git_commit_safe_wrapper_is_allowed(tmp_path):
    repo = _init_repo(tmp_path)
    r = _commit(repo, tmp_path, args=[str(SAFE), "commit", "-m", "via wrapper"])
    assert r.returncode == 0, r.stdout + r.stderr


def test_authorized_marker_env_allows_commit(tmp_path):
    repo = _init_repo(tmp_path)
    r = _commit(repo, tmp_path, args=["git", "commit", "-m", "marked"],
                extra_env={"COUNCIL_COMMIT_AUTHORIZED": "1"})
    assert r.returncode == 0, r.stdout + r.stderr


def test_no_verify_is_the_human_escape_hatch(tmp_path):
    repo = _init_repo(tmp_path)
    r = _commit(repo, tmp_path, args=["git", "commit", "--no-verify", "-m", "human"])
    assert r.returncode == 0, r.stdout + r.stderr
