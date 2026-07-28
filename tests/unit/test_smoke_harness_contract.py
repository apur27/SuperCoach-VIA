"""scripts/smoke_harness.sh must never be able to write history or reach origin.

CLAUDE.md 6.2 requires a harness change to run end to end before it merges. A smoke
runner that could commit or push would be strictly worse than no runner at all — it
would turn a validation step into an unreviewed release path.
"""
import os
import stat
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "scripts" / "smoke_harness.sh"


def test_smoke_runner_exists_and_is_executable():
    assert SMOKE.exists(), "no smoke runner; CLAUDE.md 6.2 cannot be satisfied"
    assert os.stat(SMOKE).st_mode & stat.S_IXUSR, "smoke runner is not executable"


def test_smoke_runner_parses():
    r = subprocess.run(["bash", "-n", str(SMOKE)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_commit_and_push_are_suppressed():
    """The shim must intercept BOTH, or a smoke run could mutate the real repo."""
    src = SMOKE.read_text()
    assert "commit) echo" in src, "git commit is not suppressed"
    assert "push)   echo" in src or "push) echo" in src, "git push is not suppressed"


def test_runs_in_a_worktree_not_the_repo():
    src = SMOKE.read_text()
    assert "worktree add" in src, "smoke run does not isolate itself in a worktree"


def test_limitations_are_documented():
    """A green smoke run must not be over-trusted; the header states what it does
    NOT cover (network, LLM gates, its own un-smoke-testability)."""
    src = SMOKE.read_text()
    assert "LIMITATIONS" in src.upper()
    assert "SMOKE_SKIP_SCRAPE" in src
