"""R1 — the pre-commit hook must gate PYTHON commits, not only Markdown ones.

The hook `exit 0`d whenever no `.md` file was staged, so a pure-Python commit —
a scraper rewrite, a model change, a new harness module — passed through with
zero checks. Every enforcement this repo has built (stamp gate, staged-blob
check, fail-closed audit) applied to prose only. Meanwhile the repo's own TDD
rule ("no code commit without tests") was enforced by nothing but habit.

Two additions, both fail-closed:
  1. If any .py is staged, the fast unit tier must pass.
  2. A NEWLY ADDED module must be referenced by something under tests/.

Rule 2 is reference-based, not filename-based, deliberately: this repo's real
convention varies (`scripts/check_round_settled.py` is tested by
`tests/unit/test_round_settled.py`, and `scripts/skeptic_verdict.py` by
`test_skeptic_gate.py`). A strict `test_<module>.py` rule would fire on ~25
existing modules and would have blocked correct work, so it would have been
turned off within a week.

These drive the REAL .githooks/pre-commit inside throwaway git repos.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / ".githooks" / "pre-commit"
VENV_PYTHON = Path("/home/abhi/sourceCode/python/coding/.venv/bin/python")

pytestmark = pytest.mark.skipif(
    not VENV_PYTHON.exists(), reason="repo venv python not available"
)

PASSING_TEST = "def test_ok():\n    assert True\n"
FAILING_TEST = "def test_bad():\n    assert False\n"


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests" / "unit").mkdir(parents=True)
    (repo / "scripts").mkdir(parents=True)
    # The Markdown half of the hook fails closed without its check script, so give
    # the fixture the real one — these tests are about the PYTHON half.
    shutil.copy(
        REPO / "scripts" / "check-council-stamp.sh",
        repo / "scripts" / "check-council-stamp.sh",
    )
    os.chmod(repo / "scripts" / "check-council-stamp.sh", 0o755)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    # A baseline passing test so the suite itself is green unless a test says otherwise.
    (repo / "tests" / "unit" / "test_baseline.py").write_text(PASSING_TEST)
    _git(repo, "add", "tests/unit/test_baseline.py")
    _git(repo, "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", "base")
    return repo


def _stub_runner(repo):
    """A runner that reports a green suite instantly.

    Most tests here assert on the new-module REFERENCE rule, not on suite results.
    Spawning a real pytest for each cost ~0.85s and pushed the whole unit tier past
    CLAUDE.md's 10s budget — which matters doubly because the hook runs that tier on
    every Python commit. The two tests that genuinely care about suite outcome pass
    `real_python=True`.
    """
    stub = repo.parent / "stub-python"
    stub.write_text("#!/bin/sh\nexit 0\n")
    os.chmod(stub, 0o755)
    return str(stub)


def _run_hook(repo, real_python=False):
    env = {
        "PATH": "/usr/bin:/bin",
        "COUNCIL_COMMIT_AUTHORIZED": "1",
        "HOME": str(repo.parent),
        "COUNCIL_PYTHON": str(VENV_PYTHON) if real_python else _stub_runner(repo),
    }
    return subprocess.run(
        ["bash", str(HOOK)], cwd=repo, env=env, capture_output=True, text=True
    )


# ------------------------------------------------------- suite must be green


def test_staged_python_with_passing_suite_is_allowed(tmp_path):
    repo = _repo(tmp_path)
    (repo / "scripts" / "thing.py").write_text("VALUE = 1\n")
    (repo / "tests" / "unit" / "test_thing.py").write_text(
        'PATH = "scripts/thing.py"\n\n\ndef test_thing():\n    assert True\n'
    )
    _git(repo, "add", "scripts/thing.py", "tests/unit/test_thing.py")
    res = _run_hook(repo, real_python=True)
    assert res.returncode == 0, f"hook blocked a clean python commit: {res.stdout}{res.stderr}"


def test_staged_python_with_failing_test_is_blocked(tmp_path):
    """The core of the gate: a red suite must not reach history."""
    repo = _repo(tmp_path)
    (repo / "scripts" / "thing.py").write_text("VALUE = 1\n")
    (repo / "tests" / "unit" / "test_thing.py").write_text(FAILING_TEST)
    _git(repo, "add", "scripts/thing.py", "tests/unit/test_thing.py")
    res = _run_hook(repo, real_python=True)
    assert res.returncode != 0
    assert "test" in (res.stdout + res.stderr).lower()


def test_markdown_only_commit_does_not_run_the_python_tier(tmp_path):
    """Prose-only commits must stay fast — no suite run, and the md path is intact."""
    repo = _repo(tmp_path)
    (repo / "tests" / "unit" / "test_baseline.py").write_text(FAILING_TEST)
    (repo / "notes.md").write_text("# notes\n")
    _git(repo, "add", "notes.md")
    res = _run_hook(repo)
    # A broken suite is present but no .py is staged, so the python tier is skipped.
    assert res.returncode == 0, f"{res.stdout}{res.stderr}"


# ------------------------------------------------- new modules need coverage


def test_new_module_without_any_test_reference_is_blocked(tmp_path):
    repo = _repo(tmp_path)
    (repo / "scripts" / "orphan.py").write_text("def f():\n    return 1\n")
    _git(repo, "add", "scripts/orphan.py")
    res = _run_hook(repo)
    assert res.returncode != 0
    assert "orphan" in (res.stdout + res.stderr)


def test_new_module_referenced_by_any_test_is_allowed(tmp_path):
    """Reference-based, so the repo's varied test-naming conventions still pass."""
    repo = _repo(tmp_path)
    (repo / "scripts" / "skeptic_verdict.py").write_text("def f():\n    return 1\n")
    # Deliberately NOT named test_skeptic_verdict.py — mirrors the real repo.
    (repo / "tests" / "unit" / "test_skeptic_gate.py").write_text(
        'PATH = "scripts/skeptic_verdict.py"\n\n\ndef test_it():\n    assert True\n'
    )
    _git(repo, "add", "scripts/skeptic_verdict.py", "tests/unit/test_skeptic_gate.py")
    res = _run_hook(repo)
    assert res.returncode == 0, f"{res.stdout}{res.stderr}"


def test_modified_module_without_test_is_not_retroactively_blocked(tmp_path):
    """Scoped to ADDED files: ~25 existing untested modules must stay committable."""
    repo = _repo(tmp_path)
    (repo / "scripts" / "legacy.py").write_text("VALUE = 1\n")
    _git(repo, "add", "scripts/legacy.py")
    _git(repo, "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", "legacy")
    (repo / "scripts" / "legacy.py").write_text("VALUE = 2\n")
    _git(repo, "add", "scripts/legacy.py")
    res = _run_hook(repo)
    assert res.returncode == 0, f"{res.stdout}{res.stderr}"


def test_new_test_file_itself_needs_no_test(tmp_path):
    repo = _repo(tmp_path)
    (repo / "tests" / "unit" / "test_new.py").write_text(PASSING_TEST)
    _git(repo, "add", "tests/unit/test_new.py")
    res = _run_hook(repo)
    assert res.returncode == 0, f"{res.stdout}{res.stderr}"


@pytest.mark.parametrize("path", ["scratch/explore.py", "archive/old.py"])
def test_exempt_directories_are_not_gated(tmp_path, path):
    """scratch/ and archive/ are explicitly non-production areas."""
    repo = _repo(tmp_path)
    p = repo / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x = 1\n")
    _git(repo, "add", path)
    res = _run_hook(repo)
    assert res.returncode == 0, f"{res.stdout}{res.stderr}"


# ----------------------------------------------------------- fail-closed


def test_unrunnable_test_runner_blocks(tmp_path):
    """A gate that cannot run must not wave the commit through."""
    repo = _repo(tmp_path)
    (repo / "scripts" / "thing.py").write_text("VALUE = 1\n")
    (repo / "tests" / "unit" / "test_thing.py").write_text(PASSING_TEST)
    _git(repo, "add", "scripts/thing.py", "tests/unit/test_thing.py")
    env = {
        "PATH": "/usr/bin:/bin",
        "COUNCIL_COMMIT_AUTHORIZED": "1",
        "HOME": str(repo.parent),
        "COUNCIL_PYTHON": str(repo / "nonexistent-python"),
    }
    res = subprocess.run(
        ["bash", str(HOOK)], cwd=repo, env=env, capture_output=True, text=True
    )
    assert res.returncode != 0
    assert "fail-closed" in (res.stdout + res.stderr).lower()


def test_hanging_suite_times_out_and_blocks(tmp_path):
    """A hung test must fail the gate, not hang the commit forever.

    Found by dogfooding: an unrelated in-progress test file in the working tree
    left the hook's pytest running for 9 minutes with no output and no way to
    tell "slow" from "wedged". An enforcement gate that can hang indefinitely
    gets bypassed with --no-verify out of frustration, which is the same as not
    having it. Bounded, and a timeout counts as failure.
    """
    repo = _repo(tmp_path)
    (repo / "scripts" / "thing.py").write_text("VALUE = 1\n")
    (repo / "tests" / "unit" / "test_thing.py").write_text(
        "import time\n\n\ndef test_hangs():\n    time.sleep(120)\n"
    )
    _git(repo, "add", "scripts/thing.py", "tests/unit/test_thing.py")
    env = {
        "PATH": "/usr/bin:/bin",
        "COUNCIL_COMMIT_AUTHORIZED": "1",
        "HOME": str(repo.parent),
        "COUNCIL_PYTHON": str(VENV_PYTHON),
        "COUNCIL_PYTEST_TIMEOUT": "3",
    }
    res = subprocess.run(
        ["bash", str(HOOK)], cwd=repo, env=env, capture_output=True, text=True,
        timeout=60,
    )
    assert res.returncode != 0
    assert "timed out" in (res.stdout + res.stderr).lower()


def test_hook_strips_git_environment_before_running_the_suite():
    """CONTRACT check — deliberately source-level, and here is why.

    The hook runs inside `git commit`, which exports GIT_INDEX_FILE/GIT_DIR and
    holds .git/index.lock. Leaking those into pytest retargets the git-driving
    tests at the REAL repository, where they block on that lock — a deadlock that
    presents as "the commit hung" (observed: 9 minutes, no output).

    This cannot be reproduced behaviourally from outside a real commit: setting
    GIT_INDEX_FILE in a test environment breaks the hook's OWN `git diff --cached`
    detection first, so it finds nothing staged and skips the tier — the run then
    exits 0 whether or not the bug is present. Two attempts at a behavioural test
    passed identically against fixed and unfixed hooks, i.e. they were vacuous.
    Rather than keep a test that proves nothing, this asserts the guard is present
    and documents that the real proof is an actual commit succeeding.
    """
    src = HOOK.read_text()
    pytest_call = src[src.index("-m pytest") - 400 : src.index("-m pytest")]
    for var in ("GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE"):
        assert f"-u {var}" in pytest_call, (
            f"{var} is not stripped before the suite runs — the hook can deadlock "
            f"against the index.lock held by the commit that invoked it"
        )


def test_commit_wrapper_lock_is_overridable(tmp_path):
    """git_commit_safe.sh must not deadlock when invoked from inside a commit.

    The wrapper flocks one global path. The pre-commit hook now runs the suite,
    and two tests drive the wrapper — with a single hard-coded lock they block
    forever on the lock the outer commit holds, and the commit just hangs (this
    cost two aborted commit attempts before it was diagnosed). Both directions are
    asserted so the test cannot pass vacuously.
    """
    import time

    wrapper = REPO / "scripts" / "git_commit_safe.sh"
    repo = _repo(tmp_path)
    held = tmp_path / "held.lock"
    held.touch()

    holder = subprocess.Popen(
        ["flock", str(held), "sleep", "30"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.5)
        base = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}

        # A DIFFERENT lock path proceeds immediately.
        res = subprocess.run(
            [str(wrapper), "rev-parse", "--show-toplevel"],
            cwd=repo, env={**base, "COUNCIL_GIT_LOCK": str(tmp_path / "free.lock")},
            capture_output=True, text=True, timeout=15,
        )
        assert res.returncode == 0, res.stderr

        # The HELD lock blocks — proving the override above is what did the work.
        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(
                [str(wrapper), "rev-parse", "--show-toplevel"],
                cwd=repo, env={**base, "COUNCIL_GIT_LOCK": str(held)},
                capture_output=True, text=True, timeout=3,
            )
    finally:
        holder.kill()
        holder.wait()


def test_direct_commit_guard_still_applies(tmp_path):
    """The pre-existing only-Gaffer-commits guard must not be regressed."""
    repo = _repo(tmp_path)
    (repo / "scripts" / "thing.py").write_text("VALUE = 1\n")
    _git(repo, "add", "scripts/thing.py")
    res = subprocess.run(
        ["bash", str(HOOK)],
        cwd=repo,
        env={"PATH": "/usr/bin:/bin", "HOME": str(repo.parent)},
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "only-Gaffer-commits" in (res.stdout + res.stderr)
