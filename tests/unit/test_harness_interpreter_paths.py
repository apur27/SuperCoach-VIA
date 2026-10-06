"""The harness and its hooks find their interpreter and agent CLI portably (no absolute machine paths).

2026-10: the hard-coded ``/home/abhi/sourceCode/python/coding/.venv/bin/python`` and ``/home/abhi/.claude/local/claude``
vanished after an OS upgrade, so the weekly cycle could not start and no harness change could be smoke-run. The
interpreter is now the repository's own ``.venv`` (``uv sync --locked --group dev --group legacy --extra ml``),
overridable with ``SUPERCOACH_PYTHON``; the agent CLI is ``$CLAUDE`` or the ``claude`` on PATH.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RESOLVER = REPO / "scripts" / "harness_env.sh"
SCRIPTS = [
    "scripts/weekly_refresh.sh",
    "refresh_and_rank.sh",
    "scripts/update_eval_surface.sh",
    "scripts/record-sentinel-verdict.sh",
    ".githooks/pre-commit",
]


@pytest.mark.parametrize("rel", SCRIPTS)
def test_no_harness_script_hard_codes_a_machine_interpreter_or_agent_path(rel: str) -> None:
    text = (REPO / rel).read_text(encoding="utf-8")
    assert "/home/abhi/sourceCode/python" not in text, rel
    assert "/home/abhi/.claude/local/claude" not in text, rel


def test_the_shared_resolver_prefers_the_override_then_the_repo_venv(tmp_path: Path) -> None:
    lib = RESOLVER
    assert lib.is_file()
    fake = tmp_path / "repo"
    (fake / ".venv" / "bin").mkdir(parents=True)
    py = fake / ".venv" / "bin" / "python"
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)
    def resolve(env: dict[str, str]) -> str:
        cmd = f'REPO_ROOT="{fake}"; . "{lib}"; harness_python'
        return subprocess.run(["bash", "-c", cmd], env={"PATH": "/usr/bin:/bin", **env}, capture_output=True,
                              text=True, check=True).stdout.strip()  # fmt: skip
    assert resolve({}) == str(py)
    other = tmp_path / "other-python"
    other.write_text("#!/bin/sh\n")
    other.chmod(0o755)
    assert resolve({"SUPERCOACH_PYTHON": str(other)}) == str(other)


def test_a_missing_interpreter_fails_with_the_rebuild_command(tmp_path: Path) -> None:
    lib = RESOLVER
    cmd = f'REPO_ROOT="{tmp_path}"; . "{lib}"; harness_python'
    res = subprocess.run(["bash", "-c", cmd], env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    assert res.returncode != 0 and "uv sync --locked --group dev --group legacy --extra ml" in res.stderr


def test_the_sourced_resolver_is_not_git_ignored() -> None:
    """Every harness script sources this file; an ignored path is silently left out of commits and smoke copies,
    and a fresh clone's cycle then dies on its first line (it first sat under a dir a generic `lib/` rule ignores)."""
    res = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", str(RESOLVER)], capture_output=True)
    assert res.returncode == 1, "scripts/harness_env.sh is git-ignored"
