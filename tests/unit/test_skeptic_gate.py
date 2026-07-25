"""scripts/skeptic_verdict.py — interpret a Skeptic verdict, fail closed.

docs/afl-insights.md previously skipped Skeptic entirely (the "weekly recap
exemption"): it was gated by DataSentinel alone, which checks whether numbers are
TRUE but not whether the surrounding prose overstates them. The user has since
required the adversarial pass, so the harness now needs a deterministic reading
of Skeptic's three-way verdict.

Skeptic's vocabulary (canonical, shared with record-sentinel-verdict.sh):
  PASS                 -> ship
  PASS_WITH_CONCERNS   -> ship, log the concerns
  BLOCK                -> halt

The load-bearing case is the fourth one: no verdict at all. An agent that crashed,
timed out, or emitted prose instead of JSON must HALT the cycle, never sail
through. A gate that cannot run is not a gate that passed.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "skeptic_verdict.py"


def _run(tmp_path, content):
    p = tmp_path / "skeptic_out.json"
    p.write_text(content)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(p)], capture_output=True, text=True
    )


# ------------------------------------------------------------------ clearing


@pytest.mark.parametrize("verdict", ["PASS", "PASS_WITH_CONCERNS"])
def test_clearing_verdicts_exit_zero(tmp_path, verdict):
    res = _run(tmp_path, '{"verdict": "%s", "doc": "docs/afl-insights.md"}' % verdict)
    assert res.returncode == 0, res.stderr
    assert verdict in res.stdout


def test_pass_with_concerns_surfaces_the_concerns(tmp_path):
    res = _run(
        tmp_path,
        '{"verdict": "PASS_WITH_CONCERNS", "concerns": ["ladder framing overstated"]}',
    )
    assert res.returncode == 0
    assert "ladder framing overstated" in res.stdout


def test_verdict_embedded_in_surrounding_prose_is_found(tmp_path):
    """Agents often wrap JSON in commentary; the verdict must still be read."""
    res = _run(
        tmp_path,
        'Here is my review.\n{"verdict": "PASS", "notes": "fine"}\nDone.',
    )
    assert res.returncode == 0


# -------------------------------------------------------------------- halting


def test_block_exits_nonzero(tmp_path):
    res = _run(tmp_path, '{"verdict": "BLOCK", "reason": "unsupported causal claim"}')
    assert res.returncode != 0
    assert "BLOCK" in (res.stdout + res.stderr)
    assert "unsupported causal claim" in (res.stdout + res.stderr)


def test_missing_verdict_fails_closed(tmp_path):
    res = _run(tmp_path, '{"notes": "I could not complete the review"}')
    assert res.returncode != 0


def test_empty_output_fails_closed(tmp_path):
    """The agent crashed and wrote nothing — must halt, not proceed."""
    res = _run(tmp_path, "")
    assert res.returncode != 0


def test_prose_only_output_fails_closed(tmp_path):
    res = _run(tmp_path, "The document looks good to me overall.\n")
    assert res.returncode != 0


def test_missing_file_fails_closed(tmp_path):
    res = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "nope.json")],
        capture_output=True, text=True,
    )
    assert res.returncode != 0


def test_unknown_token_fails_closed(tmp_path):
    """A mistyped/invented verdict is not a pass."""
    res = _run(tmp_path, '{"verdict": "LOOKS_FINE"}')
    assert res.returncode != 0


def test_lowercase_pass_is_not_accepted(tmp_path):
    """Exact-token match, consistent with record-sentinel-verdict.sh (F07)."""
    res = _run(tmp_path, '{"verdict": "pass"}')
    assert res.returncode != 0


def test_block_wins_when_both_tokens_appear(tmp_path):
    """A BLOCK anywhere in the output must dominate an incidental 'PASS'."""
    res = _run(
        tmp_path,
        '{"verdict": "BLOCK", "reason": "the PASS criteria were not met"}',
    )
    assert res.returncode != 0
