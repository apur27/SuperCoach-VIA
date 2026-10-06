"""Gaffer M5: the hook's unit tier ran serially (~173 s on 2,112 tests, against CLAUDE.md's ~20 s budget and inside
the 300 s hook timeout). On pytest-xdist with 4 workers the same suite takes ~66 s with identical results (12 cores,
2026-10-06; -n auto was slower, 74 s). The hook runs it on COUNCIL_PYTEST_WORKERS workers (default 4) when xdist is
importable and falls back to a serial run otherwise, so a missing plugin never blocks a commit."""

from pathlib import Path

HOOK = Path(__file__).resolve().parents[2] / ".githooks" / "pre-commit"


def test_the_unit_tier_runs_on_xdist_workers_with_a_serial_fallback():
    src = HOOK.read_text()
    assert 'COUNCIL_PYTEST_WORKERS:-4' in src
    assert "import xdist" in src, "no check that the plugin is available"
    assert '"$PYTHON" -m pytest "$REPO_ROOT/tests" -q -m "not integration" "${xdist_args[@]}"' in src
