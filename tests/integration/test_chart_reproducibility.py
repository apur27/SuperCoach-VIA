"""Committed chart PNGs must reproduce byte-identically from the current code.

BL-04. `assets/charts/top10_alltime_hall.png` regenerated deterministically in this
environment but differed from the committed blob (174,837 vs 175,862 bytes) — the
committed version having been produced under a different matplotlib/font build. The
underlying data was identical, so nothing published was WRONG; the cost was
operational. Every agent that ran a doc regeneration dirtied `assets/` as a side
effect and had to remember to revert it, and a real chart change was indistinguishable
from rendering noise in a diff.

This asserts the committed artifact matches what the code produces now. It renders into
a tmp directory rather than `assets/`, so the check itself has no side effect — the
thing it is policing.

When a matplotlib/font upgrade lands this test SHOULD fail: that is the signal to
refresh the committed charts deliberately, rather than discovering the drift as
unexplained churn in someone else's diff.
"""
import hashlib
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

pytestmark = pytest.mark.integration

COMMITTED = os.path.join(_REPO_ROOT, "assets", "charts", "top10_alltime_hall.png")


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def test_top100_chart_reproduces_byte_identically(tmp_path, monkeypatch):
    """Canary for rendering drift, on the chart BL-04 named.

    Chosen because its inputs are the all-time top-10, which change rarely — so a
    byte difference here is almost always the renderer, not the data.
    """
    import update_team_analysis as uta

    assert os.path.exists(COMMITTED), f"committed chart missing: {COMMITTED}"

    monkeypatch.setattr(uta, "CHARTS_DIR", str(tmp_path))
    produced = uta.generate_top100_chart()

    assert os.path.dirname(os.path.abspath(produced)) == str(tmp_path), (
        "chart generation escaped the tmp directory and wrote into the repo"
    )
    assert _md5(produced) == _md5(COMMITTED), (
        "the committed chart no longer reproduces from current code — most likely a "
        "matplotlib/font upgrade. Regenerate and commit the charts deliberately "
        "(BL-04), rather than leaving every doc regeneration to dirty assets/."
    )
