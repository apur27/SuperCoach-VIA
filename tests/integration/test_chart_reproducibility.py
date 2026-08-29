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

BUT AN UPGRADE IS NOT THE ONLY CAUSE, AND IT IS NOT THE COMMON ONE. Work out which
cause you have BEFORE following the recovery below, because for the other cause that
recovery is actively harmful. On 2026-08-29 this gate aborted a cycle with no upgrade
involved: matplotlib.rcParams is process-global, and a sibling generator left
font.size: 11 behind (generate_readme_charts._apply_dark_style). This chart sets an
explicit fontsize on every text object EXCEPT the x-axis tick labels, which resolve
xtick.labelsize: "medium" against that ambient value — one point, and then
bbox_inches="tight" reflows the entire figure. The byte diff looks exactly like a font
build change. It was not: the data was identical and a CLEAN interpreter still
reproduced the committed bytes.

Tell the two apart in one step — render this chart in a FRESH interpreter:
  * Fresh render MATCHES the committed file -> data and renderer are both fine; you
    have state leakage, not an upgrade. Do NOT recommit charts. That would bake
    whichever call order happened to run into the artifact, which IS the defect. Fix
    the leak instead (see _deterministic_chart_rc in update_team_analysis.py and
    tests/unit/test_chart_render_isolation.py).
  * Fresh render DIFFERS from the committed file -> the rendering environment really
    did change. Follow the recovery below.

THIS FILE IS A LIVE GATE. tests/integration is Phase 3d of scripts/weekly_refresh.sh,
which fails closed and aborts before Phase 4 stages anything — so a failure here stops a
real weekly cycle. Recovery, for whoever hits it mid-run:

  1. It means the RENDERING environment changed, not that the data is wrong.
  2. Regenerate: generate_readme_charts.py, update_team_analysis.generate_top100_chart(),
     docs/hall-of-fame/generate_records_charts.py
  3. Confirm the diff is rendering-only by re-rendering a PRIOR data vintage in the
     current environment and checking it matches today's render. Do NOT try to prove it
     by diffing the input CSVs — all_time_top_100.csv is the BIO file and carries no
     scores, so it will show "no change" whether or not the data moved. That mistake was
     made on 2026-07-28 and reported as evidence.
  4. Commit the refreshed charts, then re-run the cycle.

Do not disable this check to get a cycle out. A chart that no longer reproduces is
exactly the signal the gate exists to surface.
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
