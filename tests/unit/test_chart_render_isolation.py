"""Chart renders must not depend on what ran before them in the same process.

BL-17. `tests/integration/test_chart_reproducibility.py` aborted a live weekly
cycle at Phase 3d. The committed `assets/charts/top10_alltime_hall.png`
reproduced byte-identically from a CLEAN interpreter, but the SAME function
called mid-pipeline — after `generate_readme_charts` had already drawn its
charts in the same process — produced different bytes. The pipeline then
overwrote the good committed chart with the drifted render.

The integration test's docstring blames "a matplotlib/font upgrade". That guess
was wrong here and cost investigation time: the data was identical and the
library was unchanged. The cause was ambient `matplotlib.rcParams` leaking
between generators — `generate_readme_charts._apply_dark_style()` sets a global
`font.size`, and `generate_top100_chart` never established its own, so its
x-tick labels rendered a point larger and `bbox_inches="tight"` reflowed the
whole figure.

These tests pin the property the integration gate actually depends on: the
render is a pure function of its inputs, not of call order. They are hermetic —
tiny fixture CSVs in `tmp_path`, no real `data/`, no writes to `assets/`.
"""
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


@pytest.fixture
def restore_rcparams():
    """Undo any global rcParams mutation a test performs deliberately.

    Without this, perturbing the globals to prove the bug would leak into every
    later test in the session — the exact failure mode under test.
    """
    saved = dict(plt.rcParams)
    try:
        yield
    finally:
        plt.rcParams.update(saved)


@pytest.fixture
def top10_inputs(tmp_path, monkeypatch):
    """Minimal bio + score CSVs wired into update_team_analysis' path constants."""
    import update_team_analysis as uta

    bio = pd.DataFrame(
        {
            "Player Name": [f"Test Player {i}" for i in range(1, 11)],
            "Rank": list(range(1, 11)),
        }
    )
    scores = pd.DataFrame({"all_time_score": [3.0 - 0.1 * i for i in range(10)]})

    bio_path = tmp_path / "bio.csv"
    scores_path = tmp_path / "scores.csv"
    bio.to_csv(bio_path, index=False)
    scores.to_csv(scores_path, index=False)

    charts_dir = tmp_path / "charts"
    monkeypatch.setattr(uta, "TOP100_CSV", str(bio_path))
    monkeypatch.setattr(uta, "TOP100_SCORES_CSV", str(scores_path))
    monkeypatch.setattr(uta, "CHARTS_DIR", str(charts_dir))
    return uta, charts_dir


def _render_bytes(uta, charts_dir):
    path = uta.generate_top100_chart()
    assert path, "generate_top100_chart returned no path"
    assert os.path.dirname(os.path.abspath(path)) == str(charts_dir), (
        "chart generation escaped the tmp directory"
    )
    with open(path, "rb") as f:
        return f.read()


def test_top100_chart_is_call_order_independent(top10_inputs, restore_rcparams):
    """The exact BL-17 sequence: a sibling generator's style, then our chart."""
    uta, charts_dir = top10_inputs
    from generate_readme_charts import _apply_dark_style

    first = _render_bytes(uta, charts_dir)
    _apply_dark_style()
    second = _render_bytes(uta, charts_dir)

    assert first == second, (
        "generate_top100_chart is not isolated from ambient matplotlib state — "
        "rendering after generate_readme_charts._apply_dark_style() changed the "
        "bytes. This is BL-17: in-pipeline it overwrites the committed chart and "
        "aborts the Phase 3d gate."
    )


@pytest.mark.parametrize(
    "rc",
    [
        {"font.size": 11},
        {"font.family": "monospace"},
        {"figure.dpi": 200},
        {"axes.titlesize": 14, "axes.titleweight": "bold"},
        {"xtick.labelsize": 20, "ytick.labelsize": 20},
        {"savefig.dpi": 90, "savefig.facecolor": "#ffffff"},
    ],
)
def test_top100_chart_ignores_ambient_rcparams(top10_inputs, restore_rcparams, rc):
    """Every rcParam a sibling generator sets globally must be inert here.

    Parametrised over the union of what `generate_readme_charts._apply_dark_style`,
    `docs/hall-of-fame/generate_records_charts._apply_dark_style` and the
    in-module `plt.rcParams.update(...)` blocks in update_team_analysis set.
    """
    uta, charts_dir = top10_inputs

    baseline = _render_bytes(uta, charts_dir)
    plt.rcParams.update(rc)
    perturbed = _render_bytes(uta, charts_dir)

    assert baseline == perturbed, f"ambient rcParams leaked into the render: {rc}"
