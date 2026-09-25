"""Chart rendering: scoped rc state, determinism, alt text from the same values (R11)."""

from __future__ import annotations

import matplotlib

from supercoach_via.publish import charts


def test_bar_chart_is_deterministic_and_scoped() -> None:
    before = dict(matplotlib.rcParams)
    spec = charts.BarChartSpec(
        title="Demo leaders", unit="disposals", labels=["Demo A", "Demo B"], values=[30.25, None]
    )
    a = charts.render_bar_chart(spec)
    b = charts.render_bar_chart(spec)
    assert a.png == b.png and a.png[:4] == b"\x89PNG"
    assert dict(matplotlib.rcParams) == before
    import matplotlib.pyplot as plt

    assert plt.get_fignums() == []
    assert "Demo A: 30.2 disposals" in a.alt_text or "Demo A: 30.3 disposals" in a.alt_text
    assert "Demo B: not recorded" in a.alt_text
    assert a.plotted == [("Demo A", 30.25), ("Demo B", None)]
