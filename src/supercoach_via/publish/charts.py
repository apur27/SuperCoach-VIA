"""Scoped, reproducible chart rendering for exported images.

Charts use the Agg backend, an explicit rc context (bundled DejaVu Sans, fixed DPI and
size), strip PNG metadata, and always close their figure. Alt text and the plotted value
list come from the same spec, so image, alt text and table cannot disagree. Byte identity
is expected only in the locked rendering environment; tests assert values separately.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, cast

RC: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.titlesize": 12.0,
    "figure.dpi": 100.0,
    "savefig.dpi": 100.0,
    "svg.hashsalt": "supercoach-via",
    "path.simplify": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
NAVY = "#1b2a4a"
TEAL = "#0f766e"


@dataclass(frozen=True)
class BarChartSpec:
    title: str
    unit: str
    labels: list[str]
    values: list[float | None]


@dataclass(frozen=True)
class RenderedChart:
    png: bytes
    alt_text: str
    plotted: list[tuple[str, float | None]]


def alt_text_for(spec: BarChartSpec) -> str:
    parts = [
        f"{label}: {value:.1f} {spec.unit}" if value is not None else f"{label}: not recorded"
        for label, value in zip(spec.labels, spec.values, strict=True)
    ]
    return f"Bar chart, {spec.title}. " + "; ".join(parts) + "."


def render_bar_chart(spec: BarChartSpec) -> RenderedChart:
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if len(spec.labels) != len(spec.values):
        raise ValueError("labels and values must align")
    with matplotlib.rc_context(rc=cast(Any, RC)):
        fig = Figure(figsize=(8, max(2.0, 0.4 * len(spec.labels) + 1)))
        FigureCanvasAgg(fig)  # no pyplot: no global figure registry or backend switch
        ax = fig.add_subplot()
        try:
            ys = list(range(len(spec.labels)))[::-1]
            vals = [v if v is not None else 0.0 for v in spec.values]
            bars = ax.barh(ys, vals, color=TEAL)
            for bar, value in zip(bars, spec.values, strict=True):
                if value is None:
                    bar.set_hatch("//")
                    bar.set_facecolor("white")
                    bar.set_edgecolor(NAVY)
                label = "not recorded" if value is None else f"{value:.1f}"
                ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {label}", va="center", color=NAVY)
            ax.set_yticks(ys, spec.labels)
            ax.set_xlabel(spec.unit)
            ax.set_title(spec.title, color=NAVY)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", metadata={"Software": None})
        finally:
            fig.clear()
    return RenderedChart(
        png=buf.getvalue(),
        alt_text=alt_text_for(spec),
        plotted=list(zip(spec.labels, spec.values, strict=True)),
    )
