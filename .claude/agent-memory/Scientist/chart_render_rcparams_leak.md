---
name: chart-render-rcparams-leak
description: Phase 3d chart-reproducibility failures are usually a process-global matplotlib rcParams leak between generators, NOT a matplotlib/font upgrade — the gate's own docstring guesses wrong
metadata:
  type: project
---

A `tests/integration/test_chart_reproducibility.py` failure has TWO distinct causes, and
the test's docstring only names one of them.

**Cause A (what the docstring says):** matplotlib/font build changed. Signature — the
chart does NOT reproduce even from a CLEAN interpreter. Recovery is to regenerate and
recommit the charts deliberately.

**Cause B (BL-17, fixed 2026-08-29):** ambient `matplotlib.rcParams` leaking between chart
generators inside ONE process. Signature — the chart DOES reproduce from a clean
interpreter but the in-pipeline render differs. Data unchanged, library unchanged.
Recommitting charts here is the WRONG move; it bakes in whichever call order happened to
run.

**Why:** `matplotlib.rcParams` is process-global and no generator restores it.
`generate_readme_charts._apply_dark_style()` sets `font.size: 11`;
`update_team_analysis`'s in-module chart blocks set `font.family: "monospace"`. Any chart
that leaves a parameter unset inherits the last writer's value. With
`bbox_inches="tight"` a one-point tick-label change reflows the entire figure, so the byte
diff is large and looks like a font-build change. `plt.style.use("dark_background")` does
NOT reset first — it only overlays its own keys, so it is not protection.

Confirmed hashes for `top10_alltime_hall.png`: clean `691acd8f…`; after
`generate_readme_charts._apply_dark_style()` `02e97b9d…`; after a leaked monospace
`06411d67…`.

**How to apply:** Before concluding "renderer upgrade", render the chart twice in one
interpreter — once first, once after calling a sibling generator. If they differ, it is
cause B. Fix by isolating the render in `plt.rc_context(_deterministic_chart_rc())`
(pinned to `matplotlib.rcParamsDefault` minus the backend keys) — never by asking callers
to reset state, which regresses the moment a new generator is added.

Still latent as of 2026-08-29: `generate_readme_charts.chart_top10_alltime` and
`chart_top100_position_breakdown` are order-fragile the same way (verified — a leaked
`font.family` changes their bytes). They are NOT on the Phase 3d gate, so their drift
shows up only as unexplained `assets/` churn.

Guard: `tests/unit/test_chart_render_isolation.py` (fast tier, hermetic).
Related: [[golden-file-test-tier]], [[integration-test-tier]].
