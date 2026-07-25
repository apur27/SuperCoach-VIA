---
name: project_banner_aria_label_stale
description: docs/banner.svg aria-label text has been stuck at "R1-R13" for multiple cycles while the visible pill/text content updates correctly each week — chronic harness gap, not a per-cycle regression
metadata:
  type: project
---

`docs/banner.svg` has two places the round/MAE numbers live: the visible
`<text>` elements (pills + the "PREDICTION ACCURACY" band) and the top-level
`aria-label` attribute on the `<svg>` root (accessibility text). The weekly
refresh harness correctly rewrites the visible text each cycle (confirmed
2026-07-20: R1-R19→R1-R20, MAE 3.960→3.958, 74.2%→74.4% all updated in commit
`64b366702`), but the `aria-label` string has not been touched since it was
set to "2026 season R1–R13: MAE 4.020, 73.0% within 5 disposals" — verified via
`git log -p --follow -- docs/banner.svg | grep aria-label`, only 2 aria-label
edits exist in the file's entire history and neither is from a weekly cycle.
As of 2026-07-20 (actual state R1-R20, MAE 3.958) the aria-label is 7 rounds
stale.

**Why:** worth recording so a future QA run doesn't waste time re-diffing
banner.svg history to determine "is this new" — it isn't, it predates at
least R14-R20's worth of cycles. It also means screen-reader users have been
getting silently stale numbers for weeks; this is a real (if low-severity)
accessibility/data-freshness bug, not cosmetic noise.

**How to apply:** flag as a WARN (not a FAIL — the visible content is
correct, the artifact exists and is well-formed) on any cycle until the
harness's banner-update step (find it in `scripts/weekly_refresh.sh` or
whatever generates the pill text) is extended to also patch the aria-label
attribute. Route the fix to Gaffer (harness/process gap), not Scientist. Once
fixed, remove this memory or mark it resolved with the fix commit.

**RESOLVED 2026-07-25.** `scripts/update_eval_surface.sh` was fixed this
cycle (uncommitted at QA time, part of the harness+correctness ship) to
rewrite the `aria-label` alongside the visible pills, and also to include the
player-file count in that same label. Verified on the real file:
`docs/banner.svg` aria-label now reads "...13,357 player files...2026 season
R1–R20: MAE 3.958, 74.4% within 5 disposals..." — matches the visible pills
(R1–R20 · 2026 / MAE 3.958 / 74.4% within 5) and the visible "13,357 player
files" text exactly, and 13,357 matches
`ls data/player_data/*performance_details.csv | wc -l`. New regression test:
`tests/unit/test_eval_surface_banner.py` (5 tests, including an idempotency
check and en-dash/hyphen equivalence). No further WARN needed on this check
unless the aria-label drifts from the pills again.
