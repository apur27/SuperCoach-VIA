---
name: golden-file-test-tier
description: Where the golden-file snapshots for the doc generators live, how to re-record them, and the silent-no-op guard class they enforce in update_team_analysis.py
metadata:
  type: project
---

Golden-file regression tier for the published-doc generators, added 2026-07-26.

**Layout.** Fixtures + snapshots under `tests/fixtures/golden/`:
- `top100/` — `bio.csv`, `scores.csv`, `hof_input.md` (inputs) and
  `expected_top100_section.md`, `expected_hof_doc.md` (snapshots).
- `sections/` — `sections_input.md` and `expected_sections_doc.md` (all six
  marker replacers composed).
Tests: `tests/unit/test_golden_top100_render.py`,
`tests/unit/test_golden_section_replacers.py`. Hermetic (fast tier, ~0.5s each),
every module path constant monkeypatched — see [[tests_can_write_real_artifacts]].

**Re-recording.** `UPDATE_GOLDEN=1 pytest tests/unit/test_golden_*.py` rewrites
the snapshots and SKIPs. Always `git diff tests/fixtures/golden/` afterwards.
Re-recording to turn a red test green without reading the diff defeats the tier.

**Why:** two shipped generator defects had the same shape — a substitution that
matched nothing, returned the input verbatim, and let the caller (`if new_text
!= text:` … else write nothing, report no error) log a clean run over a frozen
published section. A substring assertion cannot catch "changed nothing"; a
snapshot can.

**Guard class now enforced in `update_team_analysis.py` — do not revert to a
`print(...)+return readme_text`:** `replace_top100_section`,
`replace_stat_leaders_section`, `replace_predictions_section` and
`replace_backtest_section` raise `ValueError` when neither the marker pair nor
the fallback anchor is found. `regenerate_top100_profiles` raises when the count
of `### ` headings in the profiles region != the count of parsed blocks (an
en-dash/entity heading is otherwise swallowed into the previous block's prose and
re-emitted as an empty placeholder — silent loss of authored narrative).
`check_top100_consistency` appends a HARD mismatch when it matched 0 ranked
blocks in a doc that has the profiles heading (a gate that verified nothing must
not report PASS; the CLAUDE.md inline-tag exemption for that doc rests on it).

**How to apply:** when adding a new published section, give its replacer the
same loud-on-zero-match contract and add it to the `REPLACERS` list in
`test_golden_section_replacers.py` — the parametrised zero-match and idempotency
tests then cover it for free. Note the fallback anchors are prose strings with
em dashes (`## {year} season — live team analysis`), so a dash re-encoding in
the target doc is a live failure mode, not a hypothetical.

Related: [[top100_profile_regen]], [[hof_full_table_regen]].
