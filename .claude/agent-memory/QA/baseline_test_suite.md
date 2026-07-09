---
name: baseline_test_suite
description: Baseline pytest count for tests/ so future QA runs can detect regressions vs growth
metadata:
  type: project
---

As of 2026-07-07 (Round 19 weekly-refresh QA gate), `pytest tests/ -v` reports
**244 passed, 0 failed, 0 skipped** in ~1s. Prior baseline (2026-07-03) was 239;
the +5 growth is `tests/unit/test_prediction_selection.py` (3 tests) added for
the CR-1 mtime-vs-lexicographic-sort fix in `scripts/weekly_refresh.sh`, plus
2 more from other untracked test files landing between cycles. Confirmed
non-regression: same green baseline, more coverage.

Note: a prior request assumed "~250+ passing" as the expected baseline — that
number was not grounded in an actual prior QA run recorded in memory (this was
the first QA memory entry). 239 passed with zero failures is a clean, complete
green run; treat 239 (now 244) as the current baseline going forward, not a shortfall.

**Why:** without a recorded baseline, QA can't distinguish "test count dropped
because something broke" from "test count is just lower than someone's
unverified estimate." Recording the actual number here closes that gap.

**How to apply:** on future QA runs, compare the new pass count against 239.
- Count drops with same file set → investigate (deleted/skipped tests).
- Count rises → expected as new modules ship (e.g. this cycle added
  test_commit_authorization.py, test_inject_trust_badge.py,
  test_skeptic_sample_tags.py, test_requires_stamp_routing.py,
  test_staged_blob_check.py, test_tag_vocabulary.py — several of these were
  untracked/unstaged at QA time, see [[project_council_stamp_gate]]).
- Update this memory's number after each QA run so the baseline stays current.
