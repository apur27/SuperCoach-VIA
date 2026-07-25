---
name: sentinel-record-vs-log
description: Two DataSentinel audit artifacts look alike but only one is authoritative; don't read a PASS as FAIL from the stale log
metadata:
  type: feedback
---

The `.claude/audit/` directory holds TWO different DataSentinel artifacts for the insights lane. Do not confuse them.

- `insights_datasentinel_<date>.json` — this is the Phase-3b `DS_OUT` **log dump** from `weekly_refresh.sh` (`$LOG_DIR/insights_datasentinel_${TODAY}.json`). It is overwritten per attempt and can hold a STALE early-morning FAIL even after the doc was later fixed and re-passed. NOT authoritative. Not keyed to content.
- `sentinel-<64hex-content-hash>-<ts>.json` — the **authoritative, hash-keyed verdict record** written by `scripts/record-sentinel-verdict.sh`. Shape: `{"doc_path","doc_hash","verdict","ts","agent_id"}`. The pre-commit stamp gate (`check-council-stamp.sh`) trusts ONLY an exact `"verdict":"PASS"` record whose `doc_hash` matches `council-content-hash.sh <doc>`.

**Why:** In the 2026-07-13 R20 cycle the log dump showed a FAIL (11:00Z, 11 untagged tags, "no source declared") while the doc had since been fixed — FootyStrategy added the methodology paragraph declaring sources — and re-passed at 23:20 with hash `9357e497…`. Reading the log alone would wrongly conclude the doc is un-shippable.

**How to apply:** To confirm a doc is gate-cleared, (1) compute `scripts/council-content-hash.sh <doc>`, (2) find `sentinel-<that-hash>-*.json` and confirm `"verdict":"PASS"`. Ignore the `*_datasentinel_<date>.json` log for verdict purposes. See [[project_weekly_r19_retro]], [[feedback_datasentinel_nondeterminism]].
