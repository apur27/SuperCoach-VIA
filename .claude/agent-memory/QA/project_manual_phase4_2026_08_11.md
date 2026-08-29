---
name: project_manual_phase4_2026_08_11
description: 2026-08-11 R24 cycle — harness aborted at Phase 3d (chart reproducibility), Phase 4 completed manually; what QA verified and the one gap found (stale completion sentinel)
metadata:
  type: project
---

The 2026-08-11 weekly-refresh cycle (R23 completed / R24 predicted) had its
harness (`scripts/weekly_refresh.sh`) abort at Phase 3d on
`test_top100_chart_reproduces_byte_identically`. Diagnosis (by the operator,
independently spot-checked by QA): a clean-interpreter render of
`assets/charts/top10_alltime_hall.png` was deterministic and byte-identical
to the blob at `30b2c174a` (174,837 bytes); Phase 1's in-pipeline render had
produced a different 175,862-byte blob from ambient matplotlib state left by
an earlier chart generator in the same process — renderer/data both fine,
gate correctly fired. The operator restored the reproducible chart and
completed Phase 4 manually by replicating the harness's `git add` allowlist
(weekly_refresh.sh lines 412-432), across two commits: `e2947dcb5` (Phase 1,
harness-run) and `8883d5285` (Phase 2/3, manual).

**QA independently verified** (did not just trust the operator's account):
- Manual commit `8883d5285`'s file list matches the allowlist exactly — all
  18 allowlisted paths present, `git status --short` on every allowlisted
  path is clean (fully committed), and the two deliberately-excluded paths
  (`data/top100/yearly/year_2026.csv`, `.claude/agent-memory/**`) are the
  only things left dirty. No missing doc, no extra file.
- `docs/hall-of-fame/_stat_leaders.json` is in the allowlist's `git add` list
  but is gitignored (see [[project_stat_leaders_json_gotchas]]) — `git add`
  on it is a documented no-op, not a gap.
- All 14 stamped council docs that changed this cycle (13 HOF stat-*.md +
  1 leaders.md in the manual commit, plus `docs/afl-backtest-2026.md` in the
  Phase 1 commit) verify via `scripts/check-council-stamp.sh --dry-run`
  against their audit-record content hash: 0 failures.
- `docs/afl-insights.md` correctly has NO inline stamp (opt-in-sticky —
  `docs/afl-*.md` only requires one if it already carries one) but DOES have
  a matching DataSentinel PASS (13/13 tags verified) and Skeptic
  PASS_WITH_CONCERNS record in `.claude/audit/insights_{datasentinel,skeptic}_2026-08-11.json`
  — those files are markdown-fenced JSON (` ```json ... ``` `), not raw JSON;
  strip the fence before `json.loads`.
- R23 backtest figures independently re-derived from
  `prediction_vs_actual_round_23_2026_20260811_102810.csv` (413 rows, 48 NaN
  actuals dropped → n=365): MAE 4.205, RMSE 5.433, bias +0.436 — exact match
  to both the operator's stated run-log figures and the published
  `docs/afl-backtest-2026.md` round-23 table row (4.21/5.43/71.2%/94.8%,
  rounding-consistent).

**One gap found — WARN, not FAIL:** the round-completion sentinel
`.claude/audit/last_refresh_complete.json` is stale at `{"round": "21", ...}`
because the harness aborted before reaching the sentinel-write step (line
454-457 of `weekly_refresh.sh`) and the manual completion did not replicate
it. Per the script's own comment, Chronicler uses this file to detect a
partial run by comparing its round to the max round actually in the data —
with it frozen at 21 against real data at round 23/24, a naive Chronicler
read would misdiagnose this as a partial/failed run. `.claude/audit/last_refresh_status.json`
correctly shows the true abort point (`{"phase":"3d","exit_code":1,...}`),
so the ground truth is discoverable, just not from the sentinel Chronicler is
documented to prefer.

**Why:** this is the first cycle QA has verified where Phase 4 ran outside
the harness. The manual-completion pattern checked out cleanly overall — the
one gap is narrow and specific, not a sign the manual process is unreliable.

**How to apply:** on any future cycle where Phase 4 is completed manually
after a harness abort, explicitly check whether `last_refresh_complete.json`
was updated to the correct round — it is easy to miss since nothing else
depends on it failing loudly. Route the fix (write the sentinel, or teach the
manual-completion runbook to do it) to Gaffer, since Gaffer owns the harness
runbook and cadence.

**RESOLVED, re-verified same cycle.** By the time of a second QA pass the same
day, `last_refresh_complete.json` read `{"round": "24", "completed_at":
"2026-08-11T18:16:27+1000", ...}` — correct round, timestamped after both
ship commits. Gap closed; no further action needed unless it recurs on a
future manual-completion cycle.

**New finding this same pass — WARN:** `docs/afl-backtest-2026.md`'s inline
`<!-- council-pipeline: ... -->` stamp text still reads
`DataSentinel:PASS(pass2)@20260727T095402Z, Skeptic:PASS_WITH_CONCERNS@20260727T081224Z,
Gaffer:SHIP@20260727T100245Z` even though Phase 1's deterministic auto-update
changed the doc's content today (new R23 row). The gate mechanism itself is
NOT broken — `check-council-stamp.sh` passes because a fresh content-hash-keyed
audit record exists (`sentinel-28686c6f...-20260811T003950Z.json`,
DataSentinel PASS, 0 findings, hash matches the current file exactly) — but
the human-readable dates embedded in the stamp comment are up to two weeks
stale and there is no evidence a fresh Skeptic/Gaffer pass ran against this
specific file this cycle (only DataSentinel re-verified the new content-hash).
This is the same class of gap as [[project_banner_aria_label_stale]] (visible/
enforced content correct, embedded provenance text not rewritten on
deterministic-only refresh cycles) — route to Gaffer to decide whether
data-only refreshes should rewrite just the DataSentinel field's date, or skip
restamping by design (as this doc's structure doesn't change) and say so in
the stamp text itself so it doesn't read as an unrefreshed review.
