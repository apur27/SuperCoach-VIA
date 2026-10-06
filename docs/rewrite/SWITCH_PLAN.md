# Entry-point switch plan (PLAN Phase 9) as a CLAUDE.md §6.2 harness change

> Local application finalized and merged at `2bcdfbeb4` on 28 September 2026. See [the finalization record](FINALIZATION_2026-09-28.md) for current artifacts, checks and grand-final status. Production activation remains pending. The dated records below preserve their original context.

Status: **not activated**. The default body of `scripts/weekly_refresh.sh` is still the legacy pipeline. An opt-in (`SCVIA_NUMERIC_ENTRY=1`) execs `scripts/scvia_weekly.sh`. Cron, `core.hooksPath`, and the installed hook are unchanged. This document is the written §6.2 scope decision, merge condition and operator answer. Evidence from the rehearsal is in [REHEARSAL.md](REHEARSAL.md).

## 1. Scope decision (§6.2, in writing)

The switch is **in scope under both tests**:

- **Bright line.** It edits `scripts/weekly_refresh.sh` and `refresh_and_rank.sh` (their bodies become `scvia` calls or are retired), `.githooks/pre-commit` (interpreter and test command), and the Python entry points the harness invokes (`refresh_data.py`, `top_players_comprehensive.py`, `backtest.py`, `refresh_readme.py`, which become archived).
- **Behaviour.** It changes phase ordering, what is staged or committed (under the new model the harness no longer commits generated outputs to `main`), what the gates verify (release validation instead of council stamps on generated numeric docs), and which artifact consumers select (release manifest instead of newest-by-mtime).

Consequences: §6.1 freeze (no landing while a cycle is active, checked by the `.claude/audit/last_refresh_status.json` marker) and §6.2 merge condition (green scratch-worktree smoke run, section 4). `tests/integration/` is not touched: the new tiers live in `tests/scvia/` (decision D1), so the legacy Phase 3d gate keeps its meaning until it is retired.

## 2. Preconditions (must hold before the switch PR is opened)

| # | Precondition | State 2026-09-25 |
|---|---|---|
| P1 | No active legacy cycle (§6.1 marker has `exit_code` set for the latest run) | Marker `{"phase":"4","exit_code":0,"round":"25"}`; no process running |
| P2 | Old-vs-new rehearsal on identical input shows only documented differences | Done (REHEARSAL.md): ties at the rank cut, duplicate-identity correction, B1 refusal |
| P3 | Publish, rollback and restore rehearsal is green | Done (9/9), after the integrity-only publish fix |
| P4 | Owner decisions in section 6 are made | **Open** |
| P5 | Release-build cost within §12 budgets, or an explicit owner acceptance | **Met 2026-09-26**: 56.9–57.8 s and 1.39–1.43 GiB against 60 s and 2 GiB (4 vCPU; small margin) |
| P6 | Artifact budget sized for at least 3 seasons of growth | **Met against the unchanged 300 MiB budget** (2026-09-27): final site 276,795,445 bytes; three further seasons at the 2021–2025 mean plus 5 MiB projects to 280.44 MiB. Headroom to 300 MiB is 36.03 MiB. The projection is not a measured future season. |

## 3. The change, in three separately smoke-tested steps

Each step is its own PR with its own smoke run. None of them deletes legacy data.

**Step A: shadow (no harness edit).** Run `scvia` after each legacy cycle as a separate operator or cron command. It reads the same `data/` bytes, imports them with the archived repairs, builds a release, and runs `rehearsal_compare.py` against that cycle's legacy outputs. Nothing is published. Exit criterion: two consecutive cycles whose only differences are the documented ones. Because the harness is not edited, this step needs no §6.2 smoke run, but it must not run while a cycle is active (§6.1), because it reads the same working tree.

**Step B: switch the weekly entry point.** `scripts/weekly_refresh.sh` becomes a thin, `set -euo pipefail` wrapper with no inline Python, heredocs or LLM calls in the numeric path:

```bash
uv run --locked scvia refresh --season "$SEASON" --data-only --allow-network --json   # exit 3/4: stop, previous snapshot stays current
uv run --locked scvia forecast --train-cutoff "$TRAIN_CUTOFF" --calibration-end "$CAL_END" \
  --cutoff "$(date -u +%FT%TZ)" --json                                                # no future fixture: exit 0, forecast_status=unavailable
uv run --locked scvia build-release --snapshot current --editorial off \
  --bundle "$BUNDLE" --predictions "$PRED_DIR" --content-manifest config/public_content.toml --json
uv run --locked scvia validate-release --release "$RELEASE" --json
# publication is a separate, manually dispatched job (scvia-pages.yml); the harness never publishes
```

The weekly run no longer commits generated docs, charts or CSVs to `main`. The release is the published artifact, and its manifest is the only thing consumers select. The CLAUDE.md "Serialize writes to main" rule is updated in the same PR, keeping its intent (one gated writer) but with the gate being `validate-release` plus `publish` receipts. `refresh_and_rank.sh`, the council stamp hops, `update_eval_surface.sh` (whose mtime vintage selection blocks fresh-checkout runs, REHEARSAL.md Finding 1) and the numeric generators are **archived, not deleted**. Every one is mapped in `docs/migration.md`. The optional editorial lane (FootyStrategy recap) moves to `editorial/` and cannot block the numeric release.

**Step C: hooks, CI and docs.** `.githooks/pre-commit` drops the hardcoded interpreter default in favour of `uv run --locked`, and keeps the fast tier (`pytest tests -m "not integration"`, about 82 s today, under the hook's 300 s timeout). CLAUDE.md's test and command contract is updated to the `scvia` commands without weakening the verification rules. The legacy `tests.yml` and `weekly-fan-pack.yml` (scheduled) workflows are replaced by `scvia-ci.yml` (push/PR checks) and `scvia-pages.yml` (manual dispatch only, never scheduled).

## 4. Smoke-run procedure (the merge condition for steps B and C)

1. Confirm P1: the marker has `exit_code` set and no `weekly_refresh`/`refresh_and_rank` process is running.
2. `scripts/smoke_harness.sh --ref <switch-branch>`. It builds a scratch worktree at the branch, applies uncommitted changes, copies the current `data/` and `.claude/audit`, and shims `git commit`/`push` to no-ops. Two additions are needed for the new harness, and they belong in the switch PR itself:
   - the source step runs against the previous week's archived raw payloads (`var/raw` + repair or refresh evidence, the same mechanism as `docs/rewrite/evidence/b1`) with network dead-ended, instead of `SMOKE_SKIP_SCRAPE`. That keeps it hermetic and lets it exercise the real parse and merge path.
   - `scvia publish` is pointed at a `LocalDirectoryDestination` inside the scratch worktree, so the smoke run also exercises publish and rollback.
3. Green means every item below:
   - the wrapper exits 0 (a missing fixture is exit 0 with `forecast_status=unavailable`);
   - `validate-release` PASS;
   - `rehearsal_compare.py` shows only documented differences against the previous week's legacy outputs;
   - `web` builds against the release;
   - the budget script passes;
   - no `git commit`/`push` was attempted outside the shim.
4. Attach the smoke log and the compare JSON to the PR. **A green smoke run is the merge condition; unit tests do not substitute.**

## 5. Operator answer: "what do I do when it fires mid-cycle?"

| Exit | Meaning | Operator action |
|---|---|---|
| 3 | A source is unavailable or the refresh is partial | Nothing is promoted. The previous snapshot and the live release stay. Re-run later. |
| 4 | Dataset or release validation failed | Read `var/runs/<id>/validation-report.json`, then repair (bounded `--repair-season` or archived evidence) and re-run. Do not disable the check. |
| 5 | Locked | Another writer is running. Wait. Never delete the lock while it runs. |
| 6 | A requested model bundle is unavailable | Rebuild with `scvia forecast`, or build without forecast inputs. A missing fixture is **not** exit 6: `forecast` exits 0 with `forecast_status=unavailable` and the release says so. |
| 7 | Publish failed | The previous release stays live and a failed receipt is written. Retry `scvia publish` with the same release ID, or `scvia rollback --release <previous>`. |

**Rolling back the switch itself:** revert the switch commit. Legacy scripts are archived in place, and the new pipeline never writes `data/` CSVs, so the legacy harness runs again from the same inputs. (It still needs REHEARSAL.md Finding 1 handled to run from a fresh checkout.)

## 6. Prepared working defaults (not activated)

These are the defaults the candidate is built to. They are not an owner signature and they do not switch production.

1. Published numeric output is the sealed release artifact. Old generated documents stay in the tree as archives. The candidate does not commit them.
2. The active artifact is the one on the local or static host. Two sealed backups stay on that host. The earlier ~880 MiB Pages-retention figure is not the working plan.
3. The site budget stays 300 MiB. It is not raised to ~320 MiB.
4. Release-build cost (P5): met; no decision needed unless the reference machine changes.
5. Editorial generation stays off (`--editorial off`).

Runtime is the static site plus the local locked Python environment. Pages deployment stays a manually triggered job that uploads an already sealed tree; this plan does not dispatch it.

## 7. Continuation status (2026-09-27, not activated)

This section records the local continuation. It does not switch, push, deploy, or enable a schedule.

The default body of `scripts/weekly_refresh.sh` still calls `/home/abhi/sourceCode/python/coding/.venv/bin/python`. An opt-in at the top execs `scripts/scvia_weekly.sh` only when `SCVIA_NUMERIC_ENTRY=1`. Cron and `core.hooksPath` were not changed. The live hook is still the main repo's `.githooks`. `docs/rewrite/switch-candidate/pre-commit-python.sh` is a proposal and is not installed.

`scripts/scvia_weekly.sh` is the candidate. `SCVIA_SOURCE_MODE=production` refreshes `--data-root` the same root later stages use, and requires `SCVIA_ALLOW_NETWORK=1`. Its forecast cutoff is the run's UTC time unless `SCVIA_FORECAST_CUTOFF` is set. The default mode is `rehearsal`: it imports `SCVIA_CAPTURED_SOURCE`, requires an explicit `SCVIA_FORECAST_CUTOFF`, and does not pass `--allow-network`. It then forecasts, builds with editorial off, runs the Astro site, checks the budget, seals, and validates. When `SCVIA_LEGACY_ROOT` is set, the comparison ranks that promoted snapshot, not a second import. Every exit, including a refused start, writes `var/scvia-weekly-status.json` (or `SCVIA_VAR_DIR`) with `run_id`, `started`, and `exit_code`. The script refuses to start when the cycle marker has no `exit_code` or a `weekly_refresh.sh` / `refresh_and_rank.sh` process is running. It does not publish. `SCVIA_SKIP_SITE=1` exists only for the command-order unit test.

**Correction persistence (30 September 2026).** The candidate now runs offline
`apply-corrections` after import plus archived repairs, or after production refresh,
and before forecast/build. It applies the season named by the forecast cutoff's year;
the correction command also checks historical replay links and blank semantics.
Forecast, release build and optional comparison all use the immutable snapshot ID
returned by the successful correction stage. Rehearsals require both
`SCVIA_CORRECTION_EVIDENCE_ROOT` and `SCVIA_CORRECTION_EVIDENCE_SNAPSHOT` (an explicit
`sha256:<id>`). The CLI verifies that evidence snapshot, copies only the requested
season's pinned page and successful source observations, and binds field corrections
to the captured bytes. It copies no donor match or player rows and sets no season
completeness flag. A grand final absent from the captured CSV source remains absent
even when the evidence snapshot contains it; the full integrity audit must report
that source scope. Legacy CSVs and the evidence donor remain unchanged.

The full §6.2 smoke inherits those exported evidence variables. After P1 is checked
and the final source is frozen, a local invocation using the retained pinned evidence is:

```bash
env SCVIA_BIN="$PWD/.venv/bin/scvia" \
  SCVIA_CAPTURED_SOURCE="$PWD" \
  SCVIA_CORRECTION_EVIDENCE_ROOT=/home/abhi/git/SuperCoach-VIA/var/finalized/data \
  SCVIA_CORRECTION_EVIDENCE_SNAPSHOT=sha256:aa836549e10e96a9039cc4642bbe563b94247e0bfaf95e745a0971cc0bb2b98f \
  SCVIA_FORECAST_CUTOFF=2026-09-25T00:00:00Z \
  SCVIA_SMOKE_ROOT=/tmp/scvia-correction-smoke-UNIQUE \
  bash scripts/smoke_scvia_candidate.sh
```

Since the AFL Tables reconciliation corrections (commit 67217df40, 2026-10-06) the legacy CSVs contain the
archived B1 repair rows, so `scvia_weekly.sh` passes no `--repair` unless `SCVIA_REPAIR` names one. For a
captured source taken before those corrections, add `SCVIA_REPAIR=docs/rewrite/evidence/b1:2026` to the
invocation above; on the corrected source, B1 cannot reproduce its rows and would add Dalton under a second id.

The cached model bundle now has knowledge cutoff 2026-09-27, so `SCVIA_FORECAST_CUTOFF=2026-09-25T00:00:00Z`
is refused by the leakage guard (correctly). Use a cutoff on or after the bundle's, e.g. `2026-09-28T10:00:00Z`
(as in `docs/operations.md`); after the Grand Final the forecast is then `no_valid_future_fixture`, the
correct outcome. Verified on 2026-10-06 (corrected source, no `SCVIA_REPAIR`): both sealed releases built,
1,005 files with 0 scratch differences, rollback left the first release live, exit 0.

This is a procedure, not a new smoke result. The chosen captured source determines
which matches can appear. Inspect `corrections.json` and the promoted snapshot to
confirm the expected fixture value and source digest, then confirm forecast/build
use that corrected snapshot. The default legacy entry remains unchanged.

**Verified correction rehearsal (1 October 2026, Melbourne).** The command above ran
on the final source at `/tmp/scvia-sol-smoke-20260930-final2`, exit 0 in 458.423 s.
All 897 selected source files matched the scratch copy (inventory `a7d9a3ae…3ff1e0`).
Both cycles imported `sha256:b02706a1…b1bfac`, then forecast and built from the corrected
child `sha256:e70e66f6…21d300`. The public R17 attendance is 62,117 **[data]**;
the correction receipts bind it to season capture `87c99c1a…bc83a`. Strict comparison
matched all 130 yearly lists exactly and the all-time score delta was zero.
The temporary host accepted two sealed releases, retained its bytes after an injected
copy failure, and rolled back to `20260930T211937Z-0495e493fc14`.
Receipts: `docs/reviews/opus55/sol-followup-verification.json` and
`var/reviews/codex-opus55-20260930/validation/scratch-smoke-final.{json,log}`.
The source CSVs predate the grand final; this rehearsal does not verify a current-season
production refresh or count toward the two real shadow cycles.

If corrections fail, the cycle stops before forecast/build. The wrapper's status
records `phase=corrections`, the failing exit code, the imported/refreshed
`source_snapshot_id`, and the last successfully selected `snapshot_id`; the stage's
full receipt is `evidence_dir/corrections.json`. Missing/unpinned/corrupt evidence
is exit 3; correction validation failure is exit 4. The source stage may already
have promoted its import or refresh, but no corrected child is promoted on these
failures and the live release is untouched. Read the receipt and the named CLI run's
validation report, supply the correct immutable capture or repair the input, then
rerun. Do not bypass the correction stage. For an existing accepted dataset, the
same bounded command is available directly:

```bash
scvia apply-corrections --data-root TARGET --season 2026 \
  --evidence-root EVIDENCE_ROOT --evidence-snapshot sha256:IMMUTABLE_ID --json
```

An already pinned accepted snapshot can omit the two evidence options. A requested
season without a pinned page fails instead of reporting a successful no-op; a
second application of the same valid evidence is idempotent.

**Rehearsal parity after corrections.** Legacy regeneration now copies the captured
inputs, applies the archived B1 repair, and aligns verified replay deltas before
running `top_players_comprehensive.py`. The comparator receives both the import's
snapshot ID and the correction's snapshot ID, verifies ancestry and the retained
input binding, and compares only changed player-game partitions with Arrow/DuckDB.
It independently checks the official draw/replay pair and score reconciliation.
Statistic-preserving relinks leave the legacy ranking inputs intact. An unresolved
replay row is removed from the scratch CSV only when the candidate preserves matching
quarantine evidence and the original CSV hash, row revision, identity and statistics
verify. All proposed removals are checked before any scratch file is changed;
unexpected additions, deletions or statistic mutations stop the comparison.
Original CSVs, donor evidence and immutable snapshots are preserved. The receipt
records each removal's source hash and row, both snapshot IDs and relink/removal counts.
Fixture attendance does not enter the legacy ranking formula. Score tolerances,
player membership checks and yearly coverage checks are unchanged.

If this comparison fails, read `legacy-regenerate.json` and `compare.json` in the
cycle's evidence directory. Resolve the named source or correction discrepancy and
rerun; do not broaden score tolerances or skip a season. A failed comparison prevents
the scratch smoke from reaching publish/rollback, even if the earlier site and seal
steps passed. A correction that only quarantines rows is supported without requiring
a relink in the same run.

Pack a sealed release, then give the job those two digests. The job does not trust a digest found inside the download:

```bash
python -m supercoach_via.publish.deploy pack RELEASE_DIR bundle.tar
# prints {"seal_sha256": "...", "archive_sha256": "..."}
```

`scvia forecast` still refuses a bundle whose code fingerprint does not match the current feature code. The final uncontended run retrained (`train` 44.71s) and returned exit 0 with `forecast_status=unavailable` / `no_valid_future_fixture`. Bundle `bundle-d8349be6008314f2f6ac`. Release `20260927T100844Z-bc1ed67add68`. A missing fixture is still success. A partial source remains exit 3. Publish failure remains exit 7 with the previous live tree left in place.

Measured final site `var/corpus-site-final` is 276,795,445 bytes (263.97 MiB). The earlier `var/corpus-site` (276,794,969 bytes) and release `20260927T085937Z-d71eb27f6618` are preserved. Mean 2021–2025 season is 4,007,870 bytes. Three further seasons at that mean plus 5 MiB is 280.44 MiB, under the 282 MiB target. 2026 in the snapshot is 4,128,965 bytes and is not a complete season.

Node is pinned to 22.23.3 in `.node-version` and `web/.node-version`. CI selects that file. The verified host binary and the Playwright run are v22.23.3. The earlier file pin 22.23.2 was not the binary those checks used.

The final local scratch smoke is `/tmp/scvia-scratch-smoke-20260927c` (exit 0, wall 557s). It used `SCVIA_NUMERIC_ENTRY=1 scripts/weekly_refresh.sh` on a scratch copy of this tree. Source inventory `fb871edaad740cd19eca046142328127747d4843f552dc5eff637d8b8b0dc38e` (857 files) matched the scratch bytes. Captured inputs `995b138997fad3efdbc310ca95910490e0357fa5b3e2fe6d23491dd2fd7e9b5e` (27,350 files). Compare verdict passed with all-time delta 0. Releases `20260927T130550Z-d424a18cd169` and `20260927T131007Z-fd89198bce47`; an injected copy failure left the first release's bytes in place, then rollback returned to it. Forecast stayed `unavailable` / `no_valid_future_fixture`. 2026 is still incomplete. Earlier scratch logs remain historical and are not this result.

A later rehearsal on the sealed corpus releases, host `/tmp/scvia-rehearsal-final/host`, published the preserved release and then `20260927T100844Z-bc1ed67add68`, refused a missing release, kept the previous tree live when upload raised, and rolled back to `20260927T085937Z-d71eb27f6618`. Nothing was sent to a remote host.

Proposed activation, not run, and only after section 6 is decided, two shadow cycles have matched, and the cycle marker has `exit_code` set:

```bash
# Not run. After two matching shadow cycles and an exit_code on the cycle marker:
# 1. Confirm no weekly_refresh.sh or refresh_and_rank.sh process is running.
# 2. Remove the SCVIA_NUMERIC_ENTRY gate so weekly_refresh.sh execs
#    scripts/scvia_weekly.sh unconditionally.
# 3. Run a production cycle with SCVIA_SOURCE_MODE=production and
#    SCVIA_ALLOW_NETWORK=1. Do not point core.hooksPath at
#    docs/rewrite/switch-candidate/.
```

Recovery if that activation is wrong: restore the legacy body of `scripts/weekly_refresh.sh` (the opt-in exec is the only new branch) and leave `scripts/scvia_weekly.sh` unused. The live host pointer can be moved back with `scvia rollback --release <previous-sealed-id> --destination <host>`. Do not delete `data/` CSVs.

Still open before an activation choice: two real shadow cycles and the owner decisions in section 6. The 2026 grand final is in the local snapshot `sha256:aa836549e10e96a9039cc4642bbe563b94247e0bfaf95e745a0971cc0bb2b98f` only. That is not a production refresh. The switch is not activated.
