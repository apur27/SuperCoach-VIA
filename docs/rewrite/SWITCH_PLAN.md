# Entry-point switch plan (PLAN Phase 9) as a CLAUDE.md §6.2 harness change

Status: **plan only**. No harness, gate or hook file has been changed. This document is the written §6.2 scope decision, merge condition and operator answer that the change must carry. Evidence from the rehearsal is in [REHEARSAL.md](REHEARSAL.md).

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
| P5 | Release-build cost within §12 budgets, or an explicit owner acceptance | **Open**: 216 s and 3.6 GiB against 60 s and 2 GiB |
| P6 | Artifact budget sized for at least 3 seasons of growth | **Open**: 7.2 MiB headroom ≈ 1.7 seasons (REHEARSAL.md) |

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

## 6. Owner decisions needed (P4)

1. Should `main` still receive committed generated docs and charts, or is the release artifact the only published output? The plan above assumes the release artifact only, with the legacy docs frozen as archive.
2. Hosting and retention: GitHub Pages with active plus 2 previous releases is about 880 MiB against its 1 GB limit. Choose 1 retained release, cross-release deduplication, or another host.
3. The artifact budget: raise it to about 320 MiB (about five more seasons), or trim game logs further.
4. Release-build cost (P5): fix first (next task), or accept the current numbers for the switch.
5. Whether the LLM editorial recap continues, as an optional non-blocking lane.
