# Phase 8 rehearsal: old vs new, publish, rollback and restore (2026-09-25)

Operator evidence for [PLAN.md](PLAN.md) Phase 8. Nothing in this rehearsal committed, pushed, published to a real host or called an LLM. The raw results are in [evidence/rehearsal/](evidence/rehearsal/), and every step can be rerun with the scripts named below.

## Summary

| Question | Answer |
|---|---|
| Did both sides read identical input? | Yes. `data/` had 27,350 files with aggregate sha256 `f5f92769…` in both the repo and the legacy worktree after its run; the only exception is `data/top100/`, which the legacy run itself rewrites. The new side imported that same worktree. |
| Legacy harness result | **Failed closed in Phase 1** (`update_eval_surface.sh`: backtest team-n reconciliation, 9,099 vs 9,135) before staging or committing anything. See the finding below. |
| New pipeline on the same bytes | **Refuses to promote**: validation FAIL with 13 blocking current-season gaps, the ones B1 later repaired. On these bytes the legacy harness would have gone on to rank and publish. |
| Rankings, legacy vs new | Yearly top 100 across 130 seasons: 124 have identical players and scores (56 in identical order, 68 differ only among exact ties). The other 6 each swap one player *tied at the 100th-place cut*. All-time top 100: same 100 players, max score delta 0.00137, two adjacent rank swaps. The drift comes from the two duplicate identity files, which legacy double-counts and the rewrite quarantines; the parity test, which excludes them from the legacy scan, gets delta 4.4e-16 with identical order. |
| Biography CSV | 96 of 100 rows byte-identical. The 4 that differ are the two adjacent swaps above (ranks 16/17 and 84/85). |
| Stubbed publish / rollback / restore | All 9 steps behaved as expected, after fixing one defect found here (rollback across a contract change). |

## Setup (container-only, nothing in the repository changed)

- Legacy side: `scripts/smoke_harness.sh` (the CLAUDE.md §6.2 tool) at HEAD `74a25df6`, run in a scratch git worktree with `FINALS_MODE=1` (the season is in finals), `SMOKE_SKIP_SCRAPE=1`, and `HTTP(S)_PROXY` pointed at a closed port so no source is contacted. `git commit` and `git push` go through the smoke shim.
- The legacy harness hardcodes `/home/abhi/sourceCode/python/coding/.venv/bin/python` and `/home/abhi/.claude/local/claude`. For the rehearsal these container paths became (a) a wrapper that execs the locked `uv` environment (pandas 2.3.3) and (b) a **stub `claude`** that logs its arguments and exits 1, so no provider write is possible. The stub was never called.
- New side: `docs/rewrite/evidence/rehearsal_compare.py <legacy-worktree> <scratch>` imports the legacy worktree's `data/` *without* the archived B1 repair, validates it, and computes `legacy_v1` rankings from the unpromoted candidate.
- Publish side: `docs/rewrite/evidence/rehearsal_publish_restore.py` publishes real validated releases with `scvia publish` to a `LocalDirectoryDestination`, the stand-in static host.

Legacy wall time was 596 s, peak RSS 947 MiB.

## Finding 1: the legacy harness cannot complete a cycle from a fresh checkout

`update_eval_surface.sh` picks the "newest vintage" of `backtest_by_team_*.csv` **by file mtime**. In a fresh worktree all 19 files share one checkout timestamp, so the choice is arbitrary. It then refuses to write, because the per-team rows (9,099) do not reconcile with the round summaries (9,135). Choosing by the timestamp in the filename instead gives 9,007, which still disagrees. The committed per-team files and round summaries are therefore genuinely inconsistent, and only the operator machine's write order made them reconcile. This is the "consumer selects latest by mtime" defect from the audit (PLAN §1 correctness list), and it now blocks a legacy smoke run.

The failure is fail-closed and happened before any staging. Under CLAUDE.md §6.1/§6.2 the harness is **not patched here**. The switch plan below replaces this step, and until then a smoke run of any legacy harness change will hit the same wall. Operator workaround for a legacy run on the original machine: none needed there, because its mtimes are real.

## Finding 2 (fixed): rollback failed across a public-contract change

`publish_release` re-ran full semantic validation with the **current** code. After the compact player/box-score contracts landed, every earlier validated release failed the new schemas (17,055 match files), so rollback to them was impossible. That breaks PLAN §16 ("rollback selects a previously validated public release"). Publication and rollback now check **integrity only**: the checksums file matches the validation record, and no file is missing, extra or changed. Semantic validation stays at `validate-release` time, under the contract the release was built with. Tests: `test_rollback_survives_a_later_contract_change`, plus `test_publish_refuses_bytes_that_differ_from_the_validated_release` for modified, added and removed files.

## Publish, rollback and restore steps (`publish-rollback-restore.json`)

| Step | Result | Live release after |
|---|---|---|
| publish `114838` (older contract) | published | `114838` |
| publish `115914` | published (receipt names previous `114838`) | `115914` |
| publish `104856` (validation FAIL) | refused, exit 7 | `115914` |
| injected upload failure (ENOSPC) | failed receipt, exit 7, previous stays live | `115914` |
| rollback to `114838` | published, `kind=rollback` | `114838` |
| back up `var/` (current pointer, snapshots, fragments, models, predictions, evaluations) and restore to a fresh root | every restored tree digest equals the backup source | - |
| `scvia validate --snapshot current` on restored root | PASS, snapshot `sha256:55e295f1…` | - |
| `scvia status` on restored root | current snapshot matches | - |
| `scvia forecast` on restored root | reused bundle `bundle-ac063296…`, no retrain | - |

Four receipts were written (3 published, 1 failed). `var/` is 52 MB, so a full backup is cheap.

## Published bundle growth per season (`evidence/season-growth.json`)

Measured on the real site built from release `20260925T115914Z-103b1f82beb7` (292.8 MiB total) with `docs/rewrite/evidence/season_growth.py`. Bytes are attributed to a season by path: game logs, match detail and indexes, team pages, yearly rankings, lists, plus each player page's season line.

| Season | MiB | | Season | MiB |
|---|---:|---|---|---:|
| 1900 | 0.78 | | 2019 | 3.97 |
| 1950 | 1.33 | | 2020 (shortened) | 3.25 |
| 1980 | 1.87 | | 2021 | 4.13 |
| 2000 | 3.38 | | 2024 | 4.26 |
| 2010 | 3.57 | | 2025 | 4.32 |
| 2015 | 3.99 | | 2026 | 4.33 |

A modern season costs **about 4.3 MiB**: game logs 2.1, match detail 1.6, team pages 0.34, player season lines 0.24, lists and rankings about 0.05. Season-independent base: 41.3 MiB.

Sizing:

- Headroom is 7.2 MiB. The 2026 grand final adds well under 0.1 MiB. **The completed 2027 season fits (about 297 MiB); 2028 exceeds 300 MiB (about 301 MiB).** Base growth from new articles and downloads comes on top of this.
- Options: set the budget to about 320 MiB (roughly five more seasons), or shard or trim further. The largest remaining lever is game logs, which repeat `match_id`, stage and opponent strings that the match index already holds.
- Host retention is a separate constraint. Keeping the active release plus the two before it (PLAN §8.1) means about 3 × 293 ≈ 880 MiB, close to GitHub Pages' 1 GB site limit. Retaining one previous release, or deduplicating unchanged files across releases, would be needed before enabling Pages.

## Reproduce

```bash
# legacy side (scratch worktree; commit/push stubbed; scrape stubbed; network dead-ended)
SMOKE_WORKTREE_DIR=<scratch>/wt SMOKE_SKIP_SCRAPE=1 FINALS_MODE=1 HTTPS_PROXY=http://127.0.0.1:9 \
  HTTP_PROXY=http://127.0.0.1:9 bash scripts/smoke_harness.sh
uv run --locked python docs/rewrite/evidence/rehearsal_compare.py <scratch>/wt/<stamp> <scratch>/new out.json
uv run --locked python docs/rewrite/evidence/rehearsal_publish_restore.py --output-root dist/real \
  --data-root var/real --work <scratch>/publish --first <id> --second <id> --unvalidated <failed id>
uv run --locked python docs/rewrite/evidence/season_growth.py <built-site> out.json
```
