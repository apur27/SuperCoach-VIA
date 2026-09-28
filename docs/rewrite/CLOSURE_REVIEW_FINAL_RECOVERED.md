**Local closure: PASS.** The two earlier blockers are closed on the current tree. No production activation is implied.

## 1. Final smoke ran the final comparison gate

`/tmp/scvia-scratch-smoke-20260927c` still matches the writer tree on the smoke inventory. Recomputing `compare_roots` from the current `docs/rewrite/evidence/source_inventory.py` gives **857 files, 18,310,694 bytes, sha256 `fb871edaad740cd19eca046142328127747d4843f552dc5eff637d8b8b0dc38e`, 0 differences**. That is the digest in `closure-finish-01/scratch-smoke-result.json` and in the smoke log. `rehearsal_compare.py` and `source_inventory.py` in the scratch checkout are byte-identical to the writer copies, so the compare step used this comparison module, not the earlier one.

From the scratch release itself, not the summary:

- Verdict `ok: true`, reasons empty. All-time delta 0, 100/100, same order. Yearly 130/130 exact, no missing season. Biography `content_matches: true`.
- Selected snapshot `sha256:55e295f184c49096c4c8c783e6d7d59f416ca70e108ce0ca7f186f16157fb09c` matches the release manifest and `current_matches_selected`.
- Both weekly statuses are exit 0, phase `complete`. Forecast on the first release is `unavailable` / `no_valid_future_fixture`.
- Seals differ and both validation records are PASS: `37854d87…` (`20260927T130550Z-d424a18cd169`) and `5243945f…` (`20260927T131007Z-fd89198bce47`).
- Injected copy failure left live on the first release with `bytes_unchanged: true`. The host symlink is `releases/20260927T130550Z-d424a18cd169`, and its `index.html` matches that release’s site file.

The earlier `/tmp/scvia-scratch-smoke-20260927b` run is historical and is not a blocker.

## 2. Inventory covers the executed source

`source_inventory.py` walks the filesystem, so untracked files under the selected prefixes are included. It skips `node_modules`, `dist`, `.astro`, `.venv`, caches, and bytecode. The recomputed set includes `src/`, `scripts/`, `web/` source and integrations, `web/package-lock.json`, `config/`, `uv.lock`, `pyproject.toml`, `.node-version`, `web/.node-version`, `.python-version`, root `*.py`/`*.sh` helpers including `top_players_comprehensive.py`, `.github/` and `.githooks/`, `docs/rewrite/evidence/` including B1 `rows.jsonl`, and all 84 article paths named by both public-content manifests. No cache path is in the 857.

`SWITCH_PLAN.md`, `CURSOR_PROGRESS.md`, and `IMPLEMENTATION_STATUS.md` were touched at 23:14, after the smoke. They are outside this inventory. A fresh compare after those edits is still 0 differences, so they do not invalidate the smoke digest.

The parent’s 838-file check (`parent-source-after-final-smoke.json`, 13:14:15Z) also reports `scratch_differences: []`. Its `missing_source_paths` are the deleted legacy demo fixture names, absent from both trees. The replacement `k.*` fixtures are what the 857-file walk hashes. That list is not a scratch mismatch.

## Other checks

Node is `22.23.3` in `.node-version`, `web/.node-version`, `closure-finish-01/node.txt`, and the `scvia-ci.yml` web job, which still uses `node-version-file: ".node-version"`. `parent-node-availability.md` matches the official v22.23.3 tarball. `parent-gen-types-check.json` ran on `v22.23.3` while the pin file still said 22.23.2; the pin was then moved to the runtime that was actually used. Contract generation does not depend on that pin file.

Hermetic evidence in `closure-finish-01/hermetic-xdist.txt` is **690 passed, 127 warnings, 28.64s**, above the previous 688, so tests were added rather than dropped. `parent-integration-01.log` is **25 passed, 688 deselected, 234.56s** from before those two extra hermetic tests. I did not re-run either suite. The 28.64s run cannot have included those real-corpus tests.

Parent growth for this smoke’s first site supersedes the handoff’s “not recomputed” sentence: **276,795,037 bytes**, projected **280.438926 MiB** after three seasons plus 5 MiB, under the 282 MiB headroom line and the 300 MiB budget. `SWITCH_PLAN.md` still marks the switch as not activated and leaves two shadow cycles, the 2026 grand final, and the section 6 owner decisions outside this local pass.

I did not re-run Playwright, the wheel demo, pack/verify, or the smoke. Pages deployment stays uninvoked.
