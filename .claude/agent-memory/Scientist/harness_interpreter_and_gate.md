---
name: harness-interpreter-and-gate
description: Weekly harness interpreter now resolved by scripts/harness_env.sh (repo .venv), the generic lib/ gitignore trap, eval-surface vintage-by-filename, and the reconciliation gate's diff-vs-working-tree quirks
metadata:
  type: project
---

- The harness hard-coded /home/abhi/sourceCode/python/coding/.venv/bin/python and /home/abhi/.claude/local/claude; both vanished in an OS upgrade (2026-10). `scripts/harness_env.sh` (sourced) resolves SUPERCOACH_PYTHON else `$REPO_ROOT/.venv` (build: `uv sync --locked --group dev --group legacy --extra ml`) and CLAUDE else PATH. CLAUDE.md still names the old path for pytest commands — use the repo .venv.
  **Why:** eval-surface tests were `skipif` on the dead path and silently skipped for weeks.
  **How to apply:** a skip count jump is a finding; grep tests for hard-coded interpreter paths.
- `.gitignore` has a generic `lib/` rule: any new `scripts/lib/...` file is silently ignored (not committed, not copied by smoke_harness.sh). Check `git check-ignore -v` for every new file before handoff.
- update_eval_surface.sh selects backtest vintages by the run timestamp in the FILENAME (was mtime; fresh checkout scrambles mtimes).
- scripts/reconciliation_gate.py diffs `--base` (default origin/main, harness env RECON_GATE_BASE) against the WORKING TREE, so smoke runs (commit suppressed, rows staged) see the same seasons. Gotcha: `git diff <tree>` shows files untracked in the index as DELETED, so run it only after `git add` (Phase 1 does). Needs RECON_DATA_ROOT (default var/finalized/data, untracked); absent -> WARN skip. A season ~1,000 polite requests (~35-40 min).

Related: [[afltables-reconciliation-run-lessons]], [[feedback-backtest-rules]]
- Smoke-run lessons (2026-10-06, 4 runs to green): post-finals `matches_<yr>.csv` has named rounds ("Elimination Final") — any int(round_num) crashes; backtest vintage selection must be by FILENAME timestamp in EVERY consumer (update_eval_surface.sh, update_team_analysis.generate_backtest_section, tests/integration/test_published_artifacts.py) or DataSentinel/Phase 3d disagree; README agent table: Codex is named in EXTERNAL_AGENTS (owner decision), don't key on the model column. Don't run the scvia fast tier while a weekly_refresh is running: scvia_weekly.sh's guard fails 19 test_weekly_candidate tests. smoke_harness.sh copies .claude/audit from its OWN repo — from a worktree that's only 11 committed records; use the main checkout's.
