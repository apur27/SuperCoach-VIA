---
name: afltables-acceptance-retro
description: 2026-10-06 AFL Tables reconciliation acceptance/ship retro — pgrep self-match breaks hook tests, worktree generator runs degrade provenance, QA caught stale banner after player deletions
metadata:
  type: project
---

Shipped 2026-10-06 (0f9a3fe90 code, 67217df40 data, 11259bcd0 banner/README, 1add8cdc5 acceptance, ad217be10 Scientist memory). Accepted WITH CONDITIONS; data verdict stays UNKNOWN.

- **Commit messages that name a harness script (e.g. weekly_refresh.sh) and go through a shell heredoc break test_weekly_candidate inside the hook.** scvia_weekly.sh's `pgrep -af` matches the calling shell's command line. **How to apply:** always use `commit -F <file>`.
- **Don't ship eval-surface generator output from a worktree.** Untracked backtest run logs exist only in the main checkout, so every round's provenance falls to "not attested". Keep only diffs that are count-only and gate-verified.
- **A change that deletes or adds player files makes the banner/README file count stale.** QA's integration tier catches this; regenerate the count before ship.
- **The auto-mode classifier denies `git checkout -B` over a dirty tree.** Commit first, then `git rebase origin/main`.
- Open queue (H1 gate skip logs "passed", H2 gate path never run in-cycle, M2, M4, M5, chart reproducibility) is in docs/reviews/AFLTABLES_RECONCILIATION_ACCEPTANCE.md. See [[open-backlog]].
