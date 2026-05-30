---
name: parallel-council-commits
description: How git races play out when 7 council agents push to origin/main simultaneously, and how to commit safely as chair
metadata:
  type: feedback
---

When chairing a council session where 6 other agents push to origin/main in parallel, expect heavy git contention on the working tree and index.

**Why:** Observed 2026-05-30 council session. HEAD advanced under me repeatedly (8064ade → 0683aec → 0da8803 → 933c9b5) within the span of one task. Symptoms seen: `git pull --rebase` aborting with "cannot pull with rebase: uncommitted/unstaged changes" even when my own scoped status looked clean; `git add` of a real on-disk file silently matching nothing ("did not match any files") because a peer's rebase/checkout was mid-flight; and `.git/index.lock` present because a peer was mid-commit. Ultimately my staged edits + new file were swept into a *peer's* bundled commit (933c9b506 "Council meeting minutes, Skeptic report, Gaffer agent def, architecture gap update") and pushed — the substance shipped correctly under a different commit message than I planned.

**How to apply:**
- Before any git index operation (`add`/`commit`), wait for the lock to clear: `until [ ! -f .git/index.lock ]; do sleep 0.3; done` run via Bash `run_in_background` (foreground `sleep` is harness-blocked). Do NOT manually `rm .git/index.lock` — that corrupts a peer's in-flight commit.
- NEVER use `git add -v` or other non-standard flags — they can trip the permission allowlist. Use the plain allowlisted `git add <paths>` form.
- Verify outcomes by content, not by command success: after the dust settles, check `git show HEAD:<file>` / `git ls-files` / `git ls-tree origin/main` to confirm your changes actually landed, rather than assuming your own commit ran. My work shipped even though my `git commit` never executed.
- Re-confirm your own edits survived each rebase with `grep -c` on the working file before staging — a peer could have touched the same gap table.
- Stage ONLY the deliberate allowlist (my docs). Untracked `??` deletions need no staging — the files are simply absent and there is no history entry to record.
