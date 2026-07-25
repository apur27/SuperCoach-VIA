---
name: shared-worktree-commit-capture
description: A concurrent session sharing the working tree can sweep YOUR uncommitted edits into ITS commit, leaving a broken intermediate HEAD
metadata:
  type: project
---

On 2026-07-14 two `claude` sessions worked the same 6-fix harness task in the SAME
working directory (one `claude --resume`, one `--resume gaffer`). While my
`weekly_refresh.sh` edits (F1 hub stamp, F2 retry, F5 Phase 0) sat uncommitted, the
OTHER session ran `git add scripts/weekly_refresh.sh` + committed (896b16e35). That
commit CAPTURED my in-flight harness lines but NOT my supporting files
(`check_round_settled.py` untracked, `check_hof_hub` uncommitted) → HEAD was BROKEN:
the committed harness called an untracked script and stamped an ungated hub (a false
provenance stamp). Recovery: commit the supporting scripts on top (1d634db75) to
restore consistency.

**Why:** a shared working tree has ONE index. Any session's broad `git add <shared file>`
captures whatever is in that file right now — including another session's uncommitted
work — while missing that work's siblings. The result is a partial, inconsistent commit
nobody intended.

**How to apply:**
- When `ps` shows another `claude`/harness process in the same repo, treat the working
  tree as contended. Do not assume your uncommitted edits are yours alone until committed.
- Detect it: after any Edit that reports "file modified on disk since you last read it",
  re-read fully and check `git log --oneline` + `git rev-parse HEAD` — HEAD may have moved.
- Diagnose with content, not diffstat: `git show HEAD:<file> | grep <your-marker>` tells you
  whether your lines were swept into someone else's commit. `git diff HEAD` alone will
  mislead (swept-in lines show as "already there").
- Recover by committing the missing siblings on top so HEAD becomes internally consistent
  (every harness reference resolves; every stamp is backed by a real gate). Verify:
  `git cat-file -e HEAD:scripts/<referenced>.py`.
- Reinforces [[feedback_parallel_council_commits]] and the "verify by content not command
  success" rule from [[feedback_flaky_output_channel]].
