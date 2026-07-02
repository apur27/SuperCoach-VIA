#!/usr/bin/env bash
#
# git_commit_safe.sh — flock-wrapped git, so concurrent pipeline steps cannot race
# on the index (harness Q6). All automated commits MUST go through this wrapper
# instead of raw `git commit`, so only one writer touches .git at a time.
#
# The lock serialises the whole git invocation; a second caller blocks until the
# first releases, rather than colliding on index.lock.
#
# Usage (drop-in for git): scripts/git_commit_safe.sh commit -m "msg"
#                          scripts/git_commit_safe.sh add -A
#
exec flock /tmp/supercoach-git.lock git "$@"
