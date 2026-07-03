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
# Authorise the commit for the pre-commit guard (only-Gaffer-commits protocol):
# the pre-commit hook blocks any `git commit` that does NOT carry this marker, so
# every automated commit is forced through this serialising wrapper. `export` before
# `exec` means the marker is inherited by flock -> git -> the pre-commit hook.
export COUNCIL_COMMIT_AUTHORIZED=1
exec flock /tmp/supercoach-git.lock git "$@"
