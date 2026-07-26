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

# Lock path is overridable so a NESTED invocation cannot deadlock against the
# outer one. The pre-commit hook now runs the test suite, and two tests drive this
# very wrapper; with a single hard-coded lock they blocked forever on the lock the
# commit that invoked them was already holding — the commit simply hung. The hook
# hands the suite a throwaway lock path, so serialisation still applies to real
# pipeline writers while nested test invocations are independent.
COUNCIL_GIT_LOCK="${COUNCIL_GIT_LOCK:-/tmp/supercoach-git.lock}"
exec flock "$COUNCIL_GIT_LOCK" git "$@"
