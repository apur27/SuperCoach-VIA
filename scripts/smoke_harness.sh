#!/usr/bin/env bash
# =============================================================================
# smoke_harness.sh — run the weekly harness end-to-end in a scratch worktree,
# with commit and push stubbed, so a harness change can be validated before it
# merges (CLAUDE.md §6.2).
#
# WHY THIS EXISTS
# ---------------
# Every harness fix that broke a cycle in the week of 2026-07-27 was a few lines
# long and passed the unit suite:
#   * a gated doc regenerated after its verdict was recorded, orphaning the record;
#   * an integration gate placed before the step whose output it asserts;
#   * a re-verify hop that read the index copy while the verifier hashed the
#     worktree copy.
# None of those are visible to a unit test. They are ordering and
# staged-versus-worktree faults, and they only appear when the whole harness runs.
#
# WHAT IT DOES
#   1. Creates a git worktree at HEAD (or --ref) under a scratch directory.
#   2. Copies the current data/ and .claude/audit into it, so the run works against
#      a real snapshot rather than an empty tree.
#   3. Puts a `git` shim first on PATH that turns `commit` and `push` into no-ops
#      and passes every other git subcommand through untouched.
#   4. Runs scripts/weekly_refresh.sh inside the worktree, logging everything.
#   5. Reports the phase reached and the exit code, and leaves the worktree for
#      inspection unless --clean is passed.
#
# HONEST LIMITATIONS — read these before trusting a green run:
#   * It exercises the REAL harness, including the network scrape and the LLM
#     gates, so it is slow (tens of minutes) and not hermetic. SMOKE_SKIP_SCRAPE=1
#     stubs the scrape for a faster ordering-only check; that trades fidelity for
#     speed and is NOT sufficient for a change that touches scraping.
#   * Verdicts recorded during the run land in the WORKTREE's .claude/audit, not
#     the repo's, so a smoke run cannot clear a gate for the real tree.
#   * This script is itself harness-adjacent and cannot be smoke-tested by itself.
#     It was validated by running it.
#
# Usage:
#   scripts/smoke_harness.sh [--ref <git-ref>] [--clean]
#   SMOKE_SKIP_SCRAPE=1 scripts/smoke_harness.sh      # faster, lower fidelity
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="HEAD"
CLEAN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)   REF="${2:-HEAD}"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "smoke_harness.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WT="${SMOKE_WORKTREE_DIR:-/tmp/supercoach-smoke}/$STAMP"
LOG="$WT.log"
mkdir -p "$(dirname "$WT")"

echo "[smoke] worktree : $WT"
echo "[smoke] log      : $LOG"
echo "[smoke] ref      : $REF"

cleanup() {
  if [ "$CLEAN" = "1" ]; then
    git -C "$REPO_ROOT" worktree remove --force "$WT" 2>/dev/null || true
  fi
}
trap cleanup EXIT

git -C "$REPO_ROOT" worktree add --detach "$WT" "$REF" >/dev/null 2>&1 || {
  echo "[smoke] FATAL: could not create worktree at $WT" >&2; exit 1; }

# Apply the WORKING TREE's uncommitted changes. 6.2 exists to validate a change
# BEFORE it merges, so smoke-testing HEAD would exercise the very code the change
# replaces. That mistake was made on this tool's first run and caught immediately.
if ! git -C "$REPO_ROOT" diff --quiet HEAD 2>/dev/null; then
  if git -C "$REPO_ROOT" diff HEAD | git -C "$WT" apply --whitespace=nowarn 2>/dev/null; then
    echo "[smoke] applied uncommitted working-tree changes"
  else
    echo "[smoke] FATAL: could not apply working-tree changes; refusing to test the wrong code" >&2
    exit 1
  fi
else
  echo "[smoke] working tree clean; testing $REF as committed"
fi

# Untracked source files are part of an unmerged change too (new scripts, new tests).
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in data/*|.claude/audit/*|assets/*) continue ;; esac
  mkdir -p "$WT/$(dirname "$f")" 2>/dev/null || true
  cp -a "$REPO_ROOT/$f" "$WT/$f" 2>/dev/null || true
done < <(git -C "$REPO_ROOT" ls-files --others --exclude-standard)

# Real data snapshot — the harness is meaningless against an empty tree. Uncommitted
# working-tree data is deliberately included: that is the state the next cycle runs on.
for d in data .claude/audit; do
  if [ -d "$REPO_ROOT/$d" ]; then
    mkdir -p "$WT/$(dirname "$d")"
    cp -a "$REPO_ROOT/$d" "$WT/$(dirname "$d")/" 2>/dev/null || true
  fi
done

# git shim: commit and push become no-ops so a smoke run can never write history
# or reach origin. Everything else passes through, because the harness legitimately
# uses git to stage, diff and inspect.
SHIM="$WT/.smoke-bin"
mkdir -p "$SHIM"
REAL_GIT="$(command -v git)"
cat > "$SHIM/git" <<SHIMEOF
#!/usr/bin/env bash
for a in "\$@"; do
  case "\$a" in
    commit) echo "[smoke] git commit suppressed"; exit 0 ;;
    push)   echo "[smoke] git push suppressed";   exit 0 ;;
  esac
done
exec "$REAL_GIT" "\$@"
SHIMEOF
chmod +x "$SHIM/git"

if [ "${SMOKE_SKIP_SCRAPE:-0}" = "1" ]; then
  cat > "$WT/refresh_data.py" <<'STUBEOF'
print("[smoke] refresh_data.py stubbed (SMOKE_SKIP_SCRAPE=1) — ordering-only run")
STUBEOF
  echo "[smoke] scrape STUBBED — lower fidelity, not valid for scraper changes"
fi

echo "[smoke] starting weekly_refresh.sh ..."
( cd "$WT" && PATH="$SHIM:$PATH" bash scripts/weekly_refresh.sh ) > "$LOG" 2>&1
RC=$?

PHASE="unknown"
[ -f "$WT/.claude/audit/last_refresh_status.json" ] && \
  PHASE="$(grep -o '"phase":"[^"]*"' "$WT/.claude/audit/last_refresh_status.json" | cut -d'"' -f4)"

echo "[smoke] exit code   : $RC"
echo "[smoke] phase reached: $PHASE"
echo "[smoke] last 15 log lines:"
tail -15 "$LOG" | sed 's/^/    /'

if [ "$RC" -eq 0 ]; then
  echo "[smoke] RESULT: PASS — harness ran end to end with commit/push stubbed."
else
  echo "[smoke] RESULT: FAIL — died in phase $PHASE. Full log: $LOG" >&2
fi
exit "$RC"
