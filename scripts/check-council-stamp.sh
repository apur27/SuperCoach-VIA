#!/usr/bin/env bash
#
# check-council-stamp.sh — council-pipeline provenance gate (harness Gap 1 + Gap 4)
#
# Every council-authored doc (news articles under docs/news/, Hall of Fame stat
# pages docs/hall-of-fame-stat-*.md) must carry a `<!-- council-pipeline: ... -->`
# provenance stamp before the commit is accepted. The stamp records that each tier
# of the six-agent council ran, and that the two gating tiers — DataSentinel and
# Skeptic — returned PASS (not FAIL / BLOCK).
#
# This is a DETERMINISTIC check: it greps the staged file content, it does not
# invoke any LLM. The LLM verdict (DataSentinel/Skeptic) is recorded in the stamp
# upstream; this script only enforces that the recorded verdict is PASS.
#
# Usage:
#   scripts/check-council-stamp.sh <file.md> [<file.md> ...]
#   git diff --cached --name-only | scripts/check-council-stamp.sh   # via xargs/stdin
#   scripts/check-council-stamp.sh --dry-run <files...>              # report, never fail (CI)
#
# Exit codes:
#   0  all council-authored files carry a valid PASS stamp (or no such files staged)
#   1  one or more council-authored files failed the check (suppressed under --dry-run)
#
set -uo pipefail

DRY_RUN=0
FILES=()

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *)         FILES+=("$arg") ;;
  esac
done

# Allow files on stdin too (e.g. `git diff --cached --name-only | check-council-stamp.sh`)
if [ "${#FILES[@]}" -eq 0 ] && [ ! -t 0 ]; then
  while IFS= read -r line; do
    [ -n "$line" ] && FILES+=("$line")
  done
fi

# Does this path require a council-pipeline stamp?
#   docs/news/*.md                 -> yes (news desk articles)
#   docs/hall-of-fame-stat-*.md    -> yes (Hall of Fame stat pages, by convention)
#   docs/hall-of-fame-*.md that
#     already carry a council marker -> yes (a council doc may not silently drop
#                                            its stamp; new council pages such as
#                                            forgotten-heroes opt in by carrying one)
#
# Legacy Hall of Fame pages without a stamp are NOT retroactively blocked — that
# would only train contributors to bypass the hook. Enforcement is opt-in on first
# stamp, then sticky.
# Everything else (README, CSV, Python, briefs, other docs) is skipped.
requires_stamp() {
  local f="$1"
  case "$f" in
    docs/news/*.md)              return 0 ;;
    docs/hall-of-fame-stat-*.md) return 0 ;;
    docs/hall-of-fame-*.md)
      # Require a stamp only if the file already declares itself a council doc.
      [ -f "$f" ] && grep -q '<!-- council-pipeline:' "$f" && return 0
      return 1
      ;;
    *)                           return 1 ;;
  esac
}

FAILED=0
CHECKED=0
SKIPPED=0

for f in "${FILES[@]}"; do
  if ! requires_stamp "$f"; then
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  # Operational meta-docs inside docs/news/ are not council articles — skip them.
  case "$f" in
    docs/news/README.md)              SKIPPED=$((SKIPPED + 1)); continue ;;
    docs/news/sentinel-log-*.md)      SKIPPED=$((SKIPPED + 1)); continue ;;
    docs/news/council-meeting-*.md)   SKIPPED=$((SKIPPED + 1)); continue ;;
  esac

  CHECKED=$((CHECKED + 1))

  if [ ! -f "$f" ]; then
    # Staged deletion or rename target that no longer exists — nothing to check.
    continue
  fi

  # 1. The stamp block must exist.
  if ! grep -q '<!-- council-pipeline:' "$f"; then
    echo "ERROR: $f is missing the <!-- council-pipeline: ... --> provenance stamp." >&2
    echo "       Council-authored docs must record the six-agent chain before commit." >&2
    FAILED=$((FAILED + 1))
    continue
  fi

  file_failed=0

  # 2. DataSentinel line must contain PASS.
  ds_line="$(grep -i 'DataSentinel:' "$f" | head -1)"
  if [ -z "$ds_line" ]; then
    echo "ERROR: $f stamp is missing a DataSentinel: line." >&2
    file_failed=1
  elif ! printf '%s' "$ds_line" | grep -qi 'PASS'; then
    echo "ERROR: $f DataSentinel verdict is not PASS ->$ds_line" >&2
    file_failed=1
  fi

  # 3. Skeptic line must contain PASS.
  sk_line="$(grep -i 'Skeptic:' "$f" | head -1)"
  if [ -z "$sk_line" ]; then
    echo "ERROR: $f stamp is missing a Skeptic: line." >&2
    file_failed=1
  elif ! printf '%s' "$sk_line" | grep -qi 'PASS'; then
    echo "ERROR: $f Skeptic verdict is not PASS ->$sk_line" >&2
    file_failed=1
  fi

  if [ "$file_failed" -ne 0 ]; then
    FAILED=$((FAILED + 1))
  else
    echo "OK: $f carries a valid council-pipeline PASS stamp."
  fi
done

echo "council-stamp check: ${CHECKED} council doc(s) checked, ${SKIPPED} skipped, ${FAILED} failed."

if [ "$FAILED" -ne 0 ]; then
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "(--dry-run: reporting only, not blocking the commit.)"
    exit 0
  fi
  echo "Commit blocked: fix the stamp(s) above, or run the council chain to PASS first." >&2
  exit 1
fi

exit 0
