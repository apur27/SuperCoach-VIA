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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HASH_SCRIPT="$SCRIPT_DIR/council-content-hash.sh"
AUDIT_DIR="${COUNCIL_AUDIT_DIR:-$SCRIPT_DIR/../.claude/audit}"
# AUDIT_ENFORCE=1 makes a stamp with NO matching audit record a hard FAIL.
# Default (0) warns instead, so the gate is safe to land before the DataSentinel
# producer side (scripts/record-sentinel-verdict.sh) is wired into the agent.
# A record that EXISTS but disagrees (tamper / non-PASS) always fails, both modes.
AUDIT_ENFORCE="${AUDIT_ENFORCE:-0}"

# verify_stamp_against_audit <file> -> 0 verified/ok, 1 hard-fail
# Cross-checks a `DataSentinel: PASS` stamp against the content-hash-keyed audit
# records. This is what makes the stamp unforgeable in practice: the PASS text is
# only trusted if a sentinel record was written for THIS doc's current content.
verify_stamp_against_audit() {
  local f="$1"
  if [ ! -x "$HASH_SCRIPT" ]; then
    echo "NOTE: $HASH_SCRIPT missing — cannot audit-verify stamp for $f (presence check only)." >&2
    return 0
  fi
  local hash; hash="$("$HASH_SCRIPT" "$f" 2>/dev/null)"
  if [ -z "$hash" ]; then
    echo "NOTE: could not compute content hash for $f — skipping audit cross-check." >&2
    return 0
  fi

  local rec verdict rhash rpath
  local matched_pass=0 records_for_doc=0
  shopt -s nullglob
  for rec in "$AUDIT_DIR"/sentinel-*.json; do
    rpath="$(grep -o '"doc_path":"[^"]*"' "$rec" | head -1 | cut -d'"' -f4)"
    [ "$rpath" = "$f" ] || continue
    records_for_doc=$((records_for_doc + 1))
    rhash="$(grep -o '"doc_hash":"[^"]*"' "$rec" | head -1 | cut -d'"' -f4)"
    verdict="$(grep -o '"verdict":"[^"]*"' "$rec" | head -1 | cut -d'"' -f4)"
    if [ "$rhash" = "$hash" ] && [ "$verdict" = "PASS" ]; then
      matched_pass=1
    fi
  done
  shopt -u nullglob

  if [ "$matched_pass" -eq 1 ]; then
    echo "OK: $f stamp verified against audit record (content hash $hash)."
    return 0
  fi

  if [ "$records_for_doc" -gt 0 ]; then
    echo "ERROR: $f carries a DataSentinel: PASS stamp, but no audit record matches its" >&2
    echo "       current content hash ($hash). The doc changed after verification, or the" >&2
    echo "       recorded verdict was not PASS. Re-run DataSentinel on the current content." >&2
    return 1
  fi

  # No record at all for this doc.
  if [ "$AUDIT_ENFORCE" -eq 1 ]; then
    echo "ERROR: $f stamp cannot be verified — no DataSentinel audit record exists for it." >&2
    echo "       (AUDIT_ENFORCE=1) Run scripts/record-sentinel-verdict.sh via DataSentinel." >&2
    return 1
  fi
  echo "WARNING: $f stamp is unverified against the audit log (no sentinel record yet)." >&2
  echo "         The stamp is trusted on text alone until DataSentinel emits audit records." >&2
  return 0
}

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
    docs/news/page-*.md)              SKIPPED=$((SKIPPED + 1)); continue ;;
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

  # Stamp text is present and says PASS. Now cross-check it against the audit log:
  # the text alone is forgeable; a content-hash-keyed sentinel record is not.
  if [ "$file_failed" -eq 0 ]; then
    if ! verify_stamp_against_audit "$f"; then
      file_failed=1
    fi
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
