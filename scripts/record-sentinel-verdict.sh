#!/usr/bin/env bash
#
# record-sentinel-verdict.sh — write a machine-verifiable DataSentinel audit record.
#
# This is the PRODUCER side of the stamp-verifiability gate (harness Q1). DataSentinel
# calls this once it has verified a full doc, BEFORE Gaffer stamps it. It records the
# canonical content hash + verdict to .claude/audit/sentinel-<hash>-<ts>.json. The
# pre-commit hook (scripts/check-council-stamp.sh) later recomputes the same hash and
# refuses to trust a `DataSentinel: PASS` stamp that has no matching PASS record.
#
# This closes the forgery vector: the stamp is text an LLM can type; the audit record
# is keyed on the content hash, so a stamp that was never earned (or was earned on
# different content) fails the cross-check.
#
# Usage:
#   scripts/record-sentinel-verdict.sh --doc <path> --verdict <VERDICT> [--agent <id>]
#
# CANONICAL VERDICT VOCABULARY (F07 — one enum across DataSentinel, Skeptic, QA):
#   PASS                 — clears the gate.
#   PASS_WITH_CONCERNS   — Skeptic: clears, caveats logged (Gaffer records them in the retro).
#   PASS_WITH_WARNINGS   — QA: clears, warnings logged.
#   FAIL                 — DataSentinel/QA: halts the ship.
#   BLOCK                — Skeptic: halts the ship.
# Per-agent subset: DataSentinel {PASS,FAIL}; Skeptic {PASS,PASS_WITH_CONCERNS,BLOCK};
# QA {PASS,PASS_WITH_WARNINGS,FAIL}. The pre-commit stamp gate (check-council-stamp.sh)
# trusts only an exact "PASS" DataSentinel record; the clearing PASS_WITH_* verdicts are
# recorded for audit trail and Gaffer-side retro logging, not stamp-gate enforcement.
# Skeptic records its verdict here (--agent Skeptic) so both gates are auditable.
#
# Env:
#   COUNCIL_AUDIT_DIR   override the audit directory (default: <repo>/.claude/audit)
#
set -euo pipefail

doc=""
verdict=""
agent="DataSentinel"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --doc)     doc="${2:-}";     shift 2 ;;
    --verdict) verdict="${2:-}"; shift 2 ;;
    --agent)   agent="${2:-}";   shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "record-sentinel-verdict.sh: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

[[ -n "$doc" && -n "$verdict" ]] || { echo "record-sentinel-verdict.sh: --doc and --verdict are required" >&2; exit 2; }
case "$verdict" in
  PASS|FAIL|BLOCK|PASS_WITH_CONCERNS|PASS_WITH_WARNINGS) ;;
  *) echo "record-sentinel-verdict.sh: --verdict must be one of PASS|FAIL|BLOCK|PASS_WITH_CONCERNS|PASS_WITH_WARNINGS (canonical vocabulary, F07)" >&2; exit 2 ;;
esac
[ -f "$doc" ] || { echo "record-sentinel-verdict.sh: no such doc: $doc" >&2; exit 2; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
audit_dir="${COUNCIL_AUDIT_DIR:-$repo_root/.claude/audit}"
mkdir -p "$audit_dir"

hash="$("$script_dir/council-content-hash.sh" "$doc")"
ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="$audit_dir/sentinel-${hash}-${ts}.json"

printf '{"doc_path":"%s","doc_hash":"%s","verdict":"%s","ts":"%s","agent_id":"%s"}\n' \
  "$doc" "$hash" "$verdict" "$ts" "$agent" > "$out"

echo "recorded sentinel verdict: $out"
