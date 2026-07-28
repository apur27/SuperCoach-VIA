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
findings_file=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --doc)     doc="${2:-}";     shift 2 ;;
    --verdict) verdict="${2:-}"; shift 2 ;;
    --agent)   agent="${2:-}";   shift 2 ;;
    --findings-file) findings_file="${2:-}"; shift 2 ;;
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

# BL-05: persist the REASONING, not just the verdict.
#
# This wrote five scalar fields and nothing else, so the findings behind a FAIL or
# BLOCK survived only in the invoking agent's stdout. On 2026-07-26 a Skeptic BLOCK
# carrying four findings was read as one; three were never routed and a full gate
# cycle was spent rediscovering them. No audit trail could have caught it, because a
# one-finding and a four-finding BLOCK record were byte-identical. A separate BLOCK
# lost its findings entirely to a truncated pipe.
#
# Assembled in Python, not printf: findings quote the document under review, so they
# carry quotes, newlines and unicode that shell interpolation would corrupt into
# invalid JSON — and a corrupt audit record is worse than none, because the gate
# reads it back.
PYBIN="${COUNCIL_PYTHON:-/home/abhi/sourceCode/python/coding/.venv/bin/python}"
[ -x "$PYBIN" ] || PYBIN="$(command -v python3)"

"$PYBIN" - "$out" "$doc" "$hash" "$verdict" "$ts" "$agent" "$findings_file" \
         "${COUNCIL_REQUIRE_FINDINGS:-0}" <<'PYEOF'
import json, sys
out, doc, hash_, verdict, ts, agent, findings_file, require = sys.argv[1:9]

findings = []
if findings_file:
    try:
        with open(findings_file, encoding="utf-8") as f:
            findings = json.load(f)
        if not isinstance(findings, list):
            raise ValueError("findings must be a JSON array")
    except Exception as exc:
        # Fail closed: never write a record derived from unparseable findings.
        sys.exit(f"record-sentinel-verdict.sh: cannot read --findings-file "
                 f"{findings_file}: {exc}")

blocking = verdict in ("FAIL", "BLOCK")
if blocking and not findings and require == "1":
    sys.exit("record-sentinel-verdict.sh: COUNCIL_REQUIRE_FINDINGS=1 and a "
             f"{verdict} verdict carries no findings — refusing to record an "
             "unactionable verdict. Pass --findings-file.")

record = {
    "doc_path": doc, "doc_hash": hash_, "verdict": verdict, "ts": ts,
    "agent_id": agent, "finding_count": len(findings), "findings": findings,
}
with open(out, "w", encoding="utf-8") as f:
    # Compact separators: the on-disk format is grepped by check-council-stamp.sh
    # and asserted by existing tests as `"verdict":"PASS"`. Default json.dump
    # separators would insert spaces and silently break both.
    json.dump(record, f, ensure_ascii=False, separators=(",", ":"))
    f.write("\n")

print(f"recorded sentinel verdict: {out}")
print(f"  {verdict} with {len(findings)} finding(s)")
if blocking and not findings:
    print(f"  WARNING: {verdict} recorded with no findings — unactionable. "
          f"Whoever routes this cannot know what to fix.", file=sys.stderr)
PYEOF
