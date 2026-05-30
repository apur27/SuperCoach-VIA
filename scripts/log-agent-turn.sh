#!/usr/bin/env bash
#
# log-agent-turn.sh — append a structured JSON audit line for an agent turn.
#
# Any agent or harness step can call this to record what it did. Output goes
# to .claude/audit/YYYY-MM-DD.jsonl (one JSON object per line). These logs are
# operational and are gitignored; only .claude/audit/.gitkeep is committed.
#
# Usage:
#   scripts/log-agent-turn.sh \
#     --agent   <name> \
#     --action  <action> \
#     --files   <comma,separated,paths> \
#     --verdict <PASS|FAIL|BLOCK|DONE|NOTE>
#
# Example:
#   scripts/log-agent-turn.sh --agent scientist --action "gap2-build" \
#     --files "scripts/log-agent-turn.sh,.gitignore" --verdict DONE
#
# Pure bash, no dependencies.

set -euo pipefail

agent=""
action=""
files=""
verdict=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent)   agent="${2:-}";   shift 2 ;;
    --action)  action="${2:-}";  shift 2 ;;
    --files)   files="${2:-}";   shift 2 ;;
    --verdict) verdict="${2:-}"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "log-agent-turn.sh: unknown argument '$1'" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$agent" || -z "$action" || -z "$verdict" ]]; then
  echo "log-agent-turn.sh: --agent, --action and --verdict are required" >&2
  exit 2
fi

case "$verdict" in
  PASS|FAIL|BLOCK|DONE|NOTE) ;;
  *)
    echo "log-agent-turn.sh: --verdict must be one of PASS|FAIL|BLOCK|DONE|NOTE (got '$verdict')" >&2
    exit 2
    ;;
esac

# Resolve repo root from this script's location so it works from any cwd.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
audit_dir="$repo_root/.claude/audit"
mkdir -p "$audit_dir"

out_file="$audit_dir/$(date +%Y-%m-%d).jsonl"
ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Escape a string for safe embedding in a JSON value: backslash, double-quote,
# then control chars (tab, CR, LF).
json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//	/\\t}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\n'/\\n}"
  printf '%s' "$s"
}

# Build the files array. Empty --files yields [].
files_json="[]"
if [[ -n "$files" ]]; then
  files_json="["
  first=1
  IFS=',' read -ra _arr <<< "$files"
  for f in "${_arr[@]}"; do
    # trim surrounding whitespace
    f="${f#"${f%%[![:space:]]*}"}"
    f="${f%"${f##*[![:space:]]}"}"
    [[ -z "$f" ]] && continue
    if [[ $first -eq 1 ]]; then first=0; else files_json+=","; fi
    files_json+="\"$(json_escape "$f")\""
  done
  files_json+="]"
fi

printf '{"ts":"%s","agent":"%s","action":"%s","files":%s,"verdict":"%s"}\n' \
  "$ts" \
  "$(json_escape "$agent")" \
  "$(json_escape "$action")" \
  "$files_json" \
  "$verdict" \
  >> "$out_file"
