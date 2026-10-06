#!/usr/bin/env bash
# Numeric weekly candidate. Not scheduled. The legacy weekly_refresh.sh execs
# this only when SCVIA_NUMERIC_ENTRY=1. Commit and push are not invoked.
#
# SCVIA_SOURCE_MODE=production  refresh sources (--allow-network). Also requires
#                               SCVIA_ALLOW_NETWORK=1 so a rehearsal cannot fetch.
# SCVIA_SOURCE_MODE=rehearsal   default. Import SCVIA_CAPTURED_SOURCE only.
#                               Requires SCVIA_CORRECTION_EVIDENCE_ROOT and an
#                               immutable SCVIA_CORRECTION_EVIDENCE_SNAPSHOT.
#                               No network. Corrections run before forecast/build.
# SCVIA_SKIP_SITE=1             command-order tests only. The scratch smoke must
#                               not set this: it builds, seals and budgets the site.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -n "${SCVIA_BIN:-}" ] && [ -x "${SCVIA_BIN}" ]; then
  SCVIA=("${SCVIA_BIN}")
  PY=("$(dirname "${SCVIA_BIN}")/python")
elif [ -x "$ROOT/.venv/bin/scvia" ]; then
  SCVIA=("$ROOT/.venv/bin/scvia")
  PY=("$ROOT/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  SCVIA=(uv run --locked --group legacy --extra ml scvia)
  PY=(uv run --locked --group legacy --extra ml python)
else
  echo "scvia_weekly: no interpreter. Set SCVIA_BIN to the locked scvia." >&2
  exit 1
fi

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/usr/bin:${PATH}"
run() { "${SCVIA[@]}" "$@"; }
VAR="${SCVIA_VAR_DIR:-$ROOT/var}"
MODE="${SCVIA_SOURCE_MODE:-rehearsal}"
DATA="${SCVIA_DATA_ROOT:-$VAR/corpus}"
OUT="${SCVIA_OUTPUT_ROOT:-$VAR/corpus-dist}"
# Archived repair evidence EVIDENCE_DIR:SEASON, passed only when named. The B1 repair (2026 games of Perez, Dalton,
# Brodie) is contained in the legacy CSVs since the AFL Tables reconciliation corrections (commit 67217df40) and is no
# longer the default: replaying it on the corrected source cannot reproduce its rows and would add Dalton twice.
# A rehearsal of an older, pre-correction capture sets SCVIA_REPAIR=docs/rewrite/evidence/b1:2026.
REPAIR="${SCVIA_REPAIR:-}"
REPAIR_ARGS=()
[ -n "$REPAIR" ] && REPAIR_ARGS=(--repair "$REPAIR")
MARKER="${SCVIA_CYCLE_MARKER:-$ROOT/.claude/audit/last_refresh_status.json}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LOCK_DIR="${SCVIA_LOCK_DIR:-$ROOT/var/run-locks}"
mkdir -p "$LOCK_DIR" "$VAR"
lock_id() { printf '%s' "$(realpath -m "$1")" | sha256sum | awk '{print $1}'; }
exec 7>"$LOCK_DIR/data-$(lock_id "$DATA").lock"
exec 8>"$LOCK_DIR/out-$(lock_id "$OUT").lock"
exec 9>"$LOCK_DIR/var-$(lock_id "$VAR").lock"
if ! flock -n 7; then
  echo "scvia_weekly: data root $DATA is locked by another numeric run" >&2
  exit 5
fi
if ! flock -n 8; then
  echo "scvia_weekly: output root $OUT is locked by another numeric run" >&2
  exit 5
fi
if ! flock -n 9; then
  echo "scvia_weekly: $VAR is locked by another numeric run" >&2
  exit 5
fi
RUN_DIR="$VAR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"
STATUS="$RUN_DIR/status.json"
POINTER="$VAR/scvia-weekly-status.json"
PHASE="init"
CUTOFF=""
SNAP=""
SOURCE_SNAP=""

write_status() {
  local phase="$1"
  local rc="${2:-}"
  python3 - "$STATUS" "$POINTER" "$MODE" "$phase" "$rc" "$RUN_ID" "$STARTED" "$DATA" "$OUT" "$CUTOFF" "$SNAP" "$SOURCE_SNAP" "$RUN_DIR" <<'PY'
import json, sys
status, pointer, mode, phase, rc, run_id, started, data_root, output_root, cutoff, snap, source_snap, evidence = sys.argv[1:]
body = {
    "run_id": run_id,
    "started": started,
    "mode": mode,
    "phase": phase,
    "exit_code": None if rc == "" else int(rc),
    "data_root": data_root,
    "output_root": output_root,
    "forecast_cutoff": cutoff or None,
    "snapshot_id": snap or None,
    "source_snapshot_id": source_snap or None,
    "evidence_dir": evidence,
}
text = json.dumps(body) + "\n"
for path in (status, pointer):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
PY
}
finish() {
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    write_status "$PHASE" "$rc"
  fi
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
PHASE="init"
write_status "$PHASE"

python3 - "$MARKER" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"scvia_weekly: cycle marker is missing: {path}")
body = json.loads(path.read_text())
if body.get("exit_code") is None:
    raise SystemExit("scvia_weekly: cycle marker has no exit_code; refusing to start")
PY
# A running cycle is bash EXECUTING a harness script, not any command line that mentions one: the old substring
# match counted `git add scripts/weekly_refresh.sh && git commit` as a cycle and broke the pre-commit hook.
HARNESS_PROC_RE='^([^ ]*/)?bash( +-[^ ]+)* +([^ ]*/)?(scripts/weekly_refresh|refresh_and_rank)\.sh( |$)'
if pgrep -f "$HARNESS_PROC_RE" >/dev/null; then
  echo "scvia_weekly: a weekly_refresh or refresh_and_rank process is running" >&2
  exit 2
fi

field() {
  python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("outputs",{}).get(sys.argv[1]) or "")' "$1"
}

if [ "$MODE" = "production" ]; then
  CUTOFF="${SCVIA_FORECAST_CUTOFF:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
elif [ "$MODE" = "rehearsal" ]; then
  if [ -z "${SCVIA_FORECAST_CUTOFF:-}" ]; then
    echo "scvia_weekly: rehearsal requires SCVIA_FORECAST_CUTOFF" >&2
    exit 2
  fi
  CUTOFF="$SCVIA_FORECAST_CUTOFF"
fi

if [ "$MODE" = "production" ]; then
  if [ "${SCVIA_ALLOW_NETWORK:-0}" != "1" ]; then
    echo "scvia_weekly: production refresh needs SCVIA_ALLOW_NETWORK=1" >&2
    exit 2
  fi
  PHASE="refresh"
  write_status "$PHASE"
  run refresh --data-only --allow-network --data-root "$DATA" --json > "$RUN_DIR/refresh.json"
elif [ "$MODE" = "rehearsal" ]; then
  if [ -z "${SCVIA_CAPTURED_SOURCE:-}" ]; then
    echo "scvia_weekly: rehearsal requires SCVIA_CAPTURED_SOURCE (no live fetch)" >&2
    exit 2
  fi
  if [ ! -d "${SCVIA_CAPTURED_SOURCE}/data" ]; then
    echo "scvia_weekly: SCVIA_CAPTURED_SOURCE must be a tree whose data/ directory is the captured input" >&2
    exit 2
  fi
  if [ -z "${SCVIA_CORRECTION_EVIDENCE_ROOT:-}" ] || [ -z "${SCVIA_CORRECTION_EVIDENCE_SNAPSHOT:-}" ]; then
    echo "scvia_weekly: rehearsal requires SCVIA_CORRECTION_EVIDENCE_ROOT and SCVIA_CORRECTION_EVIDENCE_SNAPSHOT" >&2
    exit 2
  fi
  if [[ ! "$SCVIA_CORRECTION_EVIDENCE_SNAPSHOT" =~ ^sha256:[0-9a-f]{64}$ ]]; then
    echo "scvia_weekly: correction evidence snapshot must be an immutable sha256:<id>" >&2
    exit 2
  fi
  PHASE="import"
  write_status "$PHASE"
  run import-legacy --source "$SCVIA_CAPTURED_SOURCE" --data-root "$DATA" "${REPAIR_ARGS[@]}" --json \
    > "$RUN_DIR/import.json"
else
  echo "scvia_weekly: SCVIA_SOURCE_MODE must be production or rehearsal" >&2
  exit 2
fi

if [ "$MODE" = "production" ]; then
  SOURCE_JSON="$RUN_DIR/refresh.json"
else
  SOURCE_JSON="$RUN_DIR/import.json"
fi
snapshot_field() {
  python3 - "$1" <<'PY'
import json, re, sys
try:
    body = json.load(open(sys.argv[1]))
    snap = body.get("snapshot_id")
    if (body.get("ok") is not True or body.get("exit_code", 0) != 0
            or not isinstance(snap, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", snap)):
        raise ValueError("unsuccessful result or missing snapshot id")
except (OSError, ValueError, AttributeError) as exc:
    raise SystemExit(f"scvia_weekly: step did not return a successful snapshot id: {exc}")
print(snap)
PY
}
SOURCE_SNAP="$(snapshot_field "$SOURCE_JSON")"
SNAP="$SOURCE_SNAP"

PHASE="corrections"
write_status "$PHASE"
correction=(apply-corrections --season "${CUTOFF:0:4}" --data-root "$DATA" --json)
if [ "$MODE" = "rehearsal" ]; then
  correction+=(--evidence-root "$SCVIA_CORRECTION_EVIDENCE_ROOT" --evidence-snapshot "$SCVIA_CORRECTION_EVIDENCE_SNAPSHOT")
fi
run "${correction[@]}" > "$RUN_DIR/corrections.json"
CORRECTED_SNAP="$(snapshot_field "$RUN_DIR/corrections.json")"
SNAP="$CORRECTED_SNAP"

PHASE="forecast"
write_status "$PHASE"
run forecast --train-cutoff 2025-06-01 --calibration-end 2026-05-01 --cutoff "$CUTOFF" \
  --snapshot "$SNAP" --data-root "$DATA" --json > "$RUN_DIR/forecast.json"
BUNDLE="$(field bundle_id < "$RUN_DIR/forecast.json")"
PRED="$(field prediction_dir < "$RUN_DIR/forecast.json")"

PHASE="build"
write_status "$PHASE"
build=(build-release --snapshot "$SNAP" --editorial off --data-root "$DATA" --output-root "$OUT" --json)
if [ -n "$BUNDLE" ]; then build+=(--bundle "$BUNDLE"); fi
if [ -n "$PRED" ]; then build+=(--predictions "$PRED"); fi
if [ -f "$ROOT/config/public_content.toml" ]; then
  build+=(--content-manifest "$ROOT/config/public_content.toml" --content-root "$ROOT")
fi
run "${build[@]}" > "$RUN_DIR/build.json"
RID="$(field release_id < "$RUN_DIR/build.json")"
if [ -z "$RID" ]; then
  echo "scvia_weekly: build did not return a release id" >&2
  exit 1
fi

if [ "${SCVIA_SKIP_SITE:-0}" = "1" ]; then
  PHASE="validate"
  write_status "$PHASE"
  run validate-release --release "$RID" --output-root "$OUT" --json > "$RUN_DIR/validate.json"
else
  PHASE="site"
  write_status "$PHASE"
  export SCVIA_RELEASE_DIR="$OUT/releases/$RID/public"
  export SCVIA_OUT_DIR="$OUT/releases/$RID/site"
  export SCVIA_PUBLIC_BASE="${SCVIA_PUBLIC_BASE:-/}"
  npm --prefix "$ROOT/web" run build
  npm --prefix "$ROOT/web" run budget -- --dir "$SCVIA_OUT_DIR" --out "$RUN_DIR/budget.json"
  PHASE="seal"
  write_status "$PHASE"
  run seal-site --release "$RID" --output-root "$OUT" --json > "$RUN_DIR/seal.json"
  run validate-release --release "$RID" --output-root "$OUT" --json > "$RUN_DIR/validate.json"
fi

if [ -n "${SCVIA_LEGACY_ROOT:-}" ]; then
  PHASE="compare"
  write_status "$PHASE"
  "${PY[@]}" "$ROOT/docs/rewrite/evidence/rehearsal_compare.py" --regenerate \
    "$SCVIA_CAPTURED_SOURCE" "$RUN_DIR/legacy-exports" "${REPAIR_ARGS[@]}" \
    --correction-data-root "$DATA" --source-snapshot "$SOURCE_SNAP" --corrected-snapshot "$SNAP" \
    > "$RUN_DIR/legacy-regenerate.json"
  "${PY[@]}" "$ROOT/docs/rewrite/evidence/rehearsal_compare.py" --promoted \
    "$RUN_DIR/legacy-exports" "$DATA" "$OUT/releases/$RID/public" "$RUN_DIR/compare.json" \
    --snapshot "$SNAP" > "$RUN_DIR/compare.stdout"
fi

if [ -n "${SCVIA_LOCAL_DEST:-}" ]; then
  echo "scvia_weekly: SCVIA_LOCAL_DEST is ignored. Publish two sealed releases from the smoke, not this run." >&2
fi

PHASE="complete"
write_status "$PHASE" "0"
trap - ERR
echo "scvia_weekly: release $RID validated ($MODE)"
