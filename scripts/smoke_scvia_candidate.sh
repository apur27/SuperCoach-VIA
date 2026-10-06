#!/usr/bin/env bash
# Isolated smoke of the numeric candidate. Offline captured inputs only.
# Publishes two different sealed releases to a temporary local host, rejects a
# bad publish without moving the live pointer, then rolls back to the earlier
# release. Does not edit weekly_refresh.sh, cron, hooks, or a remote host.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${SCVIA_SMOKE_ROOT:-/tmp/scvia-candidate-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
HOST="$RUN/host"
BIN="${SCVIA_BIN:-$ROOT/.venv/bin/scvia}"
SOURCE="${SCVIA_CAPTURED_SOURCE:-$ROOT}"
PY="$(dirname "$BIN")/python"

python3 - "$ROOT/.claude/audit/last_refresh_status.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"cycle marker missing: {path}")
body = json.loads(path.read_text())
if body.get("exit_code") is None:
    raise SystemExit(f"cycle marker has no exit_code: {body}")
print(f"cycle marker exit_code={body['exit_code']} phase={body.get('phase')}")
PY

# bash EXECUTING a harness script, not any command line mentioning one (same pattern as scvia_weekly.sh)
HARNESS_PROC_RE='^([^ ]*/)?bash( +-[^ ]+)* +([^ ]*/)?(scripts/weekly_refresh|refresh_and_rank)\.sh( |$)'
if pgrep -f "$HARNESS_PROC_RE" >/dev/null; then
  echo "smoke: a weekly_refresh or refresh_and_rank process is running" >&2
  exit 1
fi

if [ "${SCVIA_SKIP_SITE:-0}" = "1" ]; then
  echo "smoke: SCVIA_SKIP_SITE must not be set" >&2
  exit 2
fi
if [ -e "$RUN" ]; then
  echo "smoke: $RUN already exists; set SCVIA_SMOKE_ROOT to a new path" >&2
  exit 2
fi
mkdir -p "$RUN"
SCRATCH="$RUN/scratch"
mkdir "$SCRATCH"
rsync -a \
  --exclude .venv --exclude var --exclude node_modules --exclude web/node_modules \
  --exclude web/dist --exclude web/.astro --exclude .git \
  --exclude web/test-results --exclude web/playwright-report \
  "$ROOT/" "$SCRATCH/"
ln -s "$ROOT/web/node_modules" "$SCRATCH/web/node_modules"
export PYTHONPATH="$SCRATCH/src${PYTHONPATH:+:$PYTHONPATH}"

one() {
  local name="$1"
  local compare="${2:-}"
  mkdir "$RUN/$name"
  env -u SCVIA_ALLOW_NETWORK -u SCVIA_LOCAL_DEST -u SCVIA_SKIP_SITE \
    PYTHONPATH="$SCRATCH/src${PYTHONPATH:+:$PYTHONPATH}" \
    SCVIA_SOURCE_MODE=rehearsal \
    SCVIA_CAPTURED_SOURCE="$SOURCE" \
    SCVIA_BIN="$BIN" \
    SCVIA_NUMERIC_ENTRY=1 \
    SCVIA_DATA_ROOT="$RUN/$name/data" \
    SCVIA_OUTPUT_ROOT="$RUN/$name/dist" \
    SCVIA_VAR_DIR="$RUN/$name/var" \
    SCVIA_LOCK_DIR="$RUN/locks" \
    SCVIA_LEGACY_ROOT="$compare" \
    SCVIA_CYCLE_MARKER="$ROOT/.claude/audit/last_refresh_status.json" \
    SCVIA_FORECAST_CUTOFF="${SCVIA_FORECAST_CUTOFF:-2026-09-25T00:00:00Z}" \
    bash "$SCRATCH/scripts/weekly_refresh.sh" > "$RUN/$name/wrapper.log" 2> "$RUN/$name/wrapper.err" &
  local child=$!
  python3 - "$child" "$RUN/$name/rss-peak-kib.txt" <<'PY' &
import os, subprocess, sys, time
root_pid, out = int(sys.argv[1]), sys.argv[2]
peak = 0

def descendants(pid: int) -> list[int]:
    try:
        text = subprocess.check_output(["ps", "-o", "pid=", "--ppid", str(pid)], text=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    kids = [int(item) for item in text.split() if item.strip()]
    found = list(kids)
    for kid in kids:
        found.extend(descendants(kid))
    return found

while True:
    try:
        os.kill(root_pid, 0)
    except OSError:
        break
    total = 0
    for pid in [root_pid, *descendants(root_pid)]:
        try:
            text = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
        except (OSError, subprocess.CalledProcessError):
            continue
        total += sum(int(item) for item in text.split() if item.strip())
    peak = max(peak, total)
    time.sleep(1)
open(out, "w").write(f"{peak}\n")
PY
  local sampler=$!
  local rc=0
  wait "$child" || rc=$?
  wait "$sampler" || true
  if [ "$rc" -ne 0 ]; then
      echo "smoke: weekly candidate failed for $name" >&2
      cat "$RUN/$name/wrapper.err" >&2
      if [ -f "$RUN/$name/var/scvia-weekly-status.json" ]; then
        cat "$RUN/$name/var/scvia-weekly-status.json" >&2
      fi
      return 1
    fi
  python3 - "$RUN/$name/var/scvia-weekly-status.json" "$RUN/$name/evidence.path" <<'PY' || return 1
import json, sys
body = json.load(open(sys.argv[1]))
if body.get("exit_code") != 0 or body.get("phase") != "complete":
    raise SystemExit(f"smoke: weekly status is not complete: {body}")
open(sys.argv[2], "w").write(body["evidence_dir"] + "\n")
print(json.load(open(body["evidence_dir"] + "/build.json"))["outputs"]["release_id"])
PY
}

echo "smoke: building first sealed release"
RID_A="$(one a 1)" || exit 1
python3 - "$(cat "$RUN/a/evidence.path")/compare.json" "$RUN/a/dist/releases/$RID_A/public/release.json" <<'PY'
import json, sys
compared = json.load(open(sys.argv[1]))
manifest = json.load(open(sys.argv[2]))
if compared.get("snapshot_id") != manifest.get("snapshot_id"):
    raise SystemExit("compare did not use the promoted snapshot named by the release")
if compared.get("release_snapshot_id") != manifest.get("snapshot_id"):
    raise SystemExit("compare release snapshot does not match the built release")
if not compared.get("verdict", {}).get("ok"):
    raise SystemExit("compare verdict is not PASS: " + "; ".join(compared.get("verdict", {}).get("reasons", [])))
print(f"compare snapshot {compared['snapshot_id']}")
PY
echo "smoke: building second sealed release"
RID_B="$(one b "")" || exit 1
if [ "$RID_A" = "$RID_B" ]; then
  echo "smoke: the two releases have the same id" >&2
  exit 1
fi

seal_of() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["seal_sha256"])' "$1"
}
SEAL_A="$(seal_of "$RUN/a/dist/releases/$RID_A/validation.json")"
SEAL_B="$(seal_of "$RUN/b/dist/releases/$RID_B/validation.json")"
if [ "$SEAL_A" = "$SEAL_B" ] || [ "$SEAL_A" = "null" ] || [ "$SEAL_B" = "null" ]; then
  echo "smoke: expected two different non-null seals" >&2
  exit 1
fi

mkdir "$HOST"
"$BIN" publish --release "$RID_A" --output-root "$RUN/a/dist" --destination "$HOST" --json \
  > "$RUN/publish-a.json"
"$PY" "$SCRATCH/docs/rewrite/evidence/inject_upload_failure.py" \
  "$HOST" "$RUN/b/dist" "$RID_B" > "$RUN/inject.json"
"$BIN" publish --release "$RID_B" --output-root "$RUN/b/dist" --destination "$HOST" --json \
  > "$RUN/publish-b.json"
LIVE="$(readlink "$HOST/live")"
if [ "$LIVE" != "releases/$RID_B" ]; then
  echo "smoke: second publish did not become live ($LIVE)" >&2
  exit 1
fi
"$BIN" rollback --release "$RID_A" --output-root "$RUN/a/dist" --destination "$HOST" --json \
  > "$RUN/rollback-a.json"
LIVE="$(readlink "$HOST/live")"
if [ "$LIVE" != "releases/$RID_A" ]; then
  echo "smoke: rollback live pointer is $LIVE" >&2
  exit 1
fi
"$PY" "$SCRATCH/docs/rewrite/evidence/source_inventory.py" --compare "$ROOT" "$SCRATCH" "$RUN/source-inventory.json"
python3 - "$RUN" "$RID_A" "$RID_B" "$SEAL_A" "$SEAL_B" <<'PY'
import json, sys
from pathlib import Path
run, a, b, seal_a, seal_b = sys.argv[1:]
root = Path(run)
inventory = json.loads((root / "source-inventory.json").read_text())
if inventory["scratch_differences"]:
    raise SystemExit("scratch source bytes do not match the writer tree")
evidence = Path((root / "a" / "evidence.path").read_text().strip())
regen = json.loads((evidence / "legacy-regenerate.json").read_text())
body = {
    "host_live": f"releases/{a}",
    "first": a,
    "second": b,
    "seal_first": seal_a,
    "seal_second": seal_b,
    "injected_copy_failure": json.load(open(root / "inject.json")),
    "source_inventory_sha256": inventory["inventory_sha256"],
    "source_inventory_files": inventory["files"],
    "source_inventory_bytes": inventory["bytes"],
    "scratch_matches_source": True,
    "captured_input_sha256": regen["sha256"],
    "captured_input_files": regen["files"],
    "rss_peak_kib": {
        "first": int((root / "a" / "rss-peak-kib.txt").read_text().strip() or 0),
        "second": int((root / "b" / "rss-peak-kib.txt").read_text().strip() or 0),
    },
}
(root / "smoke-result.json").write_text(json.dumps(body, indent=2) + "\n")
print(json.dumps(body))
PY
echo "smoke: rolled back to $RID_A; $RID_B remains on the host and is not live"
