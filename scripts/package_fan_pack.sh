#!/usr/bin/env bash
# Package the weekly fan pack into a zip ready for a GitHub Release.
#
# Outputs: ./fan-pack-<tag>.zip in the repo root.
#
# Usage from repo root:
#   ./scripts/package_fan_pack.sh
#   ./scripts/package_fan_pack.sh weekly-2026-05-10   # custom tag
#
# Requires: zip, find, sort. No Python, no extra deps.
#
# This script is the local equivalent of the .github/workflows/weekly-fan-pack.yml
# GitHub Action - run it manually if you want a release artifact without
# waiting for the scheduled workflow.

set -euo pipefail

# Resolve repo root (the directory containing this script's parent dir).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TAG="${1:-weekly-$(date -u +%Y-%m-%d)}"
STAGE_DIR="$(mktemp -d)"
ZIP_PATH="${REPO_ROOT}/fan-pack-${TAG}.zip"

cleanup() { rm -rf "${STAGE_DIR}"; }
trap cleanup EXIT

# 1. Find the most recent prediction CSV.
LATEST_CSV="$(find data/prediction -maxdepth 1 -type f -name 'next_round_*_prediction_*.csv' \
  -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | awk '{print $2}')"

if [ -z "${LATEST_CSV}" ]; then
  echo "ERROR: no prediction CSV found under data/prediction/" >&2
  echo "Run prediction.py first." >&2
  exit 1
fi

echo "Latest prediction CSV: ${LATEST_CSV}"

# 2. Stage the bundle.
mkdir -p "${STAGE_DIR}/fan-pack"
cp "${LATEST_CSV}" "${STAGE_DIR}/fan-pack/latest-prediction.csv"

# Best-effort copy of fan-friendly docs (do not fail if a file is missing).
for src in \
  "docs/afl-predictions-2026.md:predictions.md" \
  "docs/afl-stat-leaders-2026.md:stat-leaders.md" \
  "docs/afl-backtest-2026.md:backtest.md" \
  "docs/glossary.md:glossary.md" \
  "docs/how-to-use-this-for-supercoach.md:how-to-use.md" \
  "templates/google-sheets-template.md:google-sheets-template.md" \
  "docs/weekly/round-current-2026.md:cheat-sheet.md" \
; do
  src_path="${src%%:*}"
  dest_name="${src##*:}"
  if [ -f "${src_path}" ]; then
    cp "${src_path}" "${STAGE_DIR}/fan-pack/${dest_name}"
  else
    echo "  (skip) ${src_path} not found"
  fi
done

# 3. Generate a README for the bundle.
cat > "${STAGE_DIR}/fan-pack/README.txt" <<EOF
SuperCoach VIA - Weekly fan pack
Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Source: https://github.com/apur27/SuperCoach-VIA

Files:
- latest-prediction.csv      Predicted disposals for the next round
- predictions.md             Top 30 predicted disposal leaders, formatted
- cheat-sheet.md             Top 30 + per-club top 3 picks
- stat-leaders.md            Current 2026 season stat leaders
- backtest.md                How accurate the model has been so far
- glossary.md                Footy and data terms in plain English
- how-to-use.md              Honest guide to what these predictions are good for
- google-sheets-template.md  How to turn the CSV into a 5-tab dashboard

See the repo for everything else, including the Hall of Fame.
EOF

# 4. Zip it up.
( cd "${STAGE_DIR}" && zip -qr "${ZIP_PATH}" "fan-pack" )

echo
echo "Wrote ${ZIP_PATH}"
ls -lh "${ZIP_PATH}"

cat <<EOF

To create a manual GitHub Release with this bundle:

  1. Go to https://github.com/apur27/SuperCoach-VIA/releases/new
  2. Tag: ${TAG}
  3. Title: Weekly fan pack - ${TAG}
  4. Body: paste from docs/start-here-no-code.md or describe what changed.
  5. Attach: ${ZIP_PATH}
  6. Publish.

Or with the gh CLI:

  gh release create ${TAG} \\
    --title "Weekly fan pack - ${TAG}" \\
    --notes "Latest prediction CSV plus fan-friendly docs." \\
    ${ZIP_PATH}

EOF
