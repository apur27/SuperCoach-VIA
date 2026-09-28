#!/usr/bin/env bash
# Proposed interpreter resolution for .githooks/pre-commit. Not installed.
# The live hook remains the copy core.hooksPath already points at, which still
# defaults to the absent /home/abhi/sourceCode/python/coding/.venv/bin/python.
# Activation is a later owner step: replace that default with this order and
# re-run the section 4 smoke. Do not point core.hooksPath at this file.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [ -n "${COUNCIL_PYTHON:-}" ] && [ -x "${COUNCIL_PYTHON}" ]; then
  PYTHON="${COUNCIL_PYTHON}"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  echo "pre-commit: no interpreter. Set COUNCIL_PYTHON or create .venv with uv sync --locked." >&2
  exit 1
fi
echo "$PYTHON"
# Proposed hermetic command, not installed and not run by this file.
# Four workers, one native thread each. Measure it under taskset -c 0-3.
# OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
#   "$PYTHON" -m pytest tests/scvia -m "not integration" -q -n 4 --dist loadscope
