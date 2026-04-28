#!/usr/bin/env bash
set -e

PYTHON=/home/abhi/sourceCode/python/coding/.venv/bin/python
REPO_ROOT=/home/abhi/git/SuperCoach-VIA

cd "$REPO_ROOT"

echo "=========================================="
echo "[1/2] Refreshing match and player data..."
echo "=========================================="
"$PYTHON" refresh_data.py

echo "=========================================="
echo "[2/2] Recalculating and formatting top 100..."
echo "=========================================="
"$PYTHON" top_players_comprehensive.py

echo "=========================================="
echo "Pipeline completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
