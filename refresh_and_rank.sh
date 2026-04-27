#!/usr/bin/env bash
set -e

PYTHON=/home/abhi/sourceCode/python/coding/.venv/bin/python
REPO_ROOT=/home/abhi/git/SuperCoach-VIA

cd "$REPO_ROOT"

echo "=========================================="
echo "[1/3] Refreshing match and player data..."
echo "=========================================="
"$PYTHON" refresh_data.py

echo "=========================================="
echo "[2/3] Recalculating top 100 rankings..."
echo "=========================================="
"$PYTHON" top_players_comprehensive.py

echo "=========================================="
echo "[3/3] Formatting all_time_top_100.csv..."
echo "=========================================="
"$PYTHON" formatTop100.py

echo "=========================================="
echo "Pipeline completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
