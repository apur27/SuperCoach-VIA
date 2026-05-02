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
echo "[2/3] Recalculating and formatting top 100..."
echo "=========================================="
"$PYTHON" top_players_comprehensive.py

echo "=========================================="
echo "[3/3] Updating 2026 team analysis in README..."
echo "=========================================="
"$PYTHON" update_team_analysis.py

echo "=========================================="
echo "Pipeline completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
