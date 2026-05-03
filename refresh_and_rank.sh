#!/usr/bin/env bash
set -e

PYTHON=/home/abhi/sourceCode/python/coding/.venv/bin/python
REPO_ROOT=/home/abhi/git/SuperCoach-VIA

cd "$REPO_ROOT"

echo "=========================================="
echo "[1/4] Refreshing match and player data..."
echo "=========================================="
"$PYTHON" refresh_data.py

echo "=========================================="
echo "[2/4] Recalculating and formatting top 100..."
echo "=========================================="
"$PYTHON" top_players_comprehensive.py

echo "=========================================="
echo "[3/4] Refreshing README charts and analysis..."
echo "=========================================="
"$PYTHON" refresh_readme.py

echo "=========================================="
echo "[4/4] Committing and pushing updated docs..."
echo "=========================================="
# Stage every doc / chart / CSV that the pipeline regenerates. The list is
# deliberate — `git add .` would risk pulling in scratch CSVs sitting in
# data/prediction/ that we don't want auto-committed.
git add \
    docs/afl-season-2026.md \
    docs/afl-team-profiles.md \
    docs/afl-insights.md \
    docs/hall-of-fame-top100.md \
    assets/charts/ \
    all_time_top_100.csv \
    data/top100/all_time_top_100.csv \
    2>/dev/null || true

if git diff --cached --quiet; then
    echo "No doc changes to commit."
else
    TODAY=$(date '+%Y-%m-%d')
    git commit -m "Auto-update: refresh AFL insights and top-100 rankings (${TODAY})"
    git push origin main
    echo "Pushed to origin/main"
fi

echo "=========================================="
echo "Pipeline completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
