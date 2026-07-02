#!/usr/bin/env bash
set -e

PYTHON=/home/abhi/sourceCode/python/coding/.venv/bin/python
REPO_ROOT=/home/abhi/git/SuperCoach-VIA

cd "$REPO_ROOT"

echo "=========================================="
echo "[1/6] Refreshing match and player data..."
echo "=========================================="
"$PYTHON" refresh_data.py

echo "=========================================="
echo "[2/6] Recalculating and formatting top 100..."
echo "=========================================="
"$PYTHON" top_players_comprehensive.py

echo "=========================================="
echo "[3/6] Predicting next round disposals..."
echo "=========================================="
# Auto-detects the current year and next round from latest player data.
# Writes: data/prediction/next_round_<N>_prediction_<timestamp>.csv
# prediction.py now lives in the supercoach/ package — invoke as a module so
# its package-relative imports and config bootstrap resolve correctly.
"$PYTHON" -m supercoach.prediction

echo "=========================================="
echo "[4/6] Backtesting prediction accuracy (incremental)..."
echo "=========================================="
# Walk-forward backtest — incremental only. Detects the last complete run
# (one that produced a backtest_summary_*.csv) and starts from the next round.
# Writes: data/prediction/backtest/backtest_summary_<timestamp>.csv
#         data/prediction/backtest/backtest_by_team_<timestamp>.csv
#         data/prediction/backtest/backtest_by_position_<timestamp>.csv
LAST_TS=$(ls data/prediction/backtest/backtest_summary_*.csv 2>/dev/null | sort | tail -1 | grep -oP '\d{8}_\d{6}')
if [ -z "$LAST_TS" ]; then
    START_ROUND=1
else
    LAST_ROUND=$(ls data/prediction/backtest/prediction_vs_actual_round_*_2026_${LAST_TS}.csv 2>/dev/null \
        | grep -oP 'round_\K[0-9]+' | sort -n | tail -1)
    START_ROUND=$((LAST_ROUND + 1))
fi
echo "Last complete backtest: round ${LAST_ROUND:-none}. Running from round ${START_ROUND}."
"$PYTHON" backtest.py --start-year 2026 --start-round "$START_ROUND" --end-year 2026 --end-round auto

echo "=========================================="
echo "[5/6] Refreshing docs, charts and analysis..."
echo "=========================================="
# Picks up the fresh prediction and backtest CSVs written in steps 3 and 4
# and embeds them into docs/afl-predictions-2026.md and docs/afl-backtest-2026.md.
"$PYTHON" refresh_readme.py

# Recompute all-time stat leaders + regenerate HOF charts from fresh player data.
# Any image that depends on data is updated here so it never goes stale.
"$PYTHON" docs/hall-of-fame/compute_stat_leaders.py
"$PYTHON" docs/hall-of-fame/generate_records_charts.py

echo "=========================================="
echo "[6/6] Committing and pushing updated docs..."
echo "=========================================="
# Stage every doc / chart / CSV that the pipeline regenerates. The list is
# deliberate — `git add .` would risk pulling in scratch CSVs sitting in
# data/prediction/ that we don't want auto-committed.
git add \
    docs/afl-season-2026.md \
    docs/afl-team-analysis-2026.md \
    docs/afl-finals-2026.md \
    docs/afl-brownlow-2026.md \
    docs/afl-stat-leaders-2026.md \
    docs/afl-predictions-2026.md \
    docs/afl-backtest-2026.md \
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
    scripts/git_commit_safe.sh commit -m "Auto-update: refresh AFL insights, predictions and backtest (${TODAY})"
    git push origin main
    echo "Pushed to origin/main"
fi

echo "=========================================="
echo "Pipeline completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
