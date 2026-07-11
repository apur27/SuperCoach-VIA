#!/usr/bin/env bash
set -e

PYTHON=/home/abhi/sourceCode/python/coding/.venv/bin/python
REPO_ROOT=/home/abhi/git/SuperCoach-VIA

cd "$REPO_ROOT"

# --- single-entry-point discipline (F04) -----------------------------------
# scripts/weekly_refresh.sh is the sole sanctioned cycle entry point; it runs
# this script as an internal phase (and adds the phantom-row gate, HOF pipeline,
# QA, and the completion sentinel around it). Running this directly does a PARTIAL
# refresh with no gates. Allowed only with the parent env var (set by weekly_refresh.sh
# and the /weekly-cycle skill) or an explicit --allow-direct override.
if [ "${WEEKLY_REFRESH_PARENT:-0}" != "1" ] && [[ "$*" != *--allow-direct* ]]; then
  echo "refresh_and_rank.sh is an internal phase of the weekly cycle, not an entry point." >&2
  echo "  Run:  bash scripts/weekly_refresh.sh            (full gated cycle)" >&2
  echo "  Or, for a deliberate partial run:  bash refresh_and_rank.sh --allow-direct" >&2
  exit 1
fi
# Past the guard — propagate to child scripts (refresh_data.py) so they don't re-block.
export WEEKLY_REFRESH_PARENT=1
# ---------------------------------------------------------------------------

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
echo "[4/6] Backtesting prediction accuracy (incremental, by-archive)..."
echo "=========================================="
# Walk-forward backtest — incremental only, and now BY ARCHIVE: instead of
# re-running the predictor (which trained+tuned ~24 min/round AND wrote a
# next_round_*.csv into the live namespace that mtime-newest resolution then
# shipped in place of the real forward prediction), we score the forward CSV
# that was ACTUALLY published for each completed round.
#
# Timing note (the load-bearing subtlety): step 3 above just predicted the
# UPCOMING round M (writes next_round_M). The rounds we can score now are those
# with recorded actuals — i.e. START_ROUND .. M-1. The forward CSV that
# predicted each of those rounds was archived in a PRIOR cycle as
# next_round_<R>_prediction_*.csv and is still on disk. We score that archived
# CSV, NOT this cycle's next_round_M (which has no actuals yet).
#
# Detects the last complete run (one that produced a backtest_summary_*.csv)
# and starts from the next round. Writes, per scored round:
#   data/prediction/backtest/backtest_summary_<timestamp>.csv
#   data/prediction/backtest/backtest_by_team_<timestamp>.csv
#   data/prediction/backtest/backtest_by_position_<timestamp>.csv
#   data/prediction/backtest/prediction_vs_actual_round_<R>_2026_<timestamp>.csv
LAST_TS=$(ls data/prediction/backtest/backtest_summary_*.csv 2>/dev/null | sort | tail -1 | grep -oP '\d{8}_\d{6}')
if [ -z "$LAST_TS" ]; then
    START_ROUND=1
else
    LAST_ROUND=$(ls data/prediction/backtest/prediction_vs_actual_round_*_2026_${LAST_TS}.csv 2>/dev/null \
        | grep -oP 'round_\K[0-9]+' | sort -n | tail -1)
    START_ROUND=$((LAST_ROUND + 1))
fi

# M = the upcoming round step 3 just predicted (newest next_round_* by mtime).
# The last round WITH actuals is M-1, so that's our upper bound to score.
LATEST_FWD=$(ls -t data/prediction/next_round_*_prediction_*.csv 2>/dev/null | head -1)
UPCOMING_ROUND=$(basename "${LATEST_FWD:-}" | grep -oP 'next_round_\K[0-9]+' || echo "")
if [ -z "$UPCOMING_ROUND" ]; then
    echo "WARNING: no forward prediction CSV found — skipping backtest."
    END_SCORE_ROUND=0
else
    END_SCORE_ROUND=$((UPCOMING_ROUND - 1))
fi
echo "Last complete backtest: round ${LAST_ROUND:-none}. Upcoming (predicted) round: ${UPCOMING_ROUND:-none}."
echo "Scoring completed rounds ${START_ROUND}..${END_SCORE_ROUND} against their archived forward CSVs."

for R in $(seq "$START_ROUND" "$END_SCORE_ROUND"); do
    # Archived forward CSV that predicted round R (written in a prior cycle).
    ARCHIVED_PRED=$(ls -t data/prediction/next_round_${R}_prediction_*.csv 2>/dev/null | head -1)
    if [ -n "$ARCHIVED_PRED" ]; then
        echo "  round $R: scoring archived $ARCHIVED_PRED"
        "$PYTHON" backtest.py --start-year 2026 --start-round "$R" --end-year 2026 \
            --end-round "$R" --from-csv "$ARCHIVED_PRED"
    else
        # No archived forward CSV for this round (e.g. a cycle where the forward
        # run never wrote one). Fall back to the full-retrain path so the round
        # is still scored — a permanent gap would violate the preserve-all-rounds
        # invariant. This is the slow path (~24 min/round); it is rare.
        echo "  round $R: no archived forward CSV — falling back to full retrain"
        "$PYTHON" backtest.py --start-year 2026 --start-round "$R" --end-year 2026 --end-round "$R"
    fi
done

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
    data/matches/ \
    data/player_data/ \
    2>/dev/null || true
# data/matches + data/player_data ARE the scraped ground truth the published docs cite;
# they MUST be committed or a remote clone fails DataSentinel re-verification (Surveyor D3,
# 2026-07-07 — the R18 actuals were stranded uncommitted for a full cycle). Still explicit
# paths, never `git add .` (which would sweep scratch CSVs under data/prediction/). Lineups
# are intentionally excluded until their scraper corruption is fixed (S3).

if git diff --cached --quiet; then
    echo "No doc changes to commit."
else
    TODAY=$(date '+%Y-%m-%d')
    scripts/git_commit_safe.sh commit -m "Auto-update: refresh AFL insights, predictions and backtest (${TODAY})"
    if [ -n "${WEEKLY_REFRESH_PARENT:-}" ]; then
        # Push is deferred to weekly_refresh.sh, which runs the phantom-row gate first.
        echo "Push deferred to parent harness (phantom-row gate runs before push)."
    else
        git push origin main
        echo "Pushed to origin/main"
    fi
fi

echo "=========================================="
echo "Pipeline completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
