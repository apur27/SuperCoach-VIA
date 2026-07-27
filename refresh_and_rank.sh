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
# Completion accounting (scripts/backtest_completeness.py). The previous detector
# took the newest backtest_summary_*.csv on disk as proof a round was scored. That
# cannot tell a finished cycle from an aborted one: on 2026-07-20 a run wrote its
# artifacts at 15:57, FATALed at 16:04 on the phantom-row gate (mass-duplicated
# player corpus), and the retry run read those orphans as "round 20 complete",
# skipped re-scoring, and published figures computed on a corpus we had rejected.
#
# Now a run counts only if a cycle MARKED it complete after a successful push.
# Sweep first so orphans from any aborted cycle are quarantined out of the
# directory — that also hides them from update_team_analysis.py and
# update_eval_surface.sh, which pick the newest summary by mtime.
"$PYTHON" scripts/backtest_completeness.py --dir data/prediction/backtest sweep
LAST_ROUND=$("$PYTHON" scripts/backtest_completeness.py --dir data/prediction/backtest last-round --year 2026)
if [ -z "$LAST_ROUND" ]; then
    START_ROUND=1
else
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

# docs/afl-backtest-2026.md is written by TWO generators: refresh_readme.py above
# (per-round table, top-30 table) and update_eval_surface.sh (the CUMULATIVE,
# TEAMBIAS and MISSES blocks). Both must run before the doc is gated and committed
# below, or the page ships internally contradictory — claiming N rounds in one
# table while the pooled figures beneath it are still at N-1. That is exactly what
# happened on the 2026-07-27 R21 refresh: this script committed the doc in step 6
# while update_eval_surface.sh did not run until Phase 2, and the doc is not in the
# Phase-4 allowlist, so the stale blocks would never have been committed at all.
# It is idempotent, so Phase 2 running it again is harmless.
bash "$REPO_ROOT/scripts/update_eval_surface.sh"

# Recompute all-time stat leaders + regenerate HOF charts from fresh player data.
# Any image that depends on data is updated here so it never goes stale.
"$PYTHON" docs/hall-of-fame/compute_stat_leaders.py
"$PYTHON" docs/hall-of-fame/generate_records_charts.py

echo "=========================================="
echo "[6/6] Committing and pushing updated docs..."

# Re-verify the gated backtest page BEFORE staging it.
#
# docs/afl-backtest-2026.md carries a council-pipeline stamp, and the pre-commit
# gate cross-checks that stamp against a DataSentinel record keyed on the doc's
# CONTENT HASH. Step [5/6] above regenerates the page every cycle, which changes
# that hash and orphans the previous record — so without this hop the commit is
# refused and the whole harness dies with a round of scraped data uncommitted.
# That is exactly what happened on 2026-07-27, the first refresh after the page
# was gated. The HOF hub and afl-insights.md already have this regenerate ->
# re-verify -> commit hop; this page was gated without one.
#
# Fail-closed: no PASS recorded for the current content means we do not stage it.
CLAUDE="${CLAUDE:-/home/abhi/.claude/local/claude}"
if grep -q '<!-- council-pipeline:' docs/afl-backtest-2026.md 2>/dev/null; then
    # Stage the doc BEFORE gating it. check-council-stamp.sh verifies the STAGED
    # blob by design (F4: staging good bytes then editing bad ones must not pass),
    # but this doc is not staged until further down, so the gate was reading the
    # INDEX copy — still the previous commit's content — while DataSentinel hashed
    # the regenerated worktree copy. The two disagreed, the hop reported success
    # against the wrong bytes, and the real commit gate then blocked on the right
    # ones. Staging first makes both look at the content that will actually ship.
    git add docs/afl-backtest-2026.md
    echo "  [gate] re-verifying regenerated docs/afl-backtest-2026.md through DataSentinel..."
    "$CLAUDE" -p "Pass 2 check on docs/afl-backtest-2026.md. It was just regenerated by the weekly refresh, so its content hash has changed and the previously recorded verdict no longer applies. Verify every **[data]** tag against source, flag untagged specific numbers, and confirm the cumulative block reconciles three ways (summary n_players, per-team n, pooled per-player rows). Note the page deliberately carries BOTH an unweighted mean-across-rounds statistic and player-weighted pooled figures — both correct as labelled, expected to differ, not an inconsistency. Record via: scripts/record-sentinel-verdict.sh --doc docs/afl-backtest-2026.md --verdict <PASS|FAIL> --agent DataSentinel. Emit ONLY the JSON verdict object." \
        --agent DataSentinel --permission-mode bypassPermissions < /dev/null 2>&1 | tail -40
    if COUNCIL_AUDIT_DIR=.claude/audit bash scripts/check-council-stamp.sh docs/afl-backtest-2026.md; then
        echo "  [gate] docs/afl-backtest-2026.md re-verified."
    else
        echo "FATAL: docs/afl-backtest-2026.md failed re-verification after regeneration." >&2
        echo "  The commit is blocked rather than shipping unverified numbers. Route to DataSentinel." >&2
        exit 1
    fi
fi
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
    data/lineups/ \
    2>/dev/null || true
# data/matches + data/player_data ARE the scraped ground truth the published docs cite;
# they MUST be committed or a remote clone fails DataSentinel re-verification (Surveyor D3,
# 2026-07-07 — the R18 actuals were stranded uncommitted for a full cycle). Still explicit
# paths, never `git add .` (which would sweep scratch CSVs under data/prediction/).
#
# Lineups were excluded while the scraper emitted jersey numbers plus Rushed/Totals/
# Opposition junk (S3). That corruption is now confined to 700 legacy rows of 33,999;
# current output is clean name-form, and the files are real pipeline output that was
# drifting uncommitted every cycle, so they are staged again. The 700 historical rows
# are a separate data-quality item for Scientist — staging does not worsen them.

# F7: stage the prediction + backtest CSVs that docs/afl-insights.md cites as sources
# and that the by-archive backtest depends on surviving between cycles. Previously these
# were left untracked, so a fresh clone lost them and the backtest lost its archived
# forward CSVs. EXPLICIT patterns only — never `git add data/prediction` wholesale (that
# would sweep experimental / scratch CSVs). The upcoming-round prediction is staged
# latest-only; the backtest artifacts accumulate as history and are all staged.
LATEST_NEXT=$(ls -t data/prediction/next_round_*_prediction_*.csv 2>/dev/null | head -1 || true)
[ -n "$LATEST_NEXT" ] && git add "$LATEST_NEXT" 2>/dev/null || true
git add \
    data/prediction/backtest/backtest_summary_*.csv \
    data/prediction/backtest/prediction_vs_actual_*.csv \
    data/prediction/backtest/backtest_by_team_*.csv \
    data/prediction/backtest/backtest_by_position_*.csv \
    data/prediction/backtest/backtest_run_*.log \
    data/prediction/backtest/completed_runs.json \
    data/prediction/optuna_best_params.json \
    2>/dev/null || true

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
        # The cycle reached origin — only now do this run's backtest artifacts
        # count as complete. Marking any earlier would re-create the orphan bug:
        # a later FATAL would leave artifacts that look finished but are not.
        "$PYTHON" scripts/backtest_completeness.py --dir data/prediction/backtest mark
    fi
fi

echo "=========================================="
echo "Pipeline completed successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
