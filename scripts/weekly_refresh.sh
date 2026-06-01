#!/usr/bin/env bash
# =============================================================================
# weekly_refresh.sh — Full weekly pipeline for SuperCoach-VIA
# =============================================================================
#
# Run every Wednesday morning — round completes Sunday, data settled by Tuesday.
# Recommended cron (Wednesday 6 AM AEST = Tuesday 8 PM UTC):
#   0 20 * * 2 cd /home/abhi/git/SuperCoach-VIA && bash scripts/weekly_refresh.sh
#
# What runs:
#   Phase 1  — refresh_and_rank.sh   (data scrape → top100 → prediction →
#                                     backtest → season docs → git push)
#   Phase 2a — generate_weekly_cheat_sheet.py  (round cheat sheet)
#   Phase 2b — compute_stat_leaders.py + generate_records_charts.py
#              + DataSentinel agent to update docs/hall-of-fame-stat-*.md
#   Phase 3  — FootyStrategy agent   (round recap in docs/afl-insights.md)
#   Phase 4  — commit + push phase 2/3 outputs
#
# Excluded (manually curated — never auto-updated):
#   docs/hall-of-fame-captains.md       Captains HOF
#   docs/hall-of-fame-courageous.md     Courageous HOF
#   docs/hall-of-fame-forgotten-heroes.md  Forgotten Heroes
#   docs/hall-of-fame-dynasties.md      Dynasties HOF
#   docs/hall-of-fame-indigenous.md     Indigenous HOF
#   docs/hall-of-fame-careers-cut-short.md
#   docs/hall-of-fame-coaches.md
#   docs/news/README.md                 Published news
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/home/abhi/sourceCode/python/coding/.venv/bin/python
CLAUDE=/home/abhi/.claude/local/claude
LOG_DIR="$REPO_ROOT/.claude/audit"
TODAY=$(date '+%Y-%m-%d')
LOG_FILE="$LOG_DIR/weekly_refresh_${TODAY}.log"

cd "$REPO_ROOT"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

mkdir -p "$LOG_DIR"
log "=================================================================="
log "Weekly refresh started — $TODAY"
log "=================================================================="

# ---------------------------------------------------------------------------
# Phase 1 — data + model pipeline
# refresh_and_rank.sh handles its own git add/commit/push for:
#   docs/afl-season-2026.md, afl-team-analysis-2026.md, afl-finals-2026.md,
#   afl-brownlow-2026.md, afl-stat-leaders-2026.md, afl-predictions-2026.md,
#   afl-backtest-2026.md, afl-team-profiles.md, hall-of-fame-top100.md,
#   assets/charts/, all_time_top_100.csv, data/top100/
# ---------------------------------------------------------------------------
log "[1/5] Running refresh_and_rank.sh (data + model + season docs)..."
bash "$REPO_ROOT/refresh_and_rank.sh" 2>&1 | tee -a "$LOG_FILE"
log "[1/5] refresh_and_rank.sh complete."

# ---------------------------------------------------------------------------
# Detect round AFTER Phase 1 so we read the CSV that prediction.py just wrote,
# not whatever was on disk before the run.
# ---------------------------------------------------------------------------
LATEST_PRED=$(ls "$REPO_ROOT/data/prediction"/next_round_*_prediction_*.csv 2>/dev/null \
    | sort | tail -1 || true)
if [ -n "$LATEST_PRED" ]; then
    ROUND=$(basename "$LATEST_PRED" | grep -oP 'next_round_\K[0-9]+' || echo "unknown")
else
    ROUND="unknown"
fi
log "Detected next round: $ROUND"

# ---------------------------------------------------------------------------
# Phase 2a — weekly cheat sheet
# Reads the latest prediction CSV, writes:
#   docs/weekly/round-<N>-<year>.md
#   docs/weekly/round-current-<year>.md  (stable link)
# ---------------------------------------------------------------------------
log "[2a/5] Generating weekly cheat sheet for round $ROUND..."
$PYTHON "$REPO_ROOT/scripts/generate_weekly_cheat_sheet.py" 2>&1 | tee -a "$LOG_FILE"
log "[2a/5] Cheat sheet generated."

# ---------------------------------------------------------------------------
# Phase 2b — Hall of Fame stat leaders refresh
# 1. Recompute career stat totals from the freshly-updated player_data corpus
# 2. Regenerate stat records charts
# 3. DataSentinel agent: compare fresh JSON to the published stat pages,
#    update any changed numbers + "Last refreshed" date across all stat docs
# ---------------------------------------------------------------------------
log "[2b/5] Recomputing all-time stat leaders from updated player data..."
$PYTHON "$REPO_ROOT/docs/hall-of-fame/compute_stat_leaders.py" 2>&1 | tee -a "$LOG_FILE"
log "[2b/5] Stat leaders JSON refreshed."

log "[2b/5] Regenerating stat records charts..."
$PYTHON "$REPO_ROOT/docs/hall-of-fame/generate_records_charts.py" 2>&1 | tee -a "$LOG_FILE"
log "[2b/5] Charts regenerated."

log "[2b/5] Invoking DataSentinel to update hall-of-fame stat pages (2 batches)..."

# Split into 2 batches to stay under the 32k output token limit.
SENTINEL_OPTS="--agent DataSentinel --allowedTools Read,Write,Edit,Glob,Grep --permission-mode bypassPermissions --model sonnet"

CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000 $CLAUDE -p "You are DataSentinel for SuperCoach-VIA. Today is $TODAY.

Ground truth: docs/hall-of-fame/_stat_leaders.json (read this first).

Check and update ONLY these docs (batch 1 of 2):
- docs/hall-of-fame-stat-leaders.md
- docs/hall-of-fame-stat-disposals.md
- docs/hall-of-fame-stat-games.md
- docs/hall-of-fame-stat-goals.md
- docs/hall-of-fame-stat-tackles.md
- docs/hall-of-fame-stat-marks.md
- docs/hall-of-fame-stat-brownlow.md

Rules: only update [data]-tagged numbers that changed. Update 'Last refreshed:' and
DataSentinel stamp date to $TODAY in every doc you touch. No narrative changes. Skip
docs where nothing changed. Be concise." \
    $SENTINEL_OPTS 2>&1 | tee -a "$LOG_FILE"

CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000 $CLAUDE -p "You are DataSentinel for SuperCoach-VIA. Today is $TODAY.

Ground truth: docs/hall-of-fame/_stat_leaders.json (read this first).

Check and update ONLY these docs (batch 2 of 2):
- docs/hall-of-fame-stat-clearances.md
- docs/hall-of-fame-stat-contested.md
- docs/hall-of-fame-stat-hitouts.md
- docs/hall-of-fame-stat-kicks-handballs.md
- docs/hall-of-fame-stat-goalassists.md
- docs/hall-of-fame-stat-single-season.md

Rules: only update [data]-tagged numbers that changed. Update 'Last refreshed:' and
DataSentinel stamp date to $TODAY in every doc you touch. No narrative changes. Skip
docs where nothing changed. Be concise." \
    $SENTINEL_OPTS 2>&1 | tee -a "$LOG_FILE"

log "[2b/5] DataSentinel stat update complete."

# ---------------------------------------------------------------------------
# Phase 3 — FootyStrategy agent: round recap + insights update
# ---------------------------------------------------------------------------
log "[3/5] Invoking FootyStrategy agent for round $ROUND weekly insights..."

$CLAUDE -p "You are FootyStrategy for SuperCoach-VIA. Today is $TODAY.

## Task
Update docs/afl-insights.md with a concise 'Week in Review' section for Round $ROUND.

## What to read first (all freshly updated by the pipeline)
- docs/afl-stat-leaders-2026.md       — current season stat leaders
- docs/afl-season-2026.md             — team analysis, ladder, finals pathway
- docs/afl-predictions-2026.md        — disposal predictions for next round
- docs/weekly/round-current-2026.md   — this round's cheat sheet picks

## What to write
Add or replace a section headed '## Round $ROUND — Week in Review' immediately
after the intro table in docs/afl-insights.md. Include:
1. Top 3 disposal performers from the stat leaders this round
2. One sentence on the most notable ladder movement
3. One player to watch next round (from the predictions cheat sheet)
4. One sentence of tactical insight grounded in the data

Keep it tight: 150–200 words maximum. Use data-backed claims only.
Tag any specific stat with [data] per the council convention.

## Hard rules
- Do NOT touch the navigation table, intro text, or any link in the file.
- Do NOT touch docs/news/, docs/hall-of-fame*.md, or any other file.
- Do NOT add new links to Hall of Fame or news pages.
- If the stat leaders or season doc do not have enough data for a claim, omit it." \
    --agent FootyStrategy \
    --allowedTools "Read,Write,Edit,Glob,Grep" \
    --permission-mode bypassPermissions \
    --model sonnet \
    2>&1 | tee -a "$LOG_FILE"

log "[3/5] FootyStrategy agent complete."

# ---------------------------------------------------------------------------
# Phase 4 — commit and push all phase 2/3 outputs
# ---------------------------------------------------------------------------
log "[4/5] Staging and committing weekly agent outputs..."

git add \
    docs/afl-insights.md \
    docs/weekly/ \
    docs/hall-of-fame-stat-leaders.md \
    docs/hall-of-fame-stat-disposals.md \
    docs/hall-of-fame-stat-games.md \
    docs/hall-of-fame-stat-goals.md \
    docs/hall-of-fame-stat-brownlow.md \
    docs/hall-of-fame-stat-tackles.md \
    docs/hall-of-fame-stat-marks.md \
    docs/hall-of-fame-stat-clearances.md \
    docs/hall-of-fame-stat-contested.md \
    docs/hall-of-fame-stat-hitouts.md \
    docs/hall-of-fame-stat-kicks-handballs.md \
    docs/hall-of-fame-stat-goalassists.md \
    docs/hall-of-fame-stat-single-season.md \
    docs/hall-of-fame/_stat_leaders.json \
    assets/charts/ \
    2>/dev/null || true

if git diff --cached --quiet; then
    log "[4/5] No new changes to commit — everything already up to date."
else
    git commit -m "Weekly refresh round $ROUND — stat leaders + cheat sheet + insights ($TODAY)"
    git push origin main
    log "[4/5] Pushed to origin/main."
fi

log "=================================================================="
log "Weekly refresh complete — round $ROUND — $TODAY"
log "=================================================================="
