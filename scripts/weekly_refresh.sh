#!/usr/bin/env bash
# =============================================================================
# weekly_refresh.sh — Full weekly pipeline for SuperCoach-VIA
# =============================================================================
#
# Run every Thursday night after the AFL round is complete.
# Recommended cron (Thursday 9 PM AEST = 11 AM UTC):
#   0 11 * * 4 cd /home/abhi/git/SuperCoach-VIA && bash scripts/weekly_refresh.sh
#
# What runs:
#   Phase 1 — refresh_and_rank.sh   (data scrape → top100 → prediction →
#                                    backtest → season docs → git push)
#   Phase 2 — generate_weekly_cheat_sheet.py  (round cheat sheet)
#   Phase 3 — FootyStrategy agent   (round recap + next-round preview in
#                                    docs/afl-insights.md)
#   Phase 4 — commit + push phase 2/3 outputs
#
# Excluded (manually curated — never auto-updated):
#   docs/hall-of-fame*.md           Hall of Fame pages
#   docs/news/README.md             Published news
#   docs/hall-of-fame-stat-*.md     Stat record pages
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/home/abhi/sourceCode/python/coding/.venv/bin/python
CLAUDE=/home/abhi/.claude/local/claude
LOG_DIR="$REPO_ROOT/.claude/audit"
TODAY=$(date '+%Y-%m-%d')
LOG_FILE="$LOG_DIR/weekly_refresh_${TODAY}.log"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
# ---------------------------------------------------------------------------

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
log "[1/4] Running refresh_and_rank.sh (data + model + season docs)..."
bash "$REPO_ROOT/refresh_and_rank.sh" 2>&1 | tee -a "$LOG_FILE"
log "[1/4] refresh_and_rank.sh complete."

# ---------------------------------------------------------------------------
# Detect the current round from the latest prediction CSV so the agent
# and commit message reference the right round number.
# ---------------------------------------------------------------------------
LATEST_PRED=$(ls "$REPO_ROOT/data/prediction"/next_round_*_prediction_*.csv 2>/dev/null \
    | sort | tail -1 || true)
if [ -n "$LATEST_PRED" ]; then
    ROUND=$(basename "$LATEST_PRED" | grep -oP 'next_round_\K[0-9]+' || echo "unknown")
else
    ROUND="unknown"
fi
log "Detected round: $ROUND"

# ---------------------------------------------------------------------------
# Phase 2 — weekly cheat sheet
# Reads the latest prediction CSV, writes:
#   docs/weekly/round-<N>-<year>.md
#   docs/weekly/round-current-<year>.md  (stable link)
# ---------------------------------------------------------------------------
log "[2/4] Generating weekly cheat sheet for round $ROUND..."
$PYTHON "$REPO_ROOT/scripts/generate_weekly_cheat_sheet.py" 2>&1 | tee -a "$LOG_FILE"
log "[2/4] Cheat sheet generated."

# ---------------------------------------------------------------------------
# Phase 3 — FootyStrategy agent: round recap + insights update
# Invokes the FootyStrategy council agent via Claude Code CLI (-p = non-interactive).
# The agent reads the freshly-updated season docs and writes a "Week in Review"
# section to docs/afl-insights.md.
# ---------------------------------------------------------------------------
log "[3/4] Invoking FootyStrategy agent for round $ROUND weekly insights..."

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
- If the stat leaders or season doc do not have enough data for a claim, omit the claim rather than estimating." \
    --agent FootyStrategy \
    --allowedTools "Read,Write,Edit,Glob,Grep" \
    --permission-mode bypassPermissions \
    --model sonnet \
    2>&1 | tee -a "$LOG_FILE"

log "[3/4] FootyStrategy agent complete."

# ---------------------------------------------------------------------------
# Phase 4 — commit and push phase 2 + phase 3 outputs
# Only commits files that were actually changed by this session.
# refresh_and_rank.sh already pushed its own outputs in phase 1.
# ---------------------------------------------------------------------------
log "[4/4] Staging and committing weekly agent outputs..."

git add \
    docs/afl-insights.md \
    docs/weekly/ \
    2>/dev/null || true

if git diff --cached --quiet; then
    log "[4/4] No new changes to commit — everything already up to date."
else
    git commit -m "Weekly refresh round $ROUND — cheat sheet + insights ($TODAY)"
    git push origin main
    log "[4/4] Pushed to origin/main."
fi

log "=================================================================="
log "Weekly refresh complete — round $ROUND — $TODAY"
log "=================================================================="
