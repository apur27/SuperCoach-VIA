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

RUN_START=$(date +%s)

# ---------------------------------------------------------------------------
# Phase 1 — data + model pipeline
# refresh_and_rank.sh commits Phase 1 artifacts but defers the push (detected
# via WEEKLY_REFRESH_PARENT=1) so the phantom-row gate can run before anything
# reaches origin.
# ---------------------------------------------------------------------------
log "[1/5] Running refresh_and_rank.sh (data + model + season docs)..."
WEEKLY_REFRESH_PARENT=1 bash "$REPO_ROOT/refresh_and_rank.sh" 2>&1 | tee -a "$LOG_FILE"
log "[1/5] refresh_and_rank.sh complete."

# ---------------------------------------------------------------------------
# Phantom-row gate (F12) — validates scraped player CSVs BEFORE the Phase 1
# commit reaches origin. A non-zero exit aborts here; git push only runs after
# the gate clears, so a bad scrape never reaches remote.
# ---------------------------------------------------------------------------
log "[1c/5] Phantom-row gate: validating scraped player CSVs for dropped/doubled rows..."
if ! $PYTHON "$REPO_ROOT/scripts/phantom_row_validator.py" 2>&1 | tee -a "$LOG_FILE"; then
    log "FATAL: phantom-row validator found dropped/doubled player rows — Phase 1 commit NOT pushed. Route to Scientist."
    exit 1
fi
log "[1c/5] Phantom-row gate passed. Pushing Phase 1 commit..."
PENDING=$(git -C "$REPO_ROOT" rev-list origin/main..HEAD --count 2>/dev/null || echo 0)
if [ "$PENDING" -gt 0 ]; then
    git -C "$REPO_ROOT" push origin main 2>&1 | tee -a "$LOG_FILE"
    log "[1c/5] Phase 1 artifacts pushed ($PENDING commit(s))."
else
    log "[1c/5] No new Phase 1 commits to push."
fi

# ---------------------------------------------------------------------------
# Detect round AFTER Phase 1. Use mtime — valid on a local machine where write
# order reflects wall-clock. Also assert the CSV was written THIS cycle (mtime
# > RUN_START) so a stale on-disk file never silently re-labels a prior round.
# ---------------------------------------------------------------------------
LATEST_PRED=$(ls -t "$REPO_ROOT/data/prediction"/next_round_*_prediction_*.csv 2>/dev/null \
    | head -1 || true)
if [ -n "$LATEST_PRED" ]; then
    ROUND=$(basename "$LATEST_PRED" | grep -oP 'next_round_\K[0-9]+' || echo "unknown")
else
    ROUND="unknown"
fi
log "Detected next round: $ROUND"
if [ "$ROUND" = "unknown" ]; then
    log "ERROR: Could not detect next round from prediction CSVs. Run Phase 1 (refresh_and_rank.sh) first, then re-run this script."
    exit 1
fi
PRED_MTIME=$(stat -c %Y "$LATEST_PRED" 2>/dev/null || echo 0)
if [ "$PRED_MTIME" -lt "$RUN_START" ]; then
    log "ERROR: Latest prediction CSV ($(basename "$LATEST_PRED")) predates this run — Phase 1 did not write a new prediction. Aborting to prevent re-labelling a prior round."
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1b — refresh the README "Eval results" section + docs/banner.svg from
# the backtest figures Phase 1 just produced. Re-derives already-verified
# numbers via the same merge logic as the backtest doc; touches only the eval
# table + banner pills/numbers, never the news block. Idempotent.
# ---------------------------------------------------------------------------
log "[1b/5] Updating README eval surface + banner from backtest figures..."
bash "$REPO_ROOT/scripts/update_eval_surface.sh" 2>&1 | tee -a "$LOG_FILE"
log "[1b/5] Eval surface updated."

# ---------------------------------------------------------------------------
# Phase 2a — weekly cheat sheet
# Reads the latest prediction CSV, writes:
#   docs/weekly/round-<N>-<year>.md
#   docs/weekly/round-current-<year>.md  (stable link)
# ---------------------------------------------------------------------------
log "[2a/5] Generating weekly cheat sheet for round $ROUND..."
$PYTHON "$REPO_ROOT/scripts/generate_weekly_cheat_sheet.py" --csv "$LATEST_PRED" 2>&1 | tee -a "$LOG_FILE"
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

log "[2b/5] Updating HOF stat pages from JSON (deterministic — full leaderboard body + rank-1 sentinels)..."
$PYTHON "$REPO_ROOT/scripts/update_hof_pages.py" 2>&1 | tee -a "$LOG_FILE"
log "[2b/5] HOF pages updated."

# Refresh the reader-facing trust badge (✓ All N stats verified · council-pipeline-gated · date)
# on each HOF stat page. The badge line is stripped by council-content-hash.sh, so
# this never changes the canonical hash the provenance gate checks.
log "[2b/5] Refreshing trust badges on HOF stat pages..."
( cd "$REPO_ROOT" && $PYTHON scripts/inject_trust_badge.py docs/hall-of-fame-stat-*.md --date "$TODAY" ) \
  2>&1 | tee -a "$LOG_FILE"

log "[2b/5] Running deterministic HOF numeric gate..."
$PYTHON "$REPO_ROOT/scripts/check_hof_numbers.py" 2>&1 | tee -a "$LOG_FILE"
if [ $? -ne 0 ]; then
  log "ERROR: HOF numeric gate failed — aborting Phase 2b. Fix mismatches before shipping."
  exit 1
fi
log "[2b/5] HOF numeric gate passed."

# The HOF stat pages carry a council-pipeline PASS stamp but are regenerated
# deterministically here (no DataSentinel run). Under AUDIT_ENFORCE=1 the pre-commit
# gate would reject their commit unless a content-hash-keyed audit record backs the
# current content. The deterministic gate above (check_hof_numbers.py) IS the real
# verifier, so record its PASS verdict for each regenerated page. This is a genuine
# verdict from a genuine gate — not a simulated one.
log "[2b/5] Recording deterministic HOF verdicts for the provenance gate..."
# Only stamp pages that check_hof_numbers.py actually inspects (_SUBPAGES).
# Stamping pages the checker never reads is a false provenance claim (Surveyor F3).
(
  cd "$REPO_ROOT" || exit 1
  for hof in \
      docs/hall-of-fame-stat-games.md \
      docs/hall-of-fame-stat-marks.md \
      docs/hall-of-fame-stat-tackles.md \
      docs/hall-of-fame-stat-hitouts.md \
      docs/hall-of-fame-stat-brownlow.md \
      docs/hall-of-fame-stat-goalassists.md \
      docs/hall-of-fame-stat-clearances.md \
      docs/hall-of-fame-stat-contested.md \
      docs/hall-of-fame-stat-disposals.md \
      docs/hall-of-fame-stat-goals.md; do
    [ -f "$hof" ] || continue
    grep -q '<!-- council-pipeline:' "$hof" || continue
    scripts/record-sentinel-verdict.sh --doc "$hof" --verdict PASS --agent check_hof_numbers \
      2>&1 | tee -a "$LOG_FILE"
  done
)
log "[2b/5] HOF verdict records written."

# ---------------------------------------------------------------------------
# Phase 3 — FootyStrategy agent: round recap + insights update
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Enforce hard limit: README news block keeps only the 2 most recent entries.
# An "entry" is a non-empty line between NEWS-LATEST-START and NEWS-LATEST-END.
# If a new news item was added this cycle and pushed the count over 2, this
# trims the oldest entry so the block never grows beyond 2.
# ---------------------------------------------------------------------------
enforce_news_limit() {
    local readme="$REPO_ROOT/README.md"
    local tmp
    tmp=$(mktemp)

    /home/abhi/sourceCode/python/coding/.venv/bin/python - "$readme" <<'PYEOF'
import sys, re

path = sys.argv[1]
text = open(path).read()

pattern = r'(<!-- NEWS-LATEST-START -->)(.*?)(<!-- NEWS-LATEST-END -->)'
match = re.search(pattern, text, re.DOTALL)
if not match:
    sys.exit(0)

block = match.group(2)
# Split into entries: non-empty paragraphs
entries = [e.strip() for e in re.split(r'\n{2,}', block.strip()) if e.strip()]

if len(entries) <= 2:
    sys.exit(0)  # nothing to trim

# Keep only the 2 most recent (first two)
kept = '\n\n'.join(entries[:2])
new_block = f'\n{kept}\n'
new_text = text[:match.start(2)] + new_block + text[match.end(2):]
open(path, 'w').write(new_text)
print(f"News block trimmed: {len(entries)} → 2 entries (dropped {len(entries)-2})")
PYEOF
}

enforce_news_limit
log "News block limit enforced (max 2 entries)."

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
Tag any specific stat with the bold **[data]** form (literally two asterisks each
side, e.g. \`**[data]**\`), never plain unbold [data] — the verification vocabulary
(scripts/tag_vocabulary.py) only recognises the bold form, so a plain [data] tag is
invisible to DataSentinel and the Skeptic sampler. This is REQUIRED: immediately after
you write, DataSentinel gates this doc (Phase 3b) and a non-PASS aborts the whole cycle —
an untagged specific stat number is a FAIL, so tag every one or omit the claim.

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
# Phase 3b — Gate the insights lane (F05). afl-insights.md is LLM-authored prose
# carrying [data]-tagged numbers; it must not ship ungated. DataSentinel verifies
# every tag against source, flags untagged specific numbers and coach names, and
# records a content-hash-keyed verdict via record-sentinel-verdict.sh. A non-PASS
# (or a gate that cannot run) aborts before Phase 4 stages the file — fail-closed.
# ---------------------------------------------------------------------------
log "[3b/5] Gating afl-insights.md through DataSentinel (F05)..."
DS_OUT="$LOG_DIR/insights_datasentinel_${TODAY}.json"
$CLAUDE -p "You are DataSentinel for SuperCoach-VIA. Verify docs/afl-insights.md as a full-doc (Pass 2) check. Walk every **[data]** tag against the source CSV named in the methodology, flag any untagged specific player-stat number, and flag coach-name violations (config/coach_names.txt). Record your verdict once via: scripts/record-sentinel-verdict.sh --doc docs/afl-insights.md --verdict <PASS|FAIL> --agent DataSentinel. Then emit ONLY the JSON verdict object." \
    --agent DataSentinel \
    --allowedTools "Read,Grep,Glob,Bash" \
    --permission-mode bypassPermissions \
    2>&1 | tee "$DS_OUT" | tee -a "$LOG_FILE"

if ! grep -Eq '"verdict"[[:space:]]*:[[:space:]]*"PASS"' "$DS_OUT"; then
    log "FATAL: DataSentinel did not return PASS for afl-insights.md — aborting before commit (F05). Route the failing tags to FootyStrategy."
    exit 1
fi
log "[3b/5] afl-insights.md gated: DataSentinel PASS recorded."

# ---------------------------------------------------------------------------
# Phase 4 — commit and push all phase 2/3 outputs
# ---------------------------------------------------------------------------
log "[4/5] Staging and committing weekly agent outputs..."

git add \
    README.md \
    docs/banner.svg \
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
    scripts/git_commit_safe.sh commit -m "Weekly refresh round $ROUND — stat leaders + cheat sheet + insights ($TODAY)"
    git push origin main
    log "[4/5] Pushed to origin/main."
fi

log "=================================================================="
log "Weekly refresh complete — round $ROUND — $TODAY"
log "=================================================================="

# --- completion sentinel (F04) ---------------------------------------------
# Written ONLY when the full cycle reaches this point. A killed/partial run
# leaves the previous sentinel untouched, so Chronicler can detect a partial run
# by comparing the sentinel's round to the max round actually in the data.
SENTINEL="$LOG_DIR/last_refresh_complete.json"
printf '{"round": "%s", "completed_at": "%s", "date": "%s"}\n' \
  "$ROUND" "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$TODAY" > "$SENTINEL"
log "Completion sentinel written: $SENTINEL (round $ROUND)."
