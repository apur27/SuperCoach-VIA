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
#              + update_hof_pages.py (deterministic, no DataSentinel agent)
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

# F3: HARNESS_PHASE is stamped into every log line and exported to child steps.
# On an abort, the last line in the log names the phase that died — no more
# guessing which phase a killed run stopped in (the 07-14 Phase-4 mid-commit death).
export HARNESS_PHASE="init"
log() { echo "[$(date '+%H:%M:%S')] [phase ${HARNESS_PHASE}] $*" | tee -a "$LOG_FILE"; }

mkdir -p "$LOG_DIR"
log "=================================================================="
log "Weekly refresh started — $TODAY"
log "=================================================================="

RUN_START=$(date +%s)

# ---------------------------------------------------------------------------
# Phase 0 — deterministic round-settlement probe (F5)
# Replaces the day-of-week timing heuristic entirely. Reads the current season's
# matches CSV, finds the current (highest) home-and-away round, and confirms
# every game present for that round has a non-zero final score. If any game is
# still 0-0 (unplayed / mid-play), abort: running the cycle on an unsettled round
# scrapes and publishes half-finished results. Fail-closed on a missing file.
# ---------------------------------------------------------------------------
export HARNESS_PHASE="0"
log "[0/5] Round-settlement probe: confirming the current round's scores are final..."
if ! $PYTHON "$REPO_ROOT/scripts/check_round_settled.py" 2>&1 | tee -a "$LOG_FILE"; then
    log "FATAL: current round is unsettled (games without final scores) — aborting before Phase 1. Re-run once the round's scores are confirmed."
    exit 1
fi
log "[0/5] Round-settlement probe passed — current round is settled."

# ---------------------------------------------------------------------------
# Phase 1 — data + model pipeline
# refresh_and_rank.sh commits Phase 1 artifacts but defers the push (detected
# via WEEKLY_REFRESH_PARENT=1) so the phantom-row gate can run before anything
# reaches origin.
# ---------------------------------------------------------------------------
export HARNESS_PHASE="1"
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
log "[1c/5] Phantom-row gate passed."

# ---------------------------------------------------------------------------
# Match-completeness gate (F8/E2) — promotes the per-round fixture audit from
# warnings-only to BLOCKING. If matches_<year>.csv is missing any scheduled
# matchup for a home-and-away round (the "R10 2026" bug, where 6 of 9 games were
# silently dropped and predictions ran on incomplete data for weeks), abort
# before the Phase 1 push. Fails OPEN on a fixture/network outage. Gate on
# audit_match_rounds — check_match_completeness cannot catch a single dropped
# round (each affected team is only 1 game short, under its >=2 threshold).
# ---------------------------------------------------------------------------
log "[1c/5] Match-completeness gate: verifying every scheduled matchup was scraped..."
if ! $PYTHON "$REPO_ROOT/scripts/match_completeness_gate.py" 2>&1 | tee -a "$LOG_FILE"; then
    log "FATAL: match-completeness gate found incomplete round(s) — Phase 1 commit NOT pushed. Backfill the missing matches (re-run the match scraper) before shipping."
    exit 1
fi
log "[1c/5] Match-completeness gate passed. Pushing Phase 1 commit..."
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
export HARNESS_PHASE="1b"
log "[1b/5] Updating README eval surface + banner from backtest figures..."
bash "$REPO_ROOT/scripts/update_eval_surface.sh" 2>&1 | tee -a "$LOG_FILE"
log "[1b/5] Eval surface updated."

# ---------------------------------------------------------------------------
# Phase 2a — weekly cheat sheet
# Reads the latest prediction CSV, writes:
#   docs/weekly/round-<N>-<year>.md
#   docs/weekly/round-current-<year>.md  (stable link)
# ---------------------------------------------------------------------------
export HARNESS_PHASE="2a"
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
export HARNESS_PHASE="2b"
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
# F1: the navigation hub (docs/hall-of-fame-stat-leaders.md) is now gated by
# check_hof_hub() inside check_hof_numbers.py above — its rank-1 figures were
# verified against _stat_leaders.json in the numeric gate that just passed. Record
# its PASS so the pre-commit stamp gate accepts the regenerated hub instead of
# fail-closing and reverting it (the Phase 4 death in the 07-13 / 07-14 cycles).
(
  cd "$REPO_ROOT" || exit 1
  hub=docs/hall-of-fame-stat-leaders.md
  if [ -f "$hub" ] && grep -q '<!-- council-pipeline:' "$hub"; then
    scripts/record-sentinel-verdict.sh --doc "$hub" --verdict PASS --agent check_hof_numbers \
      2>&1 | tee -a "$LOG_FILE"
  fi
)
log "[2b/5] HOF verdict records written (subpages + hub)."

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

export HARNESS_PHASE="3"
log "[3/5] Invoking FootyStrategy agent for round $ROUND weekly insights..."

$CLAUDE -p "Round=$ROUND. Date=$TODAY. Write '## Round $ROUND — Week in Review' in docs/afl-insights.md (immediately after the intro table). Sources: docs/afl-stat-leaders-2026.md, docs/afl-season-2026.md, docs/afl-predictions-2026.md, docs/weekly/round-current-2026.md. 150-200 words max. Do not touch the navigation table, intro text, links, or any other file." \
    --agent FootyStrategy \
    --permission-mode bypassPermissions \
    2>&1 | tee -a "$LOG_FILE"

log "[3/5] FootyStrategy agent complete."

# ---------------------------------------------------------------------------
# Phase 3b — Gate the insights lane (F05). afl-insights.md is LLM-authored prose
# carrying [data]-tagged numbers; it must not ship ungated. DataSentinel verifies
# every tag against source, flags untagged specific numbers and coach names, and
# records a content-hash-keyed verdict via record-sentinel-verdict.sh. A non-PASS
# (or a gate that cannot run) aborts before Phase 4 stages the file — fail-closed.
# ---------------------------------------------------------------------------
DS_OUT="$LOG_DIR/insights_datasentinel_${TODAY}.json"

# Run DataSentinel Pass-2 on the insights doc; verdict lands in $DS_OUT.
# Returns 0 iff DataSentinel recorded PASS.
gate_insights() {
    $CLAUDE -p "Full-doc Pass 2 check on docs/afl-insights.md. Record verdict via: scripts/record-sentinel-verdict.sh --doc docs/afl-insights.md --verdict <PASS|FAIL> --agent DataSentinel. If FAIL, include a \"failed_tags\" array naming each specific number/phrase that failed and why. Emit ONLY the JSON verdict object." \
        --agent DataSentinel \
        --permission-mode bypassPermissions \
        2>&1 | tee "$DS_OUT" | tee -a "$LOG_FILE"
    grep -Eq '"verdict"[[:space:]]*:[[:space:]]*"PASS"' "$DS_OUT"
}

export HARNESS_PHASE="3b"
log "[3b/5] Gating afl-insights.md through DataSentinel (F05, pass 1)..."
if gate_insights; then
    log "[3b/5] afl-insights.md gated: DataSentinel PASS recorded (pass 1)."
else
    # F2: bounded single retry. Feed the failing tags back to FootyStrategy,
    # let it re-write the section, then re-gate exactly once. A second FAIL
    # aborts — we never loop forever on a doc DataSentinel keeps rejecting.
    log "[3b/5] DataSentinel FAIL on pass 1 — extracting failed tags and re-invoking FootyStrategy (F2 retry)..."
    FAILED_TAGS=$(grep -oE '"failed_tags"[[:space:]]*:[[:space:]]*\[[^]]*\]' "$DS_OUT" || true)
    [ -n "$FAILED_TAGS" ] || FAILED_TAGS=$(tr -d '\000' < "$DS_OUT" | tail -c 1500)
    $CLAUDE -p "Your Round $ROUND recap in docs/afl-insights.md FAILED DataSentinel. These tags/numbers failed: ${FAILED_TAGS}. Fix ONLY the '## Round $ROUND — Week in Review' section: add a bold **[data]** tag to every specific number, name the source CSV for each in the methodology sentence, and re-verify each figure against data/. Re-write the section and hand back. Touch no other section or file." \
        --agent FootyStrategy \
        --permission-mode bypassPermissions \
        2>&1 | tee -a "$LOG_FILE"

    log "[3b/5] Re-gating afl-insights.md through DataSentinel (F05, pass 2)..."
    if gate_insights; then
        log "[3b/5] afl-insights.md gated: DataSentinel PASS recorded (pass 2, after retry)."
    else
        log "FATAL: DataSentinel FAILed afl-insights.md twice (pass 1 + retry) — aborting before commit (F05). Human review required; do not ship the recap."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Phase 4 — commit and push all phase 2/3 outputs
# ---------------------------------------------------------------------------
export HARNESS_PHASE="4"
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
    # F3: tee the commit AND push through the log. Phase 4 died mid-commit in the
    # 07-14 run with no captured output; under `set -o pipefail` a failing commit or
    # push still aborts (the pipe carries the left-hand exit code) but now the log
    # captures exactly what git said before it died.
    scripts/git_commit_safe.sh commit -m "Weekly refresh round $ROUND — stat leaders + cheat sheet + insights ($TODAY)" 2>&1 | tee -a "$LOG_FILE"
    git push origin main 2>&1 | tee -a "$LOG_FILE"
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
