# Pending tasks

Engineering / harness / presentation work items surfaced by the Surveyor health
survey (2026-07-07). Ranked by impact-per-engineering-day. Editorial *decisions*
live in [`pending-decisions.md`](pending-decisions.md); this file is *tasks*.

Effort key: S = small, M = medium. Owner in **bold**.

---

## Correctness risks (already escalated to the human)

### CR-1 — Round detection was lexicographic *(FIXED 2026-07-07)*
`scripts/weekly_refresh.sh` selected the "latest" prediction CSV with
`ls … | sort | tail -1`, a lexicographic sort that ranks `next_round_9_*` above
`next_round_18_*` because the string "9" sorts after "1". Weekly recaps shipped
under the wrong round label for weeks. **Fixed:** now selects by mtime
(`ls -t | head -1`), matching `generate_weekly_cheat_sheet.py`'s `getmtime`
semantics so cheat sheet and recap always agree. Regression-guarded by
`tests/unit/test_prediction_selection.py`. Owner: **Gaffer**. Effort: S. Status: DONE.

### CR-2 — QA gate verification snippet KeyErrors on the real JSON schema
QA's embedded check reads `career_games` at the top level of
`docs/hall-of-fame/_stat_leaders.json`, but the real top-level keys are
`['meta', 'categories', 'single_season']` — career stats live at
`categories.career_games.leaders`. Every QA PASS since Sprint 1 is therefore
suspect (the check may have been silently erroring, not verifying). Fix: point
QA.md snippets at the real `categories.*` schema; add a unit test pinning the
schema; add QA-FAIL to Gaffer's never-override table. Owner: **Gaffer**. Effort: S.

### CR-3 — Five stamped docs on main carry known-wrong numbers
Corrections stranded uncommitted 4+ days: Jeremy Cameron goals 83→88,
Pendlebury 435→436, Hewett 209→213. The Brown doc has hand-edited Skeptic
verdict text (edited *after* the verdict — a provenance violation). The
list-quality diff pre-implements pending **decision #2** before that decision is
recorded. Fix: run all five docs back through DataSentinel → fresh audit records
→ commit → push. **Decision #2 must be recorded first** (the list-quality diff
pre-implements its Option A). Owner: **Gaffer** (chain re-run). Effort: S–M.

### CR-4 — afl-insights.md ships ungated LLM-authored numbers
The weekly recap lane writes numbers with no DataSentinel gate; unbold `[data]`
tags are invisible to `tag_vocabulary.py`, DataSentinel, and the badge count, so
there is no sentinel record. Ungated LLM numerics reach users every cycle. Fix:
gate the insights lane (see task 5). Owner: **Gaffer**. Effort: M.

---

## Act now (top 3)

1. **Fix round detection (CR-1)** — use mtime or numeric sort; add unit test;
   repair the insights header; decide append-vs-replace for recap history.
   Owner: **Gaffer** + Scientist. *(Fix + test DONE 2026-07-07; append-vs-replace
   for recap history still open.)*
2. **Unstrand stat corrections (CR-3)** — five docs through DataSentinel → fresh
   audit records → commit → push. Pending decision #2 must be recorded first
   (list-quality diff pre-implements Option A). Owner: **Gaffer** (chain re-run).
3. **Restore QA gate authority (CR-2)** — fix QA.md snippets to the real
   `categories.*` schema; add a unit test pinning the schema; add QA-FAIL to
   Gaffer's never-override table. Owner: **Gaffer**.

---

## Sprint 2 backlog (next 5)

4. **Single-entry-point refresh discipline** — `weekly_refresh.sh` is the only
   sanctioned cycle entry; add a completion sentinel so Chronicler can detect
   partial runs. Owner: **Gaffer**.
5. **Gate the insights lane (CR-4)** — DataSentinel pass + stamp on
   `afl-insights.md` before Phase 4 commits; clean up unbold `[data]` tags.
   Owner: **Gaffer**.
6. **BriefBuilder gets Bash tool + de-hardcoding** — computation via venv Python;
   era/coach/FanFooty facts read from `config/`, not prompt-frozen. Owner: **Gaffer**.
7. **Verdict vocabulary unification + Skeptic records** — one enum across
   Skeptic, Gaffer stamp, and `check-council-stamp.sh`; exact-token match;
   defined policy for `PASS_WITH_CONCERNS`. Combines with the F5
   Skeptic-verdict-record backlog item. Owner: **Gaffer**.
8. **Architecture consolidation** — one canonical `architecture.md`;
   `ARCHITECTURE.md` becomes a redirect stub; all nine prompts cite the canonical
   one; FootyStrategy and Scientist get chain-position/handoff sections.
   Owner: **Gaffer**.

---

## Medium priority

9. **all_time_top_100.csv schema divergence** — root CSV (100×4:
   Serial/Player/Teams/Comment) vs `data/top100/` (100×2: player/all_time_score).
   CLAUDE.md conflates them; QA and DataSentinel reference different paths.
   Owner: **Scientist**.
10. **stat_coverage_eras.yaml missing stats** — `free_kicks_for`,
    `free_kicks_against`, `percentage_of_game_played` all exist in the player
    schema but are absent from the era config, so the gate is blind for them.
    Owner: **Scientist**.
11. **phantom_row_validator.py `gaps_in_season()` wired into nothing** — built as
    a current-season abort gate but never called by the harness. Wire into
    Phase 1. Owner: **Gaffer**. Effort: S.
12. **8 Round-14 briefs have 15–16 unresolved `FOOTYSTRATEGY INSERT`
    placeholders each.** Owner: **FootyStrategy**.
13. **Harness hygiene bundle** — `enforce_news_limit` doesn't verify the dropped
    entry exists in `docs/news/README.md`; the pre-commit hook fail-opens on a
    missing check script; `weekly_refresh.sh:121` is dead code under `set -e`;
    the phantom validator ignores finals WARNINGs in scan mode. Owner: **Gaffer**.
14. **Untested scripts** — `weekly_refresh.sh` (incl. the round-detection line),
    `update_eval_surface.sh`, `log-agent-turn.sh`, `package_fan_pack.sh`, and the
    live/R10 suite (8 scripts). Owner: **Scientist** (write) / **Gaffer** (enforce).
15. **Prompt hygiene bundle** — Gaffer defines the canonical chain twice with
    different membership (one omits QA/Chronicler); BriefBuilder hardcodes era
    years and points at a memory file instead of `config/coach_names.txt`; the
    `[unverified]`-tag ratio has no cap; the Gaffer→Chronicler handoff is
    asymmetric; `git_commit_safe.sh`'s usage comment suggests `add -A`.
    Owner: **Gaffer**.
16. **Stale user surface** — `afl-season-2026.md` last committed 2026-05-08
    despite being listed as Phase-1 auto-committed; `model-report-card.md`
    2026-05-12. Owner: **Gaffer** → **Scientist** if Phase-1 wiring is broken.
17. **Trust badge deployed on only 2 of 16 news docs** (7 are stamped and
    badge-eligible). The differentiator is missing from most of the surface it
    was built for. Owner: **Gaffer**. Effort: S.

---

## New anti-pattern (added to the Surveyor list)

> Never derive "latest file" by lexicographic sort — sort numerically on the
> extracted field or by mtime.

---
---

# Surveyor deep-read findings — 2026-07-07 (second pass)

Seven genuinely new opportunities from a deep read of the model/data code,
separate from the 18 above. Sequencing by impact-per-day: S-1 → S-2 → S-5 best;
S-4 biggest product move; S-3 unblocks named-team features; S-6/S-7 are
backtest-gated experiments. **S-1 and S-2 are escalations to the human**
(published claim / numbers contradicted by code+data).

### S-1 — Phantom model features + false published "venue effects" claim *(ESCALATION)*
`prediction.py` declares `cba_percent` and `percentage_time_played` as features
but neither column exists in training data. CBA has no data source. The TOG
column in player CSVs is `percentage_of_game_played`; the RENAMES map never maps
it, and a silent `if feat in df.columns` filter drops both with no warning.
`docs/how-to-use-this-for-supercoach.md:29` publicly claims the model handles
"venue effects" — venue is not in player data either. Evidence: prediction.py
:112-116, :184-187, :385, :450; player CSV headers; how-to-use:29.
**Action:** Scientist — add `percentage_of_game_played → percentage_time_played`
to RENAMES, hard-assert every declared extra feature survives loading (no silent
drops), backtest the delta, delete/­source `cba_percent`. Gaffer — correct or
remove the "venue effects" claim in the published doc.
Owner: **Scientist** (code) + **Gaffer** (doc). Effort: S · Impact: High.

### S-2 — team_stats_conceded_2025.csv columns scrambled (wrong, not stale) *(ESCALATION)*
Row 1 claims Hawthorn conceded 123 disposals to Sydney; re-summing Sydney's
player rows = 321. Columns are scrambled — the file is corrupt. Every brief
currently omits conceded stats ("2026 not available"); the correct numbers are a
~30-line groupby over `player_data/`. No published figures are contaminated
(briefs never used its numbers). Evidence: conceded CSV row 1 vs pandas recompute;
BriefBuilder.md:81,:216; carlton-vs-geelong-round-13-2026.md:259.
**Action:** Scientist writes `compute_conceded_stats.py` (regenerate per-season
from player_data, with a test comparing one match to manually-summed player rows),
retires the corrupt 2025 file, wires into weekly refresh; BriefBuilder reference
becomes a live table. Owner: **Scientist**. Effort: S · Impact: High.

### S-3 — Lineup scraper writing garbage since 2025 R3 (15 months undetected)
`data/lineups/team_lineups_*.csv` corruption: 0% through 2024, 96% in 2025, 100%
in 2026. Players field holds jersey numbers + table-footer junk
(`"29;41;21;…;Rushed;Totals;Oppositi"`). Source page format changed ~March 2025;
`game_scraper.py` kept parsing the wrong cells. Undetected because nothing reads
lineup output. Clean lineups unlock "is this player named this week" and
teammate-absence/role-vacancy features over 117 years. Evidence: bad-row fraction
2024=0.00/2025=0.96/2026=1.00; sample corrupt 2025 R3 row.
**Action:** Scientist fixes the parser in `scrapers/game_scraper.py`
(fixture-based regression test on the new format, TDD), backfills 2025-2026, adds
a content validator ("players field must be alphabetic names, no Totals token").
Owner: **Scientist**. Effort: M · Impact: High.

### S-4 — AFL Fantasy points prediction is one target-swap away
Fantasy scoring is a fixed public linear formula (kicks×3, handballs×2, marks×3,
tackles×4, frees_for×1, frees_against×−3, goals×6, behinds×1, hitouts×1) — every
column already exists in player CSVs. Computing historical `fantasy_points` is one
pandas expression; the pipeline (rolling features, GroupKFold, Optuna, backtest)
retrains on the new target unchanged. Directly answers the fantasy question and
fixes the disposals-only blind spot that buries rucks and key forwards. Evidence:
R19 pred CSV (disposals only); player CSV headers; prediction.py:184-186
(target-agnostic); how-to-use:53-55.
**Action:** Scientist adds derived `fantasy_points` target, trains a companion
model, full-season backtest for an MAE band before ship; cheat sheet gains a
`predicted_fantasy_points` column. Owner: **Scientist**. Effort: M · Impact: High.

### S-5 — Predictions are bare point estimates; floor/ceiling derivable
Weekly product is a single integer per player, but captaincy/trades are
distribution questions. LightGBM has native quantile objectives; the backtest
already computes `pct_within_5`/`pct_within_10` per round, so interval-coverage is
a trivial extension. Docs already gesture at a uniform ±8 band when the model
could differentiate safe 28s from volatile ones. Evidence: R19 pred CSV (3 int
cols); backtest_by_team CSV coverage cols; how-to-use:79; prediction.py:566-568.
**Action:** Scientist trains p10/p90 quantile companions, adds interval-coverage
to the backtest report, extends the prediction CSV schema; Gaffer updates the
cheat sheet template to render floor/ceiling. Owner: **Scientist** + **Gaffer**.
Effort: M · Impact: High.

### S-6 — By-position backtest dead since built (all "Unknown")
Every backtest writes `backtest_by_position_<ts>.csv` = one row: `Unknown, n=320`.
`backtest.py:24` admits the dataset stores no position; the one file with a
position column (`data/contracts/afl_2026_contracts.csv`) is empty. A role label
is derivable: hitouts→ruck, marks_inside_50+goals→forward,
rebound_50s+one_percenters→defender, clearances/contested→midfielder. Evidence:
by-position CSV single row; backtest.py:24; player CSV stat columns.
**Action:** Scientist builds `derive_positions.py` (validate vs ~30 known
players), feeds the existing by-position path, tags output "derived role, not
official". Owner: **Scientist**. Effort: M · Impact: Medium.

### S-7 — Age and career experience never enter the model
Every player has a `*_personal_details.csv` (born_date, debut_date, height,
weight); the pipeline's only use is a coarse ±40-year liveness filter.
Age-at-game and career experience are among the best AFL output predictors
(2nd-4th-year breakouts, veteran decline) — exactly where a rolling-average model
lags. Both computable at load: age from born_date vs game date; experience from
the `games_played` counter already in every performance row. Evidence:
prediction.py:295,302,856,873; personal_details schema; games_played per row;
feature list prediction.py:387 (no age/experience).
**Action:** Scientist adds `age_at_game` + `career_games` features, backtests
gated on MAE improvement in young-player and veteran cohorts (slice by
career_games tercile), ships only if those slices improve. Owner: **Scientist**.
Effort: S–M · Impact: Medium.

---

## Cross-cutting (Surveyor)

- **Root pattern across S-2, S-3, S-6:** *data collected but never consumed is
  data never validated.* Three corruptions persisted because no pipeline step
  read the output. Cheap systemic fix: a content-shape validator over every
  dataset the refresh touches.
- **New anti-pattern:** "Data collected but never consumed is data never
  validated — add a content validator for every dataset the refresh writes, even
  if no model reads it."
- **Housekeeping:** three stale agent worktrees under `.claude/worktrees/`
  (~560 MB) — clean up when convenient.

---

*Created 2026-07-07 from the Surveyor health survey (both passes). Route questions to Gaffer.*
