# Survey — 2026-07-13 — scope: PULSE (R19/R20 weekly refresh cycle, commit 9e617a167)

## Executive read
The harness fixes worked: phantom gate aborted BEFORE any push (S11-F7 fix verified live), `--from-csv` scored the archived R19 CSV without retraining, and the RUN_START guard correctly labelled round 20. But the cycle is INCOMPLETE — DataSentinel FAILed afl-insights.md (11/11 tags, no declared source) and the harness FATAL-aborted at Phase 3b; README, insights, and cheat sheet are gated-but-uncommitted. One RED: the counter-rescued Clarke game #14 shipped to main with a known-wrong date (C1 recurrence, now published).

## Subsystem status

### 1. Phantom-row gate — AMBER (worked as designed; root cause was a code bug, not data timing)
- Evidence: log 18:41:08 `ERROR Phantom-row gate: clarke_angus_08052006 has a 2026-season counter gap [14]` → RuntimeError ABORT (refresh_data.py:247), zero pushes made ("Push deferred to parent harness"). Fix commit a4c1fdc20 20:13:51 "Fix delta-scraper stranding games with non-chronological afltables round labels". Re-run 20:22:49 → 20:25:41 "Phantom-row check clean across 7 updated file(s)". Second gate pass 20:56:36 before push.
- Verdict on the re-run question: NOT afltables settlement — a scraper code bug (stranded game with non-chronological round label), fixed mid-cycle, then the rescued row scraped in. The gate fired correctly both times.
- **RED sub-finding [class 1/5, C1 recurrence — ESCALATED TO HUMAN]:** the rescued row is now on main (9e617a167) with wrong metadata: clarke_angus game 14 = `year=2025, round=1, date=2025-03-01` sitting after round 25 (measured via pandas; real date ~2025-08-27 per Scientist's own test comment). Published data row is wrong; model temporal ordering poisoned for this row.
- Owner: Scientist — fixture-date resolution for counter-rescued numeric-round rows + correct the committed Clarke row. Effort S.

### 2. Backtest `--from-csv` archive mode — GREEN
- Evidence: backtest_run_20260713_205008.log:8 "scoring archived prediction CSV next_round_19_prediction_20260707_1606.csv (no retrain)"; no new next_round_* CSV written by backtest; zero ERROR/WARN lines in the log (54 lines, all INFO).
- F13 namespace-pollution path not exercised this cycle; cheat sheet consumed round 20 correctly.

### 3. RUN_START freshness guard — GREEN
- Evidence: weekly_refresh.sh:48 `RUN_START=$(date +%s)`, :97-101 aborts if latest prediction mtime < RUN_START. R20 CSV next_round_20_prediction_20260713_2050.csv mtime 20:50 > run start 20:22:49; log "Detected next round: 20" proceeded. S11-F4 stale-round-relabel class: retired, verified live.

### 4. MAE health — AMBER (informational)
- Measured from summary CSVs: R16=3.981 (n=322) → R17=3.825 (n=320) → R18=3.767 (n=412; a 07-10 rescore on n=284 gave 3.609) → **R19=3.973 (n=371)**. Worst since R16, within normal band; bias −0.13; pct≤5 74.7% vs 77.9% at R18. Cumulative R1–R19 eval surface: MAE 3.960.
- **S1b TOG%-lag fix is UNMEASURABLE at R19**: the archived R19 prediction was generated 2026-07-07 16:06, before the fix landed. First fix-exposed prediction is the R20 CSV made this cycle — the needle reads next week. Do not attribute R19's 3.97 to S1b either way.
- Owner: Scientist — note in experiment log that R20 backtest is the first S1b-lagged datapoint. Effort S.

### 5. Commit 9e617a167 contents — AMBER
- Evidence: `git show --stat` — 449 files: player CSVs, matches_2026.csv (+9 R19 rows), charts, all_time_top_100.csv (root + copy), docs/afl-{backtest,brownlow,finals,predictions,stat-leaders,team-analysis,team-profiles}, hall-of-fame-top100.md.
- NOT in the commit (Phase 2+, blocked by abort): README.md, docs/afl-insights.md, docs/weekly/round-20-2026.md cheat sheet — all generated/gated but sitting staged-uncommitted in the tree. `last_refresh_complete.json` absent (explained by abort; F04 watch still pending a clean full cycle).
- Owner: Gaffer — complete Phase 2–5 ship through the sanctioned harness path (afl-insights.md now has a PASS at its current hash, see §6); do not commit ad hoc. Effort S.

### 6. Gate verdicts — AMBER (gates worked; remediation loop closed post-abort)
- check_hof_numbers: PASS records 20:58 for HOF stat sub-pages (e.g. sentinel-f063e94…, f1c8d14…).
- DataSentinel on afl-insights.md: **FAIL** 21:00:49 (11/11 tags failed — no source file declared in methodology paragraph; 1 untagged number) → harness FATAL 21:00:58, correctly aborted before commit. Post-abort: FAIL 23:16 (hash cc08ef9d), **PASS 23:20 (hash 9357e497)** — sha256 of current docs/afl-insights.md == 9357e497. No same-hash PASS/FAIL conflict (F1 class clean this time).
- Skeptic: not invoked — weekly recap is Skeptic-exempt by documented convention. No FAIL stamps in committed docs.

### 7. Scraper audits (bycatch) — AMBER
- Match audit (both runs): matches_2026.csv missing 9 games — R10 6/9, R17 4/7 scraped. Persists after the a4c1fdc20 fix; forward-only delta never backfills (S11-F1 class).
- Player audit run 1: 28 career-total warnings, worst brown_callum_15082000 csv=76 games vs afltables=49, goals 80 vs 19 — pattern consistent with a wrong-player merge. Run 2: unwin_rhys csv=1 vs afltables=3 (missing rows).
- Owner: Scientist — triage brown_callum merge suspicion + matches_2026 R10/R17 backfill. Effort M.

## Anti-pattern list (standing)
Unchanged from 2026-07-11 survey; this cycle produced one reinforcement: the phantom gate's value came precisely from running BEFORE push — never re-order a gate after its push.

## Watch list (speculation, unranked)
- R20 backtest (next cycle) is the first S1b-lagged MAE reading — compare against the R16–R19 band 3.77–3.98.
- Post-abort remediation edits to afl-insights.md happened outside the harness (23:16–23:20); confirm the eventual commit goes through the sanctioned path, not a REPL commit.
- unwin_rhys and the other 27 run-1 player-audit warnings — check whether they self-heal next scrape.

## What was checked and found clean
backtest_run_20260713_205008.log (0 errors); `--from-csv` no-retrain path; RUN_START guard code + live behaviour; round-20 detection and cheat sheet round labelling; phantom gate pre-push ordering (S11-F7 retired); DataSentinel same-hash consistency (PASS at 9357e497 only); check_hof_numbers PASS records; Phase 1 push a4c1fdc20..9e617a167 single-committer.
