---
name: weekly-r20-retro
description: 2026-07-13 R20 weekly refresh — two fail-closed recoveries (phantom-row scraper bug + afl-insights DataSentinel double-fail), HOF hub held back on drift; open backlog
metadata:
  type: project
---

Weekly refresh for **Round 20 prediction / Round 19 backtest**, shipped 2026-07-13. Harness fail-closed TWICE; recovered both via the council chain. Final ship: `2c50e6736` (Phase 2/3 docs) + `9e617a167` (Phase 1 data/pred/backtest) + `a4c1fdc20` (scraper fix) + `6149b1ed4` (run report).

**Why:** first genuine end-to-end weekly cycle where both a data-integrity gate and a content gate fired on the same run — captures the recovery playbook and the harness gaps that surfaced.

**How to apply:** on any future weekly refresh, expect these and use the recovery moves below.

## What broke and the recovery
1. **Phantom-row gate ABORT (run 1)** — a player's career `games_played` counter jumped (game dropped). Root cause was a scraper delta-filter bug (approximate-date filter stranded a real game afltables mislabelled with a low round number). Fix: Scientist made the delta **counter-aware** (`a4c1fdc20`, +3 tests) and restored the missing row from afltables. Recovery = route to Scientist, commit the code fix myself (harness allowlist never stages `scrapers/`), re-run harness. See [[canonical-games-metric]], [[phantom-row-dedup-gate]].
2. **DataSentinel(P2) FAIL on `afl-insights.md` (run 2, Phase 3b)** — twice: first the recap's [data] tags had no methodology paragraph declaring source; then FootyStrategy's *fix* introduced 3 untagged bare numbers **inside** the new methodology footnote. Lesson: **methodology/source footnotes must name files, not restate figures** — DataSentinel's untagged-number check is occurrence-based. Two FootyStrategy passes → PASS.

## Harness gaps this cycle exposed (backlog, ranked)
1. **HOF hub (`hall-of-fame-stat-leaders.md`) is ungated but in the Phase-4 commit allowlist.** `update_hof_pages.py` auto-updates its rank-1 cells and bumps its stamp date, but `check_hof_numbers.py` does NOT read the hub (only the 10 sub-pages via HOF-TOP sentinels), so no fresh verdict record is written → the pre-commit council-stamp gate **blocks the hub whenever its numbers change** (i.e. whenever a rank-1 player's game count ticks over). It also carries hand-written `[data]` prose that drifts. **Recurs every such week.** Workaround used: `git restore --staged docs/hall-of-fame-stat-leaders.md`, ship the rest. Permanent fix = Chronicler's top rec: bring the hub under `check_hof_numbers.py` + stamp loop. Owner: Gaffer (harness) + Scientist. See [[drawn-gf-dedup-defect-hub-drift]], [[council-stamp-gate-scope]].
2. **Kicks/handballs figures inconsistent across 3 surfaces** — `_stat_leaders.json` vs `hall-of-fame-stat-kicks-handballs.md` vs hub prose; none reconcile. The kicks-handballs page is not deeply verified by `check_hof_numbers` yet passes the stamp gate. Owner: Scientist.
3. **Scientist F1 (HIGH):** the restored scraper row carries an approximated date (year-start) not its true date — mis-sequences it for form features. Fix before next model retrain. **F2 (MED):** renumbering-duplicate vector the phantom gate can't see. **Gate-1 residual:** year-level skip can strand a retroactive earlier-season backfill.
4. **QA chart-count spec mismatch (WARN):** QA expects ≥6 `alltime_top20_*.png`; generator makes 4. Decide: update spec or add 2 categories. Owner: Surveyor/Scientist.
5. **Per-team bias gate (Chronicler rec #2):** add `check_prediction_bias.py` to the harness, threshold ~1.6× global MAE / |bias|>2.5 (backtest_by_team now stable enough to calibrate). Owner: Gaffer/Scientist.

## Chronicler top recommendation
**Gate the HOF hub** (#1 above) — converts a recurring manual exception into a permanent gate; highest impact-per-day.

## Process notes that worked
- The harness has **no resume-from-phase** entry point. After a Phase-3b abort, re-running the whole thing risks FootyStrategy regenerating the same failing recap. Recovery = **complete Phase 3b/4 manually** (route the fix to the owning agent, re-gate through DataSentinel, stamp/commit via `git_commit_safe.sh`). Faithfully replicate the harness Phase-4 `git add` allowlist rather than committing everything dirty (lineups, top100/yearly, agent-memory are intentionally left out weekly).
- `afl-insights.md` is gated by the **DataSentinel content-hash ledger**, not an inline council-pipeline stamp — do NOT add a stamp/badge (it changes the hash and breaks the ledger match at pre-commit). The harness Phase 4 adds neither.
- Consulted Surveyor before committing the scraper fix (standing directive) — confirmed the counter-aware delta is safe and corroborated the timing concern. See [[consult-surveyor-before-committing]].
