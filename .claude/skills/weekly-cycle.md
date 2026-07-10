---
name: weekly-cycle
description: Run the full weekly refresh pipeline with the Tuesday-settlement guard — scrape actuals, rank, predict, backtest, HOF recompute, weekly recap, QA gate, ship, and Chronicler. Invoke as "/weekly-cycle". Use for the routine weekly data refresh after a round settles, not for a brief.
---

# /weekly-cycle — weekly refresh pipeline

Invoked by the coordinator as `/weekly-cycle`. This is the no-brief cadence cycle:

```
scripts/weekly_refresh.sh → Chronicler
```

`scripts/weekly_refresh.sh` is the **single orchestrator** for all pipeline phases. This skill invokes it and handles the before/after bookkeeping. Never re-choreograph phases here — that is what caused drift in the allowlist and phase order.

Gaffer owns process, not truth: never override a gate FAIL, never author a `[data]` number, never `git push --force`.

## Cadence rule

Run **on Tuesday after the round settles.** afltables does not have the weekend's actuals until then. Do not hardcode a specific round or date; the evergreen Tuesday-settlement rule is the whole cadence.

## Phase 0 — Timing guard

Before doing anything, check the calendar:

- If today is **Tuesday or later** and the weekend round has completed, proceed.
- If run **before Sunday's round completes**, afltables won't have the data yet. **Warn the user** ("round not settled — afltables won't have this round's actuals; scraping now will pull stale/partial data") and **ask for explicit confirmation** before proceeding. Do not auto-proceed on an unsettled round.

## Phase 1 — Run the harness

```bash
bash scripts/weekly_refresh.sh
```

This single command runs all pipeline phases in order:
- Scrape (matches + players, delta mode)
- Rank → predict (forward) → backtest-by-archive (scores prior round's archived CSV, no retrain)
- Phantom-row gate, eval surface update, cheat sheet
- HOF pipeline + numeric gate
- Weekly recap (FootyStrategy) + DataSentinel gate
- Allowlist commit + push
- Completion sentinel (`last_refresh_complete.json`)

**Long-running (~35–45 min after Optuna cache warm).** Watch for: non-zero exit (any gated phase fail-closes — do not push around it), player game-count drops (route to Scientist), `ROUND=unknown` exit (means no prediction CSV exists — run `refresh_and_rank.sh` Phase 1 first, then retry).

**Never wrap in `timeout`.**

## Phase 2 — Chronicler

After the harness exits 0 and the push succeeds:

- **Invoke:** Chronicler agent.
- **Pass:** the commit hash and cycle type `"weekly-refresh"`.
- **It produces:** the run report and top-3 expansion recommendations.

## Retro

Verify `last_refresh_complete.json` was written (F04 sentinel). Log one line to Gaffer memory: what broke, top Chronicler recommendation, what to add to the backlog.
