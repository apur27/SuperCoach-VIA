# Model report card

> [← Back to main README](../README.md) | [← Back to fan landing page](start-here-no-code.md)

**This page has moved.** The model's pre-registered hit/miss methodology, per-round accuracy table, and weekly results have been merged into a single canonical document:

# → [Backtest results — 2026](afl-backtest-2026.md)

The merged page contains everything that used to live here, kept in sync with the latest run of `backtest.py`:

- **Per-round accuracy table** (MAE, RMSE, % within 5, % within 10, bias, n) — auto-refreshed every data update
- **Cumulative season-to-date numbers** — 3,701 player-predictions across 10 rounds, MAE 4.086, RMSE 5.195, bias −0.090 (essentially unbiased)
- **Team-level bias table** — all 18 clubs, sorted from most under-predicted to most over-predicted
- **Round-by-round notable misses** — top 5 over- and under-predictions per round, the players where the model was most wrong
- **Top-30 disposal players** — the watchlist of "model has not figured this player out yet"
- **Pre-registered methodology** — metrics, hit/miss definitions, what we commit to reporting

## Why one page instead of two

We had two pages doing the same job — a per-round MAE/RMSE/within-5 table appeared on both this report card and on the backtest page, which meant numbers could drift apart on a refresh. The backtest page is auto-updated by `update_team_analysis.py` between the `<!-- 2026-BACKTEST-START -->` markers, so it is always the freshest source. Folding the report card's pre-registered methodology and narrative into the same page keeps the audit trail and the numbers in one place.

This redirect is preserved (rather than deleted) so that existing links from the README, glossary, fan setup guides, and external bookmarks continue to work.

---

## Related

- [Backtest results — 2026](afl-backtest-2026.md) — **the canonical doc**, includes everything that was here
- [How to use this for SuperCoach](how-to-use-this-for-supercoach.md)
- [Glossary](glossary.md) — hit/miss definitions
- [Data science deep-dive](data-science.md) — the model and the backtest framework
- [Prediction model overview](prediction-model.md)
