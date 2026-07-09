---
name: dropna-denominator-coverage-bias
description: Per-game rate scans that divide by only games-with-a-recorded-stat inflate pre-1990s players; use fill-zero (all games played) as denominator
metadata:
  type: feedback
---

Cross-player "rarity" / per-game-rate scans MUST divide by **all games played** (fill-zero
convention), not by the count of games where the stat was recorded. Dividing by the recorded
subset (a `dropna` denominator) silently inflates pre-1990s / early-1970s players whose
disposals/tackles were only partially recorded.

**Why:** Concrete case (2026-07-03, Dustin Martin doc "12-in-history" rarity scan — 200+ games,
20+ disp/g, 300+ goals). A DataSentinel-style scan using a dropna denominator returned **16**
players, adding Skilton, Ashman, Bisset, Barassi. Their TRUE full-career disp/g (fill-zero):
Barassi 4.45 (only 50 of 254 games have disposals recorded → dropna gives 22.6), Skilton 11.0
(98/237 recorded → 26.6), Bisset 19.8, Ashman 19.99. None actually clear 20+ disp/g honestly.
The fill-zero scan returns **12**, which is what the doc already said and is correct. Including
Barassi as a "20+ disposals per game" player is a transparently false, reader-catchable claim —
the exact thing we're trying to avoid.

**How to apply:**
- Always compute per-game rate = `stat.sum() / canonical_games` where canonical_games =
  `max(rowcount, numeric_leading_digit_max(games_played))`. See [[games_played_gap_detector]]
  and [[player_csv_date_format]] for the games_played string-dtype trap.
- When a "corrected" count disagrees with an existing doc AND the new count adds only very old
  (pre-1990) players, suspect a dropna denominator before trusting it. Re-run with fill-zero.
- Related era-coverage note: [[data_stat_coverage_eras]] (tackles from 1987, disposals sparse
  pre-1970s).

**EDITORIAL OVERRIDE (Decision 3, human, 2026-07-07) — the fill-zero-is-correct stance above
was NOT adopted.** The human resolved the era-boundary inclusion call to **INCLUDE** (dropna
over recorded games), so the sanctioned threshold count for the Martin rarity query is the
dropna count, NOT 12. The methodology insight above is still true (dropna inflates partial-
coverage old players — Barassi's 22.6 rests on 50 of 254 games; real career rate 4.45), but the
editorial call is to include them WITH a mandatory coverage annotation as the mitigation, not to
exclude. Do not re-argue 12; implement the decided rule.
- Deterministic implementation: `scripts/era_boundary_threshold.py` (tests
  `tests/unit/test_era_boundary_threshold.py`). It qualifies on the dropna rate and emits, per
  player, `coverage = recorded of games` plus a `partial_coverage` flag (<90% recorded), and the
  aggregate `N of M`. Run: `python scripts/era_boundary_threshold.py`.
- LIVE count as of 2026 R18 = **17 of 13350**, not the 16 the decision anticipated. The +1 is
  Toby Greene, who crossed 20.0 disp/g (full coverage) during 2026 — a live-data movement, NOT an
  era-boundary case. The 4 genuine dropna-vs-fillzero additions are Skilton, Bisset, Barassi,
  Ashman (3 flagged partial-coverage; Ashman is full-coverage at 20.1). This count is LIVE and
  moves every round — see [[live_stamped_doc_figures_need_asof_freeze]].
