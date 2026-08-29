---
name: afl-insights-week-in-review-pattern
description: docs/afl-insights.md "## Round N — Week in Review" section verification pattern — cites docs/afl-stat-leaders-2026.md and docs/weekly/round-current-2026.md as intermediate (doc, not CSV) sources; games-played counts behind ties need direct per-player CSV cross-checks.
metadata:
  type: project
---

`docs/afl-insights.md` carries a rolling "Week in Review" section (round number changes weekly)
with `**[data]**` tags whose methodology paragraph names TWO kinds of source:

1. **Intermediate generated docs** — `docs/afl-stat-leaders-2026.md` (disposal-leader table,
   league mean/threshold, correlation r) and `docs/afl-predictions-2026.md` /
   `docs/weekly/round-current-2026.md` (next-round projections, cheat-sheet typical error). These
   are themselves auto-generated from `data/player_data/` + `data/matches/matches_<year>.csv` /
   `data/prediction/next_round_<N>_prediction_<ts>.csv`. Verifying against the named intermediate
   doc via grep/Read is sufficient — re-deriving stat-leaders' own means/percentiles from raw CSVs
   is redundant with that doc's own prior verification pass. See [[reference_stat_leaders_distribution_basis]]
   for how to verify afl-stat-leaders-2026.md itself if ever asked directly.
2. **Direct player CSVs** — games-played counts behind a leaderboard tie (e.g. "Gulden's 10
   games vs 22 each for Oliver/Sheezel") are sourced straight from the three players'
   `*_performance_details.csv` files, not from the stat-leaders doc. `round` is a STRING column
   with non-numeric finals values in some files (`'EF'`,`'QF'`,`'PF'`,`'GF'`) — filter to
   `year==2026` first, THEN coerce `round` numeric with `errors='coerce'` if you need a round-N
   cutoff; comparing the raw string round column directly against an int raises a TypeError.

R25 cycle (2026-08-18, hash `8807f7e5…`): all 18 `**[data]**` tag instances (10 distinct claims:
35.3, 32.3, 30.5, 10, 22, 14.91, 611, 3, +29.9, +28.2, 110.0, 29.0, 28.0, 7.3, ±4, +0.75) verified
clean — games counts matched exactly (Gulden 10, Oliver 22, Sheezel 22), club attributions matched
each player's `team` column for 2026, zero untagged numbers, zero coach-name hits. PASS.

R25 line-26 re-verify (same day, hash `f3f200bd…`, after a FootyStrategy edit closing two Skeptic
concerns): "Watch in Round 25" tie widened from Daicos-only to a three-way Daicos/Neale/Oliver tie
at 28.0 — all three (plus Sheezel/Gulden at 29.0) cross-checked directly against
`data/prediction/next_round_25_prediction_20260818_1146.csv` (Surname-first `player` col) as well
as `docs/afl-predictions-2026.md` / `docs/weekly/round-current-2026.md`; exact match, no club
attribution claimed for Neale/Oliver in-sentence so nothing to check there. The caveat clause
reword ("regress toward the league mean, running below season averages for the highest-volume
players and above them for low-volume ones") added zero new numbers — confirmed via full-doc
digit scan (`grep -noE '[0-9]+(\.[0-9]+)?'`), every digit on line 26 traces to an already-tagged
value or the Round-25 structural reference. PASS.
