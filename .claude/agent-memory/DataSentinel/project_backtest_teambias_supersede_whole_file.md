---
name: backtest-teambias-supersede-whole-file
description: Reconciling docs/afl-backtest-2026.md TEAMBIAS block requires supersede-by-whole-file-per-round on backtest_by_team_*.csv, NOT drop_duplicates(['year','round','team']) — the latter silently backfills missing teams from a stale vintage and breaks the three-way reconciliation.
metadata:
  type: project
---

## 2026-07-28 (R21 Pass-2, hash `28ee21d1…`) — confirmed trap while reproducing TEAMBIAS/n=7,524

Reproducing the team-level bias table from `data/prediction/backtest/backtest_by_team_*.csv`
the naive way — concat all files, sort by filename, `drop_duplicates(['year','round','team'],
keep='last')` — gives **342 rows summing to n=7,616**, not the doc's 7,524. The discrepancy
(92 = 4 clubs × 23) lands exactly on Geelong, Melbourne, St Kilda, Western Bulldogs — the four
clubs the doc's own "Known coverage limitation" section documents as **absent** from Round 18's
newer (284-player, 14-club) vintage. Because those four clubs have NO row at all in the newer
Round 18 team-file, a per-`(year,round,team)` dedup can't override their OLDER (412-player,
18-club) Round 18 rows — it just keeps them, silently blending two vintages of the same round.

**Fix**: use the same round→timestamp map built from the keep-last `backtest_summary_*.csv`
dedup (this doc's canonical vintage-selection method, see
[[project_backtest_doc_verification_gotchas]]), then for each round load ONLY
`backtest_by_team_<that round's winning ts>.csv` filtered to that round — never dedup the
team file independently by `(year,round,team)`. This reproduces exactly 338 rows, n=7,524,
and every team's `n`/bias to the displayed decimal. This is precisely the "supersede by whole
file per (year,round)" rule the doc's own Reproducibility section states in prose — confirmed
here as load-bearing, not boilerplate: a per-team dedup breaks the three-way reconciliation
invariant (pooled per-player == summed summary n_players == summed per-team n) the doc claims
holds.

**How to apply next time**: always build the round→ts map ONCE (from summary keep-last) and
reuse it verbatim for pulling per-player AND per-team artifacts. Never independently
`drop_duplicates` the team or player files on their own keys — that reintroduces exactly the
stale-team-row blending this note describes.

See also [[project_backtest_doc_verification_gotchas]],
[[project_backtest_partial_regen_stale_blocks]].
