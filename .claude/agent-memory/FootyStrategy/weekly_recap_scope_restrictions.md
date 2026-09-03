---
name: weekly-recap-scope-restrictions
description: Some weekly-cycle recap requests explicitly restrict sources and forbid forward-looking content — the recap template's default four items (leaders, ladder movement, player to watch, tactical note) is not mandatory when the invocation narrows scope
metadata:
  type: feedback
---

**Pattern seen (Round 25, 2026-08-29):** the invocation explicitly limited sources to
`docs/afl-stat-leaders-2026.md` and `docs/afl-season-2026.md`, forbade citing
`docs/afl-predictions-2026.md` / `docs/weekly/round-current-2026.md`, and forbade any
forward-looking or "next round" content because Round 25 was already fully played and
there was no fresh cheat sheet generated for it yet. This contradicts the general
skill-prompt default (which lists "one player to watch next round" as a standard
recap item) — the specific per-invocation instruction wins.

**How to apply:** when a weekly-recap invocation narrows sources or explicitly says
"no forward prediction this cycle," drop the "watch next round" item entirely rather
than forcing a substitute forward-looking claim from the allowed sources. Rebuild the
section from only what the allowed sources actually contain — in this case
`afl-season-2026.md` turned out to be a pure index page with no data of its own, so
the whole recap was built from `afl-stat-leaders-2026.md` alone (disposal leaders,
team score/margin table, one correlation-based tactical note). Don't manufacture a
"ladder movement" claim if no ladder/finals doc is in scope — team score/margin
tables are a legitimate substitute for "team form" when that's all that's available.

**Confirms:** [[skeptic_margin_label_mismatch]] fix pattern (relabel the "Winning
margin" section heading as "average margin across all games, wins and losses
combined" rather than copying the source table's own mislabeled heading) applied
cleanly pre-emptively this cycle — no BLOCK needed. Also confirms the games-played
disclosure practice: always grep-count `,<year>,` rows in each disposal leader's own
`_performance_details.csv` before ranking by per-game average, since the leaderboard
table itself never states games played and sample sizes can differ materially
(11 games vs 22-23 games in this case) even among top-5 leaderboard entries.
