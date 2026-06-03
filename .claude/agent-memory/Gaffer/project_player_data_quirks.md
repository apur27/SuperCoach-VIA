---
name: player-data-quirks
description: Verification gotchas for data/player_data — games-counter resets per season, goal_assist is singular, Ablett snr/jnr collide on surname glob
metadata:
  type: project
---

Verification quirks found while auditing docs against `data/player_data/` on 2026-05-31.

**Career games is NOT row count AND not a naive max.** Some files store a
`games_played` counter that RESETS each season (max ~22-25 within a year), so
`df['games_played'].max()` returns nonsense like 99 for Pendlebury/Brent Harvey.
Other files (e.g. Tuck) store it cumulatively. The HOF pages compute games via
`docs/hall-of-fame/compute_stat_leaders.py` (canonical). For a quick check, row
count is usually within a game or two of the published total but can differ by a
few (drawn GFs / collapsed finals rows). Pendlebury: rowcount 430, published 433.
Treat small games deltas as UNVERIFIABLE-by-rowcount, not WRONG — defer to the
compute script.

**Column name is `goal_assist` (singular), not `goal_assists`.** Pendlebury career
goal_assist sum = 325, which matches the hub.

**Ablett snr vs jnr collide on `ablett_gary_*` glob.** snr = `ablett_gary_01101961`
(1982-1996, 1031 goals, 99 brownlow). jnr = `ablett_gary_14051984` (2002-2020,
262 brownlow). Always disambiguate by birthdate/year-span when verifying an Ablett.

**Forgotten-heroes tables use inline `[data]:` PROSE, not pipe columns.** e.g.
"[data]: 292 games, 86 career goals, 25.0 disposal average **[data]**". A
column-position regex will match zero rows. Parse the prose, or spot-check named
players. Spot-checks done 2026-05-31 (Kirk, Owen, Sewell, Murphy, Montagna, Boyd)
all verified clean; single-season records (Hudson 150@1971, Pratt 150@1934,
Macrae 880@2021, Cripps 45 brownlow@2024) all verified clean.

See [[feedback_flaky_output_channel]].
