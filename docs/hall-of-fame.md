# AFL Hall of Fame - the greatest of all time

> [← Back to main README](../README.md)

<!-- This file is part of the SuperCoach-VIA documentation. See README.md for the project overview. -->

The rankings and profiles in this section draw on the historical match records in this project (going back to 1897), career statistics where available, premiership records, and deep football knowledge built over more than a century of the game. Where actual numbers are computed from the data, they are marked **[data]**; where they come from historical record, they are marked **[historical record]**.

The game has produced extraordinary people. The pages below are about the ones who defined it.

## Pages in this section

- **[100 Forgotten Heroes →](hall-of-fame-forgotten-heroes.md)** - Underappreciated AFL players across 8 categories, data-verified from 13,328 player CSVs. The workhorses, ruck gods, defensive anchors, goal machines, one-club servants, pre-TV era legends, handball kings, and short but brilliant.
- **[Top 100 AFL players of all time](hall-of-fame-top100.md)** - auto-generated, era-normalised composite ranking. Updated whenever `update_team_analysis.py` runs.
- **[All-time statistical leaders](hall-of-fame-stat-leaders.md)** - top 20 in every major category, hub page with links to twelve dedicated sub-pages, plus single-season records. Pure raw-volume tables, the counterpart to the era-normalised top 100.
  - [Career goals](hall-of-fame-stat-goals.md) - Lockett 1,360 **[data]**
  - [Career games](hall-of-fame-stat-games.md) - Pendlebury 433 (outright) **[data]**
  - [Career disposals](hall-of-fame-stat-disposals.md) - Pendlebury 10,933 **[data]**
  - [Career marks](hall-of-fame-stat-marks.md) - Nick Riewoldt 2,944 **[data]**
  - [Career tackles](hall-of-fame-stat-tackles.md) - Pendlebury 1,997 **[data]**
  - [Career contested possessions](hall-of-fame-stat-contested.md) - Dangerfield 4,627 **[data]**
  - [Career clearances](hall-of-fame-stat-clearances.md) - Neale 1,947 **[data]**
  - [Career hit-outs](hall-of-fame-stat-hitouts.md) - Goldstein 10,597 **[data]**
  - [Career Brownlow votes](hall-of-fame-stat-brownlow.md) - Ablett jnr 262 **[data]**
  - [Career goal assists](hall-of-fame-stat-goalassists.md) - Pendlebury 325 **[data]**
  - [Career kicks & handballs](hall-of-fame-stat-kicks-handballs.md) - Bartlett 8,293 kicks; Pendlebury 5,477 handballs **[data]**
  - [Single-season records](hall-of-fame-stat-single-season.md) - Hudson/Pratt 150 goals; Cripps 45 Brownlow votes **[data]**
- **[Top 30 AFL captains of all time](hall-of-fame-captains.md)** - Barassi, Whitten, Skilton, Dyer through to Bontempelli and Cripps.
- **[Top 10 AFL coaches of all time](hall-of-fame-coaches.md)** - McHale, Barassi, Malthouse, Sheedy, Clarkson and the rest.
- **[Top 20 most courageous AFL players of all time](hall-of-fame-courageous.md)** - physical, mental, competitive courage from Sam Newman to Jason Dunstall.
- **[Top 10 Indigenous Australian players in AFL history](hall-of-fame-indigenous.md)** - the greatest Aboriginal and Torres Strait Islander footballers, from Goodes and Buddy to Winmar, Long and Rioli.
- **[Great AFL careers cut short](hall-of-fame-careers-cut-short.md)** - players stopped early by injury, illness, war, or circumstance, from Bruce Sloss to Tom Boyd.
- **[Great AFL dynasties](hall-of-fame-dynasties.md)** - dominant club eras, coaching trees, and the cross-club lineages that shaped the modern game, from Collingwood's Machine to the Clarkson tree.

Each page has its own back-link to this hub and to the main README.

---

## How to read this section

These pages are written in two layers, and they intentionally do not collapse into one.

**The data layer.** Every specific number on every page - games played, goals kicked, Brownlow votes polled, premierships won, career span - is verified against the player performance files in `data/player_data/` (one CSV per player, every game they played) and the match files in `data/matches/` (one CSV per season, every match contested). Where the data confirms a number, it is tagged `**[data]**`. Where the claim comes from documented football history but the underlying number is not in the repo (pre-1965 stat categories, club selection records, Norm Smith Medals before 1979 for some matches), it is tagged `**[historical record]**`. Where a specific stat is referenced but cannot be verified from data, it is tagged `**[historical record - unverified in data]**` - which is also a finding, and the right way to flag it.

**The analytical layer.** Each page closes with a *FootyStrategy analytical read* section. Where the data layer answers *what happened*, this layer answers *why it mattered and how it reshaped the game.* These sections are deliberately opinionated. They name the eras, the structural changes, and the tactical innovations that the raw numbers alone do not surface. A career-goals leaderboard, on its own, does not tell you that the post-2000 game made 1,000-goal forwards almost extinct; the analytical read does.

The pages are not ranked by importance. The top-100 is era-normalised. The stat leaders are by raw volume. The captains are by leadership impact. The Indigenous players page is structured by what each player changed about the game's relationship to the country it is played in. There is no single hierarchy. The pages are designed to be read against each other, not summed into a single list.

For methodology and reproducibility, see the comments in `docs/hall-of-fame/compute_stat_leaders.py` and the data-coverage notes at the bottom of each individual page.
