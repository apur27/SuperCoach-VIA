# 2026 player performance stats - what to look for and what the data says

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-STAT-LEADERS-START -->
This section is a guide to the AFL performance statistics that fans, analysts and SuperCoach players track most closely — what each stat measures, who is leading it in 2026, what the league-wide distribution looks like, and which other stats most reliably predict it. All numbers are computed live from `data/player_data/` for 2026 (rounds 1-25, **616 eligible players** with >=3 games, **9425 player-games** included). Correlations are Pearson r on the per-game frame; with several thousand player-games, p-values are universally tiny — read the magnitude of r, not the significance star.

![2026 AFL statistical leaders](../assets/charts/player_stat_leaders_2026.png)

### Disposal-based stats — volume and quality of ball use

#### Disposals per game

**What it measures.** Total kicks plus handballs in a game — the single broadest measure of how often a player has the ball. **Why it matters.** It is the headline SuperCoach scoring stat and the prediction target this repo's main model is built around. Volume midfielders and rebounding defenders dominate this leaderboard.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 34.9 |
| 2 | Bailey Smith | Geelong | 32.6 |
| 3 | Errol Gulden | Sydney | 31.5 |
| 4 | Harry Sheezel | North Melbourne | 30.5 |
| 5 | Clayton Oliver | Greater Western Sydney | 30.2 |

League distribution (eligible players, season-to-date): mean **14.91**, std 5.64, p10 8.30 / p50 14.03 / p90 23.21, max 34.86.

Top per-game correlates: `effective_disposals` (r = +0.97 *(mechanically related)*), `uncontested_possessions` (r = +0.87), `kicks` (r = +0.83).

#### Kicks per game

**What it measures.** Just the kicked disposals. **Why it matters.** Kicks tend to come from outside-midfielders, half-backs and tall rebounders — players who clear the ball by foot rather than shovel it into a contest. A player who kicks much more than they handball is usually playing a distributor / launch role.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nasiah Wanganeen-Milera | St Kilda | 21.5 |
| 2 | Bailey Dale | Western Bulldogs | 20.3 |
| 3 | Bailey Smith | Geelong | 20.0 |
| 4 | Dayne Zorko | Brisbane Lions | 19.5 |
| 5 | Errol Gulden | Sydney | 19.5 |

League distribution (eligible players, season-to-date): mean **8.58**, std 3.50, p10 4.50 / p50 8.06 / p90 13.18, max 21.47.

Top per-game correlates: `disposals` (r = +0.83), `effective_disposals` (r = +0.81), `uncontested_possessions` (r = +0.77).

#### Handballs per game

**What it measures.** The hand-passed half of disposals. **Why it matters.** Handball volume tracks contest involvement — a player wins the ball at a stoppage, then handballs out to a runner. Inside-mids and clearance specialists tend to lead this stat.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Clayton Oliver | Greater Western Sydney | 19.1 |
| 2 | Lachie Neale | Brisbane Lions | 16.3 |
| 3 | Patrick Cripps | Carlton | 16.3 |
| 4 | Nick Daicos | Collingwood | 16.0 |
| 5 | Sam Walsh | Carlton | 16.0 |

League distribution (eligible players, season-to-date): mean **6.33**, std 3.05, p10 3.02 / p50 5.75 / p90 10.51, max 19.09.

Top per-game correlates: `disposals` (r = +0.78), `effective_disposals` (r = +0.75), `contested_possessions` (r = +0.65).

#### Effective disposals per game (disposals − clangers)

**What it measures.** Disposals that did not result in a clanger, computed here as `max(disposals - clangers, 0)` because the raw data does not carry a true effective-disposal column. **Why it matters.** It is a defensible proxy for disposal *quality* — high-volume ball-users who don't turn it over. The same proxy is used in the Brownlow predictor on this page.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 29.9 |
| 2 | Bailey Smith | Geelong | 28.5 |
| 3 | Lachie Neale | Brisbane Lions | 27.8 |
| 4 | Harry Sheezel | North Melbourne | 27.3 |
| 5 | Errol Gulden | Sydney | 26.7 |

League distribution (eligible players, season-to-date): mean **12.58**, std 5.24, p10 6.35 / p50 11.83 / p90 20.12, max 29.91.

Top per-game correlates: `disposals` (r = +0.97 *(mechanically related)*), `uncontested_possessions` (r = +0.86), `kicks` (r = +0.81).

### Scoring stats — goals, behinds and conversion

#### Goals per game

**What it measures.** Goals kicked. **Why it matters.** Forwards live and die by this stat. It is volatile game-to-game (a single missed shot can halve your score), so multi-game averages and shot-source context (marks-inside-50, contested marks) matter more than any one game.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Gunston | Hawthorn | 3.47 |
| 2 | Charlie Curnow | Sydney | 3.14 |
| 3 | Logan Morris | Brisbane Lions | 2.86 |
| 4 | Jye Amiss | Fremantle | 2.64 |
| 5 | Nick Watson | Hawthorn | 2.62 |

League distribution (eligible players, season-to-date): mean **0.52**, std 0.59, p10 0.00 / p50 0.32 / p90 1.33, max 3.47.

Top per-game correlates: `marks_inside_50` (r = +0.67), `behinds` (r = +0.32), `rebound_50s` (r = -0.30).

**Goal conversion rate.** Defined as `goals / (goals + behinds)`, season-to-date, for players with >=2 goals total. League distribution (n=431): mean **58.5%**, std 16.0pp, p10 39% / p50 57% / p90 77%.

| Rank | Player | Team | G | B | Conversion |
|---|---|---|---|---|---|
| 1 | Isaac Cumming | Adelaide | 8 | 0 | 100.0% |
| 2 | Dante Visentini | Port Adelaide | 6 | 0 | 100.0% |
| 3 | Campbell Lake | St Kilda | 5 | 0 | 100.0% |
| 4 | Jack Graham | West Coast | 4 | 0 | 100.0% |
| 5 | Noah Roberts-Thomson | Richmond | 4 | 0 | 100.0% |

#### Behinds per game

**What it measures.** Minor scores — shots that hit the post or go through the smaller posts. **Why it matters.** Rarely predicted alone — it is too noisy. Best read alongside goals to compute **conversion rate** (`goals / (goals + behinds)`), the cleanest available signal of forward accuracy.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jake Waterman | West Coast | 2.43 |
| 2 | Mitch Georgiades | Port Adelaide | 2.24 |
| 3 | Jack Gunston | Hawthorn | 2.06 |
| 4 | Jake Stringer | Greater Western Sydney | 1.96 |
| 5 | Logan Morris | Brisbane Lions | 1.91 |

League distribution (eligible players, season-to-date): mean **0.39**, std 0.39, p10 0.00 / p50 0.29 / p90 0.91, max 2.43.

Top per-game correlates: `marks_inside_50` (r = +0.55), `goals` (r = +0.32), `rebound_50s` (r = -0.24).

### Contested and ground-ball stats — the inside game

#### Contested possessions per game

**What it measures.** Wins of the ball under physical pressure — ground-balls, taps, and contested marks. **Why it matters.** This is the cleanest stat for separating a midfielder's *contest* role from an outside ball-user's *spread* role. It correlates strongly with clearances and tackles.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Clayton Oliver | Greater Western Sydney | 14.91 |
| 2 | Patrick Cripps | Carlton | 14.57 |
| 3 | Tristan Xerri | North Melbourne | 14.26 |
| 4 | Matt Rowell | Gold Coast | 14.17 |
| 5 | Lachie Neale | Brisbane Lions | 13.48 |

League distribution (eligible players, season-to-date): mean **5.24**, std 2.34, p10 3.00 / p50 4.67 / p90 8.72, max 14.91.

Top per-game correlates: `clearances` (r = +0.75), `handballs` (r = +0.65), `disposals` (r = +0.59).

#### Clearances per game

**What it measures.** Disposals that move the ball clear of a stoppage (a centre-bounce or boundary throw-in). **Why it matters.** Stoppage dominance is one of the few team-level wins a midfield can manufacture. Top clearance players are almost always the inside-mid fulcrums of their team.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jai Newcombe | Hawthorn | 7.57 |
| 2 | Patrick Cripps | Carlton | 7.57 |
| 3 | Matt Rowell | Gold Coast | 7.56 |
| 4 | Clayton Oliver | Greater Western Sydney | 7.30 |
| 5 | Lachie Neale | Brisbane Lions | 7.17 |

League distribution (eligible players, season-to-date): mean **1.40**, std 1.62, p10 0.11 / p50 0.77 / p90 4.05, max 7.57.

Top per-game correlates: `contested_possessions` (r = +0.75), `handballs` (r = +0.56), `disposals` (r = +0.50).

#### Tackles per game

**What it measures.** Pressure acts that physically stop a ball-carrier. **Why it matters.** Defensive midfield work — the unsung currency of forward-half pressure and turnover football. It correlates with clearances (you tackle the same opponent you compete against) but tells a different story.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Sam Berry | Adelaide | 7.27 |
| 2 | Matt Rowell | Gold Coast | 6.94 |
| 3 | Josh Dunkley | Brisbane Lions | 6.70 |
| 4 | James Rowbottom | Sydney | 6.48 |
| 5 | Ned Long | Collingwood | 6.29 |

League distribution (eligible players, season-to-date): mean **2.38**, std 1.22, p10 1.10 / p50 2.11 / p90 4.09, max 7.27.

Top per-game correlates: `clearances` (r = +0.39), `contested_possessions` (r = +0.37), `handballs` (r = +0.31).

#### Hit-outs per game (ruckmen only)

**What it measures.** Wins by a ruckman at a ruck contest (the tap from a centre bounce or stoppage). **Why it matters.** Ruckman-only stat — the distribution is bimodal: ~1 player per team registers double-digits, everyone else is 0. Always read this leaderboard as "top ruckmen", not "top players".

**Bimodal distribution warning.** 88% of eligible 2026 players average less than 1 hit-out per game — they are not ruckmen. The league mean below is dragged down by all the zeros; the meaningful comparison is between ruckmen, where the top of the distribution sits in the 25-35 range.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Ned Moyle | Gold Coast | 34.9 |
| 2 | Brodie Grundy | Sydney | 34.3 |
| 3 | Max Gawn | Melbourne | 33.0 |
| 4 | Lachlan Mcandrew | Adelaide | 32.2 |
| 5 | Jordon Sweet | Port Adelaide | 30.2 |

League distribution (eligible players, season-to-date): mean **1.56**, std 5.38, p10 0.00 / p50 0.00 / p90 1.94, max 34.88.

Top per-game correlates: `clearances` (r = +0.27), `uncontested_possessions` (r = -0.24), `contested_possessions` (r = +0.20).

### Territory stats — moving the ball forward

#### Inside 50s per game

**What it measures.** Disposals or carries that move the ball into the team's attacking 50m arc. **Why it matters.** Territory currency — the precondition for goals. Wing/half-forward players who launch attacks lead this stat. It correlates with kicks and disposals because most inside-50s are foot-delivered.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Bailey Smith | Geelong | 7.27 |
| 2 | Errol Gulden | Sydney | 7.00 |
| 3 | Ed Richards | Western Bulldogs | 6.77 |
| 4 | Chad Warner | Sydney | 6.77 |
| 5 | Nick Daicos | Collingwood | 6.73 |

League distribution (eligible players, season-to-date): mean **2.15**, std 1.20, p10 0.71 / p50 2.00 / p90 3.68, max 7.27.

Top per-game correlates: `disposals` (r = +0.52), `effective_disposals` (r = +0.49), `kicks` (r = +0.48).

#### Marks per game

**What it measures.** Total uncontested + contested marks taken. **Why it matters.** Aerial dominance and intercept defence. Loose-half-back roles dominate the total-marks leaderboard because they sit behind the play and fly under kicks. Tall forwards lead a separate, narrower stat — marks inside 50.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Callum Wilkie | St Kilda | 9.6 |
| 2 | Aliir Aliir | Port Adelaide | 7.9 |
| 3 | Nick Haynes | Carlton | 7.7 |
| 4 | Harris Andrews | Brisbane Lions | 7.5 |
| 5 | Jacob Weitering | Carlton | 7.2 |

League distribution (eligible players, season-to-date): mean **3.78**, std 1.47, p10 1.95 / p50 3.69 / p90 5.68, max 9.61.

Top per-game correlates: `kicks` (r = +0.56), `uncontested_possessions` (r = +0.53), `effective_disposals` (r = +0.42).

#### Marks inside 50 per game

**What it measures.** Marks taken inside the attacking 50m arc — i.e. marks that turn directly into shots on goal. **Why it matters.** This is the strongest single predictor of a forward's goal output. It is what separates a deep-forward role from a high-half-forward role, and the correlation with goals is the highest of any stat in this section.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Gunston | Hawthorn | 4.12 |
| 2 | Mitch Georgiades | Port Adelaide | 3.95 |
| 3 | Jye Amiss | Fremantle | 3.41 |
| 4 | Josh Treacy | Fremantle | 3.35 |
| 5 | Logan Morris | Brisbane Lions | 3.27 |

League distribution (eligible players, season-to-date): mean **0.51**, std 0.66, p10 0.00 / p50 0.26 / p90 1.55, max 4.12.

Top per-game correlates: `goals` (r = +0.67), `behinds` (r = +0.55), `contested_marks` (r = +0.35).

### Discipline stats — errors and free kicks

#### Clangers per game

**What it measures.** Errors — missed targets, fumbles, free kicks given away by the ball-carrier. **Why it matters.** Clangers are the friction term on disposal volume — a high-disposal player who also leads in clangers is being asked to play through traffic, not necessarily playing badly. The correlation with frees-against is mechanical: many clangers *are* frees-against.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Harley Reid | West Coast | 6.74 |
| 2 | Jacob Hopper | Richmond | 5.50 |
| 3 | Brodie Grundy | Sydney | 5.24 |
| 4 | Kysaiah Pickett | Melbourne | 5.04 |
| 5 | Patrick Cripps | Carlton | 5.00 |

League distribution (eligible players, season-to-date): mean **2.34**, std 0.81, p10 1.41 / p50 2.20 / p90 3.38, max 6.74.

Top per-game correlates: `free_kicks_against` (r = +0.61 *(mechanically related)*), `contested_possessions` (r = +0.34), `disposals` (r = +0.32).

#### Free kicks for per game

**What it measures.** Free kicks paid to the player. **Why it matters.** A weak isolated signal — frees-for tracks contest involvement (rucks especially) more than skill. Best used as a tiebreaker rather than a standalone metric.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Tristan Xerri | North Melbourne | 3.00 |
| 2 | Harley Reid | West Coast | 2.52 |
| 3 | Matt Rowell | Gold Coast | 2.39 |
| 4 | Max Gawn | Melbourne | 2.35 |
| 5 | Sam Darcy | Western Bulldogs | 2.33 |

League distribution (eligible players, season-to-date): mean **0.78**, std 0.42, p10 0.33 / p50 0.71 / p90 1.31, max 3.00.

Top per-game correlates: `contested_possessions` (r = +0.42), `clearances` (r = +0.30), `tackles` (r = +0.21).

#### Free kicks against per game

**What it measures.** Free kicks paid against the player. **Why it matters.** Discipline / aggression marker, with the caveat that ruck contest infringements inflate the number for ruckmen. Reads like a clanger when it correlates with them.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Harley Reid | West Coast | 3.00 |
| 2 | Brodie Grundy | Sydney | 2.57 |
| 3 | Patrick Cripps | Carlton | 2.04 |
| 4 | Lachlan Blakiston | Essendon | 2.00 |
| 5 | Matt Flynn | West Coast | 2.00 |

League distribution (eligible players, season-to-date): mean **0.80**, std 0.41, p10 0.33 / p50 0.74 / p90 1.33, max 3.00.

Top per-game correlates: `clangers` (r = +0.61 *(mechanically related)*), `clearances` (r = +0.16), `contested_possessions` (r = +0.16).

### Team-level stats — what the scoreboard says

Team-level stats use `data/matches/matches_2026.csv` rather than per-player aggregates. Total team score is `goals × 6 + behinds`; margin is the team's score minus the opponent's. A first-quarter score is a useful early-momentum signal — strong starters tend to keep the lead.

#### Total team score per game

| Rank | Team | Avg score | Avg margin | Avg Q1 |
|---|---|---|---|---|
| 1 | Sydney | 110.6 | +29.3 | 28.5 |
| 2 | Brisbane Lions | 108.7 | +19.3 | 24.6 |
| 3 | Geelong | 102.4 | +18.7 | 25.3 |
| 4 | Melbourne | 100.0 | +8.8 | 25.3 |
| 5 | Fremantle | 99.4 | +27.0 | 24.7 |

League distribution of per-game team scores: mean **89.1**, std 25.3, p10 60 / p50 88 / p90 122, min 29 / max 170.

#### Winning margin

| Rank | Team | Avg margin | Avg score |
|---|---|---|---|
| 1 | Sydney | +29.3 | 110.6 |
| 2 | Fremantle | +27.0 | 99.4 |
| 3 | Brisbane Lions | +19.3 | 108.7 |
| 4 | Geelong | +18.7 | 102.4 |
| 5 | Hawthorn | +16.5 | 98.3 |

League distribution of margins (signed, per team-game): mean ~0 by construction, std 41.4, p10 -54 / p50 0 / p90 54.

#### First-quarter score

| Rank | Team | Avg Q1 score | Avg full-game score |
|---|---|---|---|
| 1 | Sydney | 28.5 | 110.6 |
| 2 | Adelaide | 26.3 | 94.3 |
| 3 | Melbourne | 25.3 | 100.0 |
| 4 | Geelong | 25.3 | 102.4 |
| 5 | Fremantle | 24.7 | 99.4 |

League distribution of Q1 scores: mean **22.4**, std 11.0, p10 9 / p50 21 / p90 38.

### Going deeper with this repo's models

For the stats above, three artefacts in this repo will help you form your own view rather than just reading a leaderboard:

1. The **disposal prediction model** (`prediction.py` / `prediction_cpu.py`) forecasts a player's next-round disposal count using rolling form (3/5-game, season-to-date) and opponent context. Run it with `--player surname_first --rounds 1` to see how uncertainty is quantified for any of the leaders shown above.
2. The **backtest framework** (`backtest.py`) replays a season round-by-round so you can see how the model performed on real, out-of-sample games — the honest way to judge whether a leaderboard ranking will continue to hold.
3. The **Brownlow proxy section** above is the same per-game stat structure used here, weighted into a single composite. If you want a quick "who's having the best year overall" answer rather than per-stat leaders, that table is the one to look at.
<!-- 2026-STAT-LEADERS-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Brownlow predictor](afl-brownlow-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
