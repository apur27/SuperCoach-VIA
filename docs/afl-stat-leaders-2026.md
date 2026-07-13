# 2026 player performance stats - what to look for and what the data says

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-STAT-LEADERS-START -->
This section is a guide to the AFL performance statistics that fans, analysts and SuperCoach players track most closely — what each stat measures, who is leading it in 2026, what the league-wide distribution looks like, and which other stats most reliably predict it. All numbers are computed live from `data/player_data/` for 2026 (rounds 1-19, **585 eligible players** with >=3 games, **6933 player-games** included). Correlations are Pearson r on the per-game frame; with several thousand player-games, p-values are universally tiny — read the magnitude of r, not the significance star.

![2026 AFL statistical leaders](../assets/charts/player_stat_leaders_2026.png)

### Disposal-based stats — volume and quality of ball use

#### Disposals per game

**What it measures.** Total kicks plus handballs in a game — the single broadest measure of how often a player has the ball. **Why it matters.** It is the headline SuperCoach scoring stat and the prediction target this repo's main model is built around. Volume midfielders and rebounding defenders dominate this leaderboard.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 34.8 |
| 2 | Bailey Smith | Geelong | 32.2 |
| 3 | Clayton Oliver | Greater Western Sydney | 31.9 |
| 4 | Harry Sheezel | North Melbourne | 31.6 |
| 5 | Nasiah Wanganeen-Milera | St Kilda | 30.8 |

League distribution (eligible players, season-to-date): mean **15.08**, std 5.72, p10 8.35 / p50 14.35 / p90 23.53, max 34.75.

Top per-game correlates: `effective_disposals` (r = +0.97 *(mechanically related)*), `uncontested_possessions` (r = +0.87), `kicks` (r = +0.83).

#### Kicks per game

**What it measures.** Just the kicked disposals. **Why it matters.** Kicks tend to come from outside-midfielders, half-backs and tall rebounders — players who clear the ball by foot rather than shovel it into a contest. A player who kicks much more than they handball is usually playing a distributor / launch role.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nasiah Wanganeen-Milera | St Kilda | 22.3 |
| 2 | Archie Roberts | Essendon | 20.8 |
| 3 | Bailey Dale | Western Bulldogs | 20.2 |
| 4 | Bailey Smith | Geelong | 19.6 |
| 5 | Dayne Zorko | Brisbane Lions | 19.5 |

League distribution (eligible players, season-to-date): mean **8.73**, std 3.59, p10 4.50 / p50 8.20 / p90 13.58, max 22.31.

Top per-game correlates: `disposals` (r = +0.83), `effective_disposals` (r = +0.81), `uncontested_possessions` (r = +0.78).

#### Handballs per game

**What it measures.** The hand-passed half of disposals. **Why it matters.** Handball volume tracks contest involvement — a player wins the ball at a stoppage, then handballs out to a runner. Inside-mids and clearance specialists tend to lead this stat.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Clayton Oliver | Greater Western Sydney | 20.2 |
| 2 | Sam Walsh | Carlton | 16.5 |
| 3 | Patrick Cripps | Carlton | 16.4 |
| 4 | Lachie Neale | Brisbane Lions | 16.2 |
| 5 | Harry Sheezel | North Melbourne | 16.2 |

League distribution (eligible players, season-to-date): mean **6.35**, std 3.09, p10 3.00 / p50 5.75 / p90 10.68, max 20.18.

Top per-game correlates: `disposals` (r = +0.78), `effective_disposals` (r = +0.75), `contested_possessions` (r = +0.65).

#### Effective disposals per game (disposals − clangers)

**What it measures.** Disposals that did not result in a clanger, computed here as `max(disposals - clangers, 0)` because the raw data does not carry a true effective-disposal column. **Why it matters.** It is a defensible proxy for disposal *quality* — high-volume ball-users who don't turn it over. The same proxy is used in the Brownlow predictor on this page.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 29.8 |
| 2 | Lachie Neale | Brisbane Lions | 28.2 |
| 3 | Harry Sheezel | North Melbourne | 28.2 |
| 4 | Bailey Smith | Geelong | 28.0 |
| 5 | Clayton Oliver | Greater Western Sydney | 27.5 |

League distribution (eligible players, season-to-date): mean **12.69**, std 5.33, p10 6.33 / p50 12.00 / p90 20.24, max 29.75.

Top per-game correlates: `disposals` (r = +0.97 *(mechanically related)*), `uncontested_possessions` (r = +0.86), `kicks` (r = +0.81).

### Scoring stats — goals, behinds and conversion

#### Goals per game

**What it measures.** Goals kicked. **Why it matters.** Forwards live and die by this stat. It is volatile game-to-game (a single missed shot can halve your score), so multi-game averages and shot-source context (marks-inside-50, contested marks) matter more than any one game.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Gunston | Hawthorn | 3.55 |
| 2 | Charlie Curnow | Sydney | 3.12 |
| 3 | Ben King | Gold Coast | 2.76 |
| 4 | Logan Morris | Brisbane Lions | 2.69 |
| 5 | Nick Watson | Hawthorn | 2.60 |

League distribution (eligible players, season-to-date): mean **0.52**, std 0.59, p10 0.00 / p50 0.29 / p90 1.40, max 3.55.

Top per-game correlates: `marks_inside_50` (r = +0.67), `behinds` (r = +0.32), `rebound_50s` (r = -0.30).

**Goal conversion rate.** Defined as `goals / (goals + behinds)`, season-to-date, for players with >=2 goals total. League distribution (n=383): mean **59.3%**, std 16.8pp, p10 40% / p50 59% / p90 80%.

| Rank | Player | Team | G | B | Conversion |
|---|---|---|---|---|---|
| 1 | Isaac Cumming | Adelaide | 5 | 0 | 100.0% |
| 2 | Dan Butler | St Kilda | 4 | 0 | 100.0% |
| 3 | Noah Roberts-Thomson | Richmond | 4 | 0 | 100.0% |
| 4 | Oscar Steene | Collingwood | 4 | 0 | 100.0% |
| 5 | Nik Cox | Essendon | 3 | 0 | 100.0% |

#### Behinds per game

**What it measures.** Minor scores — shots that hit the post or go through the smaller posts. **Why it matters.** Rarely predicted alone — it is too noisy. Best read alongside goals to compute **conversion rate** (`goals / (goals + behinds)`), the cleanest available signal of forward accuracy.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jake Waterman | West Coast | 2.53 |
| 2 | Mitch Georgiades | Port Adelaide | 2.35 |
| 3 | Logan Morris | Brisbane Lions | 2.00 |
| 4 | Jack Gunston | Hawthorn | 2.00 |
| 5 | Nate Caddy | Essendon | 1.94 |

League distribution (eligible players, season-to-date): mean **0.40**, std 0.41, p10 0.00 / p50 0.29 / p90 0.94, max 2.53.

Top per-game correlates: `marks_inside_50` (r = +0.54), `goals` (r = +0.32), `rebound_50s` (r = -0.24).

### Contested and ground-ball stats — the inside game

#### Contested possessions per game

**What it measures.** Wins of the ball under physical pressure — ground-balls, taps, and contested marks. **Why it matters.** This is the cleanest stat for separating a midfielder's *contest* role from an outside ball-user's *spread* role. It correlates strongly with clearances and tackles.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Clayton Oliver | Greater Western Sydney | 15.53 |
| 2 | Patrick Cripps | Carlton | 14.76 |
| 3 | Tristan Xerri | North Melbourne | 14.69 |
| 4 | Max Gawn | Melbourne | 13.47 |
| 5 | Brodie Grundy | Sydney | 13.19 |

League distribution (eligible players, season-to-date): mean **5.29**, std 2.34, p10 3.00 / p50 4.67 / p90 8.78, max 15.53.

Top per-game correlates: `clearances` (r = +0.74), `handballs` (r = +0.65), `disposals` (r = +0.58).

#### Clearances per game

**What it measures.** Disposals that move the ball clear of a stoppage (a centre-bounce or boundary throw-in). **Why it matters.** Stoppage dominance is one of the few team-level wins a midfield can manufacture. Top clearance players are almost always the inside-mid fulcrums of their team.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Clayton Oliver | Greater Western Sydney | 7.94 |
| 2 | Jai Newcombe | Hawthorn | 7.82 |
| 3 | Lachie Neale | Brisbane Lions | 7.18 |
| 4 | Patrick Cripps | Carlton | 7.00 |
| 5 | Marcus Bontempelli | Western Bulldogs | 6.47 |

League distribution (eligible players, season-to-date): mean **1.39**, std 1.62, p10 0.09 / p50 0.76 / p90 3.98, max 7.94.

Top per-game correlates: `contested_possessions` (r = +0.74), `handballs` (r = +0.56), `disposals` (r = +0.49).

#### Tackles per game

**What it measures.** Pressure acts that physically stop a ball-carrier. **Why it matters.** Defensive midfield work — the unsung currency of forward-half pressure and turnover football. It correlates with clearances (you tackle the same opponent you compete against) but tells a different story.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Sam Berry | Adelaide | 7.82 |
| 2 | Errol Gulden | Sydney | 6.80 |
| 3 | Josh Dunkley | Brisbane Lions | 6.59 |
| 4 | Jack Steele | Melbourne | 6.41 |
| 5 | James Rowbottom | Sydney | 6.35 |

League distribution (eligible players, season-to-date): mean **2.37**, std 1.24, p10 1.07 / p50 2.11 / p90 4.19, max 7.82.

Top per-game correlates: `clearances` (r = +0.39), `contested_possessions` (r = +0.36), `handballs` (r = +0.30).

#### Hit-outs per game (ruckmen only)

**What it measures.** Wins by a ruckman at a ruck contest (the tap from a centre bounce or stoppage). **Why it matters.** Ruckman-only stat — the distribution is bimodal: ~1 player per team registers double-digits, everyone else is 0. Always read this leaderboard as "top ruckmen", not "top players".

**Bimodal distribution warning.** 88% of eligible 2026 players average less than 1 hit-out per game — they are not ruckmen. The league mean below is dragged down by all the zeros; the meaningful comparison is between ruckmen, where the top of the distribution sits in the 25-35 range.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Brodie Grundy | Sydney | 35.2 |
| 2 | Max Gawn | Melbourne | 31.9 |
| 3 | Lachlan Mcandrew | Adelaide | 31.5 |
| 4 | Ned Moyle | Gold Coast | 31.0 |
| 5 | Jordon Sweet | Port Adelaide | 30.8 |

League distribution (eligible players, season-to-date): mean **1.55**, std 5.37, p10 0.00 / p50 0.00 / p90 1.80, max 35.19.

Top per-game correlates: `clearances` (r = +0.26), `uncontested_possessions` (r = -0.24), `contested_possessions` (r = +0.20).

### Territory stats — moving the ball forward

#### Inside 50s per game

**What it measures.** Disposals or carries that move the ball into the team's attacking 50m arc. **Why it matters.** Territory currency — the precondition for goals. Wing/half-forward players who launch attacks lead this stat. It correlates with kicks and disposals because most inside-50s are foot-delivered.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Bailey Smith | Geelong | 7.06 |
| 2 | Nick Daicos | Collingwood | 6.81 |
| 3 | Errol Gulden | Sydney | 6.80 |
| 4 | Chad Warner | Sydney | 6.59 |
| 5 | Ed Richards | Western Bulldogs | 6.50 |

League distribution (eligible players, season-to-date): mean **2.17**, std 1.23, p10 0.75 / p50 2.07 / p90 3.76, max 7.06.

Top per-game correlates: `disposals` (r = +0.52), `effective_disposals` (r = +0.48), `kicks` (r = +0.48).

#### Marks per game

**What it measures.** Total uncontested + contested marks taken. **Why it matters.** Aerial dominance and intercept defence. Loose-half-back roles dominate the total-marks leaderboard because they sit behind the play and fly under kicks. Tall forwards lead a separate, narrower stat — marks inside 50.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Callum Wilkie | St Kilda | 10.2 |
| 2 | Aliir Aliir | Port Adelaide | 8.1 |
| 3 | Ryan Lester | Brisbane Lions | 7.8 |
| 4 | Lachie Ash | Greater Western Sydney | 7.6 |
| 5 | Blake Acres | Carlton | 7.5 |

League distribution (eligible players, season-to-date): mean **3.84**, std 1.55, p10 2.00 / p50 3.71 / p90 5.98, max 10.18.

Top per-game correlates: `kicks` (r = +0.57), `uncontested_possessions` (r = +0.54), `effective_disposals` (r = +0.43).

#### Marks inside 50 per game

**What it measures.** Marks taken inside the attacking 50m arc — i.e. marks that turn directly into shots on goal. **Why it matters.** This is the strongest single predictor of a forward's goal output. It is what separates a deep-forward role from a high-half-forward role, and the correlation with goals is the highest of any stat in this section.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Gunston | Hawthorn | 4.18 |
| 2 | Mitch Georgiades | Port Adelaide | 3.71 |
| 3 | Jye Amiss | Fremantle | 3.29 |
| 4 | Josh Treacy | Fremantle | 3.29 |
| 5 | Logan Morris | Brisbane Lions | 3.19 |

League distribution (eligible players, season-to-date): mean **0.50**, std 0.66, p10 0.00 / p50 0.25 / p90 1.47, max 4.18.

Top per-game correlates: `goals` (r = +0.67), `behinds` (r = +0.54), `contested_marks` (r = +0.35).

### Discipline stats — errors and free kicks

#### Clangers per game

**What it measures.** Errors — missed targets, fumbles, free kicks given away by the ball-carrier. **Why it matters.** Clangers are the friction term on disposal volume — a high-disposal player who also leads in clangers is being asked to play through traffic, not necessarily playing badly. The correlation with frees-against is mechanical: many clangers *are* frees-against.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Harley Reid | West Coast | 6.65 |
| 2 | Patrick Cripps | Carlton | 5.59 |
| 3 | Jacob Hopper | Richmond | 5.50 |
| 4 | Brodie Grundy | Sydney | 5.31 |
| 5 | Kysaiah Pickett | Melbourne | 5.12 |

League distribution (eligible players, season-to-date): mean **2.39**, std 0.85, p10 1.40 / p50 2.27 / p90 3.50, max 6.65.

Top per-game correlates: `free_kicks_against` (r = +0.61 *(mechanically related)*), `contested_possessions` (r = +0.35), `disposals` (r = +0.32).

#### Free kicks for per game

**What it measures.** Free kicks paid to the player. **Why it matters.** A weak isolated signal — frees-for tracks contest involvement (rucks especially) more than skill. Best used as a tiebreaker rather than a standalone metric.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Tristan Xerri | North Melbourne | 3.54 |
| 2 | Max Gawn | Melbourne | 2.59 |
| 3 | Harley Reid | West Coast | 2.47 |
| 4 | Jai Newcombe | Hawthorn | 2.41 |
| 5 | Sam Darcy | Western Bulldogs | 2.33 |

League distribution (eligible players, season-to-date): mean **0.80**, std 0.44, p10 0.32 / p50 0.75 / p90 1.33, max 3.54.

Top per-game correlates: `contested_possessions` (r = +0.42), `clearances` (r = +0.31), `tackles` (r = +0.20).

#### Free kicks against per game

**What it measures.** Free kicks paid against the player. **Why it matters.** Discipline / aggression marker, with the caveat that ruck contest infringements inflate the number for ruckmen. Reads like a clanger when it correlates with them.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Harley Reid | West Coast | 3.00 |
| 2 | Brodie Grundy | Sydney | 2.75 |
| 3 | Patrick Cripps | Carlton | 2.47 |
| 4 | Mitch Zadow | Port Adelaide | 2.33 |
| 5 | Jack Graham | West Coast | 2.25 |

League distribution (eligible players, season-to-date): mean **0.82**, std 0.44, p10 0.33 / p50 0.76 / p90 1.35, max 3.00.

Top per-game correlates: `clangers` (r = +0.61 *(mechanically related)*), `clearances` (r = +0.17), `contested_possessions` (r = +0.16).

### Team-level stats — what the scoreboard says

Team-level stats use `data/matches/matches_2026.csv` rather than per-player aggregates. Total team score is `goals × 6 + behinds`; margin is the team's score minus the opponent's. A first-quarter score is a useful early-momentum signal — strong starters tend to keep the lead.

#### Total team score per game

| Rank | Team | Avg score | Avg margin | Avg Q1 |
|---|---|---|---|---|
| 1 | Sydney | 109.1 | +31.3 | 27.7 |
| 2 | Brisbane Lions | 106.5 | +14.1 | 24.9 |
| 3 | Fremantle | 100.1 | +30.4 | 24.1 |
| 4 | Hawthorn | 98.7 | +13.7 | 24.3 |
| 5 | Melbourne | 97.9 | +6.6 | 23.1 |

League distribution of per-game team scores: mean **88.3**, std 25.0, p10 60 / p50 86 / p90 122, min 29 / max 170.

#### Winning margin

| Rank | Team | Avg margin | Avg score |
|---|---|---|---|
| 1 | Sydney | +31.3 | 109.1 |
| 2 | Fremantle | +30.4 | 100.1 |
| 3 | Brisbane Lions | +14.1 | 106.5 |
| 4 | Hawthorn | +13.7 | 98.7 |
| 5 | Adelaide | +13.2 | 91.5 |

League distribution of margins (signed, per team-game): mean ~0 by construction, std 41.2, p10 -53 / p50 0 / p90 53.

#### First-quarter score

| Rank | Team | Avg Q1 score | Avg full-game score |
|---|---|---|---|
| 1 | Sydney | 27.7 | 109.1 |
| 2 | Geelong | 25.3 | 97.9 |
| 3 | Brisbane Lions | 24.9 | 106.5 |
| 4 | St Kilda | 24.8 | 92.4 |
| 5 | Hawthorn | 24.3 | 98.7 |

League distribution of Q1 scores: mean **21.7**, std 11.0, p10 8 / p50 21 / p90 38.

### Going deeper with this repo's models

For the stats above, three artefacts in this repo will help you form your own view rather than just reading a leaderboard:

1. The **disposal prediction model** (`prediction.py` / `prediction_cpu.py`) forecasts a player's next-round disposal count using rolling form (3/5-game, season-to-date) and opponent context. Run it with `--player surname_first --rounds 1` to see how uncertainty is quantified for any of the leaders shown above.
2. The **backtest framework** (`backtest.py`) replays a season round-by-round so you can see how the model performed on real, out-of-sample games — the honest way to judge whether a leaderboard ranking will continue to hold.
3. The **Brownlow proxy section** above is the same per-game stat structure used here, weighted into a single composite. If you want a quick "who's having the best year overall" answer rather than per-stat leaders, that table is the one to look at.
<!-- 2026-STAT-LEADERS-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Brownlow predictor](afl-brownlow-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
