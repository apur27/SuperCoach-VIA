# 2026 player performance stats - what to look for and what the data says

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-STAT-LEADERS-START -->
This section is a guide to the AFL performance statistics that fans, analysts and SuperCoach players track most closely — what each stat measures, who is leading it in 2026, what the league-wide distribution looks like, and which other stats most reliably predict it. All numbers are computed live from `data/player_data/` for 2026 (rounds 1-10, **503 eligible players** with >=3 games, **3606 player-games** included). Correlations are Pearson r on the per-game frame; with several thousand player-games, p-values are universally tiny — read the magnitude of r, not the significance star.

![2026 AFL statistical leaders](../assets/charts/player_stat_leaders_2026.png)

### Disposal-based stats — volume and quality of ball use

#### Disposals per game

**What it measures.** Total kicks plus handballs in a game — the single broadest measure of how often a player has the ball. **Why it matters.** It is the headline SuperCoach scoring stat and the prediction target this repo's main model is built around. Volume midfielders and rebounding defenders dominate this leaderboard.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 35.6 |
| 2 | Archie Roberts | Essendon | 32.7 |
| 3 | Bailey Smith | Geelong | 32.2 |
| 4 | Harry Sheezel | North Melbourne | 31.1 |
| 5 | Lachie Neale | Brisbane Lions | 30.9 |

League distribution (eligible players, season-to-date): mean **15.51**, std 5.83, p10 8.67 / p50 14.56 / p90 23.80, max 35.62.

Top per-game correlates: `effective_disposals` (r = +0.97 *(mechanically related)*), `uncontested_possessions` (r = +0.88), `kicks` (r = +0.83).

#### Kicks per game

**What it measures.** Just the kicked disposals. **Why it matters.** Kicks tend to come from outside-midfielders, half-backs and tall rebounders — players who clear the ball by foot rather than shovel it into a contest. A player who kicks much more than they handball is usually playing a distributor / launch role.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Archie Roberts | Essendon | 23.1 |
| 2 | Bailey Smith | Geelong | 20.6 |
| 3 | Jack Sinclair | St Kilda | 20.6 |
| 4 | Nick Daicos | Collingwood | 19.8 |
| 5 | Nasiah Wanganeen-Milera | St Kilda | 19.5 |

League distribution (eligible players, season-to-date): mean **9.01**, std 3.70, p10 4.62 / p50 8.62 / p90 14.25, max 23.11.

Top per-game correlates: `disposals` (r = +0.83), `effective_disposals` (r = +0.81), `uncontested_possessions` (r = +0.79).

#### Handballs per game

**What it measures.** The hand-passed half of disposals. **Why it matters.** Handball volume tracks contest involvement — a player wins the ball at a stoppage, then handballs out to a runner. Inside-mids and clearance specialists tend to lead this stat.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Clayton Oliver | Greater Western Sydney | 20.6 |
| 2 | Ryley Sanders | Western Bulldogs | 18.2 |
| 3 | Finn Callaghan | Greater Western Sydney | 16.0 |
| 4 | Nick Daicos | Collingwood | 15.9 |
| 5 | Jack Ross | Richmond | 15.8 |

League distribution (eligible players, season-to-date): mean **6.50**, std 3.20, p10 3.00 / p50 6.00 / p90 11.00, max 20.56.

Top per-game correlates: `disposals` (r = +0.77), `effective_disposals` (r = +0.74), `contested_possessions` (r = +0.65).

#### Effective disposals per game (disposals − clangers)

**What it measures.** Disposals that did not result in a clanger, computed here as `max(disposals - clangers, 0)` because the raw data does not carry a true effective-disposal column. **Why it matters.** It is a defensible proxy for disposal *quality* — high-volume ball-users who don't turn it over. The same proxy is used in the Brownlow predictor on this page.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 30.1 |
| 2 | Archie Roberts | Essendon | 29.4 |
| 3 | Bailey Smith | Geelong | 28.3 |
| 4 | Lachie Neale | Brisbane Lions | 28.1 |
| 5 | Lachie Whitfield | Greater Western Sydney | 27.4 |

League distribution (eligible players, season-to-date): mean **13.05**, std 5.44, p10 6.41 / p50 12.22 / p90 21.11, max 30.12.

Top per-game correlates: `disposals` (r = +0.97 *(mechanically related)*), `uncontested_possessions` (r = +0.87), `kicks` (r = +0.81).

### Scoring stats — goals, behinds and conversion

#### Goals per game

**What it measures.** Goals kicked. **Why it matters.** Forwards live and die by this stat. It is volatile game-to-game (a single missed shot can halve your score), so multi-game averages and shot-source context (marks-inside-50, contested marks) matter more than any one game.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Gunston | Hawthorn | 3.75 |
| 2 | Ben King | Gold Coast | 3.56 |
| 3 | Jeremy Cameron | Geelong | 3.00 |
| 4 | Mitch Georgiades | Port Adelaide | 2.78 |
| 5 | Logan Morris | Brisbane Lions | 2.62 |

League distribution (eligible players, season-to-date): mean **0.56**, std 0.64, p10 0.00 / p50 0.33 / p90 1.44, max 3.75.

Top per-game correlates: `marks_inside_50` (r = +0.68), `behinds` (r = +0.32), `rebound_50s` (r = -0.31).

**Goal conversion rate.** Defined as `goals / (goals + behinds)`, season-to-date, for players with >=2 goals total. League distribution (n=299): mean **61.8%**, std 18.1pp, p10 38% / p50 60% / p90 86%.

| Rank | Player | Team | G | B | Conversion |
|---|---|---|---|---|---|
| 1 | Jordan Dawson | Adelaide | 7 | 0 | 100.0% |
| 2 | Isaac Cumming | Adelaide | 5 | 0 | 100.0% |
| 3 | Sam Durham | Essendon | 5 | 0 | 100.0% |
| 4 | Ed Richards | Western Bulldogs | 5 | 0 | 100.0% |
| 5 | Paddy Cross | Melbourne | 4 | 0 | 100.0% |

#### Behinds per game

**What it measures.** Minor scores — shots that hit the post or go through the smaller posts. **Why it matters.** Rarely predicted alone — it is too noisy. Best read alongside goals to compute **conversion rate** (`goals / (goals + behinds)`), the cleanest available signal of forward accuracy.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jake Waterman | West Coast | 2.89 |
| 2 | Mitch Georgiades | Port Adelaide | 2.44 |
| 3 | Nate Caddy | Essendon | 2.38 |
| 4 | Jack Gunston | Hawthorn | 2.38 |
| 5 | Tom Lynch | Richmond | 2.20 |

League distribution (eligible players, season-to-date): mean **0.41**, std 0.46, p10 0.00 / p50 0.29 / p90 1.00, max 2.89.

Top per-game correlates: `marks_inside_50` (r = +0.55), `goals` (r = +0.32), `rebound_50s` (r = -0.24).

### Contested and ground-ball stats — the inside game

#### Contested possessions per game

**What it measures.** Wins of the ball under physical pressure — ground-balls, taps, and contested marks. **Why it matters.** This is the cleanest stat for separating a midfielder's *contest* role from an outside ball-user's *spread* role. It correlates strongly with clearances and tackles.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Tristan Xerri | North Melbourne | 18.50 |
| 2 | Patrick Cripps | Carlton | 14.67 |
| 3 | Max Gawn | Melbourne | 14.67 |
| 4 | Clayton Oliver | Greater Western Sydney | 14.56 |
| 5 | Isaac Heeney | Sydney | 13.57 |

League distribution (eligible players, season-to-date): mean **5.46**, std 2.45, p10 3.00 / p50 4.88 / p90 8.85, max 18.50.

Top per-game correlates: `clearances` (r = +0.74), `handballs` (r = +0.65), `disposals` (r = +0.58).

#### Clearances per game

**What it measures.** Disposals that move the ball clear of a stoppage (a centre-bounce or boundary throw-in). **Why it matters.** Stoppage dominance is one of the few team-level wins a midfield can manufacture. Top clearance players are almost always the inside-mid fulcrums of their team.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Tristan Xerri | North Melbourne | 8.83 |
| 2 | Jai Newcombe | Hawthorn | 7.89 |
| 3 | Clayton Oliver | Greater Western Sydney | 7.78 |
| 4 | Lachie Neale | Brisbane Lions | 7.33 |
| 5 | Matthew Kennedy | Western Bulldogs | 7.22 |

League distribution (eligible players, season-to-date): mean **1.45**, std 1.67, p10 0.00 / p50 0.78 / p90 4.11, max 8.83.

Top per-game correlates: `contested_possessions` (r = +0.74), `handballs` (r = +0.56), `disposals` (r = +0.48).

#### Tackles per game

**What it measures.** Pressure acts that physically stop a ball-carrier. **Why it matters.** Defensive midfield work — the unsung currency of forward-half pressure and turnover football. It correlates with clearances (you tackle the same opponent you compete against) but tells a different story.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Sam Berry | Adelaide | 7.44 |
| 2 | James Rowbottom | Sydney | 6.89 |
| 3 | Tristan Xerri | North Melbourne | 6.83 |
| 4 | Jack Steele | Melbourne | 6.78 |
| 5 | Andrew Brayshaw | Fremantle | 6.78 |

League distribution (eligible players, season-to-date): mean **2.37**, std 1.34, p10 1.00 / p50 2.11 / p90 4.25, max 7.44.

Top per-game correlates: `clearances` (r = +0.41), `contested_possessions` (r = +0.38), `handballs` (r = +0.31).

#### Hit-outs per game (ruckmen only)

**What it measures.** Wins by a ruckman at a ruck contest (the tap from a centre bounce or stoppage). **Why it matters.** Ruckman-only stat — the distribution is bimodal: ~1 player per team registers double-digits, everyone else is 0. Always read this leaderboard as "top ruckmen", not "top players".

**Bimodal distribution warning.** 87% of eligible 2026 players average less than 1 hit-out per game — they are not ruckmen. The league mean below is dragged down by all the zeros; the meaningful comparison is between ruckmen, where the top of the distribution sits in the 25-35 range.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jarrod Witts | Gold Coast | 35.7 |
| 2 | Brodie Grundy | Sydney | 35.2 |
| 3 | Max Gawn | Melbourne | 31.2 |
| 4 | Jordon Sweet | Port Adelaide | 29.5 |
| 5 | Nick Madden | Greater Western Sydney | 28.7 |

League distribution (eligible players, season-to-date): mean **1.53**, std 5.28, p10 0.00 / p50 0.00 / p90 2.23, max 35.71.

Top per-game correlates: `clearances` (r = +0.24), `uncontested_possessions` (r = -0.24), `free_kicks_for` (r = +0.21).

### Territory stats — moving the ball forward

#### Inside 50s per game

**What it measures.** Disposals or carries that move the ball into the team's attacking 50m arc. **Why it matters.** Territory currency — the precondition for goals. Wing/half-forward players who launch attacks lead this stat. It correlates with kicks and disposals because most inside-50s are foot-delivered.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 7.62 |
| 2 | Bailey Smith | Geelong | 6.89 |
| 3 | Chad Warner | Sydney | 6.78 |
| 4 | Marcus Bontempelli | Western Bulldogs | 6.56 |
| 5 | Ed Richards | Western Bulldogs | 6.12 |

League distribution (eligible players, season-to-date): mean **2.25**, std 1.31, p10 0.71 / p50 2.11 / p90 3.89, max 7.62.

Top per-game correlates: `disposals` (r = +0.52), `effective_disposals` (r = +0.49), `kicks` (r = +0.47).

#### Marks per game

**What it measures.** Total uncontested + contested marks taken. **Why it matters.** Aerial dominance and intercept defence. Loose-half-back roles dominate the total-marks leaderboard because they sit behind the play and fly under kicks. Tall forwards lead a separate, narrower stat — marks inside 50.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Callum Wilkie | St Kilda | 11.0 |
| 2 | Aliir Aliir | Port Adelaide | 10.3 |
| 3 | Lachie Ash | Greater Western Sydney | 8.6 |
| 4 | Ryan Lester | Brisbane Lions | 8.4 |
| 5 | Jayden Short | Richmond | 8.4 |

League distribution (eligible players, season-to-date): mean **3.98**, std 1.70, p10 2.00 / p50 3.86 / p90 6.33, max 11.00.

Top per-game correlates: `kicks` (r = +0.59), `uncontested_possessions` (r = +0.54), `effective_disposals` (r = +0.44).

#### Marks inside 50 per game

**What it measures.** Marks taken inside the attacking 50m arc — i.e. marks that turn directly into shots on goal. **Why it matters.** This is the strongest single predictor of a forward's goal output. It is what separates a deep-forward role from a high-half-forward role, and the correlation with goals is the highest of any stat in this section.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Gunston | Hawthorn | 4.62 |
| 2 | Mitch Georgiades | Port Adelaide | 4.56 |
| 3 | Logan Morris | Brisbane Lions | 3.38 |
| 4 | Jeremy Cameron | Geelong | 3.38 |
| 5 | Josh Treacy | Fremantle | 3.33 |

League distribution (eligible players, season-to-date): mean **0.53**, std 0.73, p10 0.00 / p50 0.25 / p90 1.56, max 4.62.

Top per-game correlates: `goals` (r = +0.68), `behinds` (r = +0.55), `contested_marks` (r = +0.36).

### Discipline stats — errors and free kicks

#### Clangers per game

**What it measures.** Errors — missed targets, fumbles, free kicks given away by the ball-carrier. **Why it matters.** Clangers are the friction term on disposal volume — a high-disposal player who also leads in clangers is being asked to play through traffic, not necessarily playing badly. The correlation with frees-against is mechanical: many clangers *are* frees-against.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Matt Rowell | Gold Coast | 6.75 |
| 2 | Jacob Hopper | Richmond | 6.50 |
| 3 | Harley Reid | West Coast | 6.33 |
| 4 | Brodie Grundy | Sydney | 6.22 |
| 5 | Nick Daicos | Collingwood | 5.50 |

League distribution (eligible players, season-to-date): mean **2.46**, std 0.98, p10 1.33 / p50 2.25 / p90 3.67, max 6.75.

Top per-game correlates: `free_kicks_against` (r = +0.63 *(mechanically related)*), `contested_possessions` (r = +0.35), `disposals` (r = +0.32).

#### Free kicks for per game

**What it measures.** Free kicks paid to the player. **Why it matters.** A weak isolated signal — frees-for tracks contest involvement (rucks especially) more than skill. Best used as a tiebreaker rather than a standalone metric.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Tristan Xerri | North Melbourne | 4.33 |
| 2 | Darcy Cameron | Collingwood | 2.89 |
| 3 | Max Gawn | Melbourne | 2.89 |
| 4 | Jai Newcombe | Hawthorn | 2.56 |
| 5 | Tim English | Western Bulldogs | 2.33 |

League distribution (eligible players, season-to-date): mean **0.82**, std 0.51, p10 0.25 / p50 0.75 / p90 1.44, max 4.33.

Top per-game correlates: `contested_possessions` (r = +0.43), `clearances` (r = +0.30), `tackles` (r = +0.23).

#### Free kicks against per game

**What it measures.** Free kicks paid against the player. **Why it matters.** Discipline / aggression marker, with the caveat that ruck contest infringements inflate the number for ruckmen. Reads like a clanger when it correlates with them.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Brodie Grundy | Sydney | 3.11 |
| 2 | Matt Rowell | Gold Coast | 3.00 |
| 3 | Jack Graham | West Coast | 3.00 |
| 4 | Harley Reid | West Coast | 2.56 |
| 5 | Patrick Cripps | Carlton | 2.44 |

League distribution (eligible players, season-to-date): mean **0.83**, std 0.49, p10 0.30 / p50 0.75 / p90 1.44, max 3.11.

Top per-game correlates: `clangers` (r = +0.63 *(mechanically related)*), `clearances` (r = +0.17), `contested_possessions` (r = +0.16).

### Team-level stats — what the scoreboard says

Team-level stats use `data/matches/matches_2026.csv` rather than per-player aggregates. Total team score is `goals × 6 + behinds`; margin is the team's score minus the opponent's. A first-quarter score is a useful early-momentum signal — strong starters tend to keep the lead.

#### Total team score per game

| Rank | Team | Avg score | Avg margin | Avg Q1 |
|---|---|---|---|---|
| 1 | Sydney | 118.1 | +46.8 | 29.6 |
| 2 | Brisbane Lions | 107.8 | +22.2 | 22.5 |
| 3 | Hawthorn | 103.6 | +18.1 | 29.0 |
| 4 | Melbourne | 101.4 | +0.2 | 19.1 |
| 5 | Gold Coast | 100.8 | +13.9 | 26.1 |

League distribution of per-game team scores: mean **91.1**, std 25.6, p10 60 / p50 90 / p90 128, min 35 / max 163.

#### Winning margin

| Rank | Team | Avg margin | Avg score |
|---|---|---|---|
| 1 | Sydney | +46.8 | 118.1 |
| 2 | Fremantle | +23.5 | 94.8 |
| 3 | Brisbane Lions | +22.2 | 107.8 |
| 4 | Hawthorn | +18.1 | 103.6 |
| 5 | Gold Coast | +13.9 | 100.8 |

League distribution of margins (signed, per team-game): mean ~0 by construction, std 43.1, p10 -56 / p50 0 / p90 56.

#### First-quarter score

| Rank | Team | Avg Q1 score | Avg full-game score |
|---|---|---|---|
| 1 | Sydney | 29.6 | 118.1 |
| 2 | Hawthorn | 29.0 | 103.6 |
| 3 | North Melbourne | 26.6 | 95.5 |
| 4 | Gold Coast | 26.1 | 100.8 |
| 5 | Western Bulldogs | 25.5 | 90.0 |

League distribution of Q1 scores: mean **22.3**, std 11.7, p10 8 / p50 21 / p90 40.

### Going deeper with this repo's models

For the stats above, three artefacts in this repo will help you form your own view rather than just reading a leaderboard:

1. The **disposal prediction model** (`prediction.py` / `prediction_cpu.py`) forecasts a player's next-round disposal count using rolling form, opponent context and venue effects. Run it with `--player surname_first --rounds 1` to see how uncertainty is quantified for any of the leaders shown above.
2. The **backtest framework** (`backtest.py`) replays a season round-by-round so you can see how the model performed on real, out-of-sample games — the honest way to judge whether a leaderboard ranking will continue to hold.
3. The **Brownlow proxy section** above is the same per-game stat structure used here, weighted into a single composite. If you want a quick "who's having the best year overall" answer rather than per-stat leaders, that table is the one to look at.
<!-- 2026-STAT-LEADERS-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Brownlow predictor](afl-brownlow-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
