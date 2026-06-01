# 2026 player performance stats - what to look for and what the data says

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-STAT-LEADERS-START -->
This section is a guide to the AFL performance statistics that fans, analysts and SuperCoach players track most closely — what each stat measures, who is leading it in 2026, what the league-wide distribution looks like, and which other stats most reliably predict it. All numbers are computed live from `data/player_data/` for 2026 (rounds 1-13, **534 eligible players** with >=3 games, **4719 player-games** included). Correlations are Pearson r on the per-game frame; with several thousand player-games, p-values are universally tiny — read the magnitude of r, not the significance star.

![2026 AFL statistical leaders](../assets/charts/player_stat_leaders_2026.png)

### Disposal-based stats — volume and quality of ball use

#### Disposals per game

**What it measures.** Total kicks plus handballs in a game — the single broadest measure of how often a player has the ball. **Why it matters.** It is the headline SuperCoach scoring stat and the prediction target this repo's main model is built around. Volume midfielders and rebounding defenders dominate this leaderboard.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 34.9 |
| 2 | Bailey Smith | Geelong | 32.6 |
| 3 | Archie Roberts | Essendon | 32.3 |
| 4 | Clayton Oliver | Greater Western Sydney | 31.4 |
| 5 | Zak Butters | Port Adelaide | 30.7 |

League distribution (eligible players, season-to-date): mean **15.32**, std 5.78, p10 8.58 / p50 14.48 / p90 23.72, max 34.91.

Top per-game correlates: `effective_disposals` (r = +0.97 *(mechanically related)*), `uncontested_possessions` (r = +0.88), `kicks` (r = +0.83).

#### Kicks per game

**What it measures.** Just the kicked disposals. **Why it matters.** Kicks tend to come from outside-midfielders, half-backs and tall rebounders — players who clear the ball by foot rather than shovel it into a contest. A player who kicks much more than they handball is usually playing a distributor / launch role.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Archie Roberts | Essendon | 22.4 |
| 2 | Jack Sinclair | St Kilda | 21.0 |
| 3 | Bailey Smith | Geelong | 19.8 |
| 4 | Nasiah Wanganeen-Milera | St Kilda | 19.5 |
| 5 | Lachie Ash | Greater Western Sydney | 19.5 |

League distribution (eligible players, season-to-date): mean **8.87**, std 3.65, p10 4.50 / p50 8.41 / p90 13.97, max 22.42.

Top per-game correlates: `disposals` (r = +0.83), `effective_disposals` (r = +0.81), `uncontested_possessions` (r = +0.78).

#### Handballs per game

**What it measures.** The hand-passed half of disposals. **Why it matters.** Handball volume tracks contest involvement — a player wins the ball at a stoppage, then handballs out to a runner. Inside-mids and clearance specialists tend to lead this stat.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Clayton Oliver | Greater Western Sydney | 21.1 |
| 2 | Ryley Sanders | Western Bulldogs | 17.2 |
| 3 | Patrick Cripps | Carlton | 16.6 |
| 4 | Sam Walsh | Carlton | 15.8 |
| 5 | Nick Daicos | Collingwood | 15.7 |

League distribution (eligible players, season-to-date): mean **6.45**, std 3.14, p10 3.00 / p50 5.96 / p90 10.87, max 21.08.

Top per-game correlates: `disposals` (r = +0.77), `effective_disposals` (r = +0.74), `contested_possessions` (r = +0.66).

#### Effective disposals per game (disposals − clangers)

**What it measures.** Disposals that did not result in a clanger, computed here as `max(disposals - clangers, 0)` because the raw data does not carry a true effective-disposal column. **Why it matters.** It is a defensible proxy for disposal *quality* — high-volume ball-users who don't turn it over. The same proxy is used in the Brownlow predictor on this page.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 29.6 |
| 2 | Bailey Smith | Geelong | 28.9 |
| 3 | Archie Roberts | Essendon | 28.8 |
| 4 | Jack Sinclair | St Kilda | 27.3 |
| 5 | Zak Butters | Port Adelaide | 27.1 |

League distribution (eligible players, season-to-date): mean **12.88**, std 5.40, p10 6.50 / p50 12.22 / p90 20.91, max 29.64.

Top per-game correlates: `disposals` (r = +0.97 *(mechanically related)*), `uncontested_possessions` (r = +0.87), `kicks` (r = +0.81).

### Scoring stats — goals, behinds and conversion

#### Goals per game

**What it measures.** Goals kicked. **Why it matters.** Forwards live and die by this stat. It is volatile game-to-game (a single missed shot can halve your score), so multi-game averages and shot-source context (marks-inside-50, contested marks) matter more than any one game.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Gunston | Hawthorn | 3.89 |
| 2 | Jeremy Cameron | Geelong | 3.09 |
| 3 | Ben King | Gold Coast | 3.09 |
| 4 | Charlie Curnow | Sydney | 2.82 |
| 5 | Jye Amiss | Fremantle | 2.50 |

League distribution (eligible players, season-to-date): mean **0.54**, std 0.61, p10 0.00 / p50 0.33 / p90 1.42, max 3.89.

Top per-game correlates: `marks_inside_50` (r = +0.67), `behinds` (r = +0.32), `rebound_50s` (r = -0.31).

**Goal conversion rate.** Defined as `goals / (goals + behinds)`, season-to-date, for players with >=2 goals total. League distribution (n=328): mean **60.9%**, std 16.9pp, p10 40% / p50 60% / p90 81%.

| Rank | Player | Team | G | B | Conversion |
|---|---|---|---|---|---|
| 1 | Isaac Cumming | Adelaide | 5 | 0 | 100.0% |
| 2 | Sam Durham | Essendon | 5 | 0 | 100.0% |
| 3 | Charlie Spargo | North Melbourne | 5 | 0 | 100.0% |
| 4 | Joel Jeffrey | Gold Coast | 4 | 0 | 100.0% |
| 5 | Noah Roberts-Thomson | Richmond | 4 | 0 | 100.0% |

#### Behinds per game

**What it measures.** Minor scores — shots that hit the post or go through the smaller posts. **Why it matters.** Rarely predicted alone — it is too noisy. Best read alongside goals to compute **conversion rate** (`goals / (goals + behinds)`), the cleanest available signal of forward accuracy.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jake Waterman | West Coast | 2.67 |
| 2 | Mitch Georgiades | Port Adelaide | 2.36 |
| 3 | Jack Gunston | Hawthorn | 2.33 |
| 4 | Nate Caddy | Essendon | 2.18 |
| 5 | Tom Lynch | Richmond | 2.17 |

League distribution (eligible players, season-to-date): mean **0.41**, std 0.44, p10 0.00 / p50 0.29 / p90 1.00, max 2.67.

Top per-game correlates: `marks_inside_50` (r = +0.54), `goals` (r = +0.32), `rebound_50s` (r = -0.24).

### Contested and ground-ball stats — the inside game

#### Contested possessions per game

**What it measures.** Wins of the ball under physical pressure — ground-balls, taps, and contested marks. **Why it matters.** This is the cleanest stat for separating a midfielder's *contest* role from an outside ball-user's *spread* role. It correlates strongly with clearances and tackles.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Tristan Xerri | North Melbourne | 16.12 |
| 2 | Clayton Oliver | Greater Western Sydney | 15.75 |
| 3 | Patrick Cripps | Carlton | 15.08 |
| 4 | Max Gawn | Melbourne | 14.08 |
| 5 | Isaac Heeney | Sydney | 13.90 |

League distribution (eligible players, season-to-date): mean **5.40**, std 2.40, p10 3.00 / p50 4.75 / p90 8.67, max 16.12.

Top per-game correlates: `clearances` (r = +0.74), `handballs` (r = +0.66), `disposals` (r = +0.58).

#### Clearances per game

**What it measures.** Disposals that move the ball clear of a stoppage (a centre-bounce or boundary throw-in). **Why it matters.** Stoppage dominance is one of the few team-level wins a midfield can manufacture. Top clearance players are almost always the inside-mid fulcrums of their team.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Clayton Oliver | Greater Western Sydney | 8.33 |
| 2 | Tristan Xerri | North Melbourne | 8.00 |
| 3 | Jai Newcombe | Hawthorn | 7.58 |
| 4 | Lachie Neale | Brisbane Lions | 7.42 |
| 5 | Patrick Cripps | Carlton | 7.25 |

League distribution (eligible players, season-to-date): mean **1.42**, std 1.64, p10 0.09 / p50 0.76 / p90 3.90, max 8.33.

Top per-game correlates: `contested_possessions` (r = +0.74), `handballs` (r = +0.55), `disposals` (r = +0.49).

#### Tackles per game

**What it measures.** Pressure acts that physically stop a ball-carrier. **Why it matters.** Defensive midfield work — the unsung currency of forward-half pressure and turnover football. It correlates with clearances (you tackle the same opponent you compete against) but tells a different story.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Sam Berry | Adelaide | 7.70 |
| 2 | James Rowbottom | Sydney | 6.92 |
| 3 | Jack Graham | West Coast | 6.60 |
| 4 | Tom Atkins | Geelong | 6.50 |
| 5 | Willem Drew | Port Adelaide | 6.45 |

League distribution (eligible players, season-to-date): mean **2.37**, std 1.26, p10 1.03 / p50 2.14 / p90 4.21, max 7.70.

Top per-game correlates: `clearances` (r = +0.40), `contested_possessions` (r = +0.36), `handballs` (r = +0.30).

#### Hit-outs per game (ruckmen only)

**What it measures.** Wins by a ruckman at a ruck contest (the tap from a centre bounce or stoppage). **Why it matters.** Ruckman-only stat — the distribution is bimodal: ~1 player per team registers double-digits, everyone else is 0. Always read this leaderboard as "top ruckmen", not "top players".

**Bimodal distribution warning.** 87% of eligible 2026 players average less than 1 hit-out per game — they are not ruckmen. The league mean below is dragged down by all the zeros; the meaningful comparison is between ruckmen, where the top of the distribution sits in the 25-35 range.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Brodie Grundy | Sydney | 35.8 |
| 2 | Jarrod Witts | Gold Coast | 34.8 |
| 3 | Max Gawn | Melbourne | 30.9 |
| 4 | Jordon Sweet | Port Adelaide | 29.1 |
| 5 | Nick Madden | Greater Western Sydney | 28.7 |

League distribution (eligible players, season-to-date): mean **1.54**, std 5.27, p10 0.00 / p50 0.00 / p90 2.00, max 35.82.

Top per-game correlates: `clearances` (r = +0.25), `uncontested_possessions` (r = -0.25), `free_kicks_for` (r = +0.21).

### Territory stats — moving the ball forward

#### Inside 50s per game

**What it measures.** Disposals or carries that move the ball into the team's attacking 50m arc. **Why it matters.** Territory currency — the precondition for goals. Wing/half-forward players who launch attacks lead this stat. It correlates with kicks and disposals because most inside-50s are foot-delivered.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 7.00 |
| 2 | Bailey Smith | Geelong | 6.83 |
| 3 | Chad Warner | Sydney | 6.75 |
| 4 | Ed Richards | Western Bulldogs | 6.18 |
| 5 | Christian Petracca | Gold Coast | 6.11 |

League distribution (eligible players, season-to-date): mean **2.21**, std 1.27, p10 0.75 / p50 2.00 / p90 3.91, max 7.00.

Top per-game correlates: `disposals` (r = +0.53), `effective_disposals` (r = +0.49), `kicks` (r = +0.48).

#### Marks per game

**What it measures.** Total uncontested + contested marks taken. **Why it matters.** Aerial dominance and intercept defence. Loose-half-back roles dominate the total-marks leaderboard because they sit behind the play and fly under kicks. Tall forwards lead a separate, narrower stat — marks inside 50.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Callum Wilkie | St Kilda | 10.1 |
| 2 | Aliir Aliir | Port Adelaide | 8.9 |
| 3 | Ryan Lester | Brisbane Lions | 8.3 |
| 4 | James Sicily | Hawthorn | 8.1 |
| 5 | Lachie Ash | Greater Western Sydney | 8.1 |

League distribution (eligible players, season-to-date): mean **3.90**, std 1.60, p10 2.00 / p50 3.82 / p90 6.09, max 10.08.

Top per-game correlates: `kicks` (r = +0.58), `uncontested_possessions` (r = +0.53), `effective_disposals` (r = +0.43).

#### Marks inside 50 per game

**What it measures.** Marks taken inside the attacking 50m arc — i.e. marks that turn directly into shots on goal. **Why it matters.** This is the strongest single predictor of a forward's goal output. It is what separates a deep-forward role from a high-half-forward role, and the correlation with goals is the highest of any stat in this section.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Gunston | Hawthorn | 4.56 |
| 2 | Mitch Georgiades | Port Adelaide | 4.00 |
| 3 | Josh Treacy | Fremantle | 3.33 |
| 4 | Logan Morris | Brisbane Lions | 3.27 |
| 5 | Jye Amiss | Fremantle | 3.25 |

League distribution (eligible players, season-to-date): mean **0.51**, std 0.69, p10 0.00 / p50 0.27 / p90 1.53, max 4.56.

Top per-game correlates: `goals` (r = +0.67), `behinds` (r = +0.54), `contested_marks` (r = +0.33).

### Discipline stats — errors and free kicks

#### Clangers per game

**What it measures.** Errors — missed targets, fumbles, free kicks given away by the ball-carrier. **Why it matters.** Clangers are the friction term on disposal volume — a high-disposal player who also leads in clangers is being asked to play through traffic, not necessarily playing badly. The correlation with frees-against is mechanical: many clangers *are* frees-against.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Harley Reid | West Coast | 6.75 |
| 2 | Jacob Hopper | Richmond | 6.00 |
| 3 | Brodie Grundy | Sydney | 5.73 |
| 4 | Patrick Cripps | Carlton | 5.33 |
| 5 | Nick Daicos | Collingwood | 5.27 |

League distribution (eligible players, season-to-date): mean **2.44**, std 0.93, p10 1.33 / p50 2.27 / p90 3.57, max 6.75.

Top per-game correlates: `free_kicks_against` (r = +0.63 *(mechanically related)*), `contested_possessions` (r = +0.35), `disposals` (r = +0.32).

#### Free kicks for per game

**What it measures.** Free kicks paid to the player. **Why it matters.** A weak isolated signal — frees-for tracks contest involvement (rucks especially) more than skill. Best used as a tiebreaker rather than a standalone metric.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Tristan Xerri | North Melbourne | 4.25 |
| 2 | Darcy Cameron | Collingwood | 2.82 |
| 3 | Max Gawn | Melbourne | 2.58 |
| 4 | Matt Rowell | Gold Coast | 2.50 |
| 5 | Sam Darcy | Western Bulldogs | 2.33 |

League distribution (eligible players, season-to-date): mean **0.81**, std 0.48, p10 0.29 / p50 0.73 / p90 1.40, max 4.25.

Top per-game correlates: `contested_possessions` (r = +0.43), `clearances` (r = +0.31), `tackles` (r = +0.23).

#### Free kicks against per game

**What it measures.** Free kicks paid against the player. **Why it matters.** Discipline / aggression marker, with the caveat that ruck contest infringements inflate the number for ruckmen. Reads like a clanger when it correlates with them.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Graham | West Coast | 3.00 |
| 2 | Brodie Grundy | Sydney | 2.91 |
| 3 | Harley Reid | West Coast | 2.67 |
| 4 | Patrick Cripps | Carlton | 2.50 |
| 5 | Matt Rowell | Gold Coast | 2.33 |

League distribution (eligible players, season-to-date): mean **0.83**, std 0.47, p10 0.33 / p50 0.75 / p90 1.41, max 3.00.

Top per-game correlates: `clangers` (r = +0.63 *(mechanically related)*), `clearances` (r = +0.18), `contested_possessions` (r = +0.17).

### Team-level stats — what the scoreboard says

Team-level stats use `data/matches/matches_2026.csv` rather than per-player aggregates. Total team score is `goals × 6 + behinds`; margin is the team's score minus the opponent's. A first-quarter score is a useful early-momentum signal — strong starters tend to keep the lead.

#### Total team score per game

| Rank | Team | Avg score | Avg margin | Avg Q1 |
|---|---|---|---|---|
| 1 | Sydney | 116.0 | +42.5 | 28.9 |
| 2 | Gold Coast | 100.9 | +13.0 | 28.3 |
| 3 | Hawthorn | 100.4 | +15.2 | 26.1 |
| 4 | Brisbane Lions | 100.4 | +3.1 | 21.4 |
| 5 | Geelong | 99.9 | +15.0 | 25.4 |

League distribution of per-game team scores: mean **90.6**, std 25.4, p10 60 / p50 88 / p90 126, min 35 / max 170.

#### Winning margin

| Rank | Team | Avg margin | Avg score |
|---|---|---|---|
| 1 | Sydney | +42.5 | 116.0 |
| 2 | Fremantle | +26.0 | 97.2 |
| 3 | Hawthorn | +15.2 | 100.4 |
| 4 | Geelong | +15.0 | 99.9 |
| 5 | Gold Coast | +13.0 | 100.9 |

League distribution of margins (signed, per team-game): mean ~0 by construction, std 42.5, p10 -55 / p50 0 / p90 55.

#### First-quarter score

| Rank | Team | Avg Q1 score | Avg full-game score |
|---|---|---|---|
| 1 | Sydney | 28.9 | 116.0 |
| 2 | Gold Coast | 28.3 | 100.9 |
| 3 | Hawthorn | 26.1 | 100.4 |
| 4 | North Melbourne | 26.0 | 94.0 |
| 5 | Geelong | 25.4 | 99.9 |

League distribution of Q1 scores: mean **22.1**, std 11.2, p10 8 / p50 21 / p90 38.

### Going deeper with this repo's models

For the stats above, three artefacts in this repo will help you form your own view rather than just reading a leaderboard:

1. The **disposal prediction model** (`prediction.py` / `prediction_cpu.py`) forecasts a player's next-round disposal count using rolling form, opponent context and venue effects. Run it with `--player surname_first --rounds 1` to see how uncertainty is quantified for any of the leaders shown above.
2. The **backtest framework** (`backtest.py`) replays a season round-by-round so you can see how the model performed on real, out-of-sample games — the honest way to judge whether a leaderboard ranking will continue to hold.
3. The **Brownlow proxy section** above is the same per-game stat structure used here, weighted into a single composite. If you want a quick "who's having the best year overall" answer rather than per-stat leaders, that table is the one to look at.
<!-- 2026-STAT-LEADERS-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Brownlow predictor](afl-brownlow-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
