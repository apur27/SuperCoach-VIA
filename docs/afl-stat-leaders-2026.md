# 2026 player performance stats - what to look for and what the data says

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-STAT-LEADERS-START -->
This section is a guide to the AFL performance statistics that fans, analysts and SuperCoach players track most closely — what each stat measures, who is leading it in 2026, what the league-wide distribution looks like, and which other stats most reliably predict it. All numbers are computed live from `data/player_data/` for 2026 (rounds 1-15, **548 eligible players** with >=3 games, **5395 player-games** included). Correlations are Pearson r on the per-game frame; with several thousand player-games, p-values are universally tiny — read the magnitude of r, not the significance star.

![2026 AFL statistical leaders](../assets/charts/player_stat_leaders_2026.png)

### Disposal-based stats — volume and quality of ball use

#### Disposals per game

**What it measures.** Total kicks plus handballs in a game — the single broadest measure of how often a player has the ball. **Why it matters.** It is the headline SuperCoach scoring stat and the prediction target this repo's main model is built around. Volume midfielders and rebounding defenders dominate this leaderboard.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 34.9 |
| 2 | Bailey Smith | Geelong | 32.3 |
| 3 | Clayton Oliver | Greater Western Sydney | 31.7 |
| 4 | Harry Sheezel | North Melbourne | 30.5 |
| 5 | Lachie Neale | Brisbane Lions | 30.4 |

League distribution (eligible players, season-to-date): mean **15.26**, std 5.74, p10 8.52 / p50 14.41 / p90 23.69, max 34.92.

Top per-game correlates: `effective_disposals` (r = +0.97 *(mechanically related)*), `uncontested_possessions` (r = +0.88), `kicks` (r = +0.83).

#### Kicks per game

**What it measures.** Just the kicked disposals. **Why it matters.** Kicks tend to come from outside-midfielders, half-backs and tall rebounders — players who clear the ball by foot rather than shovel it into a contest. A player who kicks much more than they handball is usually playing a distributor / launch role.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Archie Roberts | Essendon | 21.0 |
| 2 | Jack Sinclair | St Kilda | 20.4 |
| 3 | Nasiah Wanganeen-Milera | St Kilda | 20.3 |
| 4 | Lachie Ash | Greater Western Sydney | 19.8 |
| 5 | Bailey Smith | Geelong | 19.6 |

League distribution (eligible players, season-to-date): mean **8.83**, std 3.62, p10 4.54 / p50 8.37 / p90 13.66, max 21.00.

Top per-game correlates: `disposals` (r = +0.83), `effective_disposals` (r = +0.81), `uncontested_possessions` (r = +0.78).

#### Handballs per game

**What it measures.** The hand-passed half of disposals. **Why it matters.** Handball volume tracks contest involvement — a player wins the ball at a stoppage, then handballs out to a runner. Inside-mids and clearance specialists tend to lead this stat.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Clayton Oliver | Greater Western Sydney | 20.9 |
| 2 | Ryley Sanders | Western Bulldogs | 16.5 |
| 3 | Patrick Cripps | Carlton | 16.3 |
| 4 | Lachie Neale | Brisbane Lions | 16.1 |
| 5 | Nick Daicos | Collingwood | 16.0 |

League distribution (eligible players, season-to-date): mean **6.44**, std 3.12, p10 3.00 / p50 5.93 / p90 10.84, max 20.92.

Top per-game correlates: `disposals` (r = +0.78), `effective_disposals` (r = +0.75), `contested_possessions` (r = +0.66).

#### Effective disposals per game (disposals − clangers)

**What it measures.** Disposals that did not result in a clanger, computed here as `max(disposals - clangers, 0)` because the raw data does not carry a true effective-disposal column. **Why it matters.** It is a defensible proxy for disposal *quality* — high-volume ball-users who don't turn it over. The same proxy is used in the Brownlow predictor on this page.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 29.6 |
| 2 | Bailey Smith | Geelong | 28.3 |
| 3 | Lachie Neale | Brisbane Lions | 27.5 |
| 4 | Clayton Oliver | Greater Western Sydney | 27.4 |
| 5 | Archie Roberts | Essendon | 26.8 |

League distribution (eligible players, season-to-date): mean **12.86**, std 5.35, p10 6.41 / p50 12.08 / p90 20.79, max 29.58.

Top per-game correlates: `disposals` (r = +0.97 *(mechanically related)*), `uncontested_possessions` (r = +0.86), `kicks` (r = +0.81).

### Scoring stats — goals, behinds and conversion

#### Goals per game

**What it measures.** Goals kicked. **Why it matters.** Forwards live and die by this stat. It is volatile game-to-game (a single missed shot can halve your score), so multi-game averages and shot-source context (marks-inside-50, contested marks) matter more than any one game.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Gunston | Hawthorn | 3.89 |
| 2 | Ben King | Gold Coast | 3.15 |
| 3 | Jeremy Cameron | Geelong | 2.85 |
| 4 | Charlie Curnow | Sydney | 2.77 |
| 5 | Logan Morris | Brisbane Lions | 2.69 |

League distribution (eligible players, season-to-date): mean **0.54**, std 0.62, p10 0.00 / p50 0.33 / p90 1.46, max 3.89.

Top per-game correlates: `marks_inside_50` (r = +0.68), `behinds` (r = +0.33), `rebound_50s` (r = -0.30).

**Goal conversion rate.** Defined as `goals / (goals + behinds)`, season-to-date, for players with >=2 goals total. League distribution (n=349): mean **60.3%**, std 16.7pp, p10 40% / p50 60% / p90 82%.

| Rank | Player | Team | G | B | Conversion |
|---|---|---|---|---|---|
| 1 | Jake Melksham | Melbourne | 6 | 0 | 100.0% |
| 2 | Isaac Cumming | Adelaide | 5 | 0 | 100.0% |
| 3 | Dan Butler | St Kilda | 4 | 0 | 100.0% |
| 4 | Joel Jeffrey | Gold Coast | 4 | 0 | 100.0% |
| 5 | Noah Roberts-Thomson | Richmond | 4 | 0 | 100.0% |

#### Behinds per game

**What it measures.** Minor scores — shots that hit the post or go through the smaller posts. **Why it matters.** Rarely predicted alone — it is too noisy. Best read alongside goals to compute **conversion rate** (`goals / (goals + behinds)`), the cleanest available signal of forward accuracy.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jake Waterman | West Coast | 2.64 |
| 2 | Mitch Georgiades | Port Adelaide | 2.38 |
| 3 | Jack Gunston | Hawthorn | 2.33 |
| 4 | Tom Lynch | Richmond | 2.17 |
| 5 | Logan Morris | Brisbane Lions | 2.15 |

League distribution (eligible players, season-to-date): mean **0.40**, std 0.43, p10 0.00 / p50 0.29 / p90 0.95, max 2.64.

Top per-game correlates: `marks_inside_50` (r = +0.55), `goals` (r = +0.33), `rebound_50s` (r = -0.24).

### Contested and ground-ball stats — the inside game

#### Contested possessions per game

**What it measures.** Wins of the ball under physical pressure — ground-balls, taps, and contested marks. **Why it matters.** This is the cleanest stat for separating a midfielder's *contest* role from an outside ball-user's *spread* role. It correlates strongly with clearances and tackles.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Clayton Oliver | Greater Western Sydney | 15.54 |
| 2 | Tristan Xerri | North Melbourne | 15.20 |
| 3 | Patrick Cripps | Carlton | 14.38 |
| 4 | Brodie Grundy | Sydney | 14.08 |
| 5 | Max Gawn | Melbourne | 13.64 |

League distribution (eligible players, season-to-date): mean **5.38**, std 2.37, p10 3.00 / p50 4.74 / p90 8.75, max 15.54.

Top per-game correlates: `clearances` (r = +0.74), `handballs` (r = +0.66), `disposals` (r = +0.58).

#### Clearances per game

**What it measures.** Disposals that move the ball clear of a stoppage (a centre-bounce or boundary throw-in). **Why it matters.** Stoppage dominance is one of the few team-level wins a midfield can manufacture. Top clearance players are almost always the inside-mid fulcrums of their team.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Clayton Oliver | Greater Western Sydney | 8.31 |
| 2 | Jai Newcombe | Hawthorn | 7.69 |
| 3 | Lachie Neale | Brisbane Lions | 7.43 |
| 4 | Tristan Xerri | North Melbourne | 7.30 |
| 5 | Patrick Cripps | Carlton | 6.92 |

League distribution (eligible players, season-to-date): mean **1.41**, std 1.64, p10 0.08 / p50 0.80 / p90 4.02, max 8.31.

Top per-game correlates: `contested_possessions` (r = +0.74), `handballs` (r = +0.56), `disposals` (r = +0.49).

#### Tackles per game

**What it measures.** Pressure acts that physically stop a ball-carrier. **Why it matters.** Defensive midfield work — the unsung currency of forward-half pressure and turnover football. It correlates with clearances (you tackle the same opponent you compete against) but tells a different story.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Sam Berry | Adelaide | 7.50 |
| 2 | Jack Graham | West Coast | 6.60 |
| 3 | James Rowbottom | Sydney | 6.57 |
| 4 | Tom Atkins | Geelong | 6.50 |
| 5 | Jack Steele | Melbourne | 6.36 |

League distribution (eligible players, season-to-date): mean **2.36**, std 1.25, p10 1.05 / p50 2.13 / p90 4.15, max 7.50.

Top per-game correlates: `clearances` (r = +0.39), `contested_possessions` (r = +0.36), `handballs` (r = +0.30).

#### Hit-outs per game (ruckmen only)

**What it measures.** Wins by a ruckman at a ruck contest (the tap from a centre bounce or stoppage). **Why it matters.** Ruckman-only stat — the distribution is bimodal: ~1 player per team registers double-digits, everyone else is 0. Always read this leaderboard as "top ruckmen", not "top players".

**Bimodal distribution warning.** 87% of eligible 2026 players average less than 1 hit-out per game — they are not ruckmen. The league mean below is dragged down by all the zeros; the meaningful comparison is between ruckmen, where the top of the distribution sits in the 25-35 range.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Lachlan Mcandrew | Adelaide | 35.5 |
| 2 | Brodie Grundy | Sydney | 34.3 |
| 3 | Jarrod Witts | Gold Coast | 32.9 |
| 4 | Max Gawn | Melbourne | 31.9 |
| 5 | Jordon Sweet | Port Adelaide | 30.3 |

League distribution (eligible players, season-to-date): mean **1.59**, std 5.42, p10 0.00 / p50 0.00 / p90 2.09, max 35.50.

Top per-game correlates: `clearances` (r = +0.25), `uncontested_possessions` (r = -0.24), `free_kicks_for` (r = +0.20).

### Territory stats — moving the ball forward

#### Inside 50s per game

**What it measures.** Disposals or carries that move the ball into the team's attacking 50m arc. **Why it matters.** Territory currency — the precondition for goals. Wing/half-forward players who launch attacks lead this stat. It correlates with kicks and disposals because most inside-50s are foot-delivered.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 7.00 |
| 2 | Bailey Smith | Geelong | 6.86 |
| 3 | Ed Richards | Western Bulldogs | 6.77 |
| 4 | Chad Warner | Sydney | 6.64 |
| 5 | Marcus Bontempelli | Western Bulldogs | 5.93 |

League distribution (eligible players, season-to-date): mean **2.19**, std 1.26, p10 0.75 / p50 2.00 / p90 3.92, max 7.00.

Top per-game correlates: `disposals` (r = +0.52), `effective_disposals` (r = +0.49), `kicks` (r = +0.47).

#### Marks per game

**What it measures.** Total uncontested + contested marks taken. **Why it matters.** Aerial dominance and intercept defence. Loose-half-back roles dominate the total-marks leaderboard because they sit behind the play and fly under kicks. Tall forwards lead a separate, narrower stat — marks inside 50.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Callum Wilkie | St Kilda | 9.4 |
| 2 | Aliir Aliir | Port Adelaide | 8.5 |
| 3 | Ryan Lester | Brisbane Lions | 8.3 |
| 4 | James Sicily | Hawthorn | 8.0 |
| 5 | Lachie Ash | Greater Western Sydney | 8.0 |

League distribution (eligible players, season-to-date): mean **3.89**, std 1.58, p10 2.00 / p50 3.75 / p90 6.15, max 9.43.

Top per-game correlates: `kicks` (r = +0.57), `uncontested_possessions` (r = +0.53), `effective_disposals` (r = +0.43).

#### Marks inside 50 per game

**What it measures.** Marks taken inside the attacking 50m arc — i.e. marks that turn directly into shots on goal. **Why it matters.** This is the strongest single predictor of a forward's goal output. It is what separates a deep-forward role from a high-half-forward role, and the correlation with goals is the highest of any stat in this section.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Gunston | Hawthorn | 4.56 |
| 2 | Mitch Georgiades | Port Adelaide | 3.85 |
| 3 | Josh Treacy | Fremantle | 3.46 |
| 4 | Logan Morris | Brisbane Lions | 3.38 |
| 5 | Jye Amiss | Fremantle | 3.31 |

League distribution (eligible players, season-to-date): mean **0.51**, std 0.69, p10 0.00 / p50 0.25 / p90 1.58, max 4.56.

Top per-game correlates: `goals` (r = +0.68), `behinds` (r = +0.55), `contested_marks` (r = +0.34).

### Discipline stats — errors and free kicks

#### Clangers per game

**What it measures.** Errors — missed targets, fumbles, free kicks given away by the ball-carrier. **Why it matters.** Clangers are the friction term on disposal volume — a high-disposal player who also leads in clangers is being asked to play through traffic, not necessarily playing badly. The correlation with frees-against is mechanical: many clangers *are* frees-against.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Harley Reid | West Coast | 6.57 |
| 2 | Jacob Hopper | Richmond | 5.73 |
| 3 | Brodie Grundy | Sydney | 5.46 |
| 4 | Nick Daicos | Collingwood | 5.33 |
| 5 | Toby Greene | Greater Western Sydney | 5.31 |

League distribution (eligible players, season-to-date): mean **2.41**, std 0.91, p10 1.33 / p50 2.28 / p90 3.54, max 6.57.

Top per-game correlates: `free_kicks_against` (r = +0.62 *(mechanically related)*), `contested_possessions` (r = +0.35), `disposals` (r = +0.32).

#### Free kicks for per game

**What it measures.** Free kicks paid to the player. **Why it matters.** A weak isolated signal — frees-for tracks contest involvement (rucks especially) more than skill. Best used as a tiebreaker rather than a standalone metric.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Tristan Xerri | North Melbourne | 3.90 |
| 2 | Darcy Cameron | Collingwood | 2.58 |
| 3 | Max Gawn | Melbourne | 2.50 |
| 4 | Jai Newcombe | Hawthorn | 2.38 |
| 5 | Sam Darcy | Western Bulldogs | 2.33 |

League distribution (eligible players, season-to-date): mean **0.81**, std 0.46, p10 0.29 / p50 0.71 / p90 1.38, max 3.90.

Top per-game correlates: `contested_possessions` (r = +0.42), `clearances` (r = +0.31), `tackles` (r = +0.21).

#### Free kicks against per game

**What it measures.** Free kicks paid against the player. **Why it matters.** Discipline / aggression marker, with the caveat that ruck contest infringements inflate the number for ruckmen. Reads like a clanger when it correlates with them.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Graham | West Coast | 3.00 |
| 2 | Brodie Grundy | Sydney | 2.77 |
| 3 | Harley Reid | West Coast | 2.71 |
| 4 | Patrick Cripps | Carlton | 2.54 |
| 5 | Jordon Sweet | Port Adelaide | 2.17 |

League distribution (eligible players, season-to-date): mean **0.82**, std 0.46, p10 0.33 / p50 0.77 / p90 1.42, max 3.00.

Top per-game correlates: `clangers` (r = +0.62 *(mechanically related)*), `clearances` (r = +0.18), `contested_possessions` (r = +0.16).

### Team-level stats — what the scoreboard says

Team-level stats use `data/matches/matches_2026.csv` rather than per-player aggregates. Total team score is `goals × 6 + behinds`; margin is the team's score minus the opponent's. A first-quarter score is a useful early-momentum signal — strong starters tend to keep the lead.

#### Total team score per game

| Rank | Team | Avg score | Avg margin | Avg Q1 |
|---|---|---|---|---|
| 1 | Sydney | 113.3 | +36.3 | 27.6 |
| 2 | Fremantle | 102.0 | +34.2 | 24.9 |
| 3 | Brisbane Lions | 101.9 | +7.7 | 22.3 |
| 4 | Geelong | 98.3 | +16.1 | 24.8 |
| 5 | Hawthorn | 97.9 | +13.4 | 25.1 |

League distribution of per-game team scores: mean **89.5**, std 25.2, p10 61 / p50 88 / p90 123, min 31 / max 170.

#### Winning margin

| Rank | Team | Avg margin | Avg score |
|---|---|---|---|
| 1 | Sydney | +36.3 | 113.3 |
| 2 | Fremantle | +34.2 | 102.0 |
| 3 | Geelong | +16.1 | 98.3 |
| 4 | Hawthorn | +13.4 | 97.9 |
| 5 | Adelaide | +10.0 | 90.6 |

League distribution of margins (signed, per team-game): mean ~0 by construction, std 42.1, p10 -54 / p50 0 / p90 54.

#### First-quarter score

| Rank | Team | Avg Q1 score | Avg full-game score |
|---|---|---|---|
| 1 | Sydney | 27.6 | 113.3 |
| 2 | Gold Coast | 26.3 | 95.3 |
| 3 | Hawthorn | 25.1 | 97.9 |
| 4 | Greater Western Sydney | 24.9 | 92.7 |
| 5 | Fremantle | 24.9 | 102.0 |

League distribution of Q1 scores: mean **22.0**, std 11.1, p10 9 / p50 21 / p90 38.

### Going deeper with this repo's models

For the stats above, three artefacts in this repo will help you form your own view rather than just reading a leaderboard:

1. The **disposal prediction model** (`prediction.py` / `prediction_cpu.py`) forecasts a player's next-round disposal count using rolling form, opponent context and venue effects. Run it with `--player surname_first --rounds 1` to see how uncertainty is quantified for any of the leaders shown above.
2. The **backtest framework** (`backtest.py`) replays a season round-by-round so you can see how the model performed on real, out-of-sample games — the honest way to judge whether a leaderboard ranking will continue to hold.
3. The **Brownlow proxy section** above is the same per-game stat structure used here, weighted into a single composite. If you want a quick "who's having the best year overall" answer rather than per-stat leaders, that table is the one to look at.
<!-- 2026-STAT-LEADERS-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Brownlow predictor](afl-brownlow-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
