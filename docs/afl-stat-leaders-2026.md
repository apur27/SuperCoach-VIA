# 2026 player performance stats - what to look for and what the data says

> [← Back to 2026 season](afl-season-2026.md) | [← Back to main README](../README.md)

*This file is auto-updated by `update_team_analysis.py` / `refresh_readme.py` on every data refresh.*

<!-- 2026-STAT-LEADERS-START -->
This section is a guide to the AFL performance statistics that fans, analysts and SuperCoach players track most closely — what each stat measures, who is leading it in 2026, what the league-wide distribution looks like, and which other stats most reliably predict it. All numbers are computed live from `data/player_data/` for 2026 (rounds 1-16, **561 eligible players** with >=3 games, **5777 player-games** included). Correlations are Pearson r on the per-game frame; with several thousand player-games, p-values are universally tiny — read the magnitude of r, not the significance star.

![2026 AFL statistical leaders](../assets/charts/player_stat_leaders_2026.png)

### Disposal-based stats — volume and quality of ball use

#### Disposals per game

**What it measures.** Total kicks plus handballs in a game — the single broadest measure of how often a player has the ball. **Why it matters.** It is the headline SuperCoach scoring stat and the prediction target this repo's main model is built around. Volume midfielders and rebounding defenders dominate this leaderboard.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 35.4 |
| 2 | Bailey Smith | Geelong | 32.3 |
| 3 | Clayton Oliver | Greater Western Sydney | 31.5 |
| 4 | Harry Sheezel | North Melbourne | 30.5 |
| 5 | Lachie Neale | Brisbane Lions | 30.4 |

League distribution (eligible players, season-to-date): mean **15.23**, std 5.74, p10 8.57 / p50 14.43 / p90 23.50, max 35.38.

Top per-game correlates: `effective_disposals` (r = +0.97 *(mechanically related)*), `uncontested_possessions` (r = +0.88), `kicks` (r = +0.83).

#### Kicks per game

**What it measures.** Just the kicked disposals. **Why it matters.** Kicks tend to come from outside-midfielders, half-backs and tall rebounders — players who clear the ball by foot rather than shovel it into a contest. A player who kicks much more than they handball is usually playing a distributor / launch role.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nasiah Wanganeen-Milera | St Kilda | 21.3 |
| 2 | Archie Roberts | Essendon | 21.0 |
| 3 | Lachie Ash | Greater Western Sydney | 19.7 |
| 4 | Bailey Smith | Geelong | 19.6 |
| 5 | Dayne Zorko | Brisbane Lions | 19.4 |

League distribution (eligible players, season-to-date): mean **8.81**, std 3.61, p10 4.57 / p50 8.33 / p90 13.64, max 21.27.

Top per-game correlates: `disposals` (r = +0.83), `effective_disposals` (r = +0.81), `uncontested_possessions` (r = +0.78).

#### Handballs per game

**What it measures.** The hand-passed half of disposals. **Why it matters.** Handball volume tracks contest involvement — a player wins the ball at a stoppage, then handballs out to a runner. Inside-mids and clearance specialists tend to lead this stat.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Clayton Oliver | Greater Western Sydney | 20.6 |
| 2 | Sam Walsh | Carlton | 16.5 |
| 3 | Patrick Cripps | Carlton | 16.5 |
| 4 | Nick Daicos | Collingwood | 16.2 |
| 5 | Ryley Sanders | Western Bulldogs | 16.1 |

League distribution (eligible players, season-to-date): mean **6.43**, std 3.12, p10 3.00 / p50 5.93 / p90 10.69, max 20.57.

Top per-game correlates: `disposals` (r = +0.78), `effective_disposals` (r = +0.75), `contested_possessions` (r = +0.66).

#### Effective disposals per game (disposals − clangers)

**What it measures.** Disposals that did not result in a clanger, computed here as `max(disposals - clangers, 0)` because the raw data does not carry a true effective-disposal column. **Why it matters.** It is a defensible proxy for disposal *quality* — high-volume ball-users who don't turn it over. The same proxy is used in the Brownlow predictor on this page.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 30.0 |
| 2 | Bailey Smith | Geelong | 28.3 |
| 3 | Lachie Neale | Brisbane Lions | 27.5 |
| 4 | Clayton Oliver | Greater Western Sydney | 27.1 |
| 5 | Harry Sheezel | North Melbourne | 27.0 |

League distribution (eligible players, season-to-date): mean **12.84**, std 5.36, p10 6.36 / p50 12.12 / p90 20.82, max 30.00.

Top per-game correlates: `disposals` (r = +0.97 *(mechanically related)*), `uncontested_possessions` (r = +0.87), `kicks` (r = +0.81).

### Scoring stats — goals, behinds and conversion

#### Goals per game

**What it measures.** Goals kicked. **Why it matters.** Forwards live and die by this stat. It is volatile game-to-game (a single missed shot can halve your score), so multi-game averages and shot-source context (marks-inside-50, contested marks) matter more than any one game.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Gunston | Hawthorn | 3.89 |
| 2 | Ben King | Gold Coast | 3.00 |
| 3 | Jeremy Cameron | Geelong | 2.79 |
| 4 | Charlie Curnow | Sydney | 2.77 |
| 5 | Logan Morris | Brisbane Lions | 2.69 |

League distribution (eligible players, season-to-date): mean **0.53**, std 0.60, p10 0.00 / p50 0.33 / p90 1.40, max 3.89.

Top per-game correlates: `marks_inside_50` (r = +0.67), `behinds` (r = +0.33), `rebound_50s` (r = -0.30).

**Goal conversion rate.** Defined as `goals / (goals + behinds)`, season-to-date, for players with >=2 goals total. League distribution (n=357): mean **59.7%**, std 16.4pp, p10 40% / p50 60% / p90 80%.

| Rank | Player | Team | G | B | Conversion |
|---|---|---|---|---|---|
| 1 | Jake Melksham | Melbourne | 6 | 0 | 100.0% |
| 2 | Isaac Cumming | Adelaide | 5 | 0 | 100.0% |
| 3 | Dan Butler | St Kilda | 4 | 0 | 100.0% |
| 4 | Noah Roberts-Thomson | Richmond | 4 | 0 | 100.0% |
| 5 | Oscar Steene | Collingwood | 4 | 0 | 100.0% |

#### Behinds per game

**What it measures.** Minor scores — shots that hit the post or go through the smaller posts. **Why it matters.** Rarely predicted alone — it is too noisy. Best read alongside goals to compute **conversion rate** (`goals / (goals + behinds)`), the cleanest available signal of forward accuracy.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jake Waterman | West Coast | 2.64 |
| 2 | Mitch Georgiades | Port Adelaide | 2.43 |
| 3 | Jack Gunston | Hawthorn | 2.33 |
| 4 | Logan Morris | Brisbane Lions | 2.15 |
| 5 | Jeremy Cameron | Geelong | 2.07 |

League distribution (eligible players, season-to-date): mean **0.40**, std 0.43, p10 0.00 / p50 0.29 / p90 0.93, max 2.64.

Top per-game correlates: `marks_inside_50` (r = +0.55), `goals` (r = +0.33), `rebound_50s` (r = -0.25).

### Contested and ground-ball stats — the inside game

#### Contested possessions per game

**What it measures.** Wins of the ball under physical pressure — ground-balls, taps, and contested marks. **Why it matters.** This is the cleanest stat for separating a midfielder's *contest* role from an outside ball-user's *spread* role. It correlates strongly with clearances and tackles.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Clayton Oliver | Greater Western Sydney | 15.29 |
| 2 | Tristan Xerri | North Melbourne | 14.82 |
| 3 | Patrick Cripps | Carlton | 14.50 |
| 4 | Brodie Grundy | Sydney | 14.08 |
| 5 | Max Gawn | Melbourne | 13.73 |

League distribution (eligible players, season-to-date): mean **5.35**, std 2.36, p10 3.00 / p50 4.67 / p90 8.79, max 15.29.

Top per-game correlates: `clearances` (r = +0.74), `handballs` (r = +0.66), `disposals` (r = +0.58).

#### Clearances per game

**What it measures.** Disposals that move the ball clear of a stoppage (a centre-bounce or boundary throw-in). **Why it matters.** Stoppage dominance is one of the few team-level wins a midfield can manufacture. Top clearance players are almost always the inside-mid fulcrums of their team.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Clayton Oliver | Greater Western Sydney | 8.07 |
| 2 | Jai Newcombe | Hawthorn | 7.86 |
| 3 | Lachie Neale | Brisbane Lions | 7.43 |
| 4 | Patrick Cripps | Carlton | 6.93 |
| 5 | Tristan Xerri | North Melbourne | 6.91 |

League distribution (eligible players, season-to-date): mean **1.41**, std 1.63, p10 0.08 / p50 0.79 / p90 4.07, max 8.07.

Top per-game correlates: `contested_possessions` (r = +0.74), `handballs` (r = +0.56), `disposals` (r = +0.49).

#### Tackles per game

**What it measures.** Pressure acts that physically stop a ball-carrier. **Why it matters.** Defensive midfield work — the unsung currency of forward-half pressure and turnover football. It correlates with clearances (you tackle the same opponent you compete against) but tells a different story.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Sam Berry | Adelaide | 7.50 |
| 2 | Jack Graham | West Coast | 6.60 |
| 3 | James Rowbottom | Sydney | 6.57 |
| 4 | Tom Atkins | Geelong | 6.40 |
| 5 | Tristan Xerri | North Melbourne | 6.36 |

League distribution (eligible players, season-to-date): mean **2.36**, std 1.25, p10 1.00 / p50 2.14 / p90 4.10, max 7.50.

Top per-game correlates: `clearances` (r = +0.39), `contested_possessions` (r = +0.36), `handballs` (r = +0.29).

#### Hit-outs per game (ruckmen only)

**What it measures.** Wins by a ruckman at a ruck contest (the tap from a centre bounce or stoppage). **Why it matters.** Ruckman-only stat — the distribution is bimodal: ~1 player per team registers double-digits, everyone else is 0. Always read this leaderboard as "top ruckmen", not "top players".

**Bimodal distribution warning.** 87% of eligible 2026 players average less than 1 hit-out per game — they are not ruckmen. The league mean below is dragged down by all the zeros; the meaningful comparison is between ruckmen, where the top of the distribution sits in the 25-35 range.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Brodie Grundy | Sydney | 34.3 |
| 2 | Lachlan Mcandrew | Adelaide | 32.1 |
| 3 | Max Gawn | Melbourne | 31.7 |
| 4 | Jarrod Witts | Gold Coast | 31.2 |
| 5 | Jordon Sweet | Port Adelaide | 30.6 |

League distribution (eligible players, season-to-date): mean **1.58**, std 5.34, p10 0.00 / p50 0.00 / p90 2.21, max 34.31.

Top per-game correlates: `clearances` (r = +0.26), `uncontested_possessions` (r = -0.25), `free_kicks_for` (r = +0.20).

### Territory stats — moving the ball forward

#### Inside 50s per game

**What it measures.** Disposals or carries that move the ball into the team's attacking 50m arc. **Why it matters.** Territory currency — the precondition for goals. Wing/half-forward players who launch attacks lead this stat. It correlates with kicks and disposals because most inside-50s are foot-delivered.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 6.92 |
| 2 | Bailey Smith | Geelong | 6.86 |
| 3 | Chad Warner | Sydney | 6.64 |
| 4 | Ed Richards | Western Bulldogs | 6.57 |
| 5 | Shai Bolton | Fremantle | 5.86 |

League distribution (eligible players, season-to-date): mean **2.19**, std 1.25, p10 0.75 / p50 2.00 / p90 3.86, max 6.92.

Top per-game correlates: `disposals` (r = +0.52), `effective_disposals` (r = +0.48), `kicks` (r = +0.47).

#### Marks per game

**What it measures.** Total uncontested + contested marks taken. **Why it matters.** Aerial dominance and intercept defence. Loose-half-back roles dominate the total-marks leaderboard because they sit behind the play and fly under kicks. Tall forwards lead a separate, narrower stat — marks inside 50.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Callum Wilkie | St Kilda | 9.7 |
| 2 | Aliir Aliir | Port Adelaide | 8.5 |
| 3 | Ryan Lester | Brisbane Lions | 8.3 |
| 4 | Lachie Ash | Greater Western Sydney | 8.0 |
| 5 | Blake Acres | Carlton | 7.7 |

League distribution (eligible players, season-to-date): mean **3.90**, std 1.58, p10 2.00 / p50 3.78 / p90 6.15, max 9.73.

Top per-game correlates: `kicks` (r = +0.57), `uncontested_possessions` (r = +0.53), `effective_disposals` (r = +0.43).

#### Marks inside 50 per game

**What it measures.** Marks taken inside the attacking 50m arc — i.e. marks that turn directly into shots on goal. **Why it matters.** This is the strongest single predictor of a forward's goal output. It is what separates a deep-forward role from a high-half-forward role, and the correlation with goals is the highest of any stat in this section.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Gunston | Hawthorn | 4.56 |
| 2 | Mitch Georgiades | Port Adelaide | 3.86 |
| 3 | Jye Amiss | Fremantle | 3.57 |
| 4 | Josh Treacy | Fremantle | 3.50 |
| 5 | Logan Morris | Brisbane Lions | 3.38 |

League distribution (eligible players, season-to-date): mean **0.50**, std 0.68, p10 0.00 / p50 0.27 / p90 1.56, max 4.56.

Top per-game correlates: `goals` (r = +0.67), `behinds` (r = +0.55), `contested_marks` (r = +0.33).

### Discipline stats — errors and free kicks

#### Clangers per game

**What it measures.** Errors — missed targets, fumbles, free kicks given away by the ball-carrier. **Why it matters.** Clangers are the friction term on disposal volume — a high-disposal player who also leads in clangers is being asked to play through traffic, not necessarily playing badly. The correlation with frees-against is mechanical: many clangers *are* frees-against.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Harley Reid | West Coast | 6.57 |
| 2 | Jacob Hopper | Richmond | 5.50 |
| 3 | Brodie Grundy | Sydney | 5.46 |
| 4 | Nick Daicos | Collingwood | 5.38 |
| 5 | Kysaiah Pickett | Melbourne | 5.27 |

League distribution (eligible players, season-to-date): mean **2.40**, std 0.90, p10 1.33 / p50 2.25 / p90 3.50, max 6.57.

Top per-game correlates: `free_kicks_against` (r = +0.62 *(mechanically related)*), `contested_possessions` (r = +0.35), `disposals` (r = +0.32).

#### Free kicks for per game

**What it measures.** Free kicks paid to the player. **Why it matters.** A weak isolated signal — frees-for tracks contest involvement (rucks especially) more than skill. Best used as a tiebreaker rather than a standalone metric.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Tristan Xerri | North Melbourne | 3.64 |
| 2 | Max Gawn | Melbourne | 2.60 |
| 3 | Jai Newcombe | Hawthorn | 2.57 |
| 4 | Darcy Cameron | Collingwood | 2.38 |
| 5 | Sam Darcy | Western Bulldogs | 2.33 |

League distribution (eligible players, season-to-date): mean **0.80**, std 0.46, p10 0.33 / p50 0.73 / p90 1.36, max 3.64.

Top per-game correlates: `contested_possessions` (r = +0.42), `clearances` (r = +0.31), `tackles` (r = +0.21).

#### Free kicks against per game

**What it measures.** Free kicks paid against the player. **Why it matters.** Discipline / aggression marker, with the caveat that ruck contest infringements inflate the number for ruckmen. Reads like a clanger when it correlates with them.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Graham | West Coast | 3.00 |
| 2 | Brodie Grundy | Sydney | 2.77 |
| 3 | Harley Reid | West Coast | 2.71 |
| 4 | Patrick Cripps | Carlton | 2.50 |
| 5 | Nick Bryan | Essendon | 2.17 |

League distribution (eligible players, season-to-date): mean **0.82**, std 0.45, p10 0.33 / p50 0.75 / p90 1.40, max 3.00.

Top per-game correlates: `clangers` (r = +0.62 *(mechanically related)*), `clearances` (r = +0.17), `contested_possessions` (r = +0.16).

### Team-level stats — what the scoreboard says

Team-level stats use `data/matches/matches_2026.csv` rather than per-player aggregates. Total team score is `goals × 6 + behinds`; margin is the team's score minus the opponent's. A first-quarter score is a useful early-momentum signal — strong starters tend to keep the lead.

#### Total team score per game

| Rank | Team | Avg score | Avg margin | Avg Q1 |
|---|---|---|---|---|
| 1 | Sydney | 113.3 | +36.3 | 27.6 |
| 2 | Brisbane Lions | 101.9 | +7.7 | 22.3 |
| 3 | Fremantle | 101.8 | +32.2 | 24.7 |
| 4 | Hawthorn | 99.1 | +13.6 | 26.3 |
| 5 | Geelong | 97.7 | +14.3 | 25.3 |

League distribution of per-game team scores: mean **88.9**, std 25.0, p10 60 / p50 88 / p90 122, min 31 / max 170.

#### Winning margin

| Rank | Team | Avg margin | Avg score |
|---|---|---|---|
| 1 | Sydney | +36.3 | 113.3 |
| 2 | Fremantle | +32.2 | 101.8 |
| 3 | Geelong | +14.3 | 97.7 |
| 4 | Hawthorn | +13.6 | 99.1 |
| 5 | Adelaide | +10.5 | 89.8 |

League distribution of margins (signed, per team-game): mean ~0 by construction, std 41.2, p10 -52 / p50 0 / p90 52.

#### First-quarter score

| Rank | Team | Avg Q1 score | Avg full-game score |
|---|---|---|---|
| 1 | Sydney | 27.6 | 113.3 |
| 2 | Hawthorn | 26.3 | 99.1 |
| 3 | Gold Coast | 25.8 | 95.5 |
| 4 | Geelong | 25.3 | 97.7 |
| 5 | Fremantle | 24.7 | 101.8 |

League distribution of Q1 scores: mean **21.9**, std 11.0, p10 8 / p50 21 / p90 38.

### Going deeper with this repo's models

For the stats above, three artefacts in this repo will help you form your own view rather than just reading a leaderboard:

1. The **disposal prediction model** (`prediction.py` / `prediction_cpu.py`) forecasts a player's next-round disposal count using rolling form, opponent context and venue effects. Run it with `--player surname_first --rounds 1` to see how uncertainty is quantified for any of the leaders shown above.
2. The **backtest framework** (`backtest.py`) replays a season round-by-round so you can see how the model performed on real, out-of-sample games — the honest way to judge whether a leaderboard ranking will continue to hold.
3. The **Brownlow proxy section** above is the same per-game stat structure used here, weighted into a single composite. If you want a quick "who's having the best year overall" answer rather than per-stat leaders, that table is the one to look at.
<!-- 2026-STAT-LEADERS-END -->

---
**Related:** [Team analysis](afl-team-analysis-2026.md) · [Finals pathway](afl-finals-2026.md) · [Brownlow predictor](afl-brownlow-2026.md) · [Predictions](afl-predictions-2026.md) · [Backtest](afl-backtest-2026.md)
