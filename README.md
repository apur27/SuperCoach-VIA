# AFL SuperCoach VIA
![AFL SuperCoach VIA Banner](/assets/banner.svg)
<div align="center">
  <img src="https://img.shields.io/github/last-commit/apur27/SuperCoach-VIA">
  <img src="https://img.shields.io/github/contributors/apur27/SuperCoach-VIA">
  <img src="https://img.shields.io/github/stars/apur27/SuperCoach-VIA?style=social">
  <img src="https://img.shields.io/github/forks/apur27/SuperCoach-VIA?style=social">
</div>

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Data](https://img.shields.io/badge/data-2026%20season%20round%208-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

⭐ **If this helped your SuperCoach team this week, please star the repo** — it helps other footy fans find it.

> **⚡ TL;DR — I just want this week's predictions (no reading required)**
>
> 1. Open Terminal → paste these 4 lines one by one:
>    ```bash
>    git clone https://github.com/apur27/SuperCoach-VIA.git
>    cd SuperCoach-VIA
>    pip install -r requirements.txt
>    python prediction_cpu.py   # works on any laptop — or prediction.py if you have a GPU
>    ```
> 2. Open the file `data/prediction/next_round_*_prediction_*.csv` in Excel or Google Sheets.
> 3. Done — you now have predicted disposals for every player this week.
>
> Takes roughly 5–10 minutes on a normal laptop. Full details below if you want them.

<br>

A personal AFL data project that does three things:
1. **Stores every AFL match and player stat** going back to 1897
2. **Ranks the greatest players of all time** using a fair, era-adjusted formula
3. **Predicts how many disposals each player will get** in the next round

## Who is this for?

| I am… | What I get from this repo |
|---|---|
| **A footy fan** who wants to understand their team better | The AFL insights section — live team stats, finals pathway, who's playing well |
| **A SuperCoach player** wanting a data edge | The disposal prediction model tells you who is likely to rack up this week |
| **Curious about AI** and want to see it applied to sport | The Claude AI agents section walks you through using AI to ask questions about the data |
| **A developer or data scientist** | Full pipeline docs, model code, backtest framework, GPU setup |
| **A casual SuperCoach player** who just wants better picks each week | The weekly predicted disposals CSV — tells you who is likely to rack up this round |

You don't need to write any code to use most of this. The AFL Insights section of this README updates automatically every week — just read it.

> **What is SuperCoach?** SuperCoach is Australia's most popular AFL fantasy competition — you pick a squad of real AFL players and score points based on their actual in-game statistics each week. Disposals (kicks + handballs) are the biggest scoring category, which is why this project focuses on predicting them. Think of it as the AFL version of fantasy football.

## Table of Contents

- [Quick start & setup](docs/quick-start.md)
- [AFL insights](#afl-insights) — live 2026 data, updated weekly
- [AFL Hall of Fame](docs/hall-of-fame.md)
- [Prediction model](docs/prediction-model.md)
- [Using AI agents](docs/ai-agents.md)
- [Technical reference](docs/technical-reference.md)
- [Roadmap & contributing](docs/roadmap.md)

## AFL insights

**Who is this section for?** Footy fans, coaches, and analysts who care more about what the data says than how the predictions work. Long-run trends across 125 years of AFL, current-season team profiles, the all-time greatest player debate, and a coach's-eye game-plan workflow.

This section is for footy fans. No coding required — just read.

### How the game has changed — 125 years of data

This repo holds **687,920 player-game records** stretching from 1897 to 2026 and **16,900 matches**. That's a long enough run to see how AFL has actually evolved — not just how it feels different, but how it's measurably different.

The numbers below come from running `era_based_statistical_analysis.py` over every player and every match. Era buckets are deliberate: stats coverage changes with the eras. **Disposals, marks, kicks and handballs were not tracked at all before 1965. Tackles weren't reliably recorded until 1987. Clearances, contested possessions and inside-50s only start appearing in 1998.** Anywhere the cell says "n/a" below, it's because the AFL didn't track that stat yet — not because the value was zero.

#### Scoring across the eras

![AFL team scoring 1965 to 2025](assets/charts/era_scoring_trends.png)

<!-- SCORING-DECADE-CHART-START -->
![Average team score by decade, 1900s to 2020s](assets/charts/scoring_by_decade.png)
<!-- SCORING-DECADE-CHART-END -->

| Per team, per game | pre-1965 | 1965–1990 | 1991–2010 | 2011–now |
|---|---:|---:|---:|---:|
| Goals | **10.1** | 13.9 | 13.9 | 12.3 |
| Behinds | 12.0 | 13.6 | 12.2 | **11.0** |
| Total points | 72.5 | **96.8** | 95.7 | 84.9 |
| Scoring shots | 22.1 | **27.5** | 26.1 | 23.3 |
| Goal accuracy | 45.2% | 50.1% | **53.0%** | 52.7% |
| Match total points | 145 | 193 | 191 | **170** |

A few things jump out that contradict the usual barbershop wisdom:

- **The high-scoring era was the 1980s, not now.** A team-game in 1965–1990 averaged 96.75 points — modern teams average 84.88. That's a 12% drop in per-team scoring from the Hudson/Lockett era to the Daicos era. Total match scoring is down 23 points per game from the peak.
- **Modern players are NOT more accurate in front of goal in any meaningful way.** Goal accuracy peaked in 1991–2010 at 53.0% and has actually slipped slightly to 52.7% in the modern era. The big jump was much earlier — pre-1965 footballers converted just 45.2% of their scoring shots, which lines up with rough grounds, leather balls that swelled in the rain, and longer drop kicks rather than today's set-shot routines.
- **Modern teams have fewer scoring shots, not just less accuracy.** Scoring shots per team-game: 27.5 (1965–1990) → 26.1 → 23.3. That's 4 fewer scoring shots per team per game vs the 80s.

#### Player workload — the most dramatic change in footy

![Per-player stat evolution by era](assets/charts/era_stat_evolution.png)

<!-- TACKLES-CLEARANCES-CHART-START -->
![Tackles and clearances per player per game over time](assets/charts/era_tackles_clearances.png)
<!-- TACKLES-CLEARANCES-CHART-END -->

| Per player, per game | pre-1965 | 1965–1990 | 1991–2010 | 2011–now |
|---|---:|---:|---:|---:|
| Kicks | n/a | 11.08 | 9.15 | 9.41 |
| Handballs | n/a | 4.31 | 5.85 | **6.95** |
| Disposals | n/a | 14.89 | 14.72 | **16.21** |
| Marks | n/a | 3.76 | 4.11 | 4.22 |
| Tackles | n/a | 1.86 | 2.42 | **3.20** |
| Clearances | n/a | n/a | 2.54 | 2.87 |
| Contested possessions | n/a | n/a | 5.65 | 6.27 |

Read the kicks/handballs row carefully — this is the single biggest shift in how footy is played:

- **Handballs are up 61%** since 1965–1990 (4.31/g → 6.95/g). **Kicks are down 15%** in the same window (11.08/g → 9.41/g). The modern game is a handball game wearing a kicking game's uniform. Players today move the ball more often, but they move it shorter and faster.
- **Tackles are up 72%** from the 1965–1990 era (1.86/g → 3.20/g). Even just inside the modern stat era, tackles per player jumped 32% from 2.42 (1991–2010) to 3.20 (2011–now). Pressure is the defining tactical innovation of the post-2010 game, and the data is unambiguous about it.
- **Disposals per player are up about 9%** from the 80s (14.89 → 16.21). Combined with smaller rosters and more interchanges, that means individual ball-winners are doing more.
- **Contested possessions are up 11%** since 1991–2010 (5.65 → 6.27). The contested-ball revolution that coaches talk about is real and measurable.
- **Cohen's d for the tackles shift between 1991–2010 and the modern era is +0.41** — a moderate effect size in statistical terms. For context, that's a bigger jump than nearly every other player metric across any era boundary.

#### What the numbers don't tell us

A few honest caveats — because nothing about the past 125 years is a controlled experiment.

- **Hit-outs jumped from ~10 per ruckman/game in 2016 to ~17 in 2017** and stayed there. That's almost certainly a recording-method change at AFL Stats, not a sudden ruck revolution. Treat any hit-out comparison crossing 2016/2017 with a grain of salt.
- **Tackles in 1965–1990** look low (1.86/g) partly because the AFL didn't actually start counting tackles until 1987. The mean is dragged down by 22 years of missing data inside that bucket. The real "old footy" baseline is roughly the 1987–1990 sub-window.
- **2026 is partial** — only the first 8 rounds — so it's included in the 2011–present bucket but doesn't change the pattern materially.
- **Statistical tests are reported with a "rough indicator" caveat.** The 687k player-game rows aren't independent (same players appear hundreds of times), so the p-values from the Welch tests in `data/era_significance_tests.csv` should be read alongside the effect sizes (Cohen's d), which are more honest about practical significance.

The full era-by-era breakdown lives in `data/era_stats.csv`, the matches-level scoring numbers in `data/era_team_scoring.csv`, and the year-by-year trends in `data/era_yearly_trends.csv` — all rebuilt by running `era_based_statistical_analysis.py`.

### 2026 season — live team analysis

<!-- 2026-TEAM-ANALYSIS-START -->
Through 8 rounds of the 2026 season, the league averages out at around 367 disposals, 215 kicks, 152 handballs, 56 tackles, 35 clearances and 13 goals per team per game. **Greater Western Sydney** lead the comp for total disposals (399.3/g), with **Brisbane Lions** winning the most clearances (42.1/g) and **Sydney** the most physical side (64.0 tackles/g). **Sydney** are the highest-scoring team at 17.3 goals/g, while **Richmond** prop up the table on both fronts — only 326.1 disposals/g and 8.4 goals/g. At the individual level, **Nick Daicos** (Collingwood) is the leading ball-winner at 37.0 disposals/game. Caveat: this is a small sample — single-season form can swing by round, so treat the rankings below as a snapshot rather than a settled hierarchy.

#### All 18 teams ranked by total disposals — 2026 season-to-date

| # | Team | G | Disp/g | Kicks | Handballs | Marks | Goals | Tackles | Clearances | I50s | CP | Form tag |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | Greater Western Sydney | 7 | 399.3 | 217.7 | 181.6 | 95.0 | 12.3 | 54.3 | 33.3 | 57.4 | 134.4 | ball-winners, territory team |
| 2 | Sydney | 7 | 392.4 | 212.9 | 179.6 | 82.7 | 17.3 | 64.0 | 37.0 | 64.6 | 138.9 | ball-winners, contested-ball team |
| 3 | Collingwood | 7 | 387.1 | 224.0 | 163.1 | 113.0 | 11.9 | 55.4 | 28.9 | 53.6 | 118.7 | ball-winners |
| 4 | Hawthorn | 7 | 383.9 | 230.1 | 153.7 | 106.9 | 15.1 | 59.0 | 36.9 | 55.6 | 129.6 | scoring threat |
| 5 | North Melbourne | 7 | 380.7 | 221.7 | 159.0 | 101.0 | 14.4 | 58.0 | 37.1 | 51.4 | 128.6 | clearance machine |
| 6 | St Kilda | 7 | 380.3 | 232.7 | 147.6 | 101.4 | 13.6 | 53.0 | 35.7 | 52.9 | 130.3 | mid-pack |
| 7 | Fremantle | 7 | 377.7 | 212.9 | 164.9 | 88.1 | 13.1 | 59.9 | 35.9 | 52.9 | 138.6 | contested-ball team, physical unit |
| 8 | Brisbane Lions | 7 | 377.3 | 247.9 | 129.4 | 117.0 | 14.9 | 46.1 | 42.1 | 57.0 | 131.7 | clearance machine, low pressure |
| 9 | Geelong | 7 | 371.9 | 225.1 | 146.7 | 96.4 | 13.6 | 63.9 | 34.9 | 57.1 | 132.4 | physical unit |
| 10 | Western Bulldogs | 7 | 368.6 | 209.1 | 159.4 | 86.1 | 12.9 | 57.9 | 38.4 | 51.0 | 126.9 | clearance machine |
| 11 | Gold Coast | 7 | 368.0 | 196.0 | 172.0 | 82.0 | 15.3 | 56.1 | 33.4 | 59.0 | 129.6 | scoring threat, territory team |
| 12 | Carlton | 7 | 365.3 | 208.9 | 156.4 | 83.7 | 11.6 | 59.0 | 36.0 | 49.9 | 138.3 | contested-ball team, struggling in front of goal |
| 13 | Essendon | 7 | 359.4 | 202.7 | 156.7 | 92.4 | 12.0 | 49.7 | 29.1 | 44.9 | 116.7 | low pressure |
| 14 | Port Adelaide | 7 | 349.9 | 224.3 | 125.6 | 108.0 | 12.9 | 55.9 | 33.9 | 52.7 | 117.7 | mid-pack |
| 15 | Melbourne | 7 | 346.6 | 209.6 | 137.0 | 85.9 | 14.9 | 55.1 | 35.0 | 57.4 | 128.0 | territory team |
| 16 | Adelaide | 7 | 345.4 | 218.1 | 127.3 | 86.9 | 12.7 | 57.0 | 30.7 | 48.4 | 128.4 | starved of the ball, strong rebound |
| 17 | West Coast | 7 | 326.7 | 194.6 | 132.1 | 82.6 | 9.3 | 52.9 | 32.7 | 46.6 | 123.0 | starved of the ball, struggling in front of goal |
| 18 | Richmond | 7 | 326.1 | 187.0 | 139.1 | 80.9 | 8.4 | 52.1 | 32.0 | 48.1 | 119.4 | starved of the ball, low pressure |

#### League leaders by stat category — 2026

| Stat | League leader | Their avg/g | League avg/g | Delta vs league |
| :--- | :--- | ---: | ---: | ---: |
| Disposals | Greater Western Sydney | 399.3 | 367.0 | +32.3 |
| Kicks | Brisbane Lions | 247.9 | 215.3 | +32.6 |
| Handballs | Greater Western Sydney | 181.6 | 151.7 | +29.9 |
| Marks | Brisbane Lions | 117.0 | 93.9 | +23.1 |
| Goals | Sydney | 17.3 | 13.1 | +4.2 |
| Tackles | Sydney | 64.0 | 56.1 | +7.9 |
| Clearances | Brisbane Lions | 42.1 | 34.6 | +7.5 |
| Inside-50s | Sydney | 64.6 | 53.4 | +11.2 |
| Contested possessions | Sydney | 138.9 | 128.4 | +10.5 |
| Rebound-50s | Adelaide | 43.9 | 39.4 | +4.5 |
| Hit-outs | Hawthorn | 46.7 | 32.6 | +14.1 |
| Contested marks | Hawthorn | 10.7 | 8.8 | +1.9 |

<!-- GOALS-DISPOSALS-CHART-START -->
![2026 season — goals vs disposals scatter](assets/charts/team_2026_goals_disposals.png)
<!-- GOALS-DISPOSALS-CHART-END -->

#### Visual snapshot — top-6 radar and league-wide rank heatmap

The radar chart below picks out the top six disposal teams of the 2026 season-to-date and plots them against the six core dimensions, normalised to a 0-1 scale relative to all 18 sides — a value of 1.0 on an axis means that team is the league best on that stat. The heatmap underneath shows every team's rank across eight key stats (1 = league best in green, 18 = worst in red), with rows ordered by disposals rank.

![Top 6 teams radar — 2026](assets/charts/team_2026_radar.png)

![Team rank heatmap — 2026](assets/charts/team_2026_heatmap.png)

<!-- FORM-TREND-CHART-START -->
![2026 disposal form trend by round](assets/charts/team_form_trend_2026.png)
<!-- FORM-TREND-CHART-END -->

#### Team-by-team — playing style, strengths and weaknesses

Every paragraph below is generated from the team's actual 2026 stat profile (averages and league ranks across 16 stat categories). The strategy descriptions are derived from rank combinations, not hand-written takes — so they update automatically as the season progresses. Rank 1 = best, rank 18 = worst (for clangers and free kicks against, lower raw values are better, so rank 1 still means the desirable end).

#### Greater Western Sydney

After 7 games of 2026, the Greater Western Sydney sit near the top of the league for ball use (1st for disposals) and average 399.3 disposals, 12.3 goals and 54.3 tackles per game. Their stat profile reads as a contest-heavy approach at the stoppages; a high-tempo handball game (45% of disposals by hand); pumping the ball forward with high volume but little control inside 50. They go inside 50 57.4 times a game (3rd in the league), win 134.4 contested possessions (4th) and tilt 45/55 between handball and kick. The strengths jump out clearly: they're 1st for disposals (399.3/g), 1st for handballs (181.6/g), 1st for uncontested possessions (256.1/g). The flip side is harder to ignore: 17th for clangers (61.0/g), 16th for free kicks against (20.9/g), 15th for free kicks for (16.7/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Clayton Oliver**, leading the team in disposals at 30.9/game.

#### Sydney

After 7 games of 2026, the Sydney sit near the top of the league for ball use (2nd for disposals) and average 392.4 disposals, 17.3 goals and 64.0 tackles per game. Their stat profile reads as a contested-ball, clearance-first identity; a high-tempo handball game (46% of disposals by hand); dominating territory and connecting cleanly inside 50. They go inside 50 64.6 times a game (1st in the league), win 138.9 contested possessions (1st) and tilt 46/54 between handball and kick. The strengths jump out clearly: they're 1st for goals (17.3/g), 1st for tackles (64.0/g), 1st for inside-50s (64.6/g). The flip side is harder to ignore: 18th for clangers (65.6/g), 15th for marks (82.7/g), 13th for free kicks against (20.6/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Justin Mcinerney**, leading the team in disposals at 25.6/game.

#### Collingwood

After 7 games of 2026, the Collingwood sit near the top of the league for ball use (3rd for disposals) and average 387.1 disposals, 11.9 goals and 55.4 tackles per game. Their stat profile reads as a possession-and-control style that shies away from the contest. They go inside 50 53.6 times a game (8th in the league), win 118.7 contested possessions (16th) and tilt 42/58 between handball and kick. The strengths jump out clearly: they're 2nd for marks (113.0/g), 2nd for uncontested possessions (254.1/g), 2nd for clangers (53.6/g). The flip side is harder to ignore: 18th for clearances (28.9/g), 16th for contested possessions (118.7/g), 15th for goals (11.9/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Nick Daicos**, leading the team in disposals at 37.0/game.

#### Hawthorn

After 7 games of 2026, the Hawthorn sit near the top of the league for ball use (4th for disposals) and average 383.9 disposals, 15.1 goals and 59.0 tackles per game. Their stat profile reads as a possession-based, uncontested-game template; backed up by elite forward and ground-ball pressure. They go inside 50 55.6 times a game (7th in the league), win 129.6 contested possessions (8th) and tilt 40/60 between handball and kick. The strengths jump out clearly: they're 1st for hit-outs (46.7/g), 1st for contested marks (10.7/g), 3rd for kicks (230.1/g). The flip side is harder to ignore: 11th for rebound-50s (39.6/g), 11th for free kicks for (17.4/g), 10th for handballs (153.7/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Jai Newcombe**, leading the team in disposals at 26.4/game.

#### North Melbourne

After 7 games of 2026, the North Melbourne are tracking inside the top half (5th for disposals) and average 380.7 disposals, 14.4 goals and 58.0 tackles per game. Their stat profile reads as a possession-based, uncontested-game template; backed up by elite forward and ground-ball pressure. They go inside 50 51.4 times a game (12th in the league), win 128.6 contested possessions (10th) and tilt 42/58 between handball and kick. The strengths jump out clearly: they're 3rd for clearances (37.1/g), 3rd for uncontested possessions (243.9/g), 3rd for free kicks for (20.4/g). The flip side is harder to ignore: 17th for free kicks against (21.4/g), 13th for hit-outs (28.7/g), 12th for inside-50s (51.4/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Harry Sheezel**, leading the team in disposals at 32.7/game.

#### St Kilda

After 7 games of 2026, the St Kilda are tracking inside the top half (6th for disposals) and average 380.3 disposals, 13.6 goals and 53.0 tackles per game. Their stat profile reads as a possession-based, uncontested-game template; with notably soft pressure around the ball. They go inside 50 52.9 times a game (9th in the league), win 130.3 contested possessions (7th) and tilt 39/61 between handball and kick. The strengths jump out clearly: they're 2nd for kicks (232.7/g), 2nd for free kicks against (15.7/g), 4th for free kicks for (20.1/g). The flip side is harder to ignore: 14th for tackles (53.0/g), 11th for handballs (147.6/g), 10th for hit-outs (31.9/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Jack Sinclair**, leading the team in disposals at 30.1/game.

#### Fremantle

After 7 games of 2026, the Fremantle are tracking inside the top half (7th for disposals) and average 377.7 disposals, 13.1 goals and 59.9 tackles per game. Their stat profile reads as a contest-heavy approach at the stoppages; backed up by elite forward and ground-ball pressure; frequently absorbing entries and rebounding from defence. They go inside 50 52.9 times a game (9th in the league), win 138.6 contested possessions (2nd) and tilt 44/56 between handball and kick. The strengths jump out clearly: they're 2nd for contested possessions (138.6/g), 3rd for tackles (59.9/g), 3rd for rebound-50s (41.9/g). The flip side is harder to ignore: 18th for free kicks for (16.1/g), 10th for kicks (212.9/g), 10th for marks (88.1/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Caleb Serong**, leading the team in disposals at 25.7/game.

#### Brisbane Lions

After 7 games of 2026, the Brisbane Lions are tracking inside the top half (8th for disposals) and average 377.3 disposals, 14.9 goals and 46.1 tackles per game. Their stat profile reads as a contested-ball, clearance-first identity; a kick-first ball movement (66% by foot); dominating territory and connecting cleanly inside 50. They go inside 50 57.0 times a game (6th in the league), win 131.7 contested possessions (6th) and tilt 34/66 between handball and kick. The strengths jump out clearly: they're 1st for kicks (247.9/g), 1st for marks (117.0/g), 1st for clearances (42.1/g). The flip side is harder to ignore: 18th for tackles (46.1/g), 16th for handballs (129.4/g), 16th for rebound-50s (36.6/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Lachie Neale**, leading the team in disposals at 31.1/game.

#### Geelong

After 7 games of 2026, the Geelong are floating around the middle of the pack (9th for disposals) and average 371.9 disposals, 13.6 goals and 63.9 tackles per game. Their stat profile reads as a contest-heavy approach at the stoppages; backed up by elite forward and ground-ball pressure. They go inside 50 57.1 times a game (5th in the league), win 132.4 contested possessions (5th) and tilt 39/61 between handball and kick. The strengths jump out clearly: they're 2nd for tackles (63.9/g), 2nd for free kicks for (23.4/g), 4th for kicks (225.1/g). The flip side is harder to ignore: 13th for rebound-50s (37.6/g), 13th for clangers (58.6/g), 12th for handballs (146.7/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Bailey Smith**, leading the team in disposals at 32.0/game.

#### Western Bulldogs

After 7 games of 2026, the Western Bulldogs are floating around the middle of the pack (10th for disposals) and average 368.6 disposals, 12.9 goals and 57.9 tackles per game. Their stat profile reads as struggling to win territory and force the issue forward; frequently absorbing entries and rebounding from defence. They go inside 50 51.0 times a game (13th in the league), win 126.9 contested possessions (13th) and tilt 43/57 between handball and kick. The strengths jump out clearly: they're 2nd for clearances (38.4/g), 6th for handballs (159.4/g), 6th for rebound-50s (41.3/g). The flip side is harder to ignore: 18th for contested marks (6.1/g), 13th for kicks (209.1/g), 13th for inside-50s (51.0/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Marcus Bontempelli**, leading the team in disposals at 26.6/game.

#### Gold Coast

After 7 games of 2026, the Gold Coast are floating around the middle of the pack (11th for disposals) and average 368.0 disposals, 15.3 goals and 56.1 tackles per game. Their stat profile reads as a high-tempo handball game (47% of disposals by hand). They go inside 50 59.0 times a game (2nd in the league), win 129.6 contested possessions (8th) and tilt 47/53 between handball and kick. The strengths jump out clearly: they're 2nd for goals (15.3/g), 2nd for inside-50s (59.0/g), 3rd for handballs (172.0/g). The flip side is harder to ignore: 18th for free kicks against (22.0/g), 17th for marks (82.0/g), 16th for kicks (196.0/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Christian Petracca**, leading the team in disposals at 26.0/game.

#### Carlton

After 7 games of 2026, the Carlton are floating around the middle of the pack (12th for disposals) and average 365.3 disposals, 11.6 goals and 59.0 tackles per game. Their stat profile reads as a contested-ball, clearance-first identity; struggling to win territory and force the issue forward; backed up by elite forward and ground-ball pressure. They go inside 50 49.9 times a game (14th in the league), win 138.3 contested possessions (3rd) and tilt 43/57 between handball and kick. The strengths jump out clearly: they're 2nd for rebound-50s (43.1/g), 3rd for contested possessions (138.3/g), 4th for tackles (59.0/g). The flip side is harder to ignore: 17th for marks inside 50 (8.9/g), 16th for goals (11.6/g), 16th for clangers (60.1/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Sam Walsh**, leading the team in disposals at 28.7/game.

#### Essendon

After 7 games of 2026, the Essendon are sitting in the lower third for ball use (13th for disposals) and average 359.4 disposals, 12.0 goals and 49.7 tackles per game. Their stat profile reads as struggling to win territory and force the issue forward; with notably soft pressure around the ball. They go inside 50 44.9 times a game (18th in the league), win 116.7 contested possessions (18th) and tilt 44/56 between handball and kick. The strengths jump out clearly: they're 1st for clangers (52.0/g), 5th for free kicks against (17.4/g), 8th for handballs (156.7/g). The flip side is harder to ignore: 18th for inside-50s (44.9/g), 18th for contested possessions (116.7/g), 17th for tackles (49.7/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Archie Roberts**, leading the team in disposals at 32.1/game.

#### Port Adelaide

After 7 games of 2026, the Port Adelaide are sitting in the lower third for ball use (14th for disposals) and average 349.9 disposals, 12.9 goals and 55.9 tackles per game. Their stat profile reads as a kick-first ball movement (64% by foot). They go inside 50 52.7 times a game (11th in the league), win 117.7 contested possessions (17th) and tilt 36/64 between handball and kick. The strengths jump out clearly: they're 2nd for marks inside 50 (14.7/g), 3rd for marks (108.0/g), 5th for kicks (224.3/g). The flip side is harder to ignore: 18th for handballs (125.6/g), 18th for rebound-50s (32.1/g), 17th for contested possessions (117.7/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Zak Butters**, leading the team in disposals at 30.7/game.

#### Melbourne

After 7 games of 2026, the Melbourne are sitting in the lower third for ball use (15th for disposals) and average 346.6 disposals, 14.9 goals and 55.1 tackles per game. Their stat profile reads as dominating territory and connecting cleanly inside 50. They go inside 50 57.4 times a game (3rd in the league), win 128.0 contested possessions (12th) and tilt 40/60 between handball and kick. The strengths jump out clearly: they're 1st for free kicks against (14.1/g), 2nd for hit-outs (44.0/g), 3rd for inside-50s (57.4/g). The flip side is harder to ignore: 17th for free kicks for (16.6/g), 16th for uncontested possessions (204.1/g), 16th for rebound-50s (36.6/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Jack Steele**, leading the team in disposals at 24.9/game.

#### Adelaide

After 7 games of 2026, the Adelaide are anchored near the bottom of the table (16th for disposals) and average 345.4 disposals, 12.7 goals and 57.0 tackles per game. Their stat profile reads as a kick-first ball movement (63% by foot); struggling to win territory and force the issue forward; frequently absorbing entries and rebounding from defence. They go inside 50 48.4 times a game (15th in the league), win 128.4 contested possessions (11th) and tilt 37/63 between handball and kick. The strengths jump out clearly: they're 1st for rebound-50s (43.9/g), 6th for free kicks against (17.6/g), 8th for kicks (218.1/g). The flip side is harder to ignore: 18th for hit-outs (9.3/g), 17th for handballs (127.3/g), 16th for disposals (345.4/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Wayne Milera**, leading the team in disposals at 25.0/game.

#### West Coast

After 7 games of 2026, the West Coast are anchored near the bottom of the table (17th for disposals) and average 326.7 disposals, 9.3 goals and 52.9 tackles per game. Their stat profile reads as struggling to win territory and force the issue forward; with notably soft pressure around the ball. They go inside 50 46.6 times a game (17th in the league), win 123.0 contested possessions (14th) and tilt 40/60 between handball and kick. The strengths jump out clearly: they're 8th for free kicks for (18.7/g), 12th for rebound-50s (38.6/g), 12th for clangers (57.6/g). The flip side is harder to ignore: 18th for uncontested possessions (196.7/g), 17th for disposals (326.7/g), 17th for kicks (194.6/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Harley Reid**, leading the team in disposals at 22.9/game.

#### Richmond

After 7 games of 2026, the Richmond are anchored near the bottom of the table (18th for disposals) and average 326.1 disposals, 8.4 goals and 52.1 tackles per game. Their stat profile reads as struggling to win territory and force the issue forward; with notably soft pressure around the ball. They go inside 50 48.1 times a game (16th in the league), win 119.4 contested possessions (15th) and tilt 43/57 between handball and kick. The strengths jump out clearly: they're 5th for free kicks for (19.6/g), 9th for free kicks against (18.7/g), 12th for contested marks (8.4/g). The flip side is harder to ignore: 18th for disposals (326.1/g), 18th for kicks (187.0/g), 18th for marks (80.9/g) — that's where opposition coaches will be drawing up plans. Key player to watch is **Jayden Short**, leading the team in disposals at 25.5/game.

<!-- 2026-TEAM-ANALYSIS-END -->

### 2026 finals pathway — what each team needs

<!-- 2026-FINALS-PATHWAY-START -->
What does each AFL team need to do — from here — to make finals this year, and what would have to go right for them to play in the grand final? After Round 8 of the 2026 season, every side has played 7 games with roughly 15 games left in the home-and-away. The paragraphs below combine the actual 2026 ladder (wins, losses, percentage) with the team's stat profile across 16 categories to write an honest, data-driven mid-season assessment for each club. The grand-final read at the end of each paragraph maps to one of: **contender** (top-4, elite stat profile), **live chance** (top-8 with weapons), **chance** (top-8 without breadth), **long shot** (9-12), **mathematically alive** (13-16) or **season effectively over** (17-18) — the cut-offs are deliberately blunt.

![2026 AFL Finals Pathway ladder chart](assets/charts/finals_pathway_2026.png)


#### 1st — Sydney

**Sydney** sit 1st on the ladder after Round 8 (6-1, 24 points, 178.1%) — their job is to hold the double-chance, not chase it — every win from here is locking in a top-4 finish, not scrambling for one. On the underlying numbers they are top-4 for goals (1st), top-4 for clearances (4th), top-4 for tackles (1st) — that's the stat fingerprint behind the ladder position. From here, the one thing they must keep doing is their goals (1st of 18) — that is the engine of the run. For finals positioning, their nearest finals rival is **Fremantle** (6-1, 2nd) — the team most likely to bump them out if they slip. Grand final read: **contender**. They have the ladder buffer and the stat profile of a side built to play deep in September — 5 of the five core categories sit top-5, which is what every recent premier has had at this point of the year.

#### 2nd — Fremantle

**Fremantle** sit 2nd on the ladder after Round 8 (6-1, 24 points, 137.6%) — their job is to hold the double-chance, not chase it — every win from here is locking in a top-4 finish, not scrambling for one. On the underlying numbers they are top-4 for tackles (3rd), top-4 for contested possessions (2nd) — that's the stat fingerprint behind the ladder position. From here, the one thing they must keep doing is their contested possessions (2nd of 18) — that is the engine of the run. For finals positioning, their nearest finals rival is **Hawthorn** (6-1, 3rd) — the team most likely to bump them out if they slip. Grand final read: **live chance**. The ladder position is right and the stat profile has real weapons, but they will need to either climb a rung or two further or peak at the right time — flag-winners from outside the top four are the historical exception, not the rule.

#### 3rd — Hawthorn

**Hawthorn** sit 3rd on the ladder after Round 8 (6-1, 24 points, 124.5%) — their job is to hold the double-chance, not chase it — every win from here is locking in a top-4 finish, not scrambling for one. On the underlying numbers they are top-4 for goals (3rd), top-4 for tackles (4th) — that's the stat fingerprint behind the ladder position. From here, the one thing they must keep doing is their goals (3rd of 18) — that is the engine of the run. For finals positioning, their nearest finals rival is **Melbourne** (5-2, 4th) — the team most likely to bump them out if they slip. Grand final read: **contender**. They have the ladder buffer and the stat profile of a side built to play deep in September — 3 of the five core categories sit top-5, which is what every recent premier has had at this point of the year.

#### 4th — Melbourne

**Melbourne** sit 4th on the ladder after Round 8 (5-2, 20 points, 102.8%) — their job is to hold the double-chance, not chase it — every win from here is locking in a top-4 finish, not scrambling for one. On the underlying numbers they are top-4 for goals (4th), top-4 for inside-50s (3rd) — that's the stat fingerprint behind the ladder position. From here, the one thing they must keep doing is their inside-50s (3rd of 18) — that is the engine of the run. For finals positioning, their nearest finals rival is **Brisbane Lions** (4-3, 5th) — the team most likely to bump them out if they slip. Grand final read: **live chance**. The ladder position is right and the stat profile has real weapons, but they will need to either climb a rung or two further or peak at the right time — flag-winners from outside the top four are the historical exception, not the rule.

#### 5th — Brisbane Lions

**Brisbane Lions** sit 5th on the ladder after Round 8 (4-3, 16 points, 118.8%) — they need roughly 8 wins from their remaining 15 games to lock the eight in, which their form line is comfortably on track to do. On the underlying numbers they are top-4 for goals (4th), top-4 for clearances (1st), bottom-4 for tackles (18th) — that's the stat fingerprint behind the ladder position. From here, the one thing they have to fix is their tackles ranking (18th of 18) — every finals-tier opponent will target it. For finals positioning, their nearest finals rival is **North Melbourne** (4-3, 6th) — the team most likely to bump them out if they slip. Grand final read: **live chance**. The ladder position is right and the stat profile has real weapons, but they will need to either climb a rung or two further or peak at the right time — flag-winners from outside the top four are the historical exception, not the rule.

#### 6th — North Melbourne

**North Melbourne** sit 6th on the ladder after Round 8 (4-3, 16 points, 115.9%) — they need roughly 8 wins from their remaining 15 games to lock the eight in, which their form line is comfortably on track to do. On the underlying numbers they are top-4 for clearances (3rd) — that's the stat fingerprint behind the ladder position. From here, no single stat screams crisis, but they need every part of their game to lift a notch to be a serious finals threat. For finals positioning, their nearest finals rival is **Gold Coast** (4-3, 7th) — the team most likely to bump them out if they slip. Grand final read: **chance, not a contender**. Sitting inside the eight is the easy bit; the stat profile doesn't yet have the breadth to beat top-4 sides three weeks running, which is what the path demands once finals arrive.

#### 7th — Gold Coast

**Gold Coast** sit 7th on the ladder after Round 8 (4-3, 16 points, 114.4%) — they need roughly 8 wins from their remaining 15 games to lock the eight in, which their form line is comfortably on track to do. On the underlying numbers they are top-4 for goals (2nd), top-4 for inside-50s (2nd) — that's the stat fingerprint behind the ladder position. From here, the one thing they have to fix is their marks ranking (17th of 18) — every finals-tier opponent will target it. For finals positioning, their nearest finals rival is **Collingwood** (4-3, 8th) — the team most likely to bump them out if they slip. Grand final read: **live chance**. The ladder position is right and the stat profile has real weapons, but they will need to either climb a rung or two further or peak at the right time — flag-winners from outside the top four are the historical exception, not the rule.

#### 8th — Collingwood

**Collingwood** sit 8th on the ladder after Round 8 (4-3, 16 points, 110.2%) — they need roughly 8 wins from their remaining 15 games to lock the eight in, which their form line is comfortably on track to do. On the underlying numbers they are bottom-4 for goals (15th), bottom-4 for clearances (18th), bottom-4 for contested possessions (16th) — that's the stat fingerprint behind the ladder position. From here, the one thing they have to fix is their clearances ranking (18th of 18) — every finals-tier opponent will target it. For finals positioning, their nearest finals rival is **Geelong** (4-3, 9th) — the team most likely to bump them out if they slip. Grand final read: **chance, not a contender**. Sitting inside the eight is the easy bit; the stat profile doesn't yet have the breadth to beat top-4 sides three weeks running, which is what the path demands once finals arrive.

#### 9th — Geelong

**Geelong** sit 9th on the ladder after Round 8 (4-3, 16 points, 108.6%) — ~8 wins from 15 games gets them back in — very doable on paper, but with no margin for a flat patch. On the underlying numbers they are top-4 for tackles (2nd) — that's the stat fingerprint behind the ladder position. From here, the one thing they have to fix is their rebound-50s ranking (13th of 18) — every finals-tier opponent will target it. For finals positioning, the side blocking their road back in is **Collingwood** (4-3, 8th) — leapfrog them and the finals door reopens. Grand final read: **realistically out of premiership contention**. Even if they sneak into the eight from here, finishing 7th or 8th means winning four straight against teams who finished above them — that's not how flags get won.

#### 10th — Western Bulldogs

**Western Bulldogs** sit 10th on the ladder after Round 8 (4-3, 16 points, 91.8%) — ~8 wins from 15 games gets them back in — very doable on paper, but with no margin for a flat patch. On the underlying numbers they are top-4 for clearances (2nd) — that's the stat fingerprint behind the ladder position. From here, the one thing they have to fix is their contested possessions ranking (13th of 18) — every finals-tier opponent will target it. For finals positioning, the side blocking their road back in is **Geelong** (4-3, 9th) — leapfrog them and the finals door reopens. Grand final read: **realistically out of premiership contention**. Even if they sneak into the eight from here, finishing 7th or 8th means winning four straight against teams who finished above them — that's not how flags get won.

#### 11th — Port Adelaide

**Port Adelaide** sit 11th on the ladder after Round 8 (3-4, 12 points, 112.5%) — ~9 wins from 15 games gets them back in — very doable on paper, but with no margin for a flat patch. On the underlying numbers they are bottom-4 for contested possessions (17th) — that's the stat fingerprint behind the ladder position. From here, the one thing they have to fix is their rebound-50s ranking (18th of 18) — every finals-tier opponent will target it. For finals positioning, the side blocking their road back in is **Western Bulldogs** (4-3, 10th) — leapfrog them and the finals door reopens. Grand final read: **realistically out of premiership contention**. Even if they sneak into the eight from here, finishing 7th or 8th means winning four straight against teams who finished above them — that's not how flags get won.

#### 12th — St Kilda

**St Kilda** sit 12th on the ladder after Round 8 (3-4, 12 points, 110.1%) — ~9 wins from 15 games gets them back in — very doable on paper, but with no margin for a flat patch. On the underlying numbers they sit broadly mid-pack across the core stat lines, which is consistent with their ladder position and means the form line is honest. From here, the one thing they have to fix is their tackles ranking (14th of 18) — every finals-tier opponent will target it. For finals positioning, the side blocking their road back in is **Port Adelaide** (3-4, 11th) — leapfrog them and the finals door reopens. Grand final read: **realistically out of premiership contention**. Even if they sneak into the eight from here, finishing 7th or 8th means winning four straight against teams who finished above them — that's not how flags get won.

#### 13th — Adelaide

**Adelaide** sit 13th on the ladder after Round 8 (3-4, 12 points, 96.1%) — they would need to win ~9 of their remaining 15 games to claim a finals spot, which would require a near-perfect run from here. On the underlying numbers they are bottom-4 for clearances (16th), bottom-4 for inside-50s (15th) — that's the stat fingerprint behind the ladder position. From here, the one thing they have to fix is their clearances ranking (16th of 18) — every finals-tier opponent will target it. For finals positioning, the side blocking their road back in is **St Kilda** (3-4, 12th) — leapfrog them and the finals door reopens. Grand final read: **no realistic path**. The maths technically still works — win out and percentage helps — but the form line and stat profile are pointing the other way, and no team has come from this deep at this point of the year to win a flag in the modern era.

#### 14th — Greater Western Sydney

**Greater Western Sydney** sit 14th on the ladder after Round 8 (3-4, 12 points, 89.8%) — they would need to win ~9 of their remaining 15 games to claim a finals spot, which would require a near-perfect run from here. On the underlying numbers they are top-4 for inside-50s (3rd), top-4 for contested possessions (4th) — that's the stat fingerprint behind the ladder position. From here, the one thing they have to fix is their goals ranking (13th of 18) — every finals-tier opponent will target it. For finals positioning, the side blocking their road back in is **Adelaide** (3-4, 13th) — leapfrog them and the finals door reopens. Grand final read: **no realistic path**. The maths technically still works — win out and percentage helps — but the form line and stat profile are pointing the other way, and no team has come from this deep at this point of the year to win a flag in the modern era.

#### 15th — West Coast

**West Coast** sit 15th on the ladder after Round 8 (2-5, 8 points, 55.8%) — they would need to win ~10 of their remaining 15 games to claim a finals spot, which would require a near-perfect run from here. On the underlying numbers they are bottom-4 for goals (17th), bottom-4 for tackles (15th), bottom-4 for inside-50s (17th) — that's the stat fingerprint behind the ladder position. From here, the one thing they have to fix is their goals ranking (17th of 18) — every finals-tier opponent will target it. For finals positioning, the side blocking their road back in is **Greater Western Sydney** (3-4, 14th) — leapfrog them and the finals door reopens. Grand final read: **no realistic path**. The maths technically still works — win out and percentage helps — but the form line and stat profile are pointing the other way, and no team has come from this deep at this point of the year to win a flag in the modern era.

#### 16th — Carlton

**Carlton** sit 16th on the ladder after Round 8 (1-6, 4 points, 80.3%) — they would need to win ~11 of their remaining 15 games to claim a finals spot, which would require a near-perfect run from here. On the underlying numbers they are bottom-4 for goals (16th), top-4 for tackles (4th), top-4 for contested possessions (3rd) — that's the stat fingerprint behind the ladder position. From here, the one thing they have to fix is their marks inside 50 ranking (17th of 18) — every finals-tier opponent will target it. For finals positioning, the side blocking their road back in is **West Coast** (2-5, 15th) — leapfrog them and the finals door reopens. Grand final read: **no realistic path**. The maths technically still works — win out and percentage helps — but the form line and stat profile are pointing the other way, and no team has come from this deep at this point of the year to win a flag in the modern era.

#### 17th — Essendon

**Essendon** sit 17th on the ladder after Round 8 (1-6, 4 points, 72.9%) — the maths is technically alive but the path requires a miracle run the form line does not support. On the underlying numbers they are bottom-4 for clearances (17th), bottom-4 for tackles (17th), bottom-4 for inside-50s (18th) — that's the stat fingerprint behind the ladder position. From here, the one thing they have to fix is their contested possessions ranking (18th of 18) — every finals-tier opponent will target it. For finals positioning, the side blocking their road back in is **Carlton** (1-6, 16th) — leapfrog them and the finals door reopens. Grand final read: **season effectively over**. Even a finals berth would require the kind of run that simply doesn't happen at AFL level — the priority from here is list build, draft position and next season.

#### 18th — Richmond

**Richmond** sit 18th on the ladder after Round 8 (0-7, 0 points, 54.2%) — the maths is technically alive but the path requires a miracle run the form line does not support. On the underlying numbers they are bottom-4 for goals (18th), bottom-4 for clearances (15th), bottom-4 for tackles (16th) — that's the stat fingerprint behind the ladder position. From here, the one thing they have to fix is their goals ranking (18th of 18) — every finals-tier opponent will target it. For finals positioning, the side blocking their road back in is **Essendon** (1-6, 17th) — leapfrog them and the finals door reopens. Grand final read: **season effectively over**. Even a finals berth would require the kind of run that simply doesn't happen at AFL level — the priority from here is list build, draft position and next season.
<!-- 2026-FINALS-PATHWAY-END -->

### 2026 Brownlow Medal predictor

<!-- 2026-BROWNLOW-PREDICTOR-START -->
The **Brownlow Medal** is the AFL's individual award for the "fairest and best" player, voted on by the on-field umpires with a 3-2-1 split per game. It is impossible to predict actual votes without modelling umpire behaviour, but we *can* build a defensible **statistical proxy** — a composite score over the stats that historically correlate with vote-earning. The weights below were validated against every player-game from 2010-2025 (n=145,150) where actual `brownlow_votes` are recorded — the top 1% of proxy games captured ~70% of vote-earning performances. Players need at least 3 games played to be ranked. Suspended players are not penalised — this proxy is a stat-profile model, not a vote forecaster.

**Composite formula** (z-scored across all eligible players, summed with weights): `0.30 × disposals + 0.22 × clearances + 0.18 × contested-poss + 0.15 × effective-disposals + 0.15 × goals`. Effective disposals are approximated as `disposals - clangers` because the raw data does not carry a true effective-disposal column. Goals are weighted higher than the conventional midfielder-only template (15% vs the ~5% common in pure-midfielder proxies) because that materially improves correlation with actual historical Brownlow votes.

![2026 Brownlow predictor](assets/charts/brownlow_predictor_2026.png)

#### Top 15 Brownlow proxy candidates — 2026 season-to-date (after Round 8)

| Rank | Player | Team | Games | Disp/g | Clear/g | CP/g | Goals/g | Proxy | Proj. votes |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Nick Daicos | Collingwood | 6 | 37.0 | 5.8 | 10.8 | 0.83 | +2.58 | +56.8 |
| 2 | Clayton Oliver | Greater Western Sydney | 7 | 30.9 | 7.1 | 14.9 | 0.29 | +2.47 | +54.3 |
| 3 | Lachie Neale | Brisbane Lions | 7 | 31.1 | 7.7 | 13.4 | 0.00 | +2.42 | +53.2 |
| 4 | Zak Butters | Port Adelaide | 7 | 30.7 | 6.0 | 12.3 | 0.29 | +2.12 | +46.6 |
| 5 | Bailey Smith | Geelong | 7 | 32.0 | 5.4 | 11.4 | 0.43 | +2.09 | +46.1 |
| 6 | Christian Petracca | Gold Coast | 5 | 26.0 | 5.8 | 11.6 | 2.00 | +2.09 | +45.9 |
| 7 | Patrick Cripps | Carlton | 7 | 25.4 | 7.4 | 15.6 | 0.29 | +2.07 | +45.6 |
| 8 | Jai Newcombe | Hawthorn | 7 | 26.4 | 7.9 | 12.3 | 0.43 | +2.06 | +45.4 |
| 9 | Isaac Heeney | Sydney | 5 | 24.6 | 6.0 | 12.8 | 2.00 | +2.04 | +44.9 |
| 10 | Harry Sheezel | North Melbourne | 7 | 32.7 | 5.0 | 10.1 | 0.43 | +2.01 | +44.2 |
| 11 | Caleb Serong | Fremantle | 7 | 25.7 | 6.4 | 12.9 | 0.71 | +1.92 | +42.2 |
| 12 | Tristan Xerri | North Melbourne | 4 | 21.0 | 7.5 | 17.5 | 0.25 | +1.89 | +41.5 |
| 13 | Marcus Bontempelli | Western Bulldogs | 7 | 26.6 | 4.9 | 11.1 | 1.43 | +1.82 | +40.0 |
| 14 | Matthew Kennedy | Western Bulldogs | 7 | 26.3 | 7.1 | 11.3 | 0.29 | +1.82 | +39.9 |
| 15 | Max Gawn | Melbourne | 7 | 21.9 | 6.4 | 15.6 | 0.57 | +1.79 | +39.3 |

On the proxy, **Nick Daicos** (Collingwood) leads the field — built on 37.0 disposals/g across 6 games. The composite score (+2.58) sits 0.12 clear of second place. **Clayton Oliver** (Greater Western Sydney) is the closest challenger at +2.47, with 30.9 disposals/g and 7.1 clearances/g. The proxy is a statistical model, not actual umpire votes — it captures the stat-profile umpires *historically* reward, but it cannot model individual game narrative, suspension impact or the umpire panel's eye for a defensive midfielder.
<!-- 2026-BROWNLOW-PREDICTOR-END -->

### 2026 player performance stats — what to look for and what the data says

<!-- 2026-STAT-LEADERS-START -->
This section is a guide to the AFL performance statistics that fans, analysts and SuperCoach players track most closely — what each stat measures, who is leading it in 2026, what the league-wide distribution looks like, and which other stats most reliably predict it. All numbers are computed live from `data/player_data/` for 2026 (rounds 1-8, **460 eligible players** with >=3 games, **2742 player-games** included). Correlations are Pearson r on the per-game frame; with several thousand player-games, p-values are universally tiny — read the magnitude of r, not the significance star.

![2026 AFL statistical leaders](assets/charts/player_stat_leaders_2026.png)

#### Disposal-based stats — volume and quality of ball use

##### Disposals per game

**What it measures.** Total kicks plus handballs in a game — the single broadest measure of how often a player has the ball. **Why it matters.** It is the headline SuperCoach scoring stat and the prediction target this repo's main model is built around. Volume midfielders and rebounding defenders dominate this leaderboard.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 37.0 |
| 2 | Harry Sheezel | North Melbourne | 32.7 |
| 3 | Archie Roberts | Essendon | 32.1 |
| 4 | Bailey Smith | Geelong | 32.0 |
| 5 | Lachie Neale | Brisbane Lions | 31.1 |

League distribution (eligible players, season-to-date): mean **15.85**, std 5.93, p10 8.75 / p50 15.00 / p90 24.40, max 37.00.

Top per-game correlates: `effective_disposals` (r = +0.97 *(mechanically related)*), `uncontested_possessions` (r = +0.88), `kicks` (r = +0.83).

##### Kicks per game

**What it measures.** Just the kicked disposals. **Why it matters.** Kicks tend to come from outside-midfielders, half-backs and tall rebounders — players who clear the ball by foot rather than shovel it into a contest. A player who kicks much more than they handball is usually playing a distributor / launch role.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Archie Roberts | Essendon | 22.4 |
| 2 | Bailey Smith | Geelong | 21.4 |
| 3 | Jack Sinclair | St Kilda | 21.0 |
| 4 | Lachie Ash | Greater Western Sydney | 20.7 |
| 5 | Jayden Short | Richmond | 20.5 |

League distribution (eligible players, season-to-date): mean **9.28**, std 3.75, p10 4.75 / p50 8.83 / p90 14.51, max 22.43.

Top per-game correlates: `disposals` (r = +0.83), `effective_disposals` (r = +0.81), `uncontested_possessions` (r = +0.79).

##### Handballs per game

**What it measures.** The hand-passed half of disposals. **Why it matters.** Handball volume tracks contest involvement — a player wins the ball at a stoppage, then handballs out to a runner. Inside-mids and clearance specialists tend to lead this stat.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Clayton Oliver | Greater Western Sydney | 20.7 |
| 2 | Ryley Sanders | Western Bulldogs | 17.5 |
| 3 | Lachie Neale | Brisbane Lions | 16.6 |
| 4 | Nick Daicos | Collingwood | 16.5 |
| 5 | Harry Sheezel | North Melbourne | 16.0 |

League distribution (eligible players, season-to-date): mean **6.56**, std 3.29, p10 3.00 / p50 6.00 / p90 11.43, max 20.71.

Top per-game correlates: `disposals` (r = +0.77), `effective_disposals` (r = +0.75), `contested_possessions` (r = +0.65).

##### Effective disposals per game (disposals − clangers)

**What it measures.** Disposals that did not result in a clanger, computed here as `max(disposals - clangers, 0)` because the raw data does not carry a true effective-disposal column. **Why it matters.** It is a defensible proxy for disposal *quality* — high-volume ball-users who don't turn it over. The same proxy is used in the Brownlow predictor on this page.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 32.7 |
| 2 | Archie Roberts | Essendon | 29.0 |
| 3 | Lachie Neale | Brisbane Lions | 28.9 |
| 4 | Harry Sheezel | North Melbourne | 28.7 |
| 5 | Jack Sinclair | St Kilda | 28.1 |

League distribution (eligible players, season-to-date): mean **13.35**, std 5.54, p10 6.56 / p50 12.71 / p90 21.18, max 32.67.

Top per-game correlates: `disposals` (r = +0.97 *(mechanically related)*), `uncontested_possessions` (r = +0.87), `kicks` (r = +0.81).

#### Scoring stats — goals, behinds and conversion

##### Goals per game

**What it measures.** Goals kicked. **Why it matters.** Forwards live and die by this stat. It is volatile game-to-game (a single missed shot can halve your score), so multi-game averages and shot-source context (marks-inside-50, contested marks) matter more than any one game.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Gunston | Hawthorn | 4.00 |
| 2 | Ben King | Gold Coast | 3.71 |
| 3 | Jeremy Cameron | Geelong | 3.33 |
| 4 | Nick Larkey | North Melbourne | 3.00 |
| 5 | Aaron Naughton | Western Bulldogs | 2.71 |

League distribution (eligible players, season-to-date): mean **0.56**, std 0.66, p10 0.00 / p50 0.33 / p90 1.50, max 4.00.

Top per-game correlates: `marks_inside_50` (r = +0.68), `behinds` (r = +0.32), `rebound_50s` (r = -0.31).

**Goal conversion rate.** Defined as `goals / (goals + behinds)`, season-to-date, for players with >=2 goals total. League distribution (n=259): mean **62.8%**, std 19.0pp, p10 40% / p50 60% / p90 100%.

| Rank | Player | Team | G | B | Conversion |
|---|---|---|---|---|---|
| 1 | Jordan Dawson | Adelaide | 6 | 0 | 100.0% |
| 2 | Sam Durham | Essendon | 5 | 0 | 100.0% |
| 3 | Isaac Cumming | Adelaide | 4 | 0 | 100.0% |
| 4 | James Jordon | Sydney | 4 | 0 | 100.0% |
| 5 | Lachie Weller | Gold Coast | 4 | 0 | 100.0% |

##### Behinds per game

**What it measures.** Minor scores — shots that hit the post or go through the smaller posts. **Why it matters.** Rarely predicted alone — it is too noisy. Best read alongside goals to compute **conversion rate** (`goals / (goals + behinds)`), the cleanest available signal of forward accuracy.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jake Waterman | West Coast | 2.86 |
| 2 | Tom Lynch | Richmond | 2.67 |
| 3 | Jack Gunston | Hawthorn | 2.67 |
| 4 | Nate Caddy | Essendon | 2.17 |
| 5 | Mitch Georgiades | Port Adelaide | 2.14 |

League distribution (eligible players, season-to-date): mean **0.42**, std 0.47, p10 0.00 / p50 0.29 / p90 1.00, max 2.86.

Top per-game correlates: `marks_inside_50` (r = +0.54), `goals` (r = +0.32), `rebound_50s` (r = -0.25).

#### Contested and ground-ball stats — the inside game

##### Contested possessions per game

**What it measures.** Wins of the ball under physical pressure — ground-balls, taps, and contested marks. **Why it matters.** This is the cleanest stat for separating a midfielder's *contest* role from an outside ball-user's *spread* role. It correlates strongly with clearances and tackles.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Tristan Xerri | North Melbourne | 17.50 |
| 2 | Patrick Cripps | Carlton | 15.57 |
| 3 | Max Gawn | Melbourne | 15.57 |
| 4 | Clayton Oliver | Greater Western Sydney | 14.86 |
| 5 | Lachie Neale | Brisbane Lions | 13.43 |

League distribution (eligible players, season-to-date): mean **5.59**, std 2.54, p10 3.00 / p50 5.00 / p90 9.34, max 17.50.

Top per-game correlates: `clearances` (r = +0.73), `handballs` (r = +0.65), `disposals` (r = +0.58).

##### Clearances per game

**What it measures.** Disposals that move the ball clear of a stoppage (a centre-bounce or boundary throw-in). **Why it matters.** Stoppage dominance is one of the few team-level wins a midfield can manufacture. Top clearance players are almost always the inside-mid fulcrums of their team.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jai Newcombe | Hawthorn | 7.86 |
| 2 | Lachie Neale | Brisbane Lions | 7.71 |
| 3 | Tristan Xerri | North Melbourne | 7.50 |
| 4 | Patrick Cripps | Carlton | 7.43 |
| 5 | Matthew Kennedy | Western Bulldogs | 7.14 |

League distribution (eligible players, season-to-date): mean **1.51**, std 1.72, p10 0.00 / p50 0.83 / p90 4.25, max 7.86.

Top per-game correlates: `contested_possessions` (r = +0.73), `handballs` (r = +0.55), `disposals` (r = +0.48).

##### Tackles per game

**What it measures.** Pressure acts that physically stop a ball-carrier. **Why it matters.** Defensive midfield work — the unsung currency of forward-half pressure and turnover football. It correlates with clearances (you tackle the same opponent you compete against) but tells a different story.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Tristan Xerri | North Melbourne | 8.00 |
| 2 | Tom Atkins | Geelong | 7.71 |
| 3 | James Rowbottom | Sydney | 7.43 |
| 4 | Jack Steele | Melbourne | 7.29 |
| 5 | Andrew Brayshaw | Fremantle | 7.00 |

League distribution (eligible players, season-to-date): mean **2.46**, std 1.41, p10 1.00 / p50 2.14 / p90 4.40, max 8.00.

Top per-game correlates: `clearances` (r = +0.40), `contested_possessions` (r = +0.37), `handballs` (r = +0.30).

##### Hit-outs per game (ruckmen only)

**What it measures.** Wins by a ruckman at a ruck contest (the tap from a centre bounce or stoppage). **Why it matters.** Ruckman-only stat — the distribution is bimodal: ~1 player per team registers double-digits, everyone else is 0. Always read this leaderboard as "top ruckmen", not "top players".

**Bimodal distribution warning.** 88% of eligible 2026 players average less than 1 hit-out per game — they are not ruckmen. The league mean below is dragged down by all the zeros; the meaningful comparison is between ruckmen, where the top of the distribution sits in the 25-35 range.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jarrod Witts | Gold Coast | 35.7 |
| 2 | Brodie Grundy | Sydney | 35.0 |
| 3 | Max Gawn | Melbourne | 33.1 |
| 4 | Nick Madden | Greater Western Sydney | 28.7 |
| 5 | Jordon Sweet | Port Adelaide | 28.5 |

League distribution (eligible players, season-to-date): mean **1.53**, std 5.39, p10 0.00 / p50 0.00 / p90 2.31, max 35.71.

Top per-game correlates: `clearances` (r = +0.25), `uncontested_possessions` (r = -0.24), `free_kicks_for` (r = +0.21).

#### Territory stats — moving the ball forward

##### Inside 50s per game

**What it measures.** Disposals or carries that move the ball into the team's attacking 50m arc. **Why it matters.** Territory currency — the precondition for goals. Wing/half-forward players who launch attacks lead this stat. It correlates with kicks and disposals because most inside-50s are foot-delivered.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Nick Daicos | Collingwood | 8.33 |
| 2 | Bailey Smith | Geelong | 7.57 |
| 3 | Chad Warner | Sydney | 6.71 |
| 4 | Toby Greene | Greater Western Sydney | 6.43 |
| 5 | Marcus Bontempelli | Western Bulldogs | 6.43 |

League distribution (eligible players, season-to-date): mean **2.30**, std 1.36, p10 0.67 / p50 2.15 / p90 4.00, max 8.33.

Top per-game correlates: `disposals` (r = +0.52), `effective_disposals` (r = +0.49), `kicks` (r = +0.48).

##### Marks per game

**What it measures.** Total uncontested + contested marks taken. **Why it matters.** Aerial dominance and intercept defence. Loose-half-back roles dominate the total-marks leaderboard because they sit behind the play and fly under kicks. Tall forwards lead a separate, narrower stat — marks inside 50.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Callum Wilkie | St Kilda | 12.1 |
| 2 | Aliir Aliir | Port Adelaide | 9.4 |
| 3 | Dan Houston | Collingwood | 8.6 |
| 4 | Jayden Short | Richmond | 8.5 |
| 5 | Lachie Ash | Greater Western Sydney | 8.4 |

League distribution (eligible players, season-to-date): mean **4.05**, std 1.72, p10 2.00 / p50 3.86 / p90 6.43, max 12.14.

Top per-game correlates: `kicks` (r = +0.59), `uncontested_possessions` (r = +0.54), `effective_disposals` (r = +0.44).

##### Marks inside 50 per game

**What it measures.** Marks taken inside the attacking 50m arc — i.e. marks that turn directly into shots on goal. **Why it matters.** This is the strongest single predictor of a forward's goal output. It is what separates a deep-forward role from a high-half-forward role, and the correlation with goals is the highest of any stat in this section.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Jack Gunston | Hawthorn | 4.67 |
| 2 | Mitch Georgiades | Port Adelaide | 4.29 |
| 3 | Jeremy Cameron | Geelong | 3.50 |
| 4 | Jye Amiss | Fremantle | 3.43 |
| 5 | Charlie Curnow | Sydney | 3.29 |

League distribution (eligible players, season-to-date): mean **0.53**, std 0.75, p10 0.00 / p50 0.29 / p90 1.67, max 4.67.

Top per-game correlates: `goals` (r = +0.68), `behinds` (r = +0.54), `contested_marks` (r = +0.36).

#### Discipline stats — errors and free kicks

##### Clangers per game

**What it measures.** Errors — missed targets, fumbles, free kicks given away by the ball-carrier. **Why it matters.** Clangers are the friction term on disposal volume — a high-disposal player who also leads in clangers is being asked to play through traffic, not necessarily playing badly. The correlation with frees-against is mechanical: many clangers *are* frees-against.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Matt Rowell | Gold Coast | 6.75 |
| 2 | Harley Reid | West Coast | 6.29 |
| 3 | Brodie Grundy | Sydney | 6.14 |
| 4 | Jacob Hopper | Richmond | 6.00 |
| 5 | Toby Greene | Greater Western Sydney | 5.86 |

League distribution (eligible players, season-to-date): mean **2.50**, std 1.01, p10 1.33 / p50 2.37 / p90 3.71, max 6.75.

Top per-game correlates: `free_kicks_against` (r = +0.63 *(mechanically related)*), `contested_possessions` (r = +0.34), `disposals` (r = +0.32).

##### Free kicks for per game

**What it measures.** Free kicks paid to the player. **Why it matters.** A weak isolated signal — frees-for tracks contest involvement (rucks especially) more than skill. Best used as a tiebreaker rather than a standalone metric.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Tristan Xerri | North Melbourne | 3.50 |
| 2 | Max Gawn | Melbourne | 3.14 |
| 3 | Bailey Williams | West Coast | 3.00 |
| 4 | Tim English | Western Bulldogs | 2.75 |
| 5 | Darcy Cameron | Collingwood | 2.71 |

League distribution (eligible players, season-to-date): mean **0.82**, std 0.53, p10 0.29 / p50 0.71 / p90 1.50, max 3.50.

Top per-game correlates: `contested_possessions` (r = +0.42), `clearances` (r = +0.29), `tackles` (r = +0.24).

##### Free kicks against per game

**What it measures.** Free kicks paid against the player. **Why it matters.** Discipline / aggression marker, with the caveat that ruck contest infringements inflate the number for ruckmen. Reads like a clanger when it correlates with them.

| Rank | Player | Team | Per game |
|---|---|---|---|
| 1 | Brodie Grundy | Sydney | 3.00 |
| 2 | Matt Rowell | Gold Coast | 3.00 |
| 3 | Jack Graham | West Coast | 3.00 |
| 4 | Liam Baker | West Coast | 2.50 |
| 5 | Sam Draper | Brisbane Lions | 2.50 |

League distribution (eligible players, season-to-date): mean **0.83**, std 0.51, p10 0.25 / p50 0.75 / p90 1.50, max 3.00.

Top per-game correlates: `clangers` (r = +0.63 *(mechanically related)*), `hit_outs` (r = +0.16), `clearances` (r = +0.16).

#### Team-level stats — what the scoreboard says

Team-level stats use `data/matches/matches_2026.csv` rather than per-player aggregates. Total team score is `goals × 6 + behinds`; margin is the team's score minus the opponent's. A first-quarter score is a useful early-momentum signal — strong starters tend to keep the lead.

##### Total team score per game

| Rank | Team | Avg score | Avg margin | Avg Q1 |
|---|---|---|---|---|
| 1 | Sydney | 116.3 | +51.0 | 29.3 |
| 2 | Hawthorn | 105.1 | +20.7 | 31.3 |
| 3 | Gold Coast | 103.3 | +13.0 | 26.7 |
| 4 | Brisbane Lions | 102.7 | +16.3 | 20.3 |
| 5 | Melbourne | 99.6 | +2.7 | 17.0 |

League distribution of per-game team scores: mean **90.4**, std 26.0, p10 60 / p50 90 / p90 126, min 35 / max 163.

##### Winning margin

| Rank | Team | Avg margin | Avg score |
|---|---|---|---|
| 1 | Sydney | +51.0 | 116.3 |
| 2 | Fremantle | +25.1 | 92.0 |
| 3 | Hawthorn | +20.7 | 105.1 |
| 4 | Brisbane Lions | +16.3 | 102.7 |
| 5 | North Melbourne | +13.3 | 96.9 |

League distribution of margins (signed, per team-game): mean ~0 by construction, std 44.7, p10 -58 / p50 0 / p90 58.

##### First-quarter score

| Rank | Team | Avg Q1 score | Avg full-game score |
|---|---|---|---|
| 1 | Hawthorn | 31.3 | 105.1 |
| 2 | Sydney | 29.3 | 116.3 |
| 3 | North Melbourne | 27.6 | 96.9 |
| 4 | Carlton | 26.9 | 80.7 |
| 5 | Gold Coast | 26.7 | 103.3 |

League distribution of Q1 scores: mean **22.4**, std 12.1, p10 8 / p50 21 / p90 40.

#### Going deeper with this repo's models

For the stats above, three artefacts in this repo will help you form your own view rather than just reading a leaderboard:

1. The **disposal prediction model** (`prediction.py` / `prediction_cpu.py`) forecasts a player's next-round disposal count using rolling form, opponent context and venue effects. Run it with `--player surname_first --rounds 1` to see how uncertainty is quantified for any of the leaders shown above.
2. The **backtest framework** (`backtest.py`) replays a season round-by-round so you can see how the model performed on real, out-of-sample games — the honest way to judge whether a leaderboard ranking will continue to hold.
3. The **Brownlow proxy section** above is the same per-game stat structure used here, weighted into a single composite. If you want a quick "who's having the best year overall" answer rather than per-stat leaders, that table is the one to look at.
<!-- 2026-STAT-LEADERS-END -->



### Team playing styles — 5 years of data (2021–2025)

<!-- 5YEAR-TEAM-PROFILES-START -->
The profiles below summarise each team's 2021–2025 statistical fingerprint — five full seasons of per-game averages aggregated across every player who pulled on the jumper. They reflect coaching philosophy and list system more than any single season's results, smoothing out the noise of a hot run or a flat year. If you want to understand *why* a team plays the way it does in 2026, this is the better baseline than the season-to-date snapshot above — it is what the data says they are at their core.

Each paragraph leans on the actual numbers (per-game averages, league ranks across 18 teams, and 5-year linear trends), so the descriptions update automatically when the window rolls forward.

![5-year team playing styles scatter — 2021-2025](assets/charts/team_2026_style_scatter.png)

The scatter above places each club on two axes that capture the most visible part of a team's identity — handball ratio (% of disposals by hand) on the X and tackles per game on the Y. The dashed lines mark the league median, splitting the field into four loose archetypes: top-right teams move the ball by handball *and* hunt with tackle pressure; top-left sides trust their kicking but still bring the heat; bottom-right are handball teams that spread rather than tackle; bottom-left are kick-and-spread possession sides.

#### Adelaide

Adelaide have built their identity around a ground-ball pressure game — ranked 1st of 18 across the last five seasons. Across 2021–2025 they averaged 356.3 disposals, 63.9 tackles and 38.2 clearances per game, with a roughly balanced 41/59 handball-to-kick split. On territory and contest, they show a mid-pack territory profile (9th for inside-50s) combined with a contested-ball, high-pressure identity (2nd for contested possessions, 1st for tackles). Forward of centre, a volume-over-precision forward template — only 1-in-4.6 entries are marked inside 50 (14th for connection), and ball security is a clear vulnerability — sitting in the bottom four for clangers or frees conceded. The profile is gradually shifting rather than locked in: forward connection has climbed ~12% across the five-year window and handball share has fallen ~11% across the five-year window. What separates them from the rest of the competition is the floor under their marks (83.8/g, 18th of 18) — a structural issue, not a single bad year. In short, the ceiling is anchored to their tackles (1st of 18), while the vulnerability is their marks (18th of 18) — and that gap is what every opposition gameplan targets.

#### Brisbane Lions

Brisbane Lions have built their identity around a territory-dominant game — ranked 1st of 18 across the last five seasons. Across 2021–2025 they averaged 354.3 disposals, 56.5 tackles and 40.4 clearances per game, with a kick-first, territory-via-foot template (64% by foot — 1st-most kick-dominant). On territory and contest, they show a relentlessly forward-territory game (+18.8 net 50-entries per game) combined with a contested-ball game without the tackle pressure to match (5th for CP, only 15th for tackles). Forward of centre, mid-pack forward efficiency (8th for marks-per-inside-50), and ball security sits roughly mid-pack. The 5-year profile is gradually shifting — modest year-on-year movement but no wholesale re-tooling of the game plan. What sets them apart from the field is their kicks (227.4/g, 1st of 18) — a true outlier rather than mid-pack noise. In short, the ceiling is anchored to their kicks (1st of 18), while the vulnerability is their tackles (15th of 18) — and that gap is what every opposition gameplan targets.

#### Carlton

Carlton's 5-year profile is defined by what they don't do — the least ruck-led side in the competition. Across 2021–2025 they averaged 365.1 disposals, 59.5 tackles and 37.7 clearances per game, with a roughly balanced 40/60 handball-to-kick split. On territory and contest, they show a mid-pack territory profile (8th for inside-50s) combined with a balanced contest profile (3rd for CP, 10th for tackles). Forward of centre, mid-pack forward efficiency (13th for marks-per-inside-50), and ball security sits roughly mid-pack. The 5-year profile is gradually shifting — modest year-on-year movement but no wholesale re-tooling of the game plan. What sets them apart is how far they sit from the league norm in hit-outs (32.5/g, 18th of 18) — that gap is the signature of their style, not a fault line. In short, the ceiling is anchored to their contested marks (1st of 18); the low hit-outs (18th of 18) is the deliberate trade-off that comes with that style.

#### Collingwood

Collingwood's 5-year profile is defined by what they don't do — the third-least mark-and-control side in the competition. Across 2021–2025 they averaged 352.0 disposals, 61.2 tackles and 35.0 clearances per game, with a roughly balanced 41/59 handball-to-kick split. On territory and contest, they show a struggle to win territory (15th for inside-50s) combined with a balanced contest profile (14th for CP, 5th for tackles). Forward of centre, a clinical use of the few entries they win (1-in-4.3 entries marked, 3rd in the league), and ball security is a clear vulnerability — sitting in the bottom four for clangers or frees conceded. The profile is evolving rather than locked in: ball use has fallen ~13% across the five-year window and forward connection has climbed ~11% across the five-year window. What separates them from the rest of the competition is the floor under their marks (88.4/g, 16th of 18) — a structural issue, not a single bad year. In short, the ceiling is anchored to their marks-per-inside-50 conversion (3rd of 18), while the vulnerability is their marks (16th of 18) — and that gap is what every opposition gameplan targets.

#### Essendon

Essendon have built their identity around a possession-and-control game — ranked 1st of 18 across the last five seasons. Across 2021–2025 they averaged 370.6 disposals, 56.5 tackles and 34.3 clearances per game, with a high-tempo, handball-heavy build-up (42% by hand — 3rd in the league). On territory and contest, they show a mid-pack territory profile (11th for inside-50s) combined with a possession-and-spread template that avoids the contest (1st for uncontested possessions, 16th for CP). Forward of centre, mid-pack forward efficiency (6th for marks-per-inside-50), and ball security is a real strength — among the cleanest sides for both clangers and frees against. The 5-year profile is gradually shifting — modest year-on-year movement but no wholesale re-tooling of the game plan. What sets them apart from the field is their uncontested possessions (234.6/g, 1st of 18) — a true outlier rather than mid-pack noise. In short, the ceiling is anchored to their uncontested possessions (1st of 18), while the vulnerability is their tackles (16th of 18) — and that gap is what every opposition gameplan targets.

#### Fremantle

Fremantle have built their identity around a handball-driven game — ranked 1st of 18 across the last five seasons. Across 2021–2025 they averaged 362.8 disposals, 57.7 tackles and 37.8 clearances per game, with a high-tempo, handball-heavy build-up (43% by hand — 1st in the league). On territory and contest, they show a mid-pack territory profile (10th for inside-50s) combined with a possession-and-spread template that avoids the contest (5th for uncontested possessions, 13th for CP). Forward of centre, a volume-over-precision forward template — only 1-in-4.7 entries are marked inside 50 (16th for connection), and ball security sits roughly mid-pack. The profile is gradually shifting rather than locked in: tackle pressure has climbed ~16% across the five-year window. What sets them apart from the field is their handball share of disposals (43.1%, 1st of 18) — a true outlier rather than mid-pack noise. In short, the ceiling is anchored to their handball share of disposals (1st of 18), while the vulnerability is their marks-per-inside-50 conversion (16th of 18) — and that gap is what every opposition gameplan targets.

#### Geelong

Geelong have built their identity around a clinical forward-50 game — ranked 1st of 18 across the last five seasons. Across 2021–2025 they averaged 361.1 disposals, 60.7 tackles and 37.7 clearances per game, with a kick-first, territory-via-foot template (61% by foot — 4th-most kick-dominant). On territory and contest, they show a relentlessly forward-territory game (+17.8 net 50-entries per game) combined with a balanced contest profile (9th for CP, 7th for tackles). Forward of centre, genuine forward-half efficiency (1-in-3.8 of their entries hit a target inside 50, and they average 13.8 goals/g), and ball security sits roughly mid-pack. The profile is evolving rather than locked in: tackle pressure has climbed ~14% across the five-year window and forward connection has climbed ~12% across the five-year window. What sets them apart from the field is their goals (13.8/g, 1st of 18) — a true outlier rather than mid-pack noise. In short, the ceiling is anchored to their goals (1st of 18); the low rebound-50s (17th of 18) is the deliberate trade-off that comes with that style.

#### Gold Coast

Gold Coast's 5-year profile is defined by what they don't do — the least possession-and-control side in the competition. Across 2021–2025 they averaged 344.6 disposals, 60.3 tackles and 37.7 clearances per game, with a kick-first, territory-via-foot template (62% by foot — 2nd-most kick-dominant). On territory and contest, they show strong territorial dominance (5th for inside-50s) combined with a balanced contest profile (6th for CP, 8th for tackles). Forward of centre, a volume-over-precision forward template — only 1-in-5.0 entries are marked inside 50 (18th for connection), and ball security sits roughly mid-pack. The profile is evolving rather than locked in: handball share has climbed ~25% across the five-year window and forward entries has climbed ~12% across the five-year window. What sets them apart is how far they sit from the league norm in uncontested possessions (202.5/g, 18th of 18) — that gap is the signature of their style, not a fault line. In short, the ceiling is anchored to their tackles-per-disposal pressure (2nd of 18), while the vulnerability is their marks-per-inside-50 conversion (18th of 18) — and that gap is what every opposition gameplan targets.

#### Greater Western Sydney

Greater Western Sydney have built their identity around a rebound-and-counter game — ranked 1st of 18 across the last five seasons. Across 2021–2025 they averaged 368.9 disposals, 61.6 tackles and 36.5 clearances per game, with a roughly balanced 42/58 handball-to-kick split. On territory and contest, they show a defensive-rebound posture, soaking up entries and breaking out (rebound-50s ranked 1st) combined with genuine ground-ball pressure (3rd for tackles). Forward of centre, mid-pack forward efficiency (10th for marks-per-inside-50), and ball security sits roughly mid-pack. The profile is evolving rather than locked in: forward connection has climbed ~24% across the five-year window. What sets them apart from the field is their rebound-50s (42.2/g, 1st of 18) — a true outlier rather than mid-pack noise. In short, the ceiling is anchored to their rebound-50s (1st of 18), while the vulnerability is their free kicks for (17th of 18) — and that gap is what every opposition gameplan targets.

#### Hawthorn

Hawthorn have built their identity around a handball-driven game — ranked 2nd of 18 across the last five seasons. Across 2021–2025 they averaged 369.1 disposals, 59.6 tackles and 35.1 clearances per game, with a high-tempo, handball-heavy build-up (43% by hand — 2nd in the league). On territory and contest, they show a mid-pack territory profile (13th for inside-50s) combined with a balanced contest profile (12th for CP, 9th for tackles). Forward of centre, mid-pack forward efficiency (9th for marks-per-inside-50), and ball security sits roughly mid-pack. The profile is evolving rather than locked in: forward connection has climbed ~30% across the five-year window and forward entries has climbed ~10% across the five-year window. What sets them apart from the field is their handballs (158.2/g, 1st of 18) — a true outlier rather than mid-pack noise. In short, the ceiling is anchored to their handballs (1st of 18) — that's the foundation everything else is built on.

#### Melbourne

Melbourne have built their identity around a contested-ball game — ranked 1st of 18 across the last five seasons. Across 2021–2025 they averaged 362.6 disposals, 59.0 tackles and 36.8 clearances per game, with a roughly balanced 41/59 handball-to-kick split. On territory and contest, they show a relentlessly forward-territory game (+16.0 net 50-entries per game) combined with a balanced contest profile (1st for CP, 12th for tackles). Forward of centre, a volume-over-precision forward template — only 1-in-4.6 entries are marked inside 50 (15th for connection), and ball security sits roughly mid-pack. The profile is gradually shifting rather than locked in: contested-ball winning has fallen ~13% across the five-year window and forward entries has fallen ~10% across the five-year window. What sets them apart from the field is their contested possessions (141.8/g, 1st of 18) — a true outlier rather than mid-pack noise. In short, the ceiling is anchored to their contested possessions (1st of 18), while the vulnerability is their marks-per-inside-50 conversion (15th of 18) — and that gap is what every opposition gameplan targets.

#### North Melbourne

North Melbourne's 5-year profile is defined by what they don't do — the second-least contested-ball side in the competition. Across 2021–2025 they averaged 351.5 disposals, 57.1 tackles and 36.6 clearances per game, with a roughly balanced 42/58 handball-to-kick split. On territory and contest, they show a defensive-rebound posture, soaking up entries and breaking out (rebound-50s ranked 2nd) combined with a balanced contest profile (17th for CP, 14th for tackles). Forward of centre, a volume-over-precision forward template — only 1-in-4.8 entries are marked inside 50 (17th for connection), and ball security is a clear vulnerability — sitting in the bottom four for clangers or frees conceded. The profile is gradually shifting rather than locked in: tackle pressure has climbed ~12% across the five-year window and handball share has climbed ~8% across the five-year window. What separates them from the rest of the competition is the floor under their marks inside 50 (9.6/g, 18th of 18) — a structural issue, not a single bad year. In short, the ceiling is anchored to their rebound-50s (2nd of 18), while the vulnerability is their marks inside 50 (18th of 18) — and that gap is what every opposition gameplan targets.

#### Port Adelaide

Port Adelaide have built their identity around a clinical forward-50 game — ranked 2nd of 18 across the last five seasons. Across 2021–2025 they averaged 358.0 disposals, 61.5 tackles and 37.4 clearances per game, with a roughly balanced 40/60 handball-to-kick split. On territory and contest, they show a mid-pack territory profile (7th for inside-50s) combined with genuine ground-ball pressure (4th for tackles). Forward of centre, genuine forward-half efficiency (1-in-4.2 of their entries hit a target inside 50, and they average 12.0 goals/g), and ball security is a clear vulnerability — sitting in the bottom four for clangers or frees conceded. The profile is evolving rather than locked in: contested-ball winning has fallen ~14% across the five-year window and ball use has fallen ~12% across the five-year window. What sets them apart from the field is their marks-per-inside-50 conversion (23.6%, 2nd of 18) — a true outlier rather than mid-pack noise. In short, the ceiling is anchored to their marks-per-inside-50 conversion (2nd of 18), while the vulnerability is their free kicks against (17th of 18) — and that gap is what every opposition gameplan targets.

#### Richmond

Richmond's 5-year profile is defined by what they don't do — the least ground-ball pressure side in the competition. Across 2021–2025 they averaged 348.4 disposals, 53.5 tackles and 33.2 clearances per game, with a roughly balanced 41/59 handball-to-kick split. On territory and contest, they show a defensive-rebound posture, soaking up entries and breaking out (rebound-50s ranked 3rd) combined with notably soft pressure around the ball (18th for tackles). Forward of centre, mid-pack forward efficiency (7th for marks-per-inside-50), and ball security is a clear vulnerability — sitting in the bottom four for clangers or frees conceded. The profile is evolving rather than locked in: forward entries has fallen ~19% across the five-year window and forward connection has fallen ~15% across the five-year window. What separates them from the rest of the competition is the floor under their tackles (53.5/g, 18th of 18) — a structural issue, not a single bad year. In short, the ceiling is anchored to their rebound-50s (3rd of 18), while the vulnerability is their tackles (18th of 18) — and that gap is what every opposition gameplan targets.

#### St Kilda

St Kilda have built their identity around a mark-and-control game — ranked 2nd of 18 across the last five seasons. Across 2021–2025 they averaged 371.7 disposals, 61.0 tackles and 35.0 clearances per game, with a roughly balanced 39/61 handball-to-kick split. On territory and contest, they show a struggle to win territory (16th for inside-50s) combined with a balanced contest profile (11th for CP, 6th for tackles). Forward of centre, a clinical use of the few entries they win (1-in-4.4 entries marked, 4th in the league), and ball security is a clear vulnerability — sitting in the bottom four for clangers or frees conceded. The 5-year profile is gradually shifting — modest year-on-year movement but no wholesale re-tooling of the game plan. What sets them apart from the field is their kicks (225.2/g, 2nd of 18) — a true outlier rather than mid-pack noise. In short, the ceiling is anchored to their kicks (2nd of 18), while the vulnerability is their inside-50s (16th of 18) — and that gap is what every opposition gameplan targets.

#### Sydney

Sydney have built their identity around a ground-ball pressure game — ranked 2nd of 18 across the last five seasons. Across 2021–2025 they averaged 358.5 disposals, 62.2 tackles and 36.8 clearances per game, with a kick-first, territory-via-foot template (61% by foot — 3rd-most kick-dominant). On territory and contest, they show strong territorial dominance (6th for inside-50s) combined with genuine ground-ball pressure (2nd for tackles). Forward of centre, mid-pack forward efficiency (11th for marks-per-inside-50), and ball security is a clear vulnerability — sitting in the bottom four for clangers or frees conceded. The 5-year profile is settled — modest year-on-year movement but no wholesale re-tooling of the game plan. What sets them apart from the field is their tackles (62.2/g, 2nd of 18) — a true outlier rather than mid-pack noise. In short, the ceiling is anchored to their tackles (2nd of 18), while the vulnerability is their contested marks (17th of 18) — and that gap is what every opposition gameplan targets.

#### West Coast

West Coast's 5-year profile is defined by what they don't do — the least contested-ball side in the competition. Across 2021–2025 they averaged 333.1 disposals, 56.3 tackles and 34.2 clearances per game, with a roughly balanced 40/60 handball-to-kick split. On territory and contest, they show a struggle to win territory (18th for inside-50s) combined with notably soft pressure around the ball (17th for tackles). Forward of centre, mid-pack forward efficiency (12th for marks-per-inside-50), and ball security sits roughly mid-pack. The profile is evolving rather than locked in: handball share has climbed ~14% across the five-year window and ball use has fallen ~11% across the five-year window. What separates them from the rest of the competition is the floor under their kicks (201.4/g, 18th of 18) — a structural issue, not a single bad year. In short, the ceiling is anchored to their clangers (3rd of 18), while the vulnerability is their kicks (18th of 18) — and that gap is what every opposition gameplan targets.

#### Western Bulldogs

Western Bulldogs' 5-year profile is defined by what they don't do — the least rebound-and-counter side in the competition. Across 2021–2025 they averaged 374.0 disposals, 59.2 tackles and 40.5 clearances per game, with a high-tempo, handball-heavy build-up (42% by hand — 4th in the league). On territory and contest, they show a relentlessly forward-territory game (+19.4 net 50-entries per game) combined with a balanced contest profile (4th for CP, 11th for tackles). Forward of centre, mid-pack forward efficiency (5th for marks-per-inside-50), and ball security is a real strength — among the cleanest sides for both clangers and frees against. The profile is unusually settled — none of the six identity stats has shifted more than ~2% across five seasons, suggesting a tightly held list and game-plan. What sets them apart from the field is their disposals (374.0/g, 1st of 18) — a true outlier rather than mid-pack noise. In short, the ceiling is anchored to their disposals (1st of 18); the low rebound-50s (18th of 18) is the deliberate trade-off that comes with that style.
<!-- 5YEAR-TEAM-PROFILES-END -->

### For the footy expert — finding the greatest 100 players of all time

If you live and breathe footy — you can rattle off Lockett's goals tally, you've argued about Carey vs. Matthews more times than you can count, you have strong opinions about whether Bontempelli is already a top-20 player of all time — but you've never opened a terminal in your life, this section is for you.

This repo gives you something you can't get anywhere else: a ranking formula you can actually **change, challenge, and re-run yourself** — without writing a single line of code. You just talk to an AI agent (Claude) in plain English, the same way you'd argue at the pub. You ask "why is Carey ranked above Matthews?" and it tells you. You say "I reckon goals should count for more — show me what happens" and it changes the formula and re-runs it.

You don't need to know what Python is. You don't need to understand machine learning. You just need to be opinionated about footy.

<!-- TOP10-CHART-START -->
![All-time top 10 AFL/VFL players](assets/charts/top10_alltime.png)
<!-- TOP10-CHART-END -->

---

#### What the ranking is actually doing (in footy terms)

The hardest problem in ranking all-time greats is that **you can't just add up stats**. Tony Lockett kicked 1,360 goals — but in his era, contested possessions weren't even tracked. Haydn Bunton Sr. won three Brownlows in the 1930s when the only stats kept were goals and behinds. Compare that to Patrick Cripps, who has 20+ different stats logged every single game.

If you just totted up everything, modern players would crush every list — not because they were better, but because more was being counted.

So the formula does three things to make it fair:

**1. It only uses the stats that existed at the time.** Bunton's score uses goals and behinds. Bartlett's uses kicks, handballs, marks, and goals. Bontempelli's uses everything modern, including contested possessions and clearances. Nobody is penalised for not having a stat that didn't exist yet.

**2. It groups players by position type so you're comparing apples to apples.** A key forward like Lockett or Dunstall will always pile up more raw "score" than a midfielder like Pendlebury — because goals are weighted heavily and forwards kick more of them. So the ranking splits players into three buckets:

- **Key forwards** (3+ goals per game) — Lockett, Dunstall, Ablett Sr, Lloyd, Franklin, Brown
- **Forward-midfielders** (0.8 to 2.99) — Carey, Matthews, Bartlett, Dangerfield, Ablett Jr, Dustin Martin
- **Midfielders / defenders** (under 0.8) — Pendlebury, Neale, Cripps, Parker

A midfielder who finishes #1 in their group is genuinely the best midfielder of the era — not just "good but not as good as Lockett."

**3. It rewards both sustained excellence and peak greatness.** The final score is the **average of a player's best 8 seasons**, plus a bonus for very long careers (capped, so a 15-year average career can't beat a 10-year brilliant one), plus a bonus for having a season where you were clearly the best player in the league. The minimum to be ranked at all is 150 games — that filters out brief careers, no matter how brilliant.

Unsure why the best 8? Because using only the best 5 let players with 2 freak years rank too high. Using the whole career punished anyone who hung on too long. Best 8 is the sweet spot — long enough to reward consistency, short enough to ignore decline years.

---

#### How to challenge the ranking by talking to Claude

Once you've followed the [Setting up Claude Code on Ubuntu](docs/ai-agents.md#setting-up-claude-code-on-ubuntu) steps, you'll have a thing called Claude running in your terminal. **Forget what "terminal" means** — just think of it as a window where you type questions to a very smart footy analyst who has read every line of code in this project and every player's stats.

To start it up:

```bash
cd /path/to/SuperCoach-VIA
claude
```

You'll see a `>` prompt. That's where you type. Hit Enter to send. Claude reads everything in this project and answers in plain English. If it needs to change something or run something, it'll tell you what it's about to do first.

Here's what conversations actually look like.

##### Asking why a player ranked where they did

```
Why is Wayne Carey ranked above Leigh Matthews on the all-time list?
```

Claude will open `all_time_top_100.csv`, look up both players' scores, see which group (forward-midfielders) they're both in, and explain the breakdown — Carey's per-season scores, Matthews' per-season scores, the era adjustment applied to each, the peak bonus, the longevity bonus. You'll get a paragraph back like "Carey's average top-8 season scored higher because the formula weights kicks and marks heavily and Matthews played in an era where marks weren't tracked until 1991 — he gets credit for goals and disposals but the formula can't see his contested marks. Try this: tell me to lower the weight on marks and we'll see if Matthews moves up."

##### Asking where a specific player ranks

```
Where does Scott Pendlebury rank and why?
Where does Gary Ablett Jr finish and what's his peak season according to the formula?
Why didn't Lance Franklin make the top 10?
```

For any of these, Claude will pull the actual numbers and explain. If a player isn't on the list, Claude will tell you exactly where they fell short — not enough games, no peak Brownlow-level season, or their group ranking just wasn't high enough.

##### Challenging the formula and re-running it

This is where it gets fun. You can change anything about the formula and see what happens.

```
I think goals should count for more in the modern era — increase the goal weight by 30% and re-run the ranking. Show me how the top 20 changes.
```

```
The formula uses the best 8 seasons. I think it should be best 10 — too many short-career players are sneaking in. Change it and re-run.
```

```
Drop the minimum games requirement from 150 to 100 and tell me which players newly make the list.
```

```
The career longevity bonus is capped at 30%. I think Pendlebury and Bartlett are getting underrated for their 350+ games — raise the cap to 40% and show me what shifts.
```

Claude will make the change in the code, re-run the ranking, save the new top 100, and tell you which players moved up, which moved down, and which dropped off entirely. If you don't like it, just say "revert that change" and you're back to the original.

##### Asking Claude to explain the era adjustments

```
Explain the era adjustment in plain English — why does a 1985 season score differently to a 2015 season for the same raw stats?
```

```
Walk me through how Haydn Bunton's 1933 season is scored. He won the Brownlow but I don't see a "Brownlow" stat in the formula — how does the model recognise his greatness?
```

```
The formula caps any single stat at 55% of a season's score. What does that actually do? Show me a player whose ranking changed because of that cap.
```

---

#### Using the Scientist agent for deeper questions

Claude (the regular one) is great for opinion arguments and quick formula tweaks. But for **proper number-crunching** — running statistical sensitivity analyses, checking whether changes are real or just noise, comparing rankings across different formula versions — there's a heavier-duty version called the **Scientist agent**.

> **READ THE DISCLAIMER** in the [STOP. READ THIS FIRST.](docs/ai-agents.md) section above before invoking the Scientist. It runs on Claude's most expensive model and burns through tokens fast. **Use it for the questions that actually require it. For everything else, just use plain Claude.**

You invoke the Scientist by typing `@"Scientist (agent)"` at the start of your message:

```
@"Scientist (agent)" double the weight on contested possessions in the ranking formula and tell me how the top 10 changes. I want a proper sensitivity analysis — not just the new list, but how confident we should be that the changes are meaningful versus formula noise.
```

```
@"Scientist (agent)" the ranking puts Wayne Carey at #1. Run a sensitivity analysis — try 5 different reasonable variations of the formula (different season-count, different group thresholds, different era adjustments) and tell me whether Carey stays #1 in all of them or whether his top spot is fragile.
```

```
@"Scientist (agent)" how much does the top 20 actually change if I remove the era adjustment entirely? Is the era adjustment doing real work or is it cosmetic?
```

```
@"Scientist (agent)" the formula gives a peak-season bonus. Strip it out and re-rank — which players were riding on one freakish season versus genuine sustained excellence?
```

```
@"Scientist (agent)" simulate what the top 100 would look like if pre-1965 players had access to the modern stats — i.e. fill in the missing stats with reasonable estimates based on similar modern players, and show me how Bunton, Coleman and Whitten move.
```

The Scientist will read the code, do the proper analysis, and report back with not just the answer but **how confident you should be in it** — which is what separates a real analysis from just another opinion.

---

#### Questions worth asking — for stress-testing the formula

If you're a footy expert who wants to genuinely challenge whether this ranking holds up, here are the questions that will tell you the most. You can paste any of these straight into Claude.

**Era fairness**

- "Show me the top 5 players from each decade in this list. Are pre-1970 players underrepresented? If so, is that a real reflection of quality or a flaw in the formula?"
- "The best 8 seasons average rewards longevity. Most pre-WWII players had shorter careers due to the war and shorter seasons. Has the formula adjusted for that, or are players like Dyer and Bunton being penalised?"
- "Modern players have GPS distance, defensive pressure acts, and other stats that aren't in this dataset. The formula scales them down to compensate — but is the scale-down enough? Try doubling it and show me what happens."

**Position group fairness**

- "There are 3 position groups. What if I split key forwards into 'true power forwards' (Lockett, Dunstall, Coleman) and 'lead-up forwards' (Franklin, Ablett Sr)? Does the ranking change in a way that feels right?"
- "Pendlebury, Neale and Cripps are in the same group as 1960s-era pure defenders. Does that make sense? Test what happens if we split midfielders out from defenders."
- "Carey is classified as a forward-midfielder. Move him into the key forwards group and re-rank. Does that change his position?"

**Stat weighting**

- "Goals dominate forwards' scores. What's the actual weight on goals in the formula, and what happens if I halve it? Do midfielders take over the top of the list?"
- "Show me what happens to the top 20 if I weight contested possessions twice as heavily as kicks. The argument is that contested ball is harder, so should count more."
- "Brownlow votes aren't used as a stat. Should they be? Add them as a 10% weighting and re-rank."

**Peak vs longevity**

- "Tony Lockett's career peak was higher than Buddy Franklin's, but Franklin played longer. The formula uses best 8 seasons + a longevity bonus. Show me what each player gets from each component."
- "Cut the longevity bonus to zero. Who falls off the top 50? Are those players great because of who they were, or because they hung on?"
- "Use only best 3 seasons (peak only — no longevity at all). Who rises to the top?"

**Specific player debates**

- "Dustin Martin won 3 Norm Smiths and a Brownlow. Where does the formula put him, and is he being properly credited for finals heroics? (The dataset is regular-season only — flag that as a limitation.)"
- "Gary Ablett Sr vs Gary Ablett Jr — which one ranks higher and what's the gap? Walk me through the breakdown."
- "Lachie Neale has two Brownlows but doesn't crack the top 30. Is that a fair reflection or is the formula missing something about his game?"
- "Bontempelli is still active. How does the formula treat current players whose best seasons might still be ahead of them?"

**Structural questions**

- "The minimum is 150 games. How many genuinely great careers does that exclude — players who were brilliant but injured early, or war-shortened careers like Dick Reynolds-era players?"
- "The list guarantees one player per decade. Is the 1900s rep just there because they had to put someone — or are they genuinely top-100 quality on the merits?"
- "What's the standard deviation of scores in the top 100? If the gap between #1 and #50 is small, the rankings within those 50 are basically noise. If it's big, the order is meaningful."

---

#### One last thing

The whole point of this is that **you can argue with the formula and win**. If you genuinely believe Leigh Matthews is the greatest of all time and the formula has him at #4 — say so to Claude, ask why, and challenge the assumptions. If your argument holds up, change the formula. If your changes produce a list you can defend at the pub, push them to the repo (just type `push the changes to main`).

This isn't a list handed down from on high. It's a starting point for the argument — and now you've got the tools to make your case with actual numbers behind it.

### For the coaching staff — building a data-driven game plan

You're an assistant coach, performance analyst, or fitness staff member at Richmond. It's Monday morning, the senior coach has just confirmed Friday night's opponent — West Coast at the MCG — and your week is now about one question: **how do we beat them?**

You already know what your eyes tell you from the last three weeks of vision. You've got the GPS reports, the contested-ball numbers from the last game, the medical list. What you don't always have on tap is **130 years of structured match and player data**, sliced however you want, with a footy-literate analyst who'll work through the night without asking for a coffee.

That's what this repo plus a Claude Code session gives you. Every Richmond–West Coast result ever played, every disposal Elliot Yeo has ever logged, every game Harley Reid has been kept under 18 touches in, every clearance Yeo has won when his side has played away from Perth — all queryable in plain English. The Scientist agent on top of that does the heavier lifting: opponent-split regressions, form-curve analysis, matchup history, sensitivity tests on which factors actually move the needle.

This section is for the staff member who knows the game inside out but wants the numbers to back the call before walking into the Tuesday review meeting.

---

#### What's in the data (and what isn't)

**You have:**

- Every Richmond vs West Coast match ever played — venue, margin, quarter-by-quarter scores, weather where recorded
- Every West Coast player's per-game record: kicks, handballs, marks, disposals, goals, behinds, tackles, clearances, contested possessions, hit-outs, frees for/against, Brownlow votes — by round, by year, by opponent
- The same for every Richmond player — recent form (last 5 rounds), season averages, career history, splits by opponent and venue
- Historical opponent splits — how a player or team performs at home vs away, against specific opposition, in specific rounds, across decades
- Disposal predictions for the upcoming round (`prediction.py`) — useful as a sanity check on form or for comparing Richmond vs West Coast personnel projections

**You don't have (be honest with yourself and the head coach):**

- **No GPS, no heart-rate, no high-speed running data** — the dataset is box-score only
- **No positional/spatial data** — you can't ask "where on the ground does Harley Reid win his ball" or "what's West Coast's average kick length inside 50"
- **No video tags** — pressure acts, spoils, intercept marks, defensive one-percenters before 2011 aren't here
- **No injury/availability data** — you'll need to overlay your own medical and team list on top
- **No live odds, weather forecasts, or in-game state** — historical match-day weather is patchy

If a question requires any of the above, the answer here is "this dataset can't tell you that — go to your video department or your GPS provider." That honesty matters when you're in front of the senior coach.

---

#### The workflow — Monday to Thursday with Claude

Below is a six-step workflow you can run end-to-end in a single Claude Code session. Each step has copy-pasteable prompts. Plain Claude handles most of it; the Scientist gets pulled in only for the questions that genuinely require analytical depth.

To start a session:

```bash
cd /path/to/SuperCoach-VIA
claude
```

Then work through the steps below.

---

##### Step 1 — Understand West Coast's current form

Before you talk personnel, get a read on the team. Are they trending up or down? Is their midfield winning the ball at expected rates? Are their forwards converting?

```
Pull the last 5 rounds of West Coast's results — opponents, margins, venues. For each game, tell me their total disposals, contested possessions, clearances, goals, and inside-50 count if it's in the data.
```

```
For each West Coast player who's played at least 3 of the last 5 rounds, give me their average disposals, goals, and tackles over that window, plus how that compares to their 2026 season average. Flag anyone trending up or down by more than 15%.
```

```
Compare West Coast's last 5 games at home (Optus Stadium) versus their last 5 games on the road. Are there meaningful differences in their disposal count, scoring, or how heavily they lean on specific players?
```

What you're looking for: which West Coast players are in form, which are wobbling, and whether their recent results are noise or signal.

---

##### Step 2 — Identify West Coast's key weapons and how to stop them

Now go individual. Who hurts you most if left free?

```
Rank West Coast's current top 8 ball-winners by 2026 average disposals. For each one, show me their disposal split when they've been tagged versus untagged this year — proxy "tagged" by games where they had under 18 disposals against a top-8 contested-possession side.
```

```
Elliot Yeo — give me a complete profile. Average disposals, contested possessions, clearances and goals this season. His best and worst opponents historically. Any pattern in how often he plays inside vs as a wingman based on his disposal-to-tackle ratio.
```

```
Harley Reid is the one we're most worried about. Pull every game he's played in 2026 and tell me — when his disposals were under 18, who was he matched up against and what was West Coast's margin in those games?
```

```
Jamie Cripps and the small forward group — what's their goal-to-shot ratio this year, and how does it change when West Coast is leading vs trailing at three-quarter time?
```

```
@"Scientist (agent)" build me a quick model on Harley Reid's disposal count this year. Use opponent strength, venue, and rest days as features and tell me which of those actually predicts his output. I want to know whether physically tagging him is worth it or whether his numbers are mostly driven by something else.
```

What you're looking for: a shortlist of 2–4 West Coast players you genuinely need a plan for, and an honest read on whether tagging or zoning each one is supported by their historical splits.

---

##### Step 3 — Identify West Coast's weaknesses to exploit

Every opponent has soft spots. Find them in the numbers.

```
Show me West Coast's defensive record in the last quarter when they've been leading vs trailing this year. Are they bleeding scores when behind, or do they tighten up?
```

```
Across West Coast's 2026 games, where have they conceded the most goals — from forward-50 stoppages, from turnovers in their defensive 50, or from open play? Use shots conceded relative to opposition inside-50 count as the proxy.
```

```
Which West Coast defenders have been getting beaten most often? For each of their backline, show me how many goals their direct opponents kicked over the last 5 rounds.
```

```
West Coast travel — historically over the last 10 years, what's their win rate and average margin when playing in Melbourne in the second half of the season? Compare to their home-Perth record over the same window.
```

```
@"Scientist (agent)" run a proper analysis on West Coast's quarter-by-quarter scoring patterns over the last 3 seasons. Specifically: do they fade in last quarters of away games? I want a real test, not just averages — confidence intervals or a sensitivity check on whether the pattern is noise.
```

What you're looking for: 2–3 structural weaknesses you can build a game plan around — e.g. "they fade late in away games" or "their defensive small Y is conceding 2.3 goals a game over the last month."

---

##### Step 4 — Analyse Richmond's own form and matchup strengths

Now turn the lens on us. Who's hot, who's matched up well, and who's been quiet against this opposition historically?

```
For Richmond's current top 22, give me each player's last 5 rounds — disposals, goals, contested possessions — versus their season average. Who's in form, who's flat?
```

```
Tim Taranto's record against West Coast — every game he's played them, his disposals, goals, and votes. Is he historically a strong performer against this opposition or below par?
```

```
Noah Cumberland — show me his games this year split by where he's played (forward vs midfield, proxy with his CBA proxy of disposal-to-goal ratio). Where has he been most damaging?
```

```
Tom Lynch and Noah Cumberland — head-to-head this year, who's converting better, and which one historically performs better against tall, slow defenders versus quicker undersized ones? (Use opposition key-back age and disposal profile as the proxy.)
```

```
Across all Richmond players in our current squad, which ones have the best historical record against West Coast — by Brownlow votes per game, by goals per game, by disposals per game?
```

What you're looking for: who you want on the ball early, who's a matchup nightmare for West Coast specifically, and who needs a role tweak to be useful Friday.

---

##### Step 5 — Use Scientist to run deeper statistical analysis

This is where you get value out of the heavier model. The plain-Claude steps above answered the "what" questions. The Scientist answers the "is it real" and "how confident should we be" questions — the ones that decide whether you actually change a structure on the back of a number.

> **Cost reminder:** the Scientist runs on Claude Opus and burns tokens hard. Read the [STOP. READ THIS FIRST.](docs/ai-agents.md) section above. Don't @ Scientist for "what's Harley Reid's average" — use it for the questions where the answer changes a coaching call.

```
@"Scientist (agent)" we're considering tagging Elliot Yeo with Marlion Pickett. Look at every game in the last 3 seasons where Yeo was matched against a similar-profile inside midfielder (use age, contested possessions per game, and tackles per game to define similar). Tell me — does tagging actually suppress his output, or does West Coast just shift his role and someone else gains the ball? Give me the answer with proper uncertainty, not a point estimate.
```

```
@"Scientist (agent)" build a matchup model — for our likely 22 versus their likely 22, predict the contested possession differential and the inside-50 differential. Show me which individual matchups contribute most to the projected margin, and which are the most uncertain. I want to know which 1–2 matchups the game probably hinges on.
```

```
@"Scientist (agent)" historical: in Richmond–West Coast games at the MCG over the last 15 years, what's the single biggest predictor of Richmond winning by 20+? Test multiple hypotheses (clearance differential, inside-50 differential, goal-kicking accuracy, contested mark count) and tell me which one actually holds up versus which are just confounded with general team form.
```

```
@"Scientist (agent)" the prediction model in this repo predicts disposal counts. Run it for both squads for this round and tell me — where are Richmond projected to outpoint West Coast, and where are we projected to be outpointed? Treat the predictions with appropriate uncertainty (the 2026 backtest MAE is ~4.1).
```

```
@"Scientist (agent)" Elliot Yeo's clearance work — sensitivity check. Across his career, what conditions predict a high-clearance day for him? I want this with a baseline comparison so I know whether the predictors are real signal or just noise around his season average.
```

What you're looking for: 2–3 statistically defensible insights you can put on a slide and stand behind in a coaches' meeting without getting torn apart on "yeah but is that just because he was playing easier opposition."

---

##### Step 6 — Build the game plan from the findings

Now consolidate. The data has done its job; the coaching judgement is on you.

```
Summarise everything we've found in this session into a one-page brief. Three sections:
1. Threats — top 3 West Coast players we need a plan for, with the data backing each call
2. Opportunities — top 3 West Coast weaknesses we should attack, with how confident we are in each
3. Our matchups — top 3 Richmond personnel decisions, with the matchup data behind each
Be honest about which conclusions are robust and which are exploratory. Flag anything where the dataset can't answer the question.
```

```
Based on the analysis, give me a list of 5 specific coaching questions I should put to the senior coach in tomorrow's meeting — questions where the data raises a structural decision (tagging vs zoning, role change, matchup pick) that needs a human call, not a numerical one.
```

That brief plus your own video and GPS work is what walks into the Tuesday meeting.

---

#### Questions worth asking — the coaching shortlist

Twenty prompts you can paste straight into Claude. Some are plain-Claude jobs; the ones marked **(Scientist)** justify the model upgrade.

**On stopping their ball-winners**

- "How does Harley Reid's disposal count change when he plays against physical, high-tackle midfield opponents versus more outside types? Use opponent average tackles per game as the split."
- "Elliot Yeo's clearance count by venue — is he genuinely better at Optus Stadium than on the road, or is that just opponent-quality noise?" **(Scientist)**
- "What's the optimal tagging target for us — of West Coast's top 5 ball-winners, who has the biggest impact on West Coast's winning margin when they have a big game?" **(Scientist)**

**On their structural patterns**

- "What is West Coast's clearance rate when playing away from Perth? Compare to their home rate over the last 3 seasons."
- "Does West Coast give up more inside 50s in the last quarter when trailing? Quantify it."
- "How does West Coast's contested possession win-rate change in wet weather games? (Use historical match-day weather where available, flag the data gap if not.)"
- "West Coast's record at the MCG in the last 10 years — wins, losses, average margin. Are they worse here than at neutral grounds?" 

**On their weaknesses**

- "Which West Coast defender has conceded the most goals to his direct opponent over the last 5 rounds?"
- "When West Coast loses the contested possession count, what's their win rate?"
- "Is there a quarter where West Coast consistently scores below their per-quarter average? Test it properly." **(Scientist)**

**On our players**

- "Which Richmond players have historically performed best against West Coast — by votes per game, by goals, by disposals?"
- "Tim Taranto's last 8 games against West Coast — disposals, goals, Brownlow votes. Is he a known West Coast performer or has his record been overstated?"
- "Rhyan Mansell at the MCG vs at the smaller grounds — is there a meaningful split in his goals or disposal count?"
- "Predict Richmond's top 5 disposal-getters for this round using prediction.py and tell me how confident the model is in each prediction."

**On matchups**

- "If we put Noah Cumberland on Tom Barrass and Tom Lynch on Jeremy McGovern, what does the historical data say about how those forward types fare against those defender types?"
- "Identify 1 sneaky matchup advantage — a Richmond player whose 2026 form profile (high contested marks, high tackles, etc.) suggests an underrated mismatch against a specific West Coast player." **(Scientist)**

**On meta-questions**

- "If we win the clearance count by 5+, what does history say about Richmond's win rate against West Coast? Watch out for confounding with general form."
- "What's the smallest margin West Coast has lost by when they had Elliot Yeo with 30+ disposals? Are they losable when he plays well?"
- "Backtest the prediction model on the last 3 Richmond–West Coast games — how accurate was it for the West Coast squad? That tells us how much to trust this round's projections."
- "Run a sensitivity check on this whole brief — which of our findings would change if we excluded the COVID-era 2020–2021 seasons from the historical comparisons?" **(Scientist)**

---

#### Honest limits — what to tell the head coach

When you walk into the meeting with this brief, lead with what the dataset can and can't do. It saves arguments.

| Question | Can this dataset answer it? |
|---|---|
| Who hurts us most if left free, by historical impact on margin? | **Yes** — disposal, goal, vote and margin data are all in there |
| How does West Coast travel? | **Yes** — venue splits go back decades |
| Which Richmond player matches up best on Harley Reid? | **Partially** — the data tells you about output suppression; it can't tell you whether your player physically holds up over four quarters |
| Where on the ground does Reid win his ball? | **No** — no spatial data |
| Is Elliot Yeo running at 95% of his peak high-speed metres? | **No** — no GPS data, ask the fitness staff |
| What's our pressure rating in the forward 50 last week? | **No** — no video-tag data |
| What's the predicted margin? | **Indirectly** — we have disposal predictions, not score predictions; build it bottom-up if needed and own the uncertainty |
| Will this play in wet weather? | **Sometimes** — historical match-day weather is patchy and inconsistent before 2010 |

**The rule:** if a question requires GPS, video tags, or positional data, the answer is "this is for the analyst with Champion Data or for the video department." This dataset is for *historical patterns and box-score-driven matchup work*. That's a lot — but it's not everything.

---

#### When to use Scientist vs plain Claude — for coaching questions

Same rule as the rest of the repo: plain Claude handles 80% of this work cheaper and faster. The Scientist is for the questions where you need an *analytically defensible* answer because a coaching decision rides on it.

| Plain Claude | Scientist |
|---|---|
| Pull Harley Reid's last 5 games | Test whether tagging Harley Reid actually suppresses his output or just shifts West Coast's structure |
| Show me Richmond's record at Optus Stadium | Run a sensitivity check on whether the home-ground effect is real or driven by opponent quality |
| List West Coast's top 5 ball-winners this season | Build a matchup model and tell me which 1–2 individual matchups the game probably hinges on |
| Average disposals per quarter for Elliot Yeo | Test whether Yeo's per-quarter pattern is statistically meaningful or within normal player variance |
| What was the score in the last Richmond–West Coast game | Across all Richmond–West Coast games, what's the strongest predictor of a Richmond win that survives controlling for general team form? |

If the answer to "would I make a coaching call on the back of this number?" is yes, use the Scientist. If you're just orienting yourself or pulling raw stats, plain Claude is the right tool.

> **Read the cost disclaimer** in the [STOP. READ THIS FIRST.](docs/ai-agents.md) section before invoking Scientist. On an entry-level Claude plan, three or four Scientist calls is meaningful spend. Be deliberate.

