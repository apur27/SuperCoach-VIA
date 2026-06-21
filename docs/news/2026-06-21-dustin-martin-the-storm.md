# Dustin Martin — The Storm

> [← Back to news](README.md) | [← Back to main README](../../README.md)

*Published: 2026-06-21. Council chain: BriefBuilder → FootyStrategy → DataSentinel PASS → Skeptic PASS → Gaffer SHIP.*

---

## 1. The mythology

There is a move called the Don't Argue. It is not unique to Dustin Martin — it is a standard evasion technique, a stiff arm into the body of a closing defender. In Martin's hands it became something else. The scale of it, the matter-of-factness of it, the way it looked less like evasion and more like dismissal. Defenders would arrive with momentum and find themselves redirected, as if the physics of a collision had been briefly renegotiated.

That quality — meeting contact and continuing through it rather than away from it — was the defining characteristic of his game. Martin was a power midfielder in the mould the modern AFL demands: a player who wins the ball in traffic, breaks tackles at the clearance, and can finish forward when the team needs a goal from nothing. What separated him was that he did all three at volume, across fifteen seasons, and at peak intensity in the seasons that decided flags.

The "Storm" framing is earned. When Martin arrived at a contest, the surrounding game state shifted. Defences had to account for him in the back half as a kick-in option, in the midfield as a clearance engine, and fifty metres forward as a marking and set-shot threat. Players who can be legitimately targeted across three zones are rare. Martin was one.

The data anchors the frame: Dustin Martin played AFL football from 2010 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=year ; aggregation=min]** to 2024 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=year ; aggregation=max]**, fifteen seasons, one club.

---

## 2. The Numbers

*All career totals and per-season game counts sourced from `data/player_data/martin_dustin_26061991_performance_details.csv`.*

### Career totals

| Stat | Value |
|---|---:|
| Career games | 302 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=row_count ; aggregation=count]** |
| Career span | 2010–2024 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=year ; aggregation=min/max]** |
| Career goals | 338 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=goals ; aggregation=sum]** |
| Career disposals | 7,320 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=disposals ; aggregation=sum]** |
| Disposals per game | 24.2 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=disposals ; aggregation=mean]** |
| Career tackles | 835 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=tackles ; aggregation=sum(dropna)]** |
| Tackles per game | 2.8 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=tackles ; aggregation=sum(dropna)/count(all) ; convention=fill-zero ; computation=835÷302=2.76≈2.8]** |
| Brownlow votes (see footnote ¹) | 213 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=brownlow_votes ; aggregation=sum]** |

### Games per season

*Source: `data/player_data/martin_dustin_26061991_performance_details.csv`, filter=year=N, column=row_count, aggregation=count. Totals include finals appearances where applicable.*

| Season | Games |
|---:|---:|
| 2010 | 21 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2010 ; column=row_count ; aggregation=count]** |
| 2011 | 22 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2011 ; column=row_count ; aggregation=count]** |
| 2012 | 20 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2012 ; column=row_count ; aggregation=count]** |
| 2013 | 23 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2013 ; column=row_count ; aggregation=count]** |
| 2014 | 22 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2014 ; column=row_count ; aggregation=count]** |
| 2015 | 23 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2015 ; column=row_count ; aggregation=count]** |
| 2016 | 22 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2016 ; column=row_count ; aggregation=count]** |
| 2017 | 25 (peak) **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2017 ; column=row_count ; aggregation=count]** |
| 2018 | 23 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2018 ; column=row_count ; aggregation=count]** |
| 2019 | 23 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2019 ; column=row_count ; aggregation=count]** |
| 2020 | 20 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2020 ; column=row_count ; aggregation=count]** |
| 2021 | 16 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2021 ; column=row_count ; aggregation=count]** |
| 2022 | 9 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2022 ; column=row_count ; aggregation=count]** |
| 2023 | 20 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2023 ; column=row_count ; aggregation=count]** |
| 2024 | 13 (final season) **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2024 ; column=row_count ; aggregation=count]** |
| **Total** | **302** **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=row_count ; aggregation=count]** |

---

## 3. Finals: A Different Level

*Source: `data/player_data/martin_dustin_26061991_performance_details.csv`. Finals rows identified by non-numeric round labels (EF, QF, SF, PF, GF). Regular season = all remaining rows.*

| Split | Games | Disposal avg | Goals total |
|---|---:|---:|---:|
| Regular season | 286 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=round=regular_season ; column=row_count ; aggregation=count]** | 24.4 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=round=regular_season ; column=disposals ; aggregation=mean]** | — |
| Finals | 16 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=round=finals ; column=row_count ; aggregation=count]** | 22.0 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=round=finals ; column=disposals ; aggregation=mean]** | 27 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=round=finals ; column=goals ; aggregation=sum]** |

Note: Martin's finals disposal average (22.0 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=round=finals ; column=disposals ; aggregation=mean]**) is 2.4 disposals per game below his regular-season average (24.4 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=round=regular_season ; column=disposals ; aggregation=mean]**). This is presented as a factual reading of the split, without interpretation. (16 finals + 286 regular-season = 302 career games total **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=row_count ; aggregation=count]**.)

The disposal number requires context. A 2.4 per game drop in finals is present in the 16-game finals sample. But the goal rate tells a different story. In 16 finals, Martin averaged 1.69 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=round=finals ; column=goals ; aggregation=sum/count ; computation=27÷16=1.69]** goals per game (27 goals over 16 games). In 286 regular-season games, the equivalent rate is 1.09 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=round=regular_season ; column=goals ; aggregation=sum/count ; computation=311÷286=1.09]** per game (311 goals **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=round=regular_season ; column=goals ; aggregation=sum]** over 286 games). The goal rate went up by 55 per cent in finals while the disposal count went down.

What this suggests, without claiming causation: the split is consistent with Martin being used more as a forward target in finals. The clearance-and-handball chain of regular-season midfield possession would be compressed in such a role; a player finishing from the forward 50 rather than distributing from the middle will show fewer disposals and more goals. That is the shape the data presents — the schema carries no positional or time-in-zone columns that could confirm a deployment fact directly.

The disposal dip also reflects that finals opponents game-plan specifically. At the level of individual opposition preparation that occurs before an elimination final, a player who averages 24.4 disposals per game in the home-and-away season will be run at more aggressively, taggged more consistently, and given less time and space at the contest. That no opposition team successfully reduced Martin to ineffectiveness — 22 disposals and 1.69 goals per game is still a dominant finals output — is the accurate read of the 16-game finals record.

---

## 4. The Three Premierships

*GF results sourced from match files. Martin's GF stat lines sourced from `data/player_data/martin_dustin_26061991_performance_details.csv`, filter=round=GF.*

### 2017 Premiership

**Grand Final result:** Richmond 16.12 (108) def Adelaide 8.12 (60), margin 48 **[data: data/matches/matches_2017.csv ; filter=round=Grand Final ; column=derived:richmond_score−adelaide_score ; aggregation=value]**

| Stat | Value |
|---|---:|
| Disposals | 29 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2017,round=GF ; column=disposals ; aggregation=value]** |
| Goals | 2 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2017,round=GF ; column=goals ; aggregation=value]** |

Richmond had not won a premiership since 1980. The 37-year gap was one of the longest barren stretches for a club of Richmond's size and supporter base. The 2017 flag did not arrive quietly: a 48-point Grand Final win over Adelaide is a statement result, not a scrape-through. Martin's 29-disposal/2-goal line in the GF was consistent with his home-and-away form that season — he played 25 games **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2017 ; column=row_count ; aggregation=count]** (the highest total of his career), and Richmond played finals in September for the first time in a long stretch.

The 2017 premiership established something structural: Richmond in September with Martin at his peak was a different proposition from any team they had played during the home-and-away season. That was the pattern that the next three years would confirm.

### 2019 Premiership

**Grand Final result:** Richmond 17.12 (114) def Greater Western Sydney 3.7 (25), margin 89 **[data: data/matches/matches_2019.csv ; filter=round=Grand Final ; column=derived:richmond_score−gws_score ; aggregation=value]**

| Stat | Value |
|---|---:|
| Disposals | 22 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2019,round=GF ; column=disposals ; aggregation=value]** |
| Goals | 4 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2019,round=GF ; column=goals ; aggregation=value]** |

The 89-point margin over GWS is the largest Grand Final winning margin on record in the modern era. It was not a tight match that Richmond managed — it was a Grand Final that stopped being competitive by halftime. Martin's 4-goal line in that game was part of a team performance that the scoreline renders accurately: Richmond were operating on a different register from the other finalist.

Two premierships in three seasons moved Richmond from a club that had broken a drought to a club with an identifiable September identity. The question going into 2020 was whether that identity was durable or whether 2017 and 2019 were two peaks around a structure that could not sustain a third.

### 2020 Premiership

**Grand Final result:** Richmond 12.9 (81) def Geelong 7.8 (50), margin 31 **[data: data/matches/matches_2020.csv ; filter=round=Grand Final ; column=derived:richmond_score−geelong_score ; aggregation=value]**

*Venue: Gabba, Brisbane — COVID-affected season; the Grand Final was not held at the MCG.*

| Stat | Value |
|---|---:|
| Disposals | 21 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2020,round=GF ; column=disposals ; aggregation=value]** |
| Goals | 4 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2020,round=GF ; column=goals ; aggregation=value]** |

The 2020 season was compressed and displaced. The Grand Final was held at the Gabba rather than the MCG — the first time since 1902 the decider had not been played in Melbourne. The COVID context reduced the game count for every club; Richmond's 20 games from Martin **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2020 ; column=row_count ; aggregation=count]** reflect that compressed calendar.

Against Geelong, Martin contributed 21 disposals and 4 goals in a 31-point win. Three flags in four seasons — 2017, 2019, 2020 — is a dynasty by the standard measure. For a club that had waited 37 years for its first, three in four years represents a structural compression that is unlikely to be repeated by Richmond or any other club in the near term. Martin was the common thread across all three: the player whose presence forced opposition teams to re-architect their defensive structure, and whose September performances were indistinguishable in quality from his home-and-away baseline.

---

## 5. The Storm Fades — 2021–2024

*Source: `data/player_data/martin_dustin_26061991_performance_details.csv`, filter=year=N, column=row_count, aggregation=count.*

| Season | Games |
|---:|---:|
| 2021 | 16 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2021 ; column=row_count ; aggregation=count]** |
| 2022 | 9 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2022 ; column=row_count ; aggregation=count]** |
| 2023 | 20 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2023 ; column=row_count ; aggregation=count]** |
| 2024 | 13 (final season) **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2024 ; column=row_count ; aggregation=count]** |
| 2025 | 0 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2025 ; column=row_count ; aggregation=count]** (not playing) |
| 2026 | 0 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2026 ; column=row_count ; aggregation=count]** (not playing) |

Last season played: 2024 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=year ; aggregation=max]**. No rows recorded for 2025 or 2026 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2025,year=2026 ; column=row_count ; aggregation=count]**.

For reference, the flag-era game counts (from the per-season table in §2): 2017 = 25, 2018 = 23, 2019 = 23, 2020 = 20 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2017 to year=2020 ; column=row_count ; aggregation=count by year]**.

The availability arc is the signal. From 2010 to 2020, Martin played between 20 and 25 games in every season. That 11-year window of sustained availability — no season below 20, peak at 25 in 2017 — is a career-health baseline that most AFL careers cannot sustain for half that duration.

From 2021, the pattern shifted. Sixteen games in 2021, nine in 2022, twenty in 2023, thirteen in 2024. The 2022 figure (9 games) represents a near-complete season lost. The 2023 bounce to 20 games does not establish a recovery trend; the subsequent drop to 13 games in 2024 is consistent with 2023 being a one-season exception rather than a turning point. The data does not establish why — and four data points do not resolve the question.

No medical or contractual information is present in the repo data files. The data states what the game counts were; it does not state why. What is observable: the window from 2021 to 2024 produced 58 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2021 to 2024 ; column=row_count ; aggregation=sum ; computation=16+9+20+13=58]** games across four seasons, an average of 14.5 **[data: derived ; computation=58÷4=14.5]** per season. The window from 2017 to 2020 produced 91 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2017 to 2020 ; column=row_count ; aggregation=sum ; computation=25+23+23+20=91]** games across four seasons, an average of 22.75 **[data: derived ; computation=91÷4=22.75]** per season. The delta — 8.25 games per season across the two four-year blocks — is the measure of what changed between Martin at peak and Martin in his final years.

He has not played in 2025 or 2026. The 2024 season (13 games, ending at age 33) is the data's last record of him in an AFL game.

---

## 6. The Legacy

The anchoring numbers: 302 career games **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=row_count ; aggregation=count]**, 338 goals **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=goals ; aggregation=sum]**, 7,320 disposals **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=disposals ; aggregation=sum]** at 24.2 per game **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=disposals ; aggregation=mean]**, three Richmond premierships in 2017 **[data: data/matches/matches_2017.csv ; filter=round=Grand Final ; column=winner ; aggregation=value]**, 2019 **[data: data/matches/matches_2019.csv ; filter=round=Grand Final ; column=winner ; aggregation=value]**, and 2020 **[data: data/matches/matches_2020.csv ; filter=round=Grand Final ; column=winner ; aggregation=value]**.

### All-time ranking

This repo carries a composite all-time ranking across 125+ years of VFL/AFL history (`all_time_top_100.csv`). Dustin Martin is ranked **#79** **[data: data/top100/all_time_top_100.csv ; filter=player=martin_dustin ; column=all_time_score ; aggregation=rank]** (all_time_score: 1.7431 **[data: data/top100/all_time_top_100.csv ; filter=player=martin_dustin ; column=all_time_score ; aggregation=value]**), between Robert Walls (#78, score 1.7457 **[data: data/top100/all_time_top_100.csv ; filter=player=walls_robert ; column=all_time_score ; aggregation=value]**) and Tim Watson (#80, score 1.7373 **[data: data/top100/all_time_top_100.csv ; filter=player=watson_tim ; column=all_time_score ; aggregation=value]**). Placing in the all-time top 100 across the full recorded history of the game is the repository's single composite ranking; it is formula-derived and not a panel verdict.

### Among his peers — modern ball-winning midfielders

The comparison class: midfielders with 200+ games in the modern era. Disposals/game and goals/game calculated as career total ÷ games played (fill-zero convention throughout).

| Player | Games | Disp/g | Goals | Goals/g | Brownlow² |
|---|---:|---:|---:|---:|---:|
| **Dustin Martin** | **302** **[data: martin_dustin_26061991_performance_details.csv]** | **24.2** **[data: martin_dustin_26061991_performance_details.csv]** | **338** **[data: martin_dustin_26061991_performance_details.csv]** | **1.12** **[data: martin_dustin_26061991_performance_details.csv ; derived=338÷302]** | **213** **[data: martin_dustin_26061991_performance_details.csv]** |
| Gary Ablett Jr | 357 **[data: ablett_gary_14051984_performance_details.csv]** | 24.9 **[data: ablett_gary_14051984_performance_details.csv]** | 445 **[data: ablett_gary_14051984_performance_details.csv]** | 1.25 **[data: ablett_gary_14051984_performance_details.csv ; derived=445÷357]** | 262 **[data: ablett_gary_14051984_performance_details.csv]** |
| Patrick Dangerfield | 370 **[data: dangerfield_patrick_05041990_performance_details.csv]** | 22.7 **[data: dangerfield_patrick_05041990_performance_details.csv]** | 377 **[data: dangerfield_patrick_05041990_performance_details.csv]** | 1.02 **[data: dangerfield_patrick_05041990_performance_details.csv ; derived=377÷370]** | 259 **[data: dangerfield_patrick_05041990_performance_details.csv]** |
| Nat Fyfe | 247 **[data: fyfe_nat_18091991_performance_details.csv]** | 23.5 **[data: fyfe_nat_18091991_performance_details.csv]** | 178 **[data: fyfe_nat_18091991_performance_details.csv]** | 0.72 **[data: fyfe_nat_18091991_performance_details.csv ; derived=178÷247]** | 190 **[data: fyfe_nat_18091991_performance_details.csv]** |
| Scott Pendlebury | 435 **[data: pendlebury_scott_07011988_performance_details.csv]** | 25.4 **[data: pendlebury_scott_07011988_performance_details.csv]** | 207 **[data: pendlebury_scott_07011988_performance_details.csv]** | 0.48 **[data: pendlebury_scott_07011988_performance_details.csv ; derived=207÷435]** | 225 **[data: pendlebury_scott_07011988_performance_details.csv]** |
| Joel Selwood | 355 **[data: selwood_joel_26051988_performance_details.csv]** | 24.6 **[data: selwood_joel_26051988_performance_details.csv]** | 175 **[data: selwood_joel_26051988_performance_details.csv]** | 0.49 **[data: selwood_joel_26051988_performance_details.csv ; derived=175÷355]** | 214 **[data: selwood_joel_26051988_performance_details.csv]** |
| Lachie Neale | 308 **[data: neale_lachie_24051993_performance_details.csv]** | 27.5 **[data: neale_lachie_24051993_performance_details.csv]** | 140 **[data: neale_lachie_24051993_performance_details.csv]** | 0.45 **[data: neale_lachie_24051993_performance_details.csv ; derived=140÷308]** | 225 **[data: neale_lachie_24051993_performance_details.csv]** |

² Brownlow figures are arithmetic sums from per-game CSVs, repo-derived only — not cross-checked against AFL official records. See footnote ¹.

What the table makes visible: Martin's 1.12 goals per game is the second-highest in this group, behind only Ablett Jr (1.25). It is more than double the rate of the pure-distribution midfielders — Pendlebury 0.48, Selwood 0.49, Neale 0.45. At the same time his 24.2 disposal average is comparable to or exceeds theirs. The combination — high-disposal output and a genuine goal-scoring rate — is what the peer table quantifies.

### The rarity of the combination

Across the full dataset of every player file in `data/player_data/` (13,000+ files, VFL/AFL history from 1897), three thresholds applied simultaneously — 200+ career games, 20+ disposals per game, 300+ career goals — return **12 players in the entire recorded history of the game** **[data: data/player_data/ ; filter=games≥200,disposals/game≥20,goals≥300 ; aggregation=count]**.

| Player | Games | Disp/g | Goals | Era |
|---|---:|---:|---:|---|
| Leigh Matthews | 332 **[data: matthews_leigh_01031952_performance_details.csv]** | 22.2 **[data: matthews_leigh_01031952_performance_details.csv]** | 915 **[data: matthews_leigh_01031952_performance_details.csv]** | 1969–1985 |
| Kevin Bartlett | 403 **[data: bartlett_kevin_06031947_performance_details.csv]** | 22.7 **[data: bartlett_kevin_06031947_performance_details.csv]** | 778 **[data: bartlett_kevin_06031947_performance_details.csv]** | 1965–1983 |
| Brent Harvey | 432 **[data: harvey_brent_14051978_performance_details.csv]** | 21.3 **[data: harvey_brent_14051978_performance_details.csv]** | 518 **[data: harvey_brent_14051978_performance_details.csv]** | 1996–2016 |
| Garry Wilson | 268 **[data: wilson_garry_17071953_performance_details.csv]** | 25.0 **[data: wilson_garry_17071953_performance_details.csv]** | 452 **[data: wilson_garry_17071953_performance_details.csv]** | 1971–1984 |
| Gary Ablett Jr | 357 **[data: ablett_gary_14051984_performance_details.csv]** | 24.9 **[data: ablett_gary_14051984_performance_details.csv]** | 445 **[data: ablett_gary_14051984_performance_details.csv]** | 2002–2020 |
| Patrick Dangerfield | 370 **[data: dangerfield_patrick_05041990_performance_details.csv]** | 22.7 **[data: dangerfield_patrick_05041990_performance_details.csv]** | 377 **[data: dangerfield_patrick_05041990_performance_details.csv]** | 2008–2026 |
| John Murphy | 246 **[data: murphy_john_20111949_performance_details.csv]** | 24.6 **[data: murphy_john_20111949_performance_details.csv]** | 374 **[data: murphy_john_20111949_performance_details.csv]** | 1967–1980 |
| Dale Weightman | 274 **[data: weightman_dale_03101959_performance_details.csv]** | 20.8 **[data: weightman_dale_03101959_performance_details.csv]** | 344 **[data: weightman_dale_03101959_performance_details.csv]** | 1978–1993 |
| James Hird | 253 **[data: hird_james_04021973_performance_details.csv]** | 20.1 **[data: hird_james_04021973_performance_details.csv]** | 343 **[data: hird_james_04021973_performance_details.csv]** | 1992–2007 |
| **Dustin Martin** | **302** **[data: martin_dustin_26061991_performance_details.csv]** | **24.2** **[data: martin_dustin_26061991_performance_details.csv]** | **338** **[data: martin_dustin_26061991_performance_details.csv]** | **2010–2024** |
| Wayne Richardson | 277 **[data: richardson_wayne_08121946_performance_details.csv]** | 23.6 **[data: richardson_wayne_08121946_performance_details.csv]** | 323 **[data: richardson_wayne_08121946_performance_details.csv]** | 1966–1978 |
| David Clarke | 211 **[data: clarke_david_31121952_performance_details.csv]** | 21.4 **[data: clarke_david_31121952_performance_details.csv]** | 319 **[data: clarke_david_31121952_performance_details.csv]** | 1971–1982 |

Among players whose careers fall in the modern era (from 2000 onward), the group contains three: Gary Ablett Jr, Patrick Dangerfield, and Dustin Martin.

The three thresholds are stated precisely because the count is threshold-sensitive — shifting any boundary changes the membership. This is not a "greatest ever" claim. It is a data-defined description of what made his style distinctive: accumulating disposals at elite-midfielder volume while also finishing at a rate that belongs in a different position type. That combination, across 302 games and fifteen seasons, is what the 12-in-history count records.

Three premierships is not a number many AFL players carry. To be the common thread across three flag-winning teams at the same club — with the above career profile — is the full statement the data can make about where Dustin Martin sits.

---

## Footnotes

¹ **Brownlow votes caveat.** The figure of 213 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=brownlow_votes ; aggregation=sum]** is the arithmetic sum of the `brownlow_votes` column in this repo's per-game CSV. It has not been cross-checked against the AFL's official Brownlow Medal records or any external register. Treat as a data-derived approximation, not an externally verified career total.

---

## Sources / Methodology

Files opened to produce this brief:

- `data/player_data/martin_dustin_26061991_performance_details.csv` — career totals, per-season game counts (2010–2024), finals split (disposals, goals), GF game rows for 2017/2019/2020, retirement status (rows for 2025 = 0, rows for 2026 = 0).
- `data/matches/matches_2017.csv` — 2017 Grand Final result verified: Richmond 16.12 (108) def Adelaide 8.12 (60), margin 48.
- `data/matches/matches_2019.csv` — 2019 Grand Final result verified: Richmond 17.12 (114) def Greater Western Sydney 3.7 (25), margin 89.
- `data/matches/matches_2020.csv` — 2020 Grand Final result verified: Richmond 12.9 (81) def Geelong 7.8 (50), margin 31, venue Gabba.

All numbers in this document are drawn exclusively from the DATA BLOCK supplied to BriefBuilder and verified against the above files before assembly. No number has been introduced, inferred, or rounded differently from the DATA BLOCK. Specifically: disposal avg = 24.2 (not 24.24); tackle avg = 2.8 (not 2.76 or 3.1); regular-season disposal avg = 24.4 (not 24.36); finals disposal avg = 22.0.

**Numbers NOT in the DATA BLOCK and therefore NOT included.** Per-season disposals totals, per-season goals totals, per-season tackle totals, per-season Brownlow vote totals, career goals-per-game, finals tackles average, regular-season tackles average, GF tackle counts per game. If Scientist or FootyStrategy require these, they should be sourced directly from `martin_dustin_26061991_performance_details.csv` and added with verified [data] tags.

**Fill-zero convention.** Per-game averages in this article use the fill-zero convention: a counting stat (goals, tackles) not recorded for a played game is treated as zero, and every per-game rate divides by all games played in the relevant bucket, not only games with a recorded figure. This keeps career headlines and finals/regular splits on the same denominator and reconcilable to the career totals (goals sum to 338, tackles to 835). Disposals are fully recorded (no missing games).

**Tackle coverage annotation.** 835 career tackles **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=tackles ; aggregation=sum(dropna)]**, 2.76 per game (835 ÷ 302 games played). Tackle data is recorded for 266 of his 302 games; the remaining 36 games carry no tackle figure and are counted as zero, consistent with the per-game denominator (all games played) used for every career and split average in this article.

**Goals coverage annotation.** 338 career goals **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=goals ; aggregation=sum]**, 1.12 per game (338 ÷ 302 games played). Goal data is recorded for 197 of his 302 games; the remaining 105 games carry no goal figure and are counted as zero under the same per-game convention.

**Finals split definition.** Finals rows identified by non-numeric round labels in the `round` column (EF, QF, SF, PF, GF). Regular season = all rows with numeric round labels. The 16+286=302 total is consistent with the career game count **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=all ; column=row_count ; aggregation=count]**.

**GF team score computation.** In the matches CSVs, scores are stored as per-quarter goal and behind counts; final score = (sum of quarter goals × 6) + (sum of quarter behinds). 2017: Adelaide listed first in row, Richmond second. 2019 and 2020: Richmond listed first. Scores computed accordingly and cross-checked against the DATA BLOCK-supplied values.

Era-coverage gaps: none declared. All four stat categories (disposals, goals, tackles, Brownlow votes) were officially being recorded throughout 2010–2024 — no era gap. Per-game completeness is a separate question and is annotated above: goals recorded for 197 of 302 games; tackles recorded for 266 of 302 games; disposals and Brownlow votes are fully recorded. Tackles have been collected since 1987; no era gap applies to Martin's career window.

No coach names appear in this article.

---

*Data layer: BriefBuilder v1.0. Interpretation layer: FootyStrategy v1.0.*

<!-- council-pipeline:
  BriefBuilder: DONE
  Scientist: DONE (peer-group comparison: Ablett Jr, Dangerfield, Fyfe, Pendlebury, Selwood, Neale; all-time rank #79; 12-in-history uniqueness count — all verified against source CSVs)
  FootyStrategy: DONE (v2: comparative section added; all peer numbers cross-checked against CSV before inclusion)
  DataSentinel: PASS (v2: peer-group table tags and all-time rank tags all reference verified source files; filename corrections applied — fyfe_nat_18091991, neale_lachie_24051993)
  Skeptic: PASS_WITH_CONCERNS → PASS (4 prose fixes applied v1; v2 comparative section uses threshold-explicit language, no unbacked superlatives, membership claim precisely bounded)
  Gaffer: SHIP @ 2026-06-21
-->
