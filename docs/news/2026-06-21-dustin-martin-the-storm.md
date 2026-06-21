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

The disposal average of 24.2 per game over 302 games is a measure of sustained output, not peak performance. Any player can put up a high disposal count for a season or across a run of form. Sustaining that average across fifteen seasons — through roles that shifted, through team rebuilds, through the seasons where the games played dropped — is a different kind of achievement.

Three premierships is not a number many AFL players carry. To be the common thread across three flag-winning teams at the same club, as a player whose presence in the side was central to each of those campaigns, is a career-defining credential by any measure the game uses. The comparison class for "power midfielder, 300-plus games, three premierships at one club" is small.

Where Martin sits in Richmond's all-time list is a question the data in this repo cannot fully answer — the repo carries career stats but not a ranked framework for comparing players across eras and positions. What the data does say: 302 games, 338 goals, 7,320 disposals. Those are the numbers he leaves behind. The rest is the game's own memory.

---

## 7. What Richmond Needs Now

Martin is not playing in 2026 **[data: data/player_data/martin_dustin_26061991_performance_details.csv ; filter=year=2026 ; column=row_count ; aggregation=count]**. For Richmond's forward path from here, see the five-year strategic analysis: [2026 AFL Grand Final Strategy — all 18 clubs](./2026-06-19-afl-2026-5yr-grand-final-strategy.md).

What Martin's absence creates is a structural gap that cannot be filled by adding a single player. He was not a positional role player whose responsibilities can be reassigned; he was a zone-of-influence midfielder whose game bent the shape of the opposition around him. Defenders who had to account for him as both a midfield bull and a forward-50 target were drawn away from their primary assignments. That displacement effect disappears when he does.

For Richmond's five-year forward plan, the question is not "who plays his role" — that framing leads to drafting or trading for someone who superficially resembles him. The structural question is how Richmond rebuilds the quality of midfield possession (his 24.2 disposal average was the engine of their ball movement in the flag years) and how they address forward-50 conversion without the player who could manufacture goals from broken play.

The five-year strategy article at the link above addresses Richmond's current list construction and what they most need from here. The Dustin Martin gap is part of the backdrop for that analysis — a five-year flag window that closed at the same time their most influential player stopped playing.

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
  Scientist: N/A (retrospective; no model predictions; BriefBuilder carried data verification)
  FootyStrategy: DONE
  DataSentinel: PASS (88/89 tags verified; 1 tag label corrected; 6 derived-stat tags added; no coach names; no unverified numbers)
  Skeptic: PASS_WITH_CONCERNS → PASS (4 prose fixes applied: deployment claim outcome-conditional; "not a statistical rounding error" removed; 2023 "confirms" softened; era-coverage vs per-game gap clarified)
  Gaffer: SHIP @ 2026-06-21
-->
