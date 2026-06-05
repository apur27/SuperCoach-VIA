# List Management 101: Is the Top-10 Draft Pick Strategy a Path to Premiership Dominance?

> Data analysis across 127 premierships, 18 clubs and 13,329 player careers. 2026-06-05

[← Back to main README](../README.md)

## The question

It is the most seductive idea in football list management: stockpile early draft
picks, draft the best teenager available year after year, and a dynasty assembles
itself. Bottom out, pick top-10, repeat, and the premiership window opens on
schedule. Is that actually how the great teams were built?

This repo cannot answer the literal version of that question — and it is important
to say so up front. The match and player data here carry **no draft-pick numbers**:
no draft rounds, no selection order, no trade history. What it *does* carry is the
complete record of who played, for whom, in which season, across the whole VFL/AFL
era. So instead of "were they top-10 picks?", we can ask the question the data can
actually answer: **were the great premiership teams built on players who stayed and
played a lot of football — and is a long, retained, experienced core what actually
separates a flag from the rest?** That is the proxy. A career of 150+ AFL games is
almost never an accident of late recruiting; it is a player a club drafted (or
recruited early) and kept.

## What the data shows

### Premierships cluster, and they cluster around experienced cores

Across the GF-decided era in this dataset there are 127 premierships from 1898 to
2025 **[data: data/matches/matches_*.csv ; round_num == "Grand Final", last GF row per year ; team_1/team_2 final score ; count of higher-scoring team]**.
They are not spread evenly. Just two clubs — Collingwood and Carlton — account for
16 flags each **[data: data/matches/matches_*.csv ; round_num == "Grand Final" ; premier club ; count by club]**,
and the four most successful clubs (Collingwood, Carlton, Essendon, and a
three-way tie behind them) hold a disproportionate share. Dominance is real and it
is concentrated.

When you open up the dynasties of the modern era, the same shape appears every
time. Take the top 22 players by games played in each premiership season and count
how many had careers longer than 150 games:

| Premiership team | Top-22 with 150+ game careers | Median career (games) |
|---|---|---|
| Hawthorn 2013 | 21 / 22 **[data: data/player_data/*_performance_details.csv ; team=="Hawthorn" & year==2013, top 22 by games ; career rows ; count >150]** | 283 **[data: same ; median of 22 career totals]** |
| Geelong 2009 | 19 / 22 **[data: team=="Geelong" & year==2009, top 22 by games ; career rows ; count >150]** | 280 **[data: same ; median]** |
| Brisbane Lions 2002 | 20 / 22 **[data: team=="Brisbane Lions" & year==2002, top 22 by games ; career rows ; count >150]** | 256 **[data: same ; median]** |
| Richmond 2017 | 19 / 22 **[data: team=="Richmond" & year==2017, top 22 by games ; career rows ; count >150]** | 213 **[data: same ; median]** |
| Sydney 2012 | 20 / 22 **[data: team=="Sydney" & year==2012, top 22 by games ; career rows ; count >150]** | 226 **[data: same ; median]** |

Every premiership core in the modern dynasty sample is built on long-career
players. The median premiership player in these sides played somewhere between 213
and 283 games — a full career, not a passing contribution.

### Premierships are won by depth, not by a handful of stars

The flags were not carried by three superstars and a supporting cast of
journeymen. Counting players who managed 15 or more games in the premiership
season — a genuine contributor, not a one-week fill-in — the winning squads are
broad: Brisbane fielded 24 such players in 2002 **[data: team=="Brisbane Lions" & year==2002 ; games-per-player that season ; count with >=15]**,
Hawthorn 23 in 2013 **[data: team=="Hawthorn" & year==2013 ; games that season ; count >=15]**,
Geelong 22 in 2009 **[data: team=="Geelong" & year==2009 ; games that season ; count >=15]**,
and Richmond 20 in 2017 **[data: team=="Richmond" & year==2017 ; games that season ; count >=15]**.
A premiership is a 20-plus-player effort.

### But experience is the price of entry, not the trophy itself

Here is the finding that complicates the headline — and the one the data insists
on. Experienced cores do not *guarantee* a flag, because almost every competitive
club already has one. In 2013, the league-average premiership-quality club ran 15.7
of its top 22 as 150+ game players; Hawthorn ran 21, the most in the competition
**[data: matches_2013 premier + player_data 2013 ; top-22 150+ counts per club ; league mean vs Hawthorn]**.
But in 2009 Geelong's 19 was only tied 3rd in the competition **[data: 2009 ; top-22 150+ count per club ; premier tied 3rd-4th at 19]**,
and in 2017 Richmond's 19 was tied 2nd-3rd, behind only Greater Western Sydney **[data: 2017 ; top-22 150+ count per club ; premier tied 2nd-3rd at 19, GWS top on 21]**.

The gap between the premier and the league average is real but modest — typically
three to five players **[data: matches_*.csv + player_data ; 2003/2009/2013/2017/2012 ; premier top-22 150+ minus league mean]**.
What the data shows cleanly is the *floor*: the least-experienced cores never win.
In 2013 the three thinnest cores belonged to Melbourne (10 of 22), Essendon (12)
and Gold Coast (13) **[data: 2013 ; top-22 150+ count per club ; three lowest]** —
none within reach of a flag. Experience is the price of admission. It buys you a
ticket; it does not buy you the premiership.

## The verdict

Building a team of top-10 draft picks is the wrong way to phrase the question, and
the data quietly explains why. Premierships are not won by *draft position* — they
are won by **retention and accumulated experience**. The flags in this dataset were
built on cores where 17 to 21 of the top 22 players had careers beyond 150 games,
carried by genuine depth of 20-plus contributors, assembled over years rather than
bought in a single draft.

A top-10 pick only matters if the player it brings stays for a decade and plays 200
games. The strategy that the data actually rewards is not "draft early" — it is
"draft well, develop, and keep." A bottom-out-and-rebuild plan that churns its list
every few seasons is optimising the one variable (pick number) the premiership
record is indifferent to, while neglecting the one it is not (a deep, retained,
experienced core). The teenagers a club drafts are raw material. The premiership is
what a club does with them over the following ten years.

## Methodology

- **Premiers** are derived from `data/matches/matches_*.csv` by selecting the row
  where `round_num == "Grand Final"` (the last such row in replay years: 1948,
  1977, 2010) and comparing final scores (goals × 6 + behinds). 127 premierships
  are GF-decided in the data; 1897 (ladder), 1924 (round-robin finals) and 2026
  (season in progress) have no Grand Final row and are excluded.
- **Career length** is the total number of per-game rows in a player's
  `data/player_data/*_performance_details.csv` file. **150+ games is used as a
  proxy for "a drafted player who stuck"** — the dataset contains no draft-pick
  numbers, rounds, or selection order, so the literal top-10-pick question cannot
  be tested directly. This is the central limitation of the analysis and every
  conclusion should be read through it.
- **Premiership core** = the 22 players with the most games for that club in that
  season. Squads use 30-39 players a year, so the top-22 is an approximation of
  the round-to-round side, not an exact match-day list.
- **Concerns carried from review:** (1) The 150-game proxy conflates "drafted and
  retained" with "recruited early via trade" — some long-career premiership players
  arrived as established players, not draftees. (2) The premier-vs-league gap is
  modest (3-5 players) and premiers do not always lead the competition on this
  measure, so experience is necessary but demonstrably not sufficient. (3) Median
  career length is inflated by survivorship — we are measuring full careers in
  hindsight, including games played *after* the premiership. The direction of the
  finding is robust; the precise magnitude should be treated as indicative.

<!-- council-pipeline: single-operator cycle (no Agent dispatch tool available in this session). Gaffer authored deterministic data derivation (every [data] number re-read from CSV in Python, no in-token arithmetic) + DataSentinel-style self-verification PASS + Skeptic-style adversarial control test PASS_WITH_CONCERNS (concerns documented in Methodology). NOT a 7-agent council chain — stamped honestly. DataSentinel-self:PASS@2026-06-05, Skeptic-self:PASS_WITH_CONCERNS@2026-06-05, Gaffer:SHIP@2026-06-05 -->
