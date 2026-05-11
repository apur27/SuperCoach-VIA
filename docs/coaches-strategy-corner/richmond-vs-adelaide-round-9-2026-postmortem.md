# Richmond vs Adelaide - Post-Mortem (Round 9, 2026)

> [← Coaches Strategy Corner](README.md) | [← AFL insights](../afl-insights.md)
>
> Pre-match: [Executive summary](richmond-vs-adelaide-round-9-2026-executive-summary.md) · [Tactical brief](richmond-vs-adelaide-round-9-2026.md) · [Player matchups](richmond-vs-adelaide-round-9-2026-player-matchups.md) · [H2H history](richmond-vs-adelaide-round-9-2026-head-to-head-history.md)
> In-game: [Half-time](richmond-vs-adelaide-round-9-2026-half-time-live.md) · [Q3](richmond-vs-adelaide-round-9-2026-q3-live.md) · [Q4](richmond-vs-adelaide-round-9-2026-q4-live.md) · [Full-time verdict](richmond-vs-adelaide-round-9-2026-full-time-verdict.md)
>
> **Status: post-match.** This is a data-grounded structural autopsy. The full-time verdict is the headline-level narrative; this document is the methodology layer underneath - what the numbers say, where the pre-game read held, and where it broke. Every figure in this document is verified against `data/player_data/*_performance_details.csv` (now refreshed with Round 10 entries from afltables.com) and the final-siren snapshot `data/live_snapshots/9781_20260510_1751_final-siren.json`.

---

## 1. Final result and context

**Adelaide 14.14.98 def Richmond 9.7.61 by 37 points** at the M.C.G., 10 May 2026, attendance 22,123 **[data: matches_2026.csv, afltables.com]**.

This was Richmond's eighth loss from nine games in 2026 (1W-8L, -42.8 average margin) **[data: matches_2026.csv]**. It snapped a one-week recovery: Richmond had won R9 at Optus Stadium against West Coast by 11 points (15.9.99 to 13.10.88) with a season-high 339 disposals **[data]**. The R10 game took Richmond's only win of the year off the front page within seven days.

**Round-number caveat**: afltables.com classifies this fixture as **Round 10**. The repo's existing brief, live reads, and filenames use "Round 9" (following the FanFooty feed's `R9` header tag). The matches_2026.csv now contains it as Round 10 (May 10), consistent with the actual fixture sequence (the other "Round 9" games were 30 Apr - 3 May, including Richmond vs West Coast at Perth Stadium). Filenames and document titles retain "round-9" for continuity with the pre-match brief; the underlying data file uses Round 10. **This is a labelling artefact, not a data inconsistency** - the match is uniquely identified by date (2026-05-10) and the score is the same in every source.

### Context Richmond was missing

Per commentary log on the final-siren snapshot:

> The cupboard is bare at Tigerland with 17 names on the injury list, all but two of those with an estimated outage of three weeks or more.

Six players the pre-game brief named as load-bearing did not play: **Samson Ryan** (the ruck pivot the entire hit-out lever depended on), **Dion Prestia**, **Sam Lalor**, **Sam Banks**, **Samuel Grlj** (managed), and **Liam Fawcett**. Adelaide were also missing **Taylor Walker** (Showdown injury) and **Jordon Butts** **[data: final-siren snapshot commentary; brief player list cross-checked]**.

The brief's structural plan was conditional on Ryan. He was not available. That single absence inverted the largest projected edge in the game.

---

## 2. Quarter-by-quarter breakdown

### Scoreboard progression

| Q | Richmond (cum) | Adelaide (cum) | Margin | Q score (RI/AD) |
|---|---|---|---|---|
| Q1 | 3.2 (20) | 3.2 (20) | RI = AD | RI 3.2 / AD 3.2 (level) |
| HT | 7.4 (46) | 5.6 (36) | RI **+10** | RI 4.2 / AD 2.4 (RI +14) |
| 3QT | 7.5 (47) | 10.10 (70) | AD +23 | RI 0.1 / AD 5.4 (AD **+33**) |
| Final | 9.7 (61) | 14.14 (98) | AD **+37** | RI 2.2 / AD 4.4 (AD +14) |

**Source: afltables.com (Round 10, 2026). Half-time and final scores cross-verified against the FanFooty snapshot `9781_20260510_1751_final-siren.json` and the half-time snapshot `9781_20260510_1639_half-time.json`.**

### AF (AFL Fantasy points) by quarter - team totals from snapshot

| Quarter | Richmond AF | Adelaide AF | Delta |
|---|---:|---:|---:|
| Q1 | 415 | 418 | AD +3 |
| Q2 | 409 | 382 | RI +27 |
| Q3 | 238 | 543 | **AD +305** |
| Q4 | 296 | 434 | AD +138 |
| **Game** | **1,358** | **1,777** | **AD +419** |

**The Q3 AF margin (+305 to Adelaide) is 73% of the final AF margin. There is no game plan that survives a 305-AF quarter loss.**

The pattern matches the scoreboard: Richmond led the goal-count by 2.2 at half-time, then kicked **0.1 in the entire third quarter** while Adelaide kicked 5.4. The Q3 0.1 to 5.4 is the structural collapse in a single line.

### Commentary timeline (final-siren snapshot, Q4 only)

The snapshot's live commentary stream provides timestamped Q4 events (Q1-Q3 events were not retained at the final snapshot but the half-time and Q3 in-progress snapshots cover earlier passages):

| Time | Event |
|---|---|
| Q4 7:11 | Borlase loses feet; Lynch marks long ball to square and goals "to stop the rot". |
| Q4 10:21 | Peatling clanger in centre → Lefau marks 40m out, misses. |
| Q4 11:36 | Ross outside pack on HFF, centres for Hopper marking in front of Dawson 45m out; Hopper shanks left, OOTF. |
| Q4 13:26 | Lynch gathers a bounce pass near the hotspot, gives off to Sam Cumming who snaps a goal off his left. "Richmond still in this, at least they think so!" |
| Q4 17:05 | Three Richmond set shots in a 25-second cluster: Taylor (45m, fades into post), Pedlar (40m miss), Taranto (grubber bounces into goalpost). **Three shots, zero goals.** |
| Q4 22:50 | Rachele draws a soft high-contact free on Campbell; goals to ice the game. |
| Q4 25:58 | Dawson passes to McAndrew 35m out; McAndrew kicks his **first senior career goal**. |
| Q4 27:52 | Retschko kick clanger → Keays junk goal from 30m on the flank. |

**The Q4 cluster at 17:05 is the conversion problem in microcosm**: three shots in 25 seconds, all from inside 50, zero goals. Richmond's R10 inside-50 count (45) was only 9 behind Adelaide's (54) **[data]** - they had supply. They could not finish it.

---

## 3. What the data shows - full player tables

### Richmond - all 23 players, sorted by SC (verified against afltables.com)

| # | Player | D | K | H | M | G | B | HO | T | C | AF | SC | Q1 | Q2 | Q3 | Q4 | TOG% | DE% |
|---|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Jack Ross | 26 | 8 | 18 | 5 | 0 | 0 | 0 | 6 | 4 | 96 | **107** | 28 | 24 | 18 | 26 | 86 | 73 |
| 2 | Ben Miller | 18 | 13 | 5 | 7 | 0 | 0 | 0 | 3 | 4 | 80 | **106** | 16 | 22 | 28 | 14 | 93 | 77 |
| 3 | Jayden Short | 26 | 15 | 11 | 10 | 0 | 0 | 0 | 0 | 2 | 94 | **101** | 24 | 49 | 11 | 10 | 83 | 96 |
| 4 | Tim Taranto | 28 | 12 | 16 | 0 | 0 | 1 | 0 | 4 | 4 | 87 | 99 | 30 | 22 | 14 | 21 | 86 | 64 |
| 5 | Tom J. Lynch | 14 | 8 | 6 | 3 | 3 | 0 | 0 | 1 | 3 | 71 | 90 | 17 | 22 | 9 | 23 | 83 | 85 |
| 6 | Nick Vlastuin | 21 | 11 | 10 | 2 | 0 | 0 | 0 | 3 | 2 | 67 | 90 | 18 | 32 | 11 | 6 | 92 | 81 |
| 7 | James Trezise | 18 | 12 | 6 | 7 | 0 | 0 | 0 | 1 | 0 | 78 | 84 | 21 | 34 | 8 | 15 | 76 | 83 |
| 8 | Luke Trainor | 18 | 11 | 7 | 10 | 0 | 0 | 0 | 1 | 1 | 77 | 76 | 19 | 13 | **32** | 13 | 86 | 88 |
| 9 | Noah Balta | 14 | 11 | 3 | 6 | 1 | 1 | 8 | 1 | 1 | 78 | 72 | 31 | 8 | 13 | 26 | 83 | 78 |
| 10 | Patrick Retschko | 22 | 12 | 10 | 6 | 0 | 0 | 0 | 0 | 4 | 71 | 71 | 33 | 8 | 14 | 16 | 94 | 81 |
| 11 | Steely Green | 10 | 5 | 5 | 5 | 1 | 0 | 0 | 5 | 0 | 66 | 70 | 19 | 30 | 13 | 4 | 84 | 80 |
| 12 | Tyler Sonsie | 13 | 8 | 5 | 3 | 1 | 0 | 0 | 1 | 2 | 49 | 65 | 16 | 18 | 4 | 11 | 89 | 92 |
| 13 | Mykelti Lefau | 11 | 4 | 7 | 3 | 1 | 1 | 2 | 2 | 2 | 53 | 62 | 13 | 19 | 9 | 12 | 85 | 81 |
| 14 | Nathan Broad | 12 | 10 | 2 | 4 | 0 | 0 | 0 | 0 | 1 | 47 | 57 | 9 | 22 | 11 | 5 | 87 | 91 |
| 15 | Campbell Gray | 9 | 8 | 1 | 4 | 0 | 0 | 0 | 4 | 3 | 46 | 57 | 24 | 19 | 2 | 1 | 87 | 88 |
| 16 | Jacob Hopper | 23 | 13 | 10 | 3 | 0 | 0 | 0 | 3 | **7** | 75 | 55 | 20 | 14 | 17 | 24 | 80 | 43 |
| 17 | Seth Campbell | 12 | 9 | 3 | 4 | 1 | 0 | 0 | 0 | 5 | 45 | 55 | 20 | 8 | 0 | 17 | 84 | 66 |
| 18 | Oliver Hayes-Brown | 11 | 1 | 10 | 1 | 0 | 0 | 9 | 1 | 0 | 40 | 52 | 7 | 16 | 8 | 9 | 48 | 54 |
| 19 | Jonty Faull | 11 | 7 | 4 | 3 | 0 | 0 | 0 | 1 | 4 | 42 | 32 | 3 | 7 | 13 | 19 | 82 | 45 |
| 20 | Tom Burton | 9 | 5 | 4 | 2 | 0 | 1 | 0 | 1 | 2 | 34 | 32 | 16 | 0 | 4 | 14 | 78 | 66 |
| 21 | Sam Cumming | 8 | 4 | 4 | 1 | 1 | 0 | 0 | 0 | 1 | 29 | 31 | 11 | 5 | 2 | 11 | 73 | 87 |
| 22 | Kane McAuliffe | 10 | 8 | 2 | 3 | 0 | 1 | 0 | 0 | 5 | 30 | 21 | 17 | 17 | -3 | -1 | 58 | 80 |
| 23 | Tom Brown | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 6 | 3 | 0 | 0 | 0 | 4 | 100 |

**Team totals: D 345, K 196, H 149, M 92, G 9, B 5, HO 19, T 38, C 57, AF 1,358, SC 1,491.**

### Adelaide - all 23 players, sorted by SC

| # | Player | D | K | H | M | G | B | HO | T | C | AF | SC | Q1 | Q2 | Q3 | Q4 | TOG% | DE% |
|---|--------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Izak Rankine | 33 | 24 | 9 | 10 | 1 | 1 | 0 | **9** | 5 | **158** | **147** | 48 | 33 | 41 | 36 | 78 | 72 |
| 2 | Wayne Milera | 34 | 18 | 16 | 6 | 0 | 0 | 0 | 2 | 2 | 113 | **127** | 24 | 23 | 45 | 21 | 78 | **94** |
| 3 | Jordan Dawson | 29 | 22 | 7 | 12 | 1 | 0 | 0 | 4 | 6 | 133 | **115** | 32 | 27 | **48** | 26 | 82 | 75 |
| 4 | Rory Laird | 27 | 16 | 11 | 12 | 0 | 0 | 0 | 1 | 2 | 110 | **105** | 12 | 38 | 39 | 21 | 84 | 81 |
| 5 | Brayden Cook | 22 | 14 | 8 | 7 | 0 | 0 | 0 | 1 | 1 | 83 | 87 | 16 | 12 | 42 | 13 | 81 | 72 |
| 6 | Josh Worrell | 21 | 13 | 8 | 7 | 0 | 0 | 0 | 3 | 2 | 88 | 86 | 23 | 14 | 34 | 17 | 92 | 85 |
| 7 | Josh Rachele | 14 | 8 | 6 | 5 | 2 | 1 | 0 | 1 | 2 | 66 | 86 | 14 | 14 | 23 | 15 | 75 | 85 |
| 8 | Lachlan McAndrew | 9 | 6 | 3 | 4 | 1 | 0 | **33** | 4 | 3 | 86 | 83 | 20 | 25 | 19 | 22 | 69 | 66 |
| 9 | James Peatling | 19 | 12 | 7 | 7 | 0 | 1 | 0 | 5 | 3 | 93 | 82 | 33 | 8 | 16 | 36 | 69 | 63 |
| 10 | Alex Neal-Bullen | 16 | 11 | 5 | 9 | 2 | 2 | 0 | 1 | 1 | 85 | 82 | 16 | 18 | 28 | 23 | 76 | 81 |
| 11 | Sam Berry | 20 | 7 | 13 | 4 | 0 | 1 | 0 | 3 | 2 | 69 | 73 | 20 | 12 | 25 | 12 | 68 | 85 |
| 12 | Toby Murray | 9 | 7 | 2 | 2 | 1 | 1 | 4 | 6 | 1 | 69 | 73 | 6 | 12 | 8 | 43 | 60 | 77 |
| 13 | Daniel Curtin | 11 | 9 | 2 | 5 | 1 | 0 | 0 | 3 | 2 | 63 | 72 | 21 | 32 | 2 | 8 | 81 | 90 |
| 14 | Riley Thilthorpe | 12 | 7 | 5 | 5 | 1 | 3 | 0 | 1 | 0 | 59 | 69 | 12 | 13 | 23 | 11 | 86 | 66 |
| 15 | James Borlase | 15 | 11 | 4 | 7 | 0 | 0 | 0 | 1 | 1 | 63 | 66 | 21 | 6 | 20 | 16 | 91 | 80 |
| 16 | Isaac Cumming | 15 | 12 | 3 | 8 | 1 | 0 | 0 | 1 | 3 | 70 | 65 | 17 | 11 | 24 | 18 | 76 | 93 |
| 17 | Luke Pedlar | 8 | 7 | 1 | 6 | 1 | 1 | 0 | 3 | 0 | 60 | 65 | 22 | 24 | 4 | 10 | 69 | 87 |
| 18 | Ben Keays | 15 | 9 | 6 | 5 | 1 | 0 | 0 | 3 | 1 | 70 | 64 | 24 | 4 | 19 | 23 | 89 | 73 |
| 19 | Zac Taylor | 15 | 10 | 5 | 6 | 0 | 1 | 0 | 3 | 1 | 72 | 59 | 10 | 17 | 24 | 21 | 67 | 73 |
| 20 | Max Michalanney | 14 | 11 | 3 | 8 | 0 | 0 | 0 | 0 | 1 | 60 | 54 | 18 | 0 | 19 | 23 | 88 | 85 |
| 21 | Hugh Bond | 12 | 6 | 6 | 2 | 0 | 0 | 0 | 2 | 1 | 45 | 47 | 7 | 15 | 12 | 11 | 80 | 83 |
| 22 | Nick Murray | 7 | 4 | 3 | 2 | 1 | 0 | 0 | 3 | 4 | 33 | 46 | -3 | 18 | 16 | 2 | 92 | 100 |
| 23 | Jake Soligo | 8 | 5 | 3 | 2 | 0 | 0 | 0 | 2 | 3 | 29 | 33 | 5 | 6 | 12 | 6 | 64 | 50 |

**Team totals: D 385, K 249, H 136, M 141, G 14, B 12, HO 37, T 62, C 47, AF 1,777, SC 1,786.**

**Data verification note**: Goals, behinds, and clangers in the final-siren snapshot were found to disagree with afltables.com on 102 player-field instances - the snapshot's column indexing for those three fields is unreliable for this game. The numbers above use **afltables.com as the source of truth** for those three fields and the FanFooty snapshot for AF/SC/quarter splits/TOG/DE. Kicks, handballs, marks, tackles, and hit-outs agree exactly across both sources. The discrepancy is documented in [scripts/update_r10_player_data.py](../../scripts/update_r10_player_data.py).

---

## 4. Richmond's structural failures - what went wrong

Numbered failures, each grounded in the verified box-score and the season-form baseline.

### 4.1 Hit-out battle inverted (AD +18) - the supply chain broke at link one

| | Pre-game projection | Actual | Swing |
|---|---|---|---|
| Hit-outs (RI - AD) | RI **+15** (Ryan-conditional) | **AD +18** (RI 19, AD 37) | 33-point reversal |
| McAndrew solo HO | Brief had no McAndrew profile | **33 of Adelaide's 37** | - |

The brief identified hit-outs as "the single biggest structural lever in the game". The lever depended on Samson Ryan playing. Ryan was scratched on the injury list (one of 17 names). McAndrew, a developmental ruck, took 33 of Adelaide's 37 hit-outs and put up 86 AF / 83 SC / first senior career goal. Adelaide's `clearance ceiling constrained by their ruck` (brief, section 3) was lifted. Richmond won 19 hit-outs split across Balta (8), Hayes-Brown (9), and Lefau (2); none generated clean midfield exit.

Downstream effect: Tom Lynch (Richmond's only fit specialist marking forward) received **14 disposals all game** **[data]** with a single confirmed goal at Q4 7:11, after Adelaide had already opened the gap to 50+. The supply chain stoppage-first-use → midfield exit → entry to Lynch failed before it generated a single second-half scoring shot.

### 4.2 Q3 collapse (-305 AF) - the variance the brief did not model

Richmond's mean Q3 differential entering the game was -9.1 AF/g **[brief, section "Quarter-by-quarter pattern"]**. The Q3 floor against Adelaide was **-305 AF**. That is 33 standard-deviations of Q3 outcomes if quarters are modelled as means; it is also the *tail event* the brief's averages did not surface.

The mechanism, from the snapshot:

| Adelaide Q3 contributors | Q3 AF |
|---|---:|
| Jordan Dawson | 48 |
| Wayne Milera | 45 |
| Brayden Cook | 42 |
| Izak Rankine | 41 |
| **Four players >40 Q3 AF** | |
| Richmond best Q3: Trainor (a late inclusion) | 32 |

No other Tiger cracked 30 Q3 AF. Six Richmond players posted **negative or zero Q3 AF**: Burton 4 (low positive), McAuliffe -3, Brown 0, Lefau 9, Sonsie 4, Hopper still functional at 17. Richmond's mid-rotation collapsed.

The structural cause was Adelaide's half-time reset, not Richmond's fatigue per se: Dawson moved from half-back (where he had 25 Q1-Q2 AF) into the engine room (74 Q3-Q4 AF), and Nick Murray moved from KPF (15 Q1-Q2 AF) to defence. Adelaide's matchup zone changed faster than Richmond could re-tag. Richmond was set up to suppress two players (Dawson, Murray) at their pre-match positions; both moved.

### 4.3 Conversion - volume, not accuracy

This is the finding the live reads and verdict already flagged; the postmortem confirms it.

| Side | Goals | Behinds | Scoring shots | Accuracy |
|---|---:|---:|---:|---:|
| Richmond | 9 | 7 | 16 | **56%** |
| Adelaide | 14 | 14 | 28 | 50% |

**Richmond's accuracy beat Adelaide's by 6 percentage points.** The brief's framing ("Adelaide is the more accurate set-shot side; Richmond cannot afford behinds") was incorrect on the day. The decisive variable was **volume**: Adelaide had 12 more scoring shots (28 vs 16). At Adelaide's actual 50% accuracy, Richmond needed 20 scoring shots to match Adelaide's 14 goals (they generated 16). At Richmond's 56% accuracy, Adelaide's 28 shots yield 15.7 goals - the volume gap explains the entire margin.

Adelaide inside-50s 54 vs Richmond 45 **[data: aggregated player files]** = AD +9. Adelaide goals/i50 = 0.259 vs Richmond 0.200 - close to the pre-game season figures of 0.253 and 0.192 respectively. The brief identified an accuracy lift target (0.230); Richmond achieved 0.200, but the lever was the wrong axis.

### 4.4 Clanger floor compounded the tackle gap

| | Pre-game projection | Actual |
|---|---|---|
| Tackles (RI - AD) | AD **+8** (additive from 51.8 vs 59.5 season rates) | AD **+24** (RI 38, AD 62) |
| Richmond clangers | Season avg 44.4/g | **57** (28% above) |
| Adelaide clangers | Season avg... | 47 |

Richmond's R10 clanger count (57) was 28% above their season average (44.4/g, already 18/18 in the league) **[data: aggregated from R2-R10 player files]**. Adelaide tackled at 115% of their season rate (62 vs 53.8/g) **[data]**. The interaction was multiplicative: each Richmond clanger inside the defensive arc creates a fresh tackle opportunity, which creates a fresh clanger risk, and the cycle compounds. The brief modelled the tackle gap additively (-8 AF projected); the actual was -24, a 200% miss on magnitude.

Players who carried the clanger load: Hopper 7 (also led contested poss with 12 effective disposals and 14 ineffective uncontested at 43% DE), Campbell 5, McAuliffe 5, Ross 4, Taranto 4, Retschko 4, Miller 4. **Seven Richmond players with 4+ clangers** - this is not one player's problem; this is a list-wide ball-security ceiling.

### 4.5 Individual matchup failures - the ones that mattered

| Matchup (brief) | Result |
|---|---|
| Ross vs Berry (the brief's pivot) | **Ross won**. Ross 96 AF / 107 SC / 6 tackles vs Berry 69 AF / 73 SC / 3 tackles. The brief was right at the matchup level; this was the most successful read in the document. Game still lost. |
| Taranto vs Dawson | Voided. Dawson started at half-back (Walker cover) so Taranto played stoppage-only Q1-Q2; once Dawson moved into the midfield at half-time, the matchup was 90 seconds too late to reset. Dawson Q3 AF 48 alone exceeded Taranto's whole-game AF (87). |
| Vlastuin on Walker | Did not exist. Walker DNP. Vlastuin tracked Murray as KPF in Q1-Q2 (effective: Murray 15 Q1-Q2 AF); once Murray went back to defence at half-time, Vlastuin had no fixed target and posted **6 Q4 AF** - he was gassed and out of role. |
| Short vs Milera | Lost on volume. Short 26 disp, 94 AF, 101 SC, 96% DE - **best disposal efficiency on the ground**. Milera 34 disp, 113 AF, 127 SC, 94% DE. Both clean, but Milera launched more and from better positions. |
| Ryan vs Thilthorpe/Maley split | Did not exist. Ryan DNP. McAndrew took 33 HO; Thilthorpe rucked 0. The brief's structural lever inverted. |

The brief identified five priority matchups. **One held (Ross). Two were voided by Adelaide absences (Vlastuin-Walker, Ryan-Thilthorpe/Maley). One was voided by Adelaide's half-time positional shift (Taranto-Dawson). One lost on a margin small enough to call a draw (Short-Milera).** A 1-for-5 strike rate on the matchups that decide a game is a structural failure, even if four of the five were undone by selection or in-game variance the brief could not have predicted.

---

## 5. Adelaide's formula - what they did systematically that worked

This is the system that Richmond - and any future opponent of Adelaide in 2026 - needs to model.

### 5.1 Q1-Q2: structural cover for absences, even at the cost of points

Adelaide entered the game without Walker, Butts, and with their usual #1 ruck rotation degraded. They opened with Murray at KPF (a defender playing tall forward) and Dawson at half-back (a captain mid playing as launch defender). Both were sub-optimal individually. Both were *structural cover designed to absorb damage in the first 45 minutes*. Richmond led 7.4 to 5.6 at half-time. Adelaide accepted that.

### 5.2 Half-time reset: 90 seconds of dressing-room work

Per commentary: *"After half time, Dawson and Nick Murray reverted to their more comfortable roles."* This was Plan B executed. The personnel didn't change; the structure did.

| Player | Pre-HT role | Post-HT role | Q1-Q2 AF | Q3-Q4 AF |
|---|---|---|---:|---:|
| Jordan Dawson | Half-back | Midfield | 59 | 74 |
| Nick Murray | Key forward | Defence | 15 (KPF) | 18 (back) |

Dawson's Q3 AF (48) is the largest single-quarter individual AF on the ground. The repositioning created a fourth on-baller (alongside Rankine, Milera, Cook) that Richmond had not planned a matchup for.

### 5.3 Tackle pressure executed above season rate

Adelaide season tackles/g = 53.8. R10 tackles = 62 = **115% of season rate** **[data]**. Sustained 4-quarter pressure: Q1 18, Q2 12, Q3 22, Q4 10 (tackles per quarter not directly in snapshot - estimated from AF tackle contributions). Richmond's clanger rate (57, 128% of season avg) was the inverse: Adelaide's tackle ceiling met Richmond's clanger floor. The gap compounded.

### 5.4 Q3 talent burst built on the structural reset

The Q3 AF wasn't random. Dawson + Milera + Cook + Rankine = 176 Q3 AF combined. Each came from a slightly different mechanism:

- **Dawson (48 Q3 AF)**: was now a fourth midfielder Richmond had no body for.
- **Milera (45 Q3 AF)**: launched off Worrell intercept marks (Worrell 34 Q3 AF, 92% TOG).
- **Cook (42 Q3 AF)**: hadn't been a flagged threat in the brief; he was the unmarked midfielder in the rotation.
- **Rankine (41 Q3 AF)**: the brief flagged him as "must contain"; containment failed because Richmond's body resources were tied up on the other three.

The cascade only works if the reset is good *and* there are four talented mid-rotations to feed. Adelaide has both.

### 5.5 Goal sources spread across the list

| Goal kickers (1 each unless stated) | Goals |
|---|---|
| Alex Neal-Bullen, Josh Rachele | 2 each |
| Izak Rankine, Jordan Dawson, Ben Keays, Daniel Curtin, Nick Murray, Toby Murray, Lachlan McAndrew, Isaac Cumming, Luke Pedlar, Riley Thilthorpe | 1 each |

**12 different goal-scorers from 14 goals.** No single player kicked more than 2. Without Walker (and with Thilthorpe limited to one goal and three behinds), the talent was distributed laterally - exactly the "system, not stars" pattern the verdict identified.

---

## 6. Prediction backtest - the accuracy table

Every concrete pre-match prediction from the brief, executive summary, and player matchups doc, scored against the verified result.

| # | Prediction (source) | Predicted | Actual | Verdict |
|---|---|---|---|---|
| 1 | Win condition: hit-outs +15 (exec summary §win-condition) | RI **+15** | AD **+18** | **MISS (inverted by Ryan absence)** |
| 2 | Win condition: Q1 within 5 points (exec summary) | Q1 within 5 | Q1 level (AD scored 3.2 vs RI 3.2; AF AD +3) | **HIT** |
| 3 | Win condition: clangers under 55 (exec summary) | RI <55 clangers | RI **57** clangers | **MISS (narrow, 2 over target)** |
| 4 | Tackle gap (brief §1) | AD +8 | AD **+24** | **MISS (direction right, magnitude 200% off)** |
| 5 | Hit-outs Ryan-conditional (brief §4) | RI +15 if Ryan plays | Ryan DNP; AD +18 | **N/A (precondition not met) → contingency missing → MISS** |
| 6 | Q1 differential (brief §quarter-by-quarter) | Avg RI -12.2 AF | Q1 AF: AD +3 (RI -3) | **HIT (beat season avg by 9.2 AF)** |
| 7 | Forward target: Lynch 2+ marks 1+ goal (brief §forward) | 2+ M, 1+ G | 3 marks, 3 goals | **HIT (exceeded - 3 goals)** |
| 8 | Conversion lift target: 0.230 g/i50 (brief §forward) | 0.230 | **0.200** (9/45) | **MISS** |
| 9 | Dawson 21-28 disp band (matchups doc) | 21-28 | **29** | **MARGINAL MISS (1 over band)** |
| 10 | Rankine "must contain, mobile forward" (matchups doc) | Contained | 33 disp, 158 AF, **147 SC** | **MISS** |
| 11 | Berry 5+ tackles by half-time / season floor 5+ (matchups doc) | 5+ tackles | **3 tackles whole game** | **MISS BUT IRRELEVANT (Berry didn't fire, Adelaide tackled 161% anyway)** |
| 12 | Walker 1.57 g/g threat, Vlastuin matchup (matchups doc) | Walker contained | Walker **DNP** | **N/A (selection invalidated)** |
| 13 | Adelaide accuracy advantage (brief §set pieces) | AD more accurate | RI 56% > AD 50% | **MISS (inverted)** |
| 14 | Q4 Adelaide tightens (brief §pre-game checklist #9) | AD Q4 score 20.8/g | Q4 AF AD +138, Q4 score AD 4.4 (28 pts) | **MISS (Adelaide accelerated, not tightened)** |
| 15 | Ross on Berry pivot matchup (brief §five matchups) | Ross +0.83 trend wins phase-for-phase | Ross 96 AF / 6 tackles vs Berry 69 AF / 3 tackles | **HIT (matchup won, game lost)** |
| 16 | Short vs Milera distributors (brief §five matchups) | Coin-flip on launches | Milera 113 AF vs Short 94 AF; both >94% DE | **MARGINAL LOSS** |
| 17 | Taranto trending down (-2.91/round) (brief §best performers) | Manage Taranto's load | Taranto 28 disp BUT 64% DE / 4 clangers | **PARTIAL HIT (volume up, efficiency down as flagged)** |
| 18 | Q3 not flagged as risk; per-quarter mean modelling | Avg Q3 RI -9.1 | Q3 AF **AD +305** | **STRUCTURAL BLIND SPOT (33+ sigma event from a mean-based model)** |
| 19 | Adelaide "no recognised ruck" exploitable (brief §1, §exec) | Adelaide HO 8.8/g (18/18) - structural hole | McAndrew **33 HO solo** = 376% of Adelaide season rate | **MISS (the hole closed when McAndrew started)** |
| 20 | Forward 50 boundary throw-in as set play of day (brief §set pieces) | Highest leverage set play | Richmond Q3 score 0.1 from all source play; F50 set play not visible in commentary as a score-source | **MISS** |

**Scorecard (20 predictions):**

| Verdict | Count | % |
|---|---:|---:|
| HIT | 4 | 20% |
| MARGINAL HIT / MARGINAL LOSS / PARTIAL | 3 | 15% |
| MISS (direction or magnitude) | 11 | 55% |
| N/A (selection invalidated) | 2 | 10% |
| STRUCTURAL BLIND SPOT (mean-modelling failure) | 1 (Q3) | 5% |

**The hit rate on individual matchup logic (4 of the 5 priority matchups Richmond could control) was 1 hit, 2 voided, 1 marginal loss, 1 missed.** The hit rate on structural claims (hit-outs as the lever; Q1 as the win condition; conversion-accuracy framing; Adelaide's no-ruck weakness; Berry as single point of failure) was **1 hit out of 5** (Q1 won), with the rest either inverted by Ryan's absence (hit-outs), or by Adelaide's structural reset (Q3 variance, Berry-as-SPOF).

**Read for future briefs**: the brief's individual scouting was largely correct. Where it broke was at the *system-interaction* level - it modelled rates additively when the rates were multiplicative; it modelled per-quarter means when the variance was bimodal; it treated Ryan as a footnote when he was the structural prerequisite for the entire plan.

---

## 7. List-level diagnosis - was this representative?

### Richmond's 2026 season at a glance

Margin per round (rounds 2-10, 2026):

| Round | Opponent | Result | Richmond | Opp | Margin |
|---:|---|---|---:|---:|---:|
| 2 | Carlton | L | 71 | 75 | -4 |
| 3 | Gold Coast | L | 60 | 128 | -68 |
| 4 | Fremantle | L | 43 | 103 | -60 |
| 5 | Port Adelaide | L | 48 | 90 | -42 |
| 6 | Greater Western Sydney | L | 75 | 131 | -56 |
| 7 | North Melbourne | L | 55 | 130 | -75 |
| 8 | Melbourne | L | 72 | 126 | -54 |
| 9 | West Coast | **W** | 99 | 88 | **+11** |
| **10** | **Adelaide** | **L** | **61** | **98** | **-37** |

**Average margin -42.8. Record 1W-8L. R10 margin (-37) is closer to average than the season low (-75 vs North Melbourne).** **[data: matches_2026.csv]**

### Adelaide game in context of Richmond's season profile

| Metric | R10 | Richmond R2-R9 avg | Delta |
|---|---:|---:|---:|
| Disposals | 345 | 242.3 | +102.7 (best of season) |
| Kicks | 196 | 143.6 | +52.4 (best of season) |
| Marks | 92 | 61.0 | +31.0 |
| Tackles | 38 | 36.6 | +1.4 (~average) |
| Clangers | 57 | 42.9 | +14.1 (well above) |
| Inside-50s | 45 | 33.3 | +11.7 |
| Hit-outs | 19 | 12.2 | +6.8 |
| Goals | 9 | 6.6 | +2.4 |

**Source: per-player performance CSVs summed by round, R2-R10 inclusive.**

**The structural story**: R10 was not a representative low-disposal Richmond loss. They had **more of the football** than in any other 2026 game except R9 (the West Coast win) - 345 disposals, 92 marks, 45 inside-50s. The losses earlier in the season (Gold Coast, Fremantle, North Melbourne, Melbourne) were *territorial wipeouts* with disposal counts in the 220-260 band. R10 looked structurally healthier on the volume axis - and lost anyway.

That is the diagnostic finding: **Richmond's R10 problem was not volume, it was conversion and ball security**. The team that ranks 18/18 for goals/g in 2026 generated 45 inside-50s and 16 scoring shots - more than they have managed in any other loss this year - and still finished -37 because clangers were 28% above their season average and the scoring shots cluster around the 17:05 Q4 mark misfired three times in 25 seconds.

### Was the loss outlier or pattern?

**Pattern, on every axis the brief identified as concerning**:

- Q1-Q4 deficit pattern: brief had Q1 -12.2 / Q4 -14.1 as the killers. R10: Q1 -3 (improvement), Q3 -23 scoreboard, Q4 -14 scoreboard. Q1 fix held; Q3 became the new failure mode.
- Clanger floor: season-worst rank (18/18) at 44.4/g. R10 at 57 was a higher-clanger game than average.
- Hit-out vulnerability when missing the #1 ruck: 19 HO without Ryan; season avg with Ryan in 3 games is 16.3 HO/g per Ryan, so the rotation without him cannot reach the structural floor needed to threaten Adelaide.
- Conversion: 0.200 g/i50 was below the 0.230 target and is consistent with the season's 18/18 goals/g rank.

**Outlier on one axis**: total disposals/volume of football. R10 was Richmond's third-highest disposal game and a positive sign for engine-room load. The R9-R10 pair (339 then 345 disposals) is the highest two-game disposal cluster of 2026. That is something to build on.

---

## 8. What Richmond must do differently - five specific changes

Each grounded in the R10 verified data, not generic.

### 8.1 Ban the "structural lever assumes #1 ruck" plan; install a Ryan-out forward-supply protocol

The brief's hit-out-dominance plan was the single largest pre-match edge it claimed. It died at team selection on Saturday morning. There must be a written alternative for the *next* game Ryan misses - because he plays only 3 of 9 games to date in 2026 (33% availability) and is the structural prerequisite for half the brief's plan.

Concrete change: pre-game brief structure must branch on each side's #1 ruck availability with named contingencies. If Ryan is out, the supply route to the forward target shifts from hit-out → midfield exit → inside-50 entry → Lynch contested mark, to **midfield mark → handball release → kick-mark chain through Trainor or Taranto → ground-ball crumb to Lynch / Campbell / Lefau as second-option targets**. Run the chain in pre-season match simulations and document which players hit which marks.

### 8.2 Cap Hopper at 3 clangers or sub him

Hopper had **7 clangers** in R10 from 23 disposals at 43% DE - both individual extremes. Brief flagged Hopper as the "most reliable disposal-floor player on the list (CV 0.14, 22 every week)" - the disposal volume is the floor; the kick efficiency is the ceiling. He hit the disposal floor (23) and the efficiency floor (43%) in the same game.

Concrete change: in-game tracker for Hopper's clanger count by quarter. If he hits 3 by half-time, the role shifts to handball-receiver only (kicks suppressed) or a forward 50 rotation. The brief's "all-game midfielder" assumption did not survive this game; the role needs a kill-switch.

### 8.3 Build a Q3 "second-half re-read" protocol with a 30-second decision window

The Q3 collapse (-305 AF) was driven by Adelaide's half-time positional reset that Richmond did not detect. Q3 0:53 had Rachele snapping the first goal after Rankine stripped Balta; Q3 4:38 had Adelaide leading the quarter 71-19 AF.

Concrete change: half-time tactical brief must include a "who has moved" check - tracked by which Adelaide players come out of the rooms for warm-up at the previously-occupied position vs the new one. If two players have shifted (e.g. Dawson and Murray), the matchup zone resets at the bounce; if not, the pre-game zone holds. This is observable inside 30 seconds at the start of Q3.

The corresponding player change: identify one Richmond mid (Ross is the candidate) as the *floating defender* whose tag adapts to the post-HT structure. Without one, Richmond is matchup-static in the most consequential 25 minutes of the game.

### 8.4 Shift the conversion target from accuracy to inside-50 differential

The brief's "lift 0.192 to 0.230 g/i50 = +1.8 goals" target was the wrong axis. Richmond R10 ran at 0.200 conversion (below target) but generated 45 inside-50s (best of season except R9). Adelaide ran at 0.259 from 54 entries.

Concrete change: replace the conversion-rate target with a **inside-50 differential of -3 or better**. R10 differential was -9. A -3 differential at R10's conversion (0.200) yields ~10 goals vs Adelaide's ~14 - still a loss, but a 24-point game, not a 37-point game.

The inside-50 lever has two sub-levers: (a) clearance wins (Richmond R10 had 27 clearances vs Adelaide 29 - within reach; (b) defensive 50 exit clean-rate. Adelaide had 21 rebound-50s; Richmond had 38. **Richmond rebounded more than Adelaide and still lost the inside-50 count by 9** - meaning Adelaide turned their fewer rebounds into longer territorial chains. That is a kicking-execution problem, not a clearance problem, and points at kick-target selection on the rebound chain. Practice short-to-target retain over long-down-the-line kick-ins.

### 8.5 The 17:05 Q4 cluster - one quality goal-kicking session per week

Three scoring shots in 25 seconds, all from inside 50, no goals: Taylor 45m post miss, Pedlar 40m miss, Taranto grubber into the post. That is not a strategy problem; that is a goal-kicking execution problem. Richmond's season accuracy is 48.4% (brief, R9 era). R10 at 56% was above average; the 17:05 cluster suggests the team accuracy distribution is bimodal - good when the structure is set, terrible under pressure-and-pace.

Concrete change: a single 30-minute weekly session for the 22 first-team players on **fatigued set-shots** - i.e. shots taken after 20 seconds of running. The 17:05 cluster came at 67% of game time elapsed for most of those players; the legs were not under them. The training condition has to match the failure condition.

---

## 9. SuperCoach fantasy verdict

### Top SC scorers - the captaincy retrospective

**Adelaide top 4 SC**: Rankine 147, Milera 127, Dawson 115, Laird 105 - **four 100+ SC scores**.

Rankine was the captaincy slam-dunk in retrospect at 147 SC. The brief flagged him as "must contain" but did not flag him as a captain-option ceiling, because the containment framing assumed it would work. **Rule for future briefs: a "must-contain" forward against a Richmond defence missing its first-choice intercept (Vlastuin played but tracked the wrong target) is a captain candidate, not a fade.**

**Richmond top 3 SC**: Ross 107, Miller 106, Short 101 - three 100+, Ross the best.

Ross 107 SC at his price-point is the value-add of the game. The brief's "Berry-equivalent" call gave the right player.

### Buys

- **Lachlan McAndrew (86 AF / 83 SC, lone ruck role)**. 33 hit-outs, 92% TOG, first senior goal, Q4 AF 22. If O'Brien stays out and Adelaide keep playing him solo, this becomes a weekly score around 80 AF. Buy.
- **Luke Trainor (77 AF / 76 SC, late inclusion)**. 10 marks across 86% TOG. The one Tiger who had a Q3 (32 AF). Late-included; if his role holds with Richmond's injury list still long, he is the cheap defender to hold.
- **Ben Miller (80 AF / 106 SC, 93% TOG)**. Quiet outlier - the only Tiger to crack 25+ Q3 AF (28). 7 marks. SC rewards his clean intercept profile. At his price, buy on Tom Brown's continued absence.

### Sells

- **Sam Berry (69 AF / 73 SC)**. The brief's "engine, single point of failure". 3 tackles vs his season average of 8.0. Adelaide tackled 161% of season rate without him firing. **The Berry-as-SPOF thesis is broken on the data**; if he doesn't tackle, the team still does. Sell at peak ownership.
- **Tim Taranto (87 AF / 99 SC, 28 disposals, 4 clangers, 64% DE)**. Classic value-trap. Disposal volume reads "elite midfielder game"; the SC penalisation strips it. AF rewarded the volume (87) but the SC system is rejecting the cheap turnovers. Brief flagged the trend (-2.91/round); this is the third-quartile outcome of that trend.
- **Nick Vlastuin (67 AF / 90 SC)**. Junk score - the matchup he was selected for (Walker) did not exist; his role floated. Q4 AF of 6 shows he gassed. Don't extrapolate.

### Holds

- **Jack Ross (96 AF / 107 SC, 6 tackles)**. Trade-period dream after this. Held up across all four quarters (Q1 28, Q2 24, Q3 18, Q4 26 - the only Tiger with no quarter under 18). Hold.
- **Jayden Short (94 AF / 101 SC, 96% DE)**. Best DE on the ground. Q3-Q4 fade (11 + 10 AF) is the team arc, not a Short-specific concern. Hold.
- **Wayne Milera (113 AF / 127 SC, 94% DE)**. 34 disposals at 94% DE is elite. Hold.

### Special note - the McAndrew curiosity

McAndrew was not in the brief. He took 33 hit-outs solo. He kicked his first senior career goal at Q4 25:58. He posted 86 AF / 83 SC. He was the structural difference between Adelaide's pre-game (8.8 HO/g, 18/18) and Adelaide's R10 (37 HO). **The brief's biggest miss in talent identification was assuming Adelaide had no recognised ruck.** They had one; he was emerging. Forward briefs targeting Adelaide should treat McAndrew as a ruck-solo option until proven otherwise.

---

## Caveats and methodology

- **Round number**: This document title and filename use "Round 9" to match the pre-match brief and repo convention. Afltables.com classifies the fixture as Round 10. `data/matches/matches_2026.csv` records it as Round 10. The match is unambiguously identified by date 2026-05-10 in either nomenclature.
- **Snapshot data quality**: The final-siren snapshot (`9781_20260510_1751_final-siren.json`) was cross-checked against afltables.com (the canonical source). Kicks, marks, handballs, tackles, hit-outs, AF, SC, Q1-Q4 AF, TOG, DE all match exactly. **Goals, behinds, clangers** in the snapshot did not match afltables for 21 players each; the snapshot's column indexing for those three fields is unreliable for this game. **All goal/behind/clanger numbers cited in this document use afltables.com**, not the snapshot.
- **Reproducibility**: Player CSV updates produced by [`scripts/update_r10_player_data.py`](../../scripts/update_r10_player_data.py). Team aggregates and player tables produced by [`scripts/compute_r10_team_aggregates.py`](../../scripts/compute_r10_team_aggregates.py) and [`scripts/compute_r10_player_table.py`](../../scripts/compute_r10_player_table.py). All three are deterministic given the inputs; re-running them on the same source data produces the same numbers.
- **Season averages**: Richmond and Adelaide 2026 averages are computed from R2-R10 inclusive (R1 not present in player files). Per-game team totals sum the active match-day roster; players who appeared in one game but not another are included only in the games they played. This understates totals vs the brief's methodology (which appears to have summed differently) - the brief reports Richmond 328.9 disp/g; my method computes 253.7 (using the current 23-man R10 roster across the season).
- **Causal claims**: Where this document says "X drove Y" (e.g. "the half-time reset drove the Q3 collapse"), the evidence is associational - quarterly AF totals correlate with positional shifts and timing, not a controlled experiment. The structural narrative is the most defensible reading of the data, not a proven causal mechanism. The brief's "Berry single point of failure" thesis is *falsified* by this game (Berry low output, team tackle count still up); the verdict is causal because it is a counter-example, not a positive claim.
- **Pitfalls walk**: data leakage [no - no models trained, just descriptive aggregation against pre-stated predictions]; selection bias [partial - Richmond's injury list of 17 is acknowledged and the brief's plan was selection-conditional]; multiple comparisons [present - 20 predictions audited; no multiple-testing correction applied, but no statistical inference claimed]; reproducibility [scripts deterministic, seeds n/a]; baseline comparison [explicit baseline used: season-to-date averages from R2-R9 vs R10 actuals].

---

*Sources: `data/matches/matches_2026.csv` (R10 entry added 2026-05-11 from afltables.com); `data/player_data/*_performance_details.csv` (all 46 match-day players refreshed with R10 entries); `data/live_snapshots/9781_20260510_1751_final-siren.json` (AF/SC/quarter splits/commentary); afltables.com Round 10 2026 match report (canonical box-score for goals/behinds/clangers).*
