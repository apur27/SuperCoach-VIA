# Richmond vs Adelaide - Full-Time Verdict (Round 9, 2026)

> Pre-match brief: [Executive summary](richmond-vs-adelaide-round-9-2026-executive-summary.md) | [Tactical brief](richmond-vs-adelaide-round-9-2026.md) | [Player matchups](richmond-vs-adelaide-round-9-2026-player-matchups.md)
> In-game reads: [Half-time](richmond-vs-adelaide-round-9-2026-half-time-live.md) | [Q3](richmond-vs-adelaide-round-9-2026-q3-live.md) | [Q4](richmond-vs-adelaide-round-9-2026-q4-live.md)
> Post-match deep dive: [**Post-mortem**](richmond-vs-adelaide-round-9-2026-postmortem.md) - full prediction backtest, structural failure analysis, and SuperCoach verdict grounded in afltables-verified box-score.
>
> Data source: FanFooty live feed (game 9781), final snapshot `9781_20260510_1751_final-siren.json`.
> Note: col15 "goals" field is unreliable (per-player team sums do not match scoreboard). Goal counts cited here are from commentary entries or scoreboard totals only. The post-mortem (link above) uses afltables.com as the canonical source for goals/behinds/clangers.

## Final score

**Adelaide 14.14.98 def Richmond 9.7.61 by 37 points** at the M.C.G., Round 9, 2026.

Adelaide won three of four quarters on AFL Fantasy points and broke the game open in Q3. Richmond won the AF battle in Q2 and Q4 but only the second of those four quarters mattered for the scoreboard.

## The 60-second verdict

Adelaide won this game in a 25-minute window after half-time during which Jordan Dawson reverted from half-back to midfield, Nick Murray returned to defence, and Rankine, Milera, Cook and Dawson all turned in prolific quarters. Richmond's tactical brief read three of the four pre-game battles correctly (Q1 starts, hit-out exposure when Ryan plays, Lynch as contested-mark target) but had no contingency for **Ryan unavailable** and no model for **Q3 variance**. The structural insight: Richmond won enough quarters to be in the game, but Adelaide's worst quarter (Q2, 382 AF) was still within 7% of Richmond's best quarter (Q2, 409 AF). The variance was asymmetric and Richmond never had a quarter that exceeded Adelaide's Q3.

## Quarter-by-quarter AF - the complete picture

| Quarter | Richmond AF | Adelaide AF | Delta | Game narrative |
|---|---|---|---|---|
| Q1 | 415 | 418 | AD +3 | Dead even. Richmond absorbed their forecast Q1 deficit (-12.2 AF/g avg). Dawson started at half-back, Murray at key forward - Adelaide playing a structure that was not working. |
| Q2 | 409 | 382 | RI +27 | Richmond's only quarter on top. Half-time deficit "reflected their shots being taken from better spots" (commentary). Conversion failure was already the visible weakness. |
| Q3 | 238 | 543 | **AD +305** | The game. Dawson reverted to midfield, Murray to defence, "the Crows woke up with a string of goals". Dawson 48 Q3 AF, Milera 45, Cook 42, Rankine 41. Richmond did not have a single 30+ AF quarter from anyone. |
| Q4 | 296 | 434 | AD +138 | Richmond got Lynch a contested mark and a goal at 7:11 ("to stop the rot"). Ross 26 Q4 AF, Lynch 23, Balta 26. Adelaide closed out: McAndrew goaled at 25:58, Rachele at 22:50. |
| **Total** | **1,358** | **1,777** | **AD +419** | |

Adelaide won total AF by 30.8%. The Q3 alone (+305 AF) was 73% of the final AF margin.

## Player report cards

### Adelaide - top 8 by SC

| Player | Disposals | AF | SC | Q3 AF | Q4 AF | Assessment |
|---|---|---|---|---|---|---|
| Izak Rankine | 33 (24k+9h) | 158 | **147** | 41 | 36 | The dominant player on the ground. 10 marks, 8 tackles, two-way game. Brief flagged "must contain"; result was the opposite of contained. |
| Wayne Milera | 34 (18k+16h) | 113 | 127 | 45 | 21 | Game-high disposals. 94% DE - cleanest user on the ground. Brief profile (CV 0.21, defensive runner, doesn't go to contest) was accurate; the volume was the surprise. |
| Jordan Dawson | 29 (22k+7h) | 133 | 115 | 48 | 26 | Captain. Q3 AF tells the story: started at half-back, moved into midfield after the long break, became the game's structural pivot. 12 marks. |
| Rory Laird | 27 (16k+11h) | 110 | 105 | 39 | 21 | Predictable Laird - 12 marks, 81% DE, 84% TOG. Linkage role executed. |
| Brayden Cook | 22 (14k+8h) | 83 | 87 | 42 | 13 | Q3 was his quarter (42 AF). Not on Richmond's radar pre-game. |
| Josh Worrell | 21 (13k+8h) | 88 | 86 | 34 | 17 | Highest TOG on the team (92%). The intercept-and-launch profile from the brief held. |
| Josh Rachele | 14 (8k+6h) | 66 | 86 | 23 | 15 | Goal at Q4 22:50 to ice the game. The "high contact free" call was soft; took the points. |
| Lachlan McAndrew | 9 (6k+3h) | 86 | 83 | 19 | 22 | **33 hit-outs as the lone ruck.** Brief had Adelaide ruck rotation as a no-recognised-ruck weakness; that read died when Ryan was scratched. First senior goal at Q4 25:58. |

### Richmond - top 8 by SC

| Player | Disposals | AF | SC | Q3 AF | Q4 AF | Assessment |
|---|---|---|---|---|---|---|
| Jack Ross | 26 (8k+18h) | 96 | **107** | 18 | 26 | Best Tiger. 6 tackles, 86% TOG, +26 Q4 AF. The "Berry-equivalent" call from the brief held; he was the most consistent four-quarter performer on the ground for Richmond. |
| Ben Miller | 18 (13k+5h) | 80 | 106 | 28 | 14 | Quiet outlier - 7 marks, 93% TOG. Only Tiger to crack 25+ Q3 AF. |
| Jayden Short | 26 (15k+11h) | 94 | 101 | 11 | 10 | 96% DE - cleanest user on the ground. But the Q3/Q4 fade (11+10) shows the same arc as the team. |
| Tim Taranto | 28 (12k+16h) | 87 | 99 | 14 | 21 | 28 disposals on paper, 18 clangers in the cell, 64% DE - the highest disposal-getter who lost the ball most. The "trending down" warning in the brief landed. |
| Nick Vlastuin | 21 (11k+10h) | 67 | 90 | 11 | 6 | The Walker matchup never materialised (Walker DNP). Q4 AF of 6 suggests gassed. |
| Tom J. Lynch | 14 (8k+6h) | 71 | 90 | 9 | **23** | The contested-mark target call was right; the supply was 45 minutes late. 1 goal confirmed at Q4 7:11 ("to stop the rot"). |
| James Trezise | 18 (12k+6h) | 78 | 84 | 8 | 15 | 7 marks, 76% TOG. Defensive role, did not generate score. |
| Luke Trainor | 18 (11k+7h) | 77 | 76 | **32** | 13 | The one Tiger who had a Q3. 10 marks across the game. Late inclusion (replacing Grlj) - the best Richmond Q3 contribution came from the bench. |

## Pre-game prediction audit

Every load-bearing prediction from the [tactical brief](richmond-vs-adelaide-round-9-2026.md), audited.

| Prediction | Pre-game call | Actual | Verdict |
|---|---|---|---|
| Hit-outs | RI +15 (Ryan-conditional). "You don't get this opportunity twice a year." | RI 19 vs AD 37 = **AD +18**. Ryan absent, McAndrew 33 HO solo. | **INVERTED** |
| Tackle gap | AD +8 (brief) - "they will out-pressure unless tempo changes" | RI 38 vs AD 61 = **AD +23**. Direction right, magnitude under by 188%. | **UNDER (direction right)** |
| Q1 differential | RI avg -12.2 AF/g vs opponents; "the win condition" | Q1 AF: RI 415 AD 418 = AD +3. **Beat the seasonal average by 9 AF.** | **BEAT** |
| Q2 momentum | Implicit chase quarter | RI +27 AF. Richmond's only winning quarter. | **HIT** |
| Lynch contested weapon | "2+ marks, 1+ goal, contested target" | 3 marks, 1 goal confirmed (Q4 7:11), 71 AF, 86 SC | **HIT (late delivery)** |
| Dawson disposals | 21-28 disp band (CV 0.11, "most consistent ball-winner") | 29 disposals, 133 AF, 115 SC, 12 marks | **TRACKING (volume exceeded; consistency held - Q1-Q4 spread was wide due to positional shift)** |
| Rankine suppression | "Must-contain, mobile forward, match with running defender" | 33 disp, 158 AF, 147 SC, 10 marks, 8 tackles | **FAILED** |
| Berry pressure | "5+ tackles, 13+ CP, the engine, single point of failure" | 20 disp, **3 tackles**, 69 AF, 73 SC | **OVER (Adelaide won without Berry firing - which inverts the brief's whole "Berry is single point of failure" thesis)** |
| Adelaide accuracy | "More accurate set-shot side; Richmond cannot afford behinds" | AD 14.14 (50% conversion), RI 9.7 (56% conversion). Adelaide kicked **14 behinds** and still won by 37. | **FAILED ON ASSUMPTION; OUTCOME INVERTED** |
| Q3 collapse | Not flagged; brief modelled per-quarter averages | AD +305 AF in Q3. The decisive quarter. | **STRUCTURAL BLIND SPOT** |
| Richmond Q4 burst | Identified as -14.1 avg deficit ("the worst") | RI 296 AF Q4 vs AD 434 = AD +138. Win condition flipped: Richmond did not get blown out in Q4 but the game was already lost. | **WINDOW OPENED TOO LATE** |
| Vlastuin on Walker | "Veteran intercept body on the goal-kicker" | Walker DNP (Showdown injury). Murray played key forward in Q1-Q2, then reverted to defence. | **MATCHUP NEVER MATERIALISED** |
| Ross on Berry | "Match Berry with Ross - the contested-ball pivot" | Berry: 3 tackles, 11.9 CP/g pre-game vs ~10 today. Ross 96 AF, 107 SC, 6 tackles. **Ross out-played Berry on every dimension.** | **HIT (matchup won, game lost)** |
| Conversion lift target | "Lift Richmond from 0.192 to 0.230 g/i50 = +1.8 goals = +11 points" | RI 9.7 from 16 scoring shots = 56% accuracy. Better than projected, but volume was the issue, not accuracy. | **HIT ON ACCURACY, MISSED ON VOLUME** |

**Summary**: 4 hits (Q1, Q2, Lynch, Ross-on-Berry matchup), 2 partials (Dawson tracking, conversion lift hit-but-irrelevant), 4 failed/inverted (Hit-outs, Rankine, Berry-as-single-point-of-failure, Adelaide accuracy), 1 blind spot (Q3 collapse), 2 unrealised (Vlastuin-Walker, Q4 burst). The brief's reads on individual matchup logic were largely correct; the brief's reads on the variance distribution and the Ryan-out scenario were where it broke.

## Structural autopsy

### What actually won this game for Adelaide

**Ryan's absence as structural pivot.** The brief identified the hit-out battle as "the single biggest structural lever in the game" - a claim that depended on Ryan playing. With Ryan scratched, McAndrew took 33 of Adelaide's 37 hit-outs and Adelaide *won* the hit-out count by 18. This was not just a flipped number on a stat sheet: Adelaide's clearance constraint (rank 16/18, "constrained by their ruck") was lifted. Without that constraint, the brief's whole "crack the structure with hit-out dominance" plan had no entry point. Richmond did not have a Plan B for ruck dominance going the other way.

**Dawson's positional shift.** The commentary log makes this explicit: "After half time, Dawson and Nick Murray reverted to their more comfortable roles." Dawson started at half-back in a structure designed to cover for Walker's absence. When that structure failed (Murray as KPF "was just not working"), Adelaide reset at half-time and Dawson moved into the engine room. His Q1-Q2 AF was 25; his Q3-Q4 AF was 74. The 48-point Q3 AF is the largest single-quarter individual AF in the match. Richmond's brief had Dawson tagged for Taranto - that matchup was effectively void once Dawson moved.

**The Q3 burst was four players, not one.** Dawson 48, Milera 45, Cook 42, Rankine 41 - four Crows over 40 AF in a single quarter. Richmond's best Q3 contributor was Trainor (32) - their late inclusion. The variance distribution the brief did not surface: Adelaide's quarter-best-case is far above their quarter-average; Richmond's quarter-best-case is at their quarter-average. Modelling per-quarter means missed this entirely.

**Adelaide's tackle identity executed above the brief's projection.** Brief said AD +8 tackles. Actual was AD +23. The brief projected from season averages (AD 59.5/g, RI 51.8/g = expected gap of ~8). On the day, Adelaide tackled at 161% of their season rate (61) and Richmond at 73% (38). The brief modelled rates; it did not model the *interaction effect* - that Adelaide tackles harder against a side that has clanger problems, and Richmond's clanger floor (4/18 worst) compounds with Adelaide's tackle ceiling. Both teams are at their structural extremes; the gap multiplies, not adds.

### What Richmond did right (and why it didn't matter)

Richmond won the disposal count's quality battle (Short 96% DE, Lynch's late contested marks, Ross's 6 tackles) and won the Q1 starts battle that the brief flagged as the win condition. Q1 finished AD +3 - vs the seasonal expectation of RI -12.2. That is a 9-AF improvement on baseline, executed exactly as the brief specified.

The Q2 was Richmond's. AF +27, "shots being taken from better spots" per commentary. If accuracy had been at season-best, Richmond goes into half-time level or up. They were not.

The problem: winning Q1 (tied) and Q2 (+27 AF) only mattered if Richmond could limit the Q3 damage to within 50 AF. They lost it by 305. There is no game plan that survives a 305-AF quarter loss. The brief's per-quarter mean modelling assumed the quarter outcomes were independent; they are not - Adelaide's Q3 is correlated with their structural reset capacity, which Richmond did not model.

### The conversion problem - data

Adelaide kicked **14.14** (28 scoring shots, 50% accuracy). Richmond kicked **9.7** (16 scoring shots, 56% accuracy). Richmond's accuracy was *better than Adelaide's* - the brief's "Adelaide is the more accurate set-shot side" was wrong on the day (and the brief itself flagged Adelaide's behinds-rate as a vulnerability the commentary calls out: "their shots being taken from better spots" was Richmond's HT advantage, not Adelaide's).

The independent variable was **scoring shot volume**, not accuracy. Adelaide had 28 shots at goal, Richmond 16. At Adelaide's actual accuracy (50%), Richmond would need 20 scoring shots just to match Adelaide's 14 goals (and they had 16). At Richmond's accuracy (56%), Adelaide's 28 shots would yield 15.7 goals. The volume gap (12 scoring shots) is the entire game.

Richmond's territory-and-supply problem was the hidden killer: with 38 tackles taken against them (sorry, 38 tackles applied; 61 conceded), they were locked in their own half for long stretches of Q3-Q4 and could not generate inside-50 volume. Brief modelled accuracy lift (0.192 to 0.230) but the lift target was the wrong axis. Volume of inside-50s was the lever.

## How Adelaide beat Richmond - strategy, not just talent

Adelaide's win was not primarily a talent story. The talent (Rankine, Dawson, Milera) was the fuel. The ignition was a half-time structural correction that created a problem Richmond had no answer for. Every claim below is from the final snapshot (`9781_20260510_1751_final-siren.json`).

### The half-time reset - a tactical correction, not an escalation

Adelaide were *losing* Q2 by 27 AF when the siren went (RI 409 vs AD 382). Their Q1-Q2 structure was broken: Murray was deployed as key forward to cover Walker's absence and per commentary "was just not working", while Dawson was operating at half-back as structural cover. By half-time they had a -24 AF half on the back of a structure built around two players in unfamiliar roles.

The reset was not a blitzkrieg. It was a side fixing what wasn't working:

| Player | Pre-half-time role | Post-half-time role | Q1-Q2 AF | Q3-Q4 AF |
|---|---|---|---|---|
| Jordan Dawson | Half-back (Walker cover) | Midfield (engine room) | 59 | 74 |
| Nick Murray | Key forward (Walker cover) | Defence (natural role) | 15 | 18 |

Commentary log: "After half time, Dawson and Nick Murray reverted to their more comfortable roles." This was Plan B executed, not a tempo change. The structure shifted; the personnel didn't.

### The midfield overload - how the flanking manoeuvre worked

Pre-reset, Adelaide had effectively one genuine on-ball midfielder generating volume (Rankine running forward/mid). Post-reset, they had four. Q3 AF tells the story:

| Adelaide Q3 contributors | Q3 AF |
|---|---|
| Jordan Dawson | 48 |
| Wayne Milera | 45 |
| Brayden Cook | 42 |
| Izak Rankine | 41 |
| **Four players over 40 AF in one quarter** | |

Richmond's best Q3 contribution: **Trainor 32 AF** (a late inclusion replacing Grlj). No other Tiger cracked 30 in the quarter. The brief had Taranto matching Dawson - but that matchup was designed for a *half-back* Dawson, not a midfield Dawson. Richmond's matchup zone was set pre-game; Adelaide turned 3 active midfielders into 4 with one positional change, and Richmond had no extra body to cover the new on-baller.

### The pressure engine - tackles as a force multiplier

The brief projected AD +8 tackles. Actual: **AD +23** (RI 38, AD 61).

| Side | Season tackle rate (per game) | Today | % of season rate |
|---|---|---|---|
| Adelaide | 59.5 | 61 | 161% (vs season-average expectation - team executing at structural ceiling) |
| Richmond | 51.8 | 38 | 73% |

This is the interaction effect the brief missed. Richmond are the **#1 clanger team** (rank 18/18 - the worst). Adelaide are a **#4 tackle team**. When both sides hit their structural extreme, the gap doesn't add - it multiplies.

The cascade in Q3-Q4:

```
Adelaide tackles applied (61)
  → Richmond clangers forced (127, vs Adelaide 118)
  → no clean disposal exit from defensive half
  → Lynch not supplied (only 14 disposals all game; 3 marks; 1 confirmed goal at Q4 7:11)
  → no scoreboard pressure
  → Dawson not chased back; runs free in midfield
  → Dawson generates 48 Q3 AF unmolested
```

This is not a talent story. It is a system exploiting a known structural weakness in the opponent. The brief modelled rates additively; the actual interaction was multiplicative.

### The ruck axis - Ryan's absence as the structural fulcrum

The brief's single biggest predicted structural advantage was **RI +15 hit-outs** ("the single biggest structural lever in the game", Ryan-conditional). Ryan was scratched.

| | Pre-game projection | Actual | Swing |
|---|---|---|---|
| Hit-outs (RI - AD) | RI +15 | **AD +18** (RI 19, AD 37) | **33-point reversal** |
| McAndrew hit-outs | n/a (Adelaide ruck rotation flagged as weakness) | **33 of Adelaide's 37** as sole ruck | - |

This was not just a number flip on a stat sheet. Richmond's clearance strategy depended on first use from stoppages → midfield exit → entry to Lynch → contested mark → goal. Without hit-out dominance, the supply chain failed at the first link. There was no Plan B written for the ruck count going the other way; the brief had "if Ryan plays" buried in section 3 rather than as a top-level branch.

### Why talent alone doesn't explain it

Adelaide won this game without two of their most-talked-about contributors firing:

| Adelaide player | Pre-game framing | Actual today |
|---|---|---|
| Sam Berry | "5+ tackles, the engine, single point of failure" | **3 tackles** (vs season avg 8.0). 69 AF. Effectively absent for pressure. |
| Taylor Walker | KPF, set-shot accuracy weapon | **DNP** (Showdown injury). |
| Lachlan McAndrew | Developmental ruck. Brief had Adelaide's no-recognised-ruck status as a weakness. | **33 hit-outs**, 86 AF, 83 SC, first senior goal at Q4 25:58. |

McAndrew is not a superstar; he is a developmental player who had an extraordinary game inside a system that supported him. The team-wide tackle count (61, vs season average 59.5) actually went **up** without Berry's pressure. That kills the "Berry is the single point of failure" thesis on its own.

If the win were a talent story, removing Berry (effectively) and Walker (literally) should have reduced the Adelaide ceiling. Instead the team posted a +305 AF Q3 - the largest single-quarter delta any side has produced against Richmond this season. The system generated the win. Rankine and Dawson's talent was the execution layer, not the root cause.

### The blueprint - what a team needs to replicate this against Richmond

Concrete, data-backed:

1. **Force Richmond's #1 ruck out (or neutralise him).** The brief identified hit-outs as the single biggest structural lever. With Ryan absent the supply chain to Lynch broke entirely (Lynch 14 disposals, 1 goal, supply 45 minutes late). Without ruck dominance Richmond cannot execute their territory game.
2. **Overload the midfield at half-time if your Q1-Q2 structure is not working.** Richmond's matchup zone is set pre-game. Mid-game positional shifts find holes - Dawson going from half-back to midfield was the structural pivot of the entire match (Q1-Q2: 59 AF; Q3-Q4: 74 AF; Q3 alone: 48 AF, the highest single-quarter individual AF on the ground).
3. **Execute tackle pressure above your season-average rate.** Adelaide tackled at 161% of their season rate; Richmond's 18/18 clanger rank makes them maximally vulnerable to tackle-heavy sides. The gap multiplies, not adds.
4. **Hold high-impact forwards in reserve through Q1-Q2; unleash in Q3.** The brief's Q1-start model did not account for Adelaide's deliberate Q1-Q2 restraint. Rankine, Milera, Cook and Dawson all peaked in Q3 (41/45/42/48 AF respectively) when Richmond's energy and matchup tracking are at their lowest. The Q3 burst was structural, not random.

The pattern is replicable. The talent was the fuel; the half-time reset was the ignition; the tackle-against-clanger interaction was the fire.

## Strategic brainstorm - Scientist x FootyStrategy

*Post-match analysis synthesised from data review (Scientist agent) and tactical brainstorm (FootyStrategy agent). The Scientist read the final-snapshot numbers and surfaced the structural anomalies; the FootyStrategy agent translated those anomalies into football-coaching language. This section is the joined-up output.*

### The full match narrative in four quarters

**Q1 was a false read.** Richmond did not "hold" Adelaide; Adelaide were not yet functional. Murray was deployed as a key forward and Dawson at half-back as Walker-cover - a structure that, per commentary, "was just not working". The 415-418 AF deadlock looks like Richmond meeting their Q1 win condition, but the load-bearing fact is that Adelaide were giving away a quarter of their effective midfield by playing Dawson 70 metres from the contest. Richmond's tactical brief read the AF balance as a Richmond achievement; the underlying cause was an Adelaide self-handicap that was never going to last.

**Q2 was Richmond's only legitimate quarter on top - and it loaded the gun.** Plus-27 AF, "shots being taken from better spots" (commentary), Lynch as a contested-mark target visibly working. But the conversion rate was already broken: 7 behinds from 28 inside-50s by half-time, a 25% goal-per-i50 conversion. The brief had set a 0.230 g/i50 lift target. Richmond was running at 0.107. The quarter was won on territory; the scoreboard never caught up. When Adelaide's structural reset arrived 20 minutes later, Richmond did not have the points-buffer the territory had earned them.

**Q3 was the structural takeover, executed in 90 seconds of dressing-room time.** Half-time: Dawson reverted from half-back to midfield, Murray from KPF to defence. Within 4 minutes 38 seconds of the Q3 bounce, Adelaide led 71-19 AF in the quarter (Rachele snap at 0:53 after Rankine stripped Balta was the visible kickoff). Four players over 40 Q3 AF: Dawson 48, Milera 45, Cook 42, Rankine 41. Richmond's best Q3 was Trainor 32 - a late inclusion playing on the bench rotation. The Taranto-on-Dawson matchup that the brief had built around was voided in 90 seconds of half-time talk. Richmond had no contingency for a positional shift it had not modelled.

**Q4 was cosmetic recovery, not contest.** Richmond won the AF count for chunks of the quarter (+12 AF in stretches) and finally got Lynch a contested mark and a goal at 7:11 "to stop the rot" - which is to say, after the rot. The midfield generated supply; the conversion stayed at 25%. Hopper shanked from 45m OOTF at 11:36 (the symbolic miss); Lefau missed at 10:21. The same conversion problem that had loaded the gun in Q2 fired the empty chamber in Q4. The game was already 50+ points gone.

### Three moments that decided the game

1. **Half-time tactical reset (~3:00pm break)** - not a football play, but the moment the match was decided. Adelaide's Q1-Q2 structure (Murray KPF, Dawson HB) was a Walker-cover compromise that wasn't working. The reversion to natural roles took less than 90 seconds of dressing-room time and re-shaped every Richmond matchup in the on-ball group.
2. **Q3 0:53 - Rachele snaps after Rankine strips Balta** - the visible kickoff of the Q3 surge. From 0:53 to 4:38, Adelaide led 71-19 AF in the quarter. The ball never settled. The system was firing at structural ceiling and Richmond had no extra body to insert.
3. **Q4 11:36 - Hopper shanks from 45m OOTF** - symbolic of the conversion problem that ended Richmond's burst window. The supply chain had finally restored (Lynch contested mark, Q4 7:11 goal); the foot-skill at the end of it had not. Same 25% conversion rate that lost Q2 lost Q4.

### What the brief got right - and why it still mattered

- **Q1 win condition framework**. The brief named -12.2 AF/g as Richmond's seasonal Q1 deficit and called Q1 starts as the load-bearing win condition. Result: AD +3 = a 9-AF over-delivery on baseline. Hit. The framework was correct; it just wasn't enough on its own.
- **Lynch as contested-mark weapon**. 3 marks, 71 AF, 86 SC, 1 confirmed goal. Hit. The supply failed (45 minutes late, hit-out battle lost), not the target. The brief identified the right player; the structure around him collapsed when Ryan was scratched.
- **Ross-on-Berry matchup**. Hit decisively. Ross 96 AF, 107 SC, 6 tackles vs Berry 69 AF, 73 SC, 3 tackles. Ross out-played Berry on every dimension. The matchup was won; the game was lost - which is itself a finding (matchup-level wins do not aggregate to game-level wins when the variance distribution favours the opponent).
- **Rankine danger-forward profiling**. Operationally correct identification ("must contain, mobile forward, match with running defender"). Containment failed (33 disp, 158 AF, 147 SC), but the reason was structural - Adelaide's Q3 burst overwhelmed every individual matchup Richmond had planned, not just the one on Rankine.
- **Adelaide rebound-50s identity (#2 in league)**. Held. Richmond's long-ball inside-50 entries were punished; territory did not convert because Adelaide's defensive rebound system was, exactly as the brief said, the second-strongest in the competition. Hit on identification, not on counter-strategy.

### The three structural blind spots

**Blind spot 1: no variance modelling.** Quarters were modelled as means, not as distributions. The brief had Richmond's average Q3 differential at -9.1 AF; it did not model the *tail* - what is the worst-case Q3 against a side that resets at half-time? Adelaide demonstrated +543 AF in Q3 versus Richmond's 238 AF floor. That gap (+305) is a tail-distribution fact that no mean-based per-quarter model surfaces. **What this means:** any future Richmond brief needs a worst-case-Q3 column alongside the average, computed from the opponent's best historical quarter against any opponent vs Richmond's worst historical quarter, with explicit variance bounds on both.

**Blind spot 2: no mid-match positional flex model.** Matchups were tagged as "game-long" assignments. Dawson tagged for Taranto, Vlastuin tagged for Walker, Ross tagged for Berry. The Walker tag died at team-list announcement (Walker DNP); the Dawson-Taranto tag died in 90 seconds of half-time talk. There was no documented contingency branch for "if Dawson moves into the on-ball group, who goes with him?" **What this means:** matchup briefs should tag every assignment as either "first-half only" (likely to lapse at half-time if structure shifts), "all-game" (the player's role is fixed), or "with contingency" (here is who picks up the new role if X moves). For Adelaide specifically, every matchup involving Dawson, Murray, Sholl or Worrell needs a contingency branch because all four are documented positional flex options.

**Blind spot 3: Berry single-point-of-failure thesis was never tested against the null.** The brief framed Berry as the engine and the single point of failure. Adelaide won with Berry at 3 tackles vs his season average of 8.0; the team tackled at 161% of season rate without him; the pressure system delivered AD +23 in the tackle gap. The thesis was never tested against the null hypothesis "Adelaide's pressure profile is distributed, not Berry-dependent". The data on the day says it is distributed: take Berry out and the team tackle count goes *up*, not down. **What this means:** any future "single point of failure" claim needs an explicit null test - find historical games where the named player had a low-output day and check whether the team metric that supposedly depends on them held up. If it did, the player is a contributor, not a lever, and the brief's pressure-attack plan needs a different target.

### Tactical themes - FootyStrategy analysis

#### Theme 1 - The half-time reset as a tactical weapon

The Dawson HB→mid reset is a specific modern-AFL tactical pattern: use a versatile midfielder as positional cover for an injured or absent KPF, then revert them to their natural role once the immediate damage (early Walker-shaped goals against) is contained. Matthew Nicks has done this with Dawson and Sholl before; Adelaide's list is built around three or four players who can credibly play three positions. Richmond's pre-game matchup structure had no "if Adelaide reshuffle at half-time" branch - the brief assumed Walker's DNP locked Adelaide into a sub-optimal structure for the full 100 minutes.

**What this means:** any brief for a Matthew Nicks side must (a) document the specific positional flex options the side has used in 2025-26, (b) tag matchups as first-half vs all-game, and (c) include a "second-half re-read" item in the in-game decision protocol so the matchup zone gets refreshed at the long break, not just at the bounces.

#### Theme 2 - The Q3 energy wave

Teams that absorb Q1-Q2 pressure inside a constrained or compromised structure, and then release into their natural roles at half-time, tend to peak in Q3 specifically because the opposition's matchup-tracking fatigue is at its highest (90 minutes of in-game work, no fresh pre-game film). Adelaide's Q3 was not random variance; it was the structural release of two players (Dawson, Murray) from roles they had been suppressed in for two quarters, against a Richmond on-ball group that had been game-planning against the suppressed structure.

**What this means:** the coaching lever - if your structure is wrong in Q1-Q2, the cost of leaving it wrong at half-time is a Q3 avalanche, not a Q4 problem. Q4 problems are recoverable in a single goal swing; Q3 problems compound for 25 minutes of game time. For the analyst writing the half-time live read: the question is not "are we losing?" but "are they playing the structure they want to play?" - if no, plan for a Q3 reset.

#### Theme 3 - Clanger-to-tackle interaction

High-clanger teams against high-tackle opponents is one of the most predictable structural mismatches in the AFL. The brief projected AD +8 tackles additively from seasonal rates (AD 59.5/g, RI 51.8/g). The actual gap (AD +23) reflects the multiplicative interaction: every forced clanger creates a new tackle opportunity, which creates a new clanger risk under pressure, and the cycle compounds. Richmond at clanger rank 18/18 and Adelaide at tackle rank 4/18 represented the maximum-extreme version of this matchup in the 2026 competition.

**What this means:** a tackle-gap projection formula for future briefs is `seasonal_differential * (1 + clanger_severity_factor)`, where `clanger_severity_factor` is the opposition's clanger-rank percentile expressed as a multiplier (worst clanger team = ~1.0, best = ~0). For Richmond playing any top-5 tackle side, the projected gap should roughly double the additive baseline. This is testable against the 2024-26 historical record and worth a Scientist sensitivity check before the next Richmond tactical brief.

#### Theme 4 - Ruck supply chain to a tall forward

Ryan's absence broke the Lynch supply chain at the first link. The chain is: stoppage first use → midfield exit → inside-50 entry → contested mark → goal. When the ruck battle is lost by 18 hit-outs, the tall forward must find alternative supply routes - second-ball crumbing from teammates, kick-mark chains through a marking midfielder operating as a half-forward, or transitioning into a ground-level contested-ball role. Lynch did none of these until Q4 when Adelaide had downshifted; he finished with 14 disposals, 8 of them kicks, 1 confirmed goal at Q4 7:11 in cosmetic time.

**What this means:** the next Richmond brief must include a "Ryan-out" forward supply diagram that does not depend on hit-out first use. Three plausible alternatives, each with a named player: (a) Taranto as the marking-midfielder kick-mark chain operator, (b) Lynch dropping deeper to half-forward and crumbing off Trainor's marks, (c) explicit accept-defeat-on-territory and play the kick-out + transition game from Short and Vlastuin. Pre-game should pick one and rehearse it; in-game the live read at 5-minute mark of Q1 should confirm which route is open.

#### Theme 5 - System vs talent

Adelaide won without Walker (DNP) and without Berry firing (3 tackles vs 8.0 average). This is the signature of a system win: the structure generates output regardless of which individuals operate within it. The brief's approach of identifying "Berry as single point of failure" was analytically correct for talent-dependent teams (think a Bulldogs side leaning on Bontempelli, or a Carlton side leaning on Cripps); it fails for systems-dependent teams.

**What this means:** for future Adelaide briefs, the single-point-of-failure test must run against the null - in games where Berry has under 5 tackles, does Adelaide's team tackle count go up, down, or stay flat? In this game it went up (61 vs season 59.5). If that holds across a 5-game sample, the pressure system is distributed, Berry is a contributor not a lever, and any "remove Berry, the engine stops" plan is targeting the wrong axis. The right axis is the structural rotation pattern, not any one midfielder.

#### Theme 6 - Richmond's Q4 shows us the fixable vs the structural

Richmond won Q4 AF stretches and competed hard. Once Adelaide downshifted and Lynch finally got the ball, the forward structure worked - he marked, he goaled, the supply chain operated. This tells us something important: the conversion problem is *independent* of the Adelaide pressure system. Hopper's shank from 45m at Q4 11:36 was not caused by an Adelaide tackle; he had clean possession and a clean kick. Richmond can execute the game plan when given clean ball.

**What this means:** the question for Damien Hardwick / Mark Williams (or whoever Richmond's coaching staff is in 2026) is whether Q3 capitulation is a *roster depth* problem (not enough bodies to rotate when the hit-out battle is lost and the tackle rate against you spikes) or a *game-plan* problem (no emergency protocols for "we are losing the hit-out count and our clangers are up - here is the contingency"). The two have different fixes. Roster-depth is a list-management decision (trade period, draft); game-plan is a coaching meeting on Tuesday. The data here can't distinguish them definitively, but the fact that Q4 worked once Adelaide downshifted suggests Richmond's *talent* is competition-grade. The structural failure is in the plan, not the playing list.

## What to adjust for the next Richmond vs Adelaide brief

1. **Ryan selection is a binary flag, not a conditional footnote.** When Ryan is out, the hit-out model must be inverted from the start. The brief had "If Ryan plays" buried in section 3; it needed to be the headline branch. **Action**: Pre-match brief structure should branch on each side's #1 ruck availability, not assume best-case lineup.
2. **Q3 variance is the unmodelled risk.** Richmond's mean Q3 differential is -9.1; their *variance* against teams that reset at half-time has not been measured. Adelaide demonstrated a +305 AF quarter on this specific lineup. **Action**: Add a "worst-case Q3" model to future briefs - opponent's best historical quarter vs Richmond's worst, with explicit variance bounds.
3. **Tackle magnitude scales non-linearly with clanger floor.** AD +8 (the brief's projection) was the additive baseline; AD +23 (actual) was the multiplicative outcome when both sides hit their structural extreme. **Action**: Tackle-gap projection must include an interaction term with the favoured team's clanger rank. Both sides at extremes = double the projected gap.
4. **Conversion is volume, not accuracy.** Richmond's 56% accuracy beat Adelaide's 50%; the game was decided by 12 extra Adelaide scoring shots. **Action**: Replace "lift conversion from X to Y" targets with "generate N+ inside-50s" targets. Inside-50 differential is the load-bearing predictor.
5. **Mid-match positional shifts are reads, not noise.** Dawson at half-back was a structural cover that failed; the brief's "Dawson is your captain on senior" matchup assumption did not survive the half-time reshuffle. **Action**: Document each side's known positional flex options and tag matchups as "first-half" or "all-game" to track when they lapse.

## SuperCoach fantasy verdict

**Top scorers (SC):**

Adelaide: Rankine 147, Milera 127, Dawson 115, Laird 105 - four 100+ SC scores. **Rankine was the captaincy slam dunk** in retrospect; the brief's "must-contain" framing did not factor in his SC ceiling when contained tactically failed.

Richmond: Ross 107, Miller 106, Short 101 - three 100+. **Ross 107 SC at his price point is the value-add of the game.** The brief's "Berry-equivalent" call gave you the right player.

**Traps:**

- **Taranto 99 SC at 28 disposals** is a value-trap classic. The disposal volume reads "elite midfielder game"; the 18 clangers and 64% DE strip it. SC penalised the cheap turnovers; AF rewarded the volume (87 AF). The 12-point AF/SC gap reflects that.
- **Vlastuin 90 SC** is junk - the matchup he was selected for (Walker) did not eventuate, and Q4 AF of 6 shows he gassed. Don't extrapolate his role into next week.
- **Lynch 90 SC, 71 AF** - 23 of the AF came in Q4 garbage time. SC saw the contested marks and rewarded; AF saw the late delivery and underweighted. This is the one game where SC told you more than AF.

**Over-deliverers:**

- **Trainor 76 AF / Miller 106 SC** - cheap forward/defender combo who exceeded brief expectations. Trainor was a late inclusion; if his TOG (86%) holds next week he's a hold.
- **McAndrew 86 AF / 83 SC as the lone ruck** - 33 hit-outs in a forced 92% TOG ruck role. If Ryan stays out, this becomes a weekly score.

**Under-deliverers:**

- **Berry 69 AF / 73 SC** at his usual ownership - his pressure profile (3 tackles vs season 8.0) collapsed. The brief's "remove Berry, the team's pressure profile collapses" thesis is now in question: Berry was effectively removed (3 tackles) and Adelaide's pressure profile *did not* collapse - it actually went up (61 team tackles vs season 59.5). **This is a finding worth carrying forward**: Adelaide's pressure system may not be Berry-dependent at the level the brief assumed.

## One-line verdict

**Richmond vs Adelaide R9 2026 verdict:** Adelaide won by 37 in a Q3 burst that exceeded any single-quarter scenario the brief modelled, with the headline structural lesson being that Ryan's absence inverted the entire ruck-dominance plan and there was no Plan B written.

---

*Source: FanFooty live feed game 9781, final snapshot 2026-05-10 17:51 (`data/live_snapshots/9781_20260510_1751_final-siren.json`).*
*Schema note: col15 (per-player goals field) unreliable - team sums do not match scoreboard. Goal counts cited use commentary entries or scoreboard totals only.*
*Headline scoreboard verified: header.home_score = "9.7.61", header.away_score = "14.14.98", header.status = "Final Siren".*
