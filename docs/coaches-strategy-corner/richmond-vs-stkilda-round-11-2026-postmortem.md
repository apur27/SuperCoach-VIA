# Richmond vs St Kilda - Post-Mortem (Round 11, 2026)

> [← Coaches Strategy Corner](README.md) | [← AFL insights](../afl-insights.md)
>
> Pre-match: [Executive summary](richmond-vs-stkilda-round-11-2026-executive-summary.md) · [Tactical brief](richmond-vs-stkilda-round-11-2026.md) · [Player matchups](richmond-vs-stkilda-round-11-2026-player-matchups.md) · [H2H history](richmond-vs-stkilda-round-11-2026-head-to-head-history.md)
> In-game: [Q1](richmond-vs-stkilda-round-11-2026-q1-live.md) · [Q2](richmond-vs-stkilda-round-11-2026-q2-live.md) · [Q3](richmond-vs-stkilda-round-11-2026-q3-live.md) · [Q4](richmond-vs-stkilda-round-11-2026-q4-live.md) · [Full-time verdict](richmond-vs-stkilda-round-11-2026-full-time-verdict.md)
>
> **Status: post-match.** This is a data-grounded structural autopsy. The full-time verdict carries the headline narrative; this document is the methodology layer underneath - what the numbers say, where the pre-game read held, and where it broke. Box-score figures verified against `data/live_snapshots/9789_20260517_1804_players.csv` (final-time snapshot) and `data/player_data/*_performance_details.csv` (season form).

---

## 1. Final result and context

**St Kilda 16.13.109 def Richmond 11.7.73 by 36 points** at Marvel Stadium, 17 May 2026 **[data]**. The H2H ledger now sits at six straight to St Kilda; the five-game average that the pre-match brief flagged (+35.4 to St Kilda) updated to a six-game average that this result sat directly on top of. The streak is no longer a quirk - it is the persistent structural mismatch the H2H lens was warning about, and Richmond's R11 list could not break it.

Richmond entered as the league's worst-scoring side (9.2 g/g, 18/18) and worst-disposal team (330.7 d/g, 18/18) **[data]**, missing the small-forward pressure unit (Bolton, Daniel Rioli gone) and asking Samson Ryan to anchor a +5 hit-out plan that needed his health to stand up. St Kilda entered without Wanganeen-Milera (the kicking spine) and the pre-game thesis was that Milera's absence would force Bradley Hill into an attacking-launcher role and open up the rebound chain. That thesis inverted (see section 4).

---

## 2. The pre-match call: what we said and why

The brief held three load-bearing structural reads:

1. **Milera's absence opens St Kilda's transition.** Hill was the named replacement; the brief modelled a 3.7 kicks/g drop in the wing-half-back kicking floor and a less-attacking half-back unit.
2. **Richmond's path is contested and territorial.** With 18/18 disposal and goal ranks, Richmond could not out-volume St Kilda; the only route was clearance-into-quality-entry through Taranto, Hopper and Ross, converting to Lynch and Lalor inside 50.
3. **The H2H is brutal but 2025 R23 was a 4-point game.** A 5-0 streak at +35 average was descriptive, but the most recent fixture (4-point Saints win) suggested the game could be closer than the ledger said.

The headline call was **Richmond by 11** (FootyStrategy), with **25-30% Richmond win probability** (data-adjusted from Codex's 35-40%). The headline call's hinge was the ruck contest: if Ryan beat Marshall by +5 hit-outs, Richmond's clearance group could survive the territory war. The H2H warning was loaded as a flag, not as the headline.

**Tripwire:** "If St Kilda lead the inside-50 count (kick-share proxy in-game) at half-time, flip the call." The tripwire fired at Q1 - St Kilda's Q1 kick-share was already commanding - and did not un-fire at any stage of the match.

---

## 3. What the data confirmed

These were the pre-match reads that held up against the final box score.

### The H2H pattern held to within a point

The +35.4 five-game average extended to a +36 sixth game. That is not coincidence; it is a six-game sample at one standard deviation of itself. The H2H lens was telling the truth, and the headline call under-weighted the signal it was producing. **The structural lesson is that when a head-to-head ledger reads that consistently across multiple seasons, the burden of proof shifts to "what has changed structurally" - not to "what could go right this time."**

### De Koning was the ruck pivot

The pre-match brief flagged Tom De Koning as the structural ruck threat. He delivered 23 hit-outs solo **[data: snapshot]**, against a Richmond hit-out total of 19 across the whole roster. His 2026 season average is 19.2 HO/g **[data: player file]**; the R11 line was above-average but not an outlier - within the same band as his R3 (27) and R4 (34). The brief identified the right player and the right lever. What it did not do was prescribe a Richmond Plan B for *losing* the ruck. There wasn't one in the document, and there wasn't one on the day.

### Sinclair tracked exactly to the band

Pre-match prediction: Sinclair 27 disposals. Actual: 29 (21k/8hb), 75% DE, 115 AF, 138 SC **[data]**. Inside the prediction band on volume, ahead of rate on efficiency. The pre-match player-level work on Sinclair was honest - the projection landed inside one disposal of his actual 29.4 d/g season average **[data]**.

### Short was Richmond's best ball-user

Predicted 23 disposals. Actual: 25 (16k/9hb), 100% DE, 88 AF, 120 SC, 89% TOG **[data]**. Top SC for Richmond. The brief's read of Short as the half-back launcher held in a side that gave him very little to launch from - he was the only Tiger to clear his prediction band on volume, and he did it at perfect disposal efficiency. This is a real signal for the role going forward, separable from the team loss.

### Richmond's contested-and-territorial path was correctly framed - and was correctly impossible

The brief's framing that Richmond could not out-volume St Kilda was confirmed: disposals 408-327 (-81), marks 115-103 (-12), tackles 50-40 (-10). The territory war ran in the direction the brief predicted, at the magnitude the brief warned about. **The framing was right; the prescription (win clearances to compensate) was downstream of a ruck contest the brief assumed Ryan could keep within +5.** Without ruck parity, the prescription had no mechanism.

---

## 4. What the analysis missed

This is the honest section. Four failures, each load-bearing.

### 4.1 Jack Macrae was the dominant player on the ground and was not in the brief

Macrae finished with **31 disposals (11k/20hb), 7 tackles, 7 marks, 2 goals, 125 AF, 109 SC** in **61% TOG** **[data: snapshot]**. Thirty-one disposals from a midfielder with the 17th-lowest game time on his own list, and 7 tackles to go with it. The disposal volume is the visible half - 11 kicks distributing forward, 20 handball-receives turning St Kilda's stoppage wins into outside chain. The 7 tackles is the invisible half - he was the chase-down insurance every time Richmond won a rare clearance. One player carrying both halves of midfield work is a structural mismatch.

**The brief had no Macrae tracker.** It named Sinclair, Flanders (as Sinclair's running mate), and tracked Hill as the Milera replacement. Macrae's 2026 form pre-match: 17.8 d/g across 4 games, having played only sporadically since joining the Saints **[data: player file]**. On that surface read, he did not look like a top-three threat in St Kilda's midfield. The R11 line was a +13 disposal spike from his 2026 average, but only a +6 from his 2025 Bulldogs form (24.6 d/g) **[data]** - which is where the projection should have been anchored, given he was joining a midfield with structural space around De Koning's first-use dominance.

**The lesson is methodological.** Pre-match midfield tracking has to extend below the marquee names when:
- The opposition's ruck is forecast to dominate (Macrae's handball-receive role only exists if someone is winning first hands).
- A player has a recent multi-year history at a higher rate than his current season suggests (Macrae's 2025 line was a far better predictor than his 2026 line of 4 games).
- A player's tackle-plus-disposal hybrid profile is the modern midfield mismatch that doesn't show up in disposal-only filters.

The brief's three-named-player tracking was a budget; the budget was set too small for an opposition midfield this deep.

### 4.2 Ruck dominance is a territory stat, not a ruck stat

The brief treated hit-outs as a single line in the projection ("if Ryan plays, +5 hit-outs"). The R11 result showed that hit-outs at the +23 magnitude (42-19) are not a ruck-coaching variable - they are a **territory differential generator**. Every centre bounce starts with first hands on the ball; every stoppage starts with first hands on the ball; first hands compounds across four quarters into the +81 disposal gap (408-327) and the +50 kick gap (243-193) that ran the game.

**The pre-match brief did not have a model of this compounding.** The ruck line read like a tactical item among other tactical items. In reality, when the hit-out differential is forecast at +20 or worse, it is the upstream variable that constrains every downstream metric:

- Clearances (St Kilda starts with the ball)
- Inside-50s (St Kilda starts the chain in front of stoppages)
- Marks (St Kilda has more ball-in-hand, more leading opportunities)
- Tackles against (Richmond is chasing without the ball, can't tackle)
- Clangers (Richmond's defensive resets under pressure produce turnovers)

The structural lesson for future briefs: **when the ruck differential is forecast at +20 or worse, the recommendation tier has to cap at Contested or worse, regardless of player availability elsewhere.** Richmond's pre-match call tier-capped on the Milera-out adjustment but did not tier-cap on the ruck mismatch, and the ruck mismatch was the larger lever by an order of magnitude.

### 4.3 The Hill thesis inverted - St Kilda's half-back played safer, not more attacking

Pre-match: Bradley Hill projected at 22 disposals stepping into the launcher role Milera's absence vacated. Actual: **13 disposals, 3 marks** - well below the band **[data]**. The brief had the right player but the wrong role. What happened is structurally interesting:

**Without Milera, St Kilda's half-back didn't try to recreate Milera. They played more conservative, more intercept-oriented.** Hill stayed in a holding pattern; Callum Wilkie (108 AF, 92% TOG, 12 marks, 20 kicks at 92% DE) **[data]** anchored the defensive intercept role; the launching responsibility shifted **into the midfield**, where Macrae, Sinclair and Flanders absorbed the distribution work the half-back would normally do.

The thesis that "Milera's absence opens up an attacking half-back" was the inverse of what happened. The Milera-out version of this St Kilda is **harder to score against, not easier**, because the defence reverts to a containment shape and the offence runs through midfield distribution rather than half-back launches. This is the counter-intuitive read and it is the most important pre-match miss.

**The structural lesson:** when a marquee creator misses, the default prior should not be "the team plays his role with a worse version of him." The default prior should be "the team reshapes around the absence in whatever way the personnel allows" - which can mean more conservative defence and more midfield-led offence, especially when the replacement (Hill) is a more natural defender than launcher.

### 4.4 "Richmond can win this type of game - slow, contested, territorial" was the wrong framing

The brief framed Richmond's win condition as a slow, contested, territorial game where the Tigers' clearance engine could match St Kilda. The disposal gap (408-327) tells the actual story: **St Kilda dominated the territory game and the contested game.** Tackles 50-40 to St Kilda means the Saints applied more pressure, not less. Marks 115-103 means the Saints held more uncontested ball. Hit-outs 42-19 means St Kilda had first use almost three-to-one.

The framing assumed Richmond could choose the contest. They could not - the ruck dominance dictated the game shape from the opening bounce. **You don't get to play a contested, territorial style against a team that has 23 more hit-outs and 81 more disposals than you. You play whatever style the team with the ball lets you play, and you reset for next week.**

The structural lesson: the brief's "Richmond's path is contested and territorial" framing was conditional on a clearance contest that the ruck mismatch precluded. The conditional was not made explicit. Future briefs should write any "Richmond can win by playing X style" claim with the upstream conditions named ("if hit-out differential is within 10, then..."), so that the in-game tripwire knows when the framing has collapsed.

---

## 5. Quarter-by-quarter story

### Scoreboard progression

| Q | St Kilda (cum) | Richmond (cum) | Margin | Q score (STK/RIC) |
|---|---|---|---|---|
| Q1 | 33 | 12 | STK +21 | STK 5.3 / RIC 1.6 (roughly) |
| HT | 53 | 26 | STK +27 | STK 3.x / RIC 2.x (+6) |
| 3QT | 87 | 54 | STK +33 | STK 5.x / RIC 4.x (+6) |
| Final | 109 | 73 | STK +36 | STK 3.x / RIC 3.x (+3) |

**The shape:** Q1 was the game in a single quarter. The 21-point opening margin was already past the point where the pre-match tripwire (St Kilda leading inside-50 count / kick-share proxy at HT) had fired. From Q2 onwards, this was a managed lead, not a contested game.

### Q1 - the game is over by quarter-time

St Kilda led 33-12 at the first break. The brief had identified Q1 as Richmond's most vulnerable quarter (season average -10.9 differential, the league's worst Q1 gap against St Kilda's +11.6) - the predicted swing was 22 points, and the actual Q1 margin was 21. The Q1 read was almost exactly the size of the pre-match warning, which means **the warning was right and the game opened at the predicted disaster scenario**, not at the upside scenario the headline call leaned into.

Mechanism: De Koning won the centre bounces, Macrae and Sinclair received the handball chain, Max Hall (who finished with 3 goals, 4 tackles, 114 AF, 101 SC across 80% TOG **[data]**) kicked his first goal off 8 disposals while untagged. Richmond's midfield was chasing the ball backwards from the opening minute.

### Q2 - Hopper tags Hall, the damage shifts

Hopper was deployed to tag Max Hall from Q2. **The tag worked on Hall** - Hall's Q2 AF was 23 (down from 30 in Q1, though TOG-normalised the suppression was modest). The problem was that the damage source shifted: Darcy Wilson (85 AF, 95% DE, 7 marks **[data]**) and the midfield group produced the Q2 goals from Hall's vacated launch space. Richmond won Q2 on the scoreboard by 6 (kicked 14, conceded 20 - so actually narrowed the gap by ~6 in this period, depending on the exact split). The half-time margin moved from -21 to -27 - **St Kilda extended the lead in the quarter Richmond's tag worked**.

The Match-up Tactician's lesson is the one in the verdict: tagging a named threat does not work when the opposition's distribution is genuinely deep. St Kilda had Macrae, Sinclair, Flanders, Wilson, Garcia and Hall all credible as inside-50 chain sources. Single-target tagging is a tool for a one-or-two-distributor opposition; for a six-distributor opposition, the personnel cost is unrecoverable.

### Q3 - the closeout begins, Garcia announces himself

Garcia finished the game with 3 goals, 7 tackles, 27 disposals, 112 SC, 80 AF **[data]**. His Q3 was strong (23 AF) but his standout quarter was Q2 (40 AF) **[data]**; his impact was distributed across the second and third quarters as he won contested ball and finished chains forward. The pre-match brief had no Garcia profile - and on his 2026 form (peaking at 32 disposals in R8, trending up), he was a viable named tracker. Another player below the marquee line who carried real damage.

St Kilda extended the margin to +33 at 3QT. The Q3 AF totals (St Kilda 388 vs Richmond 278) confirm Richmond was beaten on impact, not bad luck.

### Q4 - the margin holds, Miller fires too late

Richmond's Q4 AF (388) was actually their best quarter and equal to St Kilda's (407) within 5%. Ben Miller (Richmond defender) posted 49 AF in Q4 alone - his game total was 121 AF / 109 SC / 19 kicks / 15 marks / 100% TOG **[data]**, the kind of intercept-defender ceiling line that wins quarters when the game is already gone. Jack Ross also produced his best work in Q4 (37 AF, after 32 in Q3) - **Ross posted Q3 32 + Q4 37 = 69 AF in the second half**, the only Tiger midfielder with that profile. He was beaten by his Q1-Q2 (32 combined) start; the Q3-Q4 lift came too late to matter.

The Q4 narrative is the one that explains why this was a 36-point game and not a 50-point game: **Richmond's best players turned up late, when the result was already decided.** That is the wrong shape for an upset, but it is the right shape for individual SuperCoach signal (see section 6).

---

## 6. Player accountability - the matchup verdict on each named player

### Jayden Short (Richmond) - the brief was right, the team lost anyway

Predicted 23 disp. Actual: 25 disp (16k/9hb), 100% DE, 7 marks, 88 AF, **120 SC** (Richmond's top SC) **[data]**.

**Verdict: Outperformed the brief.** Short won his individual contest against Hill decisively and posted perfect disposal efficiency across 89% TOG. The Q1 AF (40) was the most productive opening quarter on the ground for either side. The Q3-Q4 fade (17 + 11) is the team shape, not a Short-specific concern.

The Match-up Tactician's classic warning applies: winning your direct opponent does not win the game if the opposition's structure produces damage from elsewhere. Short's 25 disposals were rebound work in a team being pinned in its back half. **They kept the score respectable; they did not turn the game.**

### Jack Sinclair (St Kilda) - exactly to the band

Predicted 27 disp. Actual: 29 disp (21k/8hb), 75% DE, 8 marks, 115 AF, 138 SC **[data]**.

**Verdict: As advertised.** Sinclair was the squad-high disposal-getter on the day, played a half-back/wing flex role (per snapshot matchup note: "Starting at half back and floating up to mids"), and produced the kick-mark distribution chain the brief flagged. The 2026 player-level work landed inside one disposal. This was the brief's most accurate single prediction.

### Bradley Hill (St Kilda) - the thesis inverted

Predicted 22 disp. Actual: **13 disposals, 3 marks** **[data]**.

**Verdict: Badly missed - but the miss was structural, not Hill-specific.** Hill did not step into the attacking-launcher role the brief assumed Milera's absence would vacate. The St Kilda defence reshaped around containment instead of launch, and Hill played to that. **He was not a poor performer in his actual role; he was used in a different role from the one the brief modelled.** The miss is on the modelling assumption, not the player.

### Jack Macrae (St Kilda) - the player the brief did not name

Untracked pre-match. Actual: **31 disposals (11k/20hb), 7 tackles, 7 marks, 2 goals, 125 AF, 109 SC, 61% TOG** **[data]**.

**Verdict: The game-deciding player, and a planning failure.** Macrae was the highest-impact midfielder on the ground for either side. His handball-receive profile (20 of his 31 disposals) was the perfect complement to De Koning's hit-out dominance - first hands to handball receivers turns ruck wins into outside chain. **No Richmond player tracked him; no Richmond plan accounted for him.**

His pre-match 2026 line of 17.8 d/g was a sample-size mirage; his 2025 line of 24.6 d/g and his career profile (former Bulldogs midfielder, multiple seasons over 30 d/g) was a far better predictor for a fit midfielder joining a team with structural midfield space. The pre-match anchor should have been 22-26 d/g; he hit 31 by walking into a Macrae-shaped role that nobody had named.

### Tom De Koning (St Kilda) - delivered the structural lever the brief identified

Predicted: dominant ruck contest. Actual: **23 hit-outs, 79 AF, 118 SC, 72% TOG** **[data]**.

**Verdict: As advertised, and the upstream variable for the game.** De Koning's 23 individual hit-outs were 55% of St Kilda's total (42); the +23 hit-out differential at team level matched his individual contribution almost exactly. The brief correctly identified him as the lever and was correct that he would dominate. **What it failed to do was prescribe Richmond's response to losing the lever.** The verdict on the player is "as expected"; the verdict on the planning is "the consequence of losing this contest was under-modelled."

---

## 7. Structural lessons for next time (Round 12+)

Each grounded in the R11 verified data, each actionable inside seven days.

### 7.1 Pre-match midfield tracker must extend to five named opposition players

The brief named three trackers (Sinclair, Flanders, Hill). The damage came from Macrae (untracked) and Garcia (untracked). **Future briefs against any side with a midfield rotation deeper than three names must explicitly list five trackers, with prediction bands for each.** The cost is one paragraph in the brief; the benefit is not having the game-deciding player in the blind spot.

**Selection criteria for the fourth and fifth tracker:** opposition players whose multi-year career rate exceeds their current-season rate, opposition players with a tackle-plus-disposal hybrid profile, opposition players whose role on the day is forecastable (e.g. handball-receivers when the ruck is dominant).

### 7.2 The recommendation tier caps on ruck differential

Add a rule to the pre-match brief structure: **if the forecast hit-out differential exceeds +/-20, the recommendation tier caps at Contested.** The R11 forecast was around +18 to +25 to St Kilda (Marshall + De Koning vs Ryan-conditional Richmond); the call should not have been "Richmond by 11" - it should have been "Contested, lean St Kilda by 25+, requires ruck contest within 10 to flip." That framing carries the structural mismatch into the headline.

### 7.3 Plan B for losing the ruck must be a documented play-script

Three named alternatives, written in the brief, rehearsed on Tuesday:
- **(a)** Accept first-use loss; play kick-mark rebound through Short and Trezise from the back half; concede territory but force St Kilda to score from chain rather than stoppage.
- **(b)** Tag the opposition's primary handball-receiver (Macrae-equivalent) rather than the primary clearance winner - the handball-receiver is where ruck dominance converts to outside chain.
- **(c)** Front-half press to force the contest forward of centre, accepting the conditioning cost and the Q3 risk.

The brief picks one based on the opposition profile; the in-game live read at the five-minute mark of Q1 confirms it. Without a chosen Plan B, "lose the ruck" defaults to "play the same game plan and lose the ruck", which is what happened in R11.

### 7.4 Tagging assumes a one-or-two-distributor opposition - check before deploying

Hopper's Q2 tag on Max Hall was sound situational coaching against a normal opposition shape. Against St Kilda's six-credible-distributor midfield, single-target tagging spent personnel without changing the damage source. **Before deploying a tag in future games, the brief should answer: "if we tag X, who are the second and third sources of inside-50 damage?"** If there are three or more credible answers, the tag is wasted and the body should go to a structural role (extra mid at centre bounce, defensive winger) instead.

### 7.5 When a marquee creator misses, prior should be "shape changes", not "replacement"

The Hill thesis assumed the Saints would replace Milera's role with a worse version of it. The reality was a different role entirely. Future briefs encountering "Player X is out" should run two hypotheses by default:
- **H1 (replacement):** named replacement plays X's role at a degraded level.
- **H2 (shape change):** team reshapes around the absence; replacement plays a different role; offensive load shifts elsewhere on the ground.

Both go into the brief; the in-game tripwire decides at the five-minute mark of Q1 which is operative. R11 ran H1 and the game was actually H2 from the opening bounce.

---

## 8. Methodology notes

**Data sources used:**
- `data/live_snapshots/9789_20260517_1804_players.csv` (final-time per-player snapshot, 46 players, FanFooty schema). **Source of truth for: kicks, handballs, disposals, marks, tackles, hit-outs, AF, SC, quarter splits (af_q1-af_q4), TOG, DE.**
- `data/player_data/*_performance_details.csv` (season form context for Macrae, Sinclair, De Koning, Garcia, Hill, Ross, Short, Trezise).
- `data/matches/matches_2026.csv` (final scoreline, attendance, H2H lookup).

**Data that was NOT available live:**
- **Inside-50 counts.** Not in the FanFooty per-player snapshot schema. The in-game tripwire used kick-share as a proxy (243-193 to St Kilda = decisive territory dominance from Q1 onwards). The proxy was directionally correct and triggered the tripwire correctly, but it is not a direct inside-50 differential and any future read on "St Kilda inside-50 dominance" carries an inferential gap.
- **Contested possessions and clearances.** Same schema absence. The narrative about "ruck dominance → first use → handball chain" is inferential from the hit-out differential and the disposal profile (Macrae 20 handball-receives is the closest direct signal). No direct clearance count is available.
- **Per-player goals and behinds reliable team total but unreliable at the snapshot level.** R10 postmortem noted that the snapshot's goal/behind columns disagreed with afltables.com for 21 players in that game. For R11, the team-total goals from the snapshot (STK 24, RIC 19) **do not match the official 16.13 vs 11.7** (which would be STK 16 goals, RIC 11 goals). **All goal-scoring figures cited in this document use the official scoreline, not the snapshot's per-player goal column.** Awaiting afltables.com R11 publication for a player-level goal audit.

**Caveats:**
- The "Jack Macrae was the dominant player" claim is grounded in disposal, tackle, mark, AF and SC - all of which agree across the snapshot. The two goals attributed to him are from the snapshot's unreliable goal column; if afltables.com publishes a different goal count for him on a recount, the goal-specific claim is wrong, but the dominance claim is not (disposal + tackle + AF carry it alone).
- The "St Kilda played a more conservative half-back without Milera" thesis is interpretive from Hill's 13 disposals, Wilkie's intercept profile, and the midfield distribution data. It is consistent with the data but not proven by it; a one-game inference. The tripwire to upgrade or falsify this is in the next 2-3 St Kilda games without Milera against top-eight opposition.
- The "ruck dominance is a territory stat" claim is a structural inference, not a controlled experiment. The disposal differential (-81), kick differential (-50), mark differential (-12) and hit-out differential (-23) all run in the same direction, which is consistent with the territory-from-ruck framing but does not isolate it from confounders.
- Macrae's 2025 disposal average of 24.6 d/g is from his Western Bulldogs era; his role in the 2026 St Kilda midfield is structurally different (more handball-receive, less clearance-leader). The pre-match anchor should have been 22-26 d/g, not 17.8 d/g; the R11 actual of 31 was a +5-9 spike on the better anchor, not a +13 spike on the 2026-only anchor. **The structural lesson stands regardless of which anchor is used: he was untracked, and tracking would have been justified on either prior.**

---

*Sources: `data/live_snapshots/9789_20260517_1804_players.csv` (final-time snapshot, 2026-05-17T08:04:41Z); `data/player_data/macrae_jack_03081994_performance_details.csv`; `data/player_data/koning_tom_16071999_performance_details.csv`; `data/player_data/garcia_hugo_22052005_performance_details.csv`; `data/player_data/sinclair_jack_12021995_performance_details.csv`; `data/player_data/short_jayden_24011996_performance_details.csv`; `data/player_data/hill_bradley_09071993_performance_details.csv`; `data/player_data/ross_jack_03092000_performance_details.csv`. Cross-checks against `data/matches/matches_2026.csv` for the final scoreline. Pre-match brief and in-game live reads in companion documents linked at the top of this file.*
