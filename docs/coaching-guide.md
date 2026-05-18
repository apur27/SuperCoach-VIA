# For the coaching staff - building a data-driven game plan

> [← Back to AFL insights](afl-insights.md) | [← Back to main README](../README.md)

## For the coaching staff - building a data-driven game plan

You're an assistant coach, performance analyst, or fitness staff member at Richmond. It's Monday morning, the senior coach has just confirmed Friday night's opponent - West Coast at the MCG - and your week is now about one question: **how do we beat them?**

You already know what your eyes tell you from the last three weeks of vision. You've got the GPS reports, the contested-ball numbers from the last game, the medical list. What you don't always have on tap is **130 years of structured match and player data**, sliced however you want, with a footy-literate analyst who'll work through the night without asking for a coffee.

That's what this repo plus a Claude Code session gives you. Every Richmond–West Coast result ever played, every disposal Elliot Yeo has ever logged, every game Harley Reid has been kept under 18 touches in, every clearance Yeo has won when his side has played away from Perth - all queryable in plain English. The Scientist agent on top of that does the heavier lifting: opponent-split regressions, form-curve analysis, matchup history, sensitivity tests on which factors actually move the needle.

This section is for the staff member who knows the game inside out but wants the numbers to back the call before walking into the Tuesday review meeting.

---

### What's in the data (and what isn't)

**You have:**

- Every Richmond vs West Coast match ever played - venue, margin, quarter-by-quarter scores, weather where recorded
- Every West Coast player's per-game record: kicks, handballs, marks, disposals, goals, behinds, tackles, clearances, contested possessions, hit-outs, frees for/against, Brownlow votes - by round, by year, by opponent
- The same for every Richmond player - recent form (last 5 rounds), season averages, career history, splits by opponent and venue
- Historical opponent splits - how a player or team performs at home vs away, against specific opposition, in specific rounds, across decades
- Disposal predictions for the upcoming round (`prediction.py`) - useful as a sanity check on form or for comparing Richmond vs West Coast personnel projections

**You don't have (be honest with yourself and the head coach):**

- **No GPS, no heart-rate, no high-speed running data** - the dataset is box-score only
- **No positional/spatial data** - you can't ask "where on the ground does Harley Reid win his ball" or "what's West Coast's average kick length inside 50"
- **No video tags** - pressure acts, spoils, intercept marks, defensive one-percenters before 2011 aren't here
- **No injury/availability data** - you'll need to overlay your own medical and team list on top
- **No live odds, weather forecasts, or in-game state** - historical match-day weather is patchy

If a question requires any of the above, the answer here is "this dataset can't tell you that - go to your video department or your GPS provider." That honesty matters when you're in front of the senior coach.

---

### The workflow - Monday to Thursday with Claude

Below is a six-step workflow you can run end-to-end in a single Claude Code session. Each step has copy-pasteable prompts. Plain Claude handles most of it; the Scientist gets pulled in only for the questions that genuinely require analytical depth.

To start a session:

```bash
cd /path/to/SuperCoach-VIA
claude
```

Then work through the steps below.

---

#### Step 1 - Understand West Coast's current form

Before you talk personnel, get a read on the team. Are they trending up or down? Is their midfield winning the ball at expected rates? Are their forwards converting?

```
Pull the last 5 rounds of West Coast's results - opponents, margins, venues. For each game, tell me their total disposals, contested possessions, clearances, goals, and inside-50 count if it's in the data.
```

```
For each West Coast player who's played at least 3 of the last 5 rounds, give me their average disposals, goals, and tackles over that window, plus how that compares to their 2026 season average. Flag anyone trending up or down by more than 15%.
```

```
Compare West Coast's last 5 games at home (Optus Stadium) versus their last 5 games on the road. Are there meaningful differences in their disposal count, scoring, or how heavily they lean on specific players?
```

What you're looking for: which West Coast players are in form, which are wobbling, and whether their recent results are noise or signal.

---

#### Step 2 - Identify West Coast's key weapons and how to stop them

Now go individual. Who hurts you most if left free?

```
Rank West Coast's current top 8 ball-winners by 2026 average disposals. For each one, show me their disposal split when they've been tagged versus untagged this year - proxy "tagged" by games where they had under 18 disposals against a top-8 contested-possession side.
```

```
Elliot Yeo - give me a complete profile. Average disposals, contested possessions, clearances and goals this season. His best and worst opponents historically. Any pattern in how often he plays inside vs as a wingman based on his disposal-to-tackle ratio.
```

```
Harley Reid is the one we're most worried about. Pull every game he's played in 2026 and tell me - when his disposals were under 18, who was he matched up against and what was West Coast's margin in those games?
```

```
Jamie Cripps and the small forward group - what's their goal-to-shot ratio this year, and how does it change when West Coast is leading vs trailing at three-quarter time?
```

```
@"Scientist (agent)" build me a quick model on Harley Reid's disposal count this year. Use opponent strength, venue, and rest days as features and tell me which of those actually predicts his output. I want to know whether physically tagging him is worth it or whether his numbers are mostly driven by something else.
```

What you're looking for: a shortlist of 2–4 West Coast players you genuinely need a plan for, and an honest read on whether tagging or zoning each one is supported by their historical splits.

---

#### Step 3 - Identify West Coast's weaknesses to exploit

Every opponent has soft spots. Find them in the numbers.

```
Show me West Coast's defensive record in the last quarter when they've been leading vs trailing this year. Are they bleeding scores when behind, or do they tighten up?
```

```
Across West Coast's 2026 games, where have they conceded the most goals - from forward-50 stoppages, from turnovers in their defensive 50, or from open play? Use shots conceded relative to opposition inside-50 count as the proxy.
```

```
Which West Coast defenders have been getting beaten most often? For each of their backline, show me how many goals their direct opponents kicked over the last 5 rounds.
```

```
West Coast travel - historically over the last 10 years, what's their win rate and average margin when playing in Melbourne in the second half of the season? Compare to their home-Perth record over the same window.
```

```
@"Scientist (agent)" run a proper analysis on West Coast's quarter-by-quarter scoring patterns over the last 3 seasons. Specifically: do they fade in last quarters of away games? I want a real test, not just averages - confidence intervals or a sensitivity check on whether the pattern is noise.
```

What you're looking for: 2–3 structural weaknesses you can build a game plan around - e.g. "they fade late in away games" or "their defensive small Y is conceding 2.3 goals a game over the last month."

---

#### Step 4 - Analyse Richmond's own form and matchup strengths

Now turn the lens on us. Who's hot, who's matched up well, and who's been quiet against this opposition historically?

```
For Richmond's current top 22, give me each player's last 5 rounds - disposals, goals, contested possessions - versus their season average. Who's in form, who's flat?
```

```
Tim Taranto's record against West Coast - every game he's played them, his disposals, goals, and votes. Is he historically a strong performer against this opposition or below par?
```

```
Noah Cumberland - show me his games this year split by where he's played (forward vs midfield, proxy with his CBA proxy of disposal-to-goal ratio). Where has he been most damaging?
```

```
Tom Lynch and Noah Cumberland - head-to-head this year, who's converting better, and which one historically performs better against tall, slow defenders versus quicker undersized ones? (Use opposition key-back age and disposal profile as the proxy.)
```

```
Across all Richmond players in our current squad, which ones have the best historical record against West Coast - by Brownlow votes per game, by goals per game, by disposals per game?
```

What you're looking for: who you want on the ball early, who's a matchup nightmare for West Coast specifically, and who needs a role tweak to be useful Friday.

---

#### Step 5 - Use Scientist to run deeper statistical analysis

This is where you get value out of the heavier model. The plain-Claude steps above answered the "what" questions. The Scientist answers the "is it real" and "how confident should we be" questions - the ones that decide whether you actually change a structure on the back of a number.

> **Cost reminder:** the Scientist runs on Claude Opus and burns tokens hard. Read the [STOP. READ THIS FIRST.](scientist-agent.md#stop-read-this-first-do-not-waste-the-scientist) section. Don't @ Scientist for "what's Harley Reid's average" - use it for the questions where the answer changes a coaching call.

```
@"Scientist (agent)" we're considering tagging Elliot Yeo with Marlion Pickett. Look at every game in the last 3 seasons where Yeo was matched against a similar-profile inside midfielder (use age, contested possessions per game, and tackles per game to define similar). Tell me - does tagging actually suppress his output, or does West Coast just shift his role and someone else gains the ball? Give me the answer with proper uncertainty, not a point estimate.
```

```
@"Scientist (agent)" build a matchup model - for our likely 22 versus their likely 22, predict the contested possession differential and the inside-50 differential. Show me which individual matchups contribute most to the projected margin, and which are the most uncertain. I want to know which 1–2 matchups the game probably hinges on.
```

```
@"Scientist (agent)" historical: in Richmond–West Coast games at the MCG over the last 15 years, what's the single biggest predictor of Richmond winning by 20+? Test multiple hypotheses (clearance differential, inside-50 differential, goal-kicking accuracy, contested mark count) and tell me which one actually holds up versus which are just confounded with general team form.
```

```
@"Scientist (agent)" the prediction model in this repo predicts disposal counts. Run it for both squads for this round and tell me - where are Richmond projected to outpoint West Coast, and where are we projected to be outpointed? Treat the predictions with appropriate uncertainty (the 2026 backtest MAE is ~4.1).
```

```
@"Scientist (agent)" Elliot Yeo's clearance work - sensitivity check. Across his career, what conditions predict a high-clearance day for him? I want this with a baseline comparison so I know whether the predictors are real signal or just noise around his season average.
```

What you're looking for: 2–3 statistically defensible insights you can put on a slide and stand behind in a coaches' meeting without getting torn apart on "yeah but is that just because he was playing easier opposition."

---

#### Step 6 - Build the game plan from the findings

Now consolidate. The data has done its job; the coaching judgement is on you.

```
Summarise everything we've found in this session into a one-page brief. Three sections:
1. Threats - top 3 West Coast players we need a plan for, with the data backing each call
2. Opportunities - top 3 West Coast weaknesses we should attack, with how confident we are in each
3. Our matchups - top 3 Richmond personnel decisions, with the matchup data behind each
Be honest about which conclusions are robust and which are exploratory. Flag anything where the dataset can't answer the question.
```

```
Based on the analysis, give me a list of 5 specific coaching questions I should put to the senior coach in tomorrow's meeting - questions where the data raises a structural decision (tagging vs zoning, role change, matchup pick) that needs a human call, not a numerical one.
```

That brief plus your own video and GPS work is what walks into the Tuesday meeting.

---

### Questions worth asking - the coaching shortlist

Twenty prompts you can paste straight into Claude. Some are plain-Claude jobs; the ones marked **(Scientist)** justify the model upgrade.

**On stopping their ball-winners**

- "How does Harley Reid's disposal count change when he plays against physical, high-tackle midfield opponents versus more outside types? Use opponent average tackles per game as the split."
- "Elliot Yeo's clearance count by venue - is he genuinely better at Optus Stadium than on the road, or is that just opponent-quality noise?" **(Scientist)**
- "What's the optimal tagging target for us - of West Coast's top 5 ball-winners, who has the biggest impact on West Coast's winning margin when they have a big game?" **(Scientist)**

**On their structural patterns**

- "What is West Coast's clearance rate when playing away from Perth? Compare to their home rate over the last 3 seasons."
- "Does West Coast give up more inside 50s in the last quarter when trailing? Quantify it."
- "How does West Coast's contested possession win-rate change in wet weather games? (Use historical match-day weather where available, flag the data gap if not.)"
- "West Coast's record at the MCG in the last 10 years - wins, losses, average margin. Are they worse here than at neutral grounds?"

**On their weaknesses**

- "Which West Coast defender has conceded the most goals to his direct opponent over the last 5 rounds?"
- "When West Coast loses the contested possession count, what's their win rate?"
- "Is there a quarter where West Coast consistently scores below their per-quarter average? Test it properly." **(Scientist)**

**On our players**

- "Which Richmond players have historically performed best against West Coast - by votes per game, by goals, by disposals?"
- "Tim Taranto's last 8 games against West Coast - disposals, goals, Brownlow votes. Is he a known West Coast performer or has his record been overstated?"
- "Rhyan Mansell at the MCG vs at the smaller grounds - is there a meaningful split in his goals or disposal count?"
- "Predict Richmond's top 5 disposal-getters for this round using prediction.py and tell me how confident the model is in each prediction."

**On matchups**

- "If we put Noah Cumberland on Tom Barrass and Tom Lynch on Jeremy McGovern, what does the historical data say about how those forward types fare against those defender types?"
- "Identify 1 sneaky matchup advantage - a Richmond player whose 2026 form profile (high contested marks, high tackles, etc.) suggests an underrated mismatch against a specific West Coast player." **(Scientist)**

**On meta-questions**

- "If we win the clearance count by 5+, what does history say about Richmond's win rate against West Coast? Watch out for confounding with general form."
- "What's the smallest margin West Coast has lost by when they had Elliot Yeo with 30+ disposals? Are they losable when he plays well?"
- "Backtest the prediction model on the last 3 Richmond–West Coast games - how accurate was it for the West Coast squad? That tells us how much to trust this round's projections."
- "Run a sensitivity check on this whole brief - which of our findings would change if we excluded the COVID-era 2020–2021 seasons from the historical comparisons?" **(Scientist)**

---

### Honest limits - what to tell the head coach

When you walk into the meeting with this brief, lead with what the dataset can and can't do. It saves arguments.

| Question | Can this dataset answer it? |
|---|---|
| Who hurts us most if left free, by historical impact on margin? | **Yes** - disposal, goal, vote and margin data are all in there |
| How does West Coast travel? | **Yes** - venue splits go back decades |
| Which Richmond player matches up best on Harley Reid? | **Partially** - the data tells you about output suppression; it can't tell you whether your player physically holds up over four quarters |
| Where on the ground does Reid win his ball? | **No** - no spatial data |
| Is Elliot Yeo running at 95% of his peak high-speed metres? | **No** - no GPS data, ask the fitness staff |
| What's our pressure rating in the forward 50 last week? | **No** - no video-tag data |
| What's the predicted margin? | **Indirectly** - we have disposal predictions, not score predictions; build it bottom-up if needed and own the uncertainty |
| Will this play in wet weather? | **Sometimes** - historical match-day weather is patchy and inconsistent before 2010 |

**The rule:** if a question requires GPS, video tags, or positional data, the answer is "this is for the analyst with Champion Data or for the video department." This dataset is for *historical patterns and box-score-driven matchup work*. That's a lot - but it's not everything.

---

### When to use Scientist vs plain Claude - for coaching questions

Same rule as the rest of the repo: plain Claude handles 80% of this work cheaper and faster. The Scientist is for the questions where you need an *analytically defensible* answer because a coaching decision rides on it.

| Plain Claude | Scientist |
|---|---|
| Pull Harley Reid's last 5 games | Test whether tagging Harley Reid actually suppresses his output or just shifts West Coast's structure |
| Show me Richmond's record at Optus Stadium | Run a sensitivity check on whether the home-ground effect is real or driven by opponent quality |
| List West Coast's top 5 ball-winners this season | Build a matchup model and tell me which 1–2 individual matchups the game probably hinges on |
| Average disposals per quarter for Elliot Yeo | Test whether Yeo's per-quarter pattern is statistically meaningful or within normal player variance |
| What was the score in the last Richmond–West Coast game | Across all Richmond–West Coast games, what's the strongest predictor of a Richmond win that survives controlling for general team form? |

If the answer to "would I make a coaching call on the back of this number?" is yes, use the Scientist. If you're just orienting yourself or pulling raw stats, plain Claude is the right tool.

> **Read the cost disclaimer** in the [STOP. READ THIS FIRST.](scientist-agent.md#stop-read-this-first-do-not-waste-the-scientist) section before invoking Scientist. On an entry-level Claude plan, three or four Scientist calls is meaningful spend. Be deliberate.

---

## Leveraging the FootyStrategy agent

Scientist is the data depth. **FootyStrategy is the football brain.** It is a tactical brainstorming agent that thinks like an AFL head coach or senior analyst - it knows zone defences, tagging conventions, ruck-rotation patterns, the difference between a stoppage forward and a structure forward, and the standard contingency plays a Nicks/Goodwin/Hardwick side will reach for at half-time. Scientist tells you *what* the numbers say. FootyStrategy tells you *what coaches will actually do* about it.

**When to use it.** After a post-match review when a structural pattern needs a tactical interpretation; when planning matchup zones for next week and the data has identified an opponent strength but not the counter-play; when a number raises a tactical question the model alone can't answer ("Adelaide tackled at 161% of season rate - is that structural or noise?"); and - increasingly important - **on match day itself**, when a quarter-time or half-time read needs a tactical second opinion in the box.

**How to invoke.** `@"FootyStrategy (agent)"` in a Claude Code session, same as Scientist.

---

### The eight-lens council - how FootyStrategy thinks

FootyStrategy is not a single coach giving you a single answer. It is a **council of eight archetypal coaching lenses** that deliberate in parallel and produce a single integrated recommendation, with disagreements made visible rather than smoothed over. Most questions activate three to five lenses (not all eight) - forcing all eight to speak produces noise. Each lens looks at your question through a different load-bearing principle:

- **The Conditioner** - asks whether you have earned the right to play this way for four quarters. Looks at work rate, repeat-effort capacity, and whether the plan survives round 18.
- **The Tempo Architect** - asks where you want to accelerate the ball and where you want to slow it down. Looks at handball receivers, play-on speed, and tempo control.
- **The Structuralist** - asks what shape you are in when you lose the ball. Looks at zones, forward-50 setups, defensive-50 exits, and the half-back rebound chain.
- **The Match-up Tactician** - asks who covers their best mover and who you can leave. Looks at named threats, tagging targets, and individual contests.
- **The Talent Developer** - asks whether each player is being asked to do what they are actually good at. Looks at role-fit, role-vs-résumé, and the third-year leap.
- **The Innovator** - asks where the league is not looking and what convention can be attacked. Looks at structural exploits and the half-life of any tactical novelty.
- **The Culture Custodian** - asks who you are when you are losing. Looks at non-negotiable standards, contested-ball identity, and what survives turnover.
- **The List Strategist** - asks where the list sits in its arc and whether you are pricing your trades correctly. Looks at multi-year horizon, re-signings, and draft capital.

The council also surfaces **convergence** (lenses agreeing on the same call - higher confidence) and **tensions** (lenses materially disagreeing - a real strategic choice you have to make, not a bug to smooth over).

---

### Confidence tiers - how to read FootyStrategy's recommendation

Every FootyStrategy output is tagged with a confidence tier. **Do not act on a tier without understanding what it permits.**

| Tier | What it means | What you do with it |
|---|---|---|
| **Settled** | Multiple lenses converge AND the upstream data is robust. The call is defensible against a senior coach's scrutiny. | Act with confidence. Build the game plan around it. Still pair with the tripwire. |
| **Probationary** | Lenses converge but the data is exploratory, partial, or has stated assumption weaknesses. The direction is right, the precision is not. | Act, but state the tripwire to your senior coach and be ready to reverse mid-game. Good for in-game wrinkles, not for list calls. |
| **Contested** | Lenses disagree materially. There is no single right answer. | Do not let FootyStrategy pick for you. Surface both options with their tripwires in the coaches' meeting. |
| **Insufficient Evidence** | Neither the data nor lens consensus supports a recommendation. | Do not invent a call. Send the missing inputs back to Scientist, then re-run FootyStrategy. |

---

### Tripwires - the falsifiable observable

**Every Settled or Probationary recommendation comes with a tripwire.** A tripwire is a specific, observable event that, if it occurs on match day, **reverses the call**.

Examples of well-formed tripwires:

- "If Yeo wins the first 3 contested balls of Q1, the tag is not working - switch Marlion Pickett off the tag and onto a free midfield role; zone Yeo instead."
- "If our forward-50 entries-per-quarter rise above 14 but our mark rate inside 50 stays under 18%, the bottleneck is entry quality, not forward structure."
- "If West Coast's clearance differential is +5 at quarter-time, our stoppage structure is being beaten - bring the wing in for the extra body at centre bounce."

If FootyStrategy gives you a Settled or Probationary recommendation **without** a tripwire, ask for one. A recommendation without a tripwire is a sermon, not strategy.

---

**Example prompts** (using the Richmond vs Adelaide R9 2026 result):

```
@"FootyStrategy (agent)" did Adelaide's half-time reset (Dawson HB→mid, Murray KPF→def) work because of their roster flexibility, or because Richmond had no contingency matchup plan? What's the actual tactical lesson for our pre-match brief structure?
```

```
@"FootyStrategy (agent)" is Richmond's Q3 structural collapse a one-off or a pattern? What does it tell us about their rotation depth when the hit-out count goes the wrong way, and what's the standard coaching response to that scenario?
```

```
@"FootyStrategy (agent)" build me a blueprint for how to attack Richmond next time we play them, based on what Adelaide did in Q3. I want zone-specific tactics, not just "tackle harder".
```

**Example prompts - live match use:**

```
@"FootyStrategy (agent)" quarter-time read - we're -12, Yeo has 9 disposals and 2 clearances, our tagger has 4 touches and zero impact. Do we stick with the tag, switch to a zone, or pull Pickett off and back our system? Give me a Probationary call with a tripwire I can watch in the first 5 minutes of Q2.
```

```
@"FootyStrategy (agent)" half-time, we're +4 but West Coast has won the last 8 minutes of Q2 and the inside-50 count is 14-22 against. Which lens do I trust here - hold structure or pre-empt their surge with a tempo change?
```

```
@"FootyStrategy (agent)" Q3 ten-minute mark, Tom Lynch is being chopped out by a double-team. Move him up the ground to drag a defender out, or hold him in the goalsquare? Two options, two tripwires, I need to call this in the next two minutes.
```

**FootyStrategy's superpower.** It can analyse each team's entire list by both quality tier *and* by what draft pick each player was - giving you a list-construction read alongside the tactical one. Adelaide's "system over stars" identity, for example, is partly a function of having no Pick-1-quality talent on the list - the structure has to compensate.

**What FootyStrategy doesn't have.** Same data limitations as Scientist - no GPS, no video, no live odds, no positional data, no Champion Data tags. If a question requires "where on the ground did Reid win his ball", it can't answer that any better than Scientist can.

**Scientist + FootyStrategy together - the workflow.** Scientist reads the data and surfaces the structural anomalies; FootyStrategy answers from football-coaching knowledge; Scientist writes the combined insight into the match doc with both halves cited. The joint output is sharper than either alone.

---
**Related:** [Using the Scientist agent](scientist-agent.md) · [2026 live season data](afl-season-2026.md)
---

## Leveraging the BriefBuilder agent

Scientist is the data depth. FootyStrategy is the football brain. **BriefBuilder is the skeleton crew.** It is a structured-assembly agent that auto-populates the data layer of a pre-match brief before any human or agent writes a single interpretive sentence. Given two team names and a round number, it opens the matches CSV, the player performance files, and the round's prediction CSV, then writes a complete tabular skeleton into `docs/coaches-strategy-corner/` — head-to-head ledger, season records for both sides, per-player form tables for the top 5 tracked players on each side, and model context including per-team prediction bias. Everywhere a tactical read would normally sit, it leaves a clearly labelled `<!-- FOOTYSTRATEGY INSERT -->` placeholder. Everywhere a methodology choice has implications, it leaves a `<!-- SCIENTIST REVIEW -->` flag. What it hands back is not a finished brief — it is a verified foundation that means nobody walks into Monday morning staring at a blank page.

**When to use it.** Monday morning, as the first thing you do after the weekend's result is confirmed and next week's opponent is known. Run BriefBuilder before you ask FootyStrategy for a tactical read and before you ask Scientist for a deeper regression. The sequence matters: BriefBuilder surfaces the numbers, FootyStrategy interprets them, Scientist tests the ones that are genuinely uncertain.

**How to invoke.** `@"BriefBuilder (agent)"` in a Claude Code session with team names and round.

**Example prompts:**

```
@"BriefBuilder (agent)" Richmond vs West Coast, round 12, 2026
```

```
@"BriefBuilder (agent)" Hawthorn vs Geelong, round 14, 2026, with H2H and exec-summary sub-docs
```

```
@"BriefBuilder (agent)" GWS vs Sydney, round 13, 2026 — once that's done, @"FootyStrategy (agent)" fill in the FOOTYSTRATEGY INSERT placeholders using the rivalry context and Sydney's current defensive structure
```

**What BriefBuilder produces vs what it leaves for humans.** BriefBuilder writes every number in the brief and nothing else. It opens the source files, reads the rows, and tags each figure `**[data]**` with the source named in the methodology paragraph. What it does not write: a single sentence about what any number means. Every tactical read, every momentum call, every "this matters because" line is left as an explicit placeholder for FootyStrategy to fill or for the coaching staff to supply from their own vision work.

**One honest limitation.** BriefBuilder applies a consistent template to every matchup. If the structural question for this week is something the template does not surface — a ruckman who only matters in wet-weather games, a player returning from a four-week absence whose season average is therefore misleading — BriefBuilder will not flag it as the key issue; it will present the numbers in the standard layout. Noticing what the skeleton is not telling you is your job.

---
**Related:** [Using the Scientist agent](scientist-agent.md) · [Leveraging the FootyStrategy agent](#leveraging-the-footystrategy-agent) · [2026 live season data](afl-season-2026.md)
---

## The DataSentinel — keeping the numbers honest

DataSentinel is a pre-commit verification gate. Before any document tagged with player statistics leaves your hands — a pre-match brief, a tactical review, a performance snapshot — DataSentinel walks every `**[data]**` tag and confirms the number against the actual source CSV in the repo. If a disposal average, a win-loss record, a Brownlow count, or a match margin doesn't match the data file, DataSentinel flags it before you print it. The goal is simple: no wrong numbers make it to the senior coach's desk.

**Why this matters.** You've built credibility over months of careful analysis. One wrong disposal average in Tuesday's review meeting — "he averaged 26 touches" when the data says 24 — erodes that credibility instantly. The senior coach starts second-guessing the whole brief. DataSentinel makes that verification automatic, not a manual checklist at 11 p.m. on Monday night.

**When it fires.** DataSentinel runs automatically as a pre-commit hook on any markdown document you're about to push that uses the repo's data-tag vocabulary. You can also invoke it manually on a draft: `@"DataSentinel (agent)"` in a Claude Code session points it at any document and runs the same check without blocking anything — you get the report, fix the draft, commit clean.

**How to invoke manually:**

```
@"DataSentinel (agent)" verify this draft document: docs/coaches-strategy-corner/richmond-vs-west-coast-brief.md
```

**Example use cases.**

*Pre-match brief before printing.* You've written a brief with "Elliot Yeo averaged 27.3 disposals this season" tagged `**[data]**`. DataSentinel opens the file, calculates the true 2026 average, and tells you: "Matched - 27.3 disposals confirmed" or "Mismatch - CSV shows 26.8." You fix the number before the brief goes anywhere.

*Player profile before showing the head coach.* You've pulled together a one-page profile with game totals, goals-per-game average, contested-mark percentage, and performance splits. Each stat is tagged. DataSentinel confirms all of them against the source files in one pass. If the contested marks are from a live snapshot that misindexes the column, you catch it before the head coach sees an unreliable number.

**What the output looks like.** DataSentinel emits a structured `PASS` or `FAIL` verdict. A `FAIL` gives you a list of which tags failed, why (CSV doesn't contain that value, source file is missing, stat from an unreliable live snapshot column), and which line of the document each error is on. Fix the number, rerun DataSentinel, commit clean.

The rule is: every number tagged `**[data]**` must be verifiable from a CSV in this repo. If you can't verify it, tag it `**[historical record - unverified in data]**` instead. DataSentinel passes that tag because you've signalled uncertainty. A number that's mismatched? That fails, every time.
---

## The Skeptic — stress-testing the brief before it goes upstairs

The Skeptic is your adversarial reviewer. Once FootyStrategy has finished drafting a brief — lens reads written, tensions surfaced, recommendation tiered — the Skeptic reads the whole thing cold and asks the questions a sceptical senior coach would ask in the meeting: is this tripwire something you can actually see from the box on match day, or is it a stat we only get on Tuesday? Did the recommendation quietly upgrade from "the data hints at this" to "we should do this"? When two lenses disagreed, did the brief honestly hold the tension, or did it smooth the disagreement into a comfortable consensus? The Skeptic does not write new analysis. It does not propose alternative recommendations. It interrogates what FootyStrategy produced and reports back.

### The three probes

1. **Tripwire observability.** Every Settled or Probationary recommendation must include a tripwire — the observable that would reverse the call. The Skeptic checks each tripwire against the data layers this repo can actually reach. A tripwire that fires on inside-50 differential is fine for a weekly review but useless at three-quarter-time — FanFooty's live snapshot does not carry it. A tripwire that says "if their tempo changes" is observable in theory and unfalsifiable in practice. The Skeptic flags both.

2. **Caveat-hierarchy fidelity.** Downstream confidence must never exceed upstream confidence. If the Scientist's finding was associational ("tagging correlates with lower output"), the recommendation cannot drift into causal framing ("tagging will suppress him"). The Skeptic walks the chain from Scientist finding to FootyStrategy verdict and flags any rung where the caveat got dropped, softened, or rewritten.

3. **Lens-tension smoothing.** If the List Strategist said "don't trade futures for a one-match edge" and the Innovator said "this is the meta window, trade for it," and the Tensions section reads "lenses broadly converged" — that is smoothed disagreement, and the Skeptic calls it out.

### When to use it

After FootyStrategy has drafted the brief and before it goes to the senior coach or gets published. The Skeptic is a gate, not a co-author — it runs once on the completed draft. If you are still iterating on the recommendation itself, the Skeptic is premature.

### How to invoke it

```
@"Skeptic (agent)" review docs/coaches-strategy-corner/2026-05-18-richmond-vs-stkilda-r11-brief.md
```

### What the Skeptic outputs

One of three verdicts:

- **PASS** — the three probes produced no material findings. Ship it.
- **PASS_WITH_CONCERNS** — the brief is not blocked, but specific issues are flagged for the author to consider. The author decides whether to incorporate.
- **BLOCK** — a clear-cut violation: a tripwire that cannot be observed in time, a tier that exceeds the upstream data confidence, causal language on associational evidence. The brief should not be committed until the blocker is addressed.

Critically, the Skeptic **never silently rewrites the draft**. Every concern comes with a line number, a verbatim quote, the rule being invoked, and a proposed fix. The author — you, with FootyStrategy's help — decides what to incorporate.

### A footy example

> `@"Skeptic (agent)"` review this draft: "Richmond should tag Elliot Yeo — historical data shows tagging suppresses his output by 18%. [Tier: Settled]. Tripwire: reverse the tag if his clearance count climbs in the second quarter."

The Skeptic will probe: (a) clearances are not in the FanFooty live snapshot schema — the tripwire is unobservable in real time, **BLOCK**; (b) "suppresses" is causal, but if the upstream finding was correlational, the language has drifted, **CONCERN**; (c) was the Match-up Tactician the only lens activated? A tag decision affects the whole midfield structure — missing-lens flag.

### Why an adversarial agent matters in a coaching environment

Match preparation rooms are high-trust, high-tempo, and densely consensual by Friday afternoon. Once a tactical read has been articulated confidently by the head of analysis and nodded along to by two assistants, it is socially expensive to be the person who says "but the tripwire isn't observable." Group-think is the enemy of good match preparation. The Skeptic carries that cost-free. It has no relationship to protect, no meeting to get through, no senior coach to keep onside. It reads the brief the way the opposition's analyst would read it: looking for the gap between what the data supports and what the recommendation claims. Every flag it raises is a flag you would rather see on Friday than explain on Monday.
---

## Getting all six agents to brainstorm together

Calling one agent gets you one perspective. Running all six in sequence gets you a brief that survives contact with Saturday afternoon. This section is the operating manual for chaining them — when to use which pipeline, what to type into Claude Code, and how to keep the brainstorm honest.

### The conductor rule (read this first)

The agents do not talk to each other. There is no shared memory between them, no orchestration layer that auto-forwards Scientist's findings to FootyStrategy. **You are the conductor.** Each agent only sees what you paste into its prompt. The quality of the brainstorm is the quality of the hand-offs — you read the previous agent's output, extract the load-bearing claims, and paste them as context into the next agent's prompt. If you skip this, FootyStrategy will hallucinate stats that Scientist never produced, and DataSentinel will have nothing to verify against.

### Mode 1 — Standard brainstorm pipeline (Mon–Thu pre-match)

**BriefBuilder → Scientist → FootyStrategy → DataSentinel → Skeptic**

Worked example: **Richmond vs West Coast, Round 12, 2026.**

**Step 1 — BriefBuilder**
```
@"BriefBuilder (agent)" Richmond vs West Coast, Round 12 2026.
Build the pre-match brief: H2H last 10 meetings, current ladder position, last 5 form for both, top-3 disposal-getters per side, predicted line-up changes, and the projected disposal/goal lines from the latest prediction file.
```

**Step 2 — Scientist**
```
@"Scientist (agent)" Using the BriefBuilder output below, test whether West Coast's last-quarter fade is real or sample noise. Pull their Q4 score-conceded vs Q1-Q3 over the last 12 games, run a paired-difference test, report effect size + CI, and tell me whether the gap survives a sensitivity check on the two biggest blow-outs.

[paste BriefBuilder output here]
```

**Step 3 — FootyStrategy**
```
@"FootyStrategy (agent)" Using the Scientist's findings + the BriefBuilder brief, run the eight-lens council on the question: "Should Richmond press high in Q4 against West Coast, or sit back and let them come?" Give me tiered recommendations and the tripwires that would force a switch.

[paste BriefBuilder + Scientist outputs here]
```

**Step 4 — DataSentinel**
```
@"DataSentinel (agent)" Verify every [data] tag and every specific number in the brief below — player game totals, H2H record, last-quarter differentials, predicted disposal lines.

[paste the full assembled brief here]
```

**Step 5 — Skeptic**
```
@"Skeptic (agent)" Review the full brief below. Three things: (1) are FootyStrategy's tripwires actually observable from the broadcast in real time? (2) did confidence drift upward between Scientist's hedged finding and FootyStrategy's recommendation? (3) what's the single weakest link in the causal chain?

[paste the full assembled, DataSentinel-verified brief here]
```

If Skeptic raises a substantive concern, loop back to the relevant agent — do not paper over it.

### Mode 2 — Rapid brainstorm (in-game, ~15 minutes)

When the match is live and you have one quarter to make a call, skip BriefBuilder and compress to two agents:

**Scientist → FootyStrategy**

```
@"Scientist (agent)" Quick: West Coast are +18 at half-time but their last-quarter concession-rate over the last 12 games is worse than their season average. Is the gap big enough to bet on, or noise? One number, one CI, one sentence.
```

```
@"FootyStrategy (agent)" Half-time, West Coast +18, Scientist says their Q4 fade is real at [effect size from above]. Do we hold structure or chase now? One tiered recommendation, one tripwire for switching.
```

No DataSentinel, no Skeptic — accept the higher residual risk. That is the price of speed.

### Mode 3 — Devil's-advocate brainstorm (stress-test a call you've already made)

You think you should tag Yeo. Before you commit, force the system to argue the other side.

**Scientist (evidence against) → FootyStrategy (best case against) → Skeptic (audit the pro-tag case)**

```
@"Scientist (agent)" I'm planning to put a hard tag on Elliot Yeo. Find evidence AGAINST this call: games where Yeo had a quiet day without a tagger and West Coast still won; correlation between Yeo's disposals and West Coast's score. Be adversarial — your job is to talk me out of it.
```

```
@"FootyStrategy (agent)" Using the Scientist's counter-evidence, build the strongest possible tactical case for NOT tagging Yeo. What does the eight-lens council look like when the prior is "don't tag"? Tripwires that would prove the no-tag call wrong in-game.
```

```
@"Skeptic (agent)" Here is my original pro-tagging brief [paste]. Here is the best case against [paste FootyStrategy output]. Review the pro-tagging brief for: caveat drift, tripwire observability, and any places I treated correlation as causation. Does the pro-tag case survive?
```

If the pro-tag brief survives Skeptic after being attacked, your conviction is earned. If it does not, you have just avoided a bad call.

### Brainstorm modes at a glance

| Mode | Agents used | Time required | Use case |
|---|---|---|---|
| **Standard** | BriefBuilder → Scientist → FootyStrategy → DataSentinel → Skeptic | 60–90 min | Pre-match preparation with runway; decision-grade brief |
| **Rapid** | Scientist → FootyStrategy | ~15 min | Live tactical call between quarters |
| **Devil's-advocate** | Scientist → FootyStrategy → Skeptic | 30–45 min | Stress-testing a call you've already drafted |
| **Code-level deep-dive** | BriefBuilder → Scientist → Codex → FootyStrategy | 90+ min | When the question touches model internals (why does the predictor rate Curnow so highly this week?) |
| **Full council (all six)** | BriefBuilder → Scientist → FootyStrategy → Codex → DataSentinel → Skeptic | 2+ hrs | High-stakes, finals-level briefs; canonical numbers cited downstream |

### Copy-paste starter block — full six-agent session

```
# === BLOCK 1: BriefBuilder ===
@"BriefBuilder (agent)" Richmond vs West Coast, Round 12 2026.
Assemble the pre-match brief: H2H last 10, ladder, last-5 form,
top-3 disposal-getters per side, predicted line-up changes,
disposal + goal projections from latest prediction file.

# [wait for output, then:]

# === BLOCK 2: Scientist ===
@"Scientist (agent)" Using the BriefBuilder brief below, test:
(a) is West Coast's last-quarter fade real or noise?
(b) is the disposal-line prediction for Yeo well-calibrated historically?
Report effect sizes, CIs, and a sensitivity check.

[paste BriefBuilder output]

# [wait, then:]

# === BLOCK 3: FootyStrategy ===
@"FootyStrategy (agent)" Using the Scientist's findings + the BriefBuilder brief,
run the eight-lens council on Richmond's match-up plan.
Tiered recommendations + tripwires.

[paste BriefBuilder + Scientist outputs]

# [wait, then:]

# === BLOCK 4: Codex ===
@"Codex (agent)" Outside-the-frame: look at the brief below and tell me what
we're NOT considering. The predictor rates Yeo at 28 disposals — pull the
relevant feature contributions and tell me whether the prediction is driven
by form, opponent, or venue.

[paste full assembled brief]

# [wait, then:]

# === BLOCK 5: DataSentinel ===
@"DataSentinel (agent)" Verify every [data] tag and every specific number
in the brief below against data/player_data/, data/matches/, data/prediction/.
Flag anything that cannot be confirmed.

[paste full assembled brief, including Codex notes]

# [wait, then:]

# === BLOCK 6: Skeptic ===
@"Skeptic (agent)" Adversarial review of the final, verified brief below.
Check: tripwire observability, caveat drift between Scientist's hedged findings
and FootyStrategy's recommendations, weakest link in the causal chain,
and any place Codex's outside-the-frame point was waved away rather than addressed.

[paste full verified brief]
```

The output of Block 6 is the brief you take into the match. If Skeptic flags a structural issue, the brief is not done — loop back to the responsible agent, fix the issue, and re-run DataSentinel + Skeptic on the patched version. The point of the pipeline is not to generate a brief; it is to generate a brief you trust on Saturday at 2:10pm.
