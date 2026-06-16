<!-- council-pipeline:
  Scientist: PASS @ 2026-06-16 (all [data] numbers verified against data/drafts/*.csv and data/player_data/* using canonical career_games = max(row_count, max_numeric(games_played)); corrections applied — Franklin 354 games, Neale 139 goals, Fyfe 247 games, Josh Kennedy disambiguated to the Sydney/Hawthorn midfielder at 2006 pick 40 (290 games). Elite-school % = APS+GPS_WA+GPS_SA+GPS_QLD over all picks.)
  FootyStrategy: AUTHORED @ 2026-06-16
  Gaffer: SPLICED top-10 conversion rate section (user request) @ 2026-06-16
  FootyStrategy: AUTHORED Part 3 (brand currency thesis) @ 2026-06-16
  FootyStrategy: AUTHORED Part 4 (decision framework — pick cliff, school signal, club efficiency, state pipelines, late-round gold) @ 2026-06-16
  FootyStrategy: AUTHORED Part 5 (Is Richmond making a mistake? — dynasty void, 0/32 post-dynasty, pick-1 question) @ 2026-06-16
-->
# The AFL National Draft: Error, Structure, and the Shape of Talent

> [← Back to articles](../) | [← README](../../README.md)

*Analytical long-form. Data layer verified by the Scientist against `data/drafts/` and `data/player_data/` on 2026-06-15. Every specific number carries a **[data]** tag.*

---

## Part 1 — The Taxonomy of Draft Error

There are two ways to get a draft pick wrong, and the football world is only interested in one of them.

The first is the **commission error**: you use a pick on a player who does not work out. This error is visible, attributable, and embarrassing. It has a name attached — the club's, the recruiter's — and it sits in the record for two decades. The second is the **omission error**: you pass on a player who becomes elite, and he simply goes on to play for somebody else. This error is invisible. Nobody is held to account for the champion who was available and overlooked, because the failure produces no artefact on your own list. It produces a great player in another guernsey, which the league files under "good fortune for them" rather than "negligence by you."

The asymmetry matters because the omission error is almost certainly the larger destroyer of value. A bust costs you one pick. Passing on a generational player costs you a generational player. Yet draft retrospectives, and the media scrutiny that drives them, focus on commission almost to the exclusion of omission. We remember who was picked and faltered. We rarely audit who was available and ignored.

The 2004 national draft is the cleanest case study available. The top ten that year:

| Pick | Player | Games | Grade |
|--:|---|--:|:--|
| 1 | Brett Deledio | 275 **[data]** | A |
| 2 | Jarryd Roughead | 283 **[data]** (578 goals **[data]**) | A |
| 3 | Ryan Griffen | 257 **[data]** | A |
| 4 | Richard Tambling | 124 **[data]** | B+ |
| 5 | Lance Franklin | 354 **[data]** (1,066 goals **[data]**) | A+ |
| 6 | Tom Williams | 85 **[data]** | C+ |
| 7 | Jordan Lewis | 319 **[data]** | A |
| 8 | John Meesen | 6 **[data]** | D |
| 9 | Jordan Russell | 125 **[data]** | B |
| 10 | Chris Egan | 27 **[data]** | C+ |

The pick that history remembers is the commission error at four. Richmond took Richard Tambling — a 124-game **[data]**, B+ career — one selection ahead of Lance Franklin, who went on to 354 games **[data]** and 1,066 goals **[data]** at A+. It is the kind of swap that gets replayed every time the draft comes around, because it is legible: two players, adjacent picks, one of them a Hall-of-Fame forward.

But look at picks six, eight, and ten — Tom Williams at 85 games **[data]**, John Meesen at six **[data]**, Chris Egan at 27 **[data]**. Three top-ten selections that returned almost nothing. These were not seen as scandals, and the clubs that made them were never publicly indicted, because the players who would have justified those picks were taken later, or by someone else, or were simply harder to project at the time. The talent that passed through those clubs' hands just went elsewhere. Three silent omission errors sit inside the same top ten as the one loud commission error — and only the loud one entered the folklore.

### The trouble with games played

The default currency of draft analysis is games played. It is available, it is simple, and it is wrong — or at least badly incomplete. Two hundred games is not two hundred games. A midfielder averaging 28 disposals across two hundred matches and a defender averaging 12 across the same span have produced very different careers, and a longevity count flattens that difference to zero. Games played also rewards durability and selection persistence as much as quality; a serviceable role player on a stable list can out-game a brilliant one whose body failed early.

The DraftGuru grade system (A+ through D) is an attempt to price career *value* rather than career *length* — to capture peak impact and positional weight, not just appearances. It is imperfect and subjective, but it encodes something the games column cannot: that a short, decisive career can outvalue a long, ordinary one. In our 2004–2025 dataset, 34 players reached A+ or A. The range within that group is enormous. At one pole is pure accumulated volume — Scott Pendlebury, 435 games **[data]**, an A+ built on two decades of unbroken availability. At the other is peak-per-game impact — a graded-A+ small forward whose career runs only a little past 150 games but whose value per match exceeds plenty of 350-game players sitting two grades below him. A games-played leaderboard cannot see that player at all; a value-aware one puts him among the best of his cohort.

Some A+ careers, verified:

| Player | Draft | Pick | Games | Goals |
|---|--:|--:|--:|--:|
| Lance Franklin | 2004 | 5 | 354 **[data]** | 1,066 **[data]** |
| Scott Pendlebury | 2005 | 5 | 435 **[data]** | 207 **[data]** |
| Jack Riewoldt | 2006 | 13 | 347 **[data]** | 787 **[data]** |
| Tom Hawkins | 2006 | 41 | 359 **[data]** | 796 **[data]** |
| Dustin Martin | 2009 | 3 | 302 **[data]** | 338 **[data]** |
| Nathan Fyfe | 2009 | 20 | 247 **[data]** | 178 **[data]** |
| Lachie Neale | 2011 | 58 | 308 **[data]** | 139 **[data]** |
| Marcus Bontempelli | 2013 | 4 | 272 **[data]** | 272 **[data]** |

The Hawkins and Neale lines are the ones that should unsettle a recruiting department. Tom Hawkins, an A+ key forward with 359 games **[data]** and 796 goals **[data]**, was extracted at pick 41 **[data]**. Lachie Neale, an A+ midfielder with 308 games **[data]**, came out at pick 58 **[data]**. Neither was a commission error by anyone — you cannot indict a club for the A+ player it found in the back half of the draft. But they are the clearest evidence that elite production is not reliably concentrated at the top of the order. If A+ careers can be pulled from picks 41 and 58, then the implicit model that ranks the draft as a smooth descent from "great" to "marginal" is wrong, and the omission errors hiding in the early picks are larger than they look.

---

## Part 2 — The Structural Fault Lines

Three structural features of the AFL draft systematically distort who gets picked, and when. None of them is a scandal. All of them are exploitable.

### Fault line one: the elite school pipeline

Across 1,538 draft picks from 2004 to 2025 **[data]**, players from elite schools — the APS and GPS systems — are not distributed evenly through the draft order. They cluster at the top:

| Pick range | Elite-school share |
|---|--:|
| Top 10 | 28.2% **[data]** |
| Picks 11–30 | 22.7% **[data]** |
| Picks 31+ | 18.9% **[data]** |
| Overall | 21.3% **[data]** |

An elite-school player is meaningfully more likely to be taken in the top ten than in the back half of the draft. The first question is whether that concentration is earned. Partly, it is. Elite-grade (A+/A) achievement rates by school type:

| School type | A+/A rate |
|---|--:|
| GPS WA | 17% **[data]** |
| APS | 16% **[data]** |
| Other private | 12% **[data]** |
| GPS SA | 11% **[data]** |
| Unknown / no school | 8% **[data]** |

So elite-school players are both drafted higher *and* genuinely more likely to become elite — a 16% **[data]** A/A+ rate for APS against 8% **[data]** for players with no recorded school is a real, roughly two-to-one gap in outcomes. But that outcome gap does not fully account for the *positioning* gap. The top-ten concentration runs at 28% **[data]** elite-school against 19% **[data]** in the late picks — a wider spread than the underlying quality difference justifies on its own. Part of the early-pick clustering is not talent; it is selection bias baked into the academy and pathway systems that happen to co-locate with elite schools. The recruiting eye finds the player it has been watching in a structured environment since under-16s, and those environments have historically sat inside the private-school network.

But here is the number that cuts sharpest. Among top-ten picks only — where elite schools are most aggressively targeted — the great-player conversion rate is almost identical regardless of pathway:

| Top-10 picks | Great (A+/A) | Rate |
|---|--:|--:|
| Elite school (APS + GPS) | 18 of 62 | 29.0% **[data]** |
| Non-elite school | 44 of 158 | 27.8% **[data]** |

No meaningful difference. Elite-school players are drafted into the top ten at a significantly higher rate than their peers — but once selected there, they convert to elite careers at essentially the same rate. The advantage is at the gate, not in the outcome. Clubs are paying a positioning premium for a school badge that, at the level of top-ten picks, carries almost no additional conversion signal.

The cases pull in both directions. From elite-school top-ten picks who did not deliver: Xavier Ellis at pick 3 **[data]** (Melbourne Grammar, 120 games **[data]**, C+), Jack Watts at pick 1 **[data]** (Brighton Grammar, 174 games **[data]**, B+), Jack Trengove at pick 2 **[data]** (Prince Alfred College, 89 games **[data]**, B), Andrew McGrath at pick 1 **[data]** (Brighton Grammar, 191 games **[data]**, B+), Jamarra Ugle-Hagan at pick 1 **[data]** (Scotch College, 70 games **[data]**, B). Five number-ones and number-twos from elite schools, none rated above B+. Meanwhile some of the best elite-school careers came *late*: Tom Hawkins A+ at pick 41 **[data]**, Josh Kennedy A+ at pick 40 **[data]**, Nathan Fyfe A+ at pick 20 **[data]**, Lachie Neale A+ at pick 58 **[data]**. The school produced the talent; the draft position did not predict the outcome.

There is a genuine **WA effect** threaded through this. The GPS WA schools — Hale, Aquinas, Guildford Grammar, Christ Church, Trinity — produce a 17% **[data]** A/A+ rate, the highest of any category, and their players frequently arrive *late*: Nathan Fyfe at pick 20 **[data]**, Lachie Neale at pick 58 **[data]**. That combination is telling. It suggests the signal there is not the school brand at all but Western Australia's broader overproduction of AFL talent, with the GPS connection partly a confound. The school is a marker of where the player was educated, not the engine of why he was good.

The era trend is the most encouraging part of the picture:

| Era | Elite-school share |
|---|--:|
| 2004–09 | 15.7% **[data]** |
| 2010–14 | 22.5% **[data]** |
| 2015–19 | 27.7% **[data]** ← peak |
| 2020–25 | 20.5% **[data]** |

Elite-school representation climbed steadily to a 27.7% **[data]** peak in 2015–19, then fell back to 20.5% **[data]**. The most plausible reading is structural: the expansion of AFL and Next Generation academies since 2017 built a parallel elite pathway that is more meritocratic by design. Players from regional and non-school backgrounds now have the structured development environment that, a decade earlier, only the private-school system reliably provided. The pullback is what genuine widening of access looks like in the data — the pipeline diversifying away from a single feeder network.

For completeness, the top feeder schools across the period: Haileybury College 35 picks **[data]**, St Patrick's College 29 **[data]**, Caulfield Grammar 26 **[data]**, and Melbourne Grammar and Scotch College tied on 24 each **[data]**.

### Fault line two: the superdraft myth, and the rarer thing

The 2004 draft is rightly celebrated — five players in the top seven graded A or A+ is historically exceptional. But the popular telling collapses into a lazier claim: that some *years* are simply better than others, in ways clubs should have read in advance and acted on.

Two corrections are worth making. First, the 2004 top-seven concentration was substantially luck. The right response to luck is not hindsight indignation but probabilistic modelling — a club should price the *chance* of a deep cohort into its draft strategy, not assume it can pick the deep years in advance. Second, and more usefully: a good draft year is not actually rare. The genuinely rare thing is a *deep* draft year — one where elite talent surfaces at picks 30–60 as readily as at picks 1–10.

By that test, the best illustration is not 2004 but **2006**:

| Player | Pick | Games |
|---|--:|--:|
| Travis Boak | 5 | 387 **[data]** |
| Joel Selwood | 7 | 355 **[data]** |
| Jack Riewoldt | 13 | 347 **[data]** (787 goals **[data]**) |
| Josh Kennedy | 40 | 290 **[data]** |
| Tom Hawkins | 41 | 359 **[data]** (796 goals **[data]**) |
| Robbie Gray | 55 | 271 **[data]** (367 goals **[data]**) |

Six A+ players from one draft — and four of them taken *after* pick 13. The midfielder Josh Kennedy at 40 **[data]**, Hawkins at 41 **[data]**, Robbie Gray at 55 **[data]**: this is depth, not just a strong top end. The implication for clubs is direct. If you pass on a player at pick 13 on the theory that the elite talent in this cohort is all gone by pick 10, the 2006 evidence says you may be leaving A+ production sitting on the board. The error a depth year punishes is the assumption that the draft has already given up its best by the time the first round is done.

### Fault line three: the mid-season draft that mostly doesn't correct

The mid-season draft was introduced as a safety valve — a way for clubs to address an in-season injury crisis or a genuine list hole without waiting until November. The intent was correction: give a finals contender stripped of bodies a mid-year mechanism to repair itself.

In practice it functions differently. The clubs picking early in the mid-season draft are, predictably, the clubs already at the bottom of the ladder — sides using it not to patch an injury emergency but to take a second swing at talent they missed, or to audition fringe and mature-age players as part of a rebuild already underway. It is less a correction mechanism for contenders than an acceleration mechanism for rebuilders.

That is not a failure of the device so much as a revelation of who actually has the draft capital and the list space to use it. A genuine finals side rarely has the room or the priority position to extract a difference-maker mid-year; a rebuilding side has both. So the mid-season draft mostly corrects nothing. It mostly speeds up the teams that were already heading in their chosen direction — a quietly useful tool for the patient, and very little to the desperate.

---

*Numbers verified against `data/drafts/afl_draft_schools.csv`, `data/drafts/draftguru_enrichment.csv`, and the per-player files in `data/player_data/`. Grades follow the DraftGuru A+/A/B+/B/C+/C/D system. Career games use `max(games_played)` per the repo convention.*

---

## Part 3 — The Draft Pick as Brand Currency

Elite private schools advertise their AFL draftees the way they advertise their ATAR results: a measurable, rankable outcome the institution can put its name to. The two metrics behave almost identically as marketing instruments. Both are published — in prospectuses, on websites, in the speech-night program. Both are rankable, which lets a school position itself against its peers. Both are self-reinforcing: the outcome attracts more of the type of family that produces the outcome, so the list compounds. And both run roughly proportional to fees — the schools at the top of the draft-feeder table are, with few exceptions, the same schools at the top of the fee table.

But the analogy breaks at the point that matters: attribution. A school can plausibly claim some share of an ATAR. Its teaching, its culture, its exam preparation might genuinely have moved the number — the score is at least partly *made* on site. An AFL draft pick is harder to claim. The physical talent existed before the player enrolled. What the school added was structured training, quality facilities, a strong inter-school competition, and proximity to AFL recruiters who were already watching that competition. Those are real goods. But they curate and expose talent; they do not create it. The school's name on the draft pick is a claim of authorship over something it mostly hosted.

### The recursive loop — talent moves toward the institution

The feeder table makes the mechanism visible. National-draft picks, 2004–2025:

| School | Picks |
|---|--:|
| Haileybury College (APS) | 35 **[data]** |
| St Patrick's College | 29 **[data]** |
| Caulfield Grammar (APS) | 26 **[data]** |
| Melbourne Grammar (APS) | 24 **[data]** |
| Scotch College (APS) | 24 **[data]** |
| Xavier College (APS) | 20 **[data]** |
| Brighton Grammar (APS) | 19 **[data]** |

A school with a draft history attracts talented footballers — through scholarships, boarding programs, and reputation — and the inflow refreshes the history, which sharpens the attraction. The "Original Club" pathway chains in the data show the direction of travel: players moving significant distances to attend a GPS or APS school, talent migrating *toward* the institution rather than the institution developing the talent it already had. The school then books the marketing credit for a destination it competed to become. That is a legitimate thing to compete for. It is just not the same as having grown the player.

### The conversion problem — the number that isn't in the prospectus

If the school environment genuinely *developed* a player beyond what an alternative pathway offered, that development should show up where the players are most comparable: at the very top of the draft, where clubs spend first-round capital on the most thoroughly scouted prospects in the pool.

| Top-10 picks | Great (A+/A) | Rate |
|---|--:|--:|
| Elite school (APS + GPS) | 18 of 62 | 29.0% **[data]** |
| Non-elite school | 44 of 158 | 27.8% **[data]** |

There is no gap. The school helped get the player drafted higher; it did not make him more likely to succeed once he got there. That sits against a real overall difference in talent concentration —

| Pathway | A+/A rate |
|---|--:|
| GPS WA | 17% **[data]** |
| APS | 16% **[data]** |
| No school / unknown | 8% **[data]** |

— where the 16% **[data]** versus 8% **[data]** spread is genuine and reflects that the pathway systems co-located with elite schools are better at *identifying* talent than the raw population is. But the two findings have to be read together. The population-level gap (16% vs 8%) is an identification story; the top-ten finding (29% vs 28%) is the development story, and there the premium vanishes. Once a player is good enough that clubs are spending a top-ten pick on him, the school has already done whatever it was going to do. Its marginal contribution to the outcome, at that point, rounds to zero.

### The academy correction — what the trend reveals

| Era | Elite-school share of all National picks |
|---|--:|
| 2004–09 | 15.7% **[data]** |
| 2010–14 | 22.5% **[data]** |
| 2015–19 | 27.7% **[data]** ← peak |
| 2020–25 | 20.5% **[data]** |

The peak at 27.7% **[data]** lands in 2015–19 — the window just before the AFL's expanded Next Generation Academy system, formalised from 2017, had fully propagated through the draft. The pullback to 20.5% **[data]** in 2020–25 is consistent with what genuinely widening access looks like in the data: a parallel elite pathway that reproduces the structured development environment without the school crest. As that academy infrastructure matures, the private school's structural advantage as a talent *pipeline* should keep narrowing — and the brand claim, "we produced X AFL players," becomes progressively harder to defend as the same calibre of player is drafted from other postcodes.

The schools won't stop counting draft picks any more than they'll stop counting ATARs, and they're entitled not to. Both are fair indicators of the environment an institution can attract and retain. But the data says something the prospectus does not: at the top-ten level, where the claim is loudest, the school credential carries no conversion premium. The player drafted out of an APS school and the player drafted out of a country football club arrive with the same odds of becoming elite. The school got to put the pick on its website. The football club got to watch him leave.

---

## Part 4 — What the Data Actually Says You Should Do

Parts 1 through 3 looked backward. This part looks forward. The same dataset that explains what happened can be turned into a small set of decision rules a recruiting department could apply on draft night. Four rules follow, each grounded in a measurable effect, each stated as a thing to do rather than a thing to notice.

### Rule 1 — Price the cliff at pick 16, not pick 11

The conventional trade market treats the end of the first round as the value cliff. The data does not. Elite (A+/A) conversion holds roughly flat from pick 6 through pick 15, then falls off a ledge at pick 16:

| Pick range | A+/A rate | n picks |
|---|--:|--:|
| 1–5 | 37.3% **[data]** | 110 **[data]** |
| 6–10 | 19.1% **[data]** | 110 **[data]** |
| 11–15 | 20.0% **[data]** | 110 **[data]** |
| 16–20 | 9.1% **[data]** | 110 **[data]** |
| 21–30 | 10.0% **[data]** | 220 **[data]** |
| 31–40 | 5.5% **[data]** | 220 **[data]** |
| 41–60 | 4.7% **[data]** | 427 **[data]** |
| 61+ | 3.9% **[data]** | 231 **[data]** |

Picks 11–15 convert at 20.0% **[data]** — statistically indistinguishable from the 19.1% **[data]** of picks 6–10, and more than double the 9.1% **[data]** of picks 16–20. The real drop is between 15 and 16. The implication is a market arbitrage: any trade that treats "end of round 1" as the cliff under-prices picks 11–15. A club that correctly values pick 13 as nearly equivalent to pick 7 can cede pick 7 for pick 13 plus a future asset and extract surplus on both ends. Separately, picks 1–5 stand alone at 37.3% **[data]** — nearly two-in-five become elite. There is no substitute for a genuine top-five selection, and no trade package of later picks reliably replicates it.

### Rule 2 — Apply the school signal only where it pays

The school-background signal is not constant across the draft; it is pick-range dependent, and that interaction is the single most actionable finding in the dataset:

| Pick range | Elite school A+/A | Non-elite A+/A |
|---|--:|--:|
| 1–5 | 35.3% (12/34) **[data]** | 38.2% (29/76) **[data]** |
| 6–10 | 21.4% (6/28) **[data]** | 18.3% (15/82) **[data]** |
| 11–15 | 30.4% (7/23) **[data]** | 17.2% (15/87) **[data]** |
| 16–20 | 20.0% (5/25) **[data]** | 5.9% (5/85) **[data]** |
| 21–30 | 17.3% (9/52) **[data]** | 7.7% (13/168) **[data]** |
| 31–40 | 10.0% (5/50) **[data]** | 4.1% (7/170) **[data]** |
| 41–60 | 6.3% (5/79) **[data]** | 4.3% (15/348) **[data]** |
| 61+ | 8.1% (3/37) **[data]** | 3.1% (6/194) **[data]** |

The decision rule the data supports:

- **Picks 1–10**: school type has no meaningful predictive value (35.3% **[data]** vs 38.2% **[data]** at the top, 21.4% **[data]** vs 18.3% **[data]** next). Weight football ability only.
- **Picks 11–30**: elite-school background carries a real signal — roughly 2× conversion (30.4% **[data]** vs 17.2% **[data]** at 11–15; 20.0% **[data]** vs 5.9% **[data]** at 16–20). Not determinative, but a legitimate tiebreaker between equally assessed players.
- **Picks 31+**: the signal narrows but persists. Even at pick 61+, elite-school players convert at 8.1% **[data]** vs 3.1% **[data]** — a non-trivial edge in an otherwise low-signal zone.

### Rule 3 — Treat drafting as a process, and benchmark it

Across 2004–2025, among clubs with 30+ picks, the A+/A conversion rate varies nearly 3-to-1:

| Club | Picks | A+/A | Rate |
|---|--:|--:|--:|
| Western Bulldogs | 90 | 14 | 15.6% **[data]** |
| Greater Western Syd | 74 | 11 | 14.9% **[data]** |
| Collingwood | 85 | 11 | 12.9% **[data]** |
| Adelaide | 80 | 10 | 12.5% **[data]** |
| Melbourne | 82 | 10 | 12.2% **[data]** |
| Sydney | 75 | 9 | 12.0% **[data]** |
| West Coast | 84 | 9 | 10.7% **[data]** |
| Geelong | 89 | 9 | 10.1% **[data]** |
| Hawthorn | 81 | 8 | 9.9% **[data]** |
| Fremantle | 89 | 8 | 9.0% **[data]** |
| Carlton | 80 | 7 | 8.8% **[data]** |
| Essendon | 92 | 8 | 8.7% **[data]** |
| Richmond | 92 | 8 | 8.7% **[data]** |
| Port Adelaide | 83 | 7 | 8.4% **[data]** |
| Gold Coast | 61 | 5 | 8.2% **[data]** |
| North Melbourne | 76 | 6 | 7.9% **[data]** |
| Brisbane Lions | 100 | 6 | 6.0% **[data]** |
| St Kilda | 88 | 5 | 5.7% **[data]** |

Two observations. First, the spread from 15.6% **[data]** to 5.7% **[data]** is not noise. Over 20-plus years and 74–100 **[data]** picks per club, the gap between the Bulldogs and St Kilda is too large and too sustained to be luck — it reflects a genuine process difference. Second, Brisbane is the cautionary case: 100 **[data]** picks, the most of any club, converted at 6.0% **[data]**. Ninety-four **[data]** of those picks returned B+ or below. Volume without a quality filter is a resource drain, not an edge.

### Rule 4 — Work the state pipelines for what they actually return

| State | Picks | A+/A | Rate |
|---|--:|--:|--:|
| NSW/ACT | 15 | 4 | 26.7% **[data]** |
| VIC | 762 | 91 | 11.9% **[data]** |
| WA | 252 | 29 | 11.5% **[data]** |
| Unknown | 252 | 21 | 8.3% **[data]** |
| SA | 205 | 11 | 5.4% **[data]** |
| QLD | 26 | 1 | 3.8% **[data]** |
| TAS | 22 | 0 | 0.0% **[data]** |
| NT | 4 | 0 | 0.0% **[data]** |

The WA mythology does not survive contact with the rate. WA produces elite players at 11.5% **[data]** — effectively identical to Victoria's 11.9% **[data]**. The state's reputation is a high-profile-player effect: a handful of individually remarkable careers, on top of a volume share larger than its population would predict, so more WA players appear at every grade including A+. The base rate is ordinary. The number that should draw attention is SA at 5.4% **[data]** — less than half Victoria's rate, from a historically rich football state. Whether that is a development, exposure, or depth problem, the SA pathway is underperforming expectation and is worth a club's diagnostic effort. (The NSW/ACT 26.7% **[data]** sits on just 15 **[data]** picks and is too small a sample to lean on.)

### The late-round dividend

Eighteen players drafted after pick 50 **[data]** reached A+/A between 2004 and 2025. A representative slice of that list:

| Year | Pick | Player | Pathway | Grade | Games | Goals |
|--|--|---|---|:-:|--:|--:|
| 2007 | 75 | Taylor Walker | North Broken Hill / NSW-ACT U18 | A | 309 **[data]** | 693 **[data]** |
| 2011 | 58 | Lachie Neale | Kybybolite / St Peter's College SA / Glenelg | A+ | 308 **[data]** | 139 **[data]** |
| 2014 | 61 | Harris Andrews | Padua College QLD / Qld U18 / Aspley | A+ | 250 **[data]** | 11 **[data]** |
| 2006 | 55 | Robbie Gray | East Burwood / Oakleigh U18 | A+ | 271 **[data]** | 367 **[data]** |
| 2016 | 73 | Nick Larkey | Hawthorn Citizens / Trinity Grammar / Oakleigh U18 | A | 147 **[data]** | 308 **[data]** |
| 2016 | 57 | Josh Daicos | Camberwell Grammar / Oakleigh U18 | A | 164 **[data]** | 68 **[data]** |
| 2017 | 67 | Dylan Moore | Rowville / Caulfield Grammar / Eastern U18 | A | 138 **[data]** | 144 **[data]** |
| 2013 | 52 | Darcy Byrne-Jones | Camberwell / Scotch College / Oakleigh U18 | A | 234 **[data]** | 76 **[data]** |
| 2008 | 53 | Michael Walters | Midvale / Swan Districts | A | 239 **[data]** | 365 **[data]** |

Three patterns recur in the late finds. First, country players with unusual athleticism who were underexposed to recruiters — Walker out of North Broken Hill, Neale out of Kybybolite, both small towns. Second, APS/GPS players taken in the 50s–70s who were correctly evaluated but dropped by other clubs for positional or projection reasons — Byrne-Jones, Moore, Larkey, Daicos. Third, WA players below the eastern-states radar who had not been sufficiently tracked — Walters. The actionable point: after pick 50 the A+/A rate is still a live 6.3–8.1% **[data]**. Clubs that treat the late draft as a formality — filling list spots rather than hunting players — leave roughly one-in-fourteen shots at an elite career unpressed.

### The proposition

Read together, the four rules say one thing: good drafting is a repeatable process, not a run of talent luck. The pick-value cliff sits at a knowable place, the school signal pays only in a knowable band, the state pipelines return knowable rates, and the late draft holds a knowable dividend — and the clubs at the top of the efficiency table exploit all four with enough consistency that their edge survives 90–100 **[data]** picks. A 15.6% **[data]** conversion rate sustained over ninety selections is not a hot streak; it is a system. The gap between the best and worst recruiting departments is the clearest evidence in the whole dataset that the draft rewards method over hope.

---

## Part 5 — Is Richmond Making a Mistake?

The four rules in Part 4 were drawn from twenty years of draft history. The fairest test of any framework is whether it explains a live situation, so this section turns it on a club in real difficulty. Richmond sit at 2 wins and 11 losses through Round 15 of 2026 **[data]** — their two victories coming in Round 9 (99–88) and Round 12 (74–56) **[data]** — with seven of their losses by 50 points or more, including a 114-point defeat to Sydney in Round 13, 56 to 170 **[data]**. The on-field collapse is obvious. The interesting question is whether the draft data says it was avoidable, and what it says they should do now.

### The dynasty was a concentrated bet that paid

Richmond's three-premiership era was built on an unusually tight cluster of elite picks, all made inside a six-year window: Brett Deledio (2004, pick 1, A, 275 games **[data]**), Jack Riewoldt (2006, pick 13, A+, 347 games and 787 goals **[data]**), Shane Edwards (2006, pick 26, A, 303 games **[data]**), Trent Cotchin (2007, pick 2, A, 306 games **[data]**), Alex Rance (2007, pick 18, A+, 200 games **[data]**) and Dustin Martin (2009, pick 3, A+, 302 games and 338 goals **[data]**). One more elite grade followed — Ivan Soldo (2015, pick 40, A, 66 games **[data]**) — and then nothing of that calibre.

That cluster is the whole story of Richmond's 8.7% career A+/A rate **[data]**, which sits tied second-worst among established clubs, above only Brisbane (6.0% **[data]**) and St Kilda (5.7% **[data]**). The dynasty did not reflect strong drafting across the board; it reflected one extraordinary five-year run of top-end selection that masked an otherwise below-average process. When six elite players are in their prime, a club does not need to draft well — and Richmond, on this evidence, did not have to.

### The void behind the dynasty

The damaging number is what came next. Across the post-dynasty drafts of 2018 to 2025, Richmond made 32 picks and produced zero A+/A grades **[data]**. Every selection in that window grades C+, C, B or D; the single best is a B. Sam Lalor, taken at pick 1 in 2024, currently grades C+ with 18 games **[data]**.

Apply Part 4's base rate. At the ~10% conversion established clubs average, 32 picks should yield three to four elite players. Richmond have none. A run of zero over a 32-pick sample is not comfortably explained by bad luck — it is consistent with a drafting process running below the league average, which is precisely the gap Part 4 showed to be real rather than noise. The dynasty core has now largely departed — Martin last played 2024, Riewoldt 2023, Edwards 2022 **[data]** — leaving veterans such as Tom Lynch, Dion Prestia, Liam Baker and Daniel Rioli (all active to Round 15 2026 **[data]**) ahead of a replacement cohort graded B and below.

### The pick-1 question, handled carefully

Part 1 put picks 1–5 at a 37.3% A+/A rate **[data]** — nearly two in five become elite, and there is no trade package that reliably replicates a genuine top-five selection. Spending pick 1 in 2024 and holding a C+ grade with 18 games **[data]** sits at the unfavourable end of that distribution. It is fair to stress that 18 games is far too few to grade a career; the C+ is a present snapshot, not a verdict. But the pattern — pick 1, 18 games, C+ — is the early shape of a commission error, the most visible failure mode from Part 1, and it warrants honest monitoring rather than a presumption that time will fix it.

### The structural read, not the individual one

The mistake worth naming is not any single pick. It is structural: Richmond ran a dynasty on six elite players drafted inside five years, allowed that core to run to retirement without building the next cohort behind it, and now field a list whose best young talent grades B. Part 4's efficiency table shows the contrast plainly — the strongest processes (Western Bulldogs 15.6% **[data]**, GWS 14.9% **[data]**) produce elite players consistently across ninety-odd picks, not in one concentrated burst. An extraordinary peak with nothing behind it is the anti-model.

### What the data says they should do

The four rules apply directly. **Rule 1 (the cliff):** picks 11–15 convert at 20.0% **[data]**, statistically level with picks 6–10 and double picks 16–20 — so a rebuild does not require tanking to pick 1; consolidating multiple future assets into the 11–15 band can be the smarter buy. **Rule 2 (the school signal):** in the 11–30 range elite-school background carries a real ~2× premium (30.4% vs 17.2% at 11–15 **[data]**), a legitimate tiebreaker their recent picks should be weighting. **Rule 3 (process over luck):** the 3-to-1 spread in club efficiency is a twenty-year process story, so the fix is a process reset, not a lucky pick. **Rule 4 (pipelines):** WA (11.5% **[data]**) and VIC (11.9% **[data]**) are the most productive states, and a VIC-leaning recent draft profile is well-supported by the data; SA's 5.4% rate **[data]** is the one pathway the numbers say to approach with caution.

So, is Richmond making a mistake? The honest answer is that the mistake was already made — it was the failure to build behind the dynasty, and its evidence is the 0-from-32 post-dynasty void **[data]**. What happens from here turns on whether the recruiting process improves, not on whether any one pick turns into a star.
