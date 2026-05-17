# Richmond vs St Kilda - Q1 live read, Round 11, 2026

> [Back to strategy corner](README.md) | [Pre-match brief](richmond-vs-stkilda-round-11-2026.md)

*Snapshot: 2026-05-17 15:21 local (05:21 UTC), Q1 4:36 elapsed. Score: St Kilda 1.1.7 - Richmond 0.0.0. Status: Q1 active. Source: FanFooty live snapshot game 9789.*

## Scoreline and Q1 snapshot

St Kilda lead by **7 points** [data] after the opening 4:36 of play, with the only goal a Max Hall snap from the hotspot at the 2:13 mark [data, commentary]. Richmond have not yet scored. Sample size warning: with under five minutes of play recorded, almost every team metric is in the small-numbers regime where one stoppage chain can flip the read. Treat everything below as a directional Q1-opening pulse, not a settled trend.

The early tempo favours the Saints: 13 disposals to Richmond's 7, six marks to one, and Tom De Koning ahead 2-0 in hit-outs [data]. The pre-match brief had Richmond rated as the contested-possession side; that has not yet shown up on the scoreboard but cannot be assessed from the snapshot (see caveat below).

## Disposal leaders - Richmond (Q1, 4:36)

| Player | Disp | K | HB | Marks | Tackles | TOG% |
|---|---|---|---|---|---|---|
| Seth Campbell | 2 | 2 | 0 | 0 | 0 | 100 |
| Ben Miller | 1 | 1 | 0 | 0 | 0 | 100 |
| Sam Cumming | 1 | 0 | 1 | 0 | 0 | 100 |
| Samuel Grlj | 1 | 0 | 1 | 0 | 0 | 100 |
| Nick Vlastuin | 1 | 1 | 0 | 1 | 0 | 100 |

*[data] - FanFooty live snapshot 9789, 2026-05-17 15:21*

Notable: none of the predicted big-three Richmond ball-winners (Short, Ross, Taranto) have a recorded touch yet [data].

## Disposal leaders - St Kilda (Q1, 4:36)

| Player | Disp | K | HB | Marks | Tackles | TOG% |
|---|---|---|---|---|---|---|
| Max Hall | 2 | 2 | 0 | 0 | 0 | 100 |
| Callum Wilkie | 2 | 2 | 0 | 2 | 0 | 100 |
| Liam Stocker | 1 | 1 | 0 | 1 | 0 | 100 |
| Sam Flanders | 1 | 1 | 0 | 1 | 0 | 100 |
| Hugo Garcia | 1 | 1 | 0 | 1 | 0 | 100 |

*[data] - FanFooty live snapshot 9789, 2026-05-17 15:21*

Wilkie has two marks already at intercept depth - early sign the rebounding role is being shared, not abandoned, in Milera's absence.

## Key metrics - the numbers that matter

| Metric | Richmond | St Kilda | Pre-match edge | Read |
|--------|----------|----------|----------------|------|
| Disposals (total) | 7 | 13 | - | STK +6, controlling early possession |
| Marks | 1 | 6 | - | STK chaining uncontested marks |
| Tackles | 0 | 1 | - | Pressure not yet showing for either side |
| Hit-outs | 0 | 2 | - | De Koning unopposed early (RIC ruck not yet on the sheet) |
| Clangers | 2 | 5 | - | STK higher but unreliable column (see caveats) |
| Inside 50s | n/a | n/a | STK (Milera absence narrows this) | **Not in snapshot schema** - cannot verify tripwire from data |
| Contested possessions | n/a | n/a | RIC (Taranto/Hopper/Ross) | **Not in snapshot schema** - awaiting full-match data |

*[data] - FanFooty live snapshot 9789, 2026-05-17 15:21*

**Caveat:** the live snapshot schema exposes kicks, handballs, marks, tackles, hit-outs, frees, clangers, DE%, TOG% per player - but **not inside-50s or contested possessions**. The tripwire requires a proxy or a different source until those columns appear post-match.

**Reliability flags from prior verification:** goals, behinds and clangers are known to be misindexed in this snapshot pipeline (see memory: `snapshot_data_quality.md`). The 1.1.7 scoreline comes from the header string (parsed separately and reliable), not the per-player goals column.

## Pre-match prediction check

Sample size is tiny (4:36 elapsed = roughly 6% of regulation time), so "behind pace" should not be over-read. A normal midfielder centred at the bounce only sees the ball every 60-90 seconds.

| Player | Team | Predicted (full) | Q1 actual | TOG% | Pace read |
|---|---|---|---|---|---|
| Jayden Short | RIC | 23 | 0 disp | 80% | Off-pace, sub-100% TOG suggests rotation - watch for first kick out of D50 |
| Jack Ross | RIC | 21 | 0 disp | 100% | On-ground but no touch - normal for first 5 min |
| Tim Taranto | RIC | 19 | 0 disp | 100% | On-ground but no touch - normal for first 5 min |
| Jack Sinclair | STK | 27 | 0 disp | 100% | On-ground but no touch - watch closely, he is the engine |
| Bradley Hill | STK | 22 | 0 disp | 100% | On-ground, no touch yet - Milera-rebound theory not yet testable |
| Callum Wilkie | STK | 24 | 2 disp, 2 marks | 100% | **On pace** - intercept marking already showing |

*[data] - FanFooty live snapshot 9789, 2026-05-17 15:21*

Wilkie is the only one of the six tracking on or above pace this early; everyone else is at zero, which at 4:36 elapsed is statistically unremarkable. Re-check at quarter time.

## Tripwire status

**Inside-50 count at this point:** **NOT AVAILABLE** - the live snapshot does not expose I50. [data, schema check]

**Tripwire trigger (STK leading I50 = flip call):** **CANNOT BE EVALUATED** from snapshot data alone.

**Available proxies for the tripwire spirit:**
- St Kilda lead disposals 13-7 and marks 6-1 [data] - consistent with STK having more time in their forward half, but disposal count alone does not distinguish defensive chains from forward entries.
- Only score is a Max Hall snap "from the hotspot" [commentary] - one Saints forward-half entry confirmed.
- Hit-out count 2-0 to STK [data] - De Koning winning the centre tap which slightly favours STK forward chains from centre bounces, but two hit-outs is one centre bounce and one ball-up - not enough to call.

**Verdict:** **HOLDING** on the available evidence. The pre-match flip call (St Kilda forward-half dominance triggers the side flip) requires inside-50 data the snapshot does not provide. Re-evaluate at quarter time when the score and event commentary give a clearer picture of which end the ball is spending time at.

---

## Methodology notes

- **Snapshot path:** `/home/abhi/git/SuperCoach-VIA/data/live_snapshots/9789_20260517_1521_q1-4-36.json`
- **Schema sentry:** passed (65 columns parsed as expected) [data]
- **Reliability:** kicks/handballs/marks/tackles/hit-outs/DE%/TOG% verified reliable; goals/behinds/clangers known to be misindexed - the header score string is used for the actual scoreline.
- **Sample size:** 4:36 elapsed = ~23% of Q1, ~6% of regulation. All disposal totals are heavily noise-dominated at this stage. The 13-7 disposal edge for St Kilda translates to a ~95% bootstrap CI of roughly +/- 4 disposals if extrapolated linearly, but linear extrapolation is itself unreliable from such a small window.
- **What I did not verify:** the predicted full-game totals (23/21/19/27/22/24) were taken from the pre-match brief as given, not re-derived from the disposal predictor. Pace-vs-prediction is a directional check only.
- **Residual risk:** the tripwire metric (I50) is unavailable from this data source. If the tripwire is load-bearing for the in-game flip call, a second data source (AFL.com.au gamecentre, Champion Data feed) is needed.
