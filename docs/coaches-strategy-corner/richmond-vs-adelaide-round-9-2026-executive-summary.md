# Richmond vs Adelaide - Executive Summary

> [← Coaches Strategy Corner](README.md) | [← AFL insights](../afl-insights.md)

## Match: 10 May 2026, 3:15 PM, M.C.G.

> **Pre-match brief.** Form data through Round 9 2026. Tactical recommendations are forward-looking. Match has not yet been played.

---

## The 3 things that decide this game

1. **Hit-outs at the centre bounce.** Adelaide are 18/18 in the league at 8.8 hit-outs/game - 24 below the league average **[data]**. With Samson Ryan (16.3 HO/g across his 3 games) Richmond would win this contest by 15+ - the largest structural mismatch of the round.
2. **Q1 sets the ceiling.** Richmond have lost the first quarter by an average of 12.2 points across the 8 games to date **[data]**. Adelaide concede first quarters at 25.1/g; Richmond score them at 13.2/g. If the bounce-back doesn't happen by the 6-minute mark, the chase is on.
3. **Sam Berry is Adelaide's pressure system.** He leads Adelaide on **both** tackles (8.0/g) and contested possessions (13.4/g) **[data]** - the only player on the list who does both. Take Berry out of the contest and Adelaide's pressure rank drops from 4/18 to mid-pack.

---

## Richmond's edge

The contested-mark contest (6/18, 9.4/g) **[data]** lets Richmond's tall forwards outwork Adelaide's mid-pack defence (13/18 contested marks). The R9 win at Optus showed the ceiling - 141 contested possessions, season-best, with man-on-man at the contest. With Ryan in the ruck, Richmond owns centre bounces. Jack Ross (5.1 tk/g, 11.9 CP/g, +0.83 trend) is the senior body to match Berry phase-for-phase. Tom Lynch and Liam Fawcett, when fit, give Richmond two contested-mark targets inside 50 - the right play given Adelaide's rebound-50 strength (2/18, 41.8/g) punishes long bombs.

## Adelaide's edge

Tackles 4/18 (59.5/g), Rebound-50s 2/18 (41.8/g), Contested possessions 7/18 (130.1/g) **[data]**. They tackle harder than Richmond and rebound interstate kicks better than 16 other teams. Their captain Jordan Dawson is the most consistent ball-winner in either squad (CV 0.11, range 21–28 disposals every week, 6 goals from 5 games **[data]**). Up forward, four players - Thilthorpe, Walker, Rachele, Keays - average over 1.25 goals per game; the spread of finishers means a Richmond defence that locks down only one tall is one short.

## The key matchup

**Samson Ryan vs the Thilthorpe/Maley combo.** This is not a tag, it is a structural exploit. Adelaide are running a forward-doubling-as-ruck rotation: Thilthorpe (1.2 HO/g across 8 games) and Maley (5.6 HO/g across 5) **[data]**. Ryan, on his 3-game 2026 return, averages 16.3 hit-outs/g. If selected and fit, the centre-bounce battle and forward-50 throw-in chains are owned for the day - a +15 hit-out swing, worth 5–8 stoppage clearances against a contested-ball side that doesn't have a recognised ruckman to reset.

---

## Win condition (Richmond)

- **First quarter within 5 points.** Average Q1 deficit is –12.2 - survive that and the Q4 burst window opens.
- **Hit-outs +15 or better.** Pick Ryan; load Ross, Taranto, Prestia at the centre bounce.
- **Clangers under 55.** Richmond average 59.2 (4/18 worst) **[data]**. Every defensive turnover at Adelaide's 0.253 goals/i50 conversion is worth 1.5 points the wrong way.

## Win condition (Adelaide)

- **Tackle count over 60.** Adelaide's pressure rank is 4/18 - Richmond's tackle floor is 51.8 (16/18). A 10+ tackle gap forces panic disposals and rebound goals.
- **Berry over 5 tackles by half-time.** When Berry's individual pressure is on, the team's rank holds.
- **Concede fewer than 8 inside-50 marks.** Richmond's contested-mark rate (6/18) is the lone Richmond aerial weapon - neutralise it and Richmond cannot score from territory.

---

## Charts at a glance

![Adelaide MCG win rate by 5-year era](../../assets/charts/strategy/adelaide_mcg_form.png)
*Adelaide's MCG win rate has been below 50% in every 5-year window since their entry - and is at a 35-year low (22%) in 2021–25.*

![Richmond vs Adelaide H2H by era](../../assets/charts/strategy/richmond_vs_adelaide_h2h.png)
*The era splits: Adelaide's lead is built almost entirely in 1997–2010 (R5–A14). The 2010s and 2020s have been close.*

![Richmond vs Adelaide team stats 2026](../../assets/charts/strategy/team_stat_comparison_2026.png)
*The shape difference: Adelaide owns tackles and rebounds; Richmond owns hit-outs (Ryan-conditional) and is competitive on contested marks; both are bottom-pack on volume.*

![Richmond quarterly differential 2026](../../assets/charts/strategy/richmond_quarterly_differential_2026.png)
*Every quarter is a deficit. Q1 (–12.2) and Q4 (–14.1) are the worst. The first 6 minutes set the ceiling.*

![Top 5 disposal-getters Richmond vs Adelaide](../../assets/charts/strategy/key_player_disposal_comparison.png)
*Half-back distributors top each side: Short (R) and Milera (A) sit within 0.4 disposals of each other. Then Adelaide's lead deepens at the contest (Ross/Taranto vs Dawson/Worrell).*

![Last 10 Richmond vs Adelaide meetings](../../assets/charts/strategy/h2h_recent_results.png)
*Richmond won 6 of the last 10 - including the 2017 Grand Final by 48. The 2025 R17 (–68) is the freshest film both staffs will watch.*

---

## Full brief

- [Tactical brief](richmond-vs-adelaide-round-9-2026.md) - stat profiles, blueprint, blueprint-by-blueprint plans
- [Player matchups](richmond-vs-adelaide-round-9-2026-player-matchups.md) - the five priority contests, full Adelaide threat list, Richmond's match-winners
- [Head-to-head history](richmond-vs-adelaide-round-9-2026-head-to-head-history.md) - 35-year ledger, MCG record, recent travel pattern

## Methodology in one line

Form numbers aggregated from `data/player_data/*_performance_details.csv` (per-player game files, summed to team-game then averaged across 8 completed 2026 games). Match results from `data/matches/matches_*.csv`. Re-running `docs/coaches-strategy-corner/generate_strategy_charts.py` reproduces the charts deterministically.
