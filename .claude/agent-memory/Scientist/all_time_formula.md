---
name: All-time ranking formula constraints
description: Methodology trade-offs in the rank-based all-time top-100 formula and what the data actually supports vs expert consensus
type: project
---

The all-time top-100 ranking in `top_players_comprehensive.py` uses a rank-based formula:

    year_score = ((101 - rank) / 100) ** RANK_GAMMA
    adj        = year_score * ERA_COMPLETENESS[era]
    mean_adj   = mean of top-N (=10) adj values
    score      = mean_adj * (1 + 0.60 * min(seasons / 18, 1))

Current constants (2026-04-27 calibration for "Bartlett #1, Pendlebury top-20, ~3 per decade"):
- `RANK_GAMMA = 0.20` (concave: rank #1=1.00, #4=0.99, #10=0.97)
- `TOP_N_SEASONS = 10`
- `ERA_COMPLETENESS = {pre_1965: 0.88, 1965_1990: 0.92, 1990_2010: 0.92, post_2010: 0.92}`
- `career_bonus = 0.60 * min(seasons / 18, 1)` — SEASONS-based, not games-based

**Why:** The game-based career bonus (`min(games/400)`) systematically rewarded modern players because pre-1990 seasons were 16-18 games vs modern 22-24 games. Switching to seasons-based eliminates this era bias. The 1990-2010 ec=0.95 also injected a structural advantage that compounded the games bonus, driving the 2000s decade to ~25 players in the top 100.

**How to apply:** When tuning this formula, know what the data actually supports vs what expert consensus claims:

- **Rank-1 frequency rules under any rank-dominance formula.** Matthews has 6 outright #1 finishes; Bartlett has 2. Carey has 2; Bartlett has 2 with much more longevity. Under any monotonic year_score curve, peak players win on rank-1 frequency. Bartlett wins #1 only because (a) seasons-based career bonus equalises era and (b) 18-season cap on bonus saturates only for him (19 seasons), so he gets max bonus while Matthews (16) and Pendlebury (15) get 0.533/0.500. Don't try to force someone else #1 by ec tweaks alone — it requires changing TOP_N or seasons cap.

- **Pendlebury's top-20 is fragile and γ-dependent.** Pendlebury's peak rank is #4, no #1 finishes; tail seasons are rank 29-81. Under γ=0.70 (convex) he ranked ~#66; under γ=0.20 (concave) he reaches #15-20 because rank #4 ≈ rank #1 in score. Lowering γ trades off rank-#1 dominance against consistency. The user's Pendlebury-top-20 constraint is a strong signal that consistency should be valued over peak.

- **TOP_N_SEASONS is the lever for "peak vs longevity."** TOP_N=8 favors peakers (Matthews, Carey); TOP_N=15 favors long-careerers (Bartlett, but kills consistency-mid players like Pendlebury whose tail seasons are weak). TOP_N=10 is the calibrated middle.

- **Seasons-based career bonus must have its cap calibrated to era-fair seasons-in-top-100 reach.** Cap=12 is too low (everyone with 12+ saturates, era_completeness gap dominates). Cap=18 lets Bartlett's 19 saturate but Matthews's 16 doesn't quite — the right separation. Cap >18 unfairly penalises early-era players who topped out at 15-16.

- **ERA_COMPLETENESS spread should be tight under rank-based aggregation.** Spread of 0.85/0.92/0.95/0.85 with games-based bonus crushed pre-1965 + 1965-1990 players. Compressed 0.88/0.92/0.92/0.92 with seasons-based bonus restored era diversity. Don't widen the 1990-2010 ec back to 0.95 without re-checking that the 2000s decade doesn't balloon back.

- **Position stratification was deleted.** It's not part of the rank-based formula — rank already normalises for position because each era's full-cohort raw-score ordering naturally produces a mix of forwards and midfielders in the top of each year. The `_career_position_group` helper is retained in the file but unused.

- **Decade balance under current calibration:** 1900s=2, 1910s=6, 1920s=4, 1930s=7, 1940s=6, 1950s=3, 1960s=10, 1970s=10, 1980s=11, 1990s=8, 2000s=15, 2010s=13, 2020s=5. The 2000s remains the highest because that era genuinely had many high-#1-frequency forwards (Lloyd, Pavlich, M Richardson, Riewoldt, etc.) — this is honest data, not formula bias.

- **DO NOT add quota/enforcement rules.** The user explicitly forbids them. The merit-based top-100 must satisfy the constraints from formula tuning alone.
