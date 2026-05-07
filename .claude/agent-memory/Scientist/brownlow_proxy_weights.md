---
name: Brownlow proxy weight validation
description: Spec weights for Brownlow vote proxy were rebalanced after EDA - goals lifted from 5% to 15% based on validation against 2010-2025 historical brownlow_votes
type: feedback
---

When building the 2026 Brownlow Medal predictor, the spec proposed weights of {disposals 35%, clearances 25%, contested-poss 20%, eff-disp 15%, goals 5%}.

EDA on 145,150 player-games from 2010-2025 (where actual `brownlow_votes` are present in `data/player_data/*_performance_details.csv`) showed:

- Per-game Pearson correlations with brownlow_votes: disposals +0.36, eff_disp +0.35, contested-poss +0.33, clearances +0.30, goals +0.25, tackles +0.14.
- OLS-fit weights on z-scored predictors gave goals ~36% normalized share - far higher than 5%.
- Practical compromise: lifting goals from 5% to 15% (Scheme B) raised pearson-vs-votes from 0.40 to 0.42 and top-1% vote-rate from 67% to 70%, without losing midfielder-first character (disp+clr+cp still total 70%).

**Why:** the spec text says "adjust after EDA if the data suggests otherwise" and the data clearly suggests otherwise. Honest reporting > impressive deference to spec.

**How to apply:** when building any AFL "X correlates with votes/wins/etc" composite, validate weights against the historical signal in `brownlow_votes` column before defaulting to spec weights. The 5% goal weighting is a common pure-midfielder convention but underweights goal-kicking forwards/wingmen.

**Effective disposals proxy:** the raw scrape does not have a true `effective_disposals` column. We use `disposals - clangers` (clipped at 0) as a proxy. Documented in update_team_analysis.py docstring.
