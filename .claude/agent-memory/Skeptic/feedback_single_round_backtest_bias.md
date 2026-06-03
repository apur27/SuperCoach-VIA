---
name: single-round-backtest-bias-propagation
description: Per-team backtest bias figures are often single-round and include the round's own actuals — circular if elevated to a standing model property in headline calls
metadata:
  type: feedback
---

When a FootyStrategy brief cites a per-team model bias (e.g. "Hawthorn −2.64 disposals/player") as a load-bearing fact in the Executive Summary or headline call, verify the backtest's scope before accepting it as a standing property.

**Why:** In the St Kilda vs Hawthorn R13 2026 brief, the −2.64 Hawthorn bias was elevated to the "match-deciding read" in the exec summary. On disk, `backtest_by_team_20260601_225644.csv` contained ONLY `round=13` rows (n=22 Hawthorn, n=23 St Kilda) — a single-round backtest scored entirely on R13's own actuals. So the bias used to explain why the leaderboard "flatters St Kilda" was computed from the very outcome being explained. The doc named this circularity once, in a Model-context footnote (L261: "post-dates and includes R13 actuals"), but did NOT propagate it up to the Executive Summary or to the standalone Caveat list, which stated −2.64 as bare fact.

**How to apply:**
- Open the cited `backtest_by_team_*.csv` / `backtest_summary_*.csv` and check the `round` column. If it is a single round, the bias is a one-round residual, NOT a season-aggregate model property. Demand `n` and the round window be disclosed inline wherever the figure is used as evidence.
- Self-referential backtest: if the backtest round == the round the brief covers, the bias "explains" its own input. Flag any causal verb built on it ("the model is shaving X off each player") as confidence the evidence does not earn.
- This is a caveat-PROPAGATION concern (Audit 2), not a data mismatch — the number itself verifies correct against the CSV. The defect is that the circularity caveat lives only in a footnote and does not travel to the headline. Verdict tends PASS_WITH_CONCERNS (caveat acknowledged somewhere) rather than BLOCK (caveat buried entirely).
- Related smoothing tell: post-game briefs that narrate each actual as confirming the pre-match read — see [[recurring-tensions]] n=1 confirmation pattern. The bias circularity and the hindsight-confirmation prose reinforce each other into apparent Settled-tier confidence under a Probationary label.
