# Skeptic Memory Index

- [Harness gap status words](feedback_harness_gap_status_words.md) — "closed" is almost always premature for harness gaps; verify the control is live on disk, prefer "built not yet wired" / "MVP-live not production-grade"
- [NaN handling for counting-stat means](feedback_nan_handling_counting_stats.md) — blank tackle/mark rows on games-played are recording gaps; prefer dropna over fill-zero, demand denominator disclosure
- [Player CSVs not chronological](feedback_player_csv_not_chronological.md) — performance_details rows aren't round-sorted; naive .tail(5) gives wrong last-5; sort_values(['year','round']) before checking
- [Single-round backtest bias propagation](feedback_single_round_backtest_bias.md) — per-team bias figures are often single-round + include the round's own actuals; circular if elevated to a headline call
- [Unfilled FootyStrategy layer tell](feedback_unfilled_footystrategy_layer.md) — "pending fill" footer + no Lens/Convergence/Tensions/Recommendation sections = mid-pipeline; inline lens-labels are not real lens reads
- [Self-flagged unverified data is a BLOCK](feedback_self_flagged_unverified_data.md) — an in-doc "data-integrity flag / taken on faith" caveat means verify the cited file, not accept the hedge; spot-check debutant tables first, diff adjacent player rows for copy-paste fabrication
- [BriefBuilder pre-round means](feedback_briefbuilder_pre_round_means.md) — player CSVs already contain the previewed round; filter round<N+1 (map 'EF'/'QF'→None) AND verify game-count, not just mean
- [Superlative contradictions & jersey collisions](feedback_superlative_and_jersey_collision.md) — ranking-column superlatives contradict the table across sections; duplicate-surname jersey-map collisions clone one player's row onto two clubs; both pass DataSentinel
- [Multi-club completeness check](feedback_multiclub_completeness_check.md) — for "all N clubs" docs, recompute ladder and check table-row count + classification tally; a skipped position is a misleading-completeness BLOCK DataSentinel can't catch
