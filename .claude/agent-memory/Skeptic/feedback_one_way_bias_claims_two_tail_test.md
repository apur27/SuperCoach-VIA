---
name: one-way-bias-claims-two-tail-test
description: Recap claims that model projections are biased in ONE direction are usually shrinkage described from one tail only — run the two-tail bucket sign test before accepting or flagging, and demand the sign-flip be stated in the prose
metadata:
  type: feedback
---

When a weekly recap explains a projection-vs-season-average gap ("that figure sits well below his season average … these projections run below season averages, most visibly for the highest-volume players"), the claim is almost always **shrinkage toward the mean narrated from the high tail only**. Stated one-way it is false for roughly half the projection set, and the sign flips by volume band.

**The decisive test (cheap, run it every time).** Join the round's `data/prediction/next_round_N_prediction_*.csv` to each named player's `data/player_data/*_performance_details.csv` season mean, bucket by season average, and report BOTH the bucket mean delta and the within-bucket percent above/below. Percentages are what settle it — bucket means alone cannot distinguish "a tendency" from "universal at the tails".

R25 result (n=396 matched, projections vs rounds 1–24 means):

| season avg | n | mean delta | % above |
|---|---|---|---|
| <10 | 46 | +1.51 | 98% |
| 10–15 | 149 | +0.21 | 54% |
| 15–20 | 101 | −0.32 | 40% |
| 20–25 | 73 | −0.95 | 27% |
| 25–30 | 22 | −1.80 | 0% |
| 30+ | 5 | −3.60 | 0% |

Pooled: 51.2% below, mean −0.14. So the pooled figure looks like "no bias", the high tail looks like "runs below", and only the banded two-tail view shows the actual mechanism. **Flag any one-way version; accept only prose that states both directions.**

**Accepted fix shape (R25 hash f3f200bd, PASS):** "a reminder that these projections regress toward the league mean, running below season averages for the highest-volume players and above them for low-volume ones." Note it is unqualified ("running below", not "on average below") and that is fine *here* because both tails are ~100% one-signed — check the percentages before allowing an unqualified group statement.

**Two companions on the same paragraph.**
- **Named-individual tie completeness.** "with X next at 28.0" invites the check: is X uniquely next? R25 had a three-way tie (Daicos/Neale/Oliver) that DataSentinel passes, since each tagged number is individually true. After the fix, re-verify the tie set is *complete* and that nothing sits between it and the value above — read the ranked source table (`docs/afl-predictions-2026.md`, `docs/weekly/round-current-2026.md`) rather than only the CSV.
- **The ±4 comparison is legitimate, do not flag it.** Comparing a projection-vs-season-average gap to the cheat sheet's prediction-vs-actual "typical error of ±4 disposals" looks like a category mismatch, but the rhetorical move is "the gap exceeds noise, therefore it is systematic shrinkage" — which the bucket table supports. `docs/weekly/round-current-2026.md` is the source of the ±4 wording.

Related: [[recap-tactical-note-causal-relapse]].
