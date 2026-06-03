---
name: self-flagged-unverified-data
description: A self-aware "data-integrity flag" in a brief is a BLOCK trigger, not a hedge — verify the cited file before accepting the author's own caveat
metadata:
  type: feedback
---

When a FootyStrategy brief contains a paragraph that says, in effect, "the data here may be wrong / cannot be substantiated, treat with caution" and then **proceeds to publish numbers and a role read anyway**, treat that as a Sentinel BLOCK trigger, not a satisfied caveat. The author noticing a defect and shipping around it is worse than not noticing — it manufactures a hedge to launder an unverified number into print.

**Why:** On 2026-06-03, the Sydney-vs-Richmond R13 brief flagged Ross Jack's line as a "data-integrity flag for Scientist/DataSentinel … taken on faith from the locked table." On direct check, `ross_jack_03092000_performance_details.csv` was a 104-game player (debut 2019-04-09) with 11 games in 2026 and a 2026 disposal mean of 23.09 — NOT the "2026 debutant, 5 games, 19.4 mean" the brief asserted. The adjacent Retschko table was byte-identical (19.4/5.2/2.8) to Ross's and also didn't match Retschko's own file — a copy-paste fabrication. Two of three spot-checked player tables failed. A self-flag had been placed over exactly the row that was fabricated.

**How to apply:**
- Any in-doc phrase like "taken on faith," "cannot substantiate," "data-integrity flag," "warrants confirmation" on a numeric table → open the cited source file and verify the numbers before reading further. Do not let the author's own hedge substitute for verification.
- Identical stat lines across two different players (same disposals AND marks AND tackles to one decimal) is a copy-paste fabrication signal — players almost never match on three counting stats simultaneously. Diff adjacent player tables.
- Verify the "debutant" claim against the personal_details.csv `debut_date` and the performance file's year span. A file with rows across 2019–2026 is not a 2026 debutant regardless of what the table says.
- The born-date in the filename (e.g. `_03092000_`) identifies WHICH player; multiple players can share a name (there were 5 `ross_jack_*` files). Confirm the cited file's born-date matches the intended player AND that its game log is plausible for the claim.
- Spot-check player tables FIRST when a brief tracks small-sample/debutant players — that is where fabrication concentrates, because there's no strong prior to contradict an invented number.
