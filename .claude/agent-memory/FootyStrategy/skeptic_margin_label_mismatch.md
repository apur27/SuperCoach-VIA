---
name: skeptic-margin-label-mismatch
description: "Average winning margin" recurs as a mislabeled claim in the weekly recap because the source table's own section heading ("#### Winning margin" in afl-stat-leaders-2026.md) names a different quantity than the column it displays (all-games average margin, losses included, league mean ~0 by construction) — verbatim BLOCK repeat, Round 24 then Round 25.
metadata:
  type: feedback
---

**The trap:** `docs/afl-stat-leaders-2026.md`'s team-level section has a heading
literally titled `#### Winning margin` directly above a table whose "Avg margin"
column is actually `team score − opponent score` averaged over **every** team-game
that season (wins AND losses) — the doc's own prose two lines above the table says
so explicitly ("margin is the team's score minus the opponent's") and states the
league-wide mean is ~0 by construction. Carrying the section heading into recap
prose as if it described the number ("Fremantle's average winning margin of
+29.9...") produces a false claim: under the real wins-only definition, the ranking
inverts (checked against `data/matches/matches_2026.csv`: Fremantle wins-only
+36.42/19 wins vs Sydney wins-only +44.82/17 wins — Sydney leads by ~8.4, not
Fremantle).

**Why it recurred (Round 24 → Round 25, same defect verbatim):** the source
document's section heading is the attractive nuisance — it reads as a ready-made,
accurate-sounding label sitting right next to the number, so pulling the heading
into prose feels like citing the source correctly rather than mislabeling it. A
one-off fix to the Round 24 recap text did not change the underlying trap, because
the trap lives in the *source table's heading*, not in anything FootyStrategy wrote
that could self-correct by pattern-matching prior recap text.

**Fix pattern applied both times:** relabel to match the actual cited quantity —
"average margin across all games" (or "margin, wins and losses combined") — keep
the existing verified number and its `**[data]**` tag unchanged. Do not switch to a
wins-only figure unless recomputing it fresh from `data/matches/matches_2026.csv`
with new tagged numbers, which is disproportionate to a labeling fix.

**Standing-guidance recommendation (for Gaffer/retro):** the durable fix is not a
FootyStrategy checklist item (that was tried after Round 24 and did not prevent
Round 25) but one of: (a) rename the `afl-stat-leaders-2026.md` section heading
from "Winning margin" to "Average margin (all games)" at the source so there is no
attractive-but-wrong label to copy, or (b) add this specific phrase
("winning margin") to a pre-submit grep check in the weekly recap step, since the
heading-vs-column mismatch is deterministic and catchable without an LLM pass.
Recommend routing to Gaffer / DataSentinel as a source-doc fix, not relying on
FootyStrategy remembering to relabel every cycle.
