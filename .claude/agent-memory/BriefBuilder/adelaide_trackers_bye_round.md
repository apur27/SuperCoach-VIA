---
name: adelaide-trackers-bye-round
description: Workflow for Adelaide tracker selection when Adelaide had a bye before the target round and has no prediction CSV output
metadata:
  type: feedback
---

Adelaide (and Gold Coast, North Melbourne, Port Adelaide) are absent from R14 2026 prediction CSVs. When building a brief for one of these teams, tracker selection falls entirely on form data from `performance_details.csv`.

**Workflow confirmed (R14 2026 Adelaide brief):**
1. Grep `Adelaide,2026` in `*performance_details.csv` to get the list of players with 2026 data.
2. For each candidate player, compute 2026 season disposal mean from raw row disposals.
3. Filter carefully — the grep pattern `Adelaide,2026` also matches "Port Adelaide,2026"; always verify the team column is "Adelaide" not "Port Adelaide".
4. Rank by season mean; top-5 are the trackers.
5. Label all prediction cells `[unavailable — Adelaide had R13 bye; no R14 model output]`.

**Why:** No model output exists for Adelaide in R14 2026. Inventions would be unverified. Form-data selection is the conservative and traceable alternative.

**How to apply:** Any brief where the home or away team is in the uncovered set (Adelaide, PA, Gold Coast, NM as of R14 2026). Check the prediction CSV for the team first; if absent, switch to this workflow. See also [[model-coverage-gap-portadelaide]].
