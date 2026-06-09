---
name: model-coverage-gap-portadelaide
description: Port Adelaide (and Adelaide, Gold Coast, North Melbourne) are absent from R14 2026 prediction CSVs — model covers only 14 of 18 clubs
metadata:
  type: project
---

As of Round 14 2026, the prediction CSV covers only 14 of 18 AFL clubs. Port Adelaide, Adelaide, Gold Coast, and North Melbourne have no players in the prediction output.

**Why:** Confirmed by exhaustive grep of both R14 prediction files (20260601_2256 and 20260601_2331). No "Port Adelaide" team name found in either CSV. The backtest_by_team file also has no Port Adelaide entry.

**How to apply:** When building a brief for any team in the uncovered set (Port Adelaide, Adelaide, Gold Coast, North Melbourne), immediately flag that no model predictions exist. PA tracker players must be selected by form-data only (2026 season disposal mean from performance_details.csv). Escalate to Scientist to confirm whether a separate prediction run covers these clubs or whether the model gap is intentional. Do NOT invent predictions or treat the absence as a data error without checking with Scientist.

Related: [[pa-player-selection-drew-macrae-archetype]]
