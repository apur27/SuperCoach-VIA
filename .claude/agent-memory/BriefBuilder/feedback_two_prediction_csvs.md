---
name: feedback-two-prediction-csvs
description: When two prediction CSVs exist for the same round, use most recent timestamp; user may specify which one
metadata:
  type: feedback
---

For R13/2026 two files existed: `next_round_13_prediction_20260525_1821.csv` and `next_round_13_prediction_20260525_1929.csv` (108 minutes apart, same date).

Rule: Use the most recent timestamp by default. If the user explicitly specifies a file (as in this case — user specified 1929), honour that choice.

**Why:** The INTERACTION MODEL spec says "use the most recent timestamp" and "note the timestamp in the methodology paragraph." User specifying a file overrides this default.

**How to apply:** Always note both files exist in the methodology paragraph even when only one is used. Do not assess MAD comparison between the two unless Scientist requests it.
