---
name: dob-source-equivalence
description: Player DOB is available two ways and they are equal — filename DOB token == personal_details.born_date (verified 40/40 sample)
metadata:
  type: reference
---

Player date-of-birth is available from two sources in `data/player_data/`, and
they are the same value:

1. The **filename token**: `surname_first_DDMMYYYY_performance_details.csv`
   (already parsed by `extract_dob_and_name` in `supercoach/prediction.py`).
2. The **`born_date`** field in `..._personal_details.csv` (format `DD-MM-YYYY`).

Verified equal on a random 40-file sample (seed=42): 0 mismatches, 0 missing.

**How to apply:** For age/experience feature work, use the filename DOB token as
the join channel — it is already parsed in the prediction pipeline and avoids a
second file read per player. No need to open `personal_details.csv` just for DOB.
Used this way in the S7 age feature (`scripts/feature_engineering.py`,
`compute_age_years`). Related: [[player_csv_date_format]] — note the performance
`date` column is off by ~1 month, negligible for age-in-years (±0.1 yr).
