---
name: model-training-corpus-scope
description: The disposal predictor trains on ~1,808 of 13,357 player files (earliest season 2005), NOT the full 1897-2026 archive — a birth-year filter, not a season filter. Docs claiming "all years 1965-2026" are wrong.
metadata:
  type: project
---

**The disposal prediction model does NOT train on the repo's full AFL history.**

`SuperCoachPredictor.load_and_prepare_data` (`supercoach/prediction.py`) applies:

```python
birth_year_threshold = self.target_year - 40
...
if pd.isna(dob) or dob.year <= birth_year_threshold:
    continue
```

**Why:** it is a *player-recency* filter — only players young enough to plausibly
be active are loaded. It is expressed as a birth-year cut, so it silently implies
a season-range cut nobody declared.

Measured 2026-07-27 for `target_year=2026` (threshold 1986, keep DOB > 1986):

| quantity | value |
|---|---:|
| player performance files on disk | 13,357 |
| files actually loaded by the predictor | **1,808** |
| earliest season anywhere in the loaded corpus | **2005** |
| median player's first season | 2016 |

**How to apply:** never describe the backtest as trained on "every game" or on
"1897-2026" / "1965-2026". The honest phrasing is *every game played by a player
born after `target_year - 40`, which for 2026 means seasons from 2005 onward*.
`docs/afl-backtest-2026.md` (Methodology → "Walk-forward, no leakage", step 1)
carried the false claim "across all years 1965-2026" — an untagged methodology
statement, and the doc has **no `<!-- council-pipeline:` marker** so DataSentinel
never gated it. Also note the threshold is relative to `target_year`, so the
corpus window slides forward one year every season — any doc that hard-codes a
start year goes stale by construction.

Related: [[backtest-artifact-vintage-selection]], [[backtest-doc-verification]].
