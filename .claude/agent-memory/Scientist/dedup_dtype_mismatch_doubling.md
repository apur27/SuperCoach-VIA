---
name: dedup-dtype-mismatch-doubling
description: R20 2026 +1 doubling — dedup_player_performance keyed on raw dtypes, so read_csv int year != scraped str year kept both copies of a re-emitted game
metadata:
  type: project
---

**The R20 2026 phantom-gate abort (833 files, missing=[] duplicated=[N], clean +1).**
Two defects compounded:

1. **Trigger (why a row re-emits):** the delta scraper skips already-seen games
   with `if since_date and game_date <= since_date and not is_new_counter: continue`.
   When this cycle's match refresh CORRECTS a fixture date forward (e.g. Steele
   Sidebottom R19 2026: stored 2026-07-05 -> re-resolved 2026-07-10; adams_taylor
   R14 2025: 2025-05-31 -> 2025-06-07), the last-recorded game's re-resolved
   `game_date` is now > `since_date` (= max stored date), so the already-on-file
   game is re-emitted as if new. The counter guard doesn't save it: its counter
   == max_counter, not > max_counter. So exactly ONE row (the drifted last game)
   re-emits per file.

2. **Root defect (why dedup didn't collapse it):** `dedup_player_performance`
   keyed `drop_duplicates` on the RAW columns. `existing_df` comes from
   `pd.read_csv` (all-numeric `year` -> int64 `2025`; a NaN-bearing numeric col
   floats to `2025.0`), while the fresh scrape row is all strings (`"2025"`).
   After `pd.concat`, `drop_duplicates` saw int `2025` != str `"2025"` (and
   `14` != `"14"`) and kept BOTH copies. The dedup — the safety net designed to
   collapse re-scrapes — was dtype-blind.

**Fix (commit-pending):** `dedup_player_performance` now normalises every key
column (team/year/round/opponent + games_played) to a stripped string with a
trailing-`.0` strip (`^(-?\d+)\.0$` -> `\1`) before keying. Dtype-agnostic.
Preserves drawn-GF pairs (distinct games_played 35 vs 36 stay distinct). Left the
delta re-emit logic ALONE (surgical): with dedup fixed the re-emit is harmless —
keep='last' keeps the corrected-date row, self-correcting the fixture date. Tests:
`test_mixed_dtype_key_columns_still_dedup`, `test_float_rendered_integer_key_collapses`
in tests/unit/test_player_scraper_dedup.py.

**Remediation gotcha — a plain harness re-run does NOT self-heal doubled files.**
After doubling, the file's max date == the drifted date, so next run `game_date
<= since_date` is True -> no re-emit -> `_write_player_details` not called -> the
double persists and the gate fails AGAIN. Must run an explicit repair pass:
read each doubled CSV -> `dedup_player_performance(df)` -> write. The FIXED dedup
collapses even same-dtype exact dups (both existing rows normalise equal), so a
read->dedup->write over the 833 flagged files repairs them deterministically
(verified: all 833 -> counter-gap ok=True, zero had actual missing rows).

**32 SINGLE_FINAL_ROW reviews are the benign layer-(b) class** (drawn-final-year
finalist with one GF row, e.g. davis_leon 2010 GF result 'D'), orthogonal to the
counter-gap (layer a) doubling. Advisory, not the hard-abort. See
[[phantom_row_validator_gate]], [[delta_scraper_approx_date_drop]].
