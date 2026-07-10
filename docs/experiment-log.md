# Experiment Log

Chronological record of model/feature changes to the disposal predictor
(`supercoach/prediction.py`) and related pipeline. Each entry states what
changed, why, how it was verified, and what was deliberately *not* verified.

---

## 2026-07-09 — S1b: Resolve phantom model features

**Author:** Scientist agent
**Files:** `supercoach/prediction.py`, `tests/unit/test_prediction_features.py`
**Blast radius:** MEDIUM (changes the feature set the production predictor trains on)

### Problem
Three declared features had no matching CSV column, so they were *silently
dropped* during feature engineering
(`extra_feats = [f for f in self.extra_features if f in df.columns]`). The
model was training on fewer features than the code implied, with no error.

### Investigation
Sampled 200 of 13,350 `*_performance_details.csv` files (seed=42). The raw
schema is uniform across all sampled files. Presence of each declared feature:

| Declared feature            | Present in raw CSV | Verdict                                   |
|-----------------------------|--------------------|-------------------------------------------|
| `percentage_time_played`    | 0 / 200            | Wrong name — real column is `percentage_of_game_played` (TOG %), present 200/200 |
| `cba_percent`               | 0 / 200            | No source column, no equivalent anywhere (searched cba/centre/bounce → only `bounces`, which is a different stat) |
| `venue`                     | 0 / 200            | No source column, no equivalent (searched venue/ground → none) |

### Changes
1. **`percentage_time_played` — WIRED.** Added
   `'percentage_of_game_played': 'percentage_time_played'` to `RENAMES`.
   The TOG% column now resolves to the declared feature name instead of being
   dropped. This *adds a real feature* the model previously ignored.
2. **`cba_percent` — REMOVED.** Dropped from `extra_features` and from
   `DTYPES`. No centre-bounce-attendance data exists in the repo, so the
   declaration was inert (always dropped). Removal is behaviour-neutral for
   training but removes a misleading phantom.
3. **`venue` — LEFT AS-IS (documented).** `venue` is not a model feature
   (not in `extra_features`/`base_rolling_features`); it is a categorical
   dummy *source* that is already defensively guarded everywhere it is used
   (`[col for col in ['venue', 'opponent'] if col in df.columns]`). Since it
   is never present, only `opponent` dummies are created — no silent
   feature-drop and no error. The `'venue': 'category'` DTYPES entry and the
   guarded `get_dummies` calls were left untouched to stay surgical; ripping
   them out is cosmetic and would touch 4 call sites for no behaviour change.
   Documented here so a future reader knows `venue` is a deliberate no-op, not
   an oversight. If venue data is ever added to the scraper, the existing
   guards will pick it up automatically.

### Verification
- TDD: `tests/unit/test_prediction_features.py` written first, confirmed RED
  (3 failing on the phantoms), then GREEN after the fix.
- Full suite: **288 passed** (`pytest tests/ -v`), no regressions.
- Reproducibility: sampling used `random.seed(42)`; libs pandas 2.2.3,
  sklearn 1.6.1, lightgbm 4.6.0, numpy 1.26.4.

### NOT verified (deferred)
- **Model-accuracy impact of adding TOG% was NOT measured.** Per task scope,
  no backtest was run (backtest is ~5-6h CPU on this host; see memory
  [[feedback_backtest_rules]] and [[prediction_lgbm_cpu]]). Adding
  `percentage_time_played` genuinely changes the training feature matrix, so
  the *next* incremental backtest will be the first run to reflect it. Whether
  TOG% improves, harms, or is neutral to disposal-prediction accuracy is an
  open question to be answered by that backtest — do NOT assume it helps.
- **Coverage/era caveat for TOG%:** `percentage_of_game_played` may be
  sparse/NaN in older seasons. Its float32 dtype + the model's imputer handle
  NaN, but if the feature is mostly-NaN in the training window it will carry
  little signal. Not audited here.

**Residual risk:** the added feature could dilute or distort predictions until
validated by a backtest; the removal of `cba_percent`/documenting `venue` is
behaviour-neutral and low-risk.

### 2026-07-10 — S1b follow-up fix: TOG% was leaking (raw → shift(1) lag)

**Defect found.** The S1b wiring (above) put `percentage_time_played` into
`extra_features`, so it entered `feature_columns` **unlagged** — the raw
*same-game* TOG% of game *i* was a feature used to predict disposals of game
*i*. Every other model feature is `.shift(1)` (strictly prior information).
Two consequences:
- **Target leakage in training.** Same-game TOG% correlates with same-game
  disposals (a player subbed off early has both low). The model learned a
  contemporaneous relationship that does not exist at serve time.
- **Train/serve skew in production.** At predict time the pipeline supplies the
  *last played* game's stats on the predict row, so the served TOG% is a prior
  game's value while the model was trained on the same-game value — the
  coefficient is miscalibrated. The prior backtest would also have been falsely
  optimistic (it retains the cutoff-round raw row, consuming the actual TOG% of
  the round being predicted).

**Fix applied (2026-07-10).** TOG% is now a strictly-lagged conditioning signal,
same temporal treatment as `base_rolling_features`:
- Removed `percentage_time_played` from `extra_features` (now `[]`).
- `_engineer_features` and `_engineer_features_for_prediction` compute
  `percentage_time_played_lag1 = groupby('player')['percentage_time_played'].shift(1)`,
  and `percentage_time_played_lag1` (not the raw column) enters `feature_columns`.
- `RENAMES` (`percentage_of_game_played → percentage_time_played`) and the
  `float32` DTYPES entry are unchanged — the raw column still exists as the lag
  source, it just never reaches the model unlagged.
- Prior-round TOG% remains a legitimate signal (a player who played 60% last
  week may be injured/managed); it is now prior information only.

**Verification.** TDD: `test_no_unlagged_raw_feature_in_feature_columns` +
`test_tog_lag_uses_prior_game_only` added to
`tests/unit/test_prediction_features.py`, confirmed RED before the fix, GREEN
after. Full suite **317 passed** (`pytest tests/`), no regressions. Guard test
also pins `percentage_time_played` and `cba_percent` OUT of `feature_columns`
to prevent recurrence.

**Git SHA:** `b5cc4db17`

---

## 2026-07-09 — Lineup scraper fix (garbage names since 2025 R3) [Task S3]

### What broke
`MatchScraper._extract_player_names` (`scrapers/game_scraper.py`) read
`cells[0]` of each row in the afltables match-stats table. On that page the
columns are `#`, `Player`, KI, MK, HB, … — **column 0 is the jersey number**,
and the player name lives in the **`Player` column (index 1)** formatted
`"Surname, Firstname"`. The parser also dropped the historical
`"Surname, Firstname"` → `"Firstname Surname"` reversal. Result: the `players`
field filled with jersey numbers, sub markers (`↑`/`↓`) and footer labels
(`Rushed`, `Totals`, `Opposition`).

This is a **code regression, not a site change** — a 2024 page and a
2025-R2-era page were fetched and both show the identical `#`/`Player`
structure. The clean historical CSV data (through 2025 R2) was produced by
older correct code; the broken code's output started landing at 2025 R3 and
poisoned every round through the current 2026 season.

Corruption extent (garbage rows detected): **~700 rows** across the 24
`data/lineups/team_lineups_*.csv` files — 2025 R3→finals and all of 2026.
(2025 R1 partial, R2 clean, R3+ all garbage.)

### Fix (committed as code + tests)
- `_extract_player_names` now locates the `Player` column from the header
  (`_find_player_column`, falls back to index 1), reverses the name
  (`_normalise_player_name`), and skips header/footer rows.
- Added `LineupParseError`: if a populated stats table yields zero valid names,
  the parser **raises** instead of writing garbage — so a future afltables
  structure change fails loudly.
- Tests: `tests/unit/test_lineup_scraper.py` (4 tests, no network).
- Verified end-to-end on the real page `stats/games/2025/091920250329.html`
  (previously garbage) → now 23 clean names in `Firstname Surname` form.

### Backfill (HUMAN must run — requires network; NOT run here)
The corrupt rows are already on disk. The scraper dedups on
`(year, date, round_num, team_name)`, so a plain re-run **skips** the corrupt
rows and will NOT overwrite them. You must (1) delete the garbage rows so their
keys are absent, then (2) re-scrape 2025 + 2026 with the fixed parser.

Run from repo root with the venv Python
(`/home/abhi/sourceCode/python/coding/.venv/bin/python`):

```python
import glob, re, pandas as pd
import config
from scrapers.game_scraper import MatchScraper

# --- 1. drop garbage rows from every team lineup CSV ---
def is_garbage(s):
    toks = str(s).split(';')
    bad = sum(1 for t in toks
              if re.fullmatch(r'\s*\d+\s*(↑|↓)?\s*', t)
              or t.strip() in ('Rushed', 'Totals', 'Opposition', ''))
    return bad > len(toks) / 2

for f in glob.glob(f"{config.LINEUPS_DIR}/team_lineups_*.csv"):
    df = pd.read_csv(f)
    keep = df[~df['players'].map(is_garbage)]
    if len(keep) != len(df):
        keep.to_csv(f, index=False)
        print(f"{f}: dropped {len(df) - len(keep)} garbage rows")

# --- 2. re-scrape the affected seasons (fixed parser fills the now-missing keys) ---
s = MatchScraper()
s._load_existing_lineup_keys(config.LINEUPS_DIR)   # AFTER the deletion above
for year in (2025, 2026):
    s._process_year(year, config.MATCHES_DIR)      # re-fetches game pages, accrues lineups
s._process_team_lineups(config.LINEUPS_DIR)        # writes only the missing (clean) rows
```

Note: a normal `main.py` / `refresh_data.py` delta run only re-scrapes from the
last-processed year (2026), so it will NOT repair 2025 on its own — the loop
above forces both years. After running, spot-check a repaired file, e.g.
`data/lineups/team_lineups_adelaide.csv` 2025 R3, should read player names not
numbers.

---

## 2026-07-09 — S7: Age & experience features (OPT-IN, NOT YET BACKTESTED)

**Author:** Scientist agent
**Files:** `scripts/feature_engineering.py` (new), `supercoach/prediction.py`,
`tests/unit/test_age_experience_features.py` (new)
**Blast radius:** HIGH (feeds the production predictor) — de-risked to
behaviour-neutral by an opt-in flag defaulting OFF.
**Status:** IMPLEMENTED, OPT-IN, **NOT production-ready** — a backtest
comparison is required first (see below).

### Feature definitions

| Feature | Definition | Source |
|---|---|---|
| `player_age_at_match` | `(match_date - born_date).days / 365.25` — age in years at each game. Float. | `born_date` = filename DOB token (`..._DDMMYYYY_...`), verified == `personal_details.born_date` on a 40-file sample (0 mismatches). `match_date` = performance-file `date`. |
| `career_games_to_date` | Count of the player's games **strictly before** the current game, in temporal order. 0-indexed `groupby.cumcount` over a `(year, round_number, date)`-sorted frame. Float. | `performance_details` rows (1 row = 1 game). |

Pure functions live in `scripts/feature_engineering.py`
(`compute_age_years`, `compute_career_games_to_date`), wired into
`AFLDisposalPredictor._add_age_experience_features` and gated behind the
constructor flag `include_age_experience` (**default False**) / CLI flag
`--include-age-experience`. With the flag OFF the training feature matrix is
byte-identical to before this change (verified: 44 → 44 columns, no `born_date`
column added).

### TEMPORAL-CUTOFF INVARIANT (a future Scientist MUST verify before promotion)

Both features use only at-or-before-match information, composing safely with
`LeakProofPredictor`'s hard cutoff:

1. `player_age_at_match` is deterministic from DOB + match date — a player's
   age at round N is independent of any later round. No future info by
   construction.
2. `career_games_to_date` is a 0-indexed cumcount over a temporally sorted
   frame: first game = 0, each later game +1, so a row counts ONLY strictly-
   prior games. The current and all future games are excluded. Under the
   backtest cutoff (future rows dropped in `LeakProofPredictor.load_and_prepare_data`
   *before* feature engineering), future rows are absent and cannot be counted.

**Regression guard:**
`tests/unit/test_age_experience_features.py::test_career_games_appending_future_game_does_not_change_earlier_counts`
proves that appending a later game leaves every earlier row's count unchanged.
Before flipping this feature on in production, confirm (a) that test passes and
(b) `_add_age_experience_features` is still invoked *after* the temporal filter,
never before it.

### Verification done
- Unit tests: 12/12 pass (age exactness, 365.25 leap handling, Series/scalar
  DOB alignment, NaN propagation; career-games strict-before, per-player
  independence, cross-season accumulation, row-order independence, index
  preservation, the leakage guard above).
- Integration smoke (Pendlebury, real data): flag OFF → feature set unchanged,
  no `born_date`; flag ON → +2 features, age 18.3–37.7 (plausible), career
  games strictly non-decreasing in time, zero NaNs.
- Repro: pandas 2.2.3 / numpy 1.26.4 / sklearn 1.6.1 / lightgbm 4.6.0.

### REQUIRED before production promotion (NOT done here)
- [ ] **Backtest OFF vs ON** on the same rounds (MAE / RMSE / pct-within-5 /
      top-10 MAE). Features are only justified if held-out error improves or is
      neutral. Each backtest round ≈ 24 min Optuna; full run ≈ 5–6h CPU — out of
      scope this session. To enable, construct `LeakProofPredictor` with
      `include_age_experience=True` (it currently is NOT — the backtest still
      runs the production feature set until this comparison is deliberately
      wired). Respect the incremental-only backtest rules
      ([[feedback_backtest_rules]]).
- [ ] Re-confirm the temporal-cutoff regression guard passes at promotion time.
- [ ] Error-analysis slice: does any signal concentrate by career stage
      (early-career vs veteran)?

### Caveats
- Performance-file `date` is known to be off by up to ~1 month from the true
  match date. Negligible for age-in-years (±~0.1 yr); irrelevant to
  `career_games_to_date` (ordinal, not absolute date).
- `career_games_to_date` counts rows in the loaded frame — NOT the afltables
  `games_played` career counter (which has drawn-GF / missing-finals-row
  quirks). Cumcount was chosen for clean, testable leak-proof semantics; the
  `games_played` column is an alternative to A/B if cumcount underperforms.

**Residual risk:** low while OFF (behaviour-neutral); the ON path is untested
for accuracy and must not be promoted before the backtest above.
