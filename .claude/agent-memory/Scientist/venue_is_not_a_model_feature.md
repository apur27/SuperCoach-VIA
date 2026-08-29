---
name: venue-is-not-a-model-feature
description: SETTLED — venue is NOT a fitted model input (no venue column exists in any player CSV); opponent enters only as an identity dummy, never as "defensive style". Do not re-derive from the get_dummies calls.
metadata:
  type: project
---

**`venue` is not, and has never been, an input to the fitted disposal model.** The
`opponent` categorical IS an input, but only as a bare club-identity dummy — there is
no defensive-style, opponent-strength, or conceded-stats feature anywhere in
`supercoach/prediction.py`.

**Why this keeps getting re-litigated:** `venue` appears in four places that *look* like
wiring, and grepping any of them produces a false positive:

- `DTYPES` (`supercoach/prediction.py:191`) declares `'venue': 'category'`
- `load_player` (~:404) `dummy_cols = [col for col in ['venue','opponent'] if col in df.columns]`
- `predict_current_season_disposals` (~:947) — same guarded pattern
- `_engineer_features` (~:556) `dummy_cols = [c for c in df.columns if c.startswith(('venue_','opponent_'))]`

All four are **defensively guarded on column presence**, and the column is never
present. All 13,357 files in `data/player_data/*_performance_details.csv` share ONE
header, and it has no `venue` (it has `opponent`). The sole loader is `load_player`;
there is no merge against `data/matches/` (which does have venue). `RENAMES` maps
nothing to `venue`. So the `venue_*` selector at :556 always matches the empty set,
and `self.feature_columns` — which becomes `training_feature_columns`, the exact list
the estimator is fitted on — contains zero venue features. This is already documented
as a deliberate no-op in `docs/experiment-log.md:213-224` (S1b).

**Empirical confirmation** (60-player modern-era sample, 4,496 training rows,
pandas 2.2.3 / sklearn 1.6.1 / lightgbm 4.6.0): `training_feature_columns` = 46 entries
= 24 rolling/form + `round_number` + `days_since_last_game` +
`percentage_time_played_lag1` + `missing_count` + **18 `opponent_*` dummies + 0
`venue_*`**.

**Why:** Council item S1a claimed the published model description falsely advertises
venue. The premise is CORRECT; a plain grep of the `get_dummies` calls makes it look
wrong. Two prior sessions reached opposite conclusions from the same grep.

**How to apply:** Never conclude a feature is fitted from an encoding call. Trace to
`self.feature_columns` (`prediction.py:556-557`) → `training_feature_columns`
(`:618`) → the matrix passed to `.fit()` (`:895`, `:1056`). If asked again whether
venue is in the model, the answer is no — cite this file, and re-verify cheaply by
checking a player CSV header for a `venue` column before answering.

**Adjacent defect (belongs to S1b, not S1a):** `get_dummies(..., drop_first=True)` runs
**per player file** inside `load_player`, so the dropped reference level is
player-specific — 11 distinct reference opponents across a 200-file sample (Adelaide
151, Brisbane Lions 20, Collingwood 8, ...). After `concat` the ragged union is
NaN-filled and median-imputed. The opponent dummies are therefore in the fitted matrix
but are not a coherent single encoding across players.

Related: [[model_training_corpus_scope]], [[conceded_stats_corruption]]
