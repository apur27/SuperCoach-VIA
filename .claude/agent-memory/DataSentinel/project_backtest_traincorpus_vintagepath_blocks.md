---
name: backtest-traincorpus-vintagepath-blocks
description: docs/afl-backtest-2026.md TRAINCORPUS/VINTAGEPATH generated blocks (added 2026-07-28, replacing stale hand-written prose) — exact reproduction recipes and what each verifies.
metadata:
  type: project
---

## 2026-07-28 (Pass-2, hash `c942d143…`) — two new generated blocks verified

`scripts/update_eval_surface.sh` converted two previously hand-written (and stale)
paragraphs into generated marker blocks. Both verified computationally, zero failures.

**`<!-- TRAINCORPUS-START/END -->`** (Methodology step-1 corpus-scope table):
- `supercoach/prediction.py:598` is `historical_data = df[df['year'] < self.target_year].copy()`
  — confirms the `year < target_year` filter claim and line number (the OLD hardcoded
  line number in this doc, `:589`, had already drifted stale before this fix — the new
  anchor-derived `:598` is correct as of this commit).
- `supercoach/prediction.py:434` is `birth_year_threshold = self.target_year - 40` — confirms
  the 1986-for-2026 birth-year filter and line number.
- Season span "2005–2025" is NOT the full corpus span (full archive of all
  `*_performance_details.csv` files reaches back to 1897) and is NOT even the raw span of
  just the birth-year-admitted files (which is 2005–**2026**, since 2026 rows exist as
  prediction targets in those same files). The doc's own caveat ("target year excluded")
  is what reconciles this — reproduce it as: filter files by the *same* birth-year
  admission rule the loader uses (`dob.year > target_year - 40`), take the min/max `year`
  across only those admitted files' rows, then drop the target year from the top of that
  range. Do not filter by `year` column directly — the admission is per-FILE via DOB in
  the filename, not per-row.
- Player-files-loaded count `1,808 of 13,357` reproduces exactly via direct replay of
  `load_and_prepare_data`'s per-file DOB-admission loop (glob `*_performance_details.csv`,
  `extract_dob_and_name`, skip if `dob.year <= threshold`). Same recipe as
  [[project_backtest_reproduction_recipes]] item 2, now doubly confirmed against a live
  `supercoach.prediction` import rather than approximated.

**`<!-- VINTAGEPATH-START/END -->`** (per-round retrain-vs-archive-path + attestation table):
- Retrain/archive classification: grep each dedup-selected round's own vintage
  `backtest_run_<ts>.log` for `[cutoff y=2026 r=N] dropped` (retrain) vs
  `scoring archived prediction CSV ... (no retrain)` (archive) — exact same method as
  [[project_backtest_doc_verification_gotchas]] §"2026-07-27 update", now generalized to
  R1–R21. As of this pass: R1–R17 retrain (17 rounds), R18/R19/R20/R21 archive (4 rounds).
- **Attestation check, new this pass**: "attested" means the forward-prediction CSV's
  first `git log --diff-filter=A` commit timestamp is strictly earlier than the round's
  first-bounce fixture time (`data/matches/matches_2026.csv`, min `date` for that
  `round_num`), with the fixture time read as if it were UTC+8 (the doc's own stated
  convention — the earliest Australian venue offset, so the test is conservative for
  eastern venues) and the git commit timestamp converted from its recorded offset (this
  repo's commits show `+1000`) to the same basis before comparing. In practice the
  margins seen so far (days, not hours) make the UTC+8-vs-AEST distinction immaterial —
  but do the timezone-aware comparison anyway rather than assuming it never matters,
  since a same-day commit-vs-bounce race is exactly the case this convention exists to
  catch correctly. R18/R19/R21 attested (committed well before first bounce); R20 has NO
  git commit for its forward CSV at all (`git log --diff-filter=A --follow` returns
  nothing) — correctly "not attested". Doc's own summary line ("17 retrain / 4 archive; of
  archive, 3 attested / 1 not") reproduces exactly.

See also [[project_backtest_doc_verification_gotchas]], [[project_backtest_reproduction_recipes]].
