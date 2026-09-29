# Claude Code (Opus 5.5) review and data-integrity checker: 29 September 2026

*Engineering review. Player and match statistics are tagged: **[data]** means read from
the retained snapshot or its CSV inputs; **[historical record]** means a public source page.
Timings, byte counts and test counts are engineering measurements.*

## 1. Identity, scope and limits

| Item | Value |
|---|---|
| Model | Claude Code 2.1.283, `claude-opus-5-5` (Opus 5.5), as configured for this session |
| Starting point | `main` at `21dff89a6` (payload commit), clean; ancestors confirmed: app `2bcdfbeb4`, handoff `c101a09e3`, Claude implementation `2aad17873` |
| Work branch | `review/opus55-integrity` in worktree `var/worktrees/opus55-integrity` (not merged; main untouched) |
| Snapshot | `sha256:aa836549e10e96a9039cc4642bbe563b94247e0bfaf95e745a0971cc0bb2b98f` (legacy_unverified) |
| Release | `20260928T111014Z-ca96603163ad`, seal `bd9a655b06795c2d83ca77a48adb9df85fb2774463fa390f814dac2ff72474fa` |
| Evidence directory | `var/reviews/opus55/20260929T074525Z-integrity/` in the main checkout (reports, execution metadata, logs, 187 screenshots) |
| Committed samples | `docs/reviews/opus55/` (canonical checker reports, findings stream, browser summary) |

The retained inputs were all present and were read in place, never modified. The
first-audit input identities are recorded in `docs/reviews/opus55/integrity-report-retained-full.json`
(`inputs.digest` `0686717696cdf170…`).

Not done:

- no deployment, publication, merge to main, schedule change or harness activation;
- no retraining, and no recomputation of the model card's holdout metrics;
- no network access;
- no archive (`site.tar`) round trip; the seal and every site byte were verified in place instead;
- no second machine.

Nothing here counts as a production shadow cycle.

## 2. Findings (most severe first)

"Confirmed" means reproduced here, with the evidence path given.

### O55-01 · High · confirmed: published per-game averages are inflated because zeros are stored as "not recorded"

- **Where.**
  - Import: `src/supercoach_via/ingest/legacy.py:1001` (`blank_to_null`) and `src/supercoach_via/ingest/afltables.py:513` (`_stat_value`).
  - Policy: `src/supercoach_via/domain/metrics.py:5-12` and `analytics/players.py:32` (`COVERAGE_NOTE`).
  - Display: `web/src/islands/PlayerView.tsx:97` ("… per game"), `CompareView.tsx:71,103,115` and the box scores in `MatchView.tsx`.
- **Trigger.** AFL Tables prints a zero count as a blank cell, both in the legacy CSVs (`data/player_data/pendlebury_scott_07011988_performance_details.csv` has 280 blank and 0 zero `goals` cells **[data]**) and on live pages. The importer maps every blank to null, and the published means divide by non-null games.
- **Evidence.** Zero is never stored in any of the 23 statistics across 695,521 player-game rows **[data]**. For Scott Pendlebury the player page shows **[data]**:
  - goals 209 in 442 games, published at 1.29 per game (209 ÷ 162) instead of 0.47;
  - Brownlow votes 230, shown as **2.1 per game** (230 ÷ 107) with "107 of 442 games with data".

  Box scores label a player who kicked no goal as "not recorded".
  - `docs/reviews/opus55/browser-review-summary.json` (`flows.pendlebury_career_table_head`)
  - `var/reviews/opus55/…/browser/screens/match-r17-w1440-light.png`
  - checker rule `football.blank_as_null`: 22 statistics **[data]**, in `integrity-report-retained-full.json`
- **Impact.** Every mean, comparison, season line and coverage percentage for sparse statistics is wrong on every player page. Coverage badges claim data is missing when the source recorded zero. ML features built from `prior-N` means of sparse stats (tackles, clearances, inside 50s) are conditioned on non-zero games. That is consistent between training and serving, but it is not what the feature names say.
- **Expected versus actual.** Per-game = total ÷ games in the stat's recorded era (0.47 goals **[data]**); actual = total ÷ games with a non-zero value.
- **Remediation.** This keeps decision 3 in `docs/pending-decisions.md` (average over recorded-era games, never fill pre-era gaps) and corrects its implementation, which treats in-era blanks as unrecorded.
  - At import, for a stat whose `recorded_from` is at or before the row's season, a blank cell in a source table that has that column becomes `0`. Pre-era and absent columns stay null.
  - Apply the same rule in `afltables._stat_value` and the legacy importer.
  - Recompute; relabel "games with data" as "games in recorded era".
  - Because this changes every published statistic, it needs an **owner decision** and a re-rehearsal of `rehearsal_compare.py` (legacy_v1 already zero-fills, so parity should improve).
- **Regression required.** Import tests: an in-era blank becomes 0 and a pre-era blank stays null. `football.blank_as_null` reports nothing on the rebuilt snapshot. A browser test asserts that the per-game value equals total ÷ games.

### O55-02 · Medium · confirmed (current-season data): R17 attendance contradicts the pinned source

- **Where.** `data/matches/matches_2026.csv` (legacy input); displayed by `web/src/islands/MatchView.tsx:61`.
- **Evidence.** The pinned season page `87c99c1a…` says 62,117 **[historical record]**; the snapshot holds 0 **[data]**. The site shows "attendance 0" (screenshot above). This is the checker's single blocking finding (`freshness.fixture_value_mismatch`), so the audit exits 4.
- **Remediation.** Bounded repair from the pinned source, or treat attendance 0 as unknown at import. Do not add an exception, because current-season defects cannot be suppressed.
- **Regression required.** The checker's fixture inventory passes on the repaired snapshot.

### O55-03 · Medium · confirmed: parser drift in one statistic cell silently becomes null

- **Where.** `src/supercoach_via/ingest/afltables.py:513-520`. `_stat_value` returns `None` on `ValueError`, and `parse_match_detail` records no issue.
- **Reproduction.** Replace one numeric cell with `8a` in the committed B1 page `08a2bf59…`. `parse_match_detail` returns outcome **PASS** with no issue, and that player's `marks` becomes `None`.
- **Impact.** A malformed or reformatted source cell is published as "not recorded" and passes the refresh gate.
- **Remediation.** Treat non-blank, non-numeric text as a parse issue (FAIL). The checker's independent reader already keeps such text and flags `source.stat_cell_mismatch`.
- **Regression required.** A unit test on a page with an `8a` cell expects FAIL.

### O55-04 · Medium · confirmed: release validation does not check compact-row width or stat-column vocabulary

- **Where.** `src/supercoach_via/publish/view_models.py:247-258` (`BoxScoreColumns._aligned` checks only that the three arrays have the same length) and `:272`, `:558` (`MatchDetail` and `PlayerSeasonGames` never compare row width with `stat_columns`).
- **Evidence.** `tests/scvia/unit/test_integrity_release.py::test_truncated_compact_row_with_hashes_recalculated`. A match detail with one short stats row, re-hashed and re-sealed, passes `validate_release` (the test's `reseal` asserts that), and only the checker's `release.compact_row_width` catches it.
- **Impact.** The browser maps values by position, so a builder bug of this kind would show values under the wrong statistic with every gate green.
- **Remediation.** Add model validators: every stats row has `len(stat_columns)` values, and `stat_columns` is a subset of `PLAYER_STAT_COLUMNS` in canonical order with no duplicates.
- **Regression required.** Make that test's `reseal` assertion fail.

### O55-05 · Low · confirmed: release metadata JSON accepts duplicate keys and lax types

- **Where.** `src/supercoach_via/publish/release.py:113` (`_strict_json` refuses NaN but not duplicate keys; `json.loads` keeps the last value). The public models are in pydantic lax mode, so `1` validates as `true`.
- **Trust boundary.** Local release directories and downloaded archives. The deploy path pins both seal and archive digests, so exploiting this needs a forged archive whose digests already match: low.
- **Evidence.** `test_integrity_release.py::test_boolean_and_number_are_not_interchangeable` (a re-sealed `"active": 1` passes `validate_release`; the checker flags it).
- **Remediation.** Add an `object_pairs_hook` that refuses duplicates, and strict boolean fields on public models.

### O55-06 · Low · confirmed (historical data): replay player rows are linked to drawn finals

- **Where.** The legacy identity-linking of the stage token plus team pair, in `ingest/legacy.py`.
- **Evidence.**
  - Seven drawn finals (1928 SF, 1946 SF, 1948 GF, 1962 PF, 1972 SF, 1977 GF, 1990 QF) hold 25 player rows **[data]** whose W/L is the replay's result. For example, the 1928 SF has 36 D, 3 W and 2 L rows on the drawn match **[data]**.
  - The same cause gives the 1990 QF Collingwood player behinds 13 against team behinds 12 **[data]**.
  - Checker rules `relations.result_mismatch` and `football.player_behinds_exceed_team`, both error, historical.
- **Remediation.** Link these rows by date to `replay_occurrence` 1. Until then, document them as historical exceptions.

### O55-07 · Low · confirmed (process): the hermetic tier exceeds its 30 s budget once the checker's tests are added

| Suite (CPUs 0,2,4,6, four workers) | Tests | Time |
|---|---|---|
| without the checker's tests | 698 | 24.4 s |
| with them | 842 | 40.2–40.9 s |

Logs are in `…/checks/hermetic-*.log`. The first version added 24 s. Targeting each negative test at its own check and sharing one demo build per worker brought that to about 16 s. The budget was **not** raised and no test was skipped. **Owner decision:** accept about 41 s, or give the demo-release checker tests (release, cache, determinism; about 35 s of CPU) their own hermetic job.

### O55-08 · Low · confirmed (UX)

- **Match titles.** A title reads "Collingwood v Richmond, 17 2026", a stage label without "Round".
- **R15 is only partially resolved.** On mobile the provenance strip (season, coverage, source checked, generated, published, status, release, method) takes about 270 px before the page heading on every page (`…/browser/screens/player-w320-dark.png`).

Neither blocks use.

### O55-09 · Info · confirmed (coverage of evidence)

Only 138 of 695,521 player-game rows **[data]** have an independent source capture that could be compared: the grand final and the two B1 match pages. All 138 match cell for cell (3,174 cells). The three B1 player pages have no comparator. Everything else is legacy_unverified, as labelled.

### Checker defects found and fixed while building it

Each was found by running the checker on real data or by a new test; the fix, with a regression test, is on this branch.

- Comparing `stage_label` flagged 29,778 rows **[data]**. The two tables use different vocabularies; `stage_id` is the identity.
- Navigation-arrow cells broke attendance parsing on real match pages.
- The Totals row's behinds include rushed behinds.
- A canonical player with no fact rows has empty `stat_names`.
- Python's `True == 1` hid type changes.
- A global `os.scandir` patch invalidated a determinism test. The test now shuffles the checker's own listing seam and asserts the shuffle was used.

## 3. Earlier findings reconciled

| ID | Status on this branch | Evidence |
|---|---|---|
| R01 site not the validated tree | fixed | `test_release.py` (sealed site is the only upload; tamper refused); checker `release.artifact` PASS on the retained seal |
| R02 model from the future | fixed | `assert_bundle_eligible`, `test_ml_predict.py`; checker `models.ineligible_bundle` |
| R03 persisted feature spec ignored | fixed | `feature_spec_from_stored`, `test_ml_predict.py:55-75`; checker `models.feature_spec` |
| R04 late observation breaks order | fixed | `test_ml_features.py:61,211` |
| R05–R09 | fixed on the Claude branch (per reuse doc); suites pass | hermetic and integration logs |
| R10 existing destination trusted | fixed | `LocalDirectoryDestination.upload` re-hashes; `test_release.py:334,396` |
| R11 unlisted resources copied | fixed | `web/integrations/release-tree.mjs:105-131` (closed inventory; extras refused) |
| R12 inferred dates in logs | fixed | logs show the linked match date; checker verifies date and label rule on 58,850 logs |
| R13 season-span search | fixed for current releases | `web/src/lib/search.ts:29` uses exact `seasons`; span only for older releases |
| R14 word wrapping | not reproduced | 184 real-site loads, no horizontal overflow at 320–1440 |
| R15 repeated banners | partially open | O55-08 |
| R16 immutable live polling | fixed | `LiveView.tsx` has no polling |
| R17 key collisions | fixed for new keys; legacy alias kept | `k.` + base64url codec; `web/src/lib/ids.ts:62` still reads legacy `__` links |

## 4. Coverage matrix

| Area | Files read / behaviour checked | Result |
|---|---|---|
| Ingestion and storage | `storage/snapshots.py` (fragments, identity, promotion, atomic writes); `ingest/reconcile.py`; `ingest/afltables.py` (match and season parsers, `_stat_value`); `ingest/legacy.py` (blank handling); `pipeline.py` season aggregates. Retained bytes: 408 fragments, identity, row counts, schemas and partitions all verified; season rows recomputed | O55-01, O55-03; storage and contract PASS |
| Analytics | `analytics/players.py`, `domain/metrics.py`; player detail aggregates for 13,366 players and 58,850 logs recomputed and matched cell for cell | O55-01 (definition, not arithmetic) |
| ML | `ml/bundles.py`, `ml/predict.py` (eligibility, spec, code hash), `ml/train.py` (knowledge cutoff); retained bundle and prediction verified without loading the model; the unavailable forecast checked against the fixture | no new defect; model card labels unchanged |
| Publication and security | `publish/release.py` (closure, seal, validation binding, upload, rollback), `publish/deploy.py` (bounded extraction, pinned digests), `publish/resources.py`, `publish/view_models.py`; seal and all 182,926 files re-hashed | O55-04, O55-05 |
| Browser and UX | real sealed site: 23 routes × 4 widths × 2 themes; GF match→player→back; search; watchlist persistence; 14 downloads; legacy and invalid ids; 404 | O55-01 display, O55-02, O55-08; no console or page errors except the intended 404 |
| Accessibility | axe WCAG 2 A/AA on 6 pages × 2 themes: 0 violations; skip link first, visible focus; reduced motion emulated; no-JS pages explain and link the CSV | no finding |
| Performance | release build 38.9 s, 1.31 GiB process tree (public tree, no forecast inputs); checker 65 s cold, 15 s warm, ≤ 1.86 GiB; hermetic tier timing | O55-07 |
| Code and tests | ruff and mypy clean; hermetic 842 pass; scvia integration 35 pass (286 s, 3.15 GiB peak during real import and training); web gen-types, check, lint, Vitest 163/2 skipped; Playwright 320 pass / 4 skipped at both bases | O55-07 |
| Operations and CI | `.github/workflows/scvia-ci.yml` (locked uv/npm, pinned actions, both bases); `scvia-pages.yml` (manual dispatch, pinned seal/archive); `docs/operations.md`; legacy `tests.yml`/`weekly-fan-pack.yml` still present (switch step C) | no new defect |

## 5. Verdicts

| Question | Verdict |
|---|---|
| **Local application use** | **Usable, with one correction needed first for statistics.** Integrity is sound: every release byte matches its seal, and every published cell matches the snapshot. But per-game averages for sparse statistics (O55-01) and R17 attendance (O55-02) are wrong on screen. Totals, games, box-score values and match facts are correct. |
| **Integrity checker readiness** | **Ready for opt-in operator use.** It is deterministic, read-only and offline. 144 hermetic tests pass, including negative controls, determinism, cache, changed-mode equivalence and read-only checks, and the real-data runs are recorded. It is not wired into any gate; doing that is a §6.2 change needing its own smoke run. |
| **Production activation** | **Not ready.** It still needs two genuine shadow cycles, the owner decisions in `SWITCH_PLAN.md` §6, O55-01 decided and rebuilt, O55-02 repaired, and a decision on O55-07. |

## 6. Remediation, in order

| # | Task | Acceptance | Depends on | Scope |
|---|---|---|---|---|
| 1 | Owner decision on in-era blank = 0 (O55-01) | written decision in `docs/pending-decisions.md` | — | decision |
| 2 | Implement in-era blank = 0 in both importers; relabel UI; update `COVERAGE_NOTE` | import tests; `football.blank_as_null` empty on the rebuilt snapshot; per-game browser test; rehearsal compare shows only documented differences | 1 | 1–2 days, full rebuild |
| 3 | Repair R17 attendance from the pinned source (O55-02) | checker fixture inventory PASS | — | bounded repair |
| 4 | Fail parse on non-numeric stat text (O55-03) | new adapter test FAILs on `8a` | — | small |
| 5 | Row-width and stat-column validators; duplicate-key refusal; strict booleans (O55-04/05) | the two checker tests' `reseal` fails `validate_release` | — | small; release contract change, so re-run the rehearsal |
| 6 | Relink replay rows of drawn finals (O55-06) | `relations.result_mismatch` empty | — | small, historical |
| 7 | Decide the hermetic budget or split the job (O55-07) | written decision; CI job timing recorded | — | decision |
| 8 | Run `scvia check-integrity` in each shadow cycle and attach its report | two shadow cycles with complete reports and no blocking findings | 2–3 | per cycle |
| 9 | Optional: a player-page comparator for B1 evidence; team/history comparisons | `coverage.public_compare.not_compared_by_model` shrinks | — | 1–2 days |

## 7. Checker commands and results

```bash
scvia check-integrity --data-root /abs/var/finalized/data --snapshot current \
  --release-dir /abs/var/finalized/releases/20260928T111014Z-ca96603163ad --scope full \
  --as-of 2026-09-28T12:00:00Z --evidence /abs/docs/rewrite/evidence/b1/raw \
  --report /abs/var/reviews/opus55/<run>/integrity-full.json --workers 4 --cache /abs/var/reviews/opus55/cache --json
```

On the retained artifacts the outcome is **FAIL (exit 4)**, and the audit is complete:

| | Count |
|---|---|
| checks | 21: 20 PASS, 1 FAIL |
| open findings | 1 blocking (O55-02), 26 error (O55-06), 40 warning (22 blank_as_null from O55-01 and 18 existing goal-mismatch warnings from `validate_dataset`), 1 info **[data]** |
| public cells compared | 24,051,313 |
| resources examined | 363,713 |

The operator documentation is in `docs/data-integrity.md`; measurements are there and under `var/reviews/opus55/20260929T074525Z-integrity/final/`.
