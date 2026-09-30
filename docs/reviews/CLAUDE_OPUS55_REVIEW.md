# Claude Code (Opus 5.5) review and data-integrity checker: 29 September 2026

*Engineering review. Player and match statistics are tagged: **[data]** means read from
the retained snapshot or its CSV inputs; **[historical record]** means a public source page.
Timings, byte counts and test counts are engineering measurements.*

## 0. Follow-up (30 September 2026)

This section records the follow-up payload (`docs/rewrite/CLAUDE_CODE_OPUS_55_FOLLOWUP_PAYLOAD.md`).
It supersedes the verdicts in §5, and qualifies two claims made below. §5 said "every
published cell matches the snapshot", but that held only for the match and player
resources then compared. §6 called the team and history comparisons optional; they were
required, and they are now implemented.

### 0.1 Identities

| Item | Value |
|---|---|
| Branch | `review/opus55-integrity`, not merged, main untouched; follow-up commits `3eb885410`..`7101482f3` plus the documentation commit |
| Retained inputs (unchanged) | snapshot `sha256:aa836549…2bb98f`, release `20260928T111014Z-ca96603163ad`, under the main checkout's `var/finalized/`; pointer digest recorded before and after in the run directory |
| Candidate snapshot | `sha256:3de6597513b5c6eaca1b102d304a2cf128d8c08f229ee1de384c1d40fe746db0` (parent `aa836549…`, made by `scvia apply-corrections` on a copy) |
| Candidate model | `bundle-2103e39a9c78444b0de6` (`hgb`), manifest sha256 `f8f86a79…`, retrained on the candidate snapshot |
| Candidate release | `20260929T213900Z-c8938f4ddb83`, checksums `c8df185d…`, seal `6154fbf3162cbd6071527d85de52a1605c30fbc942afdbb565c39c0d250c6395`, validation PASS |
| Run directory | `var/reviews/opus55/20260929T203757Z-followup/` (candidate data and release, logs, audits, smoke, browser) |
| Committed samples | `docs/reviews/opus55/integrity-report-candidate-full.json` (and its findings stream), `integrity-report-retained-followup.json` (the reference), `browser-facts-candidate.json`, `smoke-old-new-compare.json` |

### 0.2 Two further checker findings (both fixed)

**C55-01 · Medium: output destinations could alias.** `--execution` defaulted to a path
that could equal `--report`, and aliasing through relative, absolute or symlinked-parent
forms was not detected. One output silently overwrote another, and a failed write left
a receipt claiming files that were never written. Fix: `plan_outputs` resolves every
destination, refuses any two that name the same file (by path or inode) before the audit
runs, and leaves existing bytes untouched on refusal. A failed write now reports exactly
which outputs were written and which were not. Tests: `test_integrity_cli.py`, six
aliasing forms and a partial-write receipt.

**C55-02 · High: semantic coverage was incomplete, but the report said complete.**
`scope.complete` was true while team pages, history tables, lists, downloads, the overview,
the quality page, accuracy, live resources and articles had no semantic comparison. A team
total changed with every hash recalculated passed every check. Fix:

- `release.derived`, `release.forecast`, `release.content` and `release.coverage` (see `docs/data-integrity.md`);
- the B1 player-page comparator `source.player_pages`;
- `scope.semantic_complete`, true only when every public file was compared or is provenance-only by type.

A missing comparator input, or a comparison that did not run, is now UNKNOWN. The
adversarial tests change team totals, history ranks and values, download row membership
and stale summaries, in cold, warm-cache and changed-since modes.

### 0.3 Status of the O55 findings

| Finding | Status | Evidence |
|---|---|---|
| O55-01 zeros stored as "not recorded" | **fixed in the candidate** | `domain/blanks.py`, one rule for both importers and refresh. Annotated captured cells: `tests/scvia/fixtures/zero_semantics/annotations.json`. Independent checker reading: `football.blank_as_null` / `unevidenced_zero` / `brownlow_not_applicable` and `checks_source.expected_cells`. 2,903,202 cells **[data]** became recorded zeros. Pendlebury's goals are 209 over 442 games, 0.5 per game on the site (was 1.29) **[data]**. |
| O55-02 R17 attendance | **fixed in the candidate** | `apply-corrections` from pinned capture `87c99c1a…`, with a resolved `fixture_field_corrected` issue. Snapshot, public resource and embedded site show 62,117 **[historical record]**. The grand-final `venue_id` gap was fixed the same way. |
| O55-03 malformed stat text becomes null | fixed | the match and player parsers fail on non-numeric text |
| O55-04 compact rows unchecked | fixed | producer validators and a browser contract; checker tests still build invalid releases (`reseal(expect_valid=False)`) |
| O55-05 duplicate keys, lax booleans | fixed | strict JSON loading and `StrictBool` |
| O55-06 replay rows linked to drawn finals | **fixed in the candidate** | 12 rows relinked where official team scores reconcile; 13 quarantined with an actionable issue **[data]** |
| O55-07 hermetic tier over 30 s | **still open** | see 0.6 |
| O55-08 mobile clutter, "17 2026" titles | fixed | "Collingwood v Richmond, Round 17 2026"; the strip at 320 px is 133 px on the real candidate (was about 270) |
| O55-09 little independent source evidence | unchanged, now measured more fully | 138 rows on 3 captured match pages and 92 rows on the 3 B1 player pages (1,235 exact cells, 881 blanks confirmed non-positive) **[data]**; every other row stays `legacy_unverified` |

**Zero semantics: what is and is not decided.** A blank becomes 0 only when all of these
hold: the match reports the statistic, the row took the field, and the season is inside the
statistic's recorded era. Time on ground is never zero-filled, and finals Brownlow votes are
not applicable. Decision 3 (observed denominators) and the `N of M` disclosure are
unchanged. Remaining gaps stay visible, for example Brownlow votes at 409 of 442 games for
Pendlebury **[data]**, the 33 finals being not applicable. The evidence is the source's own
convention, verified on the captured pages and consistent across the corpus: every
home-and-away match since 1984 sums to 6 Brownlow votes (7,620 of 7,620) **[data]**. Applying
that convention to legacy rows that have no capture of their own is a policy choice. The
candidate makes it; **the owner should confirm it before activation**. The candidate records
no owner approval.

### 0.4 Candidate validation

- **Audit** (`--as-of 2026-09-30T00:00:00Z`, B1 evidence, curated content): PASS, complete,
  `semantic_complete`; 26 of 26 checks PASS; open findings are 4 historical goal-mismatch
  warnings (18 before), 1 `models.prediction_snapshot` warning (the retained run's old
  prediction artifact sits in the copied predictions directory) and 1 info **[data]**.
  26.6 million cells compared, 1,854 derived resources, 0 uncompared. Wall time and RSS are
  in `docs/data-integrity.md`. Cold at 1 and 4 workers, warm, and changed-since reports are
  byte-identical. A one-value edit on a copy was detected and invalidated exactly 2 cached
  results.
- **Reference** (the retained inputs): FAIL, kept as the reference. 1 blocking R17
  attendance; 1,167 blocking source-cell contradictions (captured blanks are zeros the
  retained snapshot holds as nulls); 25 replay-link errors; 44 blank-as-null warnings **[data]**.
- **Grand final end to end:** Brisbane Lions 14.12 (96) d Fremantle 12.17 (89) at the M.C.G.,
  26 September 2026, attendance 100,023 **[data]**. All 46 player rows are compared cell
  by cell with the pinned match capture `78749199…`. The snapshot, public match detail and
  embedded site copy agree byte for byte, and the page renders "Fremantle v Brisbane Lions,
  Grand Final 2026". Its presence in the data is separate from source coverage and from the
  zero semantics.
- **Retraining:** new bundle; champion `hgb`; matched holdout MAE 3.760 against the prior-5
  baseline's 3.909 (3.81%, gate passed); 80% interval holdout coverage 81.2%. These are
  model-evaluation metrics computed on the candidate, not reused. The forecast remains
  `unavailable` (`no_valid_future_fixture`). The engineering model card
  (`docs/model-card.md`) still describes an earlier bundle; the release's own model card is
  generated from the new manifest.
- **Site:** 263,922,347 bytes against a 314,572,800-byte budget, 48.3 MiB of headroom. The
  2026 season-scoped files are 3.31 MiB; at about 4.3 MiB per modern season in total that is
  about 11 seasons. Browser sweep: 23 routes × 4 widths × 2 themes, no console or page
  errors except the intended 404, 0 axe violations, grand-final and watchlist flows pass,
  and all 14 downloads return 200.
- **Rehearsal and smoke:** `SCVIA_NUMERIC_ENTRY=1 scripts/weekly_refresh.sh` ran on a scratch
  copy (source inventory `cbb62505…`, 895 files, 0 differences) in rehearsal mode, with no
  network and no commit or push. It exited 0 in 278 s at 1.92 GiB peak tree RSS; the
  import, forecast, build, site, budget, seal and validate phases all passed. Old/new
  comparison: all-time top 100 identical (max delta 0), 130 of 130 yearly lists exact,
  biography rows 100 of 100 identical. A first attempt was refused by the harness guard
  because the RSS sampler's own command line named `weekly_refresh.sh`; the guard was
  right, and the sampler was changed.
- **§6 scope determination.** No harness `.sh`, hook, gate script, legacy-tier
  `tests/integration` file or harness-invoked top-level Python entry point changed.
  `refresh`, `import-legacy` and `cli.py` change the data the opt-in numeric entry produces,
  so that path was treated as in scope and smoke-run (above). The default legacy harness
  path is unchanged. At this follow-up's commit, a rehearsal import from the legacy CSVs still carries R17 attendance 0,
  because the capture-backed correction is applied by `apply-corrections` (or a refresh),
  never by editing `data/`. The numeric cycle must run it after an import.
  The later GPT-6.1 Sol fixes in §0.7 add that step with explicit pinned evidence.

### 0.4a Tests on final code

| Tier | Result |
|---|---|
| `pytest tests -m "not integration"`, 4 workers on CPUs 0,2,4,6 | 1,479 passed, 45 skipped (pre-existing skips), 82.0 s |
| `pytest tests/scvia -m "not integration"` (the hermetic tier) | 924 passed, 51.7 s (target 30 s, see 0.6) |
| `pytest tests/scvia -m integration`, real corpus, integrity test on the candidate | 43 of 43 pass after one oracle update (below); 320 s, 3.34 GiB max process |
| `pytest tests/integration -m integration` (legacy Phase 3d gate) | 17 passed, 3 failed, 1 skipped. The failures are pre-existing and outside this branch: the two backtest reconciliations (9,099 against 9,007 per-team rows, REHEARSAL.md Finding 1) and chart reproducibility under this worktree's rendering environment. The branch changes nothing under `data/`, `assets/` or the harness. |
| ruff, mypy | clean |
| web: `gen:types:check`, `check`, `lint`, Vitest | up to date, 0 errors, clean, 174 passed / 2 skipped |
| Playwright e2e at `/` and `/SuperCoach-VIA/` | 322 passed, 4 skipped (run-once checks) |

**Original oracle update (superseded by §0.7).** `test_era_stats_match_legacy` compared the new era statistics with the
legacy script, which reads every blank as missing, so O55-01 made it fail by design. It now
states parity exactly. The number of observations may only grow. Because the added values
are zeros, each metric's sum and sum of squares are unchanged, so the new mean and SD are
checked to 1e-9 against values derived from the legacy output. Every column, median and
per-100% included, must still match exactly wherever no blank was resolved. The real-data
integrity test now supplies the curated content and asserts semantic coverage and the B1
player pages; it passes on the candidate (18 of 18).

### 0.5 Nothing here authorised

There was no merge, push, deployment, schedule change, default harness activation or
budget increase. No rehearsal is counted as a shadow cycle, and no owner approval is
recorded.

### 0.6 Test performance (O55-07), still open

Measured on the documented set-up (4 workers pinned to CPUs 0,2,4,6, no competing job):

| Run | Tests | Time |
|---|---|---|
| hermetic tier before this follow-up's test-setup change | 924 | 54.1–54.5 s |
| after (one shared DEMO build per worker, sequential season path for it) | 924 | 50.9–52.0 s |
| without the checker's tests | 735 | 29.3 s |
| checker tests alone | 189 | 30.6 s |
| all 924, 8 unpinned workers (not the documented set-up) | 924 | 42.6 s |

The dominant costs are the checker's DEMO-release tests. Each one copies, reseals
(`validate_release` 0.22 s) and audits (0.5–1 s). Per-worker DEMO construction takes 4–6 s.
No test was deleted, skipped, moved or reclassified, and the budget was not raised.
Alternatives, for the owner:

1. Run the checker tests as their own parallel hermetic job (each half is about 30 s).
2. Let `reseal` and the checker share one `validate_release` pass per test.
3. Faster tree walks in `validate_release`: `pathlib.relative_to` is about half its time on
   the DEMO. Implemented subsequently in §0.7; this alone does not meet the tier budget.
4. A budget decision.

This rewrite target (30 s) is separate from the legacy fast-tier budget in `CLAUDE.md`
(about 20 s, `tests/unit`).

### 0.7 GPT-6.1 Sol coding follow-up (30 September–1 October 2026)

At the owner's request, three GPT-6.1 Sol coding agents implemented fixes on this review
branch, followed by cross-review and local verification. The earlier candidate and
retained reference artifacts were left unchanged. This work does not approve the legacy
zero-semantics policy, activate the numeric entry, or count as a real shadow cycle.

**Audit inputs.** A new reproduction showed that editing a curated article after its
comparison could leave `inputs.stable` PASS. Auxiliary inputs now use captured content
for parsing and identity: articles, assets, policy, ranking method, predictions,
evaluations, live captures, model hashes and the source evidence selected by the checks.
Changes, deletions and newly selected inputs make stability UNKNOWN and coverage
incomplete. Regression tests cover drift before/after comparison, relocation, missing
inputs, duplicate evaluation basenames and source containment. Cross-review also caught
omitted manifest defaults and an invalid live path; both now have passing regressions.

**Numeric correction order.** The opt-in workflow now applies corrections after import
or refresh and supplies the resulting immutable snapshot to forecast, comparison and
release build. A fresh legacy import has no pinned season capture, so the rehearsal
requires an explicit donor root and immutable donor snapshot. Only verified season
captures and successful source observations are copied; donor match/player rows and
legacy CSVs are untouched. Missing, corrupt or escaping evidence stops the workflow.
Receipts distinguish the source snapshot from the corrected snapshot. See
[`SWITCH_PLAN.md`](../rewrite/SWITCH_PLAN.md) for commands and failure recovery. An import
can already have promoted its source snapshot when correction fails; no corrected child
or public release is accepted on that failure.

This procedure corrects existing match fields. It cannot add a grand final absent from
the captured CSV input. The retained corrected candidate already contains the grand
final and was checked separately; a rehearsal from the older CSVs has narrower coverage.

The first full scratch run stopped at strict ranking parity: the new replay quarantines
changed its inputs, while the legacy ranker still read the original CSVs. The comparison
now verifies snapshot ancestry, source hashes, replay identities, quarantine evidence
and row multiplicity before applying those bounded removals to scratch CSVs. Relinks
must preserve statistics, and unexplained edits still fail. The comparison tolerances
and score/order checks are unchanged. A focused real rerun matched every yearly list and
all-time score exactly. A separate regression also fixed corrections containing only
quarantines, which previously omitted a required empty upsert table.

**Independent era expectations.** The previous parity test used the new output's own
observation count to calculate expected means and skipped changed medians/per-100 values.
The replacement reads original CSV cells and independently resolves eligibility. It
asserts exact row membership and counts, then compares every mean, spread, median and
per-100 result. Only provenance links and match stage come from the snapshot. Golden
cases and deliberate denominator/zero mutations test the oracle itself.

**Release validation cost.** The directory walk now carries relative names and makes one
fresh `lstat` per entry. It retains byte hashing, seal checks and symlink refusal,
including a file replaced by a symlink after enumeration. On the same 631-file sealed
DEMO, five alternating runs reduced median validation time from 0.269 s to 0.115 s.
This is a small-fixture measurement, not a claim about the full release or tier.

**Consolidation and cleanup.** The owner requested delivery to main and fewer branches.
Useful uncommitted parser notes and browser-contract checks were preserved; an obsolete
budget-test edit was archived. The superseded phase2 scaffold, old performance/ML
experiments and patch-equivalent agent branches have a verified recovery bundle under
`var/recovery/branch-cleanup-20260930/`, with branch tips, patches and original file bytes
in `manifest.json`. The bundle is incremental and requires the history retained in main.
The obsolete daily/main-push phase2 sync workflow was removed. Production numeric
activation remains subject to §0.8. Branch names are retired after the verified merge;
linked worktrees and ignored verification artifacts are retained.

Verification receipts are under the main checkout's
`var/reviews/codex-opus55-20260930/validation/`; canonical audit runs are under
`var/reviews/codex-opus55-20260930/sol-audits/`.

| Check | Result |
|---|---|
| Final hermetic tier, four workers pinned to CPUs 0,2,4,6 | 1,025 passed in 54.56 s; the 30 s target remains open |
| Numeric integration tier | 43 passed in 462.24 s; run before the final replay-comparison and quarantine-only fixes, which have focused regressions and the final scratch rehearsal |
| Ruff / mypy | clean; mypy checked 76 source files |
| Web generated contracts, type checks and lint | pass |
| Vitest | 174 passed, 2 skipped; an additional real-release contract run passed all 12 tests |
| Corrected candidate audit | all 26 checks PASS, complete and semantically complete; 26,599,913 cells compared, 5 warnings and 1 info, no blocking/error findings |
| Audit determinism | reports byte-identical across cold 1/4 workers, warm cache and changed-since; 121.586 / 102.128 / 26.588 / 25.725 s respectively |
| Auxiliary input mutation probe | a mid-audit article edit changes stability to UNKNOWN and semantic completeness to false; a fresh audit of the changed input fails |
| Focused real ranking comparison | all 130 yearly lists identical in players, order and scores; all-time score delta 0 |
| Final scratch rehearsal | PASS in 458.423 s; two sealed releases, strict ranking parity, failed-copy protection, local publish and rollback; 897 source files match |

The canonical report digest is
`061753ff8fe3c00288157b48279736d873af49970fc9c622e1cb7c0ec93dc6c3`.
The final checker code fingerprint was checked against that report after the last
source edit. The mutation probe is restricted to release/storage checks; it is a
regression reproduction, not evidence of full-scope coverage. The audit's execution
RSS fields describe process maxima, not a sampled process-tree peak.

No tests were removed or reclassified to meet the time target. The earlier full browser
E2E results remain historical; this follow-up changed a browser contract test, with no
production browser-code edits. The legacy integration failures in §0.4a remain open.

The final rehearsal is `/tmp/scvia-sol-smoke-20260930-final2`. Its source inventory is
`a7d9a3ae41a468246bb93e3a825f8311d6925f46af263630a02ea610fa3ff1e0`.
Both cycles selected the corrected snapshot `sha256:e70e66f6…21d300`; both public
resources show R17 attendance 62,117 **[data]** from the pinned season capture.
Sampled process-tree peaks were 1,920,768 / 1,909,444 KiB (about 1.83 / 1.82 GiB).
The failed copy left the first release's bytes unchanged, and rollback returned the
temporary host to `20260930T211937Z-0495e493fc14`. A compact receipt is committed at
[`sol-followup-verification.json`](opus55/sol-followup-verification.json).
This is a local rehearsal using older captured CSVs, not a real production shadow cycle.

### 0.8 Verdicts

| Question | Verdict |
|---|---|
| **Integrity checker** | **Ready for opt-in operator use.** It now compares every published resource type with its own input and reports incomplete coverage as UNKNOWN. It is deterministic across modes and worker counts, and the hermetic and real-data tiers pass. It is still not wired into any gate. |
| **Candidate app** | **Usable locally as a candidate.** Sealed, validated, audited PASS with full semantic coverage, and browser-checked. It embeds a zero-semantics policy that the owner should confirm (0.3). The candidate release, data and bundle live only in the run directory. |
| **Production activation** | **Not ready.** It still needs two genuine shadow cycles, the `SWITCH_PLAN.md` decisions, the owner's confirmation of the zero-semantics rule for uncaptured legacy rows, and a decision or further work on O55-07. The correction step is implemented in §0.7; the default entry is unchanged. |

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

## 5. Verdicts (first review; superseded by §0.7)

| Question | Verdict |
|---|---|
| **Local application use** | **Usable, with one correction needed first for statistics.** Integrity is sound: every release byte matches its seal, and every published cell matches the snapshot. But per-game averages for sparse statistics (O55-01) and R17 attendance (O55-02) are wrong on screen. Totals, games, box-score values and match facts are correct. |
| **Integrity checker readiness** | **Ready for opt-in operator use.** It is deterministic, read-only and offline. 144 hermetic tests pass, including negative controls, determinism, cache, changed-mode equivalence and read-only checks, and the real-data runs are recorded. It is not wired into any gate; doing that is a §6.2 change needing its own smoke run. |
| **Production activation** | **Not ready.** It still needs two genuine shadow cycles, the owner decisions in `SWITCH_PLAN.md` §6, O55-01 decided and rebuilt, O55-02 repaired, and a decision on O55-07. |

## 6. Remediation, in order (first review; see §0.3 for status)

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
