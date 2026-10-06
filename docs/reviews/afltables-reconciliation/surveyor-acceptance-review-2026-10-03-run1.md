# Survey — 2026-10-03 — scope: DEEP (final acceptance, AFL Tables reconciliation) — run 1

- Surveyor: requested Opus 5.5; resolved model ID from my own system prompt: `claude-opus-5-5`.
- Read-only. Repo: worktree `var/worktrees/afltables-reconciliation`, branch `work/afltables-reconciliation`,
  HEAD `21ca71797` plus the uncommitted implementation (`git status -uall`, unchanged by me).
- Design: `docs/rewrite/afltables-reconciliation/DESIGN.md` sha256 `da64c4cf…797f` (re-hashed, matches).
- Run evidence (read only): `var/reconciliations/afltables/2026-10-01-full` and `…/2026-10-01-control`.
  I re-hashed `capture/checkpoint.sqlite` before and after a read-only (`immutable=1`) query: `b18dafaf…9b2c` both times.
- My writes: this file, my memory directory, scratch files under my session scratchpad. Two network GETs (§6c).
- Labels: **[obs]** observed by command, **[inf]** inference from observations, **[spec]** speculation (watch list only).

## Executive read

The comparison engine is real, deterministic and mostly faithful to the design. I re-derived the four
report hashes, the stream hash, every cell, appearance and aggregate partition, and traced six findings
to raw captured HTML and local rows; all traced correctly. The data verdict (snapshot FAIL, raw CSV FAIL)
is genuine and does not depend on any defect below. The software is **not acceptable yet**:

- Deviation 3 rests on a false premise. Every sampled match page prints career games-to-date, so a
  required S-07 source-consistency check is missing.
- Two evidence rules assert "proven zero" where the frozen capture shows the record is incomplete
  (1931–34 Brownlow, notes exceptions that coexist with non-blank team totals).
- The S-08 notes alias case (1975 Sydney) still passes as information, and a test enshrines that.
- `--cache` can write inside an input root and the run still exits PASS.

The single highest-leverage move is one Scientist fix batch, B1–B5 below. B1–B4 are all inside the
approved design. Then make one offline re-run of the four compares and the probes against the unchanged
frozen capture. No new mass fetch is needed, and the plan can be re-issued without changing
`capture_identity` provided `capture.py` is not touched until after that re-run (see B6).

## D01–D18

| D | Status | Evidence |
|---|---|---|
| D01 | Met | `docs/reviews/AFLTABLES_RECONCILIATION_DESIGN_REVIEW.md:1-35`: three Opus Surveyor runs, agent hashes, design hash `da64c4cf…` |
| D02 | Met at approval; reopened by B1 | Deviation 3's premise is contradicted by the frozen capture (B1) |
| D03 | Partially | T-markers exist for T01–T35 except T29 (covered only by `test_packaged_copy_equals_checkout_copy`). The T14 counter-vs-games-to-date case is absent, and so is the T23 cache-root case |
| D04 | Partially | Commands, schemas, config and docs exist. `docs/afltables-reconciliation.md:105-106` states the false deviation-3 premise. The validation schema allows PASS with a failing test (6b) |
| D05 | Partially | No model, no fuzzy join, broad `except` only to exit 9 (`cli.py:194`). But an unmappable notes exception is an info finding that still permits PASS (B4) |
| D06 | Partially | My fast tier (node_modules present): **1353 passed, 0 failed, 0 skipped, 47 deselected, 208.33 s**. Legacy tier: 555 passed, 45 skipped, all environmental (41 "repo venv python not available" from a hard-coded `/home/abhi/sourceCode/...` path; 4 "_stat_leaders.json not present"). The skips were not recorded. Fast-tier budget: measured +38 s *serial*, but design §10 requires a four-worker A/B within +10 s. It is unmeasured as specified, and the miss is not listed in `completion.json` open_items (F-M3) |
| D07 | Partially | Tests cover lock, resume, corrupt unit and failing write. My adversarial probes fail on cache containment, output residue and revalidation-interrupt (B5, B6), and code changes outside the salt are not invalidated (F-M2) |
| D08 | Met | `inputs-before.json` and `inputs-after.json` are byte-identical (sha256 `dcff7d9c…397d`), 393,772 files |
| D09 | Met | Checkpoint: first-pass letter 26, match 17,056, notes 1, profile 13,364, season 130, stats_index 1 all `done`; robots `absent`; reval letter 26 + season 130 `done`, none changed. The "30,734 done" total includes the 156 revalidations |
| D10 | Partially | All identities hold (recomputed; local rows also close: snapshot 694,561+947=695,508, legacy 694,380+1,084=695,464). But the stream holds **18 duplicate records under 5 IDs** (F-H1). Aggregate findings lack body hash and local refs (F-M4) |
| D11 | Partially | Per-game and season/stint/career aggregates are present. Missing: counter vs games-to-date (B1), the profile counter-sequence check (B1), and no-award evidence (4,308 career BR averages `unresolved_no_award_evidence`, 6c) |
| D12 | Met | `report.json` `58eff517…db28`, `findings.jsonl` `c538a53d…aefaa4`, players `9c61605e…`, coverage `2380d9ff…`, summary `77851958…` are identical across cold-1, cold-4, warm-4 and changed. `output-manifest.json` differs per run only because it hashes the non-canonical `execution.json`. The four probes recomputed only dependent units |
| D13 | Partially | `final-runs.sh` runs every compare under `unshare -rn`; exit 4 preserved. Missed targets: RSS 2150.6 > 2048 MiB and warm 122.5 > 120 s, both stated. The fast-tier budget miss is not stated (F-M3) |
| D14 | Met | `latest_completed_final`: `games/2026/081920260926.html`, snapshot matched 46, legacy missing 46 (traced, T3 below) |
| D15 | Not met | This survey; QA and Gaffer pending |
| D16 | Not met | Snapshot FAIL (traced, e.g. T1) |
| D17 | Separate claim met; app not confirmed | Checker report `46d77e23…` PASS (not re-run by me); D16 fails |
| D18 | Not met | Legacy FAIL (2026 GF absent, T3) |

## Recomputed hashes and identities (cold-4)

- Stream: sha256 recomputed `c538a53d01d182e2c063bf1372b67e0ee2c37fe79973b363f85554dc67aefaa4`, 195,668 lines. Per-(layer, category, severity) counts equal `report.findings.by_layer_category_severity`.
- Identities, both layers: `cells_expected = 23 × app_expected = 15,996,477`. Source states sum to 15,996,477.
  `app_matched+missing+unresolved = 695,499`. Career, season and stint outcome partitions are exact. `cell_equal = equal_value + equal_zero`.
  `cell_recorded_zero (2,931,420) = equal_zero + mismatch_local_null` (snapshot).
- Note: `cell_unsupported_local_numeric` (435) is a sub-count *inside* `cell_source_unavailable` (`season.py:269`).
  A naive 9-bucket sum overshoots by exactly 435. Document it in the report (F-L2).

## Traced findings (raw captured HTML → local row)

| # | Finding | Source evidence (object by sha) | Local evidence | Result |
|---|---|---|---|---|
| T1 | CELL_MISMATCH legacy, Stoneham 1976 R7 kicks/disposals | profile `7def6082…` table 12 row 5: KI 10, DI 12, Rd→`070919760515`. Match `88f779e9…` Footscray row 17: KI 10, DI 12 (R-BOTH-PAGES). Player Details "57 (25-2-30 …)" = profile Gm 57 | `data/player_data/stoneham_alan_20081955_performance_details.csv` row 57: kicks 11, disposals 13. Snapshot row also 11/13 | Correct |
| T2 | CELL_LOCAL_NULL snapshot, Shattock 2000 R16 behinds | profile `6264e767…` table 8 row 9 and match `4706fe1e…` Brisbane row 20: **all 23 cells blank**; team Totals BH "10" | snapshot fragment `1b48a2044d9fa3f1` row 5671: behinds None (all stats None) | Correct per design (DNTF cannot fire: no %P in 2000). See watch list W1 |
| T3 | APPEARANCE_MISSING_LOCAL legacy, Brayshaw 2026 GF | profile `78dac584…` table 15 row 27: Gm 197, Rd GF→`081920260926`. Match `78749199…` Fremantle row 3; Player Details "197 (105-2-90 …)" | legacy CSV has 196 rows ending 2026 PF v Sydney (196) | Correct |
| T4 | AGGREGATE_MISMATCH snapshot+legacy, Stoneham career KI 2640/2641, DI 3765/3766 | profile Totals footer KI 2640, DI 3765; per-game sums 2640, 3765 | legacy sums kicks 2641, disposals 3766 | Correct (derived from T1) |
| T5 | CELL_UNRESOLVED legacy, Aaron Black0 2013 R13 bounces | profile `c0bca34c…` table 10 row 8 BO blank; match `463596c5…` North Melbourne row 5 BO blank; team Totals BO blank | legacy row #12 | Correct (R-ALL-ZERO-UNPROVEN) |
| T6 | APPEARANCE_QUARANTINED, McEvoy 2010 GF replay | match `bed9d426…` St Kilda row 15 | `q:64e924f03c961e1c43b4181f`, candidates `m:2010:gf:…:0/1` | Correct. The career clearances 456/455 AGGREGATE_MISMATCH follows from it (no inner join) |

## Ranked findings

### B1 — Deviation 3 is false: match pages print career games-to-date; the S-07 counter check is missing  [BLOCKING] [class NEW/4]
- Evidence [obs]: every captured match page sampled (686 of 686, every 25th page per season, 1890s–2020s) contains a
  "`<Team> Player Details`" table with "Career Games (W-D-L W%)". Examples: 1976 `88f779e9…` row "57 (25-2-30 45.61%)" for
  Stoneham (profile Gm 57), and 2026 GF `78749199…` "197 (105-2-90 53.81%)" for Brayshaw (profile Gm 197).
  `read_match` (`source.py:398-560`) reads only "Match Statistics" tables. The doc repeats the false claim at
  `docs/afltables-reconciliation.md:105-106`, and RUN.md §10 item 3 does too. No profile counter-sequence check exists
  (no grep hit for a sequence or gap check in `reconciliation/`), although DESIGN §7 requires one.
- Impact: a required SOURCE_CONFLICT class (design §8 S-07, T14) is silently absent. A source counter error cannot block PASS.
- Owner: Scientist.
- Outcome: the reader extracts career games-to-date (and career W-D-L) per lineup player. A disagreement with the profile row counter is a
  `SOURCE_CONFLICT` (`R-SOURCE-COUNTER`). A profile counter that is non-contiguous or duplicated across the career is a source finding.
  The doc and RUN.md §10 are corrected.
- Failing tests to add first: (a) `read_match(fixtures/reconciliation/match_2021_r6.html)` yields a games-to-date for every lineup
  player with a profile link. (b) In an e2e world, set one Player Details value to profile counter − 1 and expect a
  `SOURCE_CONFLICT` with field `counter`, layer verdict UNKNOWN (not PASS). (c) A profile with counters 1,2,4 yields a source finding.
- Effort M · rank 1.

### B2 — Evidence rules over-assert "proven zero" where the capture proves the record incomplete (6e and notes precedence)  [BLOCKING] [class 4/NEW]
- Evidence, 1931–34 Brownlow [obs]: match pages print per-game BR and team totals in 1931–34, so BR is
  two-sided there. The report shows no BR SOURCE_CONFLICT at all, which makes the "one-sided" question moot. But **24 H&A
  matches 1931–34 print votes that do not sum to 6** (12 none, 6 four, 3 three, 2 five, 1 one). `cells.py:151-153`
  (`R-TOTAL-NONBLANK`) still marks blank player BR as RECORDED_ZERO wherever that team's total is non-blank. That yields
  **319 snapshot CELL_LOCAL_NULL BR findings inside those partial matches**. The source contradicts itself:
  18 `season_total` BR inconsistencies, e.g. Bert Mills 1932 "printed '5' game cells sum 2", so a blank in that season
  is provably not zero for him.
- Evidence, notes precedence [obs]: in 11 exception-listed team-matches the team total is non-blank
  (1975 R11 all-but-goals ×3, 1977 behinds ×6, 1978 behinds ×2). `_blank_state` checks the non-blank total before
  `R-NOTES-EXCEPTION` (`cells.py:151-176`), so blanks become RECORDED_ZERO. That is 41 snapshot CELL_LOCAL_NULL behinds findings in
  8 of those team-matches. The source's own averages exclude those games (29 behinds 1975 season-average misses, e.g.
  Noonan "printed 2.19 model 46/22"; 46/21 fits [inf]).
- Phase E (design §12) required the 1931–34 BR state to be "decided from captured evidence". RUN.md §14 records it as
  undecided ("one-sided rule not adopted") and misdescribes the shape.
- Impact: wrong FAIL findings, a few hundred out of 36,573. The FAIL verdict is unaffected. It is still a rule asserting
  facts the evidence contradicts.
- Owner: Scientist.
- Outcome: (1) For BR from 1931, a blank player cell is RECORDED_ZERO only when the two team totals sum to the award
  total for that match. Otherwise it is UNRESOLVED_BLANK. The award total is 6, and 12 in 1976–77 per the source page in 6c
  (that page must be captured as evidence first). (2) A notes exception for (season, round, team, category) takes precedence over a
  non-blank team total: NOT_RECORDED for blank cells, with printed player values kept as values. (3) The 1931–34 decision and
  its corpus measurement are written into the rules file with a locator, plus an integration test over the frozen capture
  that re-derives "sum ≠ 6 matches = 24".
- Failing tests: a synthetic 1932 match whose totals are "3" and "" → a blank BR cell on the "3" side is `UNRESOLVED_BLANK`,
  not ZERO. A synthetic 1975 R11 match with an all-but-goals exception and Totals BH "12" → a blank BH cell is `NOT_RECORDED`
  with rule `R-NOTES-EXCEPTION`.
- Effort S–M · rank 2.

### B3 — S-08 not implemented: the unmappable notes club is an info finding and PASS is still allowed  [BLOCKING; defective gate → escalate] [class 4/10]
- Evidence [obs]: config has no `notes_club_alias` (`config/reconciliation_rules.toml`).
  `compare.py:533-553` emits `NOTES_CLUB_UNMAPPED` with severity info. The test
  `test_reconciliation_e2e.py:406-417` asserts `code == 0`, i.e. PASS with an unmapped exception. In the run: notes
  "1975 R14 | Sydney, Hawthorn, Essendon" (hitouts). Hawthorn and Essendon get `R-NOTES-EXCEPTION`. South Melbourne's match
  `101619750705` is left with **19 CELL_UNRESOLVED hit-out cells** per layer, which is exactly the probe's 19-per-layer delta.
  DESIGN §8 names this exact case and says "an exception row that cannot be mapped is a schema gap, never a silent drop".
  `parse_rules` also accepts aliases with no evidence locator or reason.
- Owner: Scientist.
- Outcome: (1) Add the 1975 Sydney→South Melbourne alias with evidence locator and reason (both mandatory in the loader).
  (2) An unmapped name is `SCHEMA_GAP` (source, unknown) that blocks PASS and `comparison_complete`. (3) Invert the test:
  it must assert exit 8 and a SCHEMA_GAP.
- Effort S · rank 3.

### B4 — Output and cache containment: `--cache` inside an input root is accepted (exit 0); a refused `--out` leaves residue  [BLOCKING] [class 8/NEW, T23]
- Evidence [obs], scratch adversarial tests on synthetic worlds:
  (a) `--cache <data_root>/cachehere` gave exit 0, overall PASS, and **85 new entries written inside the snapshot data root**.
  (b) `--out <capture>/evil` was refused with exit 2, but `.evil.work-*/chunks/*.jsonl` was **left inside the capture directory**.
  Cause: `compare.py:296-306` checks `--out` only against `data_root`, creates `mkdtemp` in `out.parent` before
  `check_destination` runs, and never checks `cache_root`. `run_compare_cli` skips `cleanup` on `OutputWriteError`
  (`compare.py:1486-1495`). `_legacy_shards` also unlinks `*.jsonl` under `cache_root/work/legacy` (`compare.py:728-731`).
- Owner: Scientist.
- Outcome: `--out`, the work dir and `--cache` are all validated against every input root (data, capture, legacy dirs, plan
  dir) before any write. Refusal leaves zero bytes in any input root. Both probes become unit tests (T23).
- Effort S · rank 4.

### B5 — The validation record can say PASS for a failing step; completion is hand-typed outside the repo (owner item 6b)  [BLOCKING, owner-designated] [class 4/6]
- Evidence [obs]: `implementation-validation.json` has step "scvia fast tier" with status PASS and detail "1352 passed, 1 failed".
  `ValidationStep` (`schema.py:327-331`) has a free `status` and free-text `detail`. The writer is
  `var/reconciliations/afltables/tools/make_completion.py`, which is outside Git, untested and hard-codes `status="PASS"`
  for every step. `completion.implementation=COMPLETE` is also hard-coded. The legacy-tier skips are unlisted.
- Owner: Scientist (schema and writer). Gaffer (record placement).
- Outcome: `ValidationStep` carries structured `counts{passed,failed,errors,skipped,deselected}` and `skip_reasons`.
  A model validator rejects `status=PASS` unless failed and errors are both 0. A baseline failure is `status=FAIL` plus a `baseline_evidence`
  entry (same failure at base commit, same environment). `Completion.implementation` may be `COMPLETE` only if every required
  step is PASS. The writer is committed under `scripts/` with tests, and the record is regenerated from real pytest JSON output.
- Failing tests: `ValidationStep(status="PASS", counts={"failed":1,…})` raises. A `Completion` with implementation COMPLETE
  referencing a validation containing a FAIL step raises.
- 6a: with `web/node_modules` present I measured the fast tier at 1353 passed and 0 failed, with nothing else
  environment-skipped. The legacy tier's 45 skips are environmental and pre-existing; record them, don't fix them here.
- Effort S · rank 5.

### B6 — Deviation 2: the manifest claims `capture_complete: true` after an interrupted revalidation  [BLOCKING for merge; not for this run's data] [class 2]
- Evidence [obs]: `build_manifest` skips every `reval` row (`capture.py:820-823`), and `revalidation_started` is set before any
  revalidation fetch (`capture.py:725`). Reproduced in scratch: interrupting after the first revalidation accept gives receipt `interrupted` and
  manifest `capture_complete: true`. `compare` guards via `receipt.json` (`compare.py:287-294`), but the receipt is
  not hash-bound into the report and records the superseded `plan_id fd6bc85d…`. **This run is unaffected:** the checkpoint
  shows 156/156 revalidation tasks `done`, with no change reason.
- Owner: Scientist.
- Outcome: the manifest counts pending or failed reval tasks as incomplete, and a `revalidation_finished` marker is required. Compare
  binds the `receipt.json` sha and the checkpoint-derived revalidation tally into the report.
- **Sequencing that preserves the frozen capture:** `capture.py` is in `CAPTURE_FILES` (`inventory.py:32-39`), so editing it
  changes `capture_identity` for any newly built plan. `compare.py:271` would then refuse the frozen manifest. Therefore:
  (1) land B1–B5 and F-H1 (none touch `capture.py`, `discover.py`, `urls.py`, `http.py` or `sourcepages.py`); (2) re-issue
  the plan (same `capture_identity 9f2cbee4…`) and re-run the four compares and probes offline; (3) only then land the
  `capture.py` fix. Compare does not re-hash current capture code, so the accepted run stays reproducible with its own
  `plan.json`. If B1 needs `sourcepages.py` changes, put them in `source.py` instead or the identity breaks.
- Failing test: my scratch test `test_interrupt_during_revalidation_manifest_claims_complete` (stop after the first
  `_accept_revalidation`; assert `manifest.capture_complete is False`).
- Effort S · rank 6.

### Owner items 6c and 6d (assessed; BLOCKING only as owner-designated)

**6c — Brownlow no-award evidence.** [obs] Two bounded GETs, 9 s apart, descriptive UA:
1. `2026-10-02T23:37:47Z` `https://afltables.com/afl/afl_index.html` → 200, 4,370 B, sha256
   `c9f39ba8175b3307153e73320bd17447ee5e906265c7dd0c469d68a0c5975e15`. It links "Brownlow Medal" → `brownlow/brownlow_idx.html`.
2. `2026-10-02T23:37:56Z` `https://afltables.com/afl/brownlow/brownlow_idx.html` → 200, 67,575 B, sha256
   `dc81249c48524329ec024f29283d0524742a66da799a886b91c26d8f2aa98725`, Last-Modified `Mon, 21 Sep 2026 12:50:10 GMT`. A footer row
   (`<td colspan=10>`) reads: "No Medal awarded 1942-1945 due to WWII … Voting systems: 1924-1930: One vote per game …
   1931-present: Six votes per game: 3,2,1; (except 1976-77: 12 votes per game: 3,2,1 from each of two field umpires)".

These are identification aids only; my bytes are not audit evidence. The path is **not** allowed by
`config/reconciliation_source_policy.toml`, and changing that file changes `capture_identity`, which would orphan the frozen capture.

- Owner: Scientist.
- Outcome: a separate, committed evidence-capture path. It uses its own one-line exact policy (`^/afl/brownlow/brownlow_idx\.html$`;
  the uncommitted `tools/evidence_fetch.py` uses a broad `/afl/brownlow/.*` pattern), its own object store, and an observation
  record (URL, status, headers, sha, timestamp).
- The rules file's `[brownlow]` gains `evidence_url` and `evidence_sha256`. The `no_award_seasons` and award totals are *parsed*
  from the stored body by a tested reader, not typed in.
- Rules are pinned in `plan_id` but not in `capture_identity` (`inventory.py:248-257`). So re-issuing the plan keeps the
  30,734-record capture valid.
- The unit digest's `no_award` slice (`compare.py:846`) must include the evidence sha.
- Report its observation time separately from the acquisition window. It is a historical fact, unaffected by the event
  boundary; Last-Modified predates the window.
- The page also supplies the source locator for R-BR-SIX-PER-MATCH and the 1976–77 exception that B2 needs.
- Test: a fixture copy of the page parses to `{1942,1943,1944,1945}`. A tampered body (sha mismatch) makes compare exit 2.

**6d — Performance.** These figures are evidence, not targets met.
- cold-4 [obs]: tree peak 2,150.6 MiB over 6 processes. The parent's maxrss goes 1,015.6 MiB after identity, then
  1,676.8 MiB after season_units, i.e. the parent is about 78% of peak [inf].
- Retained memory [obs, tracemalloc, 20 of 260 unit pickles, ×13 extrapolation]: retained `SeasonResult`s ≈ 918 MiB, of which
  `club_seasons` (`ClubSeasonAgg` with 23 `StatAcc` objects plus five 23-element `Decimal` lists each) ≈ 867 MiB. They are all held in
  the parent until `reduce` (`compare.py:936-937`, `950-952`).
- Warm-4 phases [obs]: verify 2.4, parse 2.1, inventory 2.1, identity 18.5, season_units 50.4 (all 260 units cached),
  reduce 41.3 s. Read-only micro-timings, taken under load [obs]: reading all snapshot `player_games` takes 6.0 s, and the warm path does it twice
  (`compare.py:598-604`, `777`). Canonical-serialising and hashing every row for digests takes 4.0 s per layer
  (`compare.py:851-853`). A full legacy CSV parse takes 8.4 s, done twice (`identity_legacy` and `_legacy_shards`) plus a shard
  rewrite every run. `reduce` is single-threaded in the parent, and `load_core` ×13,364 takes 3.1 s per layer.
- Coverage-preserving fixes, in order of gain:
  - (1) Do not keep `club_seasons` in the parent. Spill per unit (already in the unit pickle) and run `reduce` in the worker
    pool, partitioned by profile-hash bucket. This should remove most of about 0.87 GiB and parallelise the 41 s.
  - (2) Read each local input once, and key unit digests by fragment or CSV content hashes plus an identity/match-map slice
    digest instead of re-serialising rows.
  - (3) Key the legacy shards by `legacy.content_sha256` plus an identity digest and skip the rewrite when present.
  - (4) Load the profile core once for both layers.
- Acceptance: re-measured `measure_process_tree` cold-4 ≤ 2,048 MiB and warm ≤ 120 s on the frozen capture, with byte-identical
  outputs and an unchanged unit-digest test.
- Owner: Scientist. Effort M.

### F-H1 — Duplicate finding IDs inflate PLAYER_MISSING_LOCAL  [HIGH] [class 6]
- Evidence [obs]: 195,650 distinct IDs in 195,668 records. 5 IDs are repeated, all `PLAYER_MISSING_LOCAL`, e.g. Brian_Roberts ×5
  per layer. The cause is a per-season emission guard (`season.py:802-816`, `res.emitted` is per unit).
  Snapshot 16 records cover 5 profiles (= `population.source_players_missing_locally`). RUN.md §3 reports 16/12 as if they were players.
- Owner: Scientist. Outcome: emit once per (layer, profile) in the parent, and add a unit test asserting unique IDs over the stream.
  It rides the B-batch re-run. Effort S.

### F-H2 — Deviation 6 is a material semantic change: about 674k mismatched row dates were demoted to information  [HIGH; owner decision → escalate] [class NEW]
- Evidence [obs]: `attr_row_date_unverified_mismatch` is snapshot 674,232 and legacy 674,150. The typical offset is −25 to −62 days
  (e.g. "source 2013-04-28 local 2013-03-29").
- Code (`season.py:591-598`): **every** legacy date mismatch is demoted, not only the declared ones. Snapshot mismatches are demoted
  when `date_quality != fixture_verified`.
- RUN.md §3 says "about 57,400 … in the legacy layer", which counts grouped findings and omits the snapshot.
- Owner: Gaffer (amendment and record). Scientist (correct the RUN wording; report row counts in the headline).
- Outcome: the owner decides, and a design amendment records the decision with its trade-off. Then report row counts per layer explicitly.

### F-M1 — Deviation 8 (R-BR-SIX-PER-MATCH) is sound and correctly scoped but its evidence is prose  [MEDIUM] [class 6/10]
- Evidence [obs], my independent re-measure from cached facts:
  - 7,620 H&A matches 1984–2026, 7,620 summing to 6, 3,858 with one team blank.
  - 66 drawn H&A matches, all summing to 6. No H&A match is missing a totals row or column.
  - Finals ≥1984: 364 with both totals blank; the rule excludes finals (`cells.py:158`).
  - Pre-1984 totals are blank, so the rule cannot fire; the 1984 boundary is safe.
  - It fires only when the opposition total equals "6".
  - It produced 83,565 cells (`srcrule_RECORDED_ZERO:R-BR-SIX-PER-MATCH`), 78 of them now snapshot CELL_LOCAL_NULL. RUN.md quotes 14,756 BR nulls; the stream has 14,756 + 78.
- Ruling: routine within design mechanism. It is a "substantive rule", so it needs a recorded architect sign-off (§7).
- Owner: Scientist. Outcome: an integration test re-derives the 7,620/7,620 claim from the frozen capture. The rule cites the
  captured Brownlow page (6c). Lowering `first_season` must be refused for 1976–77 (total 12).

### F-M2 — Cache keys omit code that shapes cached results  [MEDIUM] [class NEW]
- Evidence [obs]:
  - The unit salt (`compare.py:818-826`) hashes season, cells, findings, source and schema `.py`. But `season.py:19` builds
    `ClubSeasonAgg` from `aggregate.StatAcc`, and `rules.py`'s `club_alias` logic is also used. Edits to either reuse stale units.
  - `PARSER_FILES` (`cache.py:25-30`) omits `facts.py`, which shapes cached headers and per-season splits.
  - The `cache.py` docstring still says comparison is never cached.
- Owner: Scientist. Outcome: the salt and parser identity cover every module whose code shapes the cached payload. A test edits
  `aggregate.py` bytes (via a monkeypatched salt source) and expects 0 units reused.

### F-M3 — Fast-tier budget unmeasured as specified, and the miss is unreported  [MEDIUM] [class 1]
- Evidence: the design §10 four-worker +10 s budget. The validation measured serial +38 s, `pytest-xdist 3.8.0` is installed, and
  `completion.json` open_items omit it.
- Owner: Scientist (measure `-n 4` A/B). Gaffer (list it as an open target).

### F-M4 — Aggregate findings lack the evidence-locator contract  [MEDIUM]
- Evidence: AGGREGATE_MISMATCH and LOCAL_MISSING_SUMMARY_VALUE evidence is `{"locator":"career   disposals","source_url":…}`, with no
  `body_sha256` and `local: {}` (DESIGN §9 requires a body hash and the local file/row).
- Owner: Scientist. Outcome: include the profile body sha, the table/footer locator and the contributing local origins or fragment set.

### F-L1 — Smaller gaps  [LOW]
- `unknown_reasons` lists only career-level unresolved aggregates (`compare.py:1155`).
- Season-average misses list no candidate denominators (`reduce.py:310-320`).
- There is no explicit DNTF pilot-review count outside 2021–22 (coverage shows 2003: 1, 2004: 2, 2023–25: 5 each).
- Rule 3 resolves without corroboration when the local player has no memberships (`identity.py` `if not ms or …`).
- A legacy row with an unparseable year becomes season −1 and is never placed or counted as unplaced (`local.py` `-1`).
- The local-row partition holds but is not asserted in `_identities`.
- The unit digest uses `len(captured)` rather than membership (`compare.py:856`).

### F-L2 — Deviations 1, 4, 5 and 7  [LOW]
- Deviation 1: routine.
- Deviation 4: routine (the design specifies interval comparison; ties print both ways).
- Deviation 5: routine; document the sub-count relationships, including unsupported ⊂ source-unavailable.
- Deviation 7: process note for Gaffer; tests now exist.
- Out-of-tree tools (`make_completion.py`, `tree_inventory.py`, `evidence_fetch.py`) produced acceptance evidence. Commit or archive them with hashes (Gaffer).

## Deviation rulings (RUN.md §10)

| # | Ruling |
|---|---|
| 1 plan_id vs capture_identity | Routine within design |
| 2 manifest complete during revalidation | **Needs code fix** (B6); this run's evidence is unaffected (156/156 reval done) |
| 3 no games-to-date on match pages | **Needs code fix**: the premise is false (B1) |
| 4 averages on closed interval / recorded denominators | Routine within design (add season-miss candidates, F-L1) |
| 5 extra cell buckets | Routine; document it |
| 6 row-date demotion | **Material semantic change**: design amendment and owner decision (F-H2) |
| 7 capture.py not test-first | Process note (Gaffer); no code change |
| 8 R-BR-SIX-PER-MATCH | Routine mechanism, sound evidence, correct scope; needs architect sign-off and a reproducible evidence test (F-M1) |

## Escalations to the human (direct)
1. A defective gate passes content it should fail: an unmapped notes exception yields PASS, and a test enshrines it (B3).
2. Already-published claims in RUN.md are wrong:
   - deviation 3's premise
   - "PLAYER_MISSING_LOCAL 16/12" (5 profiles per layer)
   - the 1931–34 BR description
   - row-date counts
   - BR null count
   Gaffer must not package RUN.md numbers verbatim.
3. Deviation 6 is an owner decision (a classification that removes about 674k comparisons from the verdict).

## Anti-pattern list (standing)
- (seed list retained) Never trust an LLM sum; never verify by exit code; never `git add .`; never hand-edit `data/`;
  never let Pass-1 stand for Pass-2; never soften upstream caveats; never refresh before settlement; never parallel-push.
- Never define a role, model or tool scope in two places. Never leave a gate convention unwritten.
- **ADD:** never record a validation step's status by hand. Status is computed from structured counts; free-text "1 failed" under PASS is the evidence (B5).
- **ADD:** never accept a deviation whose premise is a claim about source pages without a corpus measurement. One grep over the frozen capture refuted deviation 3 (B1).
- **ADD:** an evidence rule that asserts zero must check the evidence that would contradict it (vote sums, notes exceptions, the source's own averages) (B2).

## Watch list (speculation, unranked)
- W1 [spec]: all-blank player rows in pre-%P eras are RECORDED_ZERO by design (T2). Their share of the 36,573 CELL_LOCAL_NULL is
  unmeasured. If large, the owner's "null vs zero" decision should consider it separately.
- W2 [spec]: 165 hit-out season-average misses in 1976–78 (e.g. Goad 1976: every team total non-blank, but the source's
  denominator is 22 rather than 23) suggest an undocumented per-game availability convention. Investigate before any rule changes.
- W3 [spec]: the 15 DNTF appearances in 2023–25 fall outside the observed 2021–22 convention.

## What was checked and found clean
- Independence: no imports of `ingest.afltables`, blank resolvers or analytics into `reconciliation/`. It uses only
  `integrity.sourcepages` (the designated tokenizer, plus private `_local_start` and `FINAL_TOKENS`), `ingest.http` transport, storage and serialisation.
- Additive edits to `http.py` (`retry_after_s`) and `sourcepages.py` (`rowspan=1`).
- Plan-pinned capture files equal the current worktree hashes: `capture.py 0b4dab01…`, `http.py cc1fb39b…`, `sourcepages.py 08d23e7a…`, `discover.py 5d26d58e…`, `urls.py 961ca7f6…`, `__init__ 3c10139a…`.
- Config copies under `config/` and `src/supercoach_via/config/` are byte-identical.
- Local drift recheck at finalize. Stream merge is deterministic. Output manifest is written last. Hard-link and symlink outputs are refused.
- Local-row accounting closes in both layers. The D08 inventories are identical. The no-fetch probe originals are unchanged per the probes summary (not re-hashed by me).
