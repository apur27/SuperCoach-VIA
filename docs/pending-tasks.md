# Fix Plan — 2026-07-07 — Surveyor

Source: Survey 1 (full-repo health, 18 findings, IDs F01–F18) and Survey 2 (deep-read, 7 findings, IDs S1–S7), both 2026-07-07. Prepared by Surveyor; owned and executed by the named agents; Gaffer sequences and ships.

**Escalations already with the human:** S1 (a published "venue effects" claim is false) and S2 (conceded-stats file is corrupt). Their correction items lead their sprints.

**Standing rule for this plan:** no item marked *Blocked by decision* may be started, partially started, or worked around until the human records the decision. The three open decisions are restated at the end of this document; they are human editorial/basis calls, not agent calls.

**If Sprint 2 must be cut to fit one session,** ship in this order and let the rest slip to the top of Sprint 3: F03, F14a, F12, F07, then F02a → F02. Gate repairs before surface repairs.

---

## Sprint 2 — Correctness, gates, and wrong-data prevention

### [Sprint 2] F01 (CR-1) — Verify the round-detection fix and pin it with a regression test
**Owner:** Scientist
**Depends on:** none
**Blocked by decision:** none
**Fix brief:** The lexicographic round-detection bug was fixed this cycle; this item is verification, not re-implementation. Confirm the fix is on `main` (not only in a working tree), then add a unit test in `tests/unit/` that feeds the exact failure shape (e.g. "Round 9" vs "Round 10" ordering) and asserts numeric round comparison. Touch only the test file unless the fix turns out not to have landed, in which case land it with the test per TDD.
**Acceptance criterion:** A committed failing-then-passing test exists that would have caught the original lexicographic bug, and the full suite passes.

### [Sprint 2] F03 (CR-2) — Restore QA gate authority
**Owner:** Gaffer
**Depends on:** none
**Blocked by decision:** none
**Fix brief:** The QA gate KeyErrors when run against the real pipeline JSON schema, which means the gate has never issued a verdict grounded in actual output — a defective gate. Reconcile the QA agent's validation logic (`.claude/agents/QA.md` and any helper script it invokes) with the actual schema of the artifacts it checks, sampled from a real refresh run, not from documentation. Every field access must either exist in the real schema or be an explicit, reported check failure — never an unhandled exception. Scientist executes any Python changes; per CLAUDE.md TDD, changes ship with tests against a real captured artifact fixture.
**Acceptance criterion:** QA runs end-to-end against the most recent real pipeline outputs with zero unhandled exceptions and produces a PASS/FAIL verdict whose checks reference fields that actually exist.

### [Sprint 2] F14a — Fail-open pre-commit hook must fail closed
**Owner:** Gaffer
**Depends on:** none
**Blocked by decision:** none
**Fix brief:** Split out of the F14 hygiene bundle because it is gate integrity, not hygiene: an enforcement hook that fails open means any error in the hook silently waves commits through. Audit `.githooks/pre-commit` (and the scripts it calls: `check-council-stamp.sh`, staged-blob gate) for every error path — missing file, malformed JSON, non-zero helper exit — and make each one block the commit with a clear message rather than pass. Ship with a test in `tests/unit/` that simulates a broken audit record and asserts the hook rejects.
**Acceptance criterion:** Deliberately corrupting an audit record or removing a helper script causes the pre-commit hook to block with an explanatory error, demonstrated by a committed test.

### [Sprint 2] F12 — Wire phantom_row_validator into the pipeline
**Owner:** Gaffer
**Depends on:** none (soft-order after F04 if both land in the same session, so it wires into the surviving entry point)
**Blocked by decision:** none
**Fix brief:** `scripts/phantom_row_validator.py` exists and is tested (`tests/unit/test_phantom_row_validator.py`) but nothing invokes it — a built gate wired into nothing. Add an invocation in `scripts/weekly_refresh.sh` immediately after the scrape/refresh phase writes player CSVs, before anything downstream reads them, with a non-zero exit aborting the run. The validator itself needs no changes; this is harness wiring only.
**Acceptance criterion:** `grep phantom_row_validator scripts/weekly_refresh.sh` shows the call in the post-scrape phase, and a dry-run of the refresh demonstrates the phase executes and its verdict is logged.

### [Sprint 2] F07 — Verdict vocabulary unification and Skeptic records
**Owner:** Gaffer
**Depends on:** none
**Blocked by decision:** none
**Fix brief:** The council's gates speak different verdict languages (Skeptic: PASS / PASS_WITH_CONCERNS / BLOCK; DataSentinel and the hook layer: PASS / FAIL), and Skeptic verdicts are not recorded to `.claude/audit/` at all, so the "both gates PASS before ship" rule is unauditable for one of the two gates. Define one canonical verdict vocabulary and the mapping for each gate, update the Skeptic and DataSentinel prompt contracts (`.claude/agents/Skeptic.md`, `.claude/agents/DataSentinel.md`) and `record-sentinel-verdict.sh` so Skeptic verdicts are recorded with the same content-hash discipline DataSentinel already has. Do not alter any verdict already on record.
**Acceptance criterion:** The next Skeptic-gated doc has a Skeptic verdict record in `.claude/audit/` alongside its DataSentinel record, both using the documented vocabulary.

### [Sprint 2] F05 (CR-4) — Gate the insights lane
**Owner:** Gaffer
**Depends on:** F07 (record the new lane's verdicts in the unified vocabulary)
**Blocked by decision:** none
**Fix brief:** `docs/afl-insights.md` is written by an LLM phase of `scripts/weekly_refresh.sh` and committed by the harness with no DataSentinel pass — the only regularly-published LLM prose lane with no gate. Route the insights update through the same discipline as news docs: `[data]` tags on every number, DataSentinel verification, verdict recorded before the harness stages the file. If full gating is too heavy for a weekly automated lane, the alternative is to strip all specific numbers from the prose and link to the generated tables — but do not leave numeric LLM prose ungated.
**Acceptance criterion:** The next weekly refresh either produces an insights update with a recorded DataSentinel PASS, or produces one containing no ungated specific numbers, and `weekly_refresh.sh` enforces whichever mode was chosen.

### [Sprint 2] F04 — Single-entry-point refresh discipline
**Owner:** Gaffer
**Depends on:** none
**Blocked by decision:** none
**Fix brief:** Three overlapping entry points exist (`scripts/weekly_refresh.sh`, `refresh_and_rank.sh`, `refresh_data.py`) with divergent phase coverage. Declare `weekly_refresh.sh` the sole documented entry point; make the other two either delegate to it, refuse to run interactively without an explicit override flag, or be clearly marked as internal phases it calls. Update README/architecture references so no doc instructs a human or agent to run a partial path. Do not delete anything; demote and redirect.
**Acceptance criterion:** Exactly one entry point is documented, and invoking either legacy path without the override flag prints a redirect message instead of running a partial refresh.

### [Sprint 2] F11 — stat_coverage_eras.yaml missing three stats
**Owner:** Scientist
**Depends on:** none (but Decision 3's helper will consume this file — land F11 first)
**Blocked by decision:** none
**Fix brief:** `config/stat_coverage_eras.yaml` omits three tracked stats, so any era-coverage logic silently treats them as always-recorded. Add the three stats with their correct first-recorded years, verified against the actual data (earliest non-null year per column across `data/player_data/`, computed in pandas, not asserted from memory), with a test asserting the config covers every stat column the pipeline reads.
**Acceptance criterion:** Every stat column consumed by the pipeline has an era entry in the YAML, each first-recorded year is backed by a quoted pandas measurement, and the covering test passes.

### [Sprint 2] F02a — As-of-date verification mode for DataSentinel
**Owner:** Gaffer (directive parse + prompt rule) + Scientist (round-cap compute)
**Depends on:** F07 (verdicts recorded uniformly)
**Blocked by decision:** Decision 1 AND Decision 2 (first consumers define its semantics; pending-decisions.md item 4 explicitly depends on decisions 1–3)
**Fix brief:** Build the machine-readable `<!-- verify-asof: round=N -->` directive per the design in `docs/pending-decisions.md` item 4: directive is part of the content hash; badge renders the as-of visibly; cap applies to all source tables the doc's tags touch; recurring live-re-verify lane skips as-of docs; doc-level cap only (per-tag deferred). Ship with tests for hash inclusion and cap enforcement.
**Acceptance criterion:** A test doc with `verify-asof: round=9` verifies PASS against R9-capped data, FAILs if the directive is stripped, and renders a badge that visibly states the as-of round.

### [Sprint 2] F02 (CR-3) — Unstrand the five stamped docs with wrong numbers on main
**Owner:** Scientist (re-derivation) + Gaffer (ship)
**Depends on:** F02a, F11
**Blocked by decision:** Decision 1 (grand-final-strategy basis), Decision 2 (list-quality frozen vs live), Decision 3 (era-boundary inclusion — determines whether dustin-martin's threshold count is 12 or 16)
**Fix brief:** Five published, stamped docs carry numbers known to be wrong; they cannot be fixed until the human picks a data basis for each. Once decided: Scientist re-derives every affected figure from the data at the decided basis (pandas, quoted outputs, no prose arithmetic); for Decision 3, Scientist builds the deterministic era-boundary threshold helper encoding the chosen inclusion rule and emitting `N of M` alongside averages — this must be a script, not a prompt rule; docs corrected, re-verified by DataSentinel (via F02a's as-of mode where a snapshot basis was chosen), verdicts recorded, shipped by Gaffer. The docs' modified-but-uncommitted state in the working tree must be reconciled, not clobbered.
**Acceptance criterion:** All five docs are on `main` with fresh DataSentinel PASS records at their declared basis, visible as-of badges where frozen, and zero figures that fail a live re-derivation at that basis.

### [Sprint 2] F13 — Resolve the eight Round-14 briefs' FOOTYSTRATEGY INSERT placeholders
**Owner:** FootyStrategy
**Depends on:** none
**Blocked by decision:** none
**Fix brief:** Eight published briefs under `docs/coaches-strategy-corner/` still contain unresolved `<!-- FOOTYSTRATEGY INSERT -->` placeholders. FootyStrategy writes the interpretation layer for each (or a retrospective framing — its call), drafts go through Skeptic, Gaffer ships. If any brief is judged too stale to complete, the outcome is withdrawal from the published surface — but placeholders must not remain.
**Acceptance criterion:** `grep -rl 'FOOTYSTRATEGY INSERT' docs/coaches-strategy-corner/` returns nothing, and every completed brief has a Skeptic verdict on record.

### [Sprint 2] S1a — Retract or correct the false "venue effects" published claim
**Owner:** Gaffer
**Depends on:** none (do not wait for S1b — the claim is false today)
**Blocked by decision:** none
**Fix brief:** The model does not use the venue features its published description claims. Gaffer corrects the claim wherever it appears on the user surface (model description docs, README, report card) to describe what the model actually does, with Scientist confirming the corrected wording against the training code before ship. Do not soften into ambiguity; the correction states what is and is not in the model.
**Acceptance criterion:** No published doc claims venue effects (or any phantom feature) as a model input, verified by grep across `docs/` and README, and the corrected wording is confirmed by Scientist against the actual feature list in code.

---

## Sprint 3 — Model improvements and data-source repair

### [Sprint 3] S1b — Resolve the phantom model features
**Owner:** Scientist
**Depends on:** F01 (round detection verified)
**Blocked by decision:** none
**Fix brief:** For each phantom feature (cba_percent, percentage_time_played/TOG rename, venue): either wire it genuinely into training with a measured backtest delta, or delete the dead code. No feature may exist in a state where documentation, code, and the trained model disagree. Coordinate wording with S1a. TDD applies; backtest comparison numbers come from executed runs, quoted.
**Acceptance criterion:** The trained model's actual feature list, the feature-engineering code, and the published model description are identical, and any newly-wired feature carries a quoted before/after backtest metric.

### [Sprint 3] S2 — Repair the corrupt conceded-stats file and its writer
**Owner:** Scientist
**Depends on:** none (quarantine at sprint open even if rebuild takes longer)
**Blocked by decision:** none
**Fix brief:** `data/conceded_stats/team_stats_conceded_2025.csv` has scrambled columns (escalated). Step 1: quarantine. Step 2: find and fix the writer bug with a schema/column-order assertion test. Step 3: regenerate from source data and validate sample rows against independent per-match computation in pandas.
**Acceptance criterion:** The regenerated file passes a committed schema test, three spot-checked rows match independent pandas recomputation from `data/matches/` and `data/player_data/`, and the writer cannot reproduce the scramble (test proves it).

### [Sprint 3] S3 — Fix the lineup scraper (garbage since 2025 R3) and backfill
**Owner:** Scientist
**Depends on:** none (quarantine at sprint open)
**Blocked by decision:** none
**Fix brief:** The lineup scraper has been writing garbage rows since 2025 R3. Diagnose the parse break (site markup change), fix with mocked-HTTP unit tests per CLAUDE.md, purge garbage rows from 2025 R3 onward, backfill from source. Wire the fixed scraper's output through a row-sanity check so a future markup change fails loudly.
**Acceptance criterion:** Lineup files contain validated rows for every round from 2025 R3 to current, a mocked-HTTP test suite covers the parser, and a deliberately malformed page fixture causes a loud failure rather than a written row.

### [Sprint 3] S7 — Age and experience features enter the model
**Owner:** Scientist
**Depends on:** S1b (honest feature pipeline first)
**Blocked by decision:** none
**Fix brief:** Player age (derivable from DOB in filenames) and career experience (games played to date with strict temporal cutoff) never enter the model. Engineer both with the temporal-cutoff invariant (no future leakage), evaluate under existing GroupKFold-by-player regime, keep or discard based on measured backtest delta — quoted, not asserted.
**Acceptance criterion:** A committed experiment record shows age/experience features evaluated under the standard regime with quoted metrics, and the model either includes them (docs updated) or a recorded result justifies exclusion.

### [Sprint 3] F10 — all_time_top_100.csv schema divergence
**Owner:** Scientist
**Depends on:** none
**Blocked by decision:** none
**Fix brief:** Root `all_time_top_100.csv` and `data/top100/all_time_top_100.csv` have diverged schemas. Designate one canonical generated file, make the other a byproduct of the same generator or a documented symlink/copy step in the refresh, and add a test asserting the two are identical after generation.
**Acceptance criterion:** Both paths are written by one generator in the same refresh run, a test asserts schema and content equality, and no consumer reads a stale copy.

### [Sprint 3] F06 — BriefBuilder gets Bash and loses its hardcodes
**Owner:** Gaffer
**Depends on:** F16 (do in same edit session to avoid touching BriefBuilder.md twice)
**Blocked by decision:** none
**Fix brief:** BriefBuilder has no Bash tool — it is structurally pushed toward LLM arithmetic, the exact failure class the gates exist to catch. Grant Bash in `.claude/agents/BriefBuilder.md` frontmatter with an explicit instruction that every derived number comes from an executed command; replace hardcoded values with instructions to read current state from the repo.
**Acceptance criterion:** BriefBuilder.md grants Bash, contains no round/season-specific hardcoded values, and the next brief it produces has every [data] number traceable to an executed command in its methodology notes.

---

## Sprint 4 — Product expansion

### [Sprint 4] S4 — AFL Fantasy points prediction target
**Owner:** Scientist
**Depends on:** F03 (working QA gate), S1b (honest feature pipeline)
**Blocked by decision:** none
**Fix brief:** The AFL Fantasy scoring formula (kicks×3, handballs×2, marks×3, tackles×4, frees_for×1, frees_against×−3, goals×6, behinds×1, hitouts×1) uses columns already in player CSVs. Compute the target, train the same model architecture, backtest under the existing regime, and add the output as a new prediction column/surface. All accuracy claims must come from executed backtests.
**Acceptance criterion:** The weekly prediction output includes a fantasy-points column with a backtest accuracy figure derived from an executed run, and the new surface passes the QA gate.

### [Sprint 4] S5 — Floor/ceiling intervals on predictions
**Owner:** Scientist
**Depends on:** S4 (design the interval surface once for both targets)
**Blocked by decision:** none
**Fix brief:** Predictions ship as bare point estimates. Derive per-player prediction intervals (quantile regression or empirical residual bands — Scientist's call, justified in the experiment record), validate calibration on the backtest, and add floor/ceiling columns to the prediction output.
**Acceptance criterion:** Prediction output carries floor/ceiling columns whose empirical coverage on the backtest is within a stated tolerance of nominal, with the calibration measurement quoted in the experiment record.

### [Sprint 4] S6 — Revive the by-position backtest
**Owner:** Scientist
**Depends on:** S3 (repaired lineup data is the most plausible position source)
**Blocked by decision:** none
**Fix brief:** The by-position backtest reports every player as Unknown. Source position labels (verify what repaired lineup files actually contain before committing to them), populate the field through the backtest path, and re-run the breakdown.
**Acceptance criterion:** The by-position backtest reports a plausible distribution across real position labels with Unknown below an agreed small residual.

### [Sprint 4] F17 — Refresh or de-link the stale user surface
**Owner:** Gaffer
**Depends on:** S1a (model report card must not restate the corrected claim)
**Blocked by decision:** none
**Fix brief:** `docs/afl-season-2026.md` and `docs/model-report-card.md` are stale on the user surface. For each: either wire into the weekly/periodic regeneration path, or add a prominent as-of banner and remove from primary navigation. The unacceptable state is a stale page presenting as current.
**Acceptance criterion:** Both docs either regenerate automatically in the refresh or carry a visible as-of banner and are demoted from primary navigation, and neither contradicts the corrected model description.

---

## Backlog — Hygiene (schedule opportunistically; none blocks a sprint item)

### [Backlog] F08 — Architecture doc consolidation
**Owner:** Gaffer | **Depends on:** none | **Blocked by decision:** none
One canonical architecture doc; the others are pointers. Preserve the §4 failure-history section intact. **Criterion:** One canonical doc exists; others are pointers; failure-history survives verbatim.

### [Backlog] F09 — Chain-position/handoff sections for FootyStrategy and Scientist
**Owner:** Gaffer | **Depends on:** F16 (fix chain definition first) | **Blocked by decision:** none
Both prompts state their upstream input, downstream consumer, and gate obligations, citing the single canonical chain definition. **Criterion:** Both prompts have chain-position sections citing one source of truth.

### [Backlog] F14b — Harness hygiene bundle (remainder after F14a)
**Owner:** Gaffer | **Depends on:** F14a | **Blocked by decision:** none
`enforce_news_limit` edge behavior, remaining dead code paths. Cover `enforce_news_limit` with a test since it enforces a CLAUDE.md hard rule. **Criterion:** Dead code gone, `enforce_news_limit` has a committed test covering the two-entry limit and archive-first precondition.

### [Backlog] F15 — Test coverage for untested live scripts
**Owner:** Scientist | **Depends on:** F03 | **Blocked by decision:** none
Eight live Python scripts plus `weekly_refresh.sh` run with no tests. Prioritize by blast radius (scripts that write to `data/` or `docs/` first). Standing lane — chip away per session. **Criterion:** Every script that writes to `data/` or a published doc has at least happy-path plus one failure-path test.

### [Backlog] F16 — Prompt hygiene bundle
**Owner:** Gaffer | **Depends on:** none (do jointly with F06 and F09 where files overlap) | **Blocked by decision:** none
Chain defined in exactly one place in Gaffer.md; no agent prompt contains a stale file reference; no gate instruction weakened. **Criterion:** grep-verified clean across `.claude/agents/*.md`.

### [Backlog] F18 — Trust badge backfill on news docs
**Owner:** Gaffer | **Depends on:** F02 (do not badge wrong-number docs until corrected) | **Blocked by decision:** indirectly Decisions 1–3 via F02
After F02 lands, run `scripts/inject_trust_badge.py` across the news corpus for every doc with a valid verdict record. **Criterion:** Every news doc either carries a badge backed by a verdict record or is on a short documented exception list.

### [Backlog] Cleanup — stale worktrees under .claude/worktrees/
**Owner:** Gaffer | **Depends on:** none | **Blocked by decision:** none
Three stale agent worktrees total ~560 MB and pollute repo-wide greps. Remove after verifying they contain no uncommitted work. **Criterion:** `.claude/worktrees/` is empty or contains only active worktrees.

---

## The three open decisions (human calls — from docs/pending-decisions.md 2026-07-03)

These block F02 (and via it, F02a semantics and F18). No agent may resolve them; no fix may paper over them.

**Decision 1 — `5yr-grand-final-strategy` data basis.** Option A: re-derive all 18 clubs to end of the current settled round, cap DataSentinel there *(Gaffer recommends)*. Option B: freeze at R15 as an explicit dated snapshot. **Blocks:** F02 for this doc; shapes F02a.

**Decision 2 — `list-quality-draft-pipeline` frozen vs live.** Option A: freeze at the article's stated R1–9 basis with explicit as-of, cap DataSentinel at round 9 *(Gaffer recommends)*. Option B: re-derive per-player figures to live. **Blocks:** F02 for this doc; first consumer of F02a.

**Decision 3 — Era-boundary player inclusion in threshold counts.** Include (dropna over recorded games → computes 16) or Exclude (coverage threshold → doc's figure of 12). Do NOT flip DataSentinel to fill-zero (contradicts DataSentinel.md:82 and coverage-era memo). Once decided, Scientist builds the deterministic helper — must be a script, not a prompt rule. **Blocks:** F02 for dustin-martin; helper consumes F11's completed era config.

---

## Sequencing summary

```
Sprint 2:  F01 → F03 → F14a → F12 → F07 → F05 → F04 → F11
           [Decisions 1-3] → F02a → F02 → (F18 unblocks)
           F13, S1a  (independent, any time in sprint)

Sprint 3:  S2, S3 (quarantine at sprint open) → S3 feeds S6
           S1b → S7 ; F10, F06 independent

Sprint 4:  S4 → S5 ; S6 (after S3) ; F17 (after S1a)

Backlog:   F08, F09, F14b, F15, F16, F18 (F18 after F02), Cleanup
```

---

*Last updated: 2026-07-07. Prepared by Surveyor. Route questions to Gaffer.*
