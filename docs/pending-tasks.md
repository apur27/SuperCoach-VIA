# Fix Plan — 2026-07-07 — Surveyor

Source: Survey 1 (full-repo health, 18 findings, IDs F01–F18) and Survey 2 (deep-read, 7 findings, IDs S1–S7), both 2026-07-07. Prepared by Surveyor; owned and executed by the named agents; Gaffer sequences and ships.

**Escalations already with the human:** S1 (a published "venue effects" claim is false) and S2 (conceded-stats file is corrupt). Their correction items lead their sprints.

**This file is the CANONICAL backlog surface (2026-07-27).** Open items live here, not in
agent memory. Agent-memory entries are pointers/caches that reference an ID below; if memory
and this file disagree, this file wins. Anything tracked only in memory is invisible to every
other agent and drifts — that is how "F13" came to mean two unrelated things at once (the
Round-14 briefs item below, and a backtest-namespace item that existed only in Gaffer's
memory; the latter is now BL-01). New items get a `BL-nn` ID, a namespace that cannot collide
with the `F`/`S`/`A` IDs from the 2026-07-07 surveys.

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

### [Sprint 2] F-26 — Backtest silently fails and pipeline ships anyway
**Owner:** Scientist (code fixes) + Gaffer (harness gate)
**Depends on:** none
**Blocked by decision:** none
**Fix brief:** The backtest can die silently (SIGKILL from timeout, empty round window exit-0, or predictor writes nothing) and the pipeline ships anyway — it never checks the artifact was written. This is the exact failure that shipped the R18 eval-surface gap on 2026-07-07 (two killed runs; `refresh_and_rank.sh` never noticed the missing summary). Three Scientist fixes: (1) artifact gate in `refresh_and_rank.sh` after the backtest call — check a new `backtest_summary_*.csv` exists with today's timestamp, abort with FATAL if not; (2) `backtest.py:571` — change `return 0` to nonzero for an empty round window; (3) `backtest.py:365–370` — verify the picked prediction CSV was created after this predictor run started (mtime check), hard-fail if stale. One Gaffer fix: `check_backtest_freshness.py` gate wired into both harness scripts before any doc regeneration step — same pattern as `check_hof_numbers.py`. **Standing anti-pattern rule:** *never wrap `backtest.py` in `timeout N` or a bounded foreground call — background + artifact gate only* (backtest.py needs ~24 min per round of Optuna tuning; June 29 log: 22:28:49→22:53:02).
**Acceptance criterion:** A deliberately killed backtest run causes `refresh_and_rank.sh` to abort before touching any doc, demonstrated by a test; `docs/afl-backtest-2026.md` date cannot exceed its max backtested round.

---

## Sprint 2 — Agent Architecture Fixes (Surveyor prompt/memory review, 2026-07-07)

Full survey: `.claude/surveys/2026-07-07-survey.md`. All Gaffer-owned prompt/memory/skills edits. **HANDOFF CONTRACT** template (add to FootyStrategy, Scientist, BriefBuilder; Gaffer + DataSentinel have partial):
```
## HANDOFF CONTRACT
- Invoked by: <agent/skill/user> at <chain step>
- Receives: <exact inputs — paths, round, cycle type, verdicts; nothing else>
- Reads: <files opened itself; canonical configs by path>
- Produces: <artifact path + format, or verdict schema>
- Hands off to: <next agent> with <what exactly>
- On failure: <verdict/halt> — routed to <owning agent> by <who>
- Never: <top 3 scope violations for this seat>
```

### [Sprint 2] A-01 — BriefBuilder: add Bash tool + executed-computation rule
**Owner:** Gaffer · **Depends on:** F16 (same session) · **Blocked by decision:** none
BriefBuilder.md frontmatter has no Bash, yet its rule (line 70) requires means computed from raw rows — forcing in-token LLM arithmetic. Add Bash to frontmatter + rule "every derived number from an executed venv Python one-liner; never compute in-token." Remove 3 duplicate team-name memories (team_names.md, team_name_canon.md, team_name_canonicalisation.md) → one rule-memory pointing to latest backtest_by_team CSV. **Criterion:** frontmatter includes Bash; executed-computation rule present; next brief's [data] numbers traceable to a Bash command.

### [Sprint 2] A-02 — QA: fix embedded verification code to real JSON schema
**Owner:** Gaffer · **Depends on:** none · **Blocked by decision:** none
QA.md lines 50, 52, 85, 86, 121–122 reference `career_games.rank_1.total`, `leaders['career_games']['leaders'][0]`, `rank1['player_id']` — real top-level keys are `['meta','categories','single_season']`; career stats at `categories.career_games.leaders`; leader objects have `{rank,name,teams,games,total,per_game}` — **no `player_id`**. Fix snippets to real paths (glob player CSV by `name`, not id); ship with a test. Same fix as F03/CR-2. **Criterion:** snippets run without exception against real JSON; QA memory patch obsolete.

### [Sprint 2] A-03 — FootyStrategy: write its pipeline contract into the prompt
**Owner:** Gaffer · **Depends on:** none · **Blocked by decision:** none
FootyStrategy.md never mentions FOOTYSTRATEGY INSERT filling, coaches-strategy-corner, afl-insights, [data]-tag rules, or config/coach_names.txt. Add HANDOFF CONTRACT; INSERT-placeholder instructions with [data]-tag + coach-name prohibition; pointer to config/coach_names.txt; weekly-recap responsibility for afl-insights; fix description to reflect chain role. **Criterion:** grep "INSERT\|coaches-strategy-corner\|coach_names" returns hits.

### [Sprint 2] A-04 — Scientist: reconcile contradictory NaN memories + promote backtest invariant
**Owner:** Gaffer (memory) + Scientist (confirm) · **Depends on:** none · **Blocked by decision:** none
Contradictory live memories: `blank_counting_stat_means.md` (fill-zero) vs `dropna_denominator_coverage_bias.md` (dropna, Decision 3). Rewrite blank_counting to record the fill-zero *fact* but state the operative Decision-3 convention (dropna). Also promote the backtest temporal-cutoff invariant ("Violated TWICE") verbatim into Scientist.md body. **Criterion:** no contradictory NaN memories; invariant appears in Scientist.md.

### [Sprint 2] A-05 — DataSentinel: relocate its false-FAIL lessons to its own memory
**Owner:** Gaffer · **Depends on:** none · **Blocked by decision:** none
Three DataSentinel lessons live in other agents' dirs: canonical games metric (Gaffer/feedback_canonical_games_metric.md), non-chronological CSVs (Skeptic/feedback_player_csv_not_chronological.md), DOB-collision false-WARNINGs (Gaffer/project_audit_url_collision_fp.md). Copy to `.claude/agent-memory/DataSentinel/`, update its MEMORY.md; confirm bootstrap has canonical games = max(rowcount, games_played.max()), sort_values(['year','round']) before last-N, DOB/URL collision triage. **Criterion:** DataSentinel dir has the 3 lessons; index reflects them.

### [Sprint 2] A-06 — Gaffer: add Surveyor consult point to prompt; fix architecture §3
**Owner:** Gaffer · **Depends on:** F08 (but §3 chain fix small enough now) · **Blocked by decision:** none
(1) "Consult Surveyor before complex implementations" lives only in memory; Gaffer.md has zero Surveyor mentions. Add SURVEYOR INTEGRATION section (when to consult, what to pass, advisory-never-blocking). (2) Architecture §3 says "Six LLM agents", omits QA + Chronicler. Update to nine agents + full chain. **Criterion:** grep "Surveyor" in Gaffer.md hits; architecture §3 shows 9 agents incl QA/Chronicler.

### [Sprint 2] A-07 — council-brief + weekly-cycle: fix routing gaps and stale hardcodes
**Owner:** Gaffer · **Depends on:** none · **Blocked by decision:** none
(1) council-brief.md has no step routing `<!-- SCIENTIST REVIEW -->` markers — add a Scientist bias-marker resolution step between BriefBuilder and DataSentinel Pass 1. (2) Skeptic BLOCK routing always → FootyStrategy; add a table: BLOCK-on-data-error → DataSentinel/BriefBuilder; BLOCK-on-interpretation → FootyStrategy. (3) weekly-cycle.md line 18 hardcodes "R18 plays this weekend — next run Tuesday 2026-07-07"; delete — the evergreen Tuesday-settlement rule suffices. **Criterion:** council-brief has Scientist-review step; BLOCK routing is a table; weekly-cycle.md has no hardcoded round/date.

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

## Backlog — carried forward (2026-07-27, gate-integrity cycle)

Migrated out of agent memory so they are visible to every agent, not just the one that
found them. Source: Gaffer/Surveyor/Chronicler during commits `a46ca60e8`..`7928ba12f`.
None is a regression from that ship; all are pre-existing conditions with no incorrect
published claim attached, which is why they were deferred rather than fixed.

### [Backlog] BL-01 — Backtest writes into the live `next_round_*` prediction namespace — **DONE 2026-07-27**
**Owner:** Scientist (code) + Gaffer (harness wiring)
**Depends on:** none
**Blocked by decision:** none
**Note:** Tracked informally as "F13" in Gaffer memory before this consolidation. That
collided with the unrelated Sprint-2 F13 above. This ID supersedes that usage.
**Fix brief:** `backtest.py`'s internal predictor run writes a `next_round_*.csv` into the
live prediction directory. Because downstream consumers resolve that directory by
mtime-newest, a backtest artifact can be shipped in place of the real forward prediction.
This is the collision that produced the tainted-provenance incident the 2026-07-27 cycle
was opened to fix. The completion manifest (`scripts/backtest_completeness.py`) now hides
orphaned artifacts from mtime-based consumers, but that is a downstream filter — it does not
stop a future aborted run from colliding in the namespace again. Proper fix is isolation: a
per-run output namespace, not another consumer-side guard.
**Acceptance criterion:** a backtest run cannot write any file into the directory the
forward-prediction consumers read; demonstrated by a test that runs a backtest and asserts
the live prediction directory is byte-unchanged.

### [Backlog] BL-02 — Top-30 deviation loader ignores the completion manifest — **DONE 2026-07-28**
**Owner:** Scientist
**Depends on:** none (independent of BL-01, but the same failure family)
**Blocked by decision:** none
**Fix brief:** `update_team_analysis._load_top30_player_deviation` globs every
`prediction_vs_actual_*` vintage on disk and picks a winner itself. It has no notion of
whether the run that produced a vintage ever completed, so an artifact from an aborted cycle
is eligible for selection. Its vintage key was also date-blind until 2026-07-27 (fixed:
now the full `YYYYMMDD_HHMMSS`). The durable fix is to consult
`data/prediction/backtest/completed_runs.json` and consider only summary-backed, marked-complete
runs — the same rule the harness now uses.
**Acceptance criterion:** a `prediction_vs_actual_*` file whose timestamp is absent from the
completion manifest is never selected, proven by a test with an unmarked newer vintage on disk.

### [Backlog] BL-03 — ~700 legacy garbage rows in the lineup CSVs — **DONE 2026-07-28**
**Owner:** Scientist
**Depends on:** none
**Blocked by decision:** none
**Fix brief:** ~700 of 33,999 rows across `data/lineups/` (412 from 2025, 288 from early 2026)
hold jersey numbers plus `Rushed`/`Totals`/`Opposition` tokens instead of player names — residue
of the S3 scraper defect. Current scraper output is clean name-form, and the files were returned
to the Phase-1 allowlist on 2026-07-26, so new drift has stopped; this is historical residue only.
Cleanup recipe already written at `docs/experiment-log.md:330`.
**Acceptance criterion:** zero rows in `data/lineups/` match the jersey-number/junk-token shape,
with a validator in the integration tier so the shape cannot return.

### [Backlog] BL-04 — Committed chart PNGs no longer reproduce byte-identically
**Owner:** Scientist
**Depends on:** none
**Blocked by decision:** none
**Fix brief:** `assets/charts/top10_alltime_hall.png` regenerates deterministically within the
current environment (identical md5 across runs) but differs from the committed blob
(175,862 → 174,837 bytes). The chart plots scores, not any figure changed in the 2026-07-27
cycle, so this predates that work — most likely matplotlib/font version drift. Consequence:
any agent that runs the HOF regeneration dirties `assets/` as a side effect and must revert it.
Related trap: `generate_backtest_section()` also writes `assets/charts/backtest_accuracy_2026.png`
despite its read-only-sounding name, so "just previewing" a section is not side-effect free.
**Acceptance criterion:** either the committed PNGs are refreshed and reproduce in-environment,
or chart generation is pinned/decoupled so a doc regeneration does not touch `assets/`.

### [Backlog] BL-13 — docs/hall-of-fame-stat-leaders.md is stale against its own source JSON
**Owner:** Scientist (numbers) + Gaffer (ship)
**Depends on:** none
**Blocked by decision:** none
**Fix brief:** `docs/hall-of-fame-stat-leaders.md` was last written 2026-07-20 and now
disagrees with the regenerated `docs/hall-of-fame/_stat_leaders.json` in 10 places — e.g.
career_disposals reads 11,108 where the source computes 11,137. The integration tier
catches it (`test_check_hof_numbers_passes_on_the_published_docs`) and has been red on
this since the R21 data landed. Same disease as the HOF profile drift fixed on 2026-07-27
and the five frozen blocks in afl-backtest-2026.md — a published page whose numbers are
refreshed by a step that did not run — but a DIFFERENT page, so it needs its own fix.
Normally Phase 2 of the weekly cycle regenerates it; the R21 cycle never reached Phase 2,
so this should self-heal on the next successful full run. Verify that it does rather than
assuming, and if it does not, the regeneration is not wired where it is believed to be.
**Acceptance criterion:** `check_hof_numbers.py` exits 0 against the published pages and
the integration tier is green with no per-check exemption.

### [Backlog] BL-12 — Phase 2 rewrites the backtest doc AFTER Phase 1 verified and committed it
**Owner:** Gaffer
**Depends on:** BL-11 (needs the smoke-run tooling before a harness-ordering change merges)
**Blocked by decision:** none
**Fix brief:** `refresh_and_rank.sh` (Phase 1) now regenerates, verifies and commits
`docs/afl-backtest-2026.md`. But `scripts/weekly_refresh.sh` Phase 2 then calls
`scripts/update_eval_surface.sh` AGAIN, which rewrites that doc's CUMULATIVE, TEAMBIAS,
MISSES, TRAINCORPUS and VINTAGEPATH blocks — after verification, and after the commit.
The doc is not in the Phase-4 allowlist, so that rewrite is never committed: the working
tree ends every cycle dirty, and the shipped doc is whatever Phase 1 committed. Latent
today only because no run has yet reached Phase 2. The invariant to enforce: once a gated
doc has been verified, no later phase may write it in the same cycle. Likely shape is an
opt-out flag on `update_eval_surface.sh` so the Phase-2 call refreshes README/banner
without touching the backtest doc, plus a wiring contract test.
**Acceptance criterion:** a full cycle leaves `docs/afl-backtest-2026.md` byte-identical
to what Phase 1 committed, with the working tree clean for that path.
**Deliberately deferred 2026-07-28:** this is a harness-ordering change, and CLAUDE.md
§6.2 (shipped in `1a30279bd`) requires an end-to-end smoke run before such a change
merges. That tooling does not exist yet. Applying this immediately would break the policy
in the same session it was written — and unsmoke-tested harness edits are precisely what
cost three failed relaunches this week.

### [Backlog] BL-10 — POLICY: freeze harness/gate changes during an active weekly cycle
**Owner:** Gaffer
**Depends on:** none
**Blocked by decision:** none — user adopted as standing policy 2026-07-28
**Fix brief:** Once `weekly_refresh.sh` or `refresh_and_rank.sh` has been launched for a
cycle, no harness, gate or hook change lands until that cycle ships or is abandoned.
Hardening discovered mid-cycle QUEUES for after the cycle ships; the NEXT cycle is what
validates it, never the one it lands in. This session is the cautionary case: gating
`docs/afl-backtest-2026.md` and moving the integration tier were both correct changes
made while a cycle was in flight, and between them they cost three failed relaunches and
~7 hours — each failure a gate correctly refusing content that a mid-flight change had
just invalidated. Write the rule into `CLAUDE.md` alongside the TDD rule so the whole
council reads it, not only Gaffer's memory.
**Acceptance criterion:** the rule is in `CLAUDE.md`'s process section; a cycle in flight
is detectable (the `last_refresh_status.json` marker already distinguishes running from
terminated) and the rule names that as the check.

### [Backlog] BL-11 — POLICY: mandatory offline smoke run before any harness change merges
**Owner:** Gaffer (owns the gate) + QA (authors the smoke-run tooling)
**Depends on:** none
**Blocked by decision:** none — user adopted as standing policy 2026-07-28
**Fix brief:** Any diff touching `scripts/weekly_refresh.sh`, `refresh_and_rank.sh`,
`.githooks/pre-commit`, or any script the harness invokes must first run clean through a
full `weekly_refresh.sh` pass in a scratch worktree against the previous week's data
snapshot, with commit and push stubbed out. A green smoke run is the merge condition —
no exceptions, explicitly including "small" fixes, since every harness fix this session
was small and three of them still broke the cycle. Unit tests did not catch any of these
failures: they were ordering and staged-vs-worktree problems that only appear when the
whole harness runs end to end.
**Acceptance criterion:** a documented dry-run mode plus a scratch-worktree smoke script
QA owns; a harness diff without a green smoke run is treated as unmergeable, same
authority as a QA FAIL.

### [Backlog] BL-09 — A multi-line council stamp silently invalidates its own audit record
**Owner:** Gaffer
**Depends on:** none
**Blocked by decision:** none
**Fix brief:** `scripts/council-content-hash.sh` strips lines matching
`<!-- council-pipeline:` or `council-pipeline-gated`. That works only for a
SINGLE-LINE stamp. Write the stamp across several lines — which the format in
`Gaffer.md` visually invites — and only the opening line is stripped; the
continuation lines stay in the hashed content, changing the canonical hash and
orphaning the DataSentinel/Skeptic records the stamp exists to cite. Hit on
2026-07-27: hash moved `da1fc808` → `65f66e06` on stamping, and the gate then
reported the doc as unstamped. Recovered by collapsing to one line. The gate fails
closed so nothing unverified can ship, but the failure mode is confusing and the
next person will lose the same twenty minutes.
**Acceptance criterion:** either the hash script strips a whole multi-line stamp
block, or `record-sentinel-verdict.sh`/the stamp writer rejects a multi-line stamp
with a clear message. A test covers both stamp shapes.

### [Backlog] BL-08 — Prediction output path is cwd-relative in one place, config-driven in another — **DONE 2026-07-27, as a side effect of BL-01**
**Owner:** Scientist
**Depends on:** none
**Blocked by decision:** none
**Fix brief:** `supercoach/prediction.py:1122` writes to a hardcoded, cwd-relative
`Path("./data/prediction")`, while `backtest.py` reads `config.PREDICTION_DIR`
(`config.py:146`). The two agree only when the process happens to be started from the
repo root. Every harness path does start there, so this is latent rather than live —
but it means the writer and the reader of the same directory disagree about how to
find it, which is the shape of a bug that appears the first time something runs from
elsewhere. Surfaced incidentally on 2026-07-27 and deliberately not chased.
**Acceptance criterion:** both sides resolve the prediction directory through
`config.PREDICTION_DIR`; a test that runs the writer from a non-root cwd still lands
files where the reader looks.

### [Backlog] BL-07 — Commit forward prediction CSVs before the round is played
**Owner:** Gaffer (harness)
**Depends on:** none
**Blocked by decision:** none — user deferred the harness change on 2026-07-27, prose softened in the meantime
**Fix brief:** Rounds scored by the `--from-csv` archive path are leak-proof only if
the forward prediction genuinely predates the game. Today that ordering is evidenced
by filename timestamp and mtime, both of which the writing process controls, so it is
an assertion rather than an attestation. R18 and R19 happen to be cleanly attested —
their forward CSVs were committed to git before first bounce — but R20's was not
committed until after the round, so its guarantee currently rests on self-reported
metadata. `docs/afl-backtest-2026.md` now says so explicitly rather than overclaiming.
The durable fix is to commit the upcoming round's forward prediction CSV as part of the
weekly cycle, before the round is played, making git the independent witness.
**Acceptance criterion:** every round scored by the archive path has its forward CSV in
a commit dated before that round's first bounce, checkable after the fact without
trusting any file timestamp.

### [Backlog] BL-06 — Player-name casing is flattened in the source CSVs
**Owner:** Scientist
**Depends on:** none
**Blocked by decision:** none
**Fix brief:** `prediction_vs_actual_*.csv` stores names title-cased, so intercapped
surnames lose their casing: "Mccluggage Hugh" where the player is Hugh McCluggage.
Harmless while the data was only read by code, but `docs/afl-backtest-2026.md`'s
notable-misses table is now generated from these CSVs, so the flattened casing is
published — the hand-written table it replaced had it right. Affects the Mc/Mac and
O' families at minimum. Deliberately NOT fixed with a display-time regex: a heuristic
that re-capitalises names will eventually mangle a legitimate one, and publishing a
wrong player name is worse than publishing an unfashionable one. The fix belongs at
the scrape/normalise layer, against a real name list.
**Acceptance criterion:** names round-trip from source with original casing intact,
verified against a sample including McCluggage, McKay, O'Brien.

### [Backlog] BL-05 — Gate verdict records capture the verdict but not the reasoning
**Owner:** Gaffer
**Depends on:** none
**Blocked by decision:** none
**Fix brief:** `scripts/record-sentinel-verdict.sh` writes exactly
`{doc_path, doc_hash, verdict, ts, agent_id}`. The findings behind a FAIL or BLOCK exist only
in the invoking agent's transient stdout, so nothing durable records WHY a gate blocked or how
many distinct findings it carried. On 2026-07-26 a Skeptic BLOCK carrying four findings was
read as one, three were never routed, and a full gate cycle was spent rediscovering them — an
error the audit trail could not have caught, because the record looked identical either way.
Fix: persist a `findings` array (id, severity, quote/locator, issue) alongside the verdict, and
have the harness echo the count so a multi-finding verdict cannot be mistaken for a single one.
**Acceptance criterion:** a BLOCK with N findings writes N entries to the audit record, and a
re-gate can be checked against the previous record to confirm every finding was addressed.

---

*Last updated: 2026-07-27. 2026-07-07 plan prepared by Surveyor; BL-nn backlog consolidated by Gaffer. Route questions to Gaffer.*

