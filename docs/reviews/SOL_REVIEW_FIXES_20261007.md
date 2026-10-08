# Sol repository review fixes — 7 October 2026

Source work started from `8324dc0be7a99641dc356dac9352a113270ddece` in
`work/sol-review-fixes-20261007`; final verification resumed on 8 October after a daemon
restart. This record follows [Fable's review](FABLE_REPO_REVIEW_20261007.md). It records
local fixes and evidence, rather than claiming a remote CI run, publication or promotion.
The repository name and README's “Why this repo exists” text are preserved.

| Finding | Status | Change and evidence | Remaining acceptance |
|---|---|---|---|
| F1: source notice disappears on new snapshots | Fixed locally | `provisional.ts` binds UNKNOWN and the two original report identities to corrected snapshot `b9830cbf…`; unrecognised real snapshots receive an explicit unaudited notice and noindex. Demo behaviour is separate. Component tests cover notices, robots, and independent fresh/stale dataset status. Candidate browser review confirms UNKNOWN and 63 unresolved cells. | Exact delivered commit still needs remote checks. |
| F2: corrected candidate publication chain | Local artifact complete; publication pending | [Candidate validation summary](evidence/sol-candidate-validation-20261007.json): release `20261007T110646Z-de5708d65c5f`, matching corrected snapshot, canonical release validation PASS, sealed site, full integrity PASS with complete and semantic_complete true. [Composite source coverage](evidence/afltables-candidate-source-coverage-20261007.json) preserves both original audit identities and verifies historical fragment continuity. | Owner publication/promotion decision. Source verdict remains UNKNOWN. |
| F3: CI failures and unenforced checks | Partial | Python CI installs pinned Node and locked browser dependencies; contract resolves Node through PATH and fails when missing. Negative checks prove absent Node and absent Vitest raise failures, with no skip. Obsolete `tests.yml`, `pylint.yml` and Conda workflow retired; locked `legacy` job retains legacy coverage alongside package/web jobs. Ruff/mypy are canonical; no Pylint rule parity claimed. | Exact new commit must run remotely. Broken `weekly-fan-pack.yml` remains unchanged pending owner retirement/release-policy decision. Branch protection/ruleset enforcement is an owner action. |
| F4: ignored stale snapshot audit | Partial | Gate/status records explicitly name legacy decision scope, actual captured reference snapshot and audited seasons; correction re-audit pins the same snapshot even if current.json changes. Tests cover scope metadata and pointer drift. Operator docs call the snapshot a reference. | Snapshot comparison cost remains. Gate still computes the reference layer; it does not meet legacy-only-plan or promoted-root binding acceptance. |
| F5: stale public status | Fixed locally | README, reconciler overview, acceptance/run records and `data-integrity.md` distinguish merged software, corrected UNKNOWN data and older hosted FAIL preview. Provisional HOF regenerated from corrected snapshot using pinned coverage receipt (`54fdc52c…`), retaining unresolved evidence and original scopes. README is excluded by the council stamp gate; these changed counts describe audits, not football claims. Deterministic HOF checks and the ordinary hook apply. | Scratch smoke passed; see the 8 October receipt. Remote checks are attached to the delivered commit. |
| F6: interpreter and budget contract | Partial | CLAUDE.md and pyproject describe harness/hook interpreter precedence and the combined non-integration tier. Published integration runs at Phase 3d after regeneration. Existing targets remain 20 s combined and 30 s package; measured overruns are disclosed. | Performance work/owner budget decision remains; targets were not raised. |
| F7: shell/hook edits bypass tests | Fixed locally | Hook also gates `scripts/*.sh`, `refresh_and_rank.sh` and `.githooks/*`; deleted and renamed paths are covered. Behaviour tests exercise shell-only, hook-only, deletion and rename cases, including failing test rejection. | CLAUDE.md §6.2 scratch smoke passed on 8 October. |
| F8: recurring unsourced definitional claim | Fixed locally | Finals recap prompt requires a source defining any claimed overlap and otherwise reports association. Harness wiring test rejects the old mandatory definitional claim. | Real council scratch smoke passed on 8 October; prose concerns are recorded below. |
| F9: tracked machine audit state | Pending migration | `docs/operations.md` describes preserving bytes/hashes, defining fail-closed initial state, testing absent/active/complete markers and only then untracking agreed paths. Existing markers/verdict records are preserved. | Dedicated migration and owner scope decision. |
| F10: broad local permissions | Pending owner change | Operator-owned settings remain intact. Operations docs identify stale grants/interpreter and require a separate preserved settings change. | Owner permissions review; no local/global permissions modified. |
| F11: conflicting QA failure policy | Fixed locally | QA uses interpreter resolution and explicit fast/integration commands. Every test failure, including pre-existing failures, blocks the applicable gate; warnings have one explicit verdict. Gaffer agrees. | Council smoke checks applicable weekly gates; manual QA remains a separate role. |
| F12: inconsistent weekly chain | Fixed locally | Gaffer, QA and weekly-cycle instructions match the actual harness: numeric gates, recap, DataSentinel, Skeptic, Phase 3d integration, allowlist commit/push, Chronicler. They do not invent a weekly QA-agent invocation or stamp; manual deliveries retain QA. | Full scratch chain passed on 8 October. |
| F13: recap omitted from Skeptic scope | Fixed locally | Skeptic explicitly covers the weekly recap at Phase 3c, with evidence/prose checks and no invented recommendation/lens requirements. DataSentinel's recorded pass supplies the arithmetic gate for this invocation. | Real recap invocation passed with Skeptic concerns recorded below. |

## Additional defects reproduced during verification

The installed uutils `tail` rejects obsolete `tail -40` and `tail -15`. Two regression
cases failed before editing the scripts, then passed after changing the failure-log
paths in `refresh_and_rank.sh` and `scripts/smoke_harness.sh` to `-n40` and `-n15`.
The tests also run the selected option against 50 input lines and compare exact output.

The malformed external-release web test initially received empty Astro output. Explicit
subprocess assertions identified `spawnSync /usr/bin/node EPERM`, status 1, signal null:
the sandbox prevented the subprocess. An escalated focused retry passed. The test now
requires no subprocess error, no signal, exit 1 and the manifest-verification diagnostic;
the invalid release is neither skipped nor accepted.

## Candidate evidence and limits

The [sanitized summary](evidence/sol-candidate-validation-20261007.json) records seal
`e8e8a0fe5dc6bdd12474c8fdceb3cbdf2ce585b040942dfc6aa8cfdaf27f9a05` and integrity
report `9bc9394d87ccaf2a6bc415adab0865304b8a7d88dc502b31d2854ceac627e118`.
All 26 integrity checks pass with both completeness fields true. Findings still include
five nonblocking historical early-behinds heuristic errors (1897, 1898, 1899, 1949,
1953), 11 warnings including stale fixture metadata and retained predictions, and five
informational findings. Eighty-four archived article prose resources are checked for
provenance, rather than semantic prose correctness. This is not an error-free source audit.

The site is 265,010,840 bytes (252.734 MiB), below the 300 MiB artifact budget by
47.266 MiB. Parent browser review on 8 October covered 48 page states at widths
320/390/768/1440 in both themes, 24 accessibility scans with zero violations, search,
JavaScript-disabled behaviour and exact README purpose-text preservation. Build/audit
as-of remains 7 October. No snapshot promotion, deployment or schedule change occurred.

## Verification

The full-suite results below precede the retry-context fix described afterward.
That final harness edit has focused verification only and still requires a new full
suite and a passing §6.2 smoke before merge.

Commands run from the isolated worktree were `.venv/bin/ruff check`,
`.venv/bin/mypy`, `.venv/bin/python -m pytest tests/ -q -m "not integration" -n 4`
(BLAS/OpenMP thread counts set to one), and, under `web/`, `npm run gen:types:check`,
`npm run lint`, `npm run check`, `npm test`, `npm run test:e2e` and
`npm run budget:bases`. Subprocess/browser checks ran with sandbox escalation after
the reproduced EPERM failure.

- Ruff: all checks passed. mypy: no issues in 104 source files.
- Web generated contracts match; ESLint passed; Astro/TypeScript check: 154 files,
  zero errors, warnings or hints.
- Required Python-to-TypeScript negative probes: missing Node raises the explicit
  required-runtime assertion; missing Vitest runs actual Node in an empty web directory
  and raises on MODULE_NOT_FOUND. Both fail without a skip.
- Final combined fast tier after the retry repair: **2,159 passed, no skips**, 509 warnings, 60.06 seconds
  on four workers. This still exceeds the unchanged 20-second combined target.
- Full web unit tier: **184 passed, two skipped** in 2.54 seconds (23 files passed,
  one skipped). The skipped cases require `SCVIA_PY_PUBLIC`; the required Python
  release test supplies it and passed in the combined tier above.
- Both-base production builds passed. Playwright: **374 passed, four skipped** in
  2.2 minutes. The four skipped subpath cases duplicate file-level CSP checks that
  already inspect both builds in the root project; this is the existing suite policy.
- Both-base demo payload budgets passed. Remaining workflow inventory is
  `scvia-ci.yml`, `scvia-pages.yml` and `weekly-fan-pack.yml`; the Pages workflow
  is unchanged, and the broken fan-pack workflow remains pending an owner decision.
- CLAUDE.md §6.2 full scratch smoke passed through Phase 4 with commit/push stubbed;
  see [the recorded receipt](evidence/sol-scratch-smoke-20261008.json).

Ignored local verification logs are under `var/agent-runs/sol-review-fixes-20261007/`.
The restored ignored `docs/hall-of-fame/_stat_leaders.json` fixture is retained so its
four dependent tests run rather than skip. Main's four pre-existing dirty audit records
are outside this isolated source work and must be preserved during integration.

## Scratch smoke failure and retry-context fix — 8 October

The coordinator's full scratch smoke ended with exit 1 at Phase 3b, after DataSentinel
rejected the recap twice. Its first verdict explained that a stated regular-season
count included a finals row, but FootyStrategy's retry reported receiving only
`Daicos '23 [data]`. The confirmed transport defect was the regex
`\[[^]]*\]`: it stopped at the closing bracket inside the claim's literal `[data]`
tag and dropped the remaining claim and reason. Shell expansion of the double-quoted
variable did not re-evaluate apostrophes or command syntax.

The minimal repair in `weekly_refresh.sh` structurally decodes the last FAIL verdict
with a failed_tags array, tolerating CLI warning prefixes, and sends that array to
Claude on standard input. If no such JSON decodes, it sends the complete output;
there is no 1,500-character truncation. Retry instructions explicitly ask the agent
to address each reported reason. Counts, population filters, gates and the bounded
single-retry failure policy are unchanged.

Two regression cases failed before the repair and passed afterward. The focused
behaviour test executes the actual harness retry block with a capturing Claude stub;
compact JSON, multiline JSON and malformed fallback preserve brackets, nested arrays,
apostrophes, double quotes, dollars, backticks and newlines. Literal touch expressions
do not create their sentinel file. Harness wiring plus smoke-contract tests pass:
**34 passed in 1.05 seconds**. `bash -n` and `git diff --check` pass. Focused
test output is retained in `var/agent-runs/sol-review-fixes-20261007/retry-prompt-focused.log`.
The failed scratch log is `/tmp/scvia-sol-smoke-20261007/20261007T201903Z.log`.
The owner subsequently requested merging. The coordinator reran the full fast tier
(2,159 passed in 60.06 seconds) and the complete scratch weekly harness.

## Green scratch rehearsal — 8 October

[Receipt](evidence/sol-scratch-smoke-20261008.json): exit 0 at Phase 4, with real
DataSentinel PASS, Skeptic PASS_WITH_CONCERNS and 22 integration tests passing in
2.13 seconds. Commit and push were suppressed. Scraping was stubbed because the
patch does not change scraping; reconciliation correctly reported no changed season
and no audit. This run is ordering evidence, not a new AFL Tables source capture.

Skeptic retained three wording concerns in the scratch recap: clarify the pooled
correlation population, describe the team average period, and define the league mean
as an average across player averages and positions. None met its BLOCK threshold.
The scratch recap is not part of this code merge. The first-pass numeric review
passed, so this run did not need the retry; the regression cases exercise that path.

All original main audit records were preserved; the daily Claude audit log only
appended entries. Both operator data pointers retained their bytes. The tested source
was frozen for the rehearsal. Only this evidence record and its receipt were added
afterward. Full local logs are in `var/agent-runs/sol-review-fixes-20261007/smoke-attempt-2/`.
