# SuperCoach VIA rewrite audit

Audit dates: 23–24 September 2026, Australia/Melbourne. Code baseline: `b4ce74770b20e670a43a3165913dc4e188a25bad`. Scope: the checked-out application, its data and publishing pipeline, tests, workflows, documentation, and a new browser product requested by the owner.

Read [the implementation specification](PLAN.md) next, then give [the execution prompt](CLAUDE_CODE_PROMPT.md) to Claude Code. [The refresh report](DATA_REFRESH.md) records the separately requested catch-up data update. This audit is evidence for the rewrite, not a claim that the rewrite is already implemented.

## 1. What exists

This is a Python analytics and publishing application, not an existing browser application. There is no checked-in HTTP application, frontend package, authentication system, or application database to migrate. The useful product consists of:

- AFLTables match results, historical player performances, personal details and lineups.
- Draft, rookie draft, DraftGuru, contract/free-agency and school-enrichment importers.
- Disposal forecasting, archived forecast scoring, retrospective backtests, era analysis, rankings, team profiles, stat leaders, Brownlow proxies and finals commentary.
- Generated Markdown, PNG/SVG charts, weekly cheat sheets, player cards and downloadable fan packs.
- FanFooty snapshot fetching and two overlapping live commentary scripts.
- A shell-based weekly workflow and a Claude editorial council with deterministic and agent-based publication checks.

The tracked baseline contains 128 Python files and 35,764 Python lines, including tests, archive and diagnostics. There are 62 unit-test modules and four integration-test modules, plus package initializers. The source inventory must use `git ls-files`: recursively counting hidden agent environments inflates the apparent application size.

The data tree occupied approximately 176 MiB at inspection. It contained 13,367 performance CSVs and 13,367 personal-details CSVs, 130 match-season files and 24 lineup files. These are file-inventory observations, not proof that all historical rows or statistics are complete. Historic stat coverage is explicitly uneven in [`config/stat_coverage_eras.yaml`](../../config/stat_coverage_eras.yaml).

The original checkout was already dirty: `data/top100/yearly/year_2026.csv` was modified, and several audit, survey and forecast files were untracked. Preserve them. The two similarly named all-time exports have **different schemas**, not interchangeable copies:

- `all_time_top_100.csv`: `Serial Number,Player Name,Footy Teams,Comment`.
- `data/top100/all_time_top_100.csv`: `player,all_time_score`.

## 2. Verification performed and limits

Read the executable implementation and tests rather than accepting architecture documents as current. Examined ingestion, model features/training/prediction, backtest identity, refresh ordering, trust gates, live processing, output generation, packaging and CI. Parsed Python source for syntax errors and inspected real CSV headers and operational markers. No production weekly publishing harness, model retraining, LLM council, commit or push was run for this audit.

Tests ran in temporary copies, using Python 3.12.3, so generated files and test mutations did not affect the checkout:

| Check | Observed result | Meaning |
|---|---|---|
| Fresh environment from unpinned requirements, plus pytest/tabulate | Test collection fails with pandas 3.0.6: `pandas.errors.SettingWithCopyWarning` no longer exists | A clean install is broken; dependency reproducibility is a release blocker |
| Unit suite with pandas 2.3.3 | 555 passed, 45 skipped, 16 warnings, 8.03 seconds | Existing regression coverage is valuable; skipped tests are not verified |
| Integration suite on pre-refresh data | 19 passed, one skipped, one failed after supplying the tracked agent registry | Committed top-100 PNG differs from a clean render in the new matplotlib/font environment |
| Initial isolated integration copy | Two extra failures because the audit copy initially omitted `.claude/agents` | Audit setup issue, not an application defect; both passed when the tracked registry was copied |
| Restricted-network data-only refresh | Logged DNS failures and unavailable fixtures, then exited zero | Independently reproduced fail-open behavior; zero exit status does not establish freshness |

The 45 unit skips include modules conditioned on `/home/abhi/sourceCode/python/coding/.venv/bin/python` and a missing generated `_stat_leaders.json`. The temporary copies included tracked working-tree inputs; they intentionally did not inherit that ignored generated JSON from the original checkout. The integration render used matplotlib 3.11.2 and a different environment from the producer. A byte mismatch is an established reproducibility problem, **not evidence that its plotted numbers are false**. Exact environment and result counts are in [baseline test evidence](evidence/baseline-tests.json).

No end-to-end forecast training benchmark or statistical validation of a replacement model was performed. Runtime and payload budgets in PLAN.md are targets to measure, not claimed improvements. No browser existed to measure accessibility, Core Web Vitals or browser vulnerabilities. No comprehensive dependency CVE scan or penetration test was performed. Findings below are a thorough repository review, not a guarantee that every latent defect has been discovered.

## 3. Findings and required improvements

Priorities: **P0** = fix before new results can be trusted or published; **P1** = required for the replacement release; **P2** = polish or a measured follow-up. `C` = correctness, `P` = performance, `U` = usability, `S` = security, `Q` = maintainability. Source line numbers refer to the audit baseline; functions are more durable references.

### Correctness and data trust

| ID | Priority | Evidence and consequence | Required replacement |
|---|---|---|---|
| C01 | P0 | [`prediction.py:1116`](../../supercoach/prediction.py#L1116) filters `round_number >= next_round - 1`, selects the earliest retained row and saves it under a next-round filename. With completed historical rows, the previous match's feature row can be labelled as a future forecast. [`backtest.py:433`](../../backtest.py#L433) explicitly explains its one-round filename offset. | Build a distinct future-fixture row; every forecast carries match ID, target date, season/stage, origin and cutoff. Do not copy this behavior for parity. |
| C02 | P0 | `prepare_features_and_target` fits the imputer before cross-validation; tuning and calibration use player `GroupKFold`, not time-ordered folds. Held-out players are not the same validation question as a future round. | Fit preprocessing inside each temporal fold; keep entire matches/time blocks together; isolate tuning, calibration and final holdout. Existing outer backtest cutoffs remain valuable but do not fix inner-fold preprocessing leakage. |
| C03 | P0 | `_engineer_features*` orders by player/year/round, although rescheduled rounds need not follow match date. Numeric finals mappings differ among modules. The catch-up refresh additionally reproduced unsupported `WF` rows receiving a synthetic date rather than their wildcard fixture date. | One event-time ordering and explicit fixture stages; preserve Opening Round, wildcard finals, postponements, finals, draws and replays. |
| C04 | P0 | `load_player` dummy-encodes opponent/venue separately for each player's file with `drop_first=True`; the baseline varies between players. Train and serve independently build/reindex columns. | One fitted encoder and one feature builder; controlled categories, unknown handling, strict schema/order validation. |
| C05 | P0 | Forecast exports discard IDs, target date, target round, cutoff and model identity; `_gather_actuals` matches by display-name/team. Same-name players and later club changes are ambiguous. The catch-up also verified multi-part surname mismatches and duplicate Will/William Green and Roan Steele files; see [identity evidence](DATA_REFRESH.md). | Stable player/match IDs; explicit identity maps; reconcile/quarantine duplicate input records; never silently fuzzy-join; manifest-backed legacy exports. |
| C06 | P0 | [`game_scraper.py:21`](../../scrapers/game_scraper.py#L21) returns empty soup on fetch failure; missing fixtures return no issues. [`refresh_data.py:270`](../../refresh_data.py#L270) can log “clean” without verification. Reproduced with blocked DNS. | `PASS/FAIL/UNKNOWN` result type; UNKNOWN cannot promote a new verified dataset. Preserve last accepted snapshot and stale status. |
| C07 | P0 | Match delta mode fetches each detail before skipping older dates, and skips known match identities; player delta uses last date/counter. These paths can miss interior gaps and corrections to old records. Recent-source comparison during the catch-up confirmed changed stored statistic cells and previously unavailable award-vote cells. | Source-ID upserts, bounded overlap refresh, season reconciliation and explicit repair mode; retain previous revisions. |
| C08 | P1 | Player parser still synthesizes March-plus-weeks dates when fixture lookup fails and substitutes a default DOB when parsing fails. | Preserve unknown values and date precision/source quality; quarantine ambiguous matches; exclude inferred timestamps from time-sensitive forecasting. |
| C09 | P1 | `refresh_players` audits only active files whose row count grew; unchanged-row corrections and some newly discovered files escape its report. Worker exceptions can be logged and ignored. | Hash-based changed-file/record inventory, structured per-source outcomes, validation of new and corrected records, truthful partial-failure status. |
| C10 | P1 | Ranking fills missing stats with zero; other modules use different denominators. Ranking-era constants and the coverage YAML differ in intent. | Explicit null/zero semantics and observed-game denominators. Preserve versioned legacy ranking output, then separately version any methodological correction. |
| C11 | P1 | Prediction import filters by a hard DOB-age threshold, training excludes the target season, first-season feature rows are dropped, and predictions are clipped to 1–55. These are modeling policies embedded as code. | Declare eligibility, training-window and cold-start rules; zero is a valid outcome. Benchmark policy changes with matched populations and full precision. |
| C12 | P1 | Optuna cache keys include feature names, row count and age but not the exact dataset, ordered dtypes, feature algorithm, fold policy, target or all model settings. Forecast/backtest caches are separated already. | Typed model/feature manifests and immutable cache keys covering all semantic inputs; retain the existing namespace separation. |
| C13 | P0 | Prediction selection differs across mtime, filename ordering, numeric round and timestamps in shell, integration tests, cheat-sheet and fan-pack code. Season is absent from the forward CSV. | Exactly one artifact resolver by run/season/stage/kind and completed manifest; filenames and checkout mtimes never establish provenance. |
| C14 | P1 | Multiple report surfaces contain manually copied accuracy figures and differing windows. [`how-to-use-this-for-supercoach.md`](../how-to-use-this-for-supercoach.md) also turns MAE into an informal outcome range. | Shared metric objects with denominator, origin and cutoff; genuine held-out intervals with measured coverage, or no interval. |
| C15 | P1 | Brownlow and finals outputs mix formulas and heuristics with prediction terminology. Backtests use position `Unknown`; current data does not provide universal verified playing positions. | Label proxy/rule-based outputs and inferred roles; never advertise calibrated probabilities, official positions or SuperCoach points without evidence. |

### Performance and operational reliability

| ID | Priority | Evidence and consequence | Required replacement |
|---|---|---|---|
| P01 | P1 | The predictor auto-detects season by scanning every performance CSV; active-player detection also scans every file. Team analysis has several separate corpus loaders; backtest actuals scan the corpus per round. | Import once, partition canonical data by season, query projected columns through DuckDB and persist source/content manifests. |
| P02 | P1 | Predictor repeats feature construction by player, tuning performs many fits, calibration repeats folds, and the summary can refit models just to print feature importance. | Batch features and inference, reuse chronological out-of-fold results, remove diagnostic refits, explicit training/tuning commands and budgets. |
| P03 | P1 | `_detect_lgbm_device()` fits a GPU probe at module import, while `supercoach/__init__.py` imports the predictor. This affects unrelated imports and tests. | Lazy optional ML imports, CPU default, explicit cached device selection only when training. |
| P04 | P1 | Match/player HTTP calls have no timeout in core `get_soup`, no shared retry/session policy, and per-thread sleeps do not limit aggregate request rate. | Bounded shared HTTP client, pool, global per-host limiter, timeout/retry/backoff, response cache and structured fetch accounting. |
| P05 | P1 | `_process_year` requests all season game details before applying the date skip. The weekly scripts invoke overlapping HOF/chart/eval regeneration. | Decide fetch work from fixture/source manifests first; content-hash DAG executes each changed transform once. |
| P06 | P0 | Weekly flow commits/pushes a first phase before later editorial/integration gates, then publishes again. `git_commit_safe.sh` locks Git invocation, not the entire dataset mutation/run. | One run lock, isolated staging, validation before promotion, one publication boundary, resumable content-addressed steps. |
| P07 | P1 | CSV and JSON writes usually replace files directly; minute-level filenames can collide. Live scripts can consume the newest pre-existing snapshot after a failed fetch. | Temp write + atomic replace, unique run IDs, explicit returned artifact IDs and no stale fallback presented as fresh. |
| P08 | P1 | Live scripts overlap, hardcode one match/doc routing/key players, and attempt frequent Git commits/pushes; push errors are not reliably checked. | Generic per-match monitor, append-only snapshots, idempotent state transitions, separate export and publisher; confirmed commit/deploy receipts. |
| P09 | P1 | Generated data, charts and snapshots contribute to repository size and churn; immutable historical data is reparsed weekly. | Keep existing historical source assets, move new runtime caches/raw payloads to ignored storage, use verified release bundles with retention. No history rewrite. |
| P10 | P1 | Logs and completion markers do not consistently distinguish fetched, parsed, validated, published and stale states. | Per-stage timings/counts/status, structured error codes, release receipts and a browser data-status page with actionable recovery. |

### Security and publication integrity

| ID | Priority | Evidence and consequence | Required replacement |
|---|---|---|---|
| S01 | P0 | [`weekly_refresh.sh:359`](../../scripts/weekly_refresh.sh#L359) and related turns use `--permission-mode bypassPermissions`; source text is fed into agents that may write files or execute shell tools. | Optional editorial adapter with scoped read-only evidence and one output location; no shell/network/Git credentials inside the model task. Treat retrieved text as data. |
| S02 | P0 | Core refresh/publishing owns local Git credentials and writes the shared index; live code calls raw Git independently of the wrapper. | Scraper/model/editorial jobs have no publication credentials; publisher alone consumes a complete validated bundle and an explicit publication request. |
| S03 | P0 | PASS text is not proof. The content-hash gate is a good existing control, but its audit cross-check can skip when hashing is unavailable; local JSON records share the same writer trust domain. | Strict structured verdicts bound to content, dataset and policy hashes; UNKNOWN/failure blocks the relevant lane. Explain authenticity limits; recompute deterministic checks in publication CI. |
| S04 | P1 | [`weekly-fan-pack.yml`](../../.github/workflows/weekly-fan-pack.yml) interpolates `inputs.release_tag` into shell source. Only users able to dispatch the workflow reach that input, but it need not become executable shell. | Put inputs in environment variables, validate a bounded tag grammar, quote variables; test hostile input rejection. |
| S05 | P1 | Requirements have no versions/hashes; actions use movable tags; no unified install/lock/scan policy. | Commit dependency locks, pin action SHAs, minimal workflow permissions, vulnerability/license reports and a reviewed upgrade process. No particular CVE is asserted here. |
| S06 | P1 | Remote URLs/link paths and parsed names flow through fetchers/files; no single containment/redirect policy. This is an outbound ingestion threat, not an existing public SSRF endpoint. | Trusted adapter URL construction, HTTPS host/redirect allowlists, response caps and output-root/path containment; reject unknown resource IDs. |
| S07 | P1 | New browser rendering would expose external article/feed strings and downloadable CSVs to HTML/JavaScript and spreadsheet interpretation. | Escaped text, sanitized Markdown without executable MDX/HTML, safe URL schemes, CSP, neutralized spreadsheet text formulas. This is a required new-surface defense, not a discovered deployed XSS bug. |
| S08 | P1 | Logs/agent evidence can include unbounded external text and local context. Model persistence is currently informal. | Bounded/redacted logs, explicit evidence allowlists, trusted-local model bundles only; never deserialize a user-uploaded pickle. |

### User experience, accessibility and maintainability

| ID | Priority | Evidence and consequence | Required replacement |
|---|---|---|---|
| U01 | P1 | Fans navigate many Markdown documents or import CSV into Sheets; no built-in search, filters, comparison or retained watchlist. | Browser dashboard with predictions, player/team/match explorer, historical records, accuracy and a local watchlist. |
| U02 | P0 | The baseline reports a completed refresh dated 29 August while the current date is 23 September. Dates, “current” links and forecasts can diverge. | Distinct fetched-at, complete-through, generated-at and published-at labels; visible stale/unavailable states. See refresh report for new state. |
| U03 | P1 | Dense tables and image charts lack a consistent mobile/keyboard/text-data experience; some accessibility checks exist for the banner. | WCAG 2.2 AA target, semantic tables, chart descriptions/data alternatives, keyboard filters, focus, contrast and responsive layouts. |
| U04 | P1 | Product name encourages users to infer fantasy-score/selection capabilities beyond a disposal model. | Label units, sample size, uncertainty, eligibility and freshness at the point of use; separate disposals, observed fantasy data and heuristics. |
| U05 | P1 | Quick start runs a publishing workflow using unavailable personal interpreter/CLI paths; `main.py` can start a broad scrape. | One portable CLI, offline demo, doctor, dry-run plan and explicit data/train/build/publish operations. |
| U06 | P1 | Fan-pack copies are best-effort and selections differ; relative chart links may not resolve in the ZIP. | One validated release manifest for site/Markdown/CSV/ZIP; complete asset closure, version and checksums. |
| Q01 | P1 | `update_team_analysis.py` is 5,295 lines; prediction, rendering, narrative, I/O and configuration are coupled. | Domain, ingestion, analytics, ML, publishing and CLI modules with typed boundaries and small pure transforms. |
| Q02 | P1 | CPU predictor, live monitor, metric rendering, team normalization and doc replacement logic overlap. | One implementation per domain rule; thin compatibility wrappers only while migrating. |
| Q03 | P1 | Paths, season years, round heuristics, rules and runtime clocks remain scattered despite the useful central `config.py`. | Validated configuration and injected clock; one season/stage/club/venue registry with period-aware aliases. |
| Q04 | P1 | CI advertises a unit job but runs all tests; Python 3.8/3.9 lint conflicts with modern syntax; Conda references missing `environment.yml`. | One supported runtime matrix and lock; explicit hermetic, real-data and browser tiers; remove superseded workflows after replacement passes. |
| Q05 | P1 | Important tests skip on another user's absolute interpreter; some “unit” checks read generated/real data. Integration gates require a matching unpinned renderer. | Portable executables, test-owned data paths, offline fixtures, locked renderer/fonts; semantic chart checks plus bounded image comparisons. |
| Q06 | P1 | Broad `except`/continue, global warning suppression and global plotting state conceal failure contexts. | Typed failures, narrow catches, error summaries, scoped plotting settings and no import-time side effects. |
| Q07 | P1 | `docs/ARCHITECTURE.md`, `docs/architecture.md`, `docs/ai-architecture.md`, inline comments and operational code disagree in places. | One maintained architecture/operations source, generated command/schema references and a feature/parity map. |
| Q08 | P1 | Useful regression tests encode prior bugs, but source-text assertions alone cannot prove runtime publication ordering or temporal correctness. | Preserve bug scenarios as behavioral tests; full offline pipeline fault injection and independent forecast/metric verification. |

## 4. Preserve existing strengths

The rewrite should retain the historical corpus, source fixtures, public URLs/download contracts, editorial archives and regression scenarios. Existing improvements already include vectorized grouped rolling windows, an in-process player cache, single-pass ranking aggregation, separate backtest output directories, completed-run tracking, current-season gap gates, numeric HOF checks, staged-blob verification, bounded editorial retry and a Git lock wrapper. Extend these ideas rather than claiming they do not exist.

The strongest product principle is that statistics must be reproducible from evidence. Make that principle a shared data contract for browser, reports and agents. Make editorial prose optional for the numeric pipeline while keeping an unverified draft out of publication.

## 5. Architecture recommendation

Use a **Python modular application plus a static browser site**. Keep scraping/training on the operator or scheduled runner. Normalize historical data into immutable, typed Parquet snapshots queried with embedded DuckDB. Generate release-scoped JSON, Markdown, CSV and charts from one snapshot. Build an Astro/TypeScript site with small React islands for search/filter/compare interactions. Serve only the accepted public bundle.

This fits the existing public, read-mostly product and avoids introducing accounts, a public database, an admin API, queues and always-on application servers merely to browse a weekly dataset. Genuine low-latency multi-user writes, paid accounts or a public assistant would justify a separate future architecture decision.

Official guidance checked while drafting:

- Fit preprocessing within training folds: [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html).
- Commit and verify a dependency lock: [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/).
- Use the embedded single-writer model deliberately: [DuckDB concurrency](https://duckdb.org/docs/current/connect/concurrency).
- Keep static HTML and hydrate only interactive pieces: [Astro islands](https://docs.astro.build/en/concepts/islands/); configure project-base URLs for [GitHub Pages](https://docs.astro.build/en/guides/deploy/github/).
- Avoid direct expression interpolation into shell: [GitHub script injections](https://docs.github.com/en/actions/concepts/security/script-injections).
- Apply [WCAG 2.2](https://www.w3.org/TR/WCAG22/) to the new browser experience and [OWASP outbound-request guidance](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html) to ingestion.

These sources establish implementation practices. They do not certify this repository or a future implementation as secure, accessible or statistically accurate.
