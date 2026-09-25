# Claude Code execution handoff

Paste the following prompt into Claude Code opened at the repository root. It authorizes implementing the rewrite locally; it does not authorize publishing, pushing or replacing unrelated local work.

---

Implement the SuperCoach VIA rewrite described in `docs/rewrite/PLAN.md`, using `docs/rewrite/AUDIT.md` as the evidence and `docs/rewrite/DATA_REFRESH.md` as the current data-refresh handoff. Read all three documents, root `CLAUDE.md`, and any applicable `AGENTS.md` before editing. This is an implementation request, not another planning request. Carry it through to a tested, locally runnable application and reviewable production build.

The owner has explicitly requested:

1. A clean rewrite that addresses performance, user experience, security and code quality.
2. A browser dashboard in addition to the existing analytics, reports, charts and downloads.
3. An immediate update of the existing data, whose results and remaining limitations are recorded in `DATA_REFRESH.md`. Preserve those updates as migration inputs.

Use the specification's fixed architecture: Python 3.12 package with uv lock, validated typed contracts, immutable Parquet snapshots queried by DuckDB, safe bounded ingestion, one fixture-specific forecasting/evaluation implementation, shared public view models, and an Astro/TypeScript static site with small React islands. Keep the public site read-only and the operator pipeline separate. Do not introduce a server/account/payments/chat platform or change the target from disposals to SuperCoach points.

Before changes, inspect `git status`, active refresh markers and current processes. This checkout had pre-existing modified/untracked files before the audit, and the data-refresh work adds deliberate CSV modifications. Never reset/clean/stash them or assume a new worktree at HEAD contains them. Work in an isolated rewrite branch/worktree where appropriate and copy authorized source data with a checksum manifest. If a live old-harness cycle is active, continue independent new-package work and defer production-harness changes until it exits, as `CLAUDE.md` requires.

Follow phases 0–9 in PLAN.md. Start with a baseline/schema/output inventory, migration map and small offline fixtures. Write meaningful failing tests before new behavior. Then implement contracts/storage, migration, source reconciliation, analytics, forecasting/evaluation, browser, optional live/editorial adapters, CI and migration rehearsal. Run phase checks and keep a compact progress ledger at `docs/rewrite/IMPLEMENTATION_STATUS.md` with completed work, exact test results, decisions and unresolved blockers. Do not stop after scaffolding or after building the UI.

The correctness requirements are essential:

- Every forecast targets an actual future `(player_id, match_id)` with a cutoff and origin. Never label a previous-match feature row as “next round.”
- Use true fixture chronology, including wildcard finals, postponed matches and final replays. Source round numbers are labels, not reliable temporal order.
- Fit preprocessing, tuning and calibration only inside allowed historical folds. Run future-data mutation tests and train/serve feature parity tests.
- Keep prospective scoring, retrospective replay and unverifiable legacy predictions distinct. Preserve full prediction precision internally and disclose sample/coverage.
- Unknown data stays unknown. No synthetic fixture dates or fallback DOBs acquire verified status; no stat coverage gap becomes a zero.
- Stable identities distinguish same-name players and historical club entities. Preserve source URLs/IDs and verified aliases.
- Source/parse failures cannot exit as a successful fresh dataset. The new refresh must detect interior gaps, upstream corrections, stale fixtures and partially completed work.
- One accepted snapshot/release manifest governs browser, Markdown, charts, CSV and ZIP. No consumer selects “latest” by file mtime or unvalidated filename.
- Keep deterministic numeric publication operational without AI credentials. Optional editorial drafts cannot bypass numeric/provenance checks.
- Nothing may publish before its complete release has passed validation. A failed build or publish keeps the previous release intact.

Implement all required browser routes and states in section 9, including predictions, player search/detail/comparison, teams, matches, history, accuracy, lists, articles, local watchlist, downloads and data status. Provide mobile/keyboard/zoom support, accessible table/chart alternatives, shareable filters, accurate freshness, loading/empty/error/stale states and correct project-base URLs. Use actual imported data for the real release and clearly labelled fixtures for demo mode; no hardcoded pretend statistics.

Keep legacy capability parity explicit. Preserve curated articles and frozen as-of facts, both distinct all-time CSV schemas, yearly ranking publication cadence, drafts/contracts/schools, era analysis, live snapshots, fan-pack asset closure and public report links. Port ranking executable values rather than stale comments. Where correctness requires changed numbers, document a methodology-version change and evidence instead of preserving a known bug.

Use the security and performance requirements from sections 11–12. No broad permission-bypass flags, untrusted executable Markdown, unsanitized HTML, unsafe formula CSV cells, arbitrary model deserialization, public secrets, source-controlled paths or unchecked redirects. Use bounded HTTP, manifest hashes, locks, atomic writes and checked subprocesses. Do not claim a security/accessibility/performance result without running the relevant check.

Required validation before handoff:

- Locked clean installation and offline demo, without GPU, private interpreter paths, network or AI credentials.
- Ruff/mypy and frontend lint/type checks; unit/contract and real-corpus integration tests; all specified temporal/identity/data-quality invariants.
- Production browser build with Playwright primary-route flows, axe, keyboard/mobile review, screenshots and base-path/deep-link tests.
- Independent reconciliation across public JSON, Markdown, chart labels/alt text, CSV and fan ZIP.
- Old/new behavior comparison on the same immutable source snapshot; documented intentional corrections.
- Full offline pipeline failure-injection, resume/concurrent-writer tests, a publication rehearsal with Git/network writes stubbed, and a local rollback/restore exercise.
- Measured payload/operation counts and reference benchmarks; matched model-versus-baseline holdout report. If no model qualifies, publish the validated named baseline with an honest report.

When existing production harness/gates/hooks change, satisfy the scratch-worktree full-cycle smoke requirement in `CLAUDE.md`; unit tests alone are not a substitute. Do not run the old real weekly harness as a quick smoke check because it can retrain, invoke agents, commit and push.

Make routine implementation decisions yourself within this specification. If an external source is unavailable, finish the offline/import/browser implementation and report the affected source; do not fake successful online refresh or disable its gate. Resolve genuine code/test failures rather than hiding them with skips. Do not automatically invoke the council or delegate just because legacy documentation discusses agents.

Finish with the changed areas, local startup commands, build locations, test/benchmark results, any genuine source-data limitations, and the migration/rollback instructions. Leave changes reviewable. Do not commit, push, deploy, create a public release or destructively prune history unless separately authorized.

---

The main specification contains the exact module layout, contracts, schema mapping, routes, CLI commands, acceptance scenarios and completion checklist. This prompt deliberately points to those documents instead of duplicating a shorter, divergent specification.
