# Rewrite implementation status (ledger)

Owner of this ledger: Gaffer (integration). Spec: [PLAN.md](PLAN.md). Evidence: [AUDIT.md](AUDIT.md), [DATA_REFRESH.md](DATA_REFRESH.md).
Session 2026-09-25 (cloud continuation): owner authorised commits/pushes to branch `rewrite/wip` only (never main, no PR/deploy/release/force-push).

## Phase 0 — baseline (2026-09-24)

- HEAD `b4ce74770b20e670a43a3165913dc4e188a25bad`; 777 dirty/untracked paths preserved (never reset/stashed/cleaned).
- Old-harness marker `.claude/audit/last_refresh_status.json` = `{"phase":"4","exit_code":0,"round":"25"}`; no `weekly_refresh`/`refresh_and_rank` process running → no active cycle (CLAUDE.md §6.1).
- Migration input v0 manifest: `docs/rewrite/evidence/input-v0-manifest.json` — 27,516 files under `data/`, `config/`, root `all_time_top_100.csv`; 92,066,340 bytes; aggregate sha256 `c98aa359…07867058`.
- Accepted refresh intact: all 756 files in `evidence/refresh-manifest.json` match their `after_sha256`.
- Rewrite is built **in place** in new paths only (`src/`, `web/`, `tests/scvia/`, `schemas/`, new `config/*` files, new docs). Legacy data/code/harness files are read-only inputs; a separate worktree at HEAD would lack the refreshed data, and in-place avoids a 176 MiB copy while the v0 manifest pins the input bytes.
- Toolchain: uv 0.12.18, CPython 3.12.3, Node 22.23.2 / npm 12.0.2. Legacy venv `/home/abhi/sourceCode/python/coding/.venv` does **not exist** on this machine (pre-commit hook default interpreter is missing; `COUNCIL_PYTHON` must be set).
- Legacy suites re-run in the new locked env (`uv sync --locked --group dev --group legacy --extra ml`):
  - `pytest tests/unit -m "not integration"` → **559 passed, 41 skipped, 17 warnings, 15.69s** (audit: 555/45 with pandas 2.3.3).
  - `pytest tests/integration -m integration` → **20 passed, 1 failed** (`test_top100_chart_reproduces_byte_identically`: renderer-environment byte mismatch, same as audit; not a data error). Working tree unchanged by the run (status md5 identical before/after).

## Decisions

| # | Decision | Reason |
|---|---|---|
| D1 | New tests under `tests/scvia/{unit,contract,integration,performance}` instead of `tests/{...}` | Legacy Phase-3d gate runs `pytest tests/integration`; putting new tests there would change a live gate (CLAUDE.md §6.2). Directory conftest auto-marks integration/performance so the pre-commit fast tier excludes them. |
| D2 | One lock for both packages: `uv.lock` with groups `dev`, `legacy` and extra `ml`; pandas pinned `<3` | Reproducible legacy baseline/rehearsal (audit: pandas 3 breaks legacy collection); new code avoids pandas-3-removed APIs. |
| D3 | Snapshot ID = sha256 of semantic manifest content (not creation time) | Identical input re-import is idempotent (Phase 2 gate). |

## Ownership (parallel work, disjoint files)

| Area | Owner | Files |
|---|---|---|
| Contracts, storage, settings, CLI, pipeline, publish/*, docs, integration | Gaffer | `domain/schemas.py` (non-table sections), `publish/view_models.py`, `storage/*`, `settings.py`, `cli.py`, `pipeline.py`, `publish/*` |
| Legacy import, identity, validation, canonical tables | Scientist (import) | `domain/{ids,season}.py`, `ingest/{legacy,reconcile}.py`, `TABLES` in `domain/schemas.py`, `config/{team_aliases,venue_aliases}.csv`, `config/coverage.yaml` |
| Analytics parity | Scientist (analytics) | `domain/metrics.py`, `analytics/*`, `config/ranking_legacy_v1.toml` |
| Forecasting/evaluation | Scientist (ml) | `ml/*` |
| HTTP/source adapters/refresh, live, editorial | general-purpose (ingest) | `ingest/{http,afltables,drafts,contracts,fanfooty,refresh}.py`, `live/*`, `editorial/*`, `config/source_policies.toml` |
| Browser + CI | general-purpose (web) | `web/**`, new `.github/workflows/*` |

## Test results log

| When | Command | Result |
|---|---|---|
| Phase 1 | `uv run --locked pytest tests/scvia -q` | 36 passed (schemas, storage, settings) |
| Gaffer publish slice | `uv run --locked pytest tests/scvia -q -m "not integration"` | 96 passed (+ publish safety, release lifecycle, articles, charts, CLI shell, demo corpus) |

## Interruption log

- 2026-09-25: all five sub-agents (import, analytics, ml, ingest, web) and Gaffer were stopped by an API 429 session limit before writing any files. State check: no new files in `src/`, `tests/scvia/`, `web/`; no `var/`. Worktrees under `.claude/worktrees/agent-{a0656b…,a2c502…,a8be40…}` date from 2026-06-16 (older unrelated Scientist work; only agent-memory notes uncommitted) — not part of this rewrite, left untouched. Agents resumed via SendMessage with context retained.

## Gaffer slice progress (2026-09-25)

Done (tests first, all green): `publish/bundle.py` (formula-safe CSV, deterministic contained ZIP), `publish/content.py` (Markdown -> nh3 allowlist HTML, link/asset mapping), `publish/reports.py` (fail-closed marker replacement), `publish/articles.py` + `config/public_content.toml` (84 explicitly listed docs; rewrite/agent/operator docs rejected), `publish/charts.py` (pyplot-free, scoped rc, alt text from same values), `publish/release.py` (staging + atomic rename, closure/hash/schema/allowlist/private-content/reference validation, publish with receipts, failed publish keeps previous, rollback), `cli.py` shell (doctor, schemas, validate-release, publish, rollback, preview), `demo.py` (DEMO corpus with section-13 edge cases).
Pending on agents: release builder from snapshot (needs analytics + ml APIs), pipeline.py, remaining CLI commands, `scvia demo`.
- 2026-09-25 (second 429, resets 4pm AEST): all five agents stopped again mid-slice; Gaffer stopped mid-wait. On disk at stop:
  import — `domain/{ids,season}.py`, config aliases/coverage, 50-file D-series fixture corpus, tests `test_ids/test_season/test_legacy_import` (ingest/legacy.py not yet written; no var/agent-import snapshot);
  analytics — `domain/metrics.py`, `analytics/{players,rankings}.py`, `config/ranking_legacy_v1.toml`, tests (teams/eras/awards/lists pending);
  ml — `ml/{features,splits,train,bundles,evaluate,models}.py`, 27 tests passing per agent; predict.py next;
  ingest — `ingest/{http,afltables}.py`, `config/source_policies.toml`, tests; refresh/drafts/contracts/fanfooty/live/editorial pending;
  web — Astro project scaffolded (`web/` with package-lock, integrations, scripts, src, tests).
  Gaffer — added `publish/resources.py` (match index/detail, game logs, player index; 5 tests), ruff+mypy clean on all Gaffer files (line-length 120; justified per-line noqa only). All agents resumed via SendMessage.
- Third interruption (weekly 429 limit, reset Sep 28 3pm AEST): all five agents stopped again. On disk vs previous entry:
  import — `ingest/legacy.py` written; `ingest/reconcile.py` + `test_reconcile.py` in progress; `var/agent-import/` has fragments, snapshots and `import-report.json` but no promoted `current.json` yet;
  analytics — all six modules written (`players, rankings, teams, eras, awards, lists`) with tests; was running ruff/mypy;
  ml — `predict.py` + `test_ml_evaluate/predict` added; was starting the real-corpus timing probe (`var/agent-ml`);
  ingest — `ingest/{drafts,contracts,fanfooty,refresh}.py` + tests; refresh tests in progress; live/ and editorial/ still empty;
  web — `web/src/{components,islands,layouts,lib,pages,styles}`, `web/tests/{e2e,fixtures,unit}`; no `scvia-*.yml` workflows yet.
  No integration/performance tests landed yet. All agents resumed via SendMessage.

## Slice reports

### Ingest / live / editorial — COMPLETE
- Files: `ingest/{http,afltables,drafts,contracts,fanfooty,refresh}.py`, `live/{monitor,commentary}.py`, `editorial/{evidence,adapter,verify}.py`, `config/source_policies.toml`, unit tests + `tests/scvia/integration/test_source_check.py`, fixtures `tests/scvia/fixtures/raw/`.
- Results (agent-reported): own unit tests 137 passed / 6.3s; source-check integration 1 passed; ruff + mypy strict clean.
- Authorized source check (1 request total): GET https://afltables.com/afl/seas/2026.html → HTTP 200, 239,669 bytes, sha256 `6c0d5a3b…2ac7fc0`, Last-Modified 2026-09-20 01:40:42 GMT; parsed 217 completed matches, latest 2026-09-19, 0 scheduled, GF not yet listed; identical to `data/matches/matches_2026.csv` for all 217 (dates, pairs, stages, scores); fixture diff vs import candidate `sha256:79e753bb…`: 0 new/missing/changed.
- Limits: match-detail / player-page / Wikipedia-draft / DraftGuru parsers verified only against synthetic markup (1-request budget); `refresh_sources` returns upserts (Gaffer merges into a snapshot); first real refresh would need ~430–500 requests; DNS-rebinding window between resolve and connect documented.

### Analytics — COMPLETE
- Files: `domain/metrics.py`, `analytics/{rankings,players,teams,eras,awards,lists}.py`, `config/ranking_legacy_v1.toml`, 88 unit tests, `tests/scvia/integration/test_analytics_parity.py`.
- Parity on identical bytes: all-time top-100 100/100 ranks, max |Δ| 4.4e-16; 130/130 yearly lists; 13,367 career-games equal; era stats Δ≈3e-14. Legacy itself is non-deterministic on ties (91/100 between two legacy runs); `legacy_v1` tie-breaks by player_id.
- Against import candidates: 68/100 ranks — traced to an import defect (3 extra-time finals quarantined as `unresolved_club`, 130 player rows) → routed back to import agent. Integration: 7 passed, 3 failed (all three from that defect).
- Timings: `run_legacy_v1` 0.6–1.4 s (legacy script ≈55 s); whole-corpus player stats bundle 1.95 s.
- `scipy` declared as a direct dependency on request (already present in lock via scikit-learn).

## Blockers

- **B1: RESOLVED 2026-09-25 by an owner-authorised bounded repair fetch.** 6 AFLTables requests in total, against a cap of 10: season page, 2 match pages, 3 player pages, one attempt per URL. Run 1 failed closed on real match-page markup (final score `13.12.<b>90</b>`); the parser was fixed against the archived pages. 14 `player_games` rows were added: Perez 7, Dalton 5 (new identity `src:afltables:J.Jack_Dalton1`, DOB 2007-04-05, not the 1876 namesake), Brodie 2. 14 quarantined 2026 lineup tokens were re-linked. The real candidate now validates PASS with 0 blocking, 0 error and 18 historical warnings. Evidence is in `docs/rewrite/evidence/b1/` (URLs, hashes, raw payloads, rows, driver). Offline re-application re-verifies hashes and re-parses: `ingest.refresh.replay_repair_evidence`.

### Import / identity / validation — COMPLETE (follow-up fix in progress)
- Files: `domain/{ids,season}.py`, `ingest/{legacy,reconcile}.py`, TABLES additions, `config/{team_aliases,venue_aliases}.csv`, `config/coverage.yaml`, 110 unit tests, `tests/scvia/integration/test_legacy_import_real.py` (7 passed, 62.7 s), 53-file fixture corpus.
- Real corpus: import 36.4 s + validate 2.9 s, peak RSS 1.88 GiB (budget 120 s / 2 GiB; first run was 2.09 GiB, fixed). Re-import idempotent. Candidate `sha256:79e753bb6e0233d36fcf818fd48c8016e4cefd838792b40e6e49c5ccbb4ff3c3` (unpromoted; see B1).
- Accounting: matches 17,055 read = 17,052 + 3 quarantined; player_games 695,464 = 695,331 + 133; lineup tokens 99.989% resolved; 142 files ignored-with-reason; no unrecognised files.
- Validation: PASS on schema/keys/FK/stages/scores/stats/coverage/exceptions; FAIL on reconciliation (13 blocking = the 2026 gaps in B1). 18 historical goal mismatches (warnings); 124 players with counter > rows (disclosed, no rows invented). 97% of legacy player dates are synthetic "March 1 + weeks" → `inferred`, never used to link.
- Follow-up (routed): 3 finals (1994 QF, 2007 SF, 2017 EF) are malformed in the source CSV (extra-time score written into team_2_team_name). Instruction: create the match with the opponent taken from unanimous player-row evidence, null scores, status unknown, and an error-severity `source_row_malformed` issue; link the 130 genuine player rows; never guess scores.
