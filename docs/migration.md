# Migration: legacy capability parity registry

*Operator/engineering document for the SuperCoach VIA rewrite (spec: [rewrite/PLAN.md](rewrite/PLAN.md) §6). It carries no player statistics, so it has no data tags.*

Each row maps a legacy entry point or output to its replacement in the `supercoach_via`
package, together with the check that proves it and its status:

- **ported**: same behaviour and numbers, with a parity or behaviour test.
- **corrected**: behaviour changed on purpose. The reason and evidence are linked.
- **archived**: kept as a read-only historical archive, not regenerated.
- **gap**: the replacement analytics exist but the release does not yet publish them. Listed so nothing disappears silently.

Legacy code stays in the repository as a reference until each row is ported, corrected or archived (Phase 9).

## Ingestion and data

| Legacy | Replacement | Check | Status |
|---|---|---|---|
| `refresh_data.py`, `scrapers/game_scraper.py`, `scrapers/player_scraper.py` | `ingest.afltables`, `ingest.refresh`, `scvia refresh --plan / --data-only` | `tests/scvia/unit/test_afltables.py`, `test_refresh*.py`, `integration/test_source_check.py` | corrected: bounded HTTP policy, fail-closed on source/parse failure, `WF` resolved from the season table (DATA_REFRESH findings) |
| Legacy CSV corpus (`data/`, both all-time CSVs) | `ingest.legacy.import_legacy`, `scvia import-legacy` | `integration/test_legacy_import_real.py` (every file accounted for, per-season row reconciliation, idempotent) | ported, with quarantine for duplicate identities and malformed rows |
| `scripts/fix_synthetic_dates.py`, round-to-weeks dates | `date_quality=inferred` vs `fixture_verified`; never used to link | `test_legacy_import.py`, `test_ml_features.py` | corrected: synthetic dates are no longer presented as verified |
| `scripts/phantom_row_validator.py`, `scripts/match_completeness_gate.py`, `scripts/check_round_settled.py` | `ingest.reconcile.validate_dataset` | `test_reconcile.py`, `test_legacy_import_real.py::test_validation_structure_and_known_gaps` | ported |
| `scrapers/draft_scraper.py`, `rookie_draft_scraper.py`, `draftguru_scraper.py`, `school_classifier.py` | `ingest.drafts`, `analytics.lists`, `/lists/` | `test_drafts.py`, `test_analytics_lists.py` | ported; parsers verified on synthetic markup only (request budget) |
| `scrapers/free_agency_scraper.py` | `ingest.contracts`, `analytics.lists.contract_rows` | `test_contracts.py` | ported; observations are labelled, not guaranteed current contracts |
| `scrapers/squad_builder.py` | `roster_memberships` table + `analytics.lists` | `test_analytics_lists.py` | ported |
| B1 current-season gaps (2026) | `ingest.refresh.repair_player_pages` + archived evidence replay | `docs/rewrite/evidence/b1/`, `test_refresh_repair.py` | corrected: repaired from source, never typed by hand |

## Analytics

| Legacy | Replacement | Check | Status |
|---|---|---|---|
| `top_players_comprehensive.py` (all-time and yearly top 100) | `analytics.rankings.run_legacy_v1`, `config/ranking_legacy_v1.toml` | `integration/test_analytics_parity.py` (100/100 ranks, max score delta 4.4e-16 on identical bytes; 130/130 yearly lists) | corrected: legacy tie order is non-deterministic, while `legacy_v1` tie-breaks by `player_id` |
| Root `all_time_top_100.csv` (biography) vs `data/top100/all_time_top_100.csv` (scores) | `biography_export_rows` → `downloads/all-time-top-100.csv`; `numeric_export_rows` → `downloads/all-time-top-100-scores.csv` | `test_analytics_rankings.py` | ported; the two shapes stay distinct |
| Yearly top-100 CSV (season-end cadence) | `yearly_export_rows` → `downloads/yearly-top-100-<season>.csv` for the latest final season; `history/yearly_top_100/<season>.json` for every season | `test_builder.py::test_yearly_top100_csv_only_for_the_latest_final_season` | ported: a provisional season is listed as unavailable until it is final |
| `update_team_analysis.py`, team five-year profiles, `scripts/build_conceded_stats.py` | `analytics.teams` (`ladder`, `team_form`, `five_year_ladder`, `finals_pathway`, `conceded_stats`) → `teams/<club>/<season>.json`, `reports/team-analysis.md` | `test_analytics_teams.py` | ported; ladder and finals heuristics are fixture-aware. **gap**: conceded-stat tables are computed but not yet published |
| `supercoach/era_based_statistical_analysis.py`, `scripts/era_boundary_threshold.py` | `analytics.eras` (`era_stats`, `yearly_trends`, `adjacent_era_tests`, `team_scoring`) | `test_analytics_eras.py`, `test_builder.py::test_era_summary_and_brownlow_proxy_downloads`; parity delta ~3e-14 | ported as `downloads/era-summary.csv` (in the fan pack; unrecorded stats blank). **gap**: no browser view yet |
| Brownlow proxy (`docs/afl-brownlow-2026.md` generator) | `analytics.awards.brownlow_proxy` with a versioned ineligibility source | `test_analytics_awards.py`, `test_builder.py::test_era_summary_and_brownlow_proxy_downloads` | ported as `downloads/brownlow-proxy-<season>.csv`, labelled as an index with observed votes kept separate. **gap**: no browser view yet |
| Career, stat and season leaders; HOF stat pages | `analytics.players` (`career_leaders`, `games_leaders`, `single_season_leaders`, `season_leaders`) → `history/*` | `test_analytics_players.py` | ported, with observed-denominator and coverage disclosure |
| `scripts/check_hof_numbers.py`, `update_hof_pages.py` | Curated HOF pages imported as frozen articles; the numbers come from `history/*` | `test_articles.py` | archived (frozen as-of content) + ported (numbers) |

## Forecasting and evaluation

| Legacy | Replacement | Check | Status |
|---|---|---|---|
| `supercoach/prediction.py`, `prediction_cpu.py`, `scripts/feature_engineering.py` | `ml.features` (single feature engine), `ml.train`, `ml.predict`, `scvia forecast` | `test_ml_features.py` (M01–M05), `integration/test_ml_real_corpus.py` | corrected: targets are real future `(player, match)` rows with cutoffs. A historical row is never relabelled as "next round" |
| `supercoach/prediction_accuracy.py`, `backtest.py`, `scripts/backtest_completeness.py` | `ml.evaluate` (`evaluate`, `score_archive`, `replay`), `/accuracy/` | `test_ml_evaluate.py` (M07–M09) | corrected: prospective, replay and `legacy_unknown` never pooled; full precision kept |
| Three-column predictions CSV | `downloads/predictions-legacy.csv` + `predictions-legacy.manifest.json` sidecar; rich `downloads/predictions.csv` | `test_builder.py`, `test_reconcile_outputs.py` | ported, with the sidecar |
| Legacy prediction archives | `legacy_predictions` table → `legacy_unknown_report` | `integration/test_ml_real_corpus.py::test_legacy_archive_summary_is_separate` | archived: provenance unknown, excluded from headline accuracy |

## Publication

| Legacy | Replacement | Check | Status |
|---|---|---|---|
| `refresh_readme.py`, `generate_readme_charts.py` | `publish.reports` templates, `publish.charts` | `test_charts.py`, `test_publish_safety.py` | ported: charts are pyplot-free, with scoped rc, and alt text comes from the same values |
| `scripts/generate_player_cards.py`, `generate_weekly_cheat_sheet.py`, `generate_player_status.py` | `/player/`, `/predictions/` routes and view models | web e2e `players.spec.ts`, `predictions.spec.ts` | corrected: browser routes replace static per-player Markdown |
| `scripts/package_fan_pack.sh`, `.github/workflows/weekly-fan-pack.yml` | `publish.builder.build_fan_pack` → `downloads/fan-pack.zip` (deterministic, with checksums) | `test_builder.py`, `test_reconcile_outputs.py` | ported: a missing dependency blocks packaging |
| `scripts/fetch_live_match.py`, `live_match_monitor.py`, `live_analysis_pipeline.py` | `live.monitor`, `live.commentary`, `/live/` | `test_live.py` (L01–L04) | ported: generic per-match monitor with restart-safe state |
| News, strategy and HOF Markdown | `publish.articles` + `config/public_content.toml` (explicit list) | `test_articles.py` | archived: frozen as-of, sanitized, original URLs mapped |
| Council agents, `check-council-stamp.sh`, `record-sentinel-verdict.sh`, `skeptic_*` | `editorial.{evidence,adapter,verify}` (optional; never gates numeric releases) | `test_editorial.py` | corrected: claims reference fact objects, and `PASS_WITH_CONCERNS` is not numeric certification |
| `scripts/weekly_refresh.sh`, `refresh_and_rank.sh`, `git_commit_safe.sh` | `pipeline.py` staged DAG + `scvia` commands; `scvia publish` is the only mutation | `test_pipeline.py` (never touches Git; locked writer; failed stage keeps pointer) | ported/corrected; the legacy harness stays operational until Phase 9 switch-over |

## Intentional numeric differences

1. **Rankings tie order.** Legacy is non-deterministic on exact score ties (91/100 identical ranks between two legacy runs on the same bytes), while `legacy_v1` breaks ties by `player_id`. Scores are identical within 4.4e-16.
2. **Rankings after B1.** The 14 repaired 2026 rows (Perez, Brodie, Dalton) enter 2026 season totals. Legacy never saw them.
3. **Coverage-aware means.** New general statistics use observed denominators and return coverage. The legacy imputation is retained only inside the named `legacy_v1` ranking methodology.
4. **Forecast accuracy.** Headline metrics are pooled and player-game weighted over explicitly originated rows. Old headline numbers used a different denominator and are not compared directly.
