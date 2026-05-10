# Changelog

All notable changes to SuperCoach-VIA. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html) - though this repo is mostly weekly data refreshes interleaved with model and doc work, not formal releases. Dates are calendar dates, newest first.

> **Note:** entries before this file existed are reconstructed from git history. The full commit log on `main` is canonical.

---

## Unreleased

### Added
- Fan-first repositioning - new "Start here / I want to..." navigation, fan landing page, glossary, weekly fan pack scaffolding.
- `docs/glossary.md` - footy and data terms in plain English.
- `docs/how-to-use-this-for-supercoach.md` - the honest version of what the predictions are good for.
- `docs/start-here-no-code.md` - fan landing page linking to predictions, downloads, club analysis, Hall of Fame; notes hosted dashboard as "coming soon".
- `docs/model-report-card.md` - pre-registered hit/miss methodology, commitment to weekly reporting, season-to-date numbers (MAE 4.14, 68.1% within 5, 94.2% within 10).
- `docs/weekly/round-current-2026.md` - one-page Round 9 cheat sheet (top 30 + per-club top 3, experimental v0).
- `templates/google-sheets-template.md` - 5-tab dashboard recipe for the prediction CSV.
- `.github/ISSUE_TEMPLATE/` forms - feature request, bug report, prediction feedback (non-developer friendly).
- `.github/workflows/weekly-fan-pack.yml` - Sunday 23:00 UTC scheduled action that bundles latest predictions + fan docs as a GitHub Release artifact (scaffold; does not retrain the model).
- `scripts/package_fan_pack.sh` - local equivalent of the weekly-fan-pack workflow for ad-hoc releases.
- `scripts/generate_player_cards.py` - matplotlib scaffold for per-player PNG prediction cards (top-N from CSV, with last-5 trend indicator).
- `assets/cards/` - destination directory for generated player cards.
- `CONTRIBUTING.md` - fan/dev contribution guide, plus the recommended GitHub repo topics list (afl, supercoach, fantasy-sports, australian-football, sports-analytics, machine-learning, data-science, football-data, python).

### Changed
- `docs/installation.md` restructured with For Fans / For Power Users / For Contributors sections.
- `README.md` reorganised: lead with the one-liner positioning, "Start here - I want to..." nav table, technical / data-science detail moved under a "For data scientists" heading. New "Creating a Release" section documents the local fan-pack script and `gh release create` flow.

---

## 2026-05 - 2026 season Round 9 refresh and major doc consolidation

### Added
- Collaboration contact in the Contributing section.
- Scheduled GitHub Actions workflow to sync `main` into `feature/phase2`.
- AI architecture deep dive (`docs/ai-architecture.md`) - RAG, tool router, eval harness, MCP gateway, sovereign deployment notes.
- Hall of Fame - top 10 greatest Indigenous Australian AFL players.
- `docs/data-science.md` - full technical deep dive, layered for layperson through ML practitioner.
- Coaches Strategy Corner - Round 9 Richmond vs Adelaide tactical brief, with visualisations and accuracy audit.
- Hall of Fame - all-time statistical leaders (top 20 in every major category, verified from data).
- Hall of Fame - great AFL dynasties.
- Hall of Fame - great AFL careers cut short.
- Phase 2 roadmap section in README.
- Round 9 prediction and backtest auto-refresh (8 rounds backtested - MAE 4.14, 68.1% within 5).

### Changed
- All em-dashes replaced with regular hyphens across docs (consistency pass).
- Reframed README to lead with the data-science and AI angle (removed TL;DR).
- Restructured 2026 season hub into 6 focused docs (`afl-team-analysis-2026.md`, `afl-finals-2026.md`, `afl-brownlow-2026.md`, `afl-stat-leaders-2026.md`, `afl-predictions-2026.md`, `afl-backtest-2026.md`).
- Restructured AFL insights into 6 focused docs (`afl-insights.md` hub plus 5-year team profiles, history, expert guide, coaching guide).
- Coaches Strategy Corner hub redesigned with a match index table (replaces hardcoded chart) so future briefs scale.

### Fixed
- Jonathan Brown entry - removed fabricated "vision" claim and corrected team attribution.
- `hall-of-fame-courageous.md` factual errors (Newman games and premierships, Brown goals, Voss knee count, Platten GF year, Sylvia death date, Monfries suspension year, Trengove injury type and captaincy year, Stynes record phrasing).
- Three-peat claim - Hawthorn 2013-15 was the second modern three-in-a-row, after Brisbane 2001-03.
- `CLAUDE.md` data-paths corrected for the actual `player_data` filename format.
- Anachronistic knee-generation heading - separated pre-arthroscopy era from modern players.
- Rounded predictions doc to use today's date instead of CSV filename timestamp.
- Invalid stars/forks badges (removed `style=social` parameter).

### Documentation
- Added explicit data-verification rule to `CLAUDE.md` - always check repo data before writing AFL stats.
- Tightened Hall of Fame courageous-players criteria - on-field physical and mental courage only.

---

## 2026-04 - Phase 2 groundwork and prediction pipeline maturation

### Added
- `pyproject.toml`, pinned dependencies, `src/supercoach/` package layout.
- GitHub Actions CI - first workflow (lint and tests).
- First test scaffolding - smoke tests for prediction loading.
- Scientist agent system prompt (`/.claude/agents/scientist.md`) - methodology rules: inspect-before-transform, baseline-first, leakage-aware.

### Changed
- Refresh script (`refresh_and_rank.sh`) wired end-to-end - top-100 markdown step, 5-year profiles to its own file, auto git push.
- Top-100 ranking - rank-based formula with season-count bonus and within-cohort z-scoring.

### Fixed
- All-time top-100 - corrected leakage in earlier ranking script (era-completeness shrinkage now applied).

---

## 2026-03 and earlier - foundation

The repo started life as a single-machine prediction pipeline for the author's own SuperCoach league. Highlights from this period:

- Walk-forward backtest framework using GroupKFold (player-grouped) cross-validation.
- LightGBM + HistGradientBoosting + RandomForest ensemble with post-hoc out-of-fold linear calibration to correct top-end compression.
- Initial data scrape from AFL Tables - matches and per-player per-game stats from 1897 to present.
- README hub with Hall of Fame, captains, coaches, courageous players.
- Banner artwork and chart aesthetic standardised.

For exact attribution, see `git log` on `main`.

---

## How to read this changelog

- **Added** = new feature, doc, or capability.
- **Changed** = behaviour or doc that already existed but moved or was rewritten.
- **Fixed** = a bug, factual error, or broken link.
- **Documentation** = doc-only change (no code or model behaviour change).
- **Deprecated / Removed** = something being phased out (none yet).

If you want to know exactly which commit introduced a change, run `git log` and grep for the area you care about.
