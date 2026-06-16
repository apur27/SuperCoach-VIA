# Memory Index

- [Enforcement substrate state](project_enforcement_substrate.md) — Gap 1 + Gap 2 both MVP-live as of 2026-05-30 (committed .githooks + .claude/audit), with named open halves
- [Parallel council commits](feedback_parallel_council_commits.md) — git races/locks when 7 agents push to main at once; wait for index.lock, verify by content not command success
- [Council-stamp gate scope](feedback_council_stamp_gate_scope.md) — gate blocks ALL docs/hall-of-fame-stat-*.md without a PASS stamp incl. legacy hub; put nav links in README/news-README instead
- [Root .py import architecture](project_root_py_import_architecture.md) — all root scripts standalone; only real cross-imports are refresh chain + main/refresh_data scrapers + backtest->prediction; config.py shared via cwd=root
- [Flaky output channel workaround](feedback_flaky_output_channel.md) — when tool output truncates/duplicates, one-value-per-call + grep -c; confirm surprising bulk results standalone; don't batch independent Bash calls
- [Player-data verification quirks](project_player_data_quirks.md) — games-counter resets per season, goal_assist is singular, Ablett snr/jnr surname collision, forgotten-heroes uses inline [data]: prose not pipe columns
- [Finals doc stale round labels 2026](project_finals_doc_stale_2026.md) — afl-finals-2026.md ladder data correct but prose says "Round 8"/"7 games" vs actual Round 12; script-template bug, not hand-fixable
- [Script health backlog](project_script_health_backlog.md) — charts.py Era KeyError (needs Scientist); live_analysis_pipeline.py --help auto-starts poll loop; chart re-render churn
- [BriefBuilder defect classes](project_briefbuilder_defects.md) — round-13 cycle: sort-order (Clayton/Harvey), NaN-handling (needs Scientist convention), win-count source; + DataSentinel arithmetic slip
- [Briefs are user-initiated only](feedback_brief_user_initiated.md) — NEVER autonomously publish a coaches-strategy-corner brief; wait for explicit user instruction
- [Backtest n filtering](project_backtest_n_filtering.md) — doc headline n is late-out-filtered (~9 below raw CSV pool); gap expected, doc figure is canonical; per-round canonical run timestamps
- [Eval surface auto-update](project_eval_surface_autoupdate.md) — update_eval_surface.sh refreshes README eval + banner.svg; backtest CSVs are split per-run, MUST merge all + dedupe by (year,round)
- [No Agent dispatch tool](project_no_agent_dispatch_tool.md) — this env can't spawn the 6 council agents; run single-operator cycles w/ deterministic gates + honest stamp, never fake the 7-agent chain
- [Player finals data lag](project_player_finals_data_lag.md) — player CSVs lack current-season finals rounds though matches CSV has the GF; scope player-level premiership metrics to completed seasons
- [Matches vs player-data sync](project_matches_player_sync.md) — R10 2026 truncation FIXED + fixture-aware audit live in refresh cycle; residual: R10 quarter scores=0 lower-fidelity, 2025 R1/R2 ±1 labeling boundary
