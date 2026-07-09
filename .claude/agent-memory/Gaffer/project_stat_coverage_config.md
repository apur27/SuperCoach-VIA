---
name: stat-coverage-config
description: Canonical stat-coverage era boundaries now live in config/stat_coverage_eras.yaml (gate config); several proposed years were wrong and needed Scientist re-verification
metadata:
  type: project
---

The stat-coverage era table (which year each player stat first has real data) is now a machine-readable gate config at `config/stat_coverage_eras.yaml`, read by DataSentinel and Skeptic. It supersedes the prose table in `.claude/agent-memory/Scientist/data_stat_coverage_eras.md`. Committed a6c1e646a (2026-07-03).

**Why:** promoted out of agent memory so the era gate is deterministic, versioned, and Scientist-signed-off — same pattern as `config/coach_names.txt` and `config/fanfooty_schema.yaml`. Guarded by `tests/unit/test_gate_config.py`.

**How to apply:**
- Never edit this file without Scientist sign-off — it is a read-only gate config.
- When promoting ANY data-derived config, route the numbers to the Scientist to re-verify against `data/` first — do NOT trust a proposed table. In this promotion the proposal was wrong on 4 of 20 stats: `rebound_50s`/`clangers` are 1998 (not 1999), `goal_assist` is SINGULAR and starts 2003 (not `goal_assists` 1999), `behinds` is 1965 (not 1897 — only `goals` reaches 1897).
- `brownlow_votes` gate = 1984 (first continuously-populated year). Real but isolated 1931-1934 fragment then a 1935-1983 dead zone; keying the gate to 1931 would falsely certify the empty span. This was a methodology call, taken on the Scientist's recommendation.
- Residual: `scrapers/game_scraper.py` (~line 243) still has a code COMMENT pointing at the old memory path — harmless (not a load path), Scientist-owned, left untouched.

Related: [[consult-surveyor]], the gate-config promotion pattern.
