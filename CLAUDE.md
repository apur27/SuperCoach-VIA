# Behavioral Guidelines

Adapted from Karpathy's CLAUDE.md. These bias toward caution over speed — use judgment on trivial tasks.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 5. Test-Driven Development (TDD) — Non-Negotiable

**Write the test first. Make it fail. Then make it pass.**

Every code change to this repo requires tests. No exceptions.

### Process
1. Before writing any new function or modifying existing logic, write a failing test that defines the expected behaviour.
2. Implement the minimum code to make the test pass.
3. Run the full test suite to confirm no regressions.
4. Commit tests alongside the implementation — never separately after.

### Test location
- Unit tests: `tests/unit/test_<module_name>.py`
- Run with: `/home/abhi/sourceCode/python/coding/.venv/bin/python -m pytest tests/ -v`

### Rules
- **No network calls in unit tests** — mock all HTTP (use `unittest.mock.patch` or `responses`)
- Tests must run in under 10 seconds total
- Cover: happy path, edge cases, error/failure paths (404, malformed input, empty data)
- For scraper/audit functions: always mock the HTTP layer and test the parsing and logic separately
- For data pipeline functions: use `tmp_path` fixtures for temp CSV files — never touch real data files

### Scope
This applies to all agents (Scientist, BriefBuilder, DataSentinel, FootyStrategy). When an agent writes code, it must also write the tests. The Gaffer will not accept a code commit without corresponding tests.

---

# Project-Specific Rules

## README news block — hard limit of 2 entries

The `<!-- NEWS-LATEST-START -->` / `<!-- NEWS-LATEST-END -->` block in `README.md`
must contain **at most 2 entries** at all times. The full news archive lives in
`docs/news/README.md`.

When adding a new news entry to the README block:
1. Add the new entry at the top (most recent first).
2. Remove the oldest entry if the count would exceed 2.
3. The removed entry must already be present in `docs/news/README.md` before you delete it from README.

The weekly harness (`scripts/weekly_refresh.sh`) auto-enforces this via the
`enforce_news_limit` function — any manual publish must follow the same rule.

---

## Data verification rule - AFL stats

**This repo contains the complete AFL match and player statistics history (1897–present).**

Before writing any player stat into a document - games played, goals, Brownlow votes, premierships, career averages - you MUST verify it against the actual data files in this repo. Do NOT rely on training-data memory or general knowledge for specific numbers.

### Where to look

| Stat type | Location |
|-----------|----------|
| Per-player per-game stats | `data/player_data/<surname>_<firstname>_<DDMMYYYY>_performance_details.csv` |
| Match results and team scores | `data/matches/matches_<year>.csv` |
| All-time aggregated rankings | `all_time_top_100.csv` (root) — **canonical source of truth**, written by the pipeline. `data/top100/all_time_top_100.csv` is a committed copy — read either, but the root file is authoritative. Never edit either by hand. |

### How to verify a specific player

```python
import pandas as pd, glob, os

VENV_PYTHON = "/home/abhi/sourceCode/python/coding/.venv/bin/python"
PLAYER_DIR = "data/player_data"

# Find a player's performance file (partial name match - files are named surname_firstname_DDMMYYYY_performance_details.csv)
files = glob.glob(f"{PLAYER_DIR}/*newman*performance*.csv")
if files:
    df = pd.read_csv(files[0])
    print(f"Games: {len(df)}")
    print(f"Goals: {df['goals'].sum()}")
    print(f"Disposals: {df['disposals'].sum()}")
    print(f"Years: {df['year'].min()}–{df['year'].max()}")
```

Run with: `/home/abhi/sourceCode/python/coding/.venv/bin/python`

### Non-negotiable

- Never write a specific games total, goals tally, or Brownlow count without first reading it from the data.
- If the player's data file is missing or the stat is genuinely unavailable (pre-1965 incomplete records), say so explicitly and tag the claim `**[historical record - unverified in data]**` rather than inventing a number.

---

## Data tag system

Any markdown document in this repo that contains specific player or match statistics **must** tag every number with one of the repo's verification tags. The canonical spec is `docs/data-tag-spec.md`.

- `**[data]**` — number sourced from a CSV in this repo. Bold asterisks required; plain `[data]` is invisible to DataSentinel and will FAIL the gate.
- `**[historical record]**` — public record (afltables, AFL.com.au) not in local CSVs.
- `**[historical record — unverified in data]**` — genuinely unavailable; explicitly flagged.

**Exception**: `docs/hall-of-fame-top100.md` profile prose does not use inline tags — numbers there are verified by the deterministic `check_top100_consistency()` gate. All other docs must tag.

If you edit a gated doc (anything with `<!-- council-pipeline:` in it), run DataSentinel or the pre-commit hook before pushing, or the commit will be rejected under `AUDIT_ENFORCE=1`.

---

## Serialize writes to main

Only the harness scripts (`refresh_and_rank.sh`, `weekly_refresh.sh`) commit and push pipeline outputs to `main`. Never commit pipeline-generated files (player CSVs, prediction CSVs, HOF docs, stat pages, charts) directly from a REPL, notebook, or ad-hoc agent run — that bypasses the gate chain and the allowlist. If you need to ship a one-off fix to a gated doc, route it through the appropriate council agent and wait for a DataSentinel PASS.
