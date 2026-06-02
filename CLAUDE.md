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
| All-time aggregated rankings | `all_time_top_100.csv` and `data/top100/all_time_top_100.csv` |

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
