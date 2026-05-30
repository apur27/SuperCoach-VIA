---
name: Backtest must run incrementally — never restart from scratch
description: CRITICAL repeated failure: always check docs/afl-backtest-2026.md first; only run backtest for rounds NOT already in the doc
type: feedback
---

**RULE: Before running backtest.py, read docs/afl-backtest-2026.md and find the last round already backtested. Only run the missing rounds.**

**Why:** The full walk-forward backtest takes ~5-6 hours on CPU (one round ~40 min). Re-running R1–R10 when only R11 is missing wastes hours and has made the user repeat this correction at least twice — once the week prior, once on 2026-05-18. This is a serious repeated failure.

**How to apply:**
1. Read `docs/afl-backtest-2026.md` — the per-round table shows which rounds are already done.
2. Find the last backtested round (e.g. "10 rounds backtested · R1–R10").
3. Run ONLY the new round(s): `python backtest.py --start-year 2026 --start-round 11 --end-year 2026 --end-round 11`
4. Never use `--end-round auto` when the doc already has results — auto picks the latest played round which re-runs everything.
5. After the run, update the doc: add the new row to the round table, update cumulative summary, add notable misses row, push.

**Command template for next new round (replace N with the missing round number):**
```
/home/abhi/sourceCode/python/coding/.venv/bin/python backtest.py \
  --start-year 2026 --start-round N --end-year 2026 --end-round N
```
