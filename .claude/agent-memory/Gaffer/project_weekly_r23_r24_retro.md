---
name: weekly-r23-r24-retro
description: 2026-08-11 R23→R24 refresh — Phase 3d chart-reproducibility abort, isolated to ambient matplotlib state (BL-17), manual Phase-4 completion, clean ship
metadata:
  type: project
---

2026-08-11 weekly refresh (R23 completed → R24 predicted). Shipped
`e2947dcb5` (Phase 1) → `8883d5285` (Phase 4, completed manually) → `4c144e751`
(completion manifest + BL-17).

**Stale-marker diagnosis is now a standard preflight step, and it paid.** The cycle
opened with `last_refresh_status.json` reading `{"phase":"1","exit_code":1}` from
2026-08-05. That was a REAL failure, not noise — but it was already fixed. Proof was
ordering plus content: the failure stamped 11:28, while `f49792429` ("Fix the top-30
table lagging a round") landed 13:48 the same day with a test and a DataSentinel PASS
at content hash `9f28ec05`, and that hash still matched the doc at HEAD. **A terminal
status file is evidence about one run, not about the repo.** Read it against the commits
that came after it before deciding whether to launch — don't treat exit_code=1 as a
standing blocker, and don't wave it away either.

**What broke: Phase 3d, `test_top100_chart_reproduces_byte_identically`.** Everything
upstream was green and Phase 1 had already pushed, so the abort stranded only Phase 2/3
outputs. The guard is BL-04's, and it worked.

**The isolation, and why the prescribed diagnosis was wrong.** Both BL-04 and the test's
own docstring assert this failure "means the RENDERING environment changed (typically a
matplotlib or font upgrade)". It did not. Three consecutive clean-interpreter renders of
`generate_top100_chart()` were byte-identical (`691acd8f…`, 174,837 bytes), and that
render — from CURRENT data — matched the blob committed at `30b2c174a` exactly. So
rendering today's data reproduces the prior vintage: **neither the renderer nor the data
moved.** What differed was the *process*: in-pipeline Phase 1 produced 175,862 bytes,
a fresh interpreter 174,837. That is ambient matplotlib state (rcParams/style/font cache)
left by an earlier chart generator in the same run. Filed as **BL-17** (Scientist),
which also asks for the docstring's step 1 to be corrected — a wrong recovery hint in a
live gate sends the next operator down the wrong path. BL-04 fixed the artifact and
shipped the guard; it never fixed the cause, so the cause re-fired.

**Reusable: don't accept a gate's own explanation of itself.** The docstring named the
cause confidently and was wrong. The two-render + prior-vintage experiment cost about a
minute and settled it. See [[prove-with-the-right-file]] — same discipline, and the same
docstring explicitly warns not to prove this one by diffing `all_time_top_100.csv`,
which is the bio file and carries no scores.

**Manual Phase-4 completion worked again (second validated use).** The harness still has
no resume-from-phase. Re-running the whole cycle to clear a chart byte-diff would have
re-scraped, re-fit, written a second R24 prediction vintage and a second R23 backtest
vintage, and made FootyStrategy rewrite an already-gated recap. Instead: fix the artifact,
re-run both test tiers, then stage the Phase-4 `git add` allowlist **verbatim from
`scripts/weekly_refresh.sh`** and commit via `git_commit_safe.sh`. Confirm the exclusions
survive — `data/top100/yearly/year_2026.csv` and `.claude/agent-memory/**` are
deliberately left dirty every week. See [[project_weekly_r20_retro]].

**A manual completion must replicate the harness's TAIL, not just its commit.** After the
Phase-4 push, `weekly_refresh.sh` does two more things that an abort skips, and I initially
missed both:
- **Completion sentinel** (line ~455): writes `last_refresh_complete.json` with the round.
  Chronicler is documented to detect a partial run by comparing this sentinel's round against
  the max round in the data, so leaving it frozen (it still read round 21 from 2026-07-20)
  makes a completed cycle look partial. QA caught this. I had talked myself out of writing it
  on the grounds that it would fabricate a harness success — wrong distinction: the sentinel
  records that a CYCLE completed, not that the harness ran cleanly, and
  `last_refresh_status.json` independently preserves the honest abort record. Write it.
- **Backtest mark** (line ~464): `backtest_completeness.py mark`. This IS wired into the
  harness — an earlier grep of mine for `completed_runs` missed it because the harness invokes
  the script by name, not the filename. Grep for the SCRIPT, not the data file it writes.
  Run `status` first regardless: `mark` blesses every on-disk run, so it is only safe when the
  sole orphan is the run you just pushed (it was: `20260811_102810`). Per
  [[project_backtest_completion_manifest]]; `last-round --year 2026` then reads 23.

Checklist for any future manual completion: stage the allowlist → commit → push → **mark** →
**write the sentinel**.

**Gates:** DataSentinel PASS on `afl-backtest-2026.md` (the hop that stranded the prior
cycle) and PASS on `afl-insights.md`; Skeptic PASS_WITH_CONCERNS on the recap (six prose
findings, none fatal — recurring shapes: evidentiary weight on an n=1 pairing, signed
average margin mislabelled as "average winning margin", model's own "experimental v0"
status dropped; route to FootyStrategy if they recur a third time); phantom-row PASS;
match-completeness PASS; HOF numeric gate PASS. Unit tier 551, integration tier 21.

**Closed since R20:** the HOF hub (`hall-of-fame-stat-leaders.md`) passed the council-stamp
gate with a real verdict record this cycle — the recurring `git restore --staged` workaround
from the R20 retro's gap #1 was not needed. Treat that gap as closed unless it re-fires.

**QA verdict: PASS WITH WARNINGS** (proceeds). QA independently re-derived the R23 backtest
figures to an exact match, confirmed all 18 allowlisted paths committed with the two
deliberate exclusions intact, and verified 14 changed stamped docs against their audit
records. Warnings: the completion sentinel above (fixed); the chronic HOF chart count of 4 vs
a stale checklist spec of ≥6 (no action, non-regressed — same standing item as the R20 retro);
and **BL-18**, a data-only Phase-1 regeneration leaving the Skeptic/Gaffer half of
`afl-backtest-2026.md`'s stamp dated two weeks before the content it sits under. Verified
before logging: `refresh_and_rank.sh` invokes no Skeptic, so those timestamps cannot cover
today's bytes. Not a correctness issue — the re-verify hop writes a fresh hash-keyed
DataSentinel PASS — but the visible stamp overclaims. Logged, not fixed: hand-editing it
current would be simulating a verdict.

**Two QA runs, and the slow one was not the dead one.** I dispatched QA, judged it dead when
its output file sat at 134 bytes with no live process, and relaunched. The first run then
returned normally after ~2.8 hours. **A quiet output file plus no matching process is not
evidence an agent died** — check the process list for the RIGHT repo (the ones I saw belonged
to another project entirely) before concluding. The cost was a duplicate run, but the benefit
was accidental: two independent passes agreed on every figure, and the second surfaced BL-18
that the first missed. A relaunch cannot be cancelled once running (different owner).

**Left deliberately untouched:** `last_refresh_status.json` still records the honest
Phase-3d abort — that file is about what the HARNESS did, and it failed. The sentinel is
about what the CYCLE did, and it finished. Keep the two straight.
