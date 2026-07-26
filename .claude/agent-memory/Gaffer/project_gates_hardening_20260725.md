---
name: gates-hardening-20260725
description: 2026-07-25/26 session — three unreachable/absent gates fixed and the defect class behind them (a gate that cannot run reads as a gate that passed)
metadata:
  type: project
---

Commits `a46ca60e8` (fail-closed gates), `fa76f13a3` (test tiers), `24c047286`
(README accuracy). Weekly cycle was idle; this was harness + correctness work.

**The through-line: a gate that cannot run is indistinguishable from a gate that
passed.** Every expensive finding this session was an instance of it:

- Backtest orphans: "an artifact exists" was used as proof a round was scored, so
  a FATALed cycle's output was read as complete. See [[backtest-completion-manifest]].
- `check_top100_consistency()` lived in a code path (`update_team_analysis.py`
  step [13/14]) that NO harness has ever invoked — seven cycles of frozen profile
  prose. CLAUDE.md's inline-tag exemption for `docs/hall-of-fame-top100.md` is
  justified by that gate, so the exemption was unbacked the whole time.
- The same gate could ALSO pass by matching zero profiles and returning `([], [])`
  on an encoding variant — clean-looking output from a check that examined nothing.
- Three section replacers printed to stderr and returned input verbatim, so
  `refresh_readme.py` recorded no error and the cycle exited 0 with a frozen
  published section.
- The pre-commit hook `exit 0`d whenever no `.md` was staged: pure-Python commits
  were never checked at all.

**How to apply:** when auditing any gate, ask two questions beyond "is the logic
right?" — (1) is it actually reachable from the production entry point, and (2)
what does it return when it matches nothing / its inputs are missing? Silence and
success must not look identical. Grep for `os.path.exists(...)` guards wrapping
verification steps; that pattern is how "skip" got spelled "pass" twice here.

**Dogfooding is not optional.** Making the hook run the suite exposed three
defects in the harness that were invisible until the suite ran INSIDE a commit:
unbounded run time, git's per-command env leaking into git-driving tests, and
`git_commit_safe.sh`'s single global flock deadlocking against the commit that
invoked it. Diagnosed from the process tree (`ps -eo pid,ppid,etime,cmd`), not
guessed — two commit attempts were aborted before evidence was gathered.

**Two tests were nearly shipped vacuous.** A behavioural test for the git-env leak
passed identically against fixed and unfixed hooks, twice, because setting
`GIT_INDEX_FILE` breaks the hook's own staged-file detection before pytest is
reached. Always verify a new regression test FAILS against the unfixed code — copy
the pre-fix file to scratch and run it. An honest contract check beats a
behavioural test that proves nothing.

Related: [[verify-routed-findings]], [[llm-datasentinel-arithmetic]].
