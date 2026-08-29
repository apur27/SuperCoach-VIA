---
name: weekly-r24-r25-retro
description: 2026-08-18 R24→R25 refresh — genuine Skeptic BLOCK on a recurring source-doc mislabel (BL-19), a PASS_WITH_CONCERNS I chose not to ship on, BL-17 chart gate re-fired, clean ship
metadata:
  type: project
---

2026-08-18 weekly refresh (R24 completed → R25 predicted). Shipped `4e53cc2f7`
(Phase 1, harness-pushed) → `6c9c2ff3d` (Phase 4, completed manually) → `931082ee6`
(BL-19 backlog entry). Full run ~65 min of harness plus ~4 gate cycles.

**The stale-marker preflight paid again, third cycle running.** Opened with
`last_refresh_status.json` at `{"phase":"3d","exit_code":1,"round":"24"}`. Resolved by
ordering + content: three commits landed after its timestamp and
`last_refresh_complete.json` recorded R24 complete at 18:16 the same day. Not a blocker.
This check is now cheap and reliable — keep doing it first, every cycle.

**Skeptic BLOCKed, and it was RIGHT — the same defect as last week.** S1: the recap called
`+29.9` Fremantle's "average winning margin". It is the signed all-games mean (league mean
exactly 0.0 by construction). I recomputed before routing: all-games Fremantle +29.86 /
Sydney +28.23, but **wins-only Fremantle +36.42 (19 wins) / Sydney +44.82 (17 wins)** — so
under the prose's own label the ranking INVERTS. Not a wording nit; the claim was false.

**The root cause is a source-doc trap, not agent sloppiness — this is the reusable lesson.**
FootyStrategy diagnosed it and I verified: `docs/afl-stat-leaders-2026.md` L306 emits a
heading `#### Winning margin` directly above a table whose column (L308) is `Avg margin`,
defined at L292 as all-games and confirmed at L316 as mean ~0. Citing that heading faithfully
PRODUCES the mislabel. That is why fixing the R24 text did not prevent R25 — **a text fix
cannot fix a defect whose cause is upstream of the text.** Filed as **BL-19** (Scientist:
rename the generated heading + regression test). When a defect recurs verbatim, stop patching
the artifact and go find what regenerates it.

**I declined to ship on PASS_WITH_CONCERNS, and that was the right call.** After the S1-S5
fixes Skeptic returned PASS_WITH_CONCERNS with two NEW findings. R1 was substantive: the
clause added to close S5 claimed projections "run below season averages, most visibly for the
highest-volume players". I reproduced Skeptic's analysis exactly (404/413 matched, 51.2%
below, mean -0.14) and the bucket means show the sign FLIPS: +1.45 for <10 dpg, -3.60 for 30+.
The model regresses to the mean; it is not one-way biased. A coach reading that caveat about a
low-volume player would infer the opposite of the truth. PASS_WITH_CONCERNS permits ship —
but "permits" is not "should". One more routing cycle cost ~6 minutes and removed a false
statement from a doc our primary audience acts on. **Route a concern back when it is false to
the reader, not merely imprecise.**

**Verify a routed finding before routing it.** I recomputed S1, R1 and R2 myself first. All
three held. That is cheap insurance against sending an agent to "fix" something correct — see
[[verify-routed-findings]]. It also caught my own error in the other direction: I flagged
"Clayton Oliver (Greater Western Sydney)" as a suspected club hallucination from prior-season
memory, and the 2026 data said GWS. **The repo data outranks my training-data instinct** —
exactly what CLAUDE.md's verification rule exists for.

**BL-17 re-fired at Phase 3d, symptom byte-identical to the logged one.** Clean-interpreter
render `691acd8f` / 174,837 bytes; the in-pipeline blob Phase 1 had already committed was
`02e97b9d` / 175,862. Same md5 as the R23→R24 retro records, so no re-investigation was
warranted — regenerated in a clean interpreter, integration tier went 21/21. The cause is
still unfixed (ambient matplotlib state), so **expect this every cycle where Phase 1 commits
charts**: budget two minutes for it, do not re-diagnose it.

**Manual Phase-4 completion, third validated use.** Allowlist staged verbatim from
`weekly_refresh.sh`, commit via `git_commit_safe.sh` (13 council docs stamp-verified, 0
failed), push, then the TAIL: `backtest_completeness.py status` (exactly 1 orphan — the run I
just pushed, so `mark` was safe) → `mark` → completion sentinel. `last-round --year 2026` then
reads 24. Left `last_refresh_status.json` recording the honest 3c abort: that file is about
what the HARNESS did, the sentinel about what the CYCLE did.

**Verify the ship by content, not by hash.** I hashed the blob as stored on origin/main
(`git cat-file -p origin/main:docs/afl-insights.md`) and it came to `f3f200bd` — the exact hash
carrying both the DataSentinel PASS and the Skeptic PASS. That proves the shipped bytes ARE
the gated bytes, which a matching commit hash alone does not.

**Gates:** phantom-row PASS; match-completeness PASS; HOF numeric PASS; backtest-doc re-verify
DataSentinel PASS (the hop that stranded an earlier cycle); recap DataSentinel PASS and Skeptic
PASS both at hash `f3f200bd`, 0 findings each; integration 21/21; fast tier 551.
**QA: PASS WITH WARNINGS** — independently re-derived the R24 backtest (372 players, MAE 3.84,
RMSE 4.97) to an exact match and re-checked every S1-S5/R1-R2 fix in the current text rather
than trusting the verdict records. Warnings both chronic: the HOF 4-vs-6 chart-count checklist
mismatch, and the sentinel/status split above.

**Not fixed, deliberately:** Skeptic noted Lachie Neale carries no club attribution while every
other named player does. Cosmetic, Skeptic declined to raise it as a finding, and a sixth gate
pass on one paragraph would have cost more than it returned. Noted here instead.
